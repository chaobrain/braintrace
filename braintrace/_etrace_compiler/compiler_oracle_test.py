# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Numerical oracle tests for the ETrace compiler.

These tests treat the compiler as a black box that emits a *transition
jaxpr* ``y -> h`` for every registered relation. They verify that the
transition jaxpr is *semantically correct* by:

  1. Evaluating ``dh/dy`` numerically through the compiler's transition
     jaxpr (via ``jax.jacfwd``).
  2. Computing the same Jacobian analytically from the model's known
     forward expression.
  3. Asserting the two agree to within ``rtol=1e-5, atol=1e-5``.

A failure here means the compiler is registering a relation whose
``y_to_hidden_group_jaxprs`` does not faithfully capture ``dh/dy`` —
exactly the bug downstream D-RTRL / pp_prop algorithms would hit at
runtime.

The non-parametric-tail invariant is also exercised: for
:class:`PartialPathRNN`, ``w1``'s transition jaxpr must capture *only*
the direct ``mid -> tanh(.) -> h`` path, and the indirect ``mid -> w2 ->
h`` contribution must appear as a constvar (so its Jacobian wrt ``y`` is
zero). The analytical reference takes the same view.
"""



import warnings

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braintrace import compile_etrace_graph
from braintrace._etrace_compiler.scenario_catalog import (
    ElemwiseOnlyRNN,
    PartialPathRNN,
    UnbatchedMvRNN,
)


def _silent_compile(model, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return compile_etrace_graph(model, *args, include_hidden_perturb=False)


def _transition_callable(rel, group, const_vals):
    """Wrap ``rel.y_to_hidden_groups`` as a single-argument function for
    ``jax.jacfwd``.

    The returned callable takes ``y_val`` and returns the *concatenated*
    hidden state value for the matching group.
    """
    consts = [const_vals[v] for v in rel.y_to_hidden_group_jaxprs[0].constvars]
    jaxpr = rel.y_to_hidden_group_jaxprs[0]

    def f(y_val):
        out = jax.core.eval_jaxpr(jaxpr, consts, y_val)
        return group.concat_hidden(out)

    return f


# ---------------------------------------------------------------------------
# Oracle 1 — UnbatchedMvRNN: dh/dy = diag(1 - h^2)
# ---------------------------------------------------------------------------

class TestOracle_TanhTail:
    """For ``h = tanh(y)``, ``dh/dy`` is the diagonal matrix
    ``diag(1 - tanh(y)^2) = diag(1 - h^2)``."""

    def test_unbatched_mv_rnn_dhdy_matches_analytic(self):
        model = UnbatchedMvRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = jnp.array([0.3, -0.7, 1.1])

        graph = _silent_compile(model, inp)
        _, _, _, temps = graph.module_info.jaxpr_call(inp)

        rel = graph.hidden_param_op_relations[0]
        group = rel.hidden_groups[0]
        y_val = temps[rel.y_var]

        f = _transition_callable(rel, group, temps)
        dh_dy = jax.jacfwd(f)(y_val)

        # h = tanh(y) (after squeezing the trailing concat-axis added by
        # group.concat_hidden — only one HiddenState in the group).
        h_val = jnp.tanh(y_val)
        analytic = jnp.diag(1.0 - h_val ** 2)
        # dh_dy is shape (n_out, 1, n_out): the (1) axis is concat axis.
        assert dh_dy.shape == (4, 1, 4)
        np.testing.assert_allclose(
            dh_dy.squeeze(1), analytic, rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Oracle 2 — ElemwiseOnlyRNN: dh/dy = diag(1 - h^2) where y = h_prev + x + w
# ---------------------------------------------------------------------------

class TestOracle_ElemwiseTail:

    def test_elemwise_only_rnn_dhdy_matches_analytic(self):
        model = ElemwiseOnlyRNN(4)
        brainstate.nn.init_all_states(model)
        inp = jnp.array([0.5, -0.2, 0.9, -1.0])

        graph = _silent_compile(model, inp)
        _, _, _, temps = graph.module_info.jaxpr_call(inp)

        rel = graph.hidden_param_op_relations[0]
        group = rel.hidden_groups[0]
        y_val = temps[rel.y_var]
        # The compiler's "y" for etp_elemwise_p is the processed weight
        # itself; sum with h_prev + x to get the pre-tanh quantity.
        h_prev = model.h.value  # zeros after init
        pre = h_prev + inp + y_val
        h_val = jnp.tanh(pre)
        analytic = jnp.diag(1.0 - h_val ** 2)

        f = _transition_callable(rel, group, temps)
        dh_dy = jax.jacfwd(f)(y_val)
        np.testing.assert_allclose(
            dh_dy.squeeze(-2), analytic, rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Oracle 3 — PartialPathRNN: w1 transition captures only direct path
# ---------------------------------------------------------------------------

class TestOracle_PartialPathDirectOnly:
    """In :class:`PartialPathRNN`, ``mid`` (= y_var of w1) flows into
    ``h`` along *two* paths:

      * Direct: ``h = tanh(mid + through_const)``.
      * Indirect: ``through = matmul(mid, w2)``, fed into the tanh.

    The compiler must build w1's transition jaxpr to treat the
    ``through`` value as a constvar, so ``dh/dy_w1`` captures only the
    direct contribution: ``diag(1 - h^2)``. The indirect ``w2``
    contribution disappears because ``w2`` is a non-gradient-enabled ETP
    primitive on the tail.
    """

    def test_w1_dhdy_is_direct_only(self):
        model = PartialPathRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = jnp.array([0.4, -0.6, 1.2])

        graph = _silent_compile(model, inp)
        _, _, _, temps = graph.module_info.jaxpr_call(inp)

        by_path = {r.path: r for r in graph.hidden_param_op_relations}
        rel_w1 = by_path[('w1',)]
        group = rel_w1.hidden_groups[0]
        y_val = temps[rel_w1.y_var]

        f = _transition_callable(rel_w1, group, temps)
        # h evaluated by the transition jaxpr with the actual y_val and
        # the compile-time through-const; its squeezed concat axis is h.
        h_val = f(y_val).squeeze(-1)
        dh_dy = jax.jacfwd(f)(y_val)

        # Direct-only analytic: dh/d(mid) = diag(1 - tanh(mid + through_const)^2).
        analytic = jnp.diag(1.0 - h_val ** 2)
        np.testing.assert_allclose(
            dh_dy.squeeze(-2), analytic, rtol=1e-5, atol=1e-5,
        )

    def test_w2_dhdy_is_correct(self):
        model = PartialPathRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = jnp.array([0.4, -0.6, 1.2])

        graph = _silent_compile(model, inp)
        _, _, _, temps = graph.module_info.jaxpr_call(inp)

        by_path = {r.path: r for r in graph.hidden_param_op_relations}
        rel_w2 = by_path[('w2',)]
        group = rel_w2.hidden_groups[0]
        y_val = temps[rel_w2.y_var]

        f = _transition_callable(rel_w2, group, temps)
        h_val = f(y_val).squeeze(-1)
        dh_dy = jax.jacfwd(f)(y_val)

        analytic = jnp.diag(1.0 - h_val ** 2)
        np.testing.assert_allclose(
            dh_dy.squeeze(-2), analytic, rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Oracle 4 — Finite-difference cross-check
# ---------------------------------------------------------------------------

class TestOracle_FiniteDifference:
    """The transition jaxpr's Jacobian must agree with the central
    finite-difference Jacobian of the *same* function.  This checks that
    the jaxpr has no internal book-keeping bug independent of the
    analytic reference."""

    def _check_fd(self, rel, group, temps, atol=1e-3):
        y_val = temps[rel.y_var]
        f = _transition_callable(rel, group, temps)
        ad_jac = jax.jacfwd(f)(y_val)

        eps = 1e-3
        n = y_val.shape[0]
        fd_cols = []
        for i in range(n):
            e = jnp.zeros_like(y_val).at[i].set(eps)
            fd_cols.append((f(y_val + e) - f(y_val - e)) / (2 * eps))
        fd = jnp.stack(fd_cols, axis=-1)
        np.testing.assert_allclose(ad_jac, fd, atol=atol)

    def test_unbatched_mv_rnn_fd_matches_ad(self):
        model = UnbatchedMvRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = jnp.array([0.1, 0.2, -0.3])

        graph = _silent_compile(model, inp)
        _, _, _, temps = graph.module_info.jaxpr_call(inp)
        rel = graph.hidden_param_op_relations[0]
        self._check_fd(rel, rel.hidden_groups[0], temps)

    def test_partial_path_w1_fd_matches_ad(self):
        model = PartialPathRNN(3, 4)
        brainstate.nn.init_all_states(model)
        inp = jnp.array([0.05, -0.1, 0.2])

        graph = _silent_compile(model, inp)
        _, _, _, temps = graph.module_info.jaxpr_call(inp)
        rel = next(
            r for r in graph.hidden_param_op_relations
            if r.path == ('w1',)
        )
        self._check_fd(rel, rel.hidden_groups[0], temps)
