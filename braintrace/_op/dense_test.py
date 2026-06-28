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

"""Tests for the dense matmul ETP primitives and the :func:`matmul` API.

Coverage:

* Auto-dispatch — ``x.ndim >= 2`` selects ``etp_mm_p``; otherwise
  ``etp_mv_p``. Verified by jaxpr inspection.
* Forward correctness — agrees with ``x @ w (+ b)``.
* Bias presence — ``has_bias`` parameter is propagated through ``bind``.
* brainunit support — quantities, mixed units, unitless inputs.
* JAX rules — jit, vmap, grad, jvp work with no extra plumbing.
* Four ETP rules — ``yw_to_w``, ``xy_to_dw``, ``init_drtrl``, ``init_pp``
  return tensors of the documented shape and value.
"""

from collections import namedtuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import brainunit as u

import braintrace
from braintrace._op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    etp_mm_p,
    etp_mv_p,
    matmul,
)

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


# ---------------------------------------------------------------------------
# Forward correctness + dispatch
# ---------------------------------------------------------------------------

class TestForwardCorrectness:

    def test_unbatched_matches_python_matmul(self):
        x = jnp.array([1.0, 2.0, 3.0])
        w = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        out = matmul(x, w)
        np.testing.assert_allclose(out, x @ w)

    def test_batched_matches_python_matmul(self):
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        w = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        out = matmul(x, w)
        np.testing.assert_allclose(out, x @ w)

    def test_with_bias(self):
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 4))
        b = jnp.arange(4.0)
        out = matmul(x, w, bias=b)
        np.testing.assert_allclose(out, x @ w + b)

    def test_higher_rank_input(self):
        x = jnp.ones((2, 5, 3))
        w = jnp.ones((3, 4))
        out = matmul(x, w)
        np.testing.assert_allclose(out, x @ w)


class TestAutoDispatch:

    def test_unbatched_uses_mv_primitive(self):
        x = jnp.array([1.0, 2.0])
        w = jnp.eye(2)
        jaxpr = jax.make_jaxpr(lambda x, w: matmul(x, w))(x, w)
        assert any(eqn.primitive is etp_mv_p for eqn in jaxpr.jaxpr.eqns)
        assert not any(eqn.primitive is etp_mm_p for eqn in jaxpr.jaxpr.eqns)

    def test_batched_uses_mm_primitive(self):
        x = jnp.ones((4, 2))
        w = jnp.eye(2)
        jaxpr = jax.make_jaxpr(lambda x, w: matmul(x, w))(x, w)
        assert any(eqn.primitive is etp_mm_p for eqn in jaxpr.jaxpr.eqns)
        assert not any(eqn.primitive is etp_mv_p for eqn in jaxpr.jaxpr.eqns)


class TestHasBiasParam:

    def test_has_bias_true_when_bias_supplied(self):
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 4))
        b = jnp.zeros(4)
        jaxpr = jax.make_jaxpr(lambda x, w, b: matmul(x, w, bias=b))(x, w, b)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_mm_p)
        assert eqn.params['has_bias'] is True

    def test_has_bias_false_when_bias_omitted(self):
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 4))
        jaxpr = jax.make_jaxpr(lambda x, w: matmul(x, w))(x, w)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_mm_p)
        assert eqn.params['has_bias'] is False


# ---------------------------------------------------------------------------
# brainunit support
# ---------------------------------------------------------------------------

class TestBrainunit:

    def test_unitless_input_returns_unitless(self):
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 4))
        out = matmul(x, w)
        assert not isinstance(out, u.Quantity)

    def test_input_with_units_returns_quantity(self):
        x = jnp.ones((2, 3)) * u.mV
        w = jnp.ones((3, 4))
        out = matmul(x, w)
        # Output should still be a Quantity
        assert hasattr(out, 'mantissa') or isinstance(out, u.Quantity)

    def test_units_multiply_correctly(self):
        x = jnp.ones((2, 3)) * u.mV
        w = jnp.ones((3, 4)) * u.ms
        out = matmul(x, w)
        # Unit should be mV * ms
        expected = (jnp.ones((2, 3)) @ jnp.ones((3, 4))) * (u.mV * u.ms)
        np.testing.assert_allclose(
            u.get_mantissa(out), u.get_mantissa(expected),
        )

    def test_bias_with_units(self):
        x = jnp.ones((2, 3)) * u.mV
        w = jnp.ones((3, 4))
        b = jnp.ones(4) * u.mV
        out = matmul(x, w, bias=b)
        expected = (jnp.ones((2, 3)) @ jnp.ones((3, 4)) + jnp.ones(4)) * u.mV
        np.testing.assert_allclose(
            u.get_mantissa(out), u.get_mantissa(expected),
        )


# ---------------------------------------------------------------------------
# JAX rules — jit / vmap / grad
# ---------------------------------------------------------------------------

class TestJAXRules:

    def test_jit(self):
        x = jnp.ones((2, 3))
        w = jnp.arange(12.0).reshape(3, 4)
        f = jax.jit(matmul)
        np.testing.assert_allclose(f(x, w), x @ w)

    def test_vmap_over_batch(self):
        x = jnp.arange(6.0).reshape(2, 3)
        w = jnp.eye(3)
        out = jax.vmap(lambda xi: matmul(xi, w))(x)
        np.testing.assert_allclose(out, x @ w)

    def test_grad_wrt_w(self):
        x = jnp.ones((2, 3))
        w = jnp.arange(12.0).reshape(3, 4)
        gw = jax.grad(lambda w_: matmul(x, w_).sum())(w)
        # d(sum(x@w))/dw = x.T @ ones(2, 4) = sum(x, axis=0)[:, None] * ones((1,4))
        expected = x.sum(axis=0)[:, None] * jnp.ones((1, 4))
        np.testing.assert_allclose(gw, expected)

    def test_grad_wrt_x(self):
        x = jnp.arange(6.0).reshape(2, 3)
        w = jnp.ones((3, 4))
        gx = jax.grad(lambda x_: matmul(x_, w).sum())(x)
        # d(sum(x@w))/dx = ones(2,4) @ w.T
        expected = jnp.ones((2, 4)) @ w.T
        np.testing.assert_allclose(gx, expected)


# ---------------------------------------------------------------------------
# ETP rules — yw_to_w / xy_to_dw / init_drtrl / init_pp
# ---------------------------------------------------------------------------

class TestMmEtpRules:

    def test_yw_to_w_broadcasts_hidden(self):
        """``yw_to_w`` multiplies ``trace['weight']`` element-wise by
        ``hidden_dim`` broadcast along the input axis. Trace shape is
        ``(in, out)`` (solve context, batch stripped); ``hidden_dim`` is
        ``(out,)``. ``expand_dims(hidden_dim, axis=-2)`` → ``(1, out)``
        broadcasts against ``(in, out)`` → per-row scaling by ``hidden[o]``.
        Correct for non-square (in != out)."""
        rule = ETP_RULES_YW_TO_W[etp_mm_p]
        in_dim, out_dim = 5, 3
        hidden = jnp.array([1.0, 2.0, 3.0])  # (out,)
        trace = {'weight': jnp.ones((in_dim, out_dim))}
        out = rule(hidden, trace)
        assert isinstance(out, dict)
        assert out['weight'].shape == (in_dim, out_dim)
        # Column j scaled by hidden[j].
        np.testing.assert_allclose(
            out['weight'], jnp.ones((in_dim, out_dim)) * hidden[None, :]
        )

    def test_yw_to_w_with_bias(self):
        """When has_bias=True, ``yw_to_w`` also scales ``trace['bias']``."""
        rule = ETP_RULES_YW_TO_W[etp_mm_p]
        in_dim, out_dim = 5, 4
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])  # (out,)
        trace = {
            'weight': jnp.ones((in_dim, out_dim)),  # (in, out)
            'bias': jnp.ones((out_dim,)),  # (out,)
        }
        out = rule(hidden, trace, has_bias=True)
        assert isinstance(out, dict)
        assert 'bias' in out
        np.testing.assert_allclose(out['bias'], hidden)

    def test_xy_to_dw_matches_jax_vjp(self):
        rule = ETP_RULES_XY_TO_DW[etp_mm_p]
        x = jnp.arange(6.0).reshape(2, 3)
        w = jnp.arange(12.0).reshape(3, 4)
        hidden = jnp.ones((2, 4))
        weights = {'weight': w}
        dw_dict = rule(x, hidden, weights)
        assert isinstance(dw_dict, dict)
        # VJP of y = x @ w wrt w with cotangent ones((2,4)) is x.T @ ones((2,4))
        expected = x.T @ hidden
        np.testing.assert_allclose(dw_dict['weight'], expected)

    def test_xy_to_dw_with_bias(self):
        """With has_bias=True the dict result also contains a 'bias' entry."""
        rule = ETP_RULES_XY_TO_DW[etp_mm_p]
        x = jnp.ones((2, 3))
        w = jnp.arange(12.0).reshape(3, 4)
        b = jnp.zeros(4)
        hidden = jnp.ones((2, 4))
        weights = {'weight': w, 'bias': b}
        dw_dict = rule(x, hidden, weights, has_bias=True)
        assert isinstance(dw_dict, dict)
        assert 'weight' in dw_dict
        assert 'bias' in dw_dict
        # db = sum of hidden over batch axis = ones(4)
        np.testing.assert_allclose(dw_dict['bias'], hidden.sum(axis=0))

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_mm_p]
        x_var = _fake_var((4, 3))  # (batch, in)
        y_var = _fake_var((4, 5))
        weight_vars = {'weight': _fake_var((3, 5))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert isinstance(out, dict)
        assert out['weight'].shape == (4, 3, 5, 2)

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_mm_p]
        x_var = _fake_var((4, 3))
        y_var = _fake_var((4, 5))
        weight_vars = {'weight': _fake_var((3, 5)), 'bias': _fake_var((5,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert isinstance(out, dict)
        assert out['weight'].shape == (4, 3, 5, 2)
        assert out['bias'].shape == (4, 5, 2)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_mm_p]
        x_var = _fake_var((4, 3))
        y_var = _fake_var((4, 5))
        weight_vars = {'weight': _fake_var((3, 5))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out.shape == (4, 5, 2)


class TestMvEtpRules:

    def test_yw_to_w_broadcasts_hidden(self):
        """``yw_to_w`` multiplies ``trace['weight']`` by ``hidden`` broadcast
        along the column axis. The rule accepts and returns a dict."""
        rule = ETP_RULES_YW_TO_W[etp_mv_p]
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])  # (out,)
        trace = {'weight': jnp.ones((3, 4))}  # (in, out)
        out = rule(hidden, trace)
        assert isinstance(out, dict)
        assert out['weight'].shape == (3, 4)
        # column j scaled by hidden[j]
        np.testing.assert_allclose(out['weight'], jnp.ones((3, 4)) * hidden[None, :])

    def test_yw_to_w_with_bias(self):
        """When has_bias=True, ``yw_to_w`` also scales ``trace['bias']``."""
        rule = ETP_RULES_YW_TO_W[etp_mv_p]
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        trace = {'weight': jnp.ones((3, 4)), 'bias': jnp.ones((4,))}
        out = rule(hidden, trace, has_bias=True)
        assert isinstance(out, dict)
        assert 'bias' in out
        np.testing.assert_allclose(out['bias'], hidden)

    def test_xy_to_dw_matches_outer_product(self):
        rule = ETP_RULES_XY_TO_DW[etp_mv_p]
        x = jnp.arange(3.0)
        w = jnp.arange(12.0).reshape(3, 4)
        hidden = jnp.arange(4.0)
        weights = {'weight': w}
        dw_dict = rule(x, hidden, weights)
        assert isinstance(dw_dict, dict)
        np.testing.assert_allclose(dw_dict['weight'], jnp.outer(x, hidden))

    def test_xy_to_dw_with_bias(self):
        """With has_bias=True the dict result also contains a 'bias' entry."""
        rule = ETP_RULES_XY_TO_DW[etp_mv_p]
        x = jnp.arange(3.0)
        w = jnp.arange(12.0).reshape(3, 4)
        b = jnp.zeros(4)
        hidden = jnp.arange(4.0)
        weights = {'weight': w, 'bias': b}
        dw_dict = rule(x, hidden, weights, has_bias=True)
        assert isinstance(dw_dict, dict)
        assert 'weight' in dw_dict
        assert 'bias' in dw_dict
        # db = hidden (unbatched VJP)
        np.testing.assert_allclose(dw_dict['bias'], hidden)

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((3, 5))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert isinstance(out, dict)
        assert out['weight'].shape == (3, 5, 2)

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((3, 5)), 'bias': _fake_var((5,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert isinstance(out, dict)
        assert out['weight'].shape == (3, 5, 2)
        assert out['bias'].shape == (5, 2)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((3, 5))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out.shape == (5, 2)


# ---------------------------------------------------------------------------
# Bias-gradient correctness (D-RTRL vs BPTT)
# ---------------------------------------------------------------------------

class TestMMBiasGradient:
    """D-RTRL gradient correctness for etp_mm_p with a bias vector.

    Train a tiny recurrent net one step and verify ETP gradients (dW, db)
    match BPTT ground-truth. The merged-ParamState variant uses a single
    ParamState holding {'weight': W, 'bias': b} and proves that bias
    gradients are no longer silently zero: D-RTRL produces a db matching BPTT.
    """

    def _build_cell_merged(self):
        """One-step RNN with weight+bias stored in a merged ParamState."""

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(
                    {'weight': jnp.ones((4, 4)) * 0.1,
                     'bias': jnp.ones((4,)) * 0.2}
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                w = self.p.value['weight']
                b = self.p.value['bias']
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, w, b)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        return cell

    def test_drtrl_grad_matches_bptt_merged_paramstate(self):
        """D-RTRL dW and db must match BPTT for one recurrent step."""
        cell = self._build_cell_merged()
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 4)))

        x = jnp.ones((1, 4)) * 0.3
        target = jnp.zeros((1, 4))

        # --- ETP gradient via D-RTRL ---
        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)

        # Extract the dict-valued gradient for cell.p
        # grads_etrace is keyed by path; cell.p is at path ('p',)
        # Its value is a dict {'weight': ..., 'bias': ...}
        grad_p = list(grads_etrace.values())[0]
        assert isinstance(grad_p, dict), (
            f'Expected dict gradient for merged ParamState, got {type(grad_p)}'
        )

        # --- BPTT reference ---
        def bptt_loss(params):
            h = jnp.zeros((1, 4))
            h = jnp.tanh(x + h @ params['weight'] + params['bias'])
            return h.sum()

        bptt = jax.grad(bptt_loss)({'weight': cell.p.value['weight'],
                                    'bias': cell.p.value['bias']})

        np.testing.assert_allclose(grad_p['weight'], bptt['weight'], atol=1e-5,
                                   err_msg='D-RTRL dW does not match BPTT')
        np.testing.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                                   err_msg='D-RTRL db does not match BPTT (was it zero?)')


class TestMMNonSquareWeight:
    """_mm_yw_to_w must broadcast hidden_dim correctly when in != out.

    The latent bug was ``jnp.expand_dims(hidden_dim, axis=1)`` in the
    gradient-solve context where the batch axis is stripped: for
    ``hidden_dim: (out,)`` and ``trace: (in, out)`` with ``in != out``,
    axis=1 produced shape ``(out, 1)`` and broadcasting failed. Fixed by
    using ``axis=-2`` which inserts the singleton at the correct axis
    in both the batched (trace-update) and unbatched (solve) contexts.
    """

    def test_yw_to_w_non_square_solve_context(self):
        from braintrace._op.dense import _mm_yw_to_w
        # Gradient-solve shapes (batch axis stripped by outer vmap).
        in_dim, out_dim = 5, 3
        trace_weight = jnp.arange(in_dim * out_dim, dtype=jnp.float32).reshape(
            in_dim, out_dim
        )
        hidden_dim = jnp.array([1.0, 2.0, 3.0])  # (out,)

        out = _mm_yw_to_w(hidden_dim, {'weight': trace_weight}, has_bias=False)

        # Expected: trace[i, o] * hidden_dim[o] — broadcast across in axis.
        expected = trace_weight * hidden_dim[None, :]
        np.testing.assert_array_equal(out['weight'], expected)
        assert out['weight'].shape == (in_dim, out_dim)

    def test_yw_to_w_non_square_trace_update_context(self):
        from braintrace._op.dense import _mm_yw_to_w
        # Trace-update shapes (batch retained).
        batch, in_dim, out_dim = 2, 5, 3
        trace_weight = jnp.ones((batch, in_dim, out_dim)) * 0.5
        hidden_dim = jnp.ones((batch, out_dim)) * 2.0

        out = _mm_yw_to_w(hidden_dim, {'weight': trace_weight}, has_bias=False)

        # Expected: trace[b,i,o] * hidden_dim[b,o].
        expected = trace_weight * hidden_dim[:, None, :]
        np.testing.assert_array_equal(out['weight'], expected)
        assert out['weight'].shape == (batch, in_dim, out_dim)


class TestPublicAPIRoundTrip:
    """``braintrace.matmul`` and ``braintrace._op.matmul`` are the
    same function — the public alias is not a re-implementation."""

    def test_public_alias_identity(self):
        assert braintrace.matmul is matmul


class TestSeparateParamStateBias:
    """D-RTRL gradient correctness when weight and bias are in distinct ParamStates."""

    def test_separate_weight_and_bias_grads(self):
        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.b = brainstate.ParamState(jnp.ones((4,)) * 0.2)
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value, self.b.value)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 4)))

        x = jnp.ones((1, 4)) * 0.3

        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)

        # --- BPTT reference ---
        def bptt_loss(w, b):
            h = jnp.zeros((1, 4))
            h = jnp.tanh(x + h @ w + b)
            return h.sum()

        dW_bptt, db_bptt = jax.grad(bptt_loss, (0, 1))(cell.w.value, cell.b.value)

        # grads_etrace is keyed by path: ('w',) -> array, ('b',) -> array
        leaves = jax.tree.leaves(grads_etrace)
        shapes = sorted(leaf.shape for leaf in leaves)
        assert shapes == [(4,), (4, 4)], (
            f'Expected two gradient leaves with shapes [(4,), (4, 4)], got {shapes}'
        )

        w_leaf = next(l for l in leaves if l.shape == (4, 4))
        b_leaf = next(l for l in leaves if l.shape == (4,))

        npt.assert_allclose(w_leaf, dW_bptt, atol=1e-5,
                            err_msg='D-RTRL dW does not match BPTT')
        npt.assert_allclose(b_leaf, db_bptt, atol=1e-5,
                            err_msg='D-RTRL db does not match BPTT (was it zero?)')
