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

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._compiler.canonicalize import (
    ControlFlowPolicy,
    DEFAULT_CONTROL_FLOW_POLICY,
    if_convert_conds,
)
from braintrace._compiler.diagnostics import (
    DiagnosticKind,
    diagnostic_context,
)


def _primitive_names(closed_jaxpr):
    return [eqn.primitive.name for eqn in closed_jaxpr.jaxpr.eqns]


def _convert(closed_jaxpr, weights=(), hiddens_in=(), hiddens_out=(), policy=None):
    kwargs = dict(
        weight_invars=set(weights),
        hidden_invars=set(hiddens_in),
        hidden_outvars=set(hiddens_out),
    )
    if policy is not None:
        kwargs['policy'] = policy
    return if_convert_conds(closed_jaxpr, **kwargs)


def _eval(closed_jaxpr, *args):
    return jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)


class TestControlFlowPolicy:
    def test_defaults(self):
        assert DEFAULT_CONTROL_FLOW_POLICY.cond == 'convert'
        assert DEFAULT_CONTROL_FLOW_POLICY.scan_unroll_limit == 16

    def test_opaque_returns_input_unchanged(self):
        def f(x, w):
            return jax.lax.cond(
                jnp.sum(x) > 0.,
                lambda: braintrace.matmul(x, w),
                lambda: x * 2.,
            )

        closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.ones((3, 3)))
        out = _convert(closed, policy=ControlFlowPolicy(cond='opaque'))
        assert out is closed

    def test_invalid_cond_policy_raises(self):
        closed = jax.make_jaxpr(lambda x: x * 2.)(jnp.ones(3))
        with pytest.raises(ValueError, match='cond'):
            _convert(closed, policy=ControlFlowPolicy(cond='bogus'))


class TestIfConvertConds:
    """Unit tests of the cond -> inlined branches + select_n rewrite."""

    def _etp_cond_jaxpr(self):
        def f(x, w):
            return jax.lax.cond(
                jnp.sum(x) > 0.,
                lambda: braintrace.matmul(x, w),
                lambda: x * 2.,
            )

        x = jnp.arange(3, dtype=jnp.float32) - 0.5
        w = jnp.eye(3) * 0.5
        closed = jax.make_jaxpr(f)(x, w)
        return f, closed, x, w

    def test_etp_cond_is_converted(self):
        f, closed, x, w = self._etp_cond_jaxpr()
        conv = _convert(closed)
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert 'select_n' in names
        assert 'etp_mv' in names

    def test_outvars_and_invars_preserved_by_identity(self):
        f, closed, x, w = self._etp_cond_jaxpr()
        conv = _convert(closed)
        assert all(a is b for a, b in zip(conv.jaxpr.invars, closed.jaxpr.invars))
        assert all(a is b for a, b in zip(conv.jaxpr.outvars, closed.jaxpr.outvars))

    def test_value_equivalence_both_branches(self):
        f, closed, x, w = self._etp_cond_jaxpr()
        conv = _convert(closed)
        for xi in (jnp.abs(x) + 1., -jnp.abs(x) - 1.):
            expected = f(xi, w)
            got = _eval(conv, xi, w)[0]
            assert jnp.allclose(got, expected)

    def test_gradient_equivalence_both_branches(self):
        f, closed, x, w = self._etp_cond_jaxpr()
        conv = _convert(closed)

        def f_conv(xi, wi):
            return _eval(conv, xi, wi)[0]

        for xi in (jnp.abs(x) + 1., -jnp.abs(x) - 1.):
            for argnum in (0, 1):
                g_ref = jax.grad(lambda a, b: jnp.sum(f(a, b) ** 2), argnum)(xi, w)
                g_conv = jax.grad(lambda a, b: jnp.sum(f_conv(a, b) ** 2), argnum)(xi, w)
                assert jnp.allclose(g_ref, g_conv, atol=1e-6)

    def test_three_branch_switch(self):
        def f(i, x, w):
            return jax.lax.switch(
                i,
                [
                    lambda: x * 0.,
                    lambda: braintrace.matmul(x, w),
                    lambda: x + 1.,
                ],
            )

        x = jnp.arange(3, dtype=jnp.float32)
        w = jnp.eye(3) * 2.
        closed = jax.make_jaxpr(f)(jnp.int32(0), x, w)
        conv = _convert(closed)
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert 'etp_mv' in names
        for i in range(3):
            expected = f(jnp.int32(i), x, w)
            got = _eval(conv, jnp.int32(i), x, w)[0]
            assert jnp.allclose(got, expected)

    def test_multi_output_cond(self):
        def f(x, w):
            return jax.lax.cond(
                jnp.sum(x) > 0.,
                lambda: (braintrace.matmul(x, w), x + 1.),
                lambda: (x * 2., x - 1.),
            )

        x = jnp.arange(3, dtype=jnp.float32) + 1.
        w = jnp.eye(3)
        closed = jax.make_jaxpr(f)(x, w)
        conv = _convert(closed)
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert names.count('select_n') == 2
        assert all(a is b for a, b in zip(conv.jaxpr.outvars, closed.jaxpr.outvars))
        for xi in (x, -x):
            for got, expected in zip(_eval(conv, xi, w), f(xi, w)):
                assert jnp.allclose(got, expected)

    def test_passthrough_branch_output(self):
        def f(x):
            return jax.lax.cond(jnp.sum(x) > 0., lambda: x, lambda: -x)

        x = jnp.arange(3, dtype=jnp.float32) + 1.
        closed = jax.make_jaxpr(f)(x)
        # Force relevance by marking x as a hidden invar.
        conv = _convert(closed, hiddens_in=[closed.jaxpr.invars[0]])
        assert 'cond' not in _primitive_names(conv)
        for xi in (x, -x):
            assert jnp.allclose(_eval(conv, xi)[0], f(xi))

    def test_irrelevant_cond_stays_opaque(self):
        def f(x):
            return jax.lax.cond(jnp.sum(x) > 0., lambda: x * 2., lambda: x * 3.)

        closed = jax.make_jaxpr(f)(jnp.ones(3))
        conv = _convert(closed)
        assert conv is closed

    def test_relevant_by_weight_invar(self):
        def f(x, w):
            # Plain (non-ETP) matmul: relevance comes from consuming w.
            return jax.lax.cond(jnp.sum(x) > 0., lambda: x @ w, lambda: x * 2.)

        x = jnp.ones(3)
        w = jnp.eye(3)
        closed = jax.make_jaxpr(f)(x, w)
        conv = _convert(closed, weights=[closed.jaxpr.invars[1]])
        assert 'cond' not in _primitive_names(conv)
        for xi in (x, -x):
            assert jnp.allclose(_eval(conv, xi, w)[0], f(xi, w))

    def test_relevant_by_hidden_outvar(self):
        def f(h):
            return jax.lax.cond(jnp.sum(h) > 0., lambda: h * 2., lambda: h * 3.)

        closed = jax.make_jaxpr(f)(jnp.ones(3))
        conv = _convert(closed, hiddens_out=[closed.jaxpr.outvars[0]])
        assert 'cond' not in _primitive_names(conv)

    def test_nested_cond_converted(self):
        def f(x, w):
            def outer_true():
                return jax.lax.cond(
                    jnp.max(x) > 2.,
                    lambda: braintrace.matmul(x, w),
                    lambda: x * 5.,
                )

            return jax.lax.cond(jnp.sum(x) > 0., outer_true, lambda: x * 2.)

        x = jnp.arange(3, dtype=jnp.float32)
        w = jnp.eye(3)
        closed = jax.make_jaxpr(f)(x, w)
        conv = _convert(closed)
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert 'etp_mv' in names
        for xi in (x + 10., x + 0.1, x - 10.):
            assert jnp.allclose(_eval(conv, xi, w)[0], f(xi, w))

    def test_jit_inside_branch_is_inlined(self):
        @jax.jit
        def jitted(a, b):
            return braintrace.matmul(a, b)

        def f(x, w):
            return jax.lax.cond(
                jnp.sum(x) > 0.,
                lambda: jitted(x, w),
                lambda: x * 2.,
            )

        x = jnp.ones(3)
        w = jnp.eye(3)
        closed = jax.make_jaxpr(f)(x, w)
        conv = _convert(closed)
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert 'jit' not in names and 'pjit' not in names
        assert 'etp_mv' in names
        for xi in (x, -x):
            assert jnp.allclose(_eval(conv, xi, w)[0], f(xi, w))

    def test_shared_jitted_helper_in_both_branches(self):
        # JAX's trace cache hands both branches' calls the SAME inner jaxpr
        # object; conversion + jit inlining must freshen per call site or the
        # two calls silently alias (values from the wrong branch).
        @jax.jit
        def helper(a, b):
            return braintrace.matmul(a, b)

        def f(x, wa, wb):
            return jax.lax.cond(
                jnp.sum(x) > 0.,
                lambda: helper(x, wa),
                lambda: helper(x, wb),
            )

        x = jnp.arange(1., 4.)
        wa = jnp.eye(3) * 2.
        wb = jnp.eye(3) * 3.
        closed = jax.make_jaxpr(f)(x, wa, wb)
        conv = _convert(closed)
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert 'jit' not in names and 'pjit' not in names
        for xi in (x, -x):
            assert jnp.allclose(_eval(conv, xi, wa, wb)[0], f(xi, wa, wb))
            for argnum in (1, 2):
                g_ref = jax.grad(
                    lambda a, b, c: jnp.sum(f(a, b, c) ** 2), argnum)(xi, wa, wb)
                g_conv = jax.grad(
                    lambda a, b, c: jnp.sum(_eval(conv, a, b, c)[0] ** 2),
                    argnum)(xi, wa, wb)
                assert jnp.allclose(g_ref, g_conv, atol=1e-6)

    def test_cond_inside_jit_inside_branch_converted(self):
        # A cond hidden inside a jitted function inside a branch surfaces
        # only after the surfaced jit is inlined; the fixpoint loop must
        # gate/convert it too.
        @jax.jit
        def jitted(a, w):
            return jax.lax.cond(
                jnp.max(a) > 2.,
                lambda: braintrace.matmul(a, w),
                lambda: a * 5.,
            )

        def f(x, w):
            return jax.lax.cond(
                jnp.sum(x) > 0.,
                lambda: jitted(x, w),
                lambda: braintrace.matmul(x, w) * 2.,
            )

        x = jnp.arange(3, dtype=jnp.float32)
        w = jnp.eye(3)
        closed = jax.make_jaxpr(f)(x, w)
        conv = _convert(closed)
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert 'jit' not in names and 'pjit' not in names
        for xi in (x + 10., x + 0.1, x - 10.):
            assert jnp.allclose(_eval(conv, xi, w)[0], f(xi, w))

    def test_safety_gate_effects_in_branch(self):
        def f(x, w):
            def true_fn():
                jax.debug.print('branch taken: {}', x[0])
                return x * 2.

            return jax.lax.cond(
                jnp.sum(x) > 0., true_fn, lambda: braintrace.matmul(x, w)
            )

        closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.eye(3))
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT if-converted'):
                conv = _convert(closed)
        assert 'cond' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.COND_CONVERSION_SKIPPED in kinds

    def test_safety_gate_while_in_branch(self):
        def f(x, w):
            def true_fn():
                return jax.lax.while_loop(
                    lambda v: jnp.sum(v) < 10., lambda v: v + 1., x
                )

            return jax.lax.cond(
                jnp.sum(x) > 0., true_fn, lambda: braintrace.matmul(x, w)
            )

        closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.eye(3))
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT if-converted'):
                conv = _convert(closed)
        assert 'cond' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.COND_CONVERSION_SKIPPED in kinds

    def test_safety_gate_scan_in_branch(self):
        def f(x, w):
            def true_fn():
                return jax.lax.scan(lambda c, _: (c * 1.1, None), x, length=3)[0]

            return jax.lax.cond(
                jnp.sum(x) > 0., true_fn, lambda: braintrace.matmul(x, w)
            )

        closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.eye(3))
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT if-converted'):
                conv = _convert(closed)
        assert 'cond' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.COND_CONVERSION_SKIPPED in kinds

    def test_skipped_cond_warns_once_across_fixpoint_iterations(self):
        # A convertible cond hiding another cond inside a jit forces >= 2
        # fixpoint iterations; the unsafe cond must be reported once, not
        # once per iteration.
        @jax.jit
        def jitted(a, w):
            return jax.lax.cond(
                jnp.max(a) > 2.,
                lambda: braintrace.matmul(a, w),
                lambda: a * 5.,
            )

        def f(x, w):
            unsafe = jax.lax.cond(
                jnp.sum(x) > 1.,
                lambda: jax.lax.while_loop(
                    lambda v: jnp.sum(v) < 10., lambda v: v + 1., x),
                lambda: braintrace.matmul(x, w),
            )
            safe = jax.lax.cond(
                jnp.sum(x) > 0.,
                lambda: jitted(x, w),
                lambda: x * 2.,
            )
            return unsafe + safe

        closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.eye(3))
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT if-converted'):
                _convert(closed)
        n_skip = sum(
            r.kind is DiagnosticKind.COND_CONVERSION_SKIPPED
            for r in reporter.records()
        )
        assert n_skip == 1

    def test_conversion_emits_info_diagnostic(self):
        _, closed, x, w = self._etp_cond_jaxpr()
        with diagnostic_context() as reporter:
            _convert(closed)
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.COND_IF_CONVERTED in kinds

    def test_deterministic_output(self):
        _, closed, x, w = self._etp_cond_jaxpr()
        # String form normalizes var names positionally, so equality pins
        # the full equation *and wiring* order, not just the primitives.
        assert str(_convert(closed).jaxpr) == str(_convert(closed).jaxpr)


class _CondGateCell(brainstate.nn.Module):
    """h' = tanh(cond(sum(x) > 0, etp_mv(x, w_a), etp_mv(x, w_b)) + 0.9 h)."""

    def __init__(self, n_in, n_rec):
        super().__init__()
        self.w_a = brainstate.ParamState(brainstate.random.randn(n_in, n_rec) * 0.1)
        self.w_b = brainstate.ParamState(brainstate.random.randn(n_in, n_rec) * 0.1)
        self.n_rec = n_rec

    def init_state(self, batch_size=None, **kw):
        shape = (self.n_rec,) if batch_size is None else (batch_size, self.n_rec)
        self.h = brainstate.HiddenState(jnp.zeros(shape))

    def update(self, x):
        u_val = jax.lax.cond(
            jnp.sum(x) > 0.,
            lambda: braintrace.matmul(x, self.w_a.value),
            lambda: braintrace.matmul(x, self.w_b.value),
        )
        self.h.value = jnp.tanh(u_val + 0.9 * self.h.value)
        return self.h.value


class _SelectGateCell(brainstate.nn.Module):
    """Hand-flattened equivalent of :class:`_CondGateCell`."""

    def __init__(self, n_in, n_rec):
        super().__init__()
        self.w_a = brainstate.ParamState(brainstate.random.randn(n_in, n_rec) * 0.1)
        self.w_b = brainstate.ParamState(brainstate.random.randn(n_in, n_rec) * 0.1)
        self.n_rec = n_rec

    def init_state(self, batch_size=None, **kw):
        shape = (self.n_rec,) if batch_size is None else (batch_size, self.n_rec)
        self.h = brainstate.HiddenState(jnp.zeros(shape))

    def update(self, x):
        idx = (jnp.sum(x) > 0.).astype(jnp.int32)
        y_a = braintrace.matmul(x, self.w_a.value)
        y_b = braintrace.matmul(x, self.w_b.value)
        # cond(pred, true_fn, false_fn): branches[0] is false_fn -> w_b.
        u_val = jax.lax.select_n(idx, y_b, y_a)
        self.h.value = jnp.tanh(u_val + 0.9 * self.h.value)
        return self.h.value


class TestCondModelCompilation:
    """A model with ETP ops inside `lax.cond` must compile identically to the
    hand-flattened select model (spec Phase 1)."""

    N_IN, N_REC = 3, 4

    def _graph_for(self, cell_cls, **compile_kwargs):
        with brainstate.random.seed_context(42):
            cell = cell_cls(self.N_IN, self.N_REC)
        brainstate.nn.init_all_states(cell)
        x = jnp.ones(self.N_IN)
        return braintrace.compile_etrace_graph(cell, x, **compile_kwargs)

    def test_cond_model_compiles_without_cond_eqn(self):
        graph = self._graph_for(_CondGateCell)
        names = [eqn.primitive.name for eqn in graph.module_info.jaxpr.eqns]
        assert 'cond' not in names
        assert 'select_n' in names

    def test_relation_parity_with_select_model(self):
        g_cond = self._graph_for(_CondGateCell)
        g_select = self._graph_for(_SelectGateCell)
        assert len(g_cond.hidden_param_op_relations) == 2
        assert len(g_select.hidden_param_op_relations) == 2
        cond_prims = {r.primitive for r in g_cond.hidden_param_op_relations}
        select_prims = {r.primitive for r in g_select.hidden_param_op_relations}
        assert cond_prims == select_prims

    def test_conversion_diagnostic_recorded(self):
        graph = self._graph_for(_CondGateCell)
        records = graph.explain(kind=DiagnosticKind.COND_IF_CONVERTED)
        assert len(records) == 1

    def test_opaque_policy_keeps_existing_error(self):
        with pytest.raises(NotImplementedError, match='cond'):
            with pytest.warns(UserWarning):
                self._graph_for(
                    _CondGateCell,
                    control_flow=ControlFlowPolicy(cond='opaque'),
                )

    def test_drtrl_gradient_parity_with_select_model(self):
        def build_and_grads(cell_cls):
            with brainstate.random.seed_context(42):
                model = cell_cls(self.N_IN, self.N_REC)
            learner = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros(self.N_IN))
            weights = model.states(brainstate.ParamState)
            with brainstate.random.seed_context(7):
                xs = brainstate.random.randn(6, self.N_IN)

            def total_loss(xs):
                def step(carry, x):
                    out = learner(x)
                    return carry, jnp.mean(jnp.asarray(out) ** 2)

                _, ls = brainstate.transform.scan(step, None, xs)
                return jnp.sum(ls)

            return brainstate.transform.grad(total_loss, weights)(xs)

        g_cond = build_and_grads(_CondGateCell)
        g_select = build_and_grads(_SelectGateCell)
        leaves_cond = jax.tree.leaves(g_cond)
        leaves_select = jax.tree.leaves(g_select)
        assert leaves_cond and len(leaves_cond) == len(leaves_select)
        for a, b in zip(leaves_cond, leaves_select):
            assert jnp.allclose(a, b, atol=1e-6), (a - b)
        # Gradients must be nonzero: both branches are exercised by the inputs.
        assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in leaves_cond)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
