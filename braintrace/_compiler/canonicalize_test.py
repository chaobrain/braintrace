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

import warnings

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._compiler.canonicalize import (
    ControlFlowPolicy,
    DEFAULT_CONTROL_FLOW_POLICY,
    canonicalize_control_flow,
    if_convert_conds,
    unroll_inner_scans,
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
        assert DEFAULT_CONTROL_FLOW_POLICY.while_hidden == 'opaque-fwd'
        assert DEFAULT_CONTROL_FLOW_POLICY.etp_in_control_flow == 'error'

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

    def test_scan_descent_default_off(self):
        assert DEFAULT_CONTROL_FLOW_POLICY.scan_descent == 'off'

    def test_scan_descent_accepts_auto(self):
        assert ControlFlowPolicy(scan_descent='auto').scan_descent == 'auto'

    def test_scan_descent_rejects_unknown_value(self):
        with pytest.raises(ValueError, match='scan_descent'):
            ControlFlowPolicy(scan_descent='yes-please')


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

    def test_safety_gate_unrollable_scan_in_branch_no_longer_skips(self):
        # Phase 2 gate revision: a branch scan that unroll_inner_scans could
        # flatten (short, effect-free, while-free) no longer blocks
        # conversion. if_convert_conds alone leaves the surfaced scan in
        # place; canonicalize_control_flow flattens it too.
        def f(x, w):
            def true_fn():
                return jax.lax.scan(lambda c, _: (c * 1.1, None), x, length=3)[0]

            return jax.lax.cond(
                jnp.sum(x) > 0., true_fn, lambda: braintrace.matmul(x, w)
            )

        closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.eye(3))
        conv = _convert(closed)
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert 'scan' in names  # surfaced, not unrolled by this pass alone
        for xi in (jnp.ones(3), -jnp.ones(3)):
            assert jnp.allclose(_eval(conv, xi, jnp.eye(3))[0], f(xi, jnp.eye(3)))

    def test_safety_gate_unrollable_scan_gate_with_ineligible_scan(self):
        # A branch scan the unroller would REFUSE (length > limit) still
        # blocks conversion, exactly as in Phase 1.
        def f(x, w):
            def true_fn():
                return jax.lax.scan(lambda c, _: (c * 1.1, None), x, length=64)[0]

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


def _unroll(closed_jaxpr, weights=(), hiddens_in=(), hiddens_out=(), policy=None):
    kwargs = dict(
        weight_invars=set(weights),
        hidden_invars=set(hiddens_in),
        hidden_outvars=set(hiddens_out),
    )
    if policy is not None:
        kwargs['policy'] = policy
    return unroll_inner_scans(closed_jaxpr, **kwargs)


class TestUnrollInnerScans:
    """Unit tests of the scan -> flat unrolled equation sequence rewrite."""

    L = 3

    def _etp_scan_jaxpr(self):
        def f(w, h0, xs):
            def body(h, x):
                h = jnp.tanh(braintrace.matmul(x, w) + braintrace.matmul(h, w))
                return h, 2.0 * h
            return jax.lax.scan(body, h0, xs)

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs = jnp.arange(self.L * 3, dtype=jnp.float32).reshape(self.L, 3) * 0.1
        closed = jax.make_jaxpr(f)(w, h0, xs)
        return f, closed, w, h0, xs

    def _unroll_etp(self, closed):
        return _unroll(
            closed,
            weights=[closed.jaxpr.invars[0]],
            hiddens_in=[closed.jaxpr.invars[1]],
            hiddens_out=[closed.jaxpr.outvars[0]],
        )

    def test_etp_scan_is_unrolled(self):
        f, closed, w, h0, xs = self._etp_scan_jaxpr()
        with diagnostic_context() as reporter:
            conv = self._unroll_etp(closed)
        names = _primitive_names(conv)
        assert 'scan' not in names
        assert names.count('etp_mv') == 2 * self.L
        kinds = [r.kind for r in reporter.records()]
        assert kinds.count(DiagnosticKind.SCAN_UNROLLED) == 1

    def test_outvars_and_invars_preserved_by_identity(self):
        f, closed, w, h0, xs = self._etp_scan_jaxpr()
        conv = self._unroll_etp(closed)
        assert all(a is b for a, b in zip(conv.jaxpr.invars, closed.jaxpr.invars))
        assert all(a is b for a, b in zip(conv.jaxpr.outvars, closed.jaxpr.outvars))

    def test_value_equivalence(self):
        f, closed, w, h0, xs = self._etp_scan_jaxpr()
        conv = self._unroll_etp(closed)
        h_ref, ys_ref = f(w, h0, xs)
        h_got, ys_got = _eval(conv, w, h0, xs)
        assert jnp.allclose(h_got, h_ref)
        assert jnp.allclose(ys_got, ys_ref)

    def test_gradient_equivalence(self):
        f, closed, w, h0, xs = self._etp_scan_jaxpr()
        conv = self._unroll_etp(closed)

        def loss_ref(w_, h_, xs_):
            h, ys = f(w_, h_, xs_)
            return jnp.sum(h ** 2) + jnp.sum(ys ** 2)

        def loss_conv(w_, h_, xs_):
            h, ys = _eval(conv, w_, h_, xs_)
            return jnp.sum(h ** 2) + jnp.sum(ys ** 2)

        for argnum in (0, 1, 2):
            g_ref = jax.grad(loss_ref, argnum)(w, h0, xs)
            g_conv = jax.grad(loss_conv, argnum)(w, h0, xs)
            assert jnp.allclose(g_ref, g_conv, atol=1e-6)

    def test_multiple_carries_and_ys(self):
        def f(w, a0, b0, xs):
            def body(carry, x):
                a, b = carry
                a2 = jnp.tanh(braintrace.matmul(x, w) + a)
                b2 = b + x
                return (a2, b2), (a2 * 2.0, b2 - 1.0)
            return jax.lax.scan(body, (a0, b0), xs)

        w = jnp.eye(3) * 0.3
        a0, b0 = jnp.zeros(3), jnp.ones(3)
        xs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) * 0.1
        closed = jax.make_jaxpr(f)(w, a0, b0, xs)
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' not in _primitive_names(conv)
        ref = f(w, a0, b0, xs)
        got = _eval(conv, w, a0, b0, xs)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_zero_xs_length_scan(self):
        def f(w, h0):
            def body(h, _):
                return jnp.tanh(braintrace.matmul(h, w)), h * 2.0
            return jax.lax.scan(body, h0, None, length=self.L)

        w = jnp.eye(3) * 0.5
        h0 = jnp.ones(3)
        closed = jax.make_jaxpr(f)(w, h0)
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' not in _primitive_names(conv)
        ref = f(w, h0)
        got = _eval(conv, w, h0)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_length_one(self):
        def f(w, h0, xs):
            def body(h, x):
                h = jnp.tanh(braintrace.matmul(x, w) + h)
                return h, h
            return jax.lax.scan(body, h0, xs)

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs = jnp.ones((1, 3))
        closed = jax.make_jaxpr(f)(w, h0, xs)
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' not in _primitive_names(conv)
        ref = f(w, h0, xs)
        got = _eval(conv, w, h0, xs)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_reverse_scan(self):
        def f(w, h0, xs):
            def body(h, x):
                h = jnp.tanh(braintrace.matmul(x, w) + h)
                return h, h * 2.0
            return jax.lax.scan(body, h0, xs, reverse=True)

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) * 0.1
        closed = jax.make_jaxpr(f)(w, h0, xs)
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' not in _primitive_names(conv)
        ref = f(w, h0, xs)
        got = _eval(conv, w, h0, xs)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_duplicate_body_outvars(self):
        # A body returning its new carry AS the y output produces the same
        # Var twice in body.outvars — the natural for_loop shape.
        def f(w, h0, xs):
            def body(h, x):
                h = jnp.tanh(braintrace.matmul(x, w) + h)
                return h, h
            return jax.lax.scan(body, h0, xs)

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) * 0.1
        closed = jax.make_jaxpr(f)(w, h0, xs)
        body = [e for e in closed.jaxpr.eqns if e.primitive.name == 'scan'][0]
        body_outvars = body.params['jaxpr'].jaxpr.outvars
        assert body_outvars[0] is body_outvars[1]  # fixture really is degenerate
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' not in _primitive_names(conv)
        ref = f(w, h0, xs)
        got = _eval(conv, w, h0, xs)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_passthrough_carry(self):
        # The body returns its carry INPUT unchanged: the body outvar is a
        # body invar, so the final carry resolves to the outer init atom.
        def f(w, h0, xs):
            def body(h, x):
                return h, braintrace.matmul(x, w) + h
            return jax.lax.scan(body, h0, xs)

        w = jnp.eye(3) * 0.5
        h0 = jnp.ones(3)
        xs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) * 0.1
        closed = jax.make_jaxpr(f)(w, h0, xs)
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' not in _primitive_names(conv)
        ref = f(w, h0, xs)
        got = _eval(conv, w, h0, xs)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_unused_ys_not_stacked(self):
        def f(w, h0, xs):
            def body(h, x):
                h = jnp.tanh(braintrace.matmul(x, w) + h)
                return h, h * 2.0
            h, _ = jax.lax.scan(body, h0, xs)
            return h

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs = jnp.ones((3, 3))
        closed = jax.make_jaxpr(f)(w, h0, xs)
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        names = _primitive_names(conv)
        assert 'scan' not in names
        assert 'concatenate' not in names
        assert jnp.allclose(_eval(conv, w, h0, xs)[0], f(w, h0, xs))

    def test_relevance_gate_irrelevant_scan_unchanged(self):
        def f(x):
            return jax.lax.scan(lambda c, v: (c + v, c), jnp.zeros(()), x)

        closed = jax.make_jaxpr(f)(jnp.ones(4))
        out = _unroll(closed)
        assert out is closed

    def test_skip_length_exceeds_limit(self):
        f, closed, w, h0, xs = self._etp_scan_jaxpr()
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT unrolled'):
                conv = _unroll(
                    closed,
                    weights=[closed.jaxpr.invars[0]],
                    policy=ControlFlowPolicy(scan_unroll_limit=self.L - 1),
                )
        assert 'scan' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert kinds.count(DiagnosticKind.SCAN_UNROLL_SKIPPED) == 1

    def test_skip_effects_in_body(self):
        def f(w, h0, xs):
            def body(h, x):
                jax.debug.print('h0: {}', h[0])
                return jnp.tanh(braintrace.matmul(x, w) + h), h
            return jax.lax.scan(body, h0, xs)

        closed = jax.make_jaxpr(f)(jnp.eye(3), jnp.zeros(3), jnp.ones((2, 3)))
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT unrolled'):
                conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.SCAN_UNROLL_SKIPPED in kinds

    def test_skip_while_in_body(self):
        def f(w, h0, xs):
            def body(h, x):
                h = jax.lax.while_loop(
                    lambda v: jnp.sum(v) < 1., lambda v: v + 0.5, h
                )
                return jnp.tanh(braintrace.matmul(x, w) + h), h
            return jax.lax.scan(body, h0, xs)

        closed = jax.make_jaxpr(f)(jnp.eye(3), jnp.zeros(3), jnp.ones((2, 3)))
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT unrolled'):
                conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.SCAN_UNROLL_SKIPPED in kinds

    def test_policy_limit_zero_disables(self):
        f, closed, w, h0, xs = self._etp_scan_jaxpr()
        out = _unroll(
            closed,
            weights=[closed.jaxpr.invars[0]],
            policy=ControlFlowPolicy(scan_unroll_limit=0),
        )
        assert out is closed

    def test_weights_as_xs_skipped(self):
        # Scanning over a stacked weight: the xs invar IS a weight invar.
        def f(ws, h0):
            def body(h, w_t):
                return jnp.tanh(braintrace.matmul(h, w_t)), h
            return jax.lax.scan(body, h0, ws)

        ws = jnp.stack([jnp.eye(3)] * 2) * 0.5
        h0 = jnp.ones(3)
        closed = jax.make_jaxpr(f)(ws, h0)
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='scans over a trainable weight'):
                conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert kinds.count(DiagnosticKind.RELATION_EXCLUDED_SLICED_WEIGHT) == 1

    def test_weight_derived_xs_skipped(self):
        # The xs value is COMPUTED from a weight before the scan; backward
        # reachability must still catch it (direct membership would not).
        def f(w, h0):
            ws = jnp.stack([w, w * 2.0])

            def body(h, w_t):
                return jnp.tanh(braintrace.matmul(h, w_t)), h
            return jax.lax.scan(body, h0, ws)

        closed = jax.make_jaxpr(f)(jnp.eye(3), jnp.ones(3))
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='scans over a trainable weight'):
                conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        assert 'scan' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.RELATION_EXCLUDED_SLICED_WEIGHT in kinds

    def test_jit_inside_body_inlined(self):
        @jax.jit
        def helper(a, b):
            return a + b

        def f(w, h0, xs):
            def body(h, x):
                h = jnp.tanh(helper(braintrace.matmul(x, w), h))
                return h, h
            return jax.lax.scan(body, h0, xs)

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs = jnp.ones((2, 3))
        closed = jax.make_jaxpr(f)(w, h0, xs)
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        names = _primitive_names(conv)
        assert 'scan' not in names
        assert 'jit' not in names and 'pjit' not in names
        ref = f(w, h0, xs)
        got = _eval(conv, w, h0, xs)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_nested_eligible_scans_unrolled(self):
        def f(w, h0, xs):
            def outer_body(h, x):
                def inner_body(g, _):
                    return jnp.tanh(braintrace.matmul(g, w)), None
                g, _ = jax.lax.scan(inner_body, h + x, None, length=2)
                return g, g
            return jax.lax.scan(outer_body, h0, xs)

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs = jnp.ones((2, 3)) * 0.1
        closed = jax.make_jaxpr(f)(w, h0, xs)
        conv = _unroll(closed, weights=[closed.jaxpr.invars[0]])
        names = _primitive_names(conv)
        assert 'scan' not in names
        assert names.count('etp_mv') == 4  # 2 outer x 2 inner
        ref = f(w, h0, xs)
        got = _eval(conv, w, h0, xs)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_skip_warning_once_across_fixpoint(self):
        # One eligible + one ineligible scan: the second sweep (triggered by
        # the first unroll) must not re-warn for the ineligible one.
        def f(w, h0, xs_short, xs_long):
            def body(h, x):
                return jnp.tanh(braintrace.matmul(x, w) + h), h
            h1, _ = jax.lax.scan(body, h0, xs_short)
            h2, _ = jax.lax.scan(body, h1, xs_long)
            return h2

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs_short = jnp.ones((2, 3))
        xs_long = jnp.ones((5, 3))
        closed = jax.make_jaxpr(f)(w, h0, xs_short, xs_long)
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT unrolled'):
                conv = _unroll(
                    closed,
                    weights=[closed.jaxpr.invars[0]],
                    policy=ControlFlowPolicy(scan_unroll_limit=3),
                )
        names = _primitive_names(conv)
        assert names.count('scan') == 1
        kinds = [r.kind for r in reporter.records()]
        assert kinds.count(DiagnosticKind.SCAN_UNROLL_SKIPPED) == 1

    def test_determinism(self):
        f, closed, w, h0, xs = self._etp_scan_jaxpr()
        conv1 = self._unroll_etp(closed)
        conv2 = self._unroll_etp(closed)
        assert _primitive_names(conv1) == _primitive_names(conv2)
        assert str(conv1.jaxpr) == str(conv2.jaxpr)


class TestCanonicalizeControlFlow:
    """The joint cond + scan fixpoint driver."""

    def test_cond_inside_scan_body(self):
        def f(w, h0, xs):
            def body(h, x):
                drive = jax.lax.cond(
                    jnp.sum(x) > 0.,
                    lambda: braintrace.matmul(x, w),
                    lambda: x * 2.,
                )
                return jnp.tanh(drive + h), h
            return jax.lax.scan(body, h0, xs)

        w = jnp.eye(3) * 0.5
        h0 = jnp.zeros(3)
        xs = jnp.stack([jnp.ones(3), -jnp.ones(3)])
        closed = jax.make_jaxpr(f)(w, h0, xs)
        conv = canonicalize_control_flow(
            closed,
            weight_invars={closed.jaxpr.invars[0]},
            hidden_invars=set(),
            hidden_outvars=set(),
            policy=DEFAULT_CONTROL_FLOW_POLICY,
        )
        names = _primitive_names(conv)
        assert 'scan' not in names
        assert 'cond' not in names
        assert 'select_n' in names
        ref = f(w, h0, xs)
        got = _eval(conv, w, h0, xs)
        for r, g in zip(jax.tree.leaves(ref), got):
            assert jnp.allclose(g, r)

    def test_eligible_scan_inside_cond_branch(self):
        def f(x, w, h0):
            def true_fn():
                def body(h, _):
                    return jnp.tanh(braintrace.matmul(h, w)), None
                h, _ = jax.lax.scan(body, h0, None, length=2)
                return h

            return jax.lax.cond(jnp.sum(x) > 0., true_fn, lambda: h0 * 2.)

        x = jnp.ones(3)
        w = jnp.eye(3) * 0.5
        h0 = jnp.ones(3)
        closed = jax.make_jaxpr(f)(x, w, h0)
        conv = canonicalize_control_flow(
            closed,
            weight_invars={closed.jaxpr.invars[1]},
            hidden_invars=set(),
            hidden_outvars=set(),
            policy=DEFAULT_CONTROL_FLOW_POLICY,
        )
        names = _primitive_names(conv)
        assert 'cond' not in names
        assert 'scan' not in names
        for xi in (x, -x):
            assert jnp.allclose(_eval(conv, xi, w, h0)[0], f(xi, w, h0))

    def test_ineligible_scan_inside_cond_branch_skips_cond(self):
        def f(x, w, h0):
            def true_fn():
                def body(h, _):
                    return jnp.tanh(braintrace.matmul(h, w)), None
                h, _ = jax.lax.scan(body, h0, None, length=64)
                return h

            return jax.lax.cond(jnp.sum(x) > 0., true_fn, lambda: h0 * 2.)

        closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.eye(3), jnp.ones(3))
        with diagnostic_context() as reporter:
            with pytest.warns(UserWarning, match='NOT if-converted'):
                conv = canonicalize_control_flow(
                    closed,
                    weight_invars={closed.jaxpr.invars[1]},
                    hidden_invars=set(),
                    hidden_outvars=set(),
                    policy=DEFAULT_CONTROL_FLOW_POLICY,
                )
        assert 'cond' in _primitive_names(conv)
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.COND_CONVERSION_SKIPPED in kinds

    def test_limit_zero_reproduces_phase1_gating(self):
        # With unrolling disabled, a cond guarding ANY inner scan must stay
        # opaque, exactly as in Phase 1.
        def f(x, w, h0):
            def true_fn():
                def body(h, _):
                    return jnp.tanh(braintrace.matmul(h, w)), None
                h, _ = jax.lax.scan(body, h0, None, length=2)
                return h

            return jax.lax.cond(jnp.sum(x) > 0., true_fn, lambda: h0 * 2.)

        closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.eye(3), jnp.ones(3))
        with pytest.warns(UserWarning, match='NOT if-converted'):
            conv = canonicalize_control_flow(
                closed,
                weight_invars={closed.jaxpr.invars[1]},
                hidden_invars=set(),
                hidden_outvars=set(),
                policy=ControlFlowPolicy(scan_unroll_limit=0),
            )
        names = _primitive_names(conv)
        assert 'cond' in names


class _InnerLoopCell(brainstate.nn.Module):
    """Cell whose one-step update runs an inner ``for_loop`` of ``loops``
    sub-steps, each applying two ETP matmuls to the hidden state."""

    def __init__(self, n_rec: int, loops: int = 3):
        super().__init__()
        self.loops = loops
        self.w = brainstate.ParamState(0.1 * brainstate.random.randn(n_rec, n_rec))
        self.h = brainstate.HiddenState(jnp.zeros(n_rec))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        def step(i):
            self.h.value = jnp.tanh(
                braintrace.matmul(x, self.w.value)
                + braintrace.matmul(self.h.value, self.w.value)
            )
            return self.h.value

        outs = brainstate.transform.for_loop(step, jnp.arange(self.loops))
        return outs[-1]


class _UnrolledLoopCell(brainstate.nn.Module):
    """Hand-flattened twin of :class:`_InnerLoopCell`: the same ``loops``
    sub-steps written out at trace time (no runtime loop)."""

    def __init__(self, n_rec: int, loops: int = 3):
        super().__init__()
        self.loops = loops
        self.w = brainstate.ParamState(0.1 * brainstate.random.randn(n_rec, n_rec))
        self.h = brainstate.HiddenState(jnp.zeros(n_rec))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        for _ in range(self.loops):
            self.h.value = jnp.tanh(
                braintrace.matmul(x, self.w.value)
                + braintrace.matmul(self.h.value, self.w.value)
            )
        return self.h.value


class TestScanModelCompilation:
    """A model with ETP ops inside a `for_loop`/`scan` body must compile
    identically to the hand-unrolled twin (spec Phase 2)."""

    N_REC, LOOPS = 4, 3

    def _graph_for(self, cell_cls, **compile_kwargs):
        with brainstate.random.seed_context(42):
            cell = cell_cls(self.N_REC, self.LOOPS)
        brainstate.nn.init_all_states(cell)
        x = jnp.ones(self.N_REC)
        return braintrace.compile_etrace_graph(cell, x, **compile_kwargs)

    def test_scan_model_compiles_without_scan_eqn(self):
        graph = self._graph_for(_InnerLoopCell)
        names = [eqn.primitive.name for eqn in graph.module_info.jaxpr.eqns]
        assert 'scan' not in names
        assert names.count('etp_mv') == 2 * self.LOOPS

    def test_relation_parity_with_unrolled_model(self):
        # Only the LAST sub-step's two ETP calls become relations: earlier
        # sub-steps reach the hidden outvar exclusively through later ETP
        # ops (the weight -> weight -> hidden invariant excludes them). The
        # hand-flattened twin proves 2 is the canonical count.
        g_scan = self._graph_for(_InnerLoopCell)
        g_flat = self._graph_for(_UnrolledLoopCell)
        assert len(g_scan.hidden_param_op_relations) == 2
        assert len(g_flat.hidden_param_op_relations) == 2
        assert all(
            r.trainable_paths['weight'] == ('w',)
            for r in g_scan.hidden_param_op_relations
        )
        scan_prims = sorted(
            r.primitive.name for r in g_scan.hidden_param_op_relations
        )
        flat_prims = sorted(
            r.primitive.name for r in g_flat.hidden_param_op_relations
        )
        assert scan_prims == flat_prims

    def test_unroll_diagnostic_recorded(self):
        graph = self._graph_for(_InnerLoopCell)
        records = graph.explain(kind=DiagnosticKind.SCAN_UNROLLED)
        assert len(records) == 1

    def test_limit_zero_policy_keeps_existing_error(self):
        with pytest.raises(NotImplementedError, match='scan'):
            with pytest.warns(UserWarning):
                self._graph_for(
                    _InnerLoopCell,
                    control_flow=ControlFlowPolicy(scan_unroll_limit=0),
                )

    def test_while_body_model_keeps_existing_error(self):
        # `while_loop` is out of Phase 2 scope: an ETP-relevant while in the
        # update must still hard-error, untouched by canonicalization.
        class WhileCell(brainstate.nn.Module):
            def __init__(self, n, loops):
                super().__init__()
                self.loops = loops
                self.w = brainstate.ParamState(0.1 * brainstate.random.randn(n, n))
                self.h = brainstate.HiddenState(jnp.zeros(n))

            def update(self, x):
                def cond_fn(carry):
                    i, _ = carry
                    return i < self.loops

                def body_fn(carry):
                    i, h = carry
                    h = jnp.tanh(
                        braintrace.matmul(x, self.w.value)
                        + braintrace.matmul(h, self.w.value)
                    )
                    return i + 1, h

                _, h = jax.lax.while_loop(cond_fn, body_fn, (0, self.h.value))
                self.h.value = h
                return h

        with pytest.raises(NotImplementedError, match='while'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                self._graph_for(WhileCell)

    def test_drtrl_gradient_parity_with_unrolled_model(self):
        def build_and_grads(cell_cls):
            with brainstate.random.seed_context(42):
                model = cell_cls(self.N_REC, self.LOOPS)
            learner = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros(self.N_REC))
            weights = model.states(brainstate.ParamState)
            with brainstate.random.seed_context(7):
                xs = brainstate.random.randn(6, self.N_REC)

            def total_loss(xs):
                def step(carry, x):
                    out = learner(x)
                    return carry, jnp.mean(jnp.asarray(out) ** 2)

                _, ls = brainstate.transform.scan(step, None, xs)
                return jnp.sum(ls)

            return brainstate.transform.grad(total_loss, weights)(xs)

        g_scan = build_and_grads(_InnerLoopCell)
        g_flat = build_and_grads(_UnrolledLoopCell)
        leaves_scan = jax.tree.leaves(g_scan)
        leaves_flat = jax.tree.leaves(g_flat)
        assert leaves_scan and len(leaves_scan) == len(leaves_flat)
        for a, b in zip(leaves_scan, leaves_flat):
            assert jnp.allclose(a, b, atol=1e-6), (a - b)
        assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in leaves_scan)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
