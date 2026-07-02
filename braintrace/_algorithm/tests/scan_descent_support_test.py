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

"""Phase 4 structured-scan-descent correctness: stacked executor Jacobians,
substep trace fold, descended == unrolled == BPTT on diagonal-recurrence
bodies, and gating/diagnostic pins."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace import ControlFlowPolicy

DESCEND = lambda: ControlFlowPolicy(scan_unroll_limit=4, scan_descent='auto')
UNROLL = lambda: ControlFlowPolicy(scan_unroll_limit=16, scan_descent='off')


def make_snn_scan_net(loops, n_rec=4, decay=0.9, seed=0, carry_readout=False):
    """``carry_readout=True`` returns ``self.h.value`` after the loop (the
    scan's carry outvar) instead of the stacked ys slice ``outs[-1]``. The
    two are numerically identical, but in single-step (perturbation) mode
    only the carry readout routes the same-step loss through the per-step
    hidden perturbation of a *descended* scan — see
    ``test_single_step_ys_readout_drops_same_step_signal``."""
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            with brainstate.random.seed_context(seed):
                self.w = brainstate.ParamState(
                    0.1 * brainstate.random.randn(n_rec, n_rec))
            self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

        def update(self, x):
            x_row = x.reshape(1, -1)

            def substep(_):
                self.h.value = decay * self.h.value + jnp.tanh(
                    braintrace.matmul(x_row, self.w.value))
                return self.h.value

            outs = brainstate.transform.for_loop(substep, jnp.arange(loops))
            return self.h.value if carry_readout else outs[-1]

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _compiled_algo(net, policy, **kw):
    algo = braintrace.D_RTRL(net, vjp_method='multi-step',
                             control_flow=policy, **kw)
    algo.compile_graph(jnp.ones((4,), dtype='float32'))
    algo.init_etrace_state()
    return algo


class TestStackedJacobians:
    def test_descended_diag_is_decay_identity_per_substep(self):
        L, decay = 8, 0.9
        algo = _compiled_algo(make_snn_scan_net(loops=L, decay=decay), DESCEND())
        executor = algo.graph_executor
        # drive one real step through the executor plumbing to get temps
        (_, _, _, hid2w, hid2h, _) = executor.solve_h2w_h2h_jacobian(
            jnp.ones((4,), dtype='float32'))
        g = next(gr for gr in algo.graph.hidden_groups if gr.descent is not None)
        diag = hid2h[g.index]
        assert diag.shape == (L, 1, 4, 1, 1)
        # body h-path is `decay * h` -> D_tau == decay exactly, every substep
        np.testing.assert_allclose(np.asarray(diag), decay, atol=1e-6)

    def test_descended_df_matches_manual_tanh_prime(self):
        L = 8
        net = make_snn_scan_net(loops=L)
        algo = _compiled_algo(net, DESCEND())
        executor = algo.graph_executor
        (_, _, _, (xs, dfs), _, _) = executor.solve_h2w_h2h_jacobian(
            jnp.ones((4,), dtype='float32'))
        rel = next(r for r in algo.graph.hidden_param_op_relations
                   if r.control_flow_context is not None)
        from braintrace._misc import etrace_x_key, etrace_df_key
        x_stack = xs[etrace_x_key(rel.x_var)]
        assert x_stack.shape == (L, 1, 4)
        g = rel.hidden_groups[0]
        df_stack = dfs[etrace_df_key(rel.y_var, g.index)]
        assert df_stack.shape == (L, 1, 4, 1)
        # y_tau identical every substep here (x@w is substep-invariant), so
        # df must equal tanh'(y) with y = x_row @ w
        y = jnp.ones((1, 4)) @ np.asarray(net.w.value)
        expect = 1.0 - jnp.tanh(y) ** 2
        np.testing.assert_allclose(
            np.asarray(df_stack[0, ..., 0]), np.asarray(expect), atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(df_stack[-1, ..., 0]), np.asarray(expect), atol=1e-5)


class TestSubstepFold:
    def test_descended_trace_equals_sum_of_unrolled_traces(self):
        """Replay == unroll-in-time: after identical multi-step drives, the
        descended model's single trace equals the elementwise SUM of the
        unrolled twin's per-substep-relation traces (diagonal body, L=8)."""
        L = 8
        inputs = jnp.asarray(
            np.random.RandomState(7).randn(3, 4).astype('float32'))

        algo_d = _compiled_algo(make_snn_scan_net(loops=L, seed=0), DESCEND())
        algo_u = _compiled_algo(make_snn_scan_net(loops=L, seed=0), UNROLL())
        for algo in (algo_d, algo_u):
            algo(braintrace.MultiStepData(inputs))

        d_entries = list(algo_d.etrace_bwg.values())
        u_entries = list(algo_u.etrace_bwg.values())
        assert len(d_entries) == 1 and len(u_entries) == L

        d_val = d_entries[0].value['weight']
        u_sum = sum(e.value['weight'] for e in u_entries)
        np.testing.assert_allclose(
            np.asarray(d_val), np.asarray(u_sum), atol=1e-6)

    def test_io_dim_algorithm_still_gated(self):
        net = make_snn_scan_net(loops=40)
        algo = braintrace.pp_prop(net, decay_or_rank=16,
                                  vjp_method='multi-step',
                                  control_flow=DESCEND())
        with pytest.raises(NotImplementedError, match='scan descent'):
            algo.compile_graph(jnp.ones((4,), dtype='float32'))

    def test_orphan_descended_group_still_gated(self):
        """A body whose only weight is used through plain ops descends with
        a hidden group but ZERO relations; the group's Jacobians still carry
        the substep axis, so non-supporting algorithms must be gated on
        groups too, not just relations."""

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def substep(_):
                    # plain matmul: no ETP relation is produced
                    self.h.value = 0.9 * self.h.value + jnp.tanh(
                        x_row @ self.w.value)
                    return self.h.value

                return brainstate.transform.for_loop(
                    substep, jnp.arange(40))[-1]

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        algo = braintrace.pp_prop(net, decay_or_rank=16,
                                  vjp_method='multi-step',
                                  control_flow=DESCEND())
        with pytest.warns(UserWarning, match='no ETP relation'):
            with pytest.raises(NotImplementedError, match='scan descent'):
                algo.compile_graph(jnp.ones((4,), dtype='float32'))

    def test_orphan_descended_group_runs_under_d_rtrl(self):
        """Same plain-op-weight body under a supporting algorithm: compiles
        (with the exclusion warning), holds no traces for the excluded
        weight, and a forward update executes despite the substep axis on
        the orphan group's Jacobians."""

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                x_row = x.reshape(1, -1)

                def substep(_):
                    self.h.value = 0.9 * self.h.value + jnp.tanh(
                        x_row @ self.w.value)
                    return self.h.value

                return brainstate.transform.for_loop(
                    substep, jnp.arange(40))[-1]

        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        algo = braintrace.D_RTRL(net, vjp_method='multi-step',
                                 control_flow=DESCEND())
        with pytest.warns(UserWarning, match='no ETP relation'):
            algo.compile_graph(jnp.ones((4,), dtype='float32'))
        assert len(algo.etrace_bwg) == 0
        out = algo.update(jnp.ones((4,), dtype='float32'))
        assert jnp.shape(out) == (1, 4)


from braintrace._algorithm.oracle import (
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
    online_param_gradients,
)
from braintrace._algorithm.oracle_models import (
    scan_body_rnn,
    snn_scan_rnn,
    snn_scan_two_state_rnn,
)

ATOL_BPTT = 1e-4
# float32 accumulation-order noise between the folded (inner-scan) and the
# unrolled (flat-relation) trace paths; the design probes measured <=2e-5.
ATOL_EQUIV = 3e-5


def _inputs(T, n, seed=42):
    return jnp.asarray(np.random.RandomState(seed).randn(T, n).astype('float32'))


def _drtrl_descend(m):
    return braintrace.D_RTRL(m, vjp_method='multi-step', control_flow=DESCEND())


def _drtrl_unroll(m):
    return braintrace.D_RTRL(m, vjp_method='multi-step', control_flow=UNROLL())


class TestDescentCorrectness:
    def test_wholeseq_descended_matches_bptt_diagonal_body(self):
        spec = snn_scan_rnn(n_rec=4, loops=8, seed=0)
        inputs = _inputs(6, 4)
        g_bptt = bptt_param_gradients(spec.factory, inputs)
        g_onl = online_param_gradients(spec.factory, inputs,
                                       algo_factory=_drtrl_descend)
        assert_param_gradients_close(g_onl, g_bptt, atol=ATOL_BPTT)

    @pytest.mark.parametrize('chunk_size', [3, 1])
    def test_chunked_descended_matches_bptt(self, chunk_size):
        """THE trace oracle: chunk boundaries make the gradient depend on the
        folded eligibility trace (boundary term dL/dh0 . eps0)."""
        spec = snn_scan_rnn(n_rec=4, loops=8, seed=0)
        inputs = _inputs(6, 4)
        g_bptt = bptt_param_gradients(spec.factory, inputs)
        g_chunk = chunked_online_param_gradients(
            spec.factory, inputs, algo_factory=_drtrl_descend,
            chunk_size=chunk_size)
        assert_param_gradients_close(g_chunk, g_bptt, atol=ATOL_BPTT)

    def test_descended_equals_unrolled_gradients(self):
        spec = snn_scan_rnn(n_rec=4, loops=8, seed=0)
        inputs = _inputs(6, 4)
        g_d = chunked_online_param_gradients(
            spec.factory, inputs, algo_factory=_drtrl_descend, chunk_size=3)
        g_u = chunked_online_param_gradients(
            spec.factory, inputs, algo_factory=_drtrl_unroll, chunk_size=3)
        assert_param_gradients_close(g_d, g_u, atol=ATOL_EQUIV)

    def test_two_state_group_descended_chunked_matches_bptt(self):
        """num_state == 2 through the fold (SNN learning-signal axis pin)."""
        spec = snn_scan_two_state_rnn(n_rec=3, loops=8, seed=0)
        inputs = _inputs(6, 3)
        g_bptt = bptt_param_gradients(spec.factory, inputs)
        g_chunk = chunked_online_param_gradients(
            spec.factory, inputs, algo_factory=_drtrl_descend, chunk_size=3)
        assert_param_gradients_close(g_chunk, g_bptt, atol=ATOL_BPTT)

    def test_long_scan_L100_compiles_lean_and_matches_bptt_wholeseq(self):
        spec = snn_scan_rnn(n_rec=4, loops=100, seed=0)
        inputs = _inputs(4, 4)
        g_bptt = bptt_param_gradients(spec.factory, inputs)
        # default-limit policy: L=100 must descend, not unroll
        policy = ControlFlowPolicy(scan_descent='auto')
        algo_factory = lambda m: braintrace.D_RTRL(
            m, vjp_method='multi-step', control_flow=policy)
        g = online_param_gradients(spec.factory, inputs,
                                   algo_factory=algo_factory)
        assert_param_gradients_close(g, g_bptt, atol=ATOL_BPTT)
        # no unroll blowup
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = algo_factory(model)
        algo.compile_graph(inputs[0])
        assert len(algo.graph.module_info.jaxpr.eqns) < 60

    def test_mixing_body_descended_wholeseq_matches_bptt(self):
        """scan_body_rnn's body mixes h through an ETP matmul: whole-sequence
        multi-step is trace-independent, so descent must still be
        BPTT-exact. Chunked equality is NOT asserted (both compile paths
        approximate cross-substep credit there; documented divergence)."""
        spec = scan_body_rnn(n_rec=3, loops=8, seed=0)
        inputs = _inputs(6, 3)
        g_bptt = bptt_param_gradients(spec.factory, inputs)
        g = online_param_gradients(spec.factory, inputs,
                                   algo_factory=_drtrl_descend)
        assert_param_gradients_close(g, g_bptt, atol=ATOL_BPTT)

    @staticmethod
    def _one_step_grads(policy, carry_readout):
        x = jnp.ones((4,), dtype='float32')
        net = make_snn_scan_net(loops=8, seed=0, carry_readout=carry_readout)
        algo = braintrace.D_RTRL(net, vjp_method='single-step',
                                 control_flow=policy)
        algo.compile_graph(x)
        algo.init_etrace_state()
        params = net.states(brainstate.ParamState)
        return brainstate.transform.grad(
            lambda x_: (algo(x_) ** 2).sum(), params)(x)

    def test_single_step_vjp_descended_equals_unrolled_one_step(self):
        """Perturbation path through the descended compile (single-step mode
        reverse-differentiates THROUGH the scan; weights detached). The model
        must read the hidden state from the carry (``self.h.value`` after the
        loop) for the same-step learning signal to cross the perturbation —
        see the limitation pin below."""
        g_d = self._one_step_grads(DESCEND(), carry_readout=True)
        g_u = self._one_step_grads(UNROLL(), carry_readout=True)
        assert_param_gradients_close(g_d, g_u, atol=ATOL_EQUIV)

    def test_single_step_ys_readout_drops_same_step_signal(self):
        """v1 limitation pin: when the loss reads the hidden state through
        the descended scan's stacked ys output (``for_loop(...)[-1]``)
        instead of the carry, the same-step loss path bypasses the per-step
        hidden perturbation (which is added to the scan's *carry* outvar),
        so the single-step learning signal — and hence the one-step ETP
        gradient — is zero. This parallels the documented Phase-3
        while-hidden same-step limitation. Multi-step vjp is unaffected
        (``test_wholeseq_descended_matches_bptt_diagonal_body`` uses the ys
        readout). Fix at the model level: return ``self.h.value`` after the
        loop."""
        g_ys = self._one_step_grads(DESCEND(), carry_readout=False)
        np.testing.assert_allclose(np.asarray(g_ys[('w',)]), 0.0, atol=0.0)
        g_carry = self._one_step_grads(DESCEND(), carry_readout=True)
        assert float(np.linalg.norm(np.asarray(g_carry[('w',)]))) > 1.0


def test_default_policy_descends_long_scans():
    """Phase 4 default: an over-limit ETP scan compiles without any policy."""
    net = make_snn_scan_net(loops=40)
    algo = braintrace.D_RTRL(net, vjp_method='multi-step')
    algo.compile_graph(jnp.ones((4,), dtype='float32'))
    assert any(r.control_flow_context is not None
               for r in algo.graph.hidden_param_op_relations)
