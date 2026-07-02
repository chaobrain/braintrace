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


def make_snn_scan_net(loops, n_rec=4, decay=0.9, seed=0):
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

            return brainstate.transform.for_loop(substep, jnp.arange(loops))[-1]

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
