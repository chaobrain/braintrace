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

"""Equivalence tests for the fused multi-step eligibility-trace scan.

For multi-step (``MultiStepData``) input the graph executor can fuse the
per-step eligibility-trace roll into its over-time scan (one ``lax.scan``
instead of two), driven by the subclass hook ``_make_etrace_stepper``. Fusing
must be a pure performance change: the model outputs, the final eligibility
trace, and the weight gradients must match the legacy two-scan path. The legacy
path is forced here by stubbing ``_make_etrace_stepper`` to return ``None``.

``test_fused_*`` exercises both fused executor entry points:
  * the forward outputs / final trace go through ``solve_h2w_h2h_jacobian``
    (the non-differentiated ``_update_fn``), and
  * the gradients go through ``solve_h2w_h2h_l2h_jacobian`` (the custom-VJP
    ``_update_fn_fwd`` / ``_update_fn_bwd``), which is where the trace carry is
    threaded through a ``jax.vjp``-traced scan.
"""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._etrace_algorithms.oracle import (
    assert_param_gradients_close,
    bptt_param_gradients,
)
from braintrace._etrace_algorithms.oracle_models import tanh_rnn


def _inputs(T, n_in, seed=42):
    return jnp.asarray(np.random.RandomState(seed).randn(T, n_in).astype('float32'))


# Both exact/diagonal algorithms that use the canonical scan-based trace roll.
_ALGOS = {
    'D_RTRL': lambda m: braintrace.ParamDimVjpAlgorithm(m, vjp_method='multi-step'),
    'pp_prop': lambda m: braintrace.pp_prop(m, decay_or_rank=0.9, vjp_method='multi-step'),
}


def _build(algo_factory, inputs, *, fuse):
    # The factory is deterministic (fixed seed), so the fused and legacy builds
    # start from identical weights and states.
    model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()
    if not fuse:
        # Force the legacy two-scan path by dropping the stepper at the executor
        # boundary (so the executor stacks the per-step Jacobians and
        # ``_update_etrace_data`` rolls the trace in a second scan). We patch the
        # executor rather than ``_make_etrace_stepper`` because the legacy
        # ``_update_etrace_data`` itself builds its scan body via that hook.
        exe = algo.graph_executor
        _orig_j = exe.solve_h2w_h2h_jacobian
        _orig_l = exe.solve_h2w_h2h_l2h_jacobian
        exe.solve_h2w_h2h_jacobian = (
            lambda *a, **k: _orig_j(*a, **{**k, 'etrace_stepper': None, 'init_etrace': None})
        )
        exe.solve_h2w_h2h_l2h_jacobian = (
            lambda *a, **k: _orig_l(*a, **{**k, 'etrace_stepper': None, 'init_etrace': None})
        )
    return model, algo


def _assert_tree_close(a, b, *, atol, msg):
    leaves_a, leaves_b = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(leaves_a) == len(leaves_b), f'{msg}: pytree structure differs'
    for x, y in zip(leaves_a, leaves_b):
        x, y = jnp.asarray(x), jnp.asarray(y)
        maxdiff = float(jnp.max(jnp.abs(x - y))) if x.size else 0.0
        assert bool(jnp.allclose(x, y, atol=atol)), f'{msg}: maxabsdiff={maxdiff:.3e}'


@pytest.mark.parametrize('name', list(_ALGOS))
def test_fused_multistep_matches_legacy(name):
    algo_factory = _ALGOS[name]
    inputs = _inputs(8, 3)

    # --- forward: outputs + final eligibility trace (solve_h2w_h2h_jacobian) ---
    _, algo_f = _build(algo_factory, inputs, fuse=True)
    out_f = algo_f(braintrace.MultiStepData(inputs))
    trace_f = algo_f._get_etrace_data()

    _, algo_u = _build(algo_factory, inputs, fuse=False)
    out_u = algo_u(braintrace.MultiStepData(inputs))
    trace_u = algo_u._get_etrace_data()

    _assert_tree_close(out_f, out_u, atol=1e-5, msg=f'{name} outputs')
    _assert_tree_close(trace_f, trace_u, atol=1e-5, msg=f'{name} final trace')

    # --- gradients (solve_h2w_h2h_l2h_jacobian custom-VJP path) ---
    def grads(fuse):
        model, algo = _build(algo_factory, inputs, fuse=fuse)
        return brainstate.transform.grad(
            lambda seq: (algo(braintrace.MultiStepData(seq)) ** 2).sum(),
            model.states(brainstate.ParamState),
        )(inputs)

    assert_param_gradients_close(grads(True), grads(False), atol=1e-5)


def test_fused_d_rtrl_multistep_matches_bptt():
    """End-to-end: the fused multi-step D_RTRL path reproduces the exact BPTT gradient.

    (``oracle_test.test_d_rtrl_multistep_matches_bptt`` covers the same path, since
    multi-step now fuses; kept here to keep the fusion correctness self-contained.)
    """
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(8, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)

    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = braintrace.ParamDimVjpAlgorithm(model, vjp_method='multi-step')
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()
    g_fused = brainstate.transform.grad(
        lambda seq: (algo(braintrace.MultiStepData(seq)) ** 2).sum(),
        model.states(brainstate.ParamState),
    )(inputs)

    assert_param_gradients_close(g_fused, g_bptt, atol=1e-4)
