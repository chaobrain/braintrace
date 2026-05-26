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

"""L4 transform / integration correctness: the online algorithms are invariant
under transform.jit, the internal lax.scan (multi-step) path, a batch axis, and
float64, and remain usable descent directions end-to-end (spec section 7)."""

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
from braintrace._etrace_algorithms.oracle_models import (
    batched_tanh_rnn,
    tanh_rnn,
)

ATOL_BPTT = 1e-4
ATOL_JIT = 1e-5
ATOL_SCAN = 1e-5
ATOL_BATCH = 1e-4


def _inputs(T, n_in, seed=42):
    return jnp.asarray(np.random.RandomState(seed).randn(T, n_in).astype('float32'))


def _batched_inputs(T, B, n_in, seed=1):
    return jnp.asarray(np.random.RandomState(seed).randn(T, B, n_in).astype('float32'))


# --- Task 1: batched_tanh_rnn model ------------------------------------------

def test_batched_tanh_rnn_hidden_is_batch_shaped():
    spec = batched_tanh_rnn(n_in=3, n_rec=4, batch=5, seed=0)
    assert spec.etp_param_keys == (('w',),)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=5)
    hidden = model.states(brainstate.HiddenState)
    assert hidden[('h',)].value.shape == (5, 4)


# --- Task 2: jit invariance of the multi-step gradient -----------------------

# Exact, multi-step-capable algorithms whose multi-step gradient equals BPTT.
_EXACT_MULTISTEP = {
    'D_RTRL': lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
    'pp_prop_full': lambda m: braintrace.pp_prop(m, decay_or_rank=16, vjp_method='multi-step'),
    'EProp_k0': lambda m: braintrace.EProp(
        m, feedback='symmetric', kappa_filter_decay=0.0, vjp_method='multi-step'),
    'OSTLRecurrent': lambda m: braintrace.OSTLRecurrent(m, vjp_method='multi-step'),
}


def _multistep_grad(spec, inputs, algo_factory, jit):
    """Total-sequence gradient via the multi-step (internal lax.scan) path,
    optionally wrapping the grad fn in transform.jit. Builds a fresh, freshly
    initialized algorithm so state is not shared between calls."""
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()
    params = model.states(brainstate.ParamState)

    def gfn(seq):
        return brainstate.transform.grad(
            lambda s: (algo(braintrace.MultiStepData(s)) ** 2).sum(), params)(seq)

    gfn = brainstate.transform.jit(gfn) if jit else gfn
    return gfn(inputs)


@pytest.mark.parametrize('algo_name', list(_EXACT_MULTISTEP))
def test_jit_invariance_matches_eager_and_bptt(algo_name):
    """transform.jit changes only compilation, not the result: the jitted
    multi-step gradient equals the eager one and equals BPTT."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_eager = _multistep_grad(spec, inputs, _EXACT_MULTISTEP[algo_name], jit=False)
    g_jit = _multistep_grad(spec, inputs, _EXACT_MULTISTEP[algo_name], jit=True)
    assert_param_gradients_close(g_jit, g_eager, atol=ATOL_JIT, keys=[('w',)])
    assert_param_gradients_close(g_eager, g_bptt, atol=ATOL_BPTT, keys=[('w',)])


# --- Task 3: scan-forward invariance -----------------------------------------

def test_scan_forward_outputs_match_single_step_loop():
    """The multi-step path runs the cell under an internal lax.scan. Its stacked
    forward outputs equal an explicit per-step single-step loop (the forward
    trajectory is transform-invariant; only gradient bookkeeping differs)."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)

    m_ms = spec.factory()
    brainstate.nn.init_all_states(m_ms, batch_size=1)
    a_ms = braintrace.D_RTRL(m_ms, vjp_method='multi-step')
    a_ms.compile_graph(inputs[0])
    a_ms.init_etrace_state()
    out_ms = a_ms(braintrace.MultiStepData(inputs))

    m_ss = spec.factory()
    brainstate.nn.init_all_states(m_ss, batch_size=1)
    a_ss = braintrace.D_RTRL(m_ss, vjp_method='single-step')
    a_ss.compile_graph(inputs[0])
    a_ss.init_etrace_state()
    out_loop = jnp.stack([a_ss(inputs[t]) for t in range(inputs.shape[0])])

    assert out_ms.shape == out_loop.shape == (6, 1, 4)
    assert float(jnp.max(jnp.abs(out_ms - out_loop))) < ATOL_SCAN


# --- Task 4: batch-axis invariance -------------------------------------------

def _batched_multistep_grad(batch, seq):
    spec = batched_tanh_rnn(n_in=3, n_rec=4, batch=batch, seed=0)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=batch)
    algo = braintrace.D_RTRL(model, vjp_method='multi-step')
    algo.compile_graph(seq[0])
    algo.init_etrace_state()
    params = model.states(brainstate.ParamState)
    return brainstate.transform.grad(
        lambda s: (algo(braintrace.MultiStepData(s)) ** 2).sum(), params)(seq)


def test_batch_axis_invariance_sums_per_sequence_gradients():
    """A batched run shares one parameter across B independent sequences; the
    batched gradient of the summed loss equals the sum of the per-sequence
    single-batch gradients. This pins correct batch-axis handling."""
    B, T, n_in = 3, 6, 3
    seq = _batched_inputs(T, B, n_in, seed=1)
    g_batched = _batched_multistep_grad(B, seq)

    g_sum = None
    for b in range(B):
        g_b = _batched_multistep_grad(1, seq[:, b:b + 1, :])
        g_sum = g_b if g_sum is None else {k: g_sum[k] + g_b[k] for k in g_b}

    assert_param_gradients_close(g_batched, g_sum, atol=ATOL_BATCH, keys=[('w',), ('win',)])


# --- Task 5: float64 exactness ------------------------------------------------

@pytest.fixture
def x64_enabled():
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', False)


def _f64_spec():
    from braintrace._etrace_algorithms.oracle_models import ModelSpec

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                rng = np.random.RandomState(0)
                self.w = brainstate.ParamState(jnp.asarray(0.5 * rng.randn(4, 4), dtype='float64'))
                self.win = brainstate.ParamState(jnp.asarray(0.5 * rng.randn(3, 4), dtype='float64'))
                self.h = brainstate.HiddenState(jnp.zeros((1, 4), dtype='float64'))

            def update(self, x):
                self.h.value = jax.nn.tanh(
                    x @ self.win.value + braintrace.matmul(self.h.value, self.w.value))
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=(('win',),))


def test_float64_exact_match_to_bptt(x64_enabled):
    """Under float64, the exact D_RTRL gradient still equals BPTT element-wise
    (numerical stability of the online path is dtype-independent for exact
    algorithms)."""
    spec = _f64_spec()
    inputs = jnp.asarray(np.random.RandomState(42).randn(6, 3), dtype='float64')
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = braintrace.D_RTRL(model, vjp_method='multi-step')
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()
    params = model.states(brainstate.ParamState)
    g = brainstate.transform.grad(
        lambda s: (algo(braintrace.MultiStepData(s)) ** 2).sum(), params)(inputs)
    assert np.asarray(g[('w',)]).dtype == np.float64
    assert_param_gradients_close(g, g_bptt, atol=1e-10, keys=[('w',)])


# --- Task 6: end-to-end convergence ------------------------------------------

def _conv_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3)))
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.v.value = jax.nn.tanh(0.9 * self.v.value + braintrace.matmul(x, self.w.value))
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _train_losses(algo, n_steps=15, lr=0.05):
    x = jnp.ones((1, 3), dtype='float32')
    algo.compile_graph(x)
    algo.init_etrace_state()
    losses = []
    for _ in range(n_steps):
        def loss_fn(x_):
            out = algo.update(x_)
            return ((out - jnp.ones_like(out)) ** 2).mean()
        grads, loss_val = brainstate.transform.grad(
            loss_fn, algo.param_states, return_value=True)(x)
        for path, st in algo.param_states.items():
            st.value = st.value - lr * grads[path]
        losses.append(float(loss_val))
    return losses


@pytest.mark.parametrize('algo_factory', [
    lambda m: braintrace.D_RTRL(m, vjp_method='single-step'),
    lambda m: braintrace.pp_prop(m, decay_or_rank=3, vjp_method='single-step'),
    lambda m: braintrace.EProp(m, feedback='symmetric', kappa_filter_decay=0.5),
    lambda m: braintrace.OSTLRecurrent(m, vjp_method='single-step'),
], ids=['D_RTRL', 'pp_prop', 'EProp', 'OSTLRecurrent'])
def test_algorithms_descend_loss_end_to_end(algo_factory):
    """Integration backstop: each algorithm drives an MSE-to-ones loss down over
    15 SGD steps."""
    losses = _train_losses(algo_factory(_conv_net()))
    assert losses[-1] < losses[0]


# --- Task 7: OSTTP under transform.jit (F-12 supported case) -----------------

def _osttp_net(n_in=3, n_rec=3):
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(0.1 * jax.random.normal(jax.random.PRNGKey(0), (n_in, n_rec)))
            self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

        def update(self, x):
            self.h.value = jax.nn.tanh(0.9 * self.h.value + braintrace.matmul(x, self.w.value))
            return self.h.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _osttp_first_step_grad(jit):
    """One OSTTP update under grad with a fixed y_target, on a fresh algorithm.
    Returns the param-gradient tree. Mirrors the known-good osttp_test setup
    (n_target == n_rec == 3, small B, y = ones)."""
    model = _osttp_net(n_in=3, n_rec=3)
    B = 0.1 * jax.random.normal(jax.random.PRNGKey(9), (3, 3))  # (n_target=3, n_rec=3)
    algo = braintrace.OSTTP(model, B_list=[B], target_timing='per-step')
    x = jnp.ones((1, 3), dtype='float32')
    algo.compile_graph(x)
    algo.init_etrace_state()
    y = jnp.ones((1, 3), dtype='float32')

    def gfn(x_, y_):
        return brainstate.transform.grad(
            lambda xx: ((algo.update(xx, y_target=y_) - 1.0) ** 2).mean(),
            algo.param_states)(x_)

    gfn = brainstate.transform.jit(gfn) if jit else gfn
    return gfn(x, y)


def test_osttp_jit_threads_instance_target_correctly_F12():
    """F-12: OSTTP stashes y_target in an instance attribute, raising the concern
    that transform.jit could read a stale value. The jitted first-step param
    gradient equals the eager one element-wise — the instance-attribute threading
    is correct under jit for the supported (fixed-shape) path. (The variable-shape
    y_target hazard is the still-OPEN F-12/F-14 gap, out of P6 scope.)"""
    g_eager = _osttp_first_step_grad(jit=False)
    g_jit = _osttp_first_step_grad(jit=True)
    assert_param_gradients_close(g_jit, g_eager, atol=ATOL_JIT)


def test_osttp_trains_under_jit_with_fixed_target():
    """F-12 supported case end-to-end: with the known-good setup (n_target ==
    n_rec, small B, y = ones) OSTTP's target-projected signal drives the MSE-to-
    ones loss down, and wrapping the whole train step in transform.jit preserves
    that — the instance-attribute y_target survives jit across many steps."""
    model = _osttp_net(n_in=3, n_rec=3)
    B = 0.1 * jax.random.normal(jax.random.PRNGKey(9), (3, 3))
    algo = braintrace.OSTTP(model, B_list=[B], target_timing='per-step')
    x = jnp.ones((1, 3), dtype='float32')
    algo.compile_graph(x)
    algo.init_etrace_state()

    @brainstate.transform.jit
    def train_step(x_, y_):
        def loss_fn(xx):
            out = algo.update(xx, y_target=y_)
            return ((out - jnp.ones_like(out)) ** 2).mean()
        grads, loss_val = brainstate.transform.grad(
            loss_fn, algo.param_states, return_value=True)(x_)
        for path, st in algo.param_states.items():
            st.value = st.value - 0.05 * grads[path]
        return loss_val

    y = jnp.ones((1, 3), dtype='float32')
    losses = [float(train_step(x, y)) for _ in range(10)]
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]
