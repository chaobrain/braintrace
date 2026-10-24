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

"""Outputs that bypass every new hidden value train correctly (issue #185).

Under ``vjp_method='single-step'`` a step output computed from a value that is
not a new hidden state -- a spike read from the pre-reset membrane potential,
a plain readout weight applied inside the step -- used to contribute nothing
to the online gradient, silently. Two terms were missing: the parameters'
same-step bypass gradient, and the bypass learning signal
``dL^t/dh^{t-1}`` paired with the previous trace ``e^{t-1}``.

The models here are single-layer cells with an element-wise transition, so
D-RTRL and EProp are *exact* on them: per-step ``etrace_grad`` must equal
BPTT element-wise. A mixed output (bypass plus a read of the new hidden
value) pins that neither path is counted twice. The approximate engines
(IO-dim, random projection) are checked through their contraction hooks.
"""

import warnings

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import braintrace

N_IN, N, T = 3, 4, 10
_sg = braintools.surrogate.Sigmoid()


def _shape(batch_size):
    return (N,) if batch_size is None else (batch_size, N)


class _LIF(brainstate.nn.Module):
    def __init__(self, out='pre_reset'):
        super().__init__()
        self.out = out
        self.w = brainstate.ParamState(brainstate.random.randn(N_IN, N))

    def init_state(self, batch_size=None, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(_shape(batch_size)))

    def update(self, x):
        V_pre = 0.9 * self.V.value + braintrace.matmul(x, self.w.value)
        spk = _sg(V_pre - 1.0)
        self.V.value = V_pre - jax.lax.stop_gradient(spk)
        if self.out == 'pre_reset':
            return spk
        if self.out == 'mixed':
            return _sg(self.V.value - 0.5) + 0.3 * jnp.tanh(V_pre)
        return _sg(self.V.value - 0.5)


class _CurrentLIF(brainstate.nn.Module):
    """Two hidden states in one group; only ``V`` has a hidden bypass."""

    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(N_IN, N))

    def init_state(self, batch_size=None, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(_shape(batch_size)))
        self.I = brainstate.HiddenState(jnp.zeros(_shape(batch_size)))

    def update(self, x):
        self.I.value = 0.8 * self.I.value + braintrace.matmul(x, self.w.value)
        V_pre = 0.9 * self.V.value + self.I.value
        spk = _sg(V_pre - 1.0)
        self.V.value = V_pre - jax.lax.stop_gradient(spk)
        return spk


class _CopyRead(brainstate.nn.Module):
    """A pure copy ``h2^t = h1^{t-1}`` read in the same step (#184).

    In the jaxpr ``h2^t`` and ``h1^{t-1}`` are one variable; the read is
    credited as the hidden bypass of ``h1^{t-1}``, which is the same value
    with the same trace (``e2^t = e1^{t-1}``).
    """

    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(N_IN, N))

    def init_state(self, batch_size=None, **kwargs):
        self.h1 = brainstate.HiddenState(jnp.zeros(_shape(batch_size)))
        self.h2 = brainstate.HiddenState(jnp.zeros(_shape(batch_size)))
        self.V = brainstate.HiddenState(jnp.zeros(_shape(batch_size)))

    def update(self, x):
        old1 = self.h1.value
        self.V.value = 0.9 * self.V.value + self.h2.value
        self.h1.value = 0.5 * old1 + braintrace.matmul(x, self.w.value)
        self.h2.value = old1
        return jnp.tanh(self.h2.value) + jnp.tanh(self.V.value)


class _Readout(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(N_IN, N))
        self.w_out = brainstate.ParamState(brainstate.random.randn(N, 2))

    def init_state(self, batch_size=None, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(_shape(batch_size)))

    def update(self, x):
        self.V.value = 0.9 * self.V.value + braintrace.matmul(x, self.w.value)
        return jnp.tanh(self.V.value) @ self.w_out.value


class _ReadoutVariants(brainstate.nn.Module):
    """Bypass shapes: an ETP ``Linear`` readout (a weight + bias pytree), a
    parameter applied directly to the output, a chain of two ETP ops with no
    hidden state between them, and a parameter written into a non-hidden
    state that is the output."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.w = brainstate.ParamState(brainstate.random.randn(N_IN, N))
        self.out = braintrace.nn.Linear(N, 2)
        self.b = brainstate.ParamState(jnp.full((N,), 0.1))

    def init_state(self, batch_size=None, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(_shape(batch_size)))
        self.s = brainstate.ShortTermState(jnp.zeros(_shape(batch_size)))

    def update(self, x):
        self.V.value = 0.9 * self.V.value + braintrace.matmul(x, self.w.value)
        if self.kind == 'linear':
            return self.out(jnp.tanh(self.V.value))
        if self.kind == 'param':
            return jnp.tanh(self.V.value) * self.b.value + self.b.value
        if self.kind == 'chain':
            return self.out(jnp.tanh(braintrace.matmul(x, self.w.value))) + self.out(self.V.value)
        # 'state': no recursion through ``s`` -- a non-hidden state carries no
        # trace, so only its same-step credit is online
        self.s.value = jnp.tanh(self.V.value) * self.b.value
        return self.s.value


_MODELS = {
    'pre_reset': lambda: _LIF('pre_reset'),
    'mixed': lambda: _LIF('mixed'),
    'hidden': lambda: _LIF('hidden'),
    'current': _CurrentLIF,
    'copy_read': _CopyRead,
    'readout': _Readout,
    'linear': lambda: _ReadoutVariants('linear'),
    'param': lambda: _ReadoutVariants('param'),
    'chain': lambda: _ReadoutVariants('chain'),
    'state': lambda: _ReadoutVariants('state'),
}

_N_OUT = {'readout': 2, 'linear': 2, 'chain': 2}


def _data(n_out):
    brainstate.random.seed(42)
    xs = brainstate.random.rand(T, N_IN) * 2.0
    return xs, jnp.full((T, n_out), 0.3)


def _loss(y, t):
    return jnp.mean((y - t) ** 2)


_ALGO_KWARGS = {
    'pp_prop': dict(decay_or_rank=0.9),
    'uoro': dict(vjp_method='single-step'),
}


def _online(model, algo, **kwargs):
    brainstate.random.seed(0)
    net = _MODELS[model]()
    brainstate.nn.init_all_states(net)
    xs, ys = _data(_N_OUT.get(model, N))
    learner = braintrace.compile(net, algo, xs[0], **{**_ALGO_KWARGS.get(algo, {}), **kwargs})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return learner.etrace_grad(xs, ys, step_fn=lambda x, y: _loss(learner(x), y)), learner


def _bptt(model):
    brainstate.random.seed(0)
    net = _MODELS[model]()
    brainstate.nn.init_all_states(net)
    xs, ys = _data(_N_OUT.get(model, N))
    params = net.states(brainstate.ParamState)

    def total_loss():
        return brainstate.transform.for_loop(lambda x, y: _loss(net(x), y), xs, ys).mean()

    # some variants leave a parameter unused
    return brainstate.transform.grad(total_loss, params, check_states=False)()


@pytest.mark.parametrize('model', list(_MODELS))
@pytest.mark.parametrize('algo', ['d_rtrl', 'eprop'])
def test_exact_algorithms_match_bptt(model, algo):
    online, _ = _online(model, algo)
    ref = _bptt(model)
    for path in ref:
        if all(float(jnp.abs(b).sum()) == 0.0 for b in jax.tree.leaves(ref[path])):
            continue  # an unused parameter of a variant
        for a, b in zip(jax.tree.leaves(online[path]), jax.tree.leaves(ref[path])):
            npt.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_issue_values():
    # the numbers in the issue body: 1.079097 (pre-reset), 1.861094 (hidden)
    for model, expected in [('pre_reset', 1.079097), ('hidden', 1.861094)]:
        online, _ = _online(model, 'd_rtrl')
        npt.assert_allclose(float(jnp.abs(online[('w',)]).sum()), expected, rtol=1e-5)


@pytest.mark.parametrize('model', ['pre_reset', 'readout'])
@pytest.mark.parametrize('chunk_size', [1, 4])
def test_multi_step_needs_no_bypass(model, chunk_size):
    # Multi-step differentiates the parameters and the hidden values directly,
    # so it never had the gap; the finite-window oracle pins that.
    from braintrace._testing.oracle import (
        assert_param_gradients_close, bptt_param_gradients, chunked_online_param_gradients,
    )
    def factory():
        brainstate.random.seed(0)  # the same weights on every build
        return _MODELS[model]()

    xs = jax.random.uniform(jax.random.PRNGKey(3), (8, 1, N_IN)) * 2.0
    online = chunked_online_param_gradients(
        factory, xs, chunk_size=chunk_size,
        algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    # gradients are O(10): float32 summation order differs across JAX versions
    assert_param_gradients_close(online, bptt_param_gradients(factory, xs), atol=1e-5, rtol=1e-5)


def test_modulatory_signal_compiles_without_bypass():
    brainstate.random.seed(0)
    net = _LIF('pre_reset')
    brainstate.nn.init_all_states(net)
    xs, _ = _data(N)
    algo = braintrace.ThreeFactor(net)
    algo.compile_graph(xs[0])
    hp = algo.graph.hidden_perturb
    assert list(hp.bypass_param_paths) == [] and list(hp.bypass_hidden_paths) == []


@pytest.mark.parametrize('algo', ['pp_prop', 'uoro', 'd_rtrl', 'eprop'])
def test_bypass_reaches_every_engine(algo):
    online, learner = _online('pre_reset', algo)
    hp = learner.graph.hidden_perturb
    assert list(hp.bypass_param_paths) == [('w',)]
    assert list(hp.bypass_hidden_paths) == [('V',)]
    g = online[('w',)]
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).sum()) > 1e-6


def _engine(algo, **kwargs):
    brainstate.random.seed(0)
    net = _LIF('pre_reset')
    brainstate.nn.init_all_states(net)
    xs, _ = _data(N)
    learner = braintrace.compile(net, algo, xs[0], **{**_ALGO_KWARGS.get(algo, {}), **kwargs})
    learner.init_etrace_state()
    # roll a few steps so the traces are non-trivial
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        brainstate.transform.for_loop(learner, xs[:4])
    dl = [jnp.linspace(0.1, 1.0, N).reshape(N, 1)]
    weights = {path: st.value for path, st in learner.param_states.items()}
    return learner, dl, weights


@pytest.mark.parametrize('algo', ['pp_prop', 'uoro'])
def test_previous_trace_hook_is_the_weight_contraction(algo):
    learner, dl, weights = _engine(algo)
    trace = learner._get_etrace_data()
    running_index = 4
    prev = learner._solve_previous_trace_gradients(running_index, trace, dl, weights)
    # the contraction of the previous trace, one update younger
    main = learner._solve_weight_gradients(running_index - 1, trace, dl, weights, {}, None)
    for path in weights:
        npt.assert_allclose(prev[path], main[path], rtol=1e-6)


def test_previous_trace_hook_reads_the_filtered_trace_without_advancing_it():
    learner, dl, weights = _engine('eprop', kappa_filter_decay=0.9)
    assert learner._trace_filters
    before = {k: f.value for k, f in learner._trace_filters.items()}
    trace = learner._get_etrace_data()
    prev = learner._solve_previous_trace_gradients(0, trace, dl, weights)
    for k, f in learner._trace_filters.items():
        npt.assert_array_equal(f.value, before[k])
    # identical to contracting the filter's current value directly
    filtered = {k: before.get(k, v) for k, v in trace.items()}
    learner._trace_filters, saved = {}, learner._trace_filters
    try:
        ref = learner._solve_previous_trace_gradients(0, filtered, dl, weights)
    finally:
        learner._trace_filters = saved
    for path in weights:
        npt.assert_allclose(prev[path], ref[path], rtol=1e-6)


def test_vmapped_lanes_match_per_sample_bptt():
    # The OnlineVmapTrainer flow: per-lane states, the graph compiled per lane.
    B = 3
    xs = jax.random.uniform(jax.random.PRNGKey(5), (T, B, N_IN)) * 2.0

    brainstate.random.seed(0)
    net = _LIF('pre_reset')
    model = braintrace.D_RTRL(net)

    @brainstate.transform.vmap_new_states(state_tag='new', axis_size=B)
    def init():
        brainstate.nn.init_all_states(net)
        model.compile_graph(xs[0, 0])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        init()
        vmodel = brainstate.nn.Vmap(model, vmap_states='new')
        weights = net.states().subset(brainstate.ParamState)

        def step(prev, x):
            g = brainstate.transform.grad(lambda x_: ((vmodel(x_) - 0.3) ** 2).sum(), weights)(x)
            return jax.tree.map(lambda a, b: a + b, prev, g), None

        online, _ = brainstate.transform.scan(
            step, jax.tree.map(jnp.zeros_like, weights.to_dict_values()), xs)

    ref = None
    for b in range(B):
        brainstate.random.seed(0)
        one = _LIF('pre_reset')
        brainstate.nn.init_all_states(one)
        g = brainstate.transform.grad(
            lambda: brainstate.transform.for_loop(
                lambda x: ((one(x) - 0.3) ** 2).sum(), xs[:, b]).sum(),
            one.states(brainstate.ParamState))()
        ref = g if ref is None else jax.tree.map(lambda a, c: a + c, ref, g)
    for path in ref:
        npt.assert_allclose(online[path], ref[path], rtol=1e-4, atol=1e-5)
