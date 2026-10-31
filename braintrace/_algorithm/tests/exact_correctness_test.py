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

"""L3-A/B exact-class correctness: multi-step online gradients reproduce BPTT
element-wise, cross-algorithm reduction identities hold, and the vjp_method
boundary is pinned.

The single-step-only OTTT/OTPE/OSTTP equivalence and deferral tests
(F-19/F-20) were removed in 0.2.5 along with those algorithms; see docs/specs
for the roadmap."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._testing.oracle import (
    assert_param_gradients_close,
    bptt_param_gradients,
    online_param_gradients,
)
from braintrace._testing.oracle_models import (
    cond_gate_rnn,
    leaky_linear,
    scan_body_rnn,
    stacked_tanh_rnn,
    tanh_rnn,
    tied_weight_rnn,
    while_settle_twin_rnn,
)

ATOL_BPTT = 1e-4
ATOL_EQUIV = 1e-5


def _inputs(T, n_in, seed=42):
    return jnp.asarray(np.random.RandomState(seed).randn(T, n_in).astype('float32'))


# --- Task 1: leaky_linear model ----------------------------------------------

def test_leaky_linear_builds_and_hid2hid_is_leak_identity():
    spec = leaky_linear(n_in=3, n_rec=4, leak=0.9, seed=0)
    assert spec.etp_param_keys == (('w',),)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    assert set(model.states(brainstate.ParamState).keys()) == {('w',)}
    # With zero input the recurrence is purely h <- leak * h, so two steps from a
    # known state scale by leak each time: hid2hid Jacobian == leak * I exactly.
    h0 = jnp.ones((1, 4), dtype='float32')
    model.h.value = h0
    y = model(jnp.zeros((3,), dtype='float32'))
    np.testing.assert_allclose(np.asarray(y), np.asarray(0.9 * h0), atol=1e-6)


# --- Task 2: stacked_tanh_rnn model ------------------------------------------

def test_stacked_tanh_rnn_builds_with_two_etp_weights():
    spec = stacked_tanh_rnn(n_in=3, n_rec=4, seed=0)
    assert spec.etp_param_keys == (('w1',), ('w2',))
    assert spec.plain_param_keys == (('win',), ('wmid',))
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    keys = set(model.states(brainstate.ParamState).keys())
    assert keys == {('w1',), ('w2',), ('win',), ('wmid',)}
    y = model(jnp.ones((3,), dtype='float32'))
    assert y.shape == (1, 4)
    assert bool(jnp.all(jnp.isfinite(y)))


# --- Task 3: L3-A multi-step exact vs BPTT -----------------------------------

# Multi-step algorithm factories whose total sequence gradient is EXACT (== BPTT)
# on these toy models (spike-verified maxdiff 0.0). pp_prop rank 16 is a full int
# rank for a 4x4 weight; EProp(k=0, symmetric) reduces to D_RTRL.
_EXACT_MULTISTEP_ALGOS = {
    'D_RTRL': lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
    'pp_prop_full': lambda m: braintrace.pp_prop(m, decay_or_rank=16, vjp_method='multi-step'),
    'EProp_k0': lambda m: braintrace.EProp(
        m, feedback='symmetric', kappa_filter_decay=0.0, vjp_method='multi-step'),
    'OSTLRecurrent': lambda m: braintrace.OSTLRecurrent(m, vjp_method='multi-step'),
}

# (model_name, algo_name) pairs verified exact by the P4 spikes.
# cond_gate exercises the Phase 1 cond -> select_n canonicalization: ETP
# matmuls inside `lax.cond` branches must stay BPTT-exact after conversion.
# tied_weight locks the multi-eqn-per-weight invariant (one ParamState, two
# relations): trace state keyed per relation instance + per-path gradient
# accumulation. Scan unrolling (Phase 2) multiplies relations per weight and
# depends on it.
# scan_body exercises the Phase 2 inner-scan unrolling: ETP matmuls inside a
# `for_loop` body must stay BPTT-exact after the scan is unrolled at
# extraction time.
# while_twin anchors the Phase 3 while-hidden equivalence to ground truth: it
# is the hand-composed (no-while) twin of while_settle_rnn, so proving
# twin == BPTT here plus while == twin (while_support_test.py, single-step)
# chains the while model to the oracle.
_EXACT_CASES = (
    [('tanh_rnn', a) for a in _EXACT_MULTISTEP_ALGOS]
    + [('stacked_tanh_rnn', a) for a in _EXACT_MULTISTEP_ALGOS]
    + [('tied_weight', a) for a in _EXACT_MULTISTEP_ALGOS]
    + [('scan_body', a) for a in _EXACT_MULTISTEP_ALGOS]
    + [('leaky_linear', 'D_RTRL')]
    + [('cond_gate', 'D_RTRL')]
    + [('while_twin', 'D_RTRL'), ('while_twin', 'pp_prop_full')]
)


def _model_spec(name):
    if name == 'tanh_rnn':
        return tanh_rnn(n_in=3, n_rec=4, seed=0)
    if name == 'stacked_tanh_rnn':
        return stacked_tanh_rnn(n_in=3, n_rec=4, seed=0)
    if name == 'leaky_linear':
        return leaky_linear(n_in=3, n_rec=4, leak=0.9, seed=0)
    if name == 'cond_gate':
        return cond_gate_rnn(n_in=3, n_rec=4, leak=0.9, seed=0)
    if name == 'tied_weight':
        return tied_weight_rnn(n_rec=3, seed=0)
    if name == 'scan_body':
        return scan_body_rnn(n_rec=3, loops=3, seed=0)
    if name == 'while_twin':
        return while_settle_twin_rnn(n_in=3, n_rec=4, seed=0)
    raise KeyError(name)


def test_tied_weight_traces_keyed_per_relation_instance():
    """One ParamState through two ETP call sites must yield two relations and
    two distinct D-RTRL trace states keyed by ``(id(y_var), group index)`` —
    not one shared per-weight entry."""
    spec = tied_weight_rnn(n_rec=3, seed=0)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = braintrace.D_RTRL(model, vjp_method='multi-step')
    algo.compile_graph(_inputs(1, 3)[0])
    algo.init_etrace_state()

    rels = algo.graph.hidden_param_op_relations
    assert len(rels) == 2
    assert all(r.trainable_paths['weight'] == ('w',) for r in rels)
    assert rels[0].y_var is not rels[1].y_var
    assert len(algo.etrace_bwg) == 2
    assert set(algo.etrace_bwg) == {
        (id(r.y_var), g.index) for r in rels for g in r.hidden_groups
    }


@pytest.mark.parametrize('model_name,algo_name', _EXACT_CASES,
                         ids=[f'{m}-{a}' for m, a in _EXACT_CASES])
def test_exact_multistep_matches_bptt(model_name, algo_name):
    """Each exact-class algorithm's multi-step total gradient equals BPTT for
    every parameter (ETP and plain)."""
    spec = _model_spec(model_name)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_online = online_param_gradients(
        spec.factory, inputs, algo_factory=_EXACT_MULTISTEP_ALGOS[algo_name]
    )
    assert_param_gradients_close(g_online, g_bptt, atol=ATOL_BPTT)


# --- Task 4: cross-algorithm equivalence matrix (multi-step) -----------------

def _multistep_grads(spec, inputs, algo_factory):
    return online_param_gradients(spec.factory, inputs, algo_factory=algo_factory)


def test_ostl_recurrent_equals_d_rtrl_multistep():
    spec, inputs = tanh_rnn(n_in=3, n_rec=4, seed=0), _inputs(6, 3)
    g_ostl = _multistep_grads(spec, inputs,
                              lambda m: braintrace.OSTLRecurrent(m, vjp_method='multi-step'))
    g_drtrl = _multistep_grads(spec, inputs,
                               lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    assert_param_gradients_close(g_ostl, g_drtrl, atol=ATOL_EQUIV)


def test_pp_prop_full_rank_equals_d_rtrl_multistep():
    spec, inputs = tanh_rnn(n_in=3, n_rec=4, seed=0), _inputs(6, 3)
    g_pp = _multistep_grads(spec, inputs,
                            lambda m: braintrace.pp_prop(m, decay_or_rank=16, vjp_method='multi-step'))
    g_drtrl = _multistep_grads(spec, inputs,
                               lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    assert_param_gradients_close(g_pp, g_drtrl, atol=ATOL_EQUIV)


def test_ostl_feedforward_equals_pp_prop_multistep():
    """OSTLFeedforward (subclass of pp_prop) reduces to pp_prop on the same decay."""
    spec, inputs = tanh_rnn(n_in=3, n_rec=4, seed=0), _inputs(6, 3)
    g_ff = _multistep_grads(spec, inputs,
                            lambda m: braintrace.OSTLFeedforward(m, decay_or_rank=0.9, vjp_method='multi-step'))
    g_pp = _multistep_grads(spec, inputs,
                            lambda m: braintrace.pp_prop(m, decay_or_rank=0.9, vjp_method='multi-step'))
    assert_param_gradients_close(g_ff, g_pp, atol=ATOL_EQUIV)


# --- Task 5: one-step instantaneous equivalence -------------------------------

def _onestep_grads(algo, x):
    """Weight gradient of (algo.update(x)**2).sum() at step 0 with zero trace.
    At a single step every exact algorithm computes the same instantaneous
    gradient, so this isolates correctness of the per-step weight-gradient rule
    independent of temporal credit assignment (the cross_check_test pattern)."""
    algo.compile_graph(x)
    algo.init_etrace_state()
    return brainstate.transform.grad(
        lambda x_: (algo.update(x_) ** 2).sum(), algo.param_states
    )(x)


def _build_inited(spec):
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    return model


def _assert_onestep_equiv(spec, algo_factory, x):
    g_algo = _onestep_grads(algo_factory(_build_inited(spec)), x)
    g_drtrl = _onestep_grads(braintrace.D_RTRL(_build_inited(spec)), x)
    # Compare every shared key (ETP and plain).
    assert_param_gradients_close(g_algo, g_drtrl, atol=ATOL_EQUIV)


def test_eprop_unfiltered_matches_d_rtrl_one_step_on_tanh_rnn():
    """EProp with no kappa filter and symmetric feedback is D_RTRL's trace, so
    the instantaneous gradient must coincide."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    _assert_onestep_equiv(
        spec,
        lambda m: braintrace.EProp(m, feedback='symmetric', kappa_filter_decay=0.0),
        jnp.ones((1, 3), dtype='float32'))


def test_eprop_unfiltered_matches_d_rtrl_one_step_on_leaky_linear():
    spec = leaky_linear(n_in=3, n_rec=4, leak=0.9, seed=0)
    _assert_onestep_equiv(
        spec,
        lambda m: braintrace.EProp(m, feedback='symmetric', kappa_filter_decay=0.0),
        jnp.ones((1, 3), dtype='float32'))


# --- Task 6: vjp_method consistency & boundary -------------------------------

def test_singlestep_method_rejects_multistep_data():
    """A vjp_method='single-step' algorithm cannot compute a multi-step VJP: when
    its gradient is taken over a MultiStepData sequence (the oracle path), it
    raises NotImplementedError ('only support the input data that is at a single
    time step'). This is the boundary that forces the multi-step oracle to use
    vjp_method='multi-step'. (The eager forward does not raise; the rejection
    happens when the VJP is actually evaluated under grad.)"""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    with pytest.raises(NotImplementedError, match='single time step'):
        online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='single-step'))


def test_multistep_method_is_the_exact_path():
    """Cross-reference: multi-step vjp_method is the exact path (proven in Task 3
    and in oracle_test.py). The naive per-step single-step accumulation does not
    match BPTT element-wise -- it is only direction-aligned (the former
    F-SINGLESTEP finding, now asserted positively by
    oracle_test.py::test_singlestep_naive_directionally_aligned_with_bptt).
    Here we re-confirm multi-step D_RTRL == BPTT to anchor the vjp_method dimension."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_multi = online_param_gradients(
        spec.factory, inputs,
        algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    assert_param_gradients_close(g_multi, g_bptt, atol=ATOL_BPTT)


# =============================================================================
# Single-step and in-step reads of new hidden values (#239)
# =============================================================================
#
# These models update one hidden state from another's *new* value in the same
# step: a synaptic current read by the membrane potential (``g -> V``), the
# STP utilisation read by the resources (``u -> x``), the LSTM cell read by
# the output (``c -> h``). The single-step learning signal holds such a read
# fixed when the eligibility traces carry it; before #239 it counted the path
# twice (2.000x BPTT on LSTMCell's ``Wi`` and ``Wg``).
#
# From a zero state, one step has no history for an algorithm to approximate,
# so every algorithm must equal BPTT at T = 1. Without recurrent weights the
# SNN layers are diagonal, and single-step D-RTRL must equal BPTT at any T.
# ``MinimalRNNCell`` is left out: its input projection feeds a second hidden
# state through a partial path (#237), which is not this issue.

import braintools
import brainunit as u

from braintrace._testing import models as _snn
from braintrace._testing.oracle import online_param_gradients_singlestep_naive

# ``ALIF_Delta_Dense_Layer`` is left out: its BPTT gradient is zero on these inputs.
_SNN_LAYERS = [
    'IF_Delta_Dense_Layer', 'LIF_ExpCo_Dense_Layer', 'ALIF_ExpCo_Dense_Layer', 'LIF_ExpCu_Dense_Layer',
    'LIF_STDExpCu_Dense_Layer', 'LIF_STPExpCu_Dense_Layer', 'ALIF_ExpCu_Dense_Layer',
    'ALIF_STDExpCu_Dense_Layer', 'ALIF_STPExpCu_Dense_Layer',
]
_RNN_CELLS = ['LSTMCell', 'GRUCell', 'MGUCell', 'URLSTMCell', 'LRUCell']
_SINGLE_STEP_ALGOS = {
    'D_RTRL': lambda m: braintrace.D_RTRL(m),
    'pp_prop': lambda m: braintrace.pp_prop(m, decay_or_rank=0.9),
    'EProp': lambda m: braintrace.EProp(m),
    'OSTLRecurrent': lambda m: braintrace.OSTLRecurrent(m),
    'OSTLFeedforward': lambda m: braintrace.OSTLFeedforward(m),
    'SnAp': lambda m: braintrace.SnAp(m, n=2),
}


def _layer_factory(name, diagonal=False):
    def make():
        with brainstate.random.seed_context(1):
            if name in _RNN_CELLS:
                return getattr(braintrace.nn, name)(3, 4)
            kwargs = dict(V_th=0.05 * u.mV)
            if diagonal:
                kwargs['rec_init'] = braintools.init.ZeroInit()
            return getattr(_snn, name)(3, 4, **kwargs)
    return make


def _layer_inputs(name, steps):
    with brainstate.random.seed_context(2):
        if name in _RNN_CELLS:
            return brainstate.random.randn(steps, 1, 3)
        return (brainstate.random.rand(steps, 1, 3) < 0.7).astype(float) * 4.0


def _worst_singlestep_error(name, steps, algo_factory, diagonal=False, precision=64):
    make = _layer_factory(name, diagonal)
    with brainstate.environ.context(precision=precision, dt=0.1 * u.ms):
        xs = _layer_inputs(name, steps)
        ref = bptt_param_gradients(make, xs)
        got = online_param_gradients_singlestep_naive(make, xs, algo_factory=algo_factory)
    errors = []
    for (path, r), (_, g) in zip(jax.tree_util.tree_flatten_with_path(ref, is_leaf=u.math.is_quantity)[0],
                                 jax.tree_util.tree_flatten_with_path(got, is_leaf=u.math.is_quantity)[0]):
        r = np.asarray(u.get_mantissa(r), dtype=np.float64)
        g = np.asarray(u.get_mantissa(g), dtype=np.float64)
        if np.linalg.norm(r) > 0:
            errors.append(float(np.linalg.norm(g - r) / np.linalg.norm(r)))
    assert errors, 'no non-zero BPTT gradient to compare'
    return max(errors)


@pytest.mark.parametrize('algo', list(_SINGLE_STEP_ALGOS))
@pytest.mark.parametrize('name', _SNN_LAYERS + _RNN_CELLS)
def test_single_step_equals_bptt_at_one_step(name, algo):
    assert _worst_singlestep_error(name, 1, _SINGLE_STEP_ALGOS[algo]) < 1e-12


@pytest.mark.parametrize('name', [n for n in _SNN_LAYERS if n != 'IF_Delta_Dense_Layer'])
def test_single_step_d_rtrl_equals_bptt_on_diagonal_layers(name):
    """IF_Delta is left out: it has no in-step read, and its 0.035 at T = 10
    is not this issue."""
    assert _worst_singlestep_error(name, 10, _SINGLE_STEP_ALGOS['D_RTRL'], diagonal=True) < 1e-12


@pytest.mark.parametrize('name', ['LSTMCell', 'ALIF_ExpCu_Dense_Layer', 'LIF_STPExpCu_Dense_Layer'])
def test_single_step_float32_matches_float32_bptt(name):
    assert _worst_singlestep_error(name, 1, _SINGLE_STEP_ALGOS['D_RTRL'], precision=32) < 1e-6
