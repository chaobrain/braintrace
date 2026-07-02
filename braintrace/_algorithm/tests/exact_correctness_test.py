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
element-wise; cross-algorithm reduction identities hold; single-step-only
OTTT/OTPE match D_RTRL instantaneously. Findings F-19/F-20 pin the cases the
current oracle cannot validate."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_param_gradients_close,
    bptt_param_gradients,
    online_param_gradients,
)
from braintrace._algorithm.oracle_models import (
    cond_gate_rnn,
    leaky_linear,
    stacked_tanh_rnn,
    tanh_rnn,
    tied_weight_rnn,
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
_EXACT_CASES = (
    [('tanh_rnn', a) for a in _EXACT_MULTISTEP_ALGOS]
    + [('stacked_tanh_rnn', a) for a in _EXACT_MULTISTEP_ALGOS]
    + [('tied_weight', a) for a in _EXACT_MULTISTEP_ALGOS]
    + [('leaky_linear', 'D_RTRL')]
    + [('cond_gate', 'D_RTRL')]
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


# --- Task 5: one-step instantaneous equivalence (single-step-only algos) ------

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


def test_otpe_full_matches_d_rtrl_one_step_on_tanh_rnn():
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    _assert_onestep_equiv(spec, lambda m: braintrace.OTPE(m, mode='full', leak=0.9),
                          jnp.ones((1, 3), dtype='float32'))


def test_ottt_matches_d_rtrl_one_step_on_leaky_linear():
    """OTTT(A) on its exact regime (leaky_linear): instantaneous gradient == D_RTRL."""
    spec = leaky_linear(n_in=3, n_rec=4, leak=0.9, seed=0)
    _assert_onestep_equiv(spec, lambda m: braintrace.OTTT(m, mode='A', leak=0.9),
                          jnp.ones((1, 3), dtype='float32'))


def test_otpe_full_matches_d_rtrl_one_step_on_leaky_linear():
    spec = leaky_linear(n_in=3, n_rec=4, leak=0.9, seed=0)
    _assert_onestep_equiv(spec, lambda m: braintrace.OTPE(m, mode='full', leak=0.9),
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
    and in oracle_test.py). The naive per-step single-step accumulation diverges
    (F-SINGLESTEP, oracle_test.py::test_singlestep_naive_matches_bptt_KNOWN_DIVERGENCE).
    Here we re-confirm multi-step D_RTRL == BPTT to anchor the vjp_method dimension."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_multi = online_param_gradients(
        spec.factory, inputs,
        algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    assert_param_gradients_close(g_multi, g_bptt, atol=ATOL_BPTT)


# --- Task 7: findings F-19 (OTTT/OTPE single-step-only) & F-20 (OSTTP) --------

def test_ottt_otpe_reject_multistep_oracle_F19():
    """F-19: OTTT/OTPE are single-step only, so the multi-step BPTT oracle path is
    structurally unavailable. We pin that the multi-step call raises 'single-step
    only', which (with F-SINGLESTEP blocking the naive single-step accumulation)
    is why their multi-step temporal correctness is deferred, not asserted."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    for algo_factory in (
        lambda m: braintrace.OTTT(m, mode='A', leak=0.9, vjp_method='multi-step'),
        lambda m: braintrace.OTPE(m, mode='full', leak=0.9, vjp_method='multi-step'),
    ):
        with pytest.raises(NotImplementedError, match='single-step only'):
            online_param_gradients(spec.factory, inputs, algo_factory=algo_factory)


@pytest.mark.skip(
    reason="F-19: OTTT/OTPE are single-step only and the naive single-step "
           "accumulation diverges (F-SINGLESTEP); there is no clean multi-step "
           "BPTT oracle for them yet. Needs an online-scan oracle for single-step "
           "algorithms. Deferred (P6 / dedicated work).")
def test_ottt_multistep_temporal_matches_bptt_DEFERRED():
    pass


def test_osttp_runs_through_oracle_but_signal_is_target_based_F20():
    """F-20: OSTTP runs end-to-end (sequence-end) and yields finite gradients, but
    its learning signal is B @ y_target, NOT the autodiff dL/dh. The SSE-autodiff
    BPTT/EProp oracle therefore does not constrain it — we only smoke-test that it
    runs and is finite here; the OSTTP <-> EProp(random feedback) equivalence is
    deferred (see DEFERRED test below)."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g = online_param_gradients(
        spec.factory, inputs,
        algo_factory=lambda m: braintrace.OSTTP(
            m, [jnp.eye(4, dtype='float32')],
            target_timing='sequence-end', vjp_method='multi-step'))
    assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(v)))) for v in g.values())


@pytest.mark.skip(
    reason="F-20: OSTTP uses B@y_target as the learning signal, bypassing the "
           "autodiff loss; proving OSTTP == EProp(random feedback) needs a "
           "matched-B + matched-per-step-target harness. Deferred (P5).")
def test_osttp_equals_eprop_random_feedback_DEFERRED():
    pass
