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

"""L3-C approximate-class correctness: OTTT/OTPE gradients are direction-aligned
with BPTT (cosine/sign), exact in their degenerate regime, and their known
approximation biases (F-07/F-08/F-09) are quantified. Multi-state guards
(F-01/F-04) and the rate-model approximation-exactness ceiling (F-21/F-22) are
pinned."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_direction_aligned,
    assert_param_gradients_close,
    bptt_param_gradients,
    cosine_similarity,
    online_param_gradients,
    online_param_gradients_singlestep_naive,
    relative_magnitude,
    sign_agreement,
)
from braintrace._algorithm.oracle_models import (
    leaky_linear,
    tanh_rnn,
    two_state_rnn,
)

ATOL_BPTT = 1e-4
MIN_COSINE = 0.95
MIN_SIGN = 0.99


def _inputs(T, n_in, seed=42):
    return jnp.asarray(np.random.RandomState(seed).randn(T, n_in).astype('float32'))


# --- Task 1: direction metrics ------------------------------------------------

def test_cosine_similarity_aligned_and_orthogonal():
    a = jnp.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, 2.0 * a) == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-6)
    assert cosine_similarity(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)


def test_sign_agreement_counts_matching_signs():
    a = jnp.array([1.0, -1.0, 2.0, -3.0])
    b = jnp.array([5.0, -2.0, -1.0, -1.0])  # 3 of 4 signs match
    assert sign_agreement(a, b) == pytest.approx(0.75, abs=1e-6)


def test_relative_magnitude_ratio():
    a = jnp.array([3.0, 4.0])      # norm 5
    b = jnp.array([6.0, 8.0])      # norm 10
    assert relative_magnitude(a, b) == pytest.approx(0.5, abs=1e-6)


# --- Task 2: assert_direction_aligned ----------------------------------------

def test_assert_direction_aligned_passes_for_scaled_tree():
    ref = {('w',): jnp.array([1.0, 2.0, -3.0])}
    approx = {('w',): jnp.array([2.0, 4.0, -6.0])}  # same direction, 2x magnitude
    assert_direction_aligned(approx, ref, min_cosine=0.99, min_sign_agreement=0.99)


def test_assert_direction_aligned_flags_misaligned_key():
    ref = {('w',): jnp.array([1.0, 2.0, 3.0])}
    approx = {('w',): jnp.array([-1.0, -2.0, -3.0])}  # opposite direction
    with pytest.raises(AssertionError, match=r"\('w',\)"):
        assert_direction_aligned(approx, ref, min_cosine=0.95)


def test_assert_direction_aligned_checks_magnitude_bounds():
    ref = {('w',): jnp.array([1.0, 2.0, 3.0])}
    approx = {('w',): jnp.array([10.0, 20.0, 30.0])}  # aligned but 10x magnitude
    with pytest.raises(AssertionError, match='relmag'):
        assert_direction_aligned(approx, ref, min_cosine=0.95, mag_bounds=(0.5, 2.0))


# --- Task 3: two_state_rnn model ---------------------------------------------

def test_two_state_rnn_forms_one_group_num_state_two():
    spec = two_state_rnn(n_in=3, n_rec=3, seed=0)
    assert spec.etp_param_keys == (('w',),)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    assert set(model.states(brainstate.ParamState).keys()) == {('w',)}
    # Compile under D_RTRL to inspect the discovered structure: the coupled v,a
    # states collapse to a single HiddenGroup with num_state == 2.
    algo = braintrace.D_RTRL(model)
    algo.compile_graph(jnp.ones((1, 3), dtype='float32'))
    assert len(algo.graph.hidden_groups) == 1
    assert int(algo.graph.hidden_groups[0].varshape[-1]) == 3  # n_rec


# --- Task 4: C-level direction agreement (tanh_rnn, OTTT approximate regime) --

# Single-step-only approximate algorithms; their total-sequence gradient comes
# from the naive per-step accumulation (direction-aligned with BPTT; measured
# cosine: OTTT(A) 0.986, OTPE(full) 0.983; sign agreement 1.000).
_C_LEVEL_ALGOS = {
    'OTTT_A': lambda m: braintrace.OTTT(m, mode='A', leak=0.9),
    'OTPE_full': lambda m: braintrace.OTPE(m, mode='full', leak=0.9),
}


@pytest.mark.parametrize('algo_name', list(_C_LEVEL_ALGOS))
def test_ottt_otpe_direction_aligned_with_bptt_on_tanh_rnn(algo_name):
    """Approximate algorithms must point the same way as BPTT (cosine/sign),
    even though they do not match it element-wise."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_approx = online_param_gradients_singlestep_naive(
        spec.factory, inputs, algo_factory=_C_LEVEL_ALGOS[algo_name])
    assert_direction_aligned(
        g_approx, g_bptt, min_cosine=MIN_COSINE, min_sign_agreement=MIN_SIGN,
        keys=list(spec.etp_param_keys))


# --- Task 5: F-09 OTTT bias scales with ||hid2hid - leak*I|| ------------------

def test_ottt_is_exact_on_leaky_linear():
    """leaky_linear has hid2hid Jacobian == leak*I exactly, so OTTT (which assumes
    exactly that) reproduces BPTT: cosine ~ 1 and relative magnitude ~ 1."""
    spec = leaky_linear(n_in=3, n_rec=4, leak=0.9, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_ottt = online_param_gradients_singlestep_naive(
        spec.factory, inputs, algo_factory=lambda m: braintrace.OTTT(m, mode='A', leak=0.9))
    assert_direction_aligned(
        g_ottt, g_bptt, min_cosine=0.999, mag_bounds=(0.98, 1.02),
        keys=list(spec.etp_param_keys))


def test_ottt_is_biased_but_aligned_on_tanh_rnn_F09():
    """F-09: on tanh_rnn the hidden Jacobian is NOT leak*I (it carries the tanh
    derivative and the recurrent weight), so OTTT is no longer exact — its
    magnitude is inflated (measured relmag ~1.2, outside the exact band) while
    direction stays aligned (cosine ~0.986)."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_ottt = online_param_gradients_singlestep_naive(
        spec.factory, inputs, algo_factory=lambda m: braintrace.OTTT(m, mode='A', leak=0.9))
    key = spec.etp_param_keys[0]
    # Direction is still good ...
    assert cosine_similarity(g_ottt[key], g_bptt[key]) >= MIN_COSINE
    # ... but it is demonstrably NOT exact (magnitude inflated beyond the band that
    # holds on leaky_linear). This is the F-09 bias.
    assert relative_magnitude(g_ottt[key], g_bptt[key]) > 1.05


# --- Task 6: F-07 OTPE(mode='approx') magnitude inflation --------------------

def test_otpe_approx_is_directionally_ok_but_magnitude_inflated_F07():
    """F-07: OTPE's rank-1 'approx' mode keeps the gradient direction (cosine
    ~0.985) but grossly inflates its magnitude (measured relmag ~5.6 on tanh_rnn,
    ~3.8 on leaky_linear). We pin: aligned direction AND relmag > 2."""
    for spec in (tanh_rnn(n_in=3, n_rec=4, seed=0),
                 leaky_linear(n_in=3, n_rec=4, leak=0.9, seed=0)):
        inputs = _inputs(6, 3)
        g_bptt = bptt_param_gradients(spec.factory, inputs)
        g_approx = online_param_gradients_singlestep_naive(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.OTPE(m, mode='approx', leak=0.9))
        key = spec.etp_param_keys[0]
        assert cosine_similarity(g_approx[key], g_bptt[key]) >= MIN_COSINE
        assert relative_magnitude(g_approx[key], g_bptt[key]) > 2.0


# --- Task 7: F-08 trace_clip_abs biases magnitude ----------------------------

def test_trace_clip_abs_shrinks_gradient_magnitude_F08():
    """F-08: trace_clip_abs clamps the eligibility trace with no principled bound.
    A small clip (0.01) shrinks the OTPE gradient ~10x (measured relmag 0.096) and
    degrades its direction (cosine 0.983 -> 0.936), while clip=None keeps relmag
    near 1. We pin the magnitude collapse under aggressive clipping."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    key = spec.etp_param_keys[0]

    g_noclip = online_param_gradients_singlestep_naive(
        spec.factory, inputs, algo_factory=lambda m: braintrace.OTPE(m, mode='full', leak=0.9))
    g_clip = online_param_gradients_singlestep_naive(
        spec.factory, inputs,
        algo_factory=lambda m: braintrace.OTPE(m, mode='full', leak=0.9, trace_clip_abs=0.01))

    assert relative_magnitude(g_noclip[key], g_bptt[key]) > 0.8     # near-unbiased
    assert relative_magnitude(g_clip[key], g_bptt[key]) < 0.3       # collapsed


# --- Task 8: F-01/F-04 multi-state (num_state == 2) ---------------------------

def test_d_rtrl_exact_on_two_state_group():
    """D_RTRL handles a num_state==2 HiddenGroup exactly (it threads the
    per-state axis correctly), matching BPTT element-wise."""
    spec = two_state_rnn(n_in=3, n_rec=3, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_online = online_param_gradients(
        spec.factory, inputs, algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    assert_param_gradients_close(g_online, g_bptt, atol=ATOL_BPTT)


@pytest.mark.parametrize('algo_factory', [
    lambda m: braintrace.OTTT(m, mode='A', leak=0.9),
    lambda m: braintrace.OTPE(m, mode='full', leak=0.9),
], ids=['OTTT', 'OTPE'])
def test_ottt_otpe_reject_multistate_group_F01(algo_factory):
    """F-01/F-04: OTTT and OTPE assume a single-state group; on a num_state==2
    group they raise at compile (their per-state collapse has no theoretical
    basis here). Pinning this prevents silently-wrong multi-state gradients."""
    spec = two_state_rnn(n_in=3, n_rec=3, seed=0)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    with pytest.raises(ValueError, match='num_state == 1'):
        algo.compile_graph(jnp.ones((1, 3), dtype='float32'))


# --- Task 9: F-21/F-22 IODim/EProp approximations are exact on rate models -----

# On a single-relation rate model the IODim rank / ES decay / random-feedback
# "approximations" are degenerate-to-exact (spike: even rank=1 on n_rec=8/T=12 is
# cos 1.0 / relmag 1.0). We assert exactness here and defer the genuine
# approximation stress to an SNN multi-population model zoo (F-22).
_EXACT_ON_RATE = {
    'pp_prop_rank1': lambda m: braintrace.pp_prop(m, decay_or_rank=1, vjp_method='multi-step'),
    'pp_prop_decay05': lambda m: braintrace.pp_prop(m, decay_or_rank=0.5, vjp_method='multi-step'),
    'EProp_k05': lambda m: braintrace.EProp(
        m, feedback='symmetric', kappa_filter_decay=0.5, vjp_method='multi-step'),
    'EProp_random': lambda m: braintrace.EProp(
        m, feedback='random', kappa_filter_decay=0.0,
        random_feedback_key=jax.random.PRNGKey(7), vjp_method='multi-step'),
}


@pytest.mark.parametrize('algo_name', list(_EXACT_ON_RATE))
def test_rank_decay_random_approximations_are_exact_on_rate_model_F21(algo_name):
    """F-21: these nominally-approximate configs match BPTT element-wise on a
    single-HiddenGroup rate model. The model cannot stress their approximation."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_online = online_param_gradients(
        spec.factory, inputs, algo_factory=_EXACT_ON_RATE[algo_name])
    assert_param_gradients_close(g_online, g_bptt, atol=ATOL_BPTT)


@pytest.mark.skip(
    reason="F-22: IODim rank / ES decay / EProp random-feedback approximations are "
           "degenerate-to-exact on single-relation rate models (verified up to "
           "n_rec=8, T=12, rank=1). Exposing their real bias requires an SNN "
           "multi-population model zoo (LIF/ALIF, multi-layer random feedback). "
           "Deferred to P5b.")
def test_approximations_diverge_on_snn_multipopulation_DEFERRED():
    pass


# --- Task 10: C-level convergence backstop (loss decreases) ------------------

def _train_loss_trajectory(algo, n_steps=12, lr=0.05):
    """Manual SGD on an MSE-to-ones target (the ottt_test pattern), returning the
    per-step loss. A working approximate gradient must drive the loss down."""
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
    lambda m: braintrace.OTTT(m, mode='A', leak=0.9),
    lambda m: braintrace.OTPE(m, mode='full', leak=0.9),
], ids=['OTTT_A', 'OTPE_full'])
def test_approximate_algorithm_descends_loss(algo_factory):
    """C-level backstop: the approximate gradient is a usable descent direction —
    training loss at the end is below the start."""
    def _net():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3)))
                self.v = brainstate.HiddenState(jnp.zeros((1, 3)))

            def update(self, x):
                self.v.value = jax.nn.tanh(0.9 * self.v.value + braintrace.matmul(x, self.w.value))
                return self.v.value
        net = Net()
        brainstate.nn.init_all_states(net, batch_size=1)
        return net

    losses = _train_loss_trajectory(algo_factory(_net()))
    assert losses[-1] < losses[0]
