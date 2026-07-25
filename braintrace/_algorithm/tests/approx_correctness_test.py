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

"""L3-C approximate-class correctness: direction-alignment metrics against BPTT
(cosine / sign / relative magnitude), the multi-state (num_state == 2) exactness
guard for D_RTRL, and the rate-model approximation-exactness ceiling
(F-21/F-22/F-23/F-29).

The OTTT/OTPE-specific bias tests (F-01/F-04/F-07/F-08/F-09) were removed in
0.3.0 along with those algorithms; see docs/specs for the roadmap. The metric
helpers they exercised are retained and are the basis of the benchmark suite."""

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_direction_aligned,
    assert_gradients_differ,
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
    cosine_similarity,
    online_param_gradients,
    relative_deviation,
    relative_magnitude,
    sign_agreement,
)
from braintrace._algorithm.oracle_models import (
    SNN_SPECS,
    tanh_rnn,
    two_state_rnn,
)

ATOL_BPTT = 1e-4


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


# --- Task 9: F-21/F-22 IODim/EProp approximations are exact on rate models -----

# Through the full-window multi-step path the IODim rank / ES decay /
# random-feedback "approximations" are exact -- not because the model is a
# single-relation rate model, but because that path has no truncation for them to
# approximate (F-23). We assert the exactness here as a property of the path, and
# assert the genuine approximation through a finite window below. One entry,
# 'pp_prop_rank1', stays exact through a finite window too, for a reason specific
# to the rule rather than the path (F-29), and is held out of that test.
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
    """F-21: these nominally-approximate configs match BPTT element-wise here.

    The cause is the *oracle path*, not the model. All four configurations run
    through ``online_param_gradients`` with ``vjp_method='multi-step'`` over the
    whole sequence, where the within-call gradient is exact reverse-mode and the
    eligibility trace never enters -- so every algorithm returns BPTT at every
    hyperparameter setting (F-23). An earlier reading of this test attributed the
    exactness to the model being a single-HiddenGroup rate model and concluded
    that an SNN multi-population zoo was needed to expose the bias (F-22); that
    conclusion was wrong, and F-22 is retired by
    ``test_approximations_are_measurable_through_a_finite_window`` below.

    The test is kept because the equality is still a real property of this path
    and worth pinning.
    """
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_online = online_param_gradients(
        spec.factory, inputs, algo_factory=_EXACT_ON_RATE[algo_name])
    assert_param_gradients_close(g_online, g_bptt, atol=ATOL_BPTT)


def _chunked_exact(spec, inputs, chunk_size=2):
    return chunked_online_param_gradients(
        spec.factory, inputs, chunk_size=chunk_size,
        algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))


# ``pp_prop_rank1`` is excluded deliberately -- see F-29 and
# test_rank1_pp_prop_is_degenerate_on_a_recurrent_only_relation below.
_MEASURABLE_THROUGH_A_WINDOW = [
    name for name in _EXACT_ON_RATE if name != 'pp_prop_rank1'
]


@pytest.mark.parametrize('algo_name', _MEASURABLE_THROUGH_A_WINDOW)
def test_approximations_are_measurable_through_a_finite_window(algo_name):
    """F-22, retired. The finding deferred this until an "SNN multi-population
    model zoo" existed, on the theory that the model was what made the
    approximations look exact. That premise was wrong: a multi-population SNN
    model (ALIF + E/I conductance split, 3 ETP relations, num_state 5) is
    *also* bitwise-exact through the full-window path, because the cause is the
    oracle path and not the model (F-23). No model can revive a knob the
    harness cannot see.

    Through a finite window the same nominally-approximate configurations that
    F-21 finds exact become measurably different from the exact algorithm, on
    the very same rate model. That is the assertion F-22 wanted.

    ``min_rel`` is 1e-6 rather than something near zero on purpose: float32
    round-off on these trees sits around 1e-8, so a smaller floor would let the
    test pass on numerical noise. The deviations asserted here are 2e-3 to 2e-2.

    The SNN counterpart now exists too, and asserts the same thing on realistic
    models: ``tests/snn_model_correctness_test.py::
    test_approximation_is_measurable_on_snn_models``.
    """
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(8, 3)
    g_approx = chunked_online_param_gradients(
        spec.factory, inputs, chunk_size=2, algo_factory=_EXACT_ON_RATE[algo_name])
    assert_gradients_differ(_chunked_exact(spec, inputs), g_approx, min_rel=1e-6)


def test_rank1_pp_prop_is_degenerate_on_a_recurrent_only_relation():
    """F-29: ``decay_or_rank=1`` is *not* an approximation when the relation's
    presynaptic input is the hidden state itself.

    An int ``decay_or_rank`` maps to ``decay = (rank - 1) / (rank + 1)``, so
    rank 1 means decay 0 -- the EMA over the presynaptic factor
    ``eps^t = a * eps^{t-1} + (1 - a) * x_t`` collapses to ``x_t`` and no
    presynaptic smearing is introduced. On ``tanh_rnn``, whose only ETP relation
    is the recurrent weight (``x`` *is* ``h``), that reproduces exact D_RTRL to
    round-off: measured 7.6e-10 here, and 1e-10 to 3e-8 over chunk sizes
    1/2/4, T in {8, 12, 16}, n_rec in {4, 8} and recurrent spectral radius
    0.25 to 5.0. Weak recurrence is not the cause -- the deviation does not grow
    with the spectral radius.

    It becomes a real approximation as soon as the presynaptic input carries an
    external component: 0.55 on an input-weight ETP variant, and 0.31 to 0.79 on
    every SNN spec, whose projections consume ``concat(input, spikes)``. The
    second half of this test pins that side of the boundary, so the first half
    cannot be read as "rank 1 never does anything".

    The mechanism is not established -- only the boundary is. Consequence for
    the harness: rank 1 on a recurrent-only relation is unusable as a positive
    control for approximation error.
    """
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(8, 3)
    g_rank1 = chunked_online_param_gradients(
        spec.factory, inputs, chunk_size=2,
        algo_factory=_EXACT_ON_RATE['pp_prop_rank1'])
    rel = relative_deviation(g_rank1, _chunked_exact(spec, inputs))
    assert rel < 1e-6, (
        f'rank-1 pp_prop deviated by {rel:.3e} on a recurrent-only relation; '
        'F-29 recorded it as exact to round-off there. If this now differs, the '
        'IO-dim trace semantics changed and F-29 needs revisiting.'
    )

    snn = SNN_SPECS['lif_expcu']()
    xs = snn.make_inputs(8, 4)
    with brainstate.environ.context(dt=0.1 * u.ms):
        g_snn_rank1 = chunked_online_param_gradients(
            snn.factory, xs, chunk_size=2,
            algo_factory=_EXACT_ON_RATE['pp_prop_rank1'])
        assert_gradients_differ(
            _chunked_exact(snn, xs), g_snn_rank1, min_rel=1e-6)


# --- Task 10: C-level convergence backstop (loss decreases) ------------------

def _train_loss_trajectory(algo, n_steps=12, lr=0.05):
    """Manual SGD on an MSE-to-ones target, returning the per-step loss. A working
    approximate gradient must drive the loss down."""
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
    lambda m: braintrace.pp_prop(m, decay_or_rank=1),
    lambda m: braintrace.EProp(
        m, feedback='random', random_feedback_key=jax.random.PRNGKey(7)),
], ids=['pp_prop_rank1', 'EProp_random'])
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
