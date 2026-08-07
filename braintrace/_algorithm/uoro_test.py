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

"""Acceptance suite for ``trace_factorization='random_projection'`` (UORO) — P4.

What is being claimed, precisely: UORO is an **unbiased estimator of the exact
within-group influence recursion that the compiled transition defines**. It does
not fix cross-group coupling, the F-31 instantaneous tail, or any primitive's own
solve regime. So the reference here is *not* BPTT in general — it is the exact
within-group recursion, which on a single-group, position-preserving-tail model
also equals BPTT.

The primary pin (U1) is not statistical. Conditionally on earlier draws the
update's ``nu_t`` dependence splits into an even part ``nu nu^T J_f`` and two odd
cross terms; a draw set closed under negation with ``mean(nu nu^T) == I`` kills
the odd part pairwise and maps the even part to ``J_f`` -- both *exactly*. The
``2^H`` Rademacher sign patterns are such a set, so an exhaustive enumeration
reproduces the exact recursion at machine precision, with no confidence interval
anywhere. The statistical machinery (U3) is a *secondary* pin on the sampled
estimator.

Every gradient assertion goes through the finite-window oracle
(``chunked_online_param_gradients`` with ``chunk_size < T``, F-23) and names the
parameter keys it compares.
"""

import functools
import itertools

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._testing import oracle_models as om
from braintrace._algorithm.axes import ETraceConfig
from braintrace._testing.oracle import (
    assert_unbiased_estimator,
    bptt_param_gradients,
    chunked_online_param_gradients,
    flat_gradient_leaves,
    relative_deviation,
    seed_gradient_samples,
)

# U1's fixture constants. Both are load-bearing and measured; see
# ``nonzero_init_rnn``'s docstring.
H = 2       # hidden units == the width of one Rademacher draw
T = 3       # >= 3, or the boundary trace holds only instantaneous terms
CHUNK = 1   # one-step windows: maximally sensitive to the trace
N_U3 = 64   # U3's sample size; the floor is measured in ``_u3_samples``


def _inputs(t=T, n=H, *, seed=0, scale=0.5):
    rng = np.random.RandomState(seed)
    return scale * jnp.asarray(rng.randn(t, 1, n).astype('float32'))


def _uoro(**kwargs):
    """Algorithm factory on the finite-window oracle path."""
    return lambda m: braintrace.UORO(m, vjp_method='multi-step', **kwargs)


def _etp_leaves(tree, keys=(('w',),)):
    """The compared subtree: ETP parameter keys only, as plain arrays."""
    return {k: np.asarray(u.get_mantissa(tree[k])) for k in keys}


# ---------------------------------------------------------------------------
# The draw table: a UORO whose projections come from a fixed stacked array.
#
# The seam has to be functional. The stepper body is traced *once* and run under
# scan, so an override that consumed a Python iterator (``next(it)``) would bake
# a single draw into every step. The table is an array indexed by the carried
# step counter, which is the only spelling that survives tracing.
# ---------------------------------------------------------------------------

class _TabulatedUORO(braintrace.UORO):
    """UORO whose ``nu`` draws come from ``table[step, group]``."""

    def __init__(self, model, table, **kwargs):
        super().__init__(model, **kwargs)
        # (n_steps, n_groups, *draw_shape)
        self._table = jnp.asarray(table)

    def _draw_projection(self, key, step, group_index, shape, dtype):
        row = self._table[step, group_index]
        return jnp.reshape(row, shape).astype(dtype)


def _sign_patterns(n):
    """All ``2^n`` Rademacher sign vectors of length ``n``."""
    return np.array(list(itertools.product([-1.0, 1.0], repeat=n)))


def _exhaustive_mean_gradient(spec, inputs, *, n_draw_steps, draw_shape,
                              n_groups=1, chunk_size=CHUNK, algo_kwargs=None):
    """Mean ETP gradient over *every* sign pattern of the draws that matter.

    ``n_draw_steps`` is the number of steps whose draw can still influence the
    gradient; each step contributes one independent Rademacher vector of length
    ``n_groups * prod(draw_shape)`` -- every group's draw at that step, since a
    group's own draw must be independent of its neighbour's. The enumeration is
    therefore over ``2^(n_draw_steps * n_groups * prod(draw_shape))`` patterns,
    which is why the multi-group case uses one unit per island.

    Returns the mean, the number of patterns enumerated, and the individual
    per-pattern gradients -- the last so that a caller wanting the spread rather
    than the mean can reuse this enumeration instead of repeating it. Each
    pattern bakes its draw table in as a traced constant and therefore compiles
    its own scan, so a repeated enumeration is paid for in compilations, not
    just in time.
    """
    algo_kwargs = algo_kwargs or {}
    width = n_groups * int(np.prod(draw_shape))
    patterns = _sign_patterns(width)
    total = None
    per_pattern = []
    for combo in itertools.product(patterns, repeat=n_draw_steps):
        # (n_steps, n_groups, *draw_shape)
        table = np.stack([c.reshape((n_groups,) + tuple(draw_shape))
                          for c in combo])
        g = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m, _t=table: _TabulatedUORO(
                m, _t, vjp_method='multi-step', **algo_kwargs),
            chunk_size=chunk_size,
        )
        flat = {k: np.asarray(u.get_mantissa(v)) for k, v in g.items()}
        per_pattern.append(flat)
        total = flat if total is None else {
            k: total[k] + flat[k] for k in flat}
    count = len(per_pattern)
    return {k: v / count for k, v in total.items()}, count, per_pattern


@functools.cache
def _u1_enumeration():
    """The one 16-run enumeration that U1, its BPTT companion and U1a share.

    Three tests assert three different things -- agreement with the exact
    recursion, agreement with BPTT, and that no single pattern *is* the mean --
    about the identical enumeration over the identical fixture. Computing it
    once and asserting against it three times keeps each assertion intact while
    paying for 16 compiled scans instead of 64; the arrays are read-only.
    """
    spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
    inputs = _inputs()
    mean, count, per_pattern = _exhaustive_mean_gradient(
        spec, inputs, n_draw_steps=2, draw_shape=(1, H, 1))
    return spec, inputs, mean, count, per_pattern


def _saturating_snap_reference(spec, inputs, *, chunk_size=CHUNK, n=4):
    """The exact within-group influence recursion, via a saturating SnAp order.

    This -- not BPTT -- is what UORO estimates. On the single-group,
    elementwise-tail fixtures used here the two coincide, and both are asserted.
    """
    return chunked_online_param_gradients(
        spec.factory, inputs,
        algo_factory=lambda m: braintrace.SnAp(m, n=n, vjp_method='multi-step'),
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# U1 + companions: exhaustive exactness
# ---------------------------------------------------------------------------

class TestExhaustiveExactness:
    """U1: the mean over all sign patterns *is* the exact recursion."""

    def test_the_enumeration_mean_matches_the_exact_within_group_recursion(self):
        spec, inputs, mean, count, _ = _u1_enumeration()
        assert count == 4 ** 2, f'expected 16 runs, enumerated {count}'

        ref = _saturating_snap_reference(spec, inputs)
        dev = relative_deviation(
            {('w',): mean[('w',)]}, {('w',): ref[('w',)]})
        assert dev < 1e-5, f'deviation from the exact recursion: {dev}'

    def test_the_enumeration_mean_also_matches_bptt_on_this_fixture(self):
        # Single group, position-preserving elementwise tail: the within-group
        # recursion is itself exact here, so the BPTT comparison is meaningful.
        spec, inputs, mean, _, _ = _u1_enumeration()
        ref = bptt_param_gradients(spec.factory, inputs)
        dev = relative_deviation(
            {('w',): mean[('w',)]}, {('w',): ref[('w',)]})
        assert dev < 1e-5, f'deviation from BPTT: {dev}'

    def test_rolling_the_block_diagonal_would_fail_the_same_pin(self):
        # The negative control that gives U1 its teeth: an implementation that
        # rolled the per-position block diagonal instead of the full Jacobian
        # converges -- just as cleanly -- onto the *biased* trace. Asserted by
        # comparing the exact recursion against the diagonal one on the same
        # fixture: if these were close, U1 could not tell the two apart.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        exact = _saturating_snap_reference(spec, inputs)
        diagonal = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            chunk_size=CHUNK)
        dev = relative_deviation(
            {('w',): exact[('w',)]}, {('w',): diagonal[('w',)]})
        assert dev > 1e-3, (
            f'the fixture cannot separate full from diagonal D (dev={dev}); '
            f'U1 would pass for the wrong implementation')


class TestSingleRunsDeviateFromTheMean:
    """U1a: the enumeration must be doing work."""

    def test_individual_sign_patterns_are_not_the_mean(self):
        # The same 16 patterns U1 averages: the per-pattern gradients come back
        # from the shared enumeration rather than being enumerated a second time.
        _, _, mean, _, per_pattern = _u1_enumeration()

        deviations = [
            relative_deviation({('w',): g[('w',)]}, {('w',): mean[('w',)]})
            for g in per_pattern
        ]
        assert min(deviations) > 1e-4, (
            f'some single run equals the mean exactly (min dev {min(deviations)}); '
            f'the estimator is not stochastic')


class TestHandComputedFactors:
    """U1b: the only test that pins the *normalisers*.

    Unbiasedness is invariant to the choice of ``rho0`` (any positive
    ``nu_t``-independent scalar) and ``rho1`` (any positive even function of
    ``nu_t``), so U1 cannot see a normaliser mistake. These do.
    """

    def _factors(self, algo):
        data = algo._get_etrace_data()
        return data['s_tilde'], data['theta_tilde']

    def test_the_first_step_factors_match_the_hand_computed_values(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        x = _inputs(1)[0]
        table = np.array([[[[1.0], [-1.0]]]])  # step 0, group 0, (1, H, 1)
        table = table.reshape(1, 1, 1, H, 1)

        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = _TabulatedUORO(model, table, vjp_method='multi-step')
        algo.compile_graph(braintrace.MultiStepData(x[None]))
        algo.init_etrace_state()

        w = np.asarray(model.states(brainstate.ParamState)[('w',)].value)
        h0 = np.full((1, H), 0.4)
        algo(braintrace.MultiStepData(x[None]))
        s, theta = self._factors(algo)

        # Hand-computed. At the first step s_tilde == 0 and theta_tilde == 0, so
        # D s_tilde == 0 and rho0 == sqrt(eps / eps) == 1 exactly; rho1 is the
        # only live normaliser.
        pre = np.asarray(x) + h0 @ w
        dphi = 1.0 - np.tanh(pre) ** 2           # (1, H)
        nu = np.array([[1.0], [-1.0]]).reshape(1, H, 1)
        # nu^T J_f for a dense recurrent matmul: outer(h_prev, nu * dphi).
        proj = h0.T @ (nu[..., 0] * dphi)        # (H, H)
        rho1 = np.sqrt((np.linalg.norm(proj) + 1e-12)
                       / (np.linalg.norm(nu) + 1e-12))

        want_s = rho1 * nu
        want_theta = proj / rho1
        np.testing.assert_allclose(np.asarray(s[0]), want_s, atol=1e-5)
        got_theta = np.asarray(u.get_mantissa(theta[(0, ('w',))]))
        np.testing.assert_allclose(got_theta, want_theta, rtol=1e-4, atol=1e-6)

    def test_the_second_step_factors_pin_rho0_which_the_first_step_cannot(self):
        """``rho0`` is exactly ``1`` at step 0, so one step pins only ``rho1``.

        At the first step ``s_tilde`` and ``theta_tilde`` are both zero, so
        ``rho0 == sqrt(eps / eps) == 1`` whatever the formula says: swapping its
        numerator and denominator, or dropping it entirely, leaves the one-step
        test green. The second step is the first one where it is live, and this is
        the only test that observes it -- U1's exhaustive enumeration cannot,
        because unbiasedness holds for *any* draw-independent positive ``rho0``.
        """
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        # `scale` is load-bearing: at the suite's default 0.5 this fixture puts
        # rho0 at 0.92, so inverting its ratio would move the expectation by only
        # 19% -- detectable, but with no margin. At 5.0 the transition saturates,
        # rho0 lands near 16, and an inverted ratio is off by 264x.
        inputs = _inputs(2, scale=5.0)
        nu0 = np.array([1.0, -1.0]).reshape(1, H, 1)
        nu1 = np.array([-1.0, -1.0]).reshape(1, H, 1)
        table = np.stack([nu0[None], nu1[None]])   # (2 steps, 1 group, 1, H, 1)

        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = _TabulatedUORO(model, table, vjp_method='multi-step')
        algo.compile_graph(braintrace.MultiStepData(inputs))
        algo.init_etrace_state()
        w = np.asarray(model.states(brainstate.ParamState)[('w',)].value)
        algo(braintrace.MultiStepData(inputs))
        s, theta = self._factors(algo)

        eps = 1e-12
        x0, x1 = np.asarray(inputs[0]), np.asarray(inputs[1])
        h0 = np.full((1, H), 0.4)

        # --- step 0: d_s == 0 and theta == 0, so rho0 == 1 exactly ------------
        pre0 = x0 + h0 @ w
        dphi0 = 1.0 - np.tanh(pre0) ** 2                     # (1, H)
        h1 = np.tanh(pre0)
        proj0 = h0.T @ (nu0[..., 0] * dphi0)                 # (H, H)
        rho1_0 = np.sqrt((np.linalg.norm(proj0) + eps)
                         / (np.linalg.norm(nu0) + eps))
        s1 = rho1_0 * nu0
        theta1 = proj0 / rho1_0

        # --- step 1: both normalisers live ------------------------------------
        pre1 = x1 + h1 @ w
        dphi1 = 1.0 - np.tanh(pre1) ** 2
        # d f(h)[0, o] / d h[0, q] == dphi[o] * w[q, o], contracted with s1.
        d_s = (dphi1[0] * (w.T @ s1[0, :, 0])).reshape(1, H, 1)
        proj1 = h1.T @ (nu1[..., 0] * dphi1)
        rho0_1 = np.sqrt((np.linalg.norm(theta1) + eps)
                         / (np.linalg.norm(d_s) + eps))
        rho1_1 = np.sqrt((np.linalg.norm(proj1) + eps)
                         / (np.linalg.norm(nu1) + eps))
        assert rho0_1 > 4.0 or rho0_1 < 0.25, (
            f'rho0 == {rho0_1} is too close to 1 for this fixture to pin it')

        want_s = rho0_1 * d_s + rho1_1 * nu1
        want_theta = theta1 / rho0_1 + proj1 / rho1_1
        np.testing.assert_allclose(np.asarray(s[0]), want_s, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(u.get_mantissa(theta[(0, ('w',))])),
            want_theta, rtol=1e-4, atol=1e-6)

    def test_the_outer_product_of_the_factors_is_the_influence_estimate(self):
        # The representation claim: eps_tilde[j, u] == s_tilde[u] * theta_tilde[j].
        # After one step the estimate must be exactly the instantaneous term
        # scaled by the sign pattern, which is checkable without any recursion.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        x = _inputs(1)[0]
        table = np.ones((1, 1, 1, H, 1))  # nu == +1 everywhere

        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = _TabulatedUORO(model, table, vjp_method='multi-step')
        algo.compile_graph(braintrace.MultiStepData(x[None]))
        algo.init_etrace_state()
        algo(braintrace.MultiStepData(x[None]))

        s, theta = self._factors(algo)
        outer = np.einsum(
            'pas,ij->ijpas',
            np.asarray(s[0]),
            np.asarray(u.get_mantissa(theta[(0, ('w',))])))
        assert outer.shape == (H, H, 1, H, 1)

        # rho0/rho1 cancel in the product, so after one step from zero the
        # estimate is exactly nu nu^T J_f. With nu == 1 that is, for every
        # hidden index u, the row sum of J_f -- and for a dense recurrent matmul
        # J_f[(i,o), (q,0)] == h0[i] * dphi[o] * delta(q, o), whose row sum is
        # h0[i] * dphi[o].
        w = np.asarray(model.states(brainstate.ParamState)[('w',)].value)
        h0 = np.full((1, H), 0.4)
        dphi = 1.0 - np.tanh(np.asarray(x) + h0 @ w) ** 2   # (1, H)
        want = np.einsum('bi,bo->io', h0, dphi)             # (H, H)
        for q in range(H):
            np.testing.assert_allclose(
                outer[:, :, 0, q, 0], want, rtol=1e-4, atol=1e-6)


class TestKeyingAndSharing:
    """U1c: several relations and several groups."""

    def test_a_tied_weight_keeps_one_hidden_factor_and_stays_unbiased(self):
        spec = om.tied_weight_rnn(n_rec=H)
        inputs = _inputs(T, H, scale=0.4)
        mean, count, _ = _exhaustive_mean_gradient(
            spec, inputs, n_draw_steps=2, draw_shape=(1, H, 1))
        assert count == 16
        ref = _saturating_snap_reference(spec, inputs)
        dev = relative_deviation(
            {('w',): mean[('w',)]}, {('w',): ref[('w',)]})
        assert dev < 1e-4, f'tied-weight deviation: {dev}'

    def test_two_groups_each_get_their_own_hidden_factor(self):
        # ``two_island_rnn`` is the only fixture with two groups at coupled
        # scope: grouping follows the transition, and coupled scope merges every
        # mutually reachable hidden state, so ``stacked_tanh_rnn`` -- the obvious
        # candidate -- compiles to ONE group holding both ``h1`` and ``h2``.
        spec = om.two_island_rnn(n_in=H, n_rec=H)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step')
        algo.compile_graph(braintrace.MultiStepData(_inputs(1, H)))
        algo.init_etrace_state()

        data = algo._get_etrace_data()
        assert len(algo.graph.hidden_groups) == 2
        assert set(data['s_tilde']) == {0, 1}
        # one parameter-shaped factor per (group, ETP path)
        assert set(data['theta_tilde']) == {(0, ('wa',)), (1, ('wb',))}

    def test_two_groups_stay_unbiased_with_per_group_draws(self):
        # Structure is not correctness: a stepper that reused one group's draw
        # for the other would keep the keying above intact and still be biased,
        # because the cross-group parity argument needs the two draws to be
        # independent. One unit per island keeps the enumeration at 2^4.
        spec = om.two_island_rnn(n_in=1, n_rec=1)
        inputs = _inputs(T, 1, scale=0.6)
        mean, count, _ = _exhaustive_mean_gradient(
            spec, inputs, n_draw_steps=2, draw_shape=(1, 1), n_groups=2)
        assert count == 16, f'expected 2^(2 steps * 2 groups), got {count}'

        ref = _saturating_snap_reference(spec, inputs)
        for key in (('wa',), ('wb',)):
            dev = relative_deviation({key: mean[key]}, {key: ref[key]})
            assert dev < 1e-4, f'{key} deviation across two groups: {dev}'


class TestTheDrawIndexAdvances:
    """U1d: the trace-once trap."""

    def test_consecutive_steps_consume_different_draws(self):
        # The stepper body is traced once. A draw taken outside the carry would
        # be identical at every step and the estimator would silently become a
        # fixed-projection rule.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        x = _inputs(1)[0]
        # step 0 -> +1, step 1 -> -1. If the index did not advance, the two
        # windows would apply the same draw and s_tilde would keep its sign.
        table = np.stack([np.ones((1, 1, H, 1)), -np.ones((1, 1, H, 1))])

        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = _TabulatedUORO(model, table, vjp_method='multi-step')
        algo.compile_graph(braintrace.MultiStepData(x[None]))
        algo.init_etrace_state()

        algo(braintrace.MultiStepData(x[None]))
        first = np.asarray(algo._get_etrace_data()['s_tilde'][0]).copy()
        step_after_first = int(algo._get_etrace_data()['step'])
        algo(braintrace.MultiStepData(x[None]))
        second = np.asarray(algo._get_etrace_data()['s_tilde'][0])

        assert step_after_first == 1, f'the step counter is {step_after_first}'
        assert int(algo._get_etrace_data()['step']) == 2
        assert float(np.sign(first).sum()) > 0
        assert float(np.sign(second).sum()) < 0, (
            'the second window reused the first window\'s draw')

    def test_the_carried_key_advances_when_drawing_for_real(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        x = _inputs(1)[0]
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step',
                               projection_key=7)
        algo.compile_graph(braintrace.MultiStepData(x[None]))
        algo.init_etrace_state()

        before = jax.random.key_data(algo._get_etrace_data()['key'])
        algo(braintrace.MultiStepData(x[None]))
        after = jax.random.key_data(algo._get_etrace_data()['key'])
        assert not bool(np.all(np.asarray(before) == np.asarray(after)))


# ---------------------------------------------------------------------------
# U3: the sampled estimator, with a real confidence interval
# ---------------------------------------------------------------------------

@pytest.mark.slow
@functools.cache
def _u3_samples():
    """The 256-seed draw and its reference, shared by both U3 tests.

    The two tests differ only in which reference they hold the interval
    against -- the honest one, and one skewed by 1.15 -- while drawing the
    identical seeded gradients beforehand. That draw is where effectively all
    of U3's cost sits (each seed compiles its own run), so it is done once and
    asserted against twice. Neither caller mutates what it gets back.

    ``N_U3`` is set by the *rejection* half, not the acceptance half. Rejecting
    a 1.15-skewed reference needs ``0.15 |<ref, d>| > z sd / sqrt(N)``, so the
    interval has to be tight enough to exclude the bias; the non-vacuity half
    is satisfied here from ``N = 16`` upwards and never binds. Measured over
    the same 256-sample draw, re-checked at twelve direction seeds: at ``N =
    16`` and ``24`` the skewed reference is *accepted* and the test asserts
    nothing; at ``32`` one direction seed in twelve fails to reject; from
    ``48`` up all twelve reject. 64 keeps a margin above that floor -- the
    rejection gap grows as ``sqrt(N)`` -- while costing a quarter of the 256
    the suite used to draw.
    """
    spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
    inputs = _inputs()
    samples = seed_gradient_samples(
        spec.factory, inputs,
        algo_factory=lambda m, seed: braintrace.UORO(
            m, vjp_method='multi-step', projection_key=seed),
        seeds=range(N_U3), chunk_size=CHUNK)
    return samples, _saturating_snap_reference(spec, inputs)


class TestSampledUnbiasedness:
    """U3: fixed directions, a two-sided interval, and a non-vacuity bound."""

    def test_the_sampled_mean_lands_inside_its_confidence_interval(self):
        samples, ref = _u3_samples()
        assert_unbiased_estimator(samples, ref, keys=["w|"], seed=0)

    def test_a_biased_reference_is_rejected(self):
        # The interval must be able to fail. Scaling the reference by 1.15 is a
        # bias no honest estimator could match.
        samples, ref = _u3_samples()
        skewed = {k: v * 1.15 for k, v in ref.items()}
        with pytest.raises(AssertionError, match='confidence interval'):
            assert_unbiased_estimator(samples, skewed, keys=["w|"], seed=0)


# ---------------------------------------------------------------------------
# U4: negative controls, plural
# ---------------------------------------------------------------------------

class TestNegativeControls:
    """U4: not the biased trace, and not deterministic."""

    def test_a_single_run_differs_from_the_diagonal_rule(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        uoro = chunked_online_param_gradients(
            spec.factory, inputs, algo_factory=_uoro(projection_key=0),
            chunk_size=CHUNK)
        diagonal = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            chunk_size=CHUNK)
        dev = relative_deviation(
            {('w',): np.asarray(uoro[('w',)])},
            {('w',): np.asarray(diagonal[('w',)])})
        assert dev > 1e-3, f'UORO reproduced the diagonal rule (dev={dev})'

    def test_two_projection_keys_give_different_gradients(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        a = chunked_online_param_gradients(
            spec.factory, inputs, algo_factory=_uoro(projection_key=0),
            chunk_size=CHUNK)
        b = chunked_online_param_gradients(
            spec.factory, inputs, algo_factory=_uoro(projection_key=1),
            chunk_size=CHUNK)
        dev = relative_deviation(
            {('w',): np.asarray(a[('w',)])}, {('w',): np.asarray(b[('w',)])})
        assert dev > 1e-4, 'the estimator is deterministic across keys'


# ---------------------------------------------------------------------------
# U5: structure
# ---------------------------------------------------------------------------

class TestStructure:
    """U5: shapes, carrier size, refusals."""

    def _compiled(self, spec, n_in=H, **kwargs):
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step', **kwargs)
        x = _inputs(1, n_in)[0]
        algo.compile_graph(braintrace.MultiStepData(x[None]))
        algo.init_etrace_state()
        return model, algo, x

    def test_the_hidden_factor_has_the_group_shape_and_no_batch_surprise(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        _, algo, _ = self._compiled(spec)
        group, = algo.graph.hidden_groups
        s = algo._get_etrace_data()['s_tilde'][0]
        assert s.shape == (*group.varshape, group.num_state)

    def test_the_parameter_factor_carries_no_hidden_and_no_state_axis(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        model, algo, _ = self._compiled(spec)
        theta = algo._get_etrace_data()['theta_tilde'][(0, ('w',))]
        want = np.shape(model.states(brainstate.ParamState)[('w',)].value)
        assert np.shape(u.get_mantissa(theta)) == want

    def test_the_carrier_totals_the_documented_element_count(self):
        # tanh_rnn(3, 4): |theta| == 16, B*P*S == 1*4*1 == 4 -> 20 elements.
        # Asserted as *carrier storage*, which is not peak memory: the full
        # Jacobian is an O((P*S)^2) transient per step (F-32).
        spec = om.tanh_rnn(n_in=3, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step')
        x = jnp.ones((1, 3))
        algo.compile_graph(braintrace.MultiStepData(x[None]))
        algo.init_etrace_state()
        data = algo._get_etrace_data()
        elements = sum(int(np.size(u.get_mantissa(a)))
                       for a in jax.tree.leaves(
                           (data['s_tilde'], data['theta_tilde'])))
        assert elements == 20, f'carrier holds {elements} elements'

    def test_heterogeneous_units_survive_a_step_and_a_gradient(self):
        spec = om.unit_weight_rnn(n_in=3, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step')
        x = jnp.ones((1, 3))
        algo.compile_graph(braintrace.MultiStepData(x[None]))
        algo.init_etrace_state()
        g = brainstate.transform.grad(
            lambda a: u.get_mantissa((algo(braintrace.MultiStepData(a)) ** 2).sum()),
            model.states(brainstate.ParamState),
        )(jnp.stack([x, x]))
        assert float(jnp.abs(u.get_mantissa(g[('w',)])).max()) > 0.0
        assert bool(jnp.all(jnp.isfinite(u.get_mantissa(g[('w',)]))))

    @pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'float32'])
    def test_a_narrow_model_dtype_survives_the_scan_carry(self, dtype):
        """The factors are scan carries, so their dtype is a hard contract.

        Two independent ways to break it, both of which did: allocating the
        factors at a hard-coded ``float32`` (so a float16 or -- under
        ``jax_enable_x64`` -- a float64 model mismatches on entry), and letting
        the ``rho`` normalisers set the output dtype. The normalisers are
        deliberately computed at float32-or-wider, because a sum of squares in
        float16 underflows, so the *result* has to be narrowed back. Neither
        failure is a silent downcast: ``jax.lax.scan`` raises ``carry input and
        carry output must have equal types`` on the first windowed call.
        """
        dt = jnp.dtype(dtype)

        class Narrow(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.full((H, H), 0.1, dt))
                self.h = brainstate.HiddenState(jnp.full((1, H), 0.4, dt))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value))
                return self.h.value

        model = Narrow()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step')
        x = braintrace.MultiStepData(jnp.ones((2, 1, H), dt))
        algo.compile_graph(x)
        algo.init_etrace_state()
        algo(x)   # the call that used to raise
        data = algo._get_etrace_data()
        for leaf in jax.tree.leaves((data['s_tilde'], data['theta_tilde'])):
            assert jnp.dtype(u.get_mantissa(leaf).dtype) == dt, (
                f'factor came back as {u.get_mantissa(leaf).dtype}, not {dt}')

    @pytest.mark.parametrize('scope', ['diagonal', 'sparse_n'])
    def test_a_non_coupled_scope_is_refused_by_name(self, scope):
        kwargs = {'sparse_n': 3} if scope == 'sparse_n' else {}
        with pytest.raises(ValueError, match='random_projection'):
            ETraceConfig(trace_factorization='random_projection',
                         recurrence_scope=scope, **kwargs)

    def test_kappa_filtering_is_refused_by_name(self):
        with pytest.raises(ValueError, match='random_projection'):
            ETraceConfig(trace_factorization='random_projection',
                         recurrence_scope='coupled',
                         trace_filter='kappa', kappa=0.5)

    def test_the_io_factorized_engine_refuses_the_coordinate(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        cfg = ETraceConfig(trace_factorization='random_projection',
                           recurrence_scope='coupled')
        with pytest.raises(ValueError, match='random_projection'):
            # ``decay_or_rank`` is a required positional for this engine.
            braintrace.IODimVjpAlgorithm(model, 0.9, config=cfg)

    def test_a_descended_scan_is_refused_naming_the_coordinate(self):
        # The descent path forces include_recurrent_mixing=False, so a full
        # Jacobian computed in a descended body would silently omit exactly the
        # mixing UORO needs.
        spec = om.snn_scan_rnn(n_rec=4, loops=40)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step')
        with pytest.raises(NotImplementedError, match='coupled|descend'):
            algo.compile_graph(braintrace.MultiStepData(jnp.ones((1, 1, 4))))

    def test_the_config_alone_selects_this_engine(self):
        from braintrace._algorithm.random_projection_vjp import (
            RandomProjectionVjpAlgorithm)
        from braintrace._compile import _resolve_algorithm
        cfg = ETraceConfig(trace_factorization='random_projection',
                           recurrence_scope='coupled')
        assert _resolve_algorithm(cfg) is RandomProjectionVjpAlgorithm


# ---------------------------------------------------------------------------
# U7: reproducibility and the epsilon guard
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_reset_state_reproduces_a_run_bit_for_bit(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step',
                              projection_key=3)
        algo.compile_graph(braintrace.MultiStepData(inputs[:CHUNK]))
        algo.init_etrace_state()

        def run():
            outs = []
            for start in range(0, inputs.shape[0], CHUNK):
                outs.append(np.asarray(algo(
                    braintrace.MultiStepData(inputs[start:start + CHUNK]))))
            return np.concatenate(outs)

        first = run()
        algo.reset_state(batch_size=1)
        brainstate.nn.init_all_states(model, batch_size=1)
        second = run()
        np.testing.assert_array_equal(first, second)

    def test_reset_state_reproduces_the_gradients_not_just_the_outputs(self):
        """The forward pass is the wrong observable for this claim.

        A model's outputs do not depend on the projection factors, the step
        counter or the key at all -- they are a function of the hidden state and the
        parameters. So the output-equality test above stays green even if
        ``reset_state`` forgot to reset every one of UORO's own carriers, as long
        as the *model* states were re-initialised. The gradients are what carry the
        stream, so they are what must be compared.
        """
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step', projection_key=3)
        algo.compile_graph(braintrace.MultiStepData(inputs[:CHUNK]))
        algo.init_etrace_state()
        params = model.states(brainstate.ParamState)

        def run():
            total = None
            for start in range(0, inputs.shape[0], CHUNK):
                g = brainstate.transform.grad(
                    lambda s: (algo(braintrace.MultiStepData(s)) ** 2).sum(),
                    params)(inputs[start:start + CHUNK])
                total = g if total is None else jax.tree.map(
                    lambda a, b: a + b, total, g)
            return _etp_leaves(total)[('w',)]

        first = run()
        algo.reset_state(batch_size=1)
        brainstate.nn.init_all_states(model, batch_size=1)
        second = run()
        assert np.abs(first).max() > 1e-6, 'a zero gradient would pin nothing'
        np.testing.assert_array_equal(first, second)

    def test_the_first_step_factors_are_finite_with_the_default_epsilon(self):
        # At the first step both norms are zero, so both rho ratios are 0/0.
        # With projection_eps == 0 this is NaN at every T; the default guard is
        # the reason the exactness pins carry a tolerance above float64 epsilon.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        x = _inputs(1)[0]
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step')
        algo.compile_graph(braintrace.MultiStepData(x[None]))
        algo.init_etrace_state()
        algo(braintrace.MultiStepData(x[None]))
        data = algo._get_etrace_data()
        for leaf in jax.tree.leaves((data['s_tilde'], data['theta_tilde'])):
            assert bool(jnp.all(jnp.isfinite(u.get_mantissa(leaf))))

    def test_zeroing_the_epsilon_is_refused_rather_than_producing_nan(self):
        """It used to be allowed, and it made every gradient NaN.

        This test previously asserted the NaN, which recorded the behaviour
        faithfully and left a configuration in the API whose only possible outcome
        is a poisoned carrier. There is no window length and no model for which
        ``projection_eps=0`` is useful, so the constructor now refuses it.
        """
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        with pytest.raises(ValueError, match='projection_eps must be positive'):
            braintrace.UORO(model, vjp_method='multi-step', projection_eps=0.0)

    def test_an_epsilon_that_underflows_float32_is_refused_too(self):
        """The subtler half: positive in Python, zero where it is used.

        The norms are accumulated in float32, so ``1e-50`` is exactly the same
        guard as ``0.0`` by the time it reaches the ratio -- and would pass a naive
        ``> 0`` check.
        """
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        with pytest.raises(ValueError, match='remain positive in float32'):
            braintrace.UORO(model, vjp_method='multi-step', projection_eps=1e-50)


# ---------------------------------------------------------------------------
# The production draw, and the key schedule that feeds it.
#
# The tests above establish that the step counter advances and that the carried
# key changes. Neither observes the draw ``nu`` that production actually uses:
# ``_TabulatedUORO`` replaces it, and a key that advances in the carry can still be
# ignored by the draw.
#
# The freshness is not in ``_draw_projection`` -- that is a deterministic function
# of one key, which is exactly what makes it a usable seam. It is in the schedule
# at ``random_projection_vjp.py:553``: one ``split_key(len(groups))`` per step, one
# subkey per group, carried key advanced. So the schedule is what these tests
# replay, feeding the *production* draw function. An implementation that reused
# the initial key at every step -- one ``nu`` forever, which breaks the
# conditional parity argument -- fails the first test; one that handed both groups
# the same subkey fails the second, and no mean-based pin can see either, since
# each group's marginal expectation is untouched by correlating the two.
#
# Replaying beats capturing here: an ``io_callback`` placed inside the stepper is
# eliminated, because the stepper is traced under ``jax.custom_vjp`` where the
# callback's unused result is dead.
# ---------------------------------------------------------------------------

class TestTheProductionDrawAndItsKeySchedule:

    def _algo(self, spec=None, n_rec=6):
        spec = spec or om.nonzero_init_rnn(n_rec=n_rec, h0=0.4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.UORO(model, vjp_method='multi-step', projection_key=11)
        algo.compile_graph(braintrace.MultiStepData(_inputs(1, n=n_rec)))
        algo.init_etrace_state()
        return algo

    @staticmethod
    def _schedule(key, n_groups, n_steps):
        """Replay production's per-step split: subkeys per step, key advanced."""
        per_step = []
        for _ in range(n_steps):
            rng = brainstate.random.RandomState(key)
            per_step.append(jnp.asarray(rng.split_key(n_groups)))
            key = rng.value
        return per_step

    def test_the_draw_changes_from_step_to_step(self):
        """The freshness property, on the production draw itself.

        Six units make an accidental collision a 1-in-64 event per pair, and four
        consecutive steps of the real schedule are compared.
        """
        algo = self._algo()
        step0 = jnp.asarray(0, jnp.int32)
        draws = [
            np.asarray(algo._draw_projection(
                subkeys[0], step0, 0, (1, 6, 1), jnp.float32)).tobytes()
            for subkeys in self._schedule(algo._initial_projection_key(), 1, 4)
        ]
        assert len(set(draws)) > 1, (
            'four steps of the production schedule drew the same vector')

    def test_the_two_hidden_groups_draw_independently(self):
        algo = self._algo(om.two_island_rnn(n_in=6, n_rec=6))
        assert len(algo.graph.hidden_groups) == 2, 'the fixture must be two groups'
        subkeys = self._schedule(algo._initial_projection_key(), 2, 1)[0]
        step0 = jnp.asarray(0, jnp.int32)
        a, b = (
            np.asarray(algo._draw_projection(
                subkeys[gi], step0, gi, (1, 6, 1), jnp.float32)).tobytes()
            for gi in (0, 1)
        )
        assert a != b, 'both hidden groups drew the same projection vector'

    def test_the_draw_is_rademacher_which_is_what_the_parity_argument_needs(self):
        # +-1 valued (so E[nu nu^T] == I) and hence negation-symmetric.
        algo = self._algo()
        nu = np.asarray(algo._draw_projection(
            algo._initial_projection_key(), jnp.asarray(0, jnp.int32),
            0, (1, 6, 1), jnp.float32))
        assert set(np.unique(nu).tolist()) <= {-1.0, 1.0}, np.unique(nu)

    def test_one_key_gives_one_draw_which_is_why_the_seam_works(self):
        # The determinism the fwd/bwd replay depends on: the same key must
        # reproduce the same nu, or the backward pass would re-draw.
        algo = self._algo()
        key, step0 = algo._initial_projection_key(), jnp.asarray(0, jnp.int32)
        first = algo._draw_projection(key, step0, 0, (1, 6, 1), jnp.float32)
        second = algo._draw_projection(key, step0, 0, (1, 6, 1), jnp.float32)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
