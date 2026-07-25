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

"""The SNN specs must be deterministic (F-24) and live (F-25) before any
gradient assertion built on them means anything."""

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_model_is_live,
    bptt_param_gradients,
    chunked_online_param_gradients,
    flat_gradient_leaves,
    gradient_norm,
    online_param_gradients_singlestep_naive,
)
from braintrace._algorithm.oracle_models import SNN_SPECS


@pytest.mark.parametrize('name', sorted(SNN_SPECS))
def test_snn_spec_construction_is_deterministic(name):
    """F-24: the underlying layer classes seed from the global RNG, so two
    factory() calls must be pinned to produce identical weights."""
    spec = SNN_SPECS[name]()
    with brainstate.environ.context(dt=0.1 * u.ms):
        w1 = flat_gradient_leaves(
            {k: v.value for k, v in spec.factory().states(brainstate.ParamState).items()})
        w2 = flat_gradient_leaves(
            {k: v.value for k, v in spec.factory().states(brainstate.ParamState).items()})
    assert set(w1) == set(w2)
    for key in w1:
        assert bool(jnp.allclose(w1[key], w2[key])), f'{name}: {key} differs across calls'


@pytest.mark.parametrize('name', sorted(SNN_SPECS))
def test_snn_spec_is_live(name):
    """F-25: at the default input scale these networks never spike, so their
    gradients are identically zero and every comparison is vacuous. Each spec
    records a scale that produces a non-trivial gradient."""
    spec = SNN_SPECS[name]()
    with brainstate.environ.context(dt=0.1 * u.ms):
        xs = spec.make_inputs(6, 4)
        norm = assert_model_is_live(spec.factory, xs, min_norm=1e-6)
    assert norm > 1e-6


def test_underdriven_input_scale_is_dead():
    """The counterpart of the above: pins *why* the scale field exists. At
    scale 1.0 a conductance-based model never reaches threshold and its gradient
    is exactly zero, so any comparison on it would be vacuous."""
    spec = SNN_SPECS['lif_expcu']()
    with brainstate.environ.context(dt=0.1 * u.ms):
        dead_xs = spec.make_inputs(6, 4) / spec.input_scale  # undo the scaling
        assert gradient_norm(bptt_param_gradients(spec.factory, dead_xs)) == 0.0


def test_overdriven_input_scale_is_also_dead_while_still_spiking():
    """The live window is bounded *above* as well as below, and this is the half
    that is easy to miss.

    Driven hard, ``ALIF_Delta`` keeps spiking (rate 0.60) but the surrogate
    derivative saturates and the BPTT gradient returns to exactly zero. A
    liveness check keyed on spike rate would pass here and the comparison would
    still assert nothing -- which is why ``assert_model_is_live`` keys on the
    gradient norm instead. See F-25.
    """
    spec = SNN_SPECS['alif_delta']()
    with brainstate.environ.context(dt=0.1 * u.ms):
        live_xs = spec.make_inputs(6, 4)
        over_xs = live_xs * 20.0

        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        outs = brainstate.transform.for_loop(lambda x: model(x), over_xs)
        spike_rate = float(jnp.mean(jnp.asarray(u.get_mantissa(outs)) > 0.0))

        assert gradient_norm(bptt_param_gradients(spec.factory, live_xs)) > 1e-6
        assert spike_rate > 0.1, 'the over-driven network must still be spiking'
        assert gradient_norm(bptt_param_gradients(spec.factory, over_xs)) == 0.0


# ---------------------------------------------------------------------------
# P4 fixtures.
#
# Each of the four exists because an acceptance criterion is vacuous without it.
# The tests below assert exactly that reason, so a later "simplification" of a
# fixture fails here rather than silently weakening the criterion it serves.
# ---------------------------------------------------------------------------


def _inputs(T, n, *, seed=0, scale=1.0):
    rng = np.random.RandomState(seed)
    return scale * jnp.asarray(rng.randn(T, n).astype('float32'))


class TestNonzeroInitRnn:
    """``nonzero_init_rnn`` -- the fixture that can tell full ``D`` from its
    block diagonal."""

    def test_the_initial_hidden_state_is_nonzero_after_init(self):
        from braintrace._algorithm.oracle_models import nonzero_init_rnn
        spec = nonzero_init_rnn(n_rec=2, h0=0.4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        h = np.asarray(model.states(brainstate.HiddenState)[('h',)].value)
        np.testing.assert_allclose(h, 0.4)
        assert h.shape == (1, 2)

    def test_reinitialization_restores_h0_after_a_run(self):
        # The other fixtures in this module set their hidden value in __init__
        # only, so init_all_states leaves a *used* model where the last step left
        # it. A UORO reproducibility pin needs the value back.
        from braintrace._algorithm.oracle_models import nonzero_init_rnn
        spec = nonzero_init_rnn(n_rec=2, h0=0.4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        model(jnp.ones((1, 2)))
        moved = np.asarray(model.states(brainstate.HiddenState)[('h',)].value)
        assert float(np.abs(moved - 0.4).max()) > 1e-3

        brainstate.nn.init_all_states(model, batch_size=1)
        np.testing.assert_allclose(
            np.asarray(model.states(brainstate.HiddenState)[('h',)].value), 0.4)

    def test_the_recurrence_genuinely_mixes_positions(self):
        # The whole point of the fixture: perturbing position p must move
        # position q != p one step later. Where it does not, "roll the full
        # hidden-to-hidden Jacobian" and "roll its block diagonal" are the same
        # instruction and the UORO pin cannot discriminate.
        #
        # Measured on the model, by finite difference, rather than off a
        # compiled group: whether the recurrent mixing reaches a *group
        # transition* depends on the algorithm's recurrence_scope, which is a
        # property of the rule, not of this fixture.
        from braintrace._algorithm.oracle_models import nonzero_init_rnn
        spec = nonzero_init_rnn(n_rec=2, h0=0.4)
        x = jnp.zeros((1, 2))
        eps = 1e-3

        def step_from(h):
            model = spec.factory()
            brainstate.nn.init_all_states(model, batch_size=1)
            model.states(brainstate.HiddenState)[('h',)].value = h
            model(x)
            return np.asarray(model.states(brainstate.HiddenState)[('h',)].value)

        base_h = jnp.full((1, 2), 0.4)
        base = step_from(base_h)
        bumped = step_from(base_h.at[0, 0].add(eps))
        off = abs(float((bumped[0, 1] - base[0, 1]) / eps))
        assert off > 1e-2, f'd h[1] / d h[0] is {off}: positions do not mix'

    def test_the_first_step_trace_is_nonzero_unlike_the_h0_zero_control(self):
        # Why h0 != 0 is load-bearing: with h0 == 0 the recurrent weight's first
        # instantaneous term vanishes, so the transition acts on a zero influence
        # and no choice of transition can be distinguished.
        from braintrace._algorithm.oracle_models import nonzero_init_rnn
        x = jnp.ones((1, 2))
        for h0, want_nonzero in [(0.4, True), (0.0, False)]:
            spec = nonzero_init_rnn(n_rec=2, h0=h0)
            model = spec.factory()
            brainstate.nn.init_all_states(model, batch_size=1)
            algo = braintrace.D_RTRL(model, vjp_method='multi-step')
            algo.compile_graph(braintrace.MultiStepData(x[None]))
            algo.init_etrace_state()
            algo(braintrace.MultiStepData(x[None]))
            mags = [float(jnp.abs(u.get_mantissa(v)).max())
                    for v in jax.tree.leaves(algo._get_etrace_data())]
            assert mags, 'the algorithm has no eligibility trace state'
            if want_nonzero:
                assert max(mags) > 1e-3, f'h0={h0}: trace is {mags}'
            else:
                assert max(mags) == 0.0, f'h0={h0}: trace is {mags}'

    def test_it_has_no_plain_parameters(self):
        from braintrace._algorithm.oracle_models import nonzero_init_rnn
        spec = nonzero_init_rnn()
        assert spec.etp_param_keys == (('w',),)
        assert spec.plain_param_keys == ()


class TestUnitWeightRnn:
    """``unit_weight_rnn`` -- two states, two *different* units, one group."""

    def test_one_group_holds_both_states_with_different_units(self):
        from braintrace._algorithm.oracle_models import unit_weight_rnn
        spec = unit_weight_rnn(n_in=3, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model)
        algo.compile_graph(jnp.ones((1, 3)))
        algo.init_etrace_state()

        group, = algo.graph.hidden_groups
        assert group.num_state == 2
        assert group.varshape == (1, 4)
        hidden = model.states(brainstate.HiddenState)
        units = {u.get_unit(hidden[p].value) for p in group.hidden_paths}
        assert len(units) == 2, f'the units must differ, got {units}'

    def test_a_step_and_a_gradient_both_survive(self):
        from braintrace._algorithm.oracle_models import unit_weight_rnn
        spec = unit_weight_rnn(n_in=3, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        algo(x)  # move off the zero initial state so the trace is live

        g = brainstate.transform.grad(
            lambda a: u.get_mantissa((algo(a) ** 2).sum()),
            model.states(brainstate.ParamState),
        )(x)
        assert set(g) == {('w',)}
        assert float(jnp.abs(u.get_mantissa(g[('w',)])).max()) > 1e-6

    def test_concatenating_the_group_strips_both_units(self):
        # The claim a normaliser depends on: the concatenated hidden vector is a
        # plain mantissa array, so a scalar computed from it is unitless.
        from braintrace._algorithm.oracle_models import unit_weight_rnn
        spec = unit_weight_rnn(n_in=3, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model)
        algo.compile_graph(jnp.ones((1, 3)))
        algo.init_etrace_state()

        group, = algo.graph.hidden_groups
        hidden = model.states(brainstate.HiddenState)
        concat = group.concat_hidden(
            [u.get_mantissa(hidden[p].value) for p in group.hidden_paths])
        assert u.get_unit(concat).dim == u.get_unit(1.0).dim
        assert concat.shape == (1, 4, 2)


class TestPlainAndEtpRnn:
    """``plain_and_etp_rnn`` -- truncated plain credit, with a control."""

    def test_the_windowed_gradient_truncates_win_but_not_wout(self):
        from braintrace._algorithm.oracle_models import plain_and_etp_rnn
        spec = plain_and_etp_rnn(n_in=3, n_rec=4, n_out=2)
        inputs = _inputs(4, 3)
        ref = bptt_param_gradients(spec.factory, inputs)
        got = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            chunk_size=2)

        # w: the trace carries it, so the window costs nothing.
        np.testing.assert_allclose(
            np.asarray(got[('w',)]), np.asarray(ref[('w',)]), atol=1e-5)
        # wout: reaches only its own step's loss, so nothing to truncate.
        np.testing.assert_allclose(
            np.asarray(got[('wout',)]), np.asarray(ref[('wout',)]), atol=1e-5)
        # win: reaches every future loss through the recurrence -- truncated.
        gap = float(np.abs(np.asarray(got[('win',)])
                           - np.asarray(ref[('win',)])).max())
        assert gap > 1e-3, f'win must be truncated by the window, gap={gap}'

    def test_it_declares_both_plain_kinds(self):
        from braintrace._algorithm.oracle_models import plain_and_etp_rnn
        spec = plain_and_etp_rnn()
        assert spec.etp_param_keys == (('w',),)
        assert spec.plain_param_keys == (('win',), ('wout',))

    def test_single_step_zeroes_every_plain_key(self):
        # F-33. Not a truncation -- an absence. Pinned on the fixture that makes
        # it visible, so a single-step rule's silence about plain parameters is a
        # recorded property rather than a surprise.
        from braintrace._algorithm.oracle_models import plain_and_etp_rnn
        spec = plain_and_etp_rnn(n_in=3, n_rec=4, n_out=2)
        inputs = _inputs(4, 3)
        got = online_param_gradients_singlestep_naive(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='single-step'))
        assert float(jnp.abs(got[('win',)]).max()) == 0.0
        assert float(jnp.abs(got[('wout',)]).max()) == 0.0
        assert float(jnp.abs(got[('w',)]).max()) > 1e-6


class TestDelayedRewardRnn:
    """``delayed_reward_rnn`` -- credit that actually spans windows."""

    def test_credit_for_an_early_input_survives_to_the_end(self):
        from braintrace._algorithm.oracle_models import delayed_reward_rnn
        spec = delayed_reward_rnn(n_in=2, n_rec=8, leak=0.95)
        T = 20
        inputs = _inputs(T, 2)

        # d(final output)/d(input at step 0) must be non-negligible, which is
        # the property a bandit task lacks and DNI needs.
        def final_out(xs):
            model = spec.factory()
            brainstate.nn.init_all_states(model, batch_size=1)
            outs = brainstate.transform.for_loop(lambda x: model(x), xs)
            return outs[-1].sum()

        g = jax.grad(final_out)(inputs)
        early = float(np.abs(np.asarray(g[0])).max())
        late = float(np.abs(np.asarray(g[-1])).max())
        assert early > 1e-3, f'no credit reaches step 0 (|g|={early})'
        assert early > 0.05 * late, (
            f'credit decays too fast to span windows: early={early}, late={late}')

    def test_the_output_is_scalar_per_step(self):
        from braintrace._algorithm.oracle_models import delayed_reward_rnn
        spec = delayed_reward_rnn(n_in=2, n_rec=8)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        out = model(jnp.ones((1, 2)))
        assert out.shape == (1, 1)

    def test_a_shorter_leak_kills_the_span_which_is_why_leak_is_a_parameter(self):
        from braintrace._algorithm.oracle_models import delayed_reward_rnn

        def early_credit(leak):
            spec = delayed_reward_rnn(n_in=2, n_rec=8, leak=leak)
            inputs = _inputs(20, 2)

            def final_out(xs):
                model = spec.factory()
                brainstate.nn.init_all_states(model, batch_size=1)
                outs = brainstate.transform.for_loop(lambda x: model(x), xs)
                return outs[-1].sum()

            return float(np.abs(np.asarray(jax.grad(final_out)(inputs)[0])).max())

        assert early_credit(0.95) > 10 * early_credit(0.4)
