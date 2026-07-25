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

"""P2 acceptance: the axes are real, reachable, and mean what the spec says.

``axis_golden_test.py`` pins that the refactor moved nothing. This module pins
the other half — that the axes it introduced actually *do* something, and do the
specific thing claimed. See
``docs/specs/2026-07-25-p2-axis-decomposition.md`` § Test plan.

Every gradient comparison here runs through a **finite window**. The full-window
multi-step path is exact reverse-mode and blind to every learning-rule axis
(F-23), so an assertion made there would pass for all of these regardless of the
configuration.
"""

from __future__ import annotations

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace import ETraceConfig
from braintrace._algorithm import oracle_models as om
from braintrace._algorithm.oracle import (
    assert_gradients_differ,
    assert_model_is_live,
    chunked_online_param_gradients,
    flat_gradient_leaves,
    relative_deviation,
)

T = 8
CHUNK = 2


def _spec(name: str):
    return {
        'tanh_rnn': lambda: om.tanh_rnn(n_in=3, n_rec=4, seed=0),
        'two_state_rnn': lambda: om.two_state_rnn(n_in=3, n_rec=3, seed=0),
        'leaky_linear': lambda: om.leaky_linear(n_in=3, n_rec=4, seed=0),
    }[name]()


def _grads(model_name: str, algo_factory) -> dict:
    """Finite-window gradients of one configuration."""
    spec = _spec(model_name)
    xs = spec.make_inputs(T, 3, seed=0)
    return flat_gradient_leaves(chunked_online_param_gradients(
        _spec(model_name).factory, xs,
        algo_factory=algo_factory, chunk_size=CHUNK,
    ))


def _d_rtrl(config=None, **kwargs):
    return lambda m: braintrace.D_RTRL(
        m, vjp_method='multi-step', config=config, **kwargs)


def _pp_prop(decay=0.9, config=None, **kwargs):
    return lambda m: braintrace.pp_prop(
        m, decay_or_rank=decay, vjp_method='multi-step', config=config, **kwargs)


# ---------------------------------------------------------------------------
# temporal_recursion
# ---------------------------------------------------------------------------

class TestTemporalRecursion:
    """The axis is realised by substituting the executor's Jacobian ``D``."""

    def test_scalar_leak_reproduces_a_truly_leaky_jacobian_exactly(self):
        """The positive control, and the strongest statement available.

        ``leaky_linear``'s recurrence is ``h_t = 0.9 * h_{t-1} + matmul(x, w)``,
        whose hidden-to-hidden Jacobian is ``0.9 * I`` *exactly*. So substituting
        ``scalar_leak`` at the model's own leak must reproduce the true Jacobian
        bit for bit. That pins both halves at once: the substitution is installed
        on this path, and the array it installs is numerically right — a test
        that only checked "the knob changes something" could not distinguish a
        correct substitution from a corrupting one.
        """
        xs = _spec('leaky_linear').make_inputs(T, 3, seed=0)
        assert_model_is_live(_spec('leaky_linear').factory, xs, min_norm=1e-6)

        true_jacobian = _grads('leaky_linear', _d_rtrl())
        substituted = _grads('leaky_linear', _d_rtrl(
            ETraceConfig(temporal_recursion='scalar_leak', decay=0.9)))
        for label, deviation in _leafwise(substituted, true_jacobian).items():
            assert deviation == 0.0, (
                f'{label}: scalar_leak(0.9) deviates from the true 0.9*I '
                f'Jacobian by {deviation:.3e}; it should be identical.'
            )

    def test_a_mismatched_leak_does_move_the_gradient(self):
        """Negative control for the test above: 0.0 there is not vacuous."""
        assert_gradients_differ(
            _grads('leaky_linear', _d_rtrl()),
            _grads('leaky_linear', _d_rtrl(
                ETraceConfig(temporal_recursion='scalar_leak', decay=0.5))),
            min_rel=1e-6,
        )

    def test_none_drops_the_temporal_term(self):
        # Measured 2.9e-01 on this model; `two_state_rnn` has a genuine
        # non-ETP diagonal Jacobian (its v/a coupling is hand-written), so
        # zeroing the transition is visible.
        assert_gradients_differ(
            _grads('two_state_rnn', _d_rtrl()),
            _grads('two_state_rnn', _d_rtrl(
                ETraceConfig(temporal_recursion='none'))),
            min_rel=1e-6,
        )

    def test_none_is_degenerate_where_the_jacobian_is_already_zero(self):
        """An axis is a property of a (rule, model) pair, not of a rule.

        ``tanh_rnn``'s only hidden-to-hidden path runs through its recurrent ETP
        weight, which ``recurrence_scope='diagonal'`` excludes from the
        transition by construction. So ``D`` is identically zero there and
        ``'none'`` — which substitutes zeros — is a no-op. Recording it keeps a
        later "the axis stopped working" regression from hiding behind a model
        that never exercised it.
        """
        deviation = relative_deviation(
            _grads('tanh_rnn', _d_rtrl()),
            _grads('tanh_rnn', _d_rtrl(ETraceConfig(temporal_recursion='none'))),
        )
        assert deviation == 0.0, (
            f'tanh_rnn now distinguishes temporal_recursion="none" from '
            f'"jacobian" ({deviation:.3e}). Its diagonal Jacobian used to be '
            'identically zero; if that changed deliberately, move this model '
            'to the live set.'
        )

    def test_scalar_leak_is_live_even_there(self):
        """...because it *adds* a transition the model does not have."""
        assert_gradients_differ(
            _grads('tanh_rnn', _d_rtrl()),
            _grads('tanh_rnn', _d_rtrl(
                ETraceConfig(temporal_recursion='scalar_leak', decay=0.9))),
            min_rel=1e-6,
        )

    @pytest.mark.parametrize('chunked_trace', [True, False])
    def test_substitution_is_applied_exactly_once_on_both_trace_paths(
        self, chunked_trace
    ):
        """``chunked_trace`` picks the path; the answer must not depend on it.

        ``True`` rolls the trace in ``_update_etrace_data`` (substituted by the
        base before the call); ``False`` fuses the roll into the executor's scan
        (substituted inside the wrapped stepper). Substituting twice on either
        path would square the coefficient and show up here.
        """
        config = ETraceConfig(temporal_recursion='scalar_leak', decay=0.9)
        # Both sides hold `chunked_trace` fixed, so the substitution is the only
        # thing that varies. (Comparing across the two paths instead would fold
        # in their reassociation difference — 3.3e-08 on this model — and
        # measure two things at once.)
        substituted = _grads('leaky_linear',
                             _d_rtrl(config, chunked_trace=chunked_trace))
        unsubstituted = _grads('leaky_linear',
                               _d_rtrl(chunked_trace=chunked_trace))
        # 0.9 is leaky_linear's own leak, so substituting it must reproduce the
        # true Jacobian exactly. A doubled substitution would make the effective
        # leak 0.81 and show up as a large deviation, not a round-off one.
        for label, deviation in _leafwise(substituted, unsubstituted).items():
            assert deviation == 0.0, (
                f'{label} (chunked_trace={chunked_trace}): deviates by '
                f'{deviation:.3e}. If the substitution ran twice the effective '
                'leak would be 0.81 rather than 0.9.'
            )


# ---------------------------------------------------------------------------
# recurrence_scope
# ---------------------------------------------------------------------------

class TestRecurrenceScope:

    def test_coupled_matches_the_ostl_recurrent_preset(self):
        """The lifted axis must reproduce what the class attribute produced."""
        for label, deviation in _leafwise(
            _grads('tanh_rnn', _d_rtrl(ETraceConfig(recurrence_scope='coupled'))),
            _grads('tanh_rnn',
                   lambda m: braintrace.OSTLRecurrent(m, vjp_method='multi-step')),
        ).items():
            assert deviation == 0.0, f'{label}: deviates by {deviation:.3e}'

    def test_coupled_raises_inside_a_descended_scan(self):
        """A scope that cannot be delivered must say so, not degrade silently.

        ``_compiler/scan_descent.py`` analyses a descended body with
        ``include_recurrent_mixing=False`` unconditionally. That was defensible
        while the flag was private and only ``OSTLRecurrent`` set it; once
        ``recurrence_scope`` is a public axis, asking for ``'coupled'`` and
        getting ``'diagonal'`` inside the scan is a trap.
        """
        from braintrace._algorithm.tests.scan_descent_support_test import (
            DESCEND, make_snn_scan_net,
        )
        # loops must exceed the policy's unroll limit or nothing descends and
        # the guard has nothing to fire on.
        x = jnp.ones((4,), dtype='float32')
        braintrace.D_RTRL(  # the diagonal scope compiles fine
            make_snn_scan_net(loops=8), vjp_method='multi-step',
            control_flow=DESCEND()).compile_graph(x)
        with pytest.raises(NotImplementedError, match='descended scan'):
            braintrace.D_RTRL(
                make_snn_scan_net(loops=8), vjp_method='multi-step',
                control_flow=DESCEND(),
                config=ETraceConfig(recurrence_scope='coupled'),
            ).compile_graph(x)

    def test_coupled_is_legal_and_live_under_io_factorization(self):
        """The x-side never consumes ``D``, so only the f-side gates the scope.

        Nothing in-tree exercised this cell before P2; it was measured legal
        (finite, distinguishable at 3.7e-04) before the matrix admitted it.
        """
        config = ETraceConfig(trace_factorization='io_factorized', decay=0.9,
                              recurrence_scope='coupled')
        coupled = _grads('tanh_rnn', _pp_prop(config=config))
        assert all(np.all(np.isfinite(v)) for v in coupled.values())
        assert_gradients_differ(
            _grads('tanh_rnn', _pp_prop()), coupled, min_rel=1e-6)


# ---------------------------------------------------------------------------
# learning_signal
# ---------------------------------------------------------------------------

class TestLearningSignal:

    def test_random_feedback_matches_the_eprop_preset(self):
        """The lift must not have changed what ``EProp(feedback='random')`` does."""
        key = jax.random.PRNGKey(7)
        lifted = _grads('tanh_rnn', _d_rtrl(
            ETraceConfig(learning_signal='random_feedback'),
            random_feedback_key=key))
        preset = _grads('tanh_rnn', lambda m: braintrace.EProp(
            m, feedback='random', random_feedback_key=key,
            vjp_method='multi-step'))
        for label, deviation in _leafwise(lifted, preset).items():
            assert deviation == 0.0, f'{label}: deviates by {deviation:.3e}'

    def test_random_feedback_reaches_the_io_dim_engine(self):
        """The generalisation the decomposition buys.

        The hook only ever sees per-hidden-group signals, which carry no
        trace-factorization structure — so lifting it to the base gives the
        IO-dim engine feedback alignment for free. Measured 1.3e-02 against
        symmetric before this test was written.
        """
        config = ETraceConfig(trace_factorization='io_factorized', decay=0.9,
                              learning_signal='random_feedback')
        projected = _grads('tanh_rnn', _pp_prop(
            config=config, random_feedback_key=jax.random.PRNGKey(7)))
        assert all(np.all(np.isfinite(v)) for v in projected.values())
        assert_gradients_differ(
            _grads('tanh_rnn', _pp_prop()), projected, min_rel=1e-6)


# ---------------------------------------------------------------------------
# trace_filter
# ---------------------------------------------------------------------------

class TestTraceFilter:

    def test_kappa_matches_the_eprop_preset(self):
        lifted = _grads('tanh_rnn', _d_rtrl(
            ETraceConfig(trace_filter='kappa', kappa=0.9)))
        preset = _grads('tanh_rnn', lambda m: braintrace.EProp(
            m, kappa_filter_decay=0.9, vjp_method='multi-step'))
        for label, deviation in _leafwise(lifted, preset).items():
            assert deviation == 0.0, f'{label}: deviates by {deviation:.3e}'

    def test_zero_kappa_is_exactly_d_rtrl(self):
        """``EProp(kappa_filter_decay=0)`` documents this reduction; now it is
        enforced by canonicalisation rather than by a branch at the use site."""
        for label, deviation in _leafwise(
            _grads('tanh_rnn', lambda m: braintrace.EProp(
                m, kappa_filter_decay=0.0, vjp_method='multi-step')),
            _grads('tanh_rnn', _d_rtrl()),
        ).items():
            assert deviation == 0.0, f'{label}: deviates by {deviation:.3e}'


# ---------------------------------------------------------------------------
# the IODim decay split
# ---------------------------------------------------------------------------

class TestDecaySplit:

    def test_a_scalar_decay_is_the_symmetric_pair(self):
        for label, deviation in _leafwise(
            _grads('tanh_rnn', _pp_prop(decay=0.9)),
            _grads('tanh_rnn', _pp_prop(decay=(0.9, 0.9))),
        ).items():
            assert deviation == 0.0, f'{label}: deviates by {deviation:.3e}'

    def test_an_asymmetric_pair_is_a_different_rule(self):
        # x-side leak, f-side instantaneous. Measured 4.7e-03.
        assert_gradients_differ(
            _grads('tanh_rnn', _pp_prop(decay=0.9)),
            _grads('tanh_rnn', _pp_prop(decay=(0.9, 0.0))),
            min_rel=1e-6,
        )

    def test_the_decay_property_reports_the_shared_value(self):
        model = _spec('tanh_rnn').factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.pp_prop(model, decay_or_rank=0.9)
        assert algo.decay == 0.9
        assert algo.decay_x == 0.9 and algo.decay_f == 0.9

    def test_the_decay_property_refuses_to_pick_a_side(self):
        model = _spec('tanh_rnn').factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.pp_prop(model, decay_or_rank=(0.9, 0.5))
        assert algo.decay_x == 0.9 and algo.decay_f == 0.5
        with pytest.raises(AttributeError, match='asymmetric'):
            algo.decay


# ---------------------------------------------------------------------------
# preset coordinates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('factory,expected', [
    (lambda m: braintrace.D_RTRL(m), dict(
        trace_factorization='per_param', temporal_recursion='jacobian',
        recurrence_scope='diagonal', learning_signal='symmetric',
        trace_filter='none')),
    (lambda m: braintrace.OSTLRecurrent(m), dict(
        trace_factorization='per_param', temporal_recursion='jacobian',
        recurrence_scope='coupled', learning_signal='symmetric',
        trace_filter='none')),
    (lambda m: braintrace.EProp(m, kappa_filter_decay=0.9), dict(
        trace_factorization='per_param', temporal_recursion='jacobian',
        recurrence_scope='diagonal', learning_signal='symmetric',
        trace_filter='kappa', kappa=0.9)),
    (lambda m: braintrace.EProp(
        m, feedback='random', random_feedback_key=jax.random.PRNGKey(0)), dict(
        trace_factorization='per_param', learning_signal='random_feedback',
        trace_filter='none')),
    (lambda m: braintrace.pp_prop(m, decay_or_rank=0.9), dict(
        trace_factorization='io_factorized',
        temporal_recursion=('scalar_leak', 'jacobian'),
        recurrence_scope='diagonal', learning_signal='symmetric',
        trace_filter='none', decay=(0.9, 0.9))),
    (lambda m: braintrace.OSTLFeedforward(m), dict(
        trace_factorization='io_factorized',
        temporal_recursion=('scalar_leak', 'jacobian'),
        decay=(1e-6, 1e-6))),
])
def test_preset_coordinates_match_the_spec_table(factory, expected):
    """Each preset's ``.config``, field by field, against the spec's table.

    ``OSTLFeedforward`` is deliberately ``('scalar_leak', 'jacobian')`` and not
    ``'none'``: its default ``decay_or_rank=1e-6`` leaves both recursion terms
    structurally present with a negligible coefficient. The exact ``'none'``
    coordinate is ``decay_or_rank=0.0``.
    """
    model = _spec('tanh_rnn').factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    config = factory(model).config
    for field, value in expected.items():
        assert getattr(config, field) == value, (
            f'{field}: expected {value!r}, got {getattr(config, field)!r}')


def test_ostl_feedforward_reaches_the_exact_none_coordinate():
    model = _spec('tanh_rnn').factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    config = braintrace.OSTLFeedforward(model, decay_or_rank=0.0).config
    assert config.temporal_recursion == ('none', 'none')


# ---------------------------------------------------------------------------
# compile() and the guards
# ---------------------------------------------------------------------------

class TestCompileIntegration:

    def test_a_config_selects_the_engine(self):
        x0 = jnp.ones((1, 3))
        per_param = braintrace.compile(
            _spec('tanh_rnn').factory(), ETraceConfig(), x0, batch_size=1)
        assert isinstance(per_param, braintrace.ParamDimVjpAlgorithm)
        factorized = braintrace.compile(
            _spec('tanh_rnn').factory(),
            ETraceConfig(trace_factorization='io_factorized', decay=0.9),
            x0, batch_size=1)
        assert isinstance(factorized, braintrace.IODimVjpAlgorithm)
        assert factorized.decay_f == 0.9

    def test_an_unnamed_coordinate_compiles_and_runs(self):
        """The point of the decomposition: a rule with no preset name."""
        x0 = jnp.ones((1, 3))
        learner = braintrace.compile(
            _spec('tanh_rnn').factory(),
            ETraceConfig(trace_factorization='io_factorized', decay=(0.9, 0.0)),
            x0, batch_size=1)
        out = learner(x0)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_a_config_in_both_positions_is_rejected(self):
        with pytest.raises(TypeError, match='Pass it once'):
            braintrace.compile(
                _spec('tanh_rnn').factory(), ETraceConfig(),
                jnp.ones((1, 3)), batch_size=1, config=ETraceConfig())

    def test_the_engine_rejects_a_config_for_the_other_factorization(self):
        model = _spec('tanh_rnn').factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        with pytest.raises(ValueError, match='per-parameter trace engine'):
            braintrace.ParamDimVjpAlgorithm(
                model,
                config=ETraceConfig(
                    trace_factorization='io_factorized', decay=0.9),
            )


# ---------------------------------------------------------------------------
# P4 preset coordinates
#
# Field by field, not by equality against a hand-written config: an equality
# assertion silently follows the preset if someone edits it, whereas naming each
# field records what the preset *is supposed to be* and fails when it drifts.
# ---------------------------------------------------------------------------

class TestP4PresetCoordinates:

    def test_uoro_is_random_projection_at_coupled_scope(self):
        cfg = braintrace.UORO._default_config
        assert cfg.trace_factorization == 'random_projection'
        assert cfg.recurrence_scope == 'coupled'
        assert cfg.learning_signal == 'symmetric'
        assert cfg.trace_filter == 'none'
        assert cfg.update_schedule == 'per_step'
        assert cfg.kappa is None and cfg.decay is None and cfg.sparse_n is None

    def test_three_factor_is_d_rtrl_with_a_modulatory_signal(self):
        cfg = braintrace.ThreeFactor._default_config
        ref = braintrace.D_RTRL._default_config
        assert cfg.learning_signal == 'modulatory'
        assert cfg.trace_factorization == ref.trace_factorization == 'per_param'
        assert cfg.temporal_recursion == ref.temporal_recursion
        assert cfg.recurrence_scope == ref.recurrence_scope == 'diagonal'
        assert cfg.trace_filter == ref.trace_filter
        assert cfg.update_schedule == ref.update_schedule

    def test_dni_is_d_rtrl_with_a_bootstrapped_signal(self):
        cfg = braintrace.DNI._default_config
        ref = braintrace.D_RTRL._default_config
        assert cfg.learning_signal == 'bootstrapped'
        assert cfg.trace_factorization == ref.trace_factorization
        assert cfg.temporal_recursion == ref.temporal_recursion
        assert cfg.recurrence_scope == ref.recurrence_scope
        assert cfg.trace_filter == ref.trace_filter
        assert cfg.update_schedule == ref.update_schedule

    def test_the_three_presets_occupy_three_distinct_coordinates(self):
        coords = {
            braintrace.UORO._default_config,
            braintrace.ThreeFactor._default_config,
            braintrace.DNI._default_config,
            braintrace.D_RTRL._default_config,
        }
        assert len(coords) == 4, coords

    def test_the_vjp_method_defaults_match_each_rule(self):
        # Not cosmetic: `modulatory` is meaningless under multi-step (the
        # in-window half would stay unmodulated) and `bootstrapped` is
        # meaningless under single-step (there is no window to have an exit).
        import inspect
        assert inspect.signature(
            braintrace.ThreeFactor.__init__).parameters[
                'vjp_method'].default == 'single-step'
        assert inspect.signature(
            braintrace.DNI.__init__).parameters[
                'vjp_method'].default == 'multi-step'
        assert inspect.signature(
            braintrace.UORO.__init__).parameters[
                'vjp_method'].default == 'multi-step'


def _leafwise(actual: dict, reference: dict) -> dict:
    """Per-leaf relative deviation, keyed by leaf label.

    Per leaf rather than joint: these trees are badly scaled against each other,
    so a joint norm can absorb a large error in the smaller leaf.
    """
    assert set(actual) == set(reference), (
        f'leaf labels differ: {sorted(set(actual) ^ set(reference))}')
    out = {}
    for label in reference:
        ref = np.asarray(reference[label], dtype=np.float64)
        got = np.asarray(actual[label], dtype=np.float64)
        den = float(np.sqrt((ref ** 2).sum()))
        num = float(np.sqrt(((got - ref) ** 2).sum()))
        out[label] = num / den if den > 0.0 else num
    return out
