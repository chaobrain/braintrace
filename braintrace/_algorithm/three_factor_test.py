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

"""Acceptance tests for ``learning_signal='modulatory'`` (M1--M4).

Every criterion here runs on the **single-step** recipe of
:func:`~braintrace._algorithm.oracle.online_param_gradients_singlestep_naive`,
because :class:`~braintrace.ThreeFactor` is single-step by construction. Under
multi-step, ``_solve_weight_gradients`` adds the within-window reverse-AD
gradient of the ETP parameters on top of the trace contraction, so replacing the
*boundary* signal would leave an unmodulated in-window half -- a hybrid that is
not a three-factor rule. M3 pins that the multi-step spelling is refused.

The signals themselves are captured with ``io_callback``, which is the only
mechanism that yields **concrete** values: ``_compute_learning_signal`` runs
inside the ``custom_vjp`` backward pass, so under an outer
``brainstate.transform.grad`` everything it sees is a tracer. Asserting on
tracers would pin nothing, and asserting on a hand-rolled reimplementation of
the signal would pin the reimplementation.
"""

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import io_callback

import braintrace
from braintrace._algorithm import oracle_models as om
from braintrace._algorithm.axes import ETraceConfig
from braintrace._algorithm.oracle import (
    online_param_gradients_singlestep_naive,
    relative_deviation,
)

H = 2   # hidden width of the single-group fixture
T = 4   # steps in the single-step sweep


def _inputs(t=T, n=H, *, seed=0, scale=0.5):
    rng = np.random.RandomState(seed)
    return scale * jnp.asarray(rng.randn(t, 1, n).astype('float32'))


def _arrays(tree, keys):
    """The compared subtree as plain arrays."""
    return {k: np.asarray(u.get_mantissa(tree[k])) for k in keys}


# ---------------------------------------------------------------------------
# Concrete signal capture.
# ---------------------------------------------------------------------------

_SINK: list = []


def _store_signals(*arrays):
    _SINK.append([np.asarray(a) for a in arrays])
    return np.int32(0)


def _capturing(base):
    """Subclass of ``base`` that records every per-group learning signal."""

    class _Capturing(base):
        def _compute_learning_signal(self, dl_to_hidden_from_autodiff, args):
            sig = super()._compute_learning_signal(dl_to_hidden_from_autodiff, args)
            io_callback(_store_signals, jax.ShapeDtypeStruct((), np.int32), *sig)
            return sig

    _Capturing.__name__ = f'_Capturing{base.__name__}'
    return _Capturing


_CapturingDRTRL = _capturing(braintrace.D_RTRL)


def _capture_symmetric_signals(spec, inputs):
    """The per-step, per-group ``symmetric`` signals of the single-step rule."""
    _SINK.clear()
    online_param_gradients_singlestep_naive(
        spec.factory, inputs,
        algo_factory=lambda m: _CapturingDRTRL(m, vjp_method='single-step'))
    assert len(_SINK) == inputs.shape[0], (
        f'expected one capture per step, got {len(_SINK)}')
    return [list(entry) for entry in _SINK]


# ---------------------------------------------------------------------------
# The single-step runner with a per-step modulator.
#
# Same recipe as ``online_param_gradients_singlestep_naive`` -- one grad call per
# step, summed -- with the modulator refreshed before each step.
# ``test_the_local_runner_agrees_with_the_oracle_helper`` pins the two together
# on a constant modulator, so the local copy cannot drift.
# ---------------------------------------------------------------------------

def _singlestep_gradients(spec, inputs, *, modulator, algo_factory=None,
                          capture=False):
    """Sum of per-step single-step gradients, modulator set before each step.

    Parameters
    ----------
    modulator : callable or array
        ``modulator(t)`` per step, or one array reused at every step.
    """
    algo_factory = algo_factory or (
        lambda m: braintrace.ThreeFactor(m, vjp_method='single-step'))
    if capture:
        _SINK.clear()

    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()
    params = model.states(brainstate.ParamState)

    total = None
    for t in range(inputs.shape[0]):
        algo.modulator = modulator(t) if callable(modulator) else modulator
        g = brainstate.transform.grad(
            lambda x: (algo(x) ** 2).sum(), params)(inputs[t])
        total = g if total is None else jax.tree.map(
            lambda a, b: a + b, total, g)
    return total


# ---------------------------------------------------------------------------
# M1: degenerate equality, with the negative control that gives it teeth.
# ---------------------------------------------------------------------------

class TestDegenerateEquality:
    """M1: modulator == dL/dh reproduces ``symmetric`` exactly.

    Single group deliberately: an exactly-``dL/dh`` modulator for a multi-group
    model would need a per-group sequence, which is the binding this axis
    refuses (see :class:`TestTheAntiOSTTPContract`).
    """

    def test_feeding_the_symmetric_signal_back_reproduces_symmetric(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        captured = _capture_symmetric_signals(spec, inputs)
        assert len(captured[0]) == 1, 'the fixture must be single-group'

        symmetric = online_param_gradients_singlestep_naive(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='single-step'))
        modulated = _singlestep_gradients(
            spec, inputs, modulator=lambda t: captured[t][0])

        got = _arrays(modulated, [('w',)])
        want = _arrays(symmetric, [('w',)])
        np.testing.assert_allclose(got[('w',)], want[('w',)],
                                   rtol=1e-6, atol=1e-7)

    def test_a_scaled_modulator_does_not_reproduce_symmetric(self):
        # Without this, "ignore the modulator and return the symmetric signal"
        # passes the criterion above.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        captured = _capture_symmetric_signals(spec, inputs)

        symmetric = online_param_gradients_singlestep_naive(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='single-step'))
        doubled = _singlestep_gradients(
            spec, inputs, modulator=lambda t: 2.0 * captured[t][0])

        dev = relative_deviation(_arrays(doubled, [('w',)]),
                                 _arrays(symmetric, [('w',)]))
        assert dev > 1e-2, f'a 2x modulator must change the gradient (dev={dev})'

    def test_the_etp_gradient_is_linear_in_the_modulator(self):
        # Sharper than "differs": for the single-step rule the trace is
        # signal-independent, so the ETP gradient is exactly linear in the
        # modulator. A rule that mixed in any unmodulated term would fail this
        # while still passing the inequality above.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        captured = _capture_symmetric_signals(spec, inputs)

        once = _singlestep_gradients(
            spec, inputs, modulator=lambda t: captured[t][0])
        twice = _singlestep_gradients(
            spec, inputs, modulator=lambda t: 2.0 * captured[t][0])
        np.testing.assert_allclose(
            _arrays(twice, [('w',)])[('w',)],
            2.0 * _arrays(once, [('w',)])[('w',)],
            rtol=1e-6, atol=1e-7)

    def test_the_local_runner_agrees_with_the_oracle_helper(self):
        # Pins the per-step runner above against the mandated oracle recipe, so
        # M1's equality cannot be an artefact of a divergent local harness.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        r = jnp.asarray(0.7, dtype=jnp.float32)
        local = _singlestep_gradients(spec, inputs, modulator=r)
        oracle = online_param_gradients_singlestep_naive(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.ThreeFactor(
                m, vjp_method='single-step', modulator=r))
        np.testing.assert_allclose(
            _arrays(local, [('w',)])[('w',)],
            _arrays(oracle, [('w',)])[('w',)], rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# M2: the expansion is shape-driven, pinned by captured signal.
# ---------------------------------------------------------------------------

class TestTheExpansionIsShapeDriven:
    """M2: what each group's signal *actually is*, not merely that it is finite."""

    def test_a_scalar_modulator_expands_to_every_group(self):
        # Two groups at diagonal scope (the ThreeFactor coordinate). Asserting
        # only that gradients are finite would also pass for a symmetric
        # fallback, so the captured signal is compared element-wise.
        spec = om.stacked_tanh_rnn(n_in=4, n_rec=4)
        inputs = spec.make_inputs(2, 4)
        r = 0.3
        _singlestep_gradients(
            spec, inputs, modulator=r, capture=True,
            algo_factory=lambda m: _capturing(braintrace.ThreeFactor)(
                m, vjp_method='single-step'))

        assert len(_SINK) == inputs.shape[0]
        for step_signals in _SINK:
            assert len(step_signals) == 2, 'the fixture must have two groups'
            for sig in step_signals:
                # Materialised to the group's full shape, not left as a bare
                # scalar -- otherwise this comparison would be vacuous.
                assert sig.shape == (1, 4, 1), sig.shape
                np.testing.assert_allclose(sig, np.full(sig.shape, r),
                                           rtol=1e-7, atol=0)

    def test_a_varshape_modulator_gains_the_trailing_state_axis(self):
        # NumPy broadcasting aligns trailing axes, so a (1, n_rec) modulator does
        # NOT broadcast to a (1, n_rec, 1) signal on its own. The contract is
        # ``expand_to``: a modulator shaped like varshape gains a trailing
        # size-1 state axis.
        n_rec = 4
        spec = om.stacked_tanh_rnn(n_in=n_rec, n_rec=n_rec)
        inputs = spec.make_inputs(2, n_rec)
        m = jnp.asarray(np.arange(n_rec, dtype='float32')).reshape(1, n_rec)
        _singlestep_gradients(
            spec, inputs, modulator=m, capture=True,
            algo_factory=lambda m_: _capturing(braintrace.ThreeFactor)(
                m_, vjp_method='single-step'))

        # n_rec == 4 while the group count is 2: the expansion is driven by the
        # group's shape, never indexed by group number.
        assert len(_SINK[0]) == 2 != n_rec
        want = np.asarray(m).reshape(1, n_rec, 1)
        for step_signals in _SINK:
            for sig in step_signals:
                assert sig.shape == (1, n_rec, 1)
                np.testing.assert_allclose(sig, want, rtol=1e-7, atol=0)

    def test_a_fully_shaped_modulator_is_used_as_is(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        m = jnp.asarray(np.arange(H, dtype='float32')).reshape(1, H, 1)
        _singlestep_gradients(
            spec, inputs, modulator=m, capture=True,
            algo_factory=lambda m_: _capturing(braintrace.ThreeFactor)(
                m_, vjp_method='single-step'))
        for step_signals in _SINK:
            np.testing.assert_allclose(step_signals[0], np.asarray(m),
                                       rtol=1e-7, atol=0)


# ---------------------------------------------------------------------------
# M3: refusals and lifecycle.
# ---------------------------------------------------------------------------

class TestRefusalsAndLifecycle:
    """M3."""

    def _algo(self, spec=None, **kwargs):
        spec = spec or om.nonzero_init_rnn(n_rec=H, h0=0.4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        kwargs.setdefault('vjp_method', 'single-step')
        algo = braintrace.ThreeFactor(model, **kwargs)
        algo.compile_graph(_inputs()[0])
        algo.init_etrace_state()
        return algo

    def test_multi_step_is_refused_naming_the_in_window_term(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        with pytest.raises(ValueError, match='single-step'):
            braintrace.ThreeFactor(model, vjp_method='multi-step')

    def test_a_missing_modulator_raises_instead_of_falling_back(self):
        algo = self._algo()
        with pytest.raises(RuntimeError, match='modulator'):
            algo(_inputs()[0])

    def test_a_per_group_sequence_raises_with_the_anti_osttp_reason(self):
        algo = self._algo()
        for bad in ([jnp.ones((1, H, 1))] * 2, (jnp.ones(()), jnp.ones(()))):
            algo.modulator = bad
            with pytest.raises((TypeError, ValueError), match='per-group|group'):
                algo(_inputs()[0])

    def test_a_non_broadcastable_modulator_names_the_group_and_both_shapes(self):
        # Note the call below is a *forward-only* update, with no outer grad.
        # `_compute_learning_signal` -- where the expansion actually happens --
        # runs inside the custom_vjp backward pass, so validating only there
        # would let a malformed modulator through this call silently and fail
        # later, from inside JAX. The refusal is therefore raised eagerly, at the
        # top of `update()`, against each group's declared signal shape.
        algo = self._algo()
        algo.modulator = jnp.ones((3, 7))
        with pytest.raises(ValueError, match=r'\(3, 7\)') as exc:
            algo(_inputs()[0])
        msg = str(exc.value)
        assert '(1, 2, 1)' in msg, msg     # the group's own shape
        assert 'group' in msg.lower(), msg

    def test_the_keyword_overrides_the_attribute(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        by_attr = _singlestep_gradients(spec, inputs, modulator=1.0)

        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.ThreeFactor(model, vjp_method='single-step',
                                     modulator=-5.0)
        algo.compile_graph(inputs[0])
        algo.init_etrace_state()
        params = model.states(brainstate.ParamState)
        total = None
        for t in range(inputs.shape[0]):
            g = brainstate.transform.grad(
                lambda x: (algo.update(x, modulator=1.0) ** 2).sum(),
                params)(inputs[t])
            total = g if total is None else jax.tree.map(
                lambda a, b: a + b, total, g)
        np.testing.assert_allclose(
            _arrays(total, [('w',)])[('w',)],
            _arrays(by_attr, [('w',)])[('w',)], rtol=1e-6, atol=1e-7)

    def test_an_exception_mid_update_does_not_leave_a_stale_modulator(self):
        # The *keyword* is per call and must be cleared in a ``finally``. The
        # standing attribute is deliberately durable, so it is left unset here --
        # otherwise this test could not tell a leaked keyword from the attribute.
        algo = self._algo()
        assert algo.modulator is None
        with pytest.raises(Exception):
            algo.update(jnp.ones((1, H + 3)), modulator=1.0)  # wrong input width
        with pytest.raises(RuntimeError, match='modulator'):
            algo(_inputs()[0])

    def test_the_standing_attribute_persists_across_calls(self):
        algo = self._algo()
        algo.modulator = 0.5
        for _ in range(2):
            assert jnp.all(jnp.isfinite(algo(_inputs()[0])))

    def test_two_consecutive_calls_with_different_modulators_differ(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        a = _singlestep_gradients(spec, inputs, modulator=1.0)
        b = _singlestep_gradients(spec, inputs, modulator=lambda t: float(t + 1))
        dev = relative_deviation(_arrays(a, [('w',)]), _arrays(b, [('w',)]))
        assert dev > 1e-2, f'a time-varying modulator must matter (dev={dev})'

    def test_the_modulator_is_not_forwarded_to_the_model(self):
        # The model's ``update(self, x)`` takes exactly one argument; if the
        # modulator leaked into the forward call this would raise.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        algo = self._algo(spec)
        algo.modulator = 0.5
        out = algo(_inputs()[0])
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# M4: it does something, measurably.
# ---------------------------------------------------------------------------

class TestItDoesSomethingMeasurably:
    """M4."""

    def test_a_reward_like_scalar_differs_from_symmetric(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        modulated = _singlestep_gradients(spec, inputs, modulator=1.0)
        symmetric = online_param_gradients_singlestep_naive(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='single-step'))
        dev = relative_deviation(_arrays(modulated, [('w',)]),
                                 _arrays(symmetric, [('w',)]))
        assert dev > 1e-2, f'the axis must change the rule (dev={dev})'

    def test_a_zero_modulator_gives_exactly_zero_etp_gradient(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        g = _singlestep_gradients(spec, _inputs(), modulator=0.0)
        np.testing.assert_array_equal(
            _arrays(g, [('w',)])[('w',)], np.zeros((H, H), dtype='float32'))

    def test_flipping_the_modulator_sign_flips_the_gradient(self):
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs()
        pos = _arrays(_singlestep_gradients(spec, inputs, modulator=1.0),
                      [('w',)])[('w',)]
        neg = _arrays(_singlestep_gradients(spec, inputs, modulator=-1.0),
                      [('w',)])[('w',)]
        assert np.abs(pos).max() > 1e-4, 'the gradient must not be trivially zero'
        np.testing.assert_allclose(neg, -pos, rtol=1e-6, atol=1e-7)

    def test_every_plain_parameter_gets_an_exact_zero_which_is_f_33(self):
        """The half of ``ThreeFactor`` a caller is most likely to be bitten by.

        ``ThreeFactor`` is single-step-only (the ctor refuses multi-step, M3), and
        single-step zeroes every **plain** parameter's gradient *exactly* -- F-33
        in ``docs/specs/2026-07-25-known-limitations.md``. So a modulated learner
        trains its ETP-routed weights and nothing else: ``win`` and ``wout`` here
        come back as exact zeros, not as truncated approximations.

        This is a property of ``vjp_method``, not of the modulator, but the two are
        inseparable for this preset, so it is asserted here rather than left for a
        caller to discover by watching a readout never move.
        """
        spec = om.plain_and_etp_rnn(n_in=H, n_rec=H)
        g = _arrays(_singlestep_gradients(spec, _inputs(), modulator=1.0),
                    [('w',), ('win',), ('wout',)])
        assert np.abs(g[('w',)]).max() > 1e-6, (
            'the ETP parameter must be trained, else this test is vacuous')
        for key in [('win',), ('wout',)]:
            np.testing.assert_array_equal(
                g[key], np.zeros_like(g[key]),
                err_msg=f'{key} must be an exact zero under single-step (F-33)')

    def test_a_scalar_reward_drives_total_activity_in_the_signalled_direction(self):
        # The descent smoke test. With a scalar modulator ``m`` broadcast to every
        # hidden unit, the rule's gradient is ``m * d(sum_t sum_u h_t[u])/dtheta``
        # within its own approximation -- so stepping along ``-grad`` with ``m=+1``
        # must *decrease* total activity, and the same optimiser with ``m=-1``
        # must *increase* it. The mirrored control is the point: a rule that
        # ignored the modulator's sign would move the same way both times.
        #
        # ETP-only fixture on purpose: under single-step every *plain*
        # parameter's gradient is exactly zero (F-33), so a model with plain
        # parameters would leave part of itself untrained and muddy the reading.
        spec = om.nonzero_init_rnn(n_rec=H, h0=0.4)
        inputs = _inputs(t=6, scale=0.4)

        def activity_after_training(reward, epochs=6, lr=0.05):
            model = spec.factory()
            brainstate.nn.init_all_states(model, batch_size=1)
            algo = braintrace.ThreeFactor(model, vjp_method='single-step',
                                          modulator=reward)
            algo.compile_graph(inputs[0])
            algo.init_etrace_state()
            params = model.states(brainstate.ParamState)

            def total_activity():
                probe = spec.factory()
                brainstate.nn.init_all_states(probe, batch_size=1)
                for key, st in probe.states(brainstate.ParamState).items():
                    st.value = params[key].value
                outs = brainstate.transform.for_loop(
                    lambda x: probe(x), inputs)
                return float(jnp.sum(outs))

            before = total_activity()
            for _ in range(epochs):
                brainstate.nn.init_all_states(model, batch_size=1)
                algo.init_etrace_state()
                for t in range(inputs.shape[0]):
                    g = brainstate.transform.grad(
                        lambda x: (algo(x) ** 2).sum(), params)(inputs[t])
                    for key, st in params.items():
                        st.value = st.value - lr * g[key]
            return before, total_activity()

        before_pos, after_pos = activity_after_training(1.0)
        before_neg, after_neg = activity_after_training(-1.0)
        assert before_pos == before_neg, 'both runs must start from one model'
        assert after_pos < before_pos - 1e-4, (
            f'm=+1 must reduce total activity: {before_pos} -> {after_pos}')
        assert after_neg > before_neg + 1e-4, (
            f'm=-1 must raise total activity: {before_neg} -> {after_neg}')


# ---------------------------------------------------------------------------
# Coordinate and structure.
# ---------------------------------------------------------------------------

class TestStructure:

    def test_the_preset_coordinate_is_d_rtrl_plus_modulatory(self):
        cfg = braintrace.ThreeFactor._default_config
        assert cfg.learning_signal == 'modulatory'
        assert cfg == ETraceConfig(
            trace_factorization='per_param', temporal_recursion='jacobian',
            recurrence_scope='diagonal', learning_signal='modulatory',
            trace_filter='none', update_schedule='per_step')

    def test_the_config_alone_selects_a_per_param_engine(self):
        from braintrace._algorithm.param_dim_vjp import ParamDimVjpAlgorithm
        from braintrace._compile import _resolve_algorithm
        assert _resolve_algorithm(
            ETraceConfig(learning_signal='modulatory')) is ParamDimVjpAlgorithm

    def test_the_axis_value_is_accepted_by_the_matrix(self):
        cfg = ETraceConfig(learning_signal='modulatory')
        assert 'modulatory' in cfg.describe()
