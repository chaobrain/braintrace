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

"""Acceptance tests for ``learning_signal='bootstrapped'`` (B1--B5).

The central claim under test is a *routing* claim, not an accuracy claim: the
synthetic exit cotangent must reach the plain parameters (where it makes the sum
over windows telescope to the exact gradient) and must **not** reach the ETP
parameters (whose cross-window credit the eligibility trace already carries).
B2 pins both halves element-wise against an oracle synthesiser, which is the
sharpest available check that the second linear pass is wired correctly.
"""

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm import oracle_models as om
from braintrace._algorithm.axes import ETraceConfig
from braintrace._algorithm.dni import (
    DNI,
    SyntheticGradient,
    train_synthetic_gradient,
)
from braintrace._algorithm.oracle import (
    bptt_param_gradients,
    chunked_online_param_gradients,
    future_hidden_gradients,
    relative_deviation,
)

T = 6      # sequence length
CHUNK = 2  # window size: 3 windows, so 2 interior boundaries carry credit
N_IN = 3
N_REC = 4


def _inputs(t=T, n=N_IN, *, seed=0, scale=0.5):
    rng = np.random.RandomState(seed)
    return scale * jnp.asarray(rng.randn(t, 1, n).astype('float32'))


def _arrays(tree, keys):
    return {k: np.asarray(u.get_mantissa(tree[k])) for k in keys}


def _group_shapes(n_rec=N_REC):
    return {0: (1, n_rec, 1)}


# ---------------------------------------------------------------------------
# The window-indexed oracle synthesiser.
#
# B2 needs ``M(h^{b_k})`` pinned to the *true* future gradient, which is a
# per-window constant rather than a function of ``h``. It rides in through
# ``param_values()`` -- the same synchronous read at the top of ``update()`` that
# a learned synthesiser's weights use -- so the oracle exercises exactly the
# production plumbing rather than a bypass.
# ---------------------------------------------------------------------------

class _OracleSynthesizer(SyntheticGradient):
    def __init__(self, group_shapes, table):
        super().__init__(group_shapes)
        self._table = table      # list over windows of {gid: array}
        self.window = 0

    def param_values(self):
        return {'estimate': self._table[self.window]}

    def apply(self, param_values, group_hiddens):
        return {gid: jnp.asarray(param_values['estimate'][gid])
                for gid in group_hiddens}


class _ConstantSynthesizer(SyntheticGradient):
    """Emits the same non-zero cotangent at every window: a *live* control.

    ``emit_shape`` overrides the shape, to exercise the mismatch refusal.
    """

    def __init__(self, group_shapes, value=0.05, emit_shape=None):
        super().__init__(group_shapes)
        self._value = value
        self._emit_shape = emit_shape

    def param_values(self):
        return {'value': self._value}

    def apply(self, param_values, group_hiddens):
        return {
            gid: jnp.full(self._emit_shape or u.math.shape(h),
                          param_values['value'], dtype=jnp.float32)
            for gid, h in group_hiddens.items()
        }


# ---------------------------------------------------------------------------
# The windowed runner, with a per-window hook.
#
# Same recipe as ``chunked_online_param_gradients``; the hook is what lets the
# oracle synthesiser advance its window index.
# ``test_the_local_runner_agrees_with_the_oracle_helper`` pins the two together.
# ---------------------------------------------------------------------------

def _windowed_gradients(spec, inputs, *, algo_factory, chunk_size=CHUNK,
                        on_window=None):
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()
    params = model.states(brainstate.ParamState)

    total = None
    for k, start in enumerate(range(0, inputs.shape[0], chunk_size)):
        if on_window is not None:
            on_window(algo, k)
        chunk = inputs[start:start + chunk_size]
        g = brainstate.transform.grad(
            lambda seq: (algo(braintrace.MultiStepData(seq)) ** 2).sum(),
            params)(chunk)
        total = g if total is None else jax.tree.map(
            lambda a, b: a + b, total, g)
    return total


def _window_boundaries(t=T, chunk_size=CHUNK):
    """The exit index of each window ``[a_k, b_k)``."""
    return [min(start + chunk_size, t)
            for start in range(0, t, chunk_size)]


def _oracle_table(spec, inputs, chunk_size=CHUNK):
    """``M(h^{b_k})`` pinned to the true future gradient, per window."""
    boundaries = _window_boundaries(inputs.shape[0], chunk_size)
    per_boundary = future_hidden_gradients(spec.factory, inputs, boundaries)

    probe = spec.factory()
    brainstate.nn.init_all_states(probe, batch_size=1)
    algo = braintrace.D_RTRL(probe, vjp_method='multi-step')
    algo.compile_graph(inputs[0])
    groups = algo.graph.hidden_groups

    table = []
    for grads in per_boundary:
        table.append({
            g.index: np.asarray(u.get_mantissa(g.concat_hidden(
                [u.get_mantissa(grads[p]) for p in g.hidden_paths])))
            for g in groups
        })
    return table


# ---------------------------------------------------------------------------
# B1: the no-op, and its other half.
# ---------------------------------------------------------------------------

class TestTheZeroSynthesiserIsANoOp:
    """B1."""

    def test_a_zero_synthesiser_is_bit_exact_against_the_plain_rule(self):
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        keys = [('w',), ('win',), ('wout',)]

        plain = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            chunk_size=CHUNK)
        # A freshly built SyntheticGradient is zero-initialised, so it predicts
        # exactly zero: "DNI off" and "DNI untrained" are the same run.
        dni = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: DNI(
                m, synthesizer=SyntheticGradient(_group_shapes())),
            chunk_size=CHUNK)

        for key in keys:
            np.testing.assert_array_equal(
                _arrays(dni, [key])[key], _arrays(plain, [key])[key],
                err_msg=f'{key} must be bit-identical with M == 0')

    def test_a_live_synthesiser_changes_the_plain_parameter_gradients(self):
        # Without this half, an entirely ignored synthesiser passes B1.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        plain = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            chunk_size=CHUNK)
        live = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: DNI(
                m, synthesizer=_ConstantSynthesizer(_group_shapes())),
            chunk_size=CHUNK)

        dev = relative_deviation(_arrays(live, [('win',)]),
                                 _arrays(plain, [('win',)]))
        assert dev > 1e-3, (
            f'a live synthesiser must move the truncated plain key (dev={dev})')

        # ``wout`` is the readout: ``h @ wout`` uses it only at the step that
        # produces the loss, so it has no cross-window credit to be truncated and
        # nothing for DNI to restore. It must stay put -- a built-in control
        # against over-injection, which no amount of tuning could fake.
        np.testing.assert_array_equal(
            _arrays(live, [('wout',)])[('wout',)],
            _arrays(plain, [('wout',)])[('wout',)],
            err_msg='a readout weight has no truncated future to restore')

    def test_a_live_synthesiser_leaves_the_etp_gradient_untouched(self):
        # The no-double-counting claim, in its cheapest form.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        plain = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            chunk_size=CHUNK)
        live = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: DNI(
                m, synthesizer=_ConstantSynthesizer(_group_shapes())),
            chunk_size=CHUNK)
        np.testing.assert_array_equal(
            _arrays(live, [('w',)])[('w',)], _arrays(plain, [('w',)])[('w',)],
            err_msg='the ETP gradient must not see the injected cotangent')


# ---------------------------------------------------------------------------
# B2: the oracle synthesiser, both halves element-wise.
# ---------------------------------------------------------------------------

class TestTheOracleSynthesiser:
    """B2."""

    def test_the_local_runner_agrees_with_the_oracle_helper(self):
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        factory = (lambda m: DNI(
            m, synthesizer=SyntheticGradient(_group_shapes())))
        local = _windowed_gradients(spec, inputs, algo_factory=factory)
        helper = chunked_online_param_gradients(
            spec.factory, inputs, algo_factory=factory, chunk_size=CHUNK)
        for key in (('w',), ('win',), ('wout',)):
            np.testing.assert_allclose(
                _arrays(local, [key])[key], _arrays(helper, [key])[key],
                rtol=1e-6, atol=1e-7)

    def test_the_plain_keys_equal_full_sequence_bptt(self):
        # The telescoping claim: with the exact future cotangent injected at every
        # window exit, each plain parameter's occurrence inside a window reaches
        # every future loss exactly once.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        table = _oracle_table(spec, inputs)
        synth = _OracleSynthesizer(_group_shapes(), table)

        got = _windowed_gradients(
            spec, inputs,
            algo_factory=lambda m: DNI(m, synthesizer=synth),
            on_window=lambda algo, k: setattr(synth, 'window', k))
        want = bptt_param_gradients(spec.factory, inputs)

        for key in (('win',), ('wout',)):
            dev = relative_deviation(_arrays(got, [key]), _arrays(want, [key]))
            assert dev < 1e-4, f'{key} must telescope onto BPTT (dev={dev})'

    def test_a_zero_synthesiser_does_not_equal_bptt_on_those_keys(self):
        # Otherwise the fixture has no cross-window plain credit and the
        # criterion above is vacuous.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        truncated = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: DNI(
                m, synthesizer=SyntheticGradient(_group_shapes())),
            chunk_size=CHUNK)
        want = bptt_param_gradients(spec.factory, inputs)
        dev = relative_deviation(_arrays(truncated, [('win',)]),
                                 _arrays(want, [('win',)]))
        assert dev > 1e-3, (
            f'the fixture must have truncated cross-window plain credit '
            f'for B2 to mean anything (dev={dev})')

    def test_the_etp_keys_are_bit_identical_to_the_non_dni_run(self):
        # The no-double-counting claim against the *oracle* synthesiser, which is
        # where a mis-routed second pass would show up most strongly.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        table = _oracle_table(spec, inputs)
        synth = _OracleSynthesizer(_group_shapes(), table)

        got = _windowed_gradients(
            spec, inputs,
            algo_factory=lambda m: DNI(m, synthesizer=synth),
            on_window=lambda algo, k: setattr(synth, 'window', k))
        plain = chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
            chunk_size=CHUNK)
        np.testing.assert_array_equal(
            _arrays(got, [('w',)])[('w',)], _arrays(plain, [('w',)])[('w',)])

    def test_the_last_window_gets_a_zero_estimate(self):
        # The half-open interval: the final window has no future, so its injected
        # cotangent must be exactly zero. An off-by-one here would double-count
        # one loss per boundary.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        table = _oracle_table(spec, inputs)
        assert _window_boundaries()[-1] == T
        for gid, arr in table[-1].items():
            np.testing.assert_array_equal(arr, np.zeros_like(arr))
        # ... and the interior boundaries are not zero, or the pin is vacuous.
        assert any(np.abs(arr).max() > 1e-5
                   for arr in table[0].values()), table[0]


# ---------------------------------------------------------------------------
# B3: a learned synthesiser helps, honestly.
# ---------------------------------------------------------------------------

class TestALearnedSynthesiserHelps:
    """B3."""

    def test_training_the_synthesiser_beats_the_zero_estimate_held_out(self):
        """Measured, and both numbers matter.

        Held-out relative deviation from full-sequence BPTT on ``win``:

        =====================================  =========  ==========
        synthesiser                            deviation  regression
        =====================================  =========  ==========
        ``M == 0`` (untrained)                 0.368      --
        trained, 4 boundaries (T=8, 20 ep)     0.412      0.83 -> 0.12
        trained, 20 boundaries (T=40, 10 ep)   0.345      0.61 -> 0.24
        trained, 30 boundaries (T=60, 15 ep)   0.432      0.49 -> 0.16
        =====================================  =========  ==========

        Row 2 is why the training sequence is long: a 4x4 map with a bias is 20
        parameters, and T=8 at ``chunk_size=2`` offers only 4 distinct
        boundaries. The regression fits those four almost perfectly and
        generalises worse than predicting nothing.

        Row 4 is why this criterion is stated as a *demonstration* and not as a
        property of DNI: a lower regression loss does **not** monotonically buy a
        better gradient. The win here is real but small (6%) and sensitive to the
        configuration, because a linear map from ``h`` fitted against a
        *bootstrapped* target -- one that starts at the within-window gradient and
        only improves as ``M`` does -- is a weak predictor of the true future
        cotangent. What B2 establishes is that the *routing* is exact when the
        estimate is; how good an estimate a given synthesiser and optimiser
        produce is a modelling question this repository does not settle.

        ``wout`` is excluded from the comparison: it is the readout, so it has no
        cross-window credit to restore and its (already exact) gradient would only
        dilute the measurement.
        """
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        train_inputs = _inputs(t=40, seed=0)
        held_out = _inputs(t=T, seed=17)
        want = bptt_param_gradients(spec.factory, held_out)
        keys = [('win',)]

        def deviation(synth_factory):
            return relative_deviation(
                _arrays(chunked_online_param_gradients(
                    spec.factory, held_out,
                    algo_factory=lambda m: DNI(m, synthesizer=synth_factory()),
                    chunk_size=CHUNK), keys),
                _arrays(want, keys))

        zero_dev = deviation(lambda: SyntheticGradient(_group_shapes()))

        # Train on the *training* sequence only, with the model frozen and the
        # targets detached (both enforced inside the helper).
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        synth = SyntheticGradient(_group_shapes())
        learner = DNI(model, synthesizer=synth)
        learner.compile_graph(train_inputs[0])
        learner.init_etrace_state()
        before = {k: np.asarray(st.value) for k, st in
                  model.states(brainstate.ParamState).items()}
        # `chunk_size` must match the window the learner is *evaluated* with:
        # the synthesiser predicts the future at a window boundary, and the
        # boundaries move when the window size does.
        history = train_synthetic_gradient(
            learner, train_inputs, chunk_size=CHUNK, epochs=10, lr=0.05)
        after = {k: np.asarray(st.value) for k, st in
                 model.states(brainstate.ParamState).items()}
        for k in before:
            np.testing.assert_array_equal(
                before[k], after[k],
                err_msg=f'the model parameter {k} must stay frozen')
        assert history[-1] < history[0], (
            f'the regression must make progress: {history[0]} -> {history[-1]}')

        trained_dev = deviation(lambda: synth)
        assert trained_dev < zero_dev, (
            f'a trained synthesiser must beat the zero estimate on held-out '
            f'data: {trained_dev} vs {zero_dev}')

    def test_training_on_the_wrong_window_size_is_worse_than_not_training(self):
        """The trap the ``chunk_size`` parameter exists to prevent.

        ``chunk_size=1`` fits ``M`` against the future at *single-step*
        boundaries, then the learner is driven with two-step windows whose
        boundaries are elsewhere and whose futures are longer. Measured: 0.412
        against 0.368 for no synthesiser at all -- so getting this wrong is not
        merely suboptimal, it is worse than leaving DNI off.
        """
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        train_inputs = _inputs(t=8, seed=0)
        held_out = _inputs(t=T, seed=17)
        want = bptt_param_gradients(spec.factory, held_out)
        keys = [('win',)]

        def deviation(synth):
            return relative_deviation(
                _arrays(chunked_online_param_gradients(
                    spec.factory, held_out,
                    algo_factory=lambda m: DNI(m, synthesizer=synth),
                    chunk_size=CHUNK), keys),
                _arrays(want, keys))

        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        synth = SyntheticGradient(_group_shapes())
        learner = DNI(model, synthesizer=synth)
        learner.compile_graph(train_inputs[0])
        learner.init_etrace_state()
        train_synthetic_gradient(learner, train_inputs, chunk_size=1,
                                 epochs=20, lr=0.1)
        mismatched = deviation(synth)
        zero = deviation(SyntheticGradient(_group_shapes()))
        assert mismatched > zero, (
            f'this test documents a hazard; if a window-mismatched fit has '
            f'started helping ({mismatched} vs {zero}), the hazard note in '
            f'`train_synthetic_gradient` needs revisiting')


# ---------------------------------------------------------------------------
# B5: structure.
# ---------------------------------------------------------------------------

class TestStructure:
    """B5."""

    def _compiled(self, spec=None, synth=True, n_rec=N_REC):
        spec = spec or om.plain_and_etp_rnn(n_in=N_IN, n_rec=n_rec)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        s = SyntheticGradient(_group_shapes(n_rec)) if synth else None
        algo = DNI(model, synthesizer=s)
        algo.compile_graph(_inputs(n=N_IN)[0])
        algo.init_etrace_state()
        return algo, s

    def test_the_synthesiser_parameters_are_in_no_etp_relation(self):
        algo, synth = self._compiled()
        synth_ids = {id(st) for st in synth.states_dict().values()}
        for relation in algo.graph.hidden_param_op_relations:
            for path in relation.trainable_paths:
                assert path not in synth.states_dict(), path
        for st in algo.param_states.values():
            assert id(st) not in synth_ids, (
                'the synthesiser must not appear among the model parameters')

    def test_the_injected_cotangent_is_provably_non_zero(self):
        algo, _ = self._compiled(synth=False)
        algo.attach_synthesizer(_ConstantSynthesizer(_group_shapes(), 0.25))
        exit_hiddens = {p: st.value for p, st in algo.hidden_states.items()}
        injected = algo._inject_exit_cotangent(
            exit_hiddens, algo.synthesizer.param_values())
        assert set(injected) == set(exit_hiddens)
        for path, ct in injected.items():
            assert np.abs(np.asarray(u.get_mantissa(ct))).max() > 0.0, path
            assert u.math.shape(ct) == u.math.shape(exit_hiddens[path])

    def test_the_online_loss_gives_zero_gradient_for_the_synthesiser(self):
        # `stop_gradient` on the estimate: the online loss must not be able to
        # train the synthesiser through the injection path.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        synth = _ConstantSynthesizer(_group_shapes(), 0.25)
        algo = DNI(model, synthesizer=synth)
        algo.compile_graph(inputs[0])
        algo.init_etrace_state()

        real_synth = SyntheticGradient(_group_shapes(), scale=0.1, seed=1)
        algo.attach_synthesizer(real_synth)
        g = brainstate.transform.grad(
            lambda seq: (algo(braintrace.MultiStepData(seq)) ** 2).sum(),
            real_synth.states_dict())(inputs[:CHUNK])
        for key, val in g.items():
            np.testing.assert_array_equal(
                np.asarray(val), np.zeros_like(np.asarray(val)),
                err_msg=f'the online loss must not reach {key}')

    def test_the_auxiliary_loss_gives_non_zero_gradient_for_the_synthesiser(self):
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs()
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        synth = SyntheticGradient(_group_shapes(), scale=0.1, seed=1)
        algo = DNI(model, synthesizer=synth)
        algo.compile_graph(inputs[0])
        algo.init_etrace_state()
        history = train_synthetic_gradient(algo, inputs, epochs=2)
        assert len(history) == 2
        assert all(np.isfinite(h) for h in history)
        assert history[0] > 0.0, 'the regression loss must be live'

    def test_the_window_body_is_traced_once_not_once_per_window(self):
        """AGENTS.md rule 10, observed rather than asserted by inspection.

        The window loop drives the learner, so a Python ``for`` re-traces the
        whole ``custom_vjp`` machinery -- the model, the trace update, the
        regression, the optimiser step -- once per window. Under
        ``for_loop`` the body is traced once and ``lax.scan`` repeats it.

        The observable is the number of times the synthesiser's ``apply`` runs at
        *trace* time. Under a Python loop it grows with the window count; under
        ``for_loop`` it does not. Comparing two sequence lengths rather than
        asserting an absolute count keeps this insensitive to how many times the
        body's trace touches ``apply`` (twice: once under ``grad``, once for the
        reported error) and to the terminal pair.
        """
        counts = {}

        class _CountingSynthesizer(SyntheticGradient):
            def apply(self, param_values, hidden):
                counts[self] = counts.get(self, 0) + 1
                return super().apply(param_values, hidden)

        def traces_for(n_steps):
            spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
            model = spec.factory()
            brainstate.nn.init_all_states(model, batch_size=1)
            synth = _CountingSynthesizer(_group_shapes(), scale=0.1, seed=1)
            algo = DNI(model, synthesizer=synth)
            algo.compile_graph(_inputs(t=n_steps)[0])
            algo.init_etrace_state()
            train_synthetic_gradient(algo, _inputs(t=n_steps), chunk_size=1,
                                     epochs=1)
            return counts[synth]

        four, twelve = traces_for(4), traces_for(12)
        assert four == twelve, (
            f'the body was traced {four} times for 4 windows and {twelve} for '
            f'12, so it is being re-traced per window rather than compiled once')

    @pytest.mark.parametrize('n_steps,chunk', [(7, 2), (5, 3), (6, 4)])
    def test_a_ragged_final_window_is_refused_rather_than_truncated(
            self, n_steps, chunk):
        """A short last window fits the wrong target.

        It is not merely that ``for_loop`` needs uniform windows. The
        synthesiser is fit against the future span of a window of *this* length
        and deployed on windows of that same length, so a truncated tail is the
        same mismatch that ``chunk_size``'s own docstring warns about -- just
        introduced by the helper instead of by the caller.
        """
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = DNI(model, synthesizer=SyntheticGradient(_group_shapes()))
        algo.compile_graph(_inputs()[0])
        algo.init_etrace_state()
        with pytest.raises(ValueError, match='chunk_size') as exc:
            train_synthetic_gradient(algo, _inputs(t=n_steps), chunk_size=chunk)
        msg = str(exc.value)
        assert str(n_steps) in msg and str(chunk) in msg, msg

    def test_a_chunk_size_below_one_is_refused(self):
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = DNI(model, synthesizer=SyntheticGradient(_group_shapes()))
        algo.compile_graph(_inputs()[0])
        algo.init_etrace_state()
        with pytest.raises(ValueError, match='chunk_size'):
            train_synthetic_gradient(algo, _inputs(), chunk_size=0)

    def test_the_per_group_mapping_is_correct_on_a_two_group_model(self):
        spec = om.two_island_rnn(n_in=3, n_rec=3)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = DNI(model, synthesizer=SyntheticGradient({}))
        algo.compile_graph(spec.make_inputs(1, 3)[0])
        algo.init_etrace_state()
        shapes = algo.group_signal_shapes()
        assert len(shapes) == 2, shapes

        synth = _ConstantSynthesizer(shapes, 0.25)
        algo.attach_synthesizer(synth)
        exit_hiddens = {p: st.value for p, st in algo.hidden_states.items()}
        injected = algo._inject_exit_cotangent(
            exit_hiddens, synth.param_values())
        # One cotangent per hidden path, each routed back through its own group.
        assert set(injected) == {('ha',), ('hb',)}
        for path, ct in injected.items():
            assert u.math.shape(ct) == u.math.shape(exit_hiddens[path])

    def test_a_shape_mismatch_raises_naming_both_shapes(self):
        algo, _ = self._compiled()
        algo.attach_synthesizer(_ConstantSynthesizer(
            _group_shapes(), emit_shape=(1, N_REC + 5, 1)))
        inputs = _inputs()
        with pytest.raises(ValueError, match=r'shape') as exc:
            brainstate.transform.grad(
                lambda seq: (algo(braintrace.MultiStepData(seq)) ** 2).sum(),
                algo.param_states)(inputs[:CHUNK])
        msg = str(exc.value)
        assert str((1, N_REC + 5, 1)) in msg and str((1, N_REC, 1)) in msg, msg

    def test_a_wrong_state_axis_raises_instead_of_being_truncated(self):
        """The mismatch that used to be swallowed.

        ``group.split_hidden`` cuts the estimate along its **last** axis at the
        group's ``num_state`` boundaries. An estimate with a wider state axis
        therefore splits into more parts than the group has hidden paths, and a
        plain ``zip`` dropped the surplus -- injecting a truncated cotangent whose
        per-path pieces each have an entirely plausible shape, so nothing
        downstream could notice. The check now runs on the whole slab before the
        split, so this case is named rather than trimmed.
        """
        algo, _ = self._compiled()
        algo.attach_synthesizer(_ConstantSynthesizer(
            _group_shapes(), emit_shape=(1, N_REC, 2)))
        inputs = _inputs()
        with pytest.raises(ValueError, match=r'shape') as exc:
            brainstate.transform.grad(
                lambda seq: (algo(braintrace.MultiStepData(seq)) ** 2).sum(),
                algo.param_states)(inputs[:CHUNK])
        msg = str(exc.value)
        assert str((1, N_REC, 2)) in msg and str((1, N_REC, 1)) in msg, msg

    def test_bootstrapped_without_a_synthesiser_raises(self):
        algo, _ = self._compiled(synth=False)
        assert algo.synthesizer is None
        with pytest.raises(RuntimeError, match='synthesizer'):
            algo(braintrace.MultiStepData(_inputs()[:CHUNK]))

    def test_the_preset_coordinate_is_d_rtrl_plus_bootstrapped(self):
        assert DNI._default_config == ETraceConfig(
            trace_factorization='per_param', temporal_recursion='jacobian',
            recurrence_scope='diagonal', learning_signal='bootstrapped',
            trace_filter='none', update_schedule='per_step')

    def test_the_config_alone_selects_a_per_param_engine(self):
        from braintrace._algorithm.param_dim_vjp import ParamDimVjpAlgorithm
        from braintrace._compile import _resolve_algorithm
        assert _resolve_algorithm(
            ETraceConfig(learning_signal='bootstrapped')) is ParamDimVjpAlgorithm

    def test_no_second_pass_runs_without_a_synthesiser_hook(self):
        # The zero-cost path: every other algorithm's `_inject_exit_cotangent`
        # returns None, so the extra `eval_jaxpr` never happens.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='multi-step')
        algo.compile_graph(_inputs()[0])
        algo.init_etrace_state()
        assert algo._inject_exit_cotangent(
            {p: st.value for p, st in algo.hidden_states.items()}, None) is None


# ---------------------------------------------------------------------------
# B4: end-to-end delayed-reward smoke test.
# ---------------------------------------------------------------------------

T_B4, CHUNK_B4, N_REC_B4 = 24, 4, 8
TARGET_B4 = jnp.asarray(0.6, dtype=jnp.float32)
BOUNDS_B4 = [min(s + CHUNK_B4, T_B4) for s in range(0, T_B4, CHUNK_B4)]


def _b4_step_loss(out):
    """The one objective every arm shares -- see the note in B4's docstring."""
    return ((out - TARGET_B4) ** 2).sum()


class _WindowOracleSynthesizer(SyntheticGradient):
    """``M(h^{b_k})`` pinned to the true future gradient of :func:`_b4_step_loss`.

    Unlike :class:`_OracleSynthesizer`, whose table is fixed, this one is
    recomputed against the *current* parameters at the top of every window: the
    true future cotangent is a function of the model, and the model is moving.
    """

    def __init__(self, group_shapes):
        super().__init__(group_shapes)
        self.estimate = None

    def param_values(self):
        return {'estimate': self.estimate}

    def apply(self, param_values, group_hiddens):
        return {gid: jnp.asarray(param_values['estimate'][gid])
                for gid in group_hiddens}


def _b4_oracle_estimate(spec, seq, params, groups, b):
    """The true ``d(sum_{t >= b} l_t)/dh^b`` for the parameters as they stand.

    One window bound, not the whole table. The caller has to refresh this every
    window anyway -- the parameters move -- and only the current window's entry
    is ever read, so building all of ``BOUNDS_B4`` cost six prefix rollouts and
    six suffix VJPs per window to use one of each.
    """
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    for key, st in model.states(brainstate.ParamState).items():
        st.value = params[key].value
    hidden = model.states(brainstate.HiddenState)
    if b > 0:
        brainstate.transform.for_loop(lambda x: model(x), seq[:b])
    if b >= T_B4:
        grads = {k: jax.tree.map(u.math.zeros_like, st.value)
                 for k, st in hidden.items()}
    else:
        grads = brainstate.transform.grad(
            lambda: brainstate.transform.for_loop(
                lambda x: _b4_step_loss(model(x)), seq[b:]).sum(), hidden)()
    return {
        g.index: np.asarray(u.get_mantissa(g.concat_hidden(
            [u.get_mantissa(grads[p]) for p in g.hidden_paths])))
        for g in groups
    }


@pytest.mark.slow
class TestTheDelayedRewardTask:
    """B4: a task whose credit genuinely spans windows."""

    def test_dni_beats_both_controls_on_a_delayed_reward_task(self):
        """Four arms on one model, and the ordering theory predicts.

        Measured, mean squared error over the whole sequence after 15 epochs of
        Adam at ``3e-3`` (the ``seed=0`` row is the one asserted here):

        ====  =======  ========  =========  ========  =======
        seed  before   oracle    trained    frozen    M == 0
        ====  =======  ========  =========  ========  =======
        0     0.595    0.131     0.221      ~0.271    0.271
        1     0.224    0.078     0.121      ~0.128    0.144
        2     0.472    0.130     0.168      ~0.245    0.252
        ====  =======  ========  =========  ========  =======

        ``oracle < trained < M == 0`` on every seed, which is the shape the claim
        has to have: an *exact* future cotangent is the ceiling, a *learned* one
        captures part of the gap, and both beat truncation. The frozen arm --
        live but never trained, the spec's stopped-gradient control -- lands on
        the ``M == 0`` number, as a small random linear map should.

        Two harness properties are load-bearing, and getting either wrong makes
        DNI *worse* than leaving it off. Both were measured that way before being
        fixed, so neither is a hypothetical:

        1. **The synthesiser's ``loss_fn`` must be the objective actually
           descended.** ``train_synthetic_gradient`` defaults to sum-of-squares;
           training here on ``(out - target) ** 2`` while fitting ``M`` against
           ``out ** 2`` fits the gradient of a different function at a different
           scale. Measured with the mismatch: 0.577 against 0.140 for ``M == 0``.
        2. **The synthesiser must be refitted as the model moves.** ``M`` fitted
           once to the initial parameters is stale by the end of training; that
           arm won on one seed of three and lost on the other two.

        The fixture's ``(1 - leak)`` factor is the third: without it the state is
        an accumulator bounded only by 20, the regression target reaches ``1e5``,
        and every arm diverges to ``nan``. See :func:`delayed_reward_rnn`.

        A bandit would not serve as the task: with no temporal credit to carry
        there is nothing for a synthetic gradient to supply and every arm would
        tie.
        """
        spec = om.delayed_reward_rnn(n_in=2, n_rec=N_REC_B4, leak=0.95, seed=0)
        seq = spec.make_inputs(T_B4, 2, seed=100)
        shapes = {0: (1, N_REC_B4, 1)}

        def train(synth, *, mode, epochs=15, lr=3e-3, synth_lr=0.02):
            model = spec.factory()
            brainstate.nn.init_all_states(model, batch_size=1)
            algo = DNI(model, synthesizer=synth)
            algo.compile_graph(seq[0])
            algo.init_etrace_state()
            params = model.states(brainstate.ParamState)
            opt = braintools.optim.Adam(lr)
            opt.register_trainable_weights(params)
            groups = algo.graph.hidden_groups
            opt_synth = None
            if mode == 'trained':
                opt_synth = braintools.optim.Adam(synth_lr)
                opt_synth.register_trainable_weights(synth.states_dict())

            def evaluate():
                probe = spec.factory()
                brainstate.nn.init_all_states(probe, batch_size=1)
                for key, st in probe.states(brainstate.ParamState).items():
                    st.value = params[key].value
                outs = brainstate.transform.for_loop(lambda x: probe(x), seq)
                return float(jnp.mean((outs - TARGET_B4) ** 2))

            first = evaluate()
            for _ in range(epochs):
                if mode == 'trained':
                    # Same window size and same loss as the descent below.
                    train_synthetic_gradient(
                        algo, seq, chunk_size=CHUNK_B4, epochs=1,
                        optimizer=opt_synth, loss_fn=_b4_step_loss)
                brainstate.nn.init_all_states(model, batch_size=1)
                algo.init_etrace_state()
                for k, start in enumerate(range(0, T_B4, CHUNK_B4)):
                    if mode == 'oracle':
                        # Refreshed per *window*, not per epoch. The optimiser
                        # steps after every window, so a table built once at the
                        # top of the epoch describes a trajectory the learner has
                        # already left -- from the second window on it would hold
                        # cotangents of stale parameters, and the arm would stop
                        # being an oracle exactly where it starts mattering.
                        synth.estimate = _b4_oracle_estimate(
                            spec, seq, params, groups, BOUNDS_B4[k])
                    g = brainstate.transform.grad(
                        lambda s: _b4_step_loss(
                            algo(braintrace.MultiStepData(s))),
                        params)(seq[start:start + CHUNK_B4])
                    opt.update(brainstate.nn.clip_grad_norm(g, 1.0))
            return first, evaluate()

        oracle_first, oracle_last = train(
            _WindowOracleSynthesizer(shapes), mode='oracle')
        trained_first, trained_last = train(
            SyntheticGradient(shapes, scale=0.05, seed=2), mode='trained')
        frozen_first, frozen_last = train(
            SyntheticGradient(shapes, scale=0.05, seed=2), mode='off')
        zero_first, zero_last = train(SyntheticGradient(shapes), mode='off')

        assert oracle_first == trained_first == frozen_first == zero_first, (
            'every arm must start from one model')
        assert trained_last < trained_first, (
            f'the task must be learnable at all: {trained_first} -> {trained_last}')
        # The exact future cotangent is the ceiling: it must beat truncation.
        assert oracle_last < zero_last, (
            f'an exact estimate must beat the truncated run: '
            f'{oracle_last} vs {zero_last}')
        # A *learned* synthesiser must beat both controls -- having none, and
        # having a live one that was never fitted.
        assert trained_last < zero_last, (
            f'a trained synthesiser must beat the truncated run: '
            f'{trained_last} vs {zero_last}')
        assert trained_last < frozen_last, (
            f'a trained synthesiser must beat an untrained live one: '
            f'{trained_last} vs {frozen_last}')
        # ...and must not beat the exact estimate it is approximating.
        assert oracle_last <= trained_last + 1e-6, (
            f'nothing may beat the oracle: {oracle_last} vs {trained_last}')


# ---------------------------------------------------------------------------
# The two structural boundaries of the pass-2 routing (F-36, F-37).
#
# Both are limitations rather than bugs, and both are pinned here so that a
# future change to the routing has to decide about them on purpose.
# ---------------------------------------------------------------------------

class TestMixedRoutingIsPathGranular:
    """F-36: one parameter used both as an ETP weight and plainly."""

    @staticmethod
    def _spec():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(0):
                    self.w = brainstate.ParamState(
                        0.2 * brainstate.random.randn(N_REC, N_REC))
                self.h = brainstate.HiddenState(jnp.zeros((1, N_REC)))

            def update(self, x):
                # `w` twice: once through the ETP primitive, once plainly.
                self.h.value = jnp.tanh(
                    braintrace.matmul(self.h.value, self.w.value)
                    + x @ self.w.value)
                return self.h.value

        return om.ModelSpec(factory=Net, etp_param_keys=(('w',),),
                            plain_param_keys=())

    def test_the_plain_occurrence_gets_no_future_credit(self):
        """The whole leaf is skipped, plain occurrence included.

        `_etp_routed_paths()` answers per *path*, and the gradient dictionaries are
        keyed per path too, so "add the future term to the plain occurrences of
        `w` but not the ETP ones" is not a statement the routing can make. The
        conservative choice is taken -- skip the leaf, since double-counting the
        ETP half would be the worse error -- and the consequence is that the plain
        half's cross-window credit is lost.

        Pinned by equality against the `M == 0` run: if a future change makes the
        routing occurrence-granular, this test fails and F-36 comes off the list.
        """
        spec = self._spec()
        inputs = _inputs(t=T, n=N_REC)
        live = _arrays(chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: DNI(
                m, synthesizer=_ConstantSynthesizer(_group_shapes(), 0.3)),
            chunk_size=CHUNK), [('w',)])
        off = _arrays(chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: DNI(
                m, synthesizer=SyntheticGradient(_group_shapes())),
            chunk_size=CHUNK), [('w',)])
        np.testing.assert_allclose(live[('w',)], off[('w',)], rtol=1e-6, atol=1e-7)

    def test_a_purely_plain_parameter_does_get_it_which_gives_the_above_teeth(self):
        # Without this half, an implementation that injected nothing anywhere
        # would pass the test above.
        spec = om.plain_and_etp_rnn(n_in=N_IN, n_rec=N_REC)
        inputs = _inputs(t=T)
        live = _arrays(chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: DNI(
                m, synthesizer=_ConstantSynthesizer(_group_shapes(), 0.3)),
            chunk_size=CHUNK), [('win',)])
        off = _arrays(chunked_online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: DNI(
                m, synthesizer=SyntheticGradient(_group_shapes())),
            chunk_size=CHUNK), [('win',)])
        assert np.abs(live[('win',)] - off[('win',)]).max() > 1e-6


class TestOtherStateCreditIsNotInjected:
    """F-37: the injected template zeroes the non-hidden persistent states."""

    def test_the_other_state_slot_is_zeroed_on_a_model_that_has_one(self):
        """Non-vacuously, which the earlier version of this assertion was not.

        The template's `oth_states` slot is zeroed, so a synthesiser cannot supply
        future credit that would arrive through a persistent *non-hidden* state.
        The previous test asserted the same zeroing on a model whose other-state
        tree was empty, where it holds no matter what the code does.
        """

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                with brainstate.random.seed_context(0):
                    self.w = brainstate.ParamState(
                        0.2 * brainstate.random.randn(N_REC, N_REC))
                self.h = brainstate.HiddenState(jnp.zeros((1, N_REC)))
                # Persistent, carried across windows, and not a HiddenState.
                self.trail = brainstate.ShortTermState(jnp.zeros((1, N_REC)))

            def update(self, x):
                self.trail.value = 0.5 * self.trail.value + x
                self.h.value = jnp.tanh(
                    self.trail.value
                    + braintrace.matmul(self.h.value, self.w.value))
                return self.h.value

        model = Net()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = DNI(model, synthesizer=_ConstantSynthesizer(_group_shapes(), 0.3))
        algo.compile_graph(_inputs(n=N_REC)[0])
        algo.init_etrace_state()

        hiddens = {p: st.value for p, st in algo.hidden_states.items()}
        template = (
            {},
            {p: jnp.ones_like(v) for p, v in hiddens.items()},
            {('trail',): jnp.ones((1, N_REC))},
        )
        _dg_out, dg_hidden, dg_oth = algo._exit_cotangent_grads(
            template, algo._inject_exit_cotangent(hiddens, {'value': 0.3}))
        # The model really does have a non-hidden persistent state...
        assert set(dg_oth) == {('trail',)}
        # ...and its slot is zeroed, which is the limitation.
        np.testing.assert_array_equal(
            dg_oth[('trail',)], np.zeros((1, N_REC), dtype='float32'))
        # The hidden slot, by contrast, carries the estimate.
        assert np.abs(np.asarray(u.get_mantissa(dg_hidden[('h',)]))).max() > 0
