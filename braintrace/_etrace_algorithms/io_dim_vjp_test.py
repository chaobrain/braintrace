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

"""Comprehensive tests for ``io_dim_vjp.IODimVjpAlgorithm`` (pp_prop / ES-D-RTRL).

The input-output-dimension VJP algorithm factorises the eligibility trace as an
outer product ``eps ~= eps_f (x) eps_x`` smoothed by a decay factor, so it is an
*approximate* online estimator: its gradient is expected to align *directionally*
with BPTT rather than match it element-wise. Coverage:

* ``_format_decay_and_rank`` (the decay<->rank conversion and its guards);
* the smoothing primitives ``_expon_smooth`` / ``_low_pass_filter``;
* construction & validation (decay_or_rank, vjp_method, fast_solve, aliases);
* eligibility-trace state lifecycle over the *two* trace dicts (xs and dfs);
* forward / update mechanics; and
* gradient behaviour — directional alignment with BPTT and fast/legacy parity.
"""

import brainstate
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
import brainunit as u

import braintrace
from braintrace._etrace_algorithms import EligibilityTrace, oracle
from braintrace._etrace_algorithms import oracle_models as om
from braintrace._etrace_algorithms.io_dim_vjp import (
    IODimVjpAlgorithm,
    _expon_smooth,
    _format_decay_and_rank,
    _low_pass_filter,
)
from braintrace._etrace_algorithms.pp_prop import ES_D_RTRL, pp_prop

EXACT_MODELS = {
    'tanh_rnn': om.tanh_rnn,
    'leaky_linear': om.leaky_linear,
    'stacked_tanh_rnn': om.stacked_tanh_rnn,
    'two_state_rnn': om.two_state_rnn,
}

RNN_CELLS = [
    braintrace.nn.GRUCell,
    braintrace.nn.LSTMCell,
    braintrace.nn.MGUCell,
    braintrace.nn.MinimalRNNCell,
]


def _build(spec_factory, *, batch_size=1):
    model = spec_factory().factory()
    brainstate.nn.init_all_states(model, batch_size=batch_size)
    return model


def _compiled(model, *, x=None, decay_or_rank=0.9, **kwargs):
    x = jnp.ones((3,), dtype='float32') if x is None else x
    algo = IODimVjpAlgorithm(model, decay_or_rank=decay_or_rank, **kwargs)
    algo.compile_graph(x)
    algo.init_etrace_state()
    return algo


# ---------------------------------------------------------------------------
# _format_decay_and_rank
# ---------------------------------------------------------------------------

class TestFormatDecayAndRank:

    @pytest.mark.parametrize('decay,rank', [(0.5, 3), (0.9, 19), (0.99, 199)])
    def test_float_decay_to_rank(self, decay, rank):
        out_decay, out_rank = _format_decay_and_rank(decay)
        assert out_decay == decay
        assert out_rank == rank

    @pytest.mark.parametrize('rank,decay', [(1, 0.0), (3, 0.5), (19, 0.9)])
    def test_int_rank_to_decay(self, rank, decay):
        out_decay, out_rank = _format_decay_and_rank(rank)
        assert out_rank == rank
        assert out_decay == pytest.approx(decay)

    def test_decay_rank_roundtrip(self):
        # 0.9 <-> 19 is the canonical pairing used throughout the suite.
        assert _format_decay_and_rank(0.9) == (0.9, 19)
        assert _format_decay_and_rank(19)[0] == pytest.approx(0.9)

    @pytest.mark.parametrize('bad', [0.0, 1.0, 1.5, -0.1])
    def test_float_out_of_range_raises(self, bad):
        with pytest.raises(AssertionError):
            _format_decay_and_rank(bad)

    @pytest.mark.parametrize('bad', [0, -1, -10])
    def test_nonpositive_rank_raises(self, bad):
        with pytest.raises(AssertionError):
            _format_decay_and_rank(bad)

    @pytest.mark.parametrize('bad', ['0.9', None, (0.9,)])
    def test_invalid_type_raises(self, bad):
        with pytest.raises(ValueError):
            _format_decay_and_rank(bad)


# ---------------------------------------------------------------------------
# Smoothing primitives
# ---------------------------------------------------------------------------

class TestSmoothingHelpers:

    def test_expon_smooth_blends(self):
        old = jnp.array([2.0, 4.0])
        new = jnp.array([4.0, 8.0])
        out = _expon_smooth(old, new, 0.25)
        npt.assert_allclose(out, 0.25 * old + 0.75 * new)

    def test_expon_smooth_none_decays_old(self):
        old = jnp.array([2.0, 4.0])
        npt.assert_allclose(_expon_smooth(old, None, 0.25), 0.25 * old)

    def test_low_pass_filter_accumulates(self):
        old = jnp.array([2.0, 4.0])
        new = jnp.array([1.0, 1.0])
        out = _low_pass_filter(old, new, 0.25)
        npt.assert_allclose(out, 0.25 * old + new)

    def test_low_pass_filter_none_decays_old(self):
        old = jnp.array([2.0, 4.0])
        npt.assert_allclose(_low_pass_filter(old, None, 0.25), 0.25 * old)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_decay_float_stored(self):
        algo = IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=0.9)
        assert algo.decay == 0.9

    def test_rank_int_sets_decay(self):
        algo = IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=19)
        assert algo.decay == pytest.approx(0.9)

    def test_missing_decay_or_rank_raises(self):
        with pytest.raises(TypeError):
            IODimVjpAlgorithm(_build(om.tanh_rnn))

    @pytest.mark.parametrize('bad', [1.5, 0.0, -0.2])
    def test_invalid_decay_float_raises(self, bad):
        with pytest.raises(AssertionError):
            IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=bad)

    @pytest.mark.parametrize('bad', [0, -3])
    def test_invalid_rank_int_raises(self, bad):
        with pytest.raises(AssertionError):
            IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=bad)

    def test_invalid_decay_type_raises(self):
        with pytest.raises(ValueError):
            IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank='bad')

    @pytest.mark.parametrize('method', ['single-step', 'multi-step'])
    def test_vjp_method_stored(self, method):
        algo = IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=0.9, vjp_method=method)
        assert algo.vjp_method == method

    def test_invalid_vjp_method_raises(self):
        with pytest.raises((AssertionError, ValueError)):
            IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=0.9, vjp_method='nope')

    @pytest.mark.parametrize('flag', [True, False])
    def test_fast_solve_stored(self, flag):
        algo = IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=0.9, fast_solve=flag)
        assert algo.fast_solve is flag

    def test_pp_prop_subclass_and_alias(self):
        assert issubclass(pp_prop, IODimVjpAlgorithm)
        assert ES_D_RTRL is pp_prop


# ---------------------------------------------------------------------------
# Eligibility-trace state lifecycle (two dicts: xs and dfs)
# ---------------------------------------------------------------------------

class TestStateLifecycle:

    def test_compile_sets_flag(self):
        algo = IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=0.9)
        assert algo.is_compiled is False
        algo.compile_graph(jnp.ones((3,)))
        assert algo.is_compiled is True

    def test_both_trace_dicts_zero_initialised(self):
        algo = _compiled(_build(om.tanh_rnn))
        assert len(algo.etrace_xs) >= 1
        assert len(algo.etrace_dfs) >= 1
        for state in (*algo.etrace_xs.values(), *algo.etrace_dfs.values()):
            assert isinstance(state, EligibilityTrace)
            for leaf in jax.tree.leaves(state.value):
                npt.assert_array_equal(u.get_mantissa(leaf), jnp.zeros_like(u.get_mantissa(leaf)))

    def test_reset_zeros_both_dicts_and_index(self):
        algo = _compiled(_build(om.tanh_rnn))
        algo.update(jnp.ones((3,)))
        algo.reset_state(batch_size=1)
        assert int(algo.running_index.value) == 0
        for state in (*algo.etrace_xs.values(), *algo.etrace_dfs.values()):
            for leaf in jax.tree.leaves(state.value):
                npt.assert_array_equal(u.get_mantissa(leaf), jnp.zeros_like(u.get_mantissa(leaf)))

    def test_get_etrace_of_returns_xs_dfs_tuple(self):
        model = _build(om.tanh_rnn)
        algo = _compiled(model)
        result = algo.get_etrace_of(model.w)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_etrace_of_plain_weight_raises(self):
        model = _build(om.tanh_rnn)
        algo = _compiled(model)
        with pytest.raises(ValueError):
            algo.get_etrace_of(model.win)

    def test_get_etrace_of_before_compile_raises(self):
        model = _build(om.tanh_rnn)
        algo = IODimVjpAlgorithm(model, decay_or_rank=0.9)
        with pytest.raises(ValueError):
            algo.get_etrace_of(model.w)


# ---------------------------------------------------------------------------
# Forward / update mechanics
# ---------------------------------------------------------------------------

class TestForwardUpdate:

    def test_single_step_output_shape(self):
        algo = _compiled(_build(om.tanh_rnn))
        out = algo(jnp.ones((3,)))
        assert out.shape == (1, 4)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_running_index_increments(self):
        algo = _compiled(_build(om.tanh_rnn))
        assert int(algo.running_index.value) == 0
        algo(jnp.ones((3,)))
        algo(jnp.ones((3,)))
        assert int(algo.running_index.value) == 2

    def test_multi_step_output_leading_dim(self):
        model = _build(om.tanh_rnn)
        inputs = brainstate.random.randn(6, 3)
        algo = IODimVjpAlgorithm(model, decay_or_rank=0.9, vjp_method='multi-step')
        algo.compile_graph(inputs[0])
        algo.init_etrace_state()
        outs = algo(braintrace.MultiStepData(inputs))
        assert outs.shape[0] == 6
        assert bool(jnp.all(jnp.isfinite(outs)))

    def test_traces_change_after_update(self):
        algo = _compiled(_build(om.tanh_rnn))
        before = [u.get_mantissa(jax.tree.leaves(v.value)[0]).copy()
                  for v in (*algo.etrace_xs.values(), *algo.etrace_dfs.values())]
        algo(jnp.ones((3,)))
        after = [u.get_mantissa(jax.tree.leaves(v.value)[0])
                 for v in (*algo.etrace_xs.values(), *algo.etrace_dfs.values())]
        assert any(not bool(jnp.allclose(b, a)) for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# Gradient behaviour — approximate-algorithm contract
# ---------------------------------------------------------------------------

class TestGradientBehavior:

    @pytest.mark.parametrize('name', list(EXACT_MODELS))
    def test_direction_aligned_with_bptt(self, name):
        spec = EXACT_MODELS[name]()
        inputs = brainstate.random.randn(8, 3)
        bptt = oracle.bptt_param_gradients(spec.factory, inputs)
        approx = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: IODimVjpAlgorithm(m, decay_or_rank=0.9, vjp_method='multi-step'),
        )
        oracle.assert_direction_aligned(
            approx, bptt, min_cosine=0.9, min_sign_agreement=0.8, keys=spec.etp_param_keys
        )

    @pytest.mark.parametrize('name', list(EXACT_MODELS))
    def test_fast_solve_matches_legacy_path(self, name):
        spec = EXACT_MODELS[name]()
        inputs = brainstate.random.randn(8, 3)
        fast = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: IODimVjpAlgorithm(
                m, decay_or_rank=0.9, vjp_method='multi-step', fast_solve=True),
        )
        legacy = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: IODimVjpAlgorithm(
                m, decay_or_rank=0.9, vjp_method='multi-step', fast_solve=False),
        )
        oracle.assert_param_gradients_close(
            fast, legacy, atol=1e-6, rtol=1e-6, keys=spec.etp_param_keys
        )

    def test_pp_prop_alias_matches_base_class(self):
        spec = om.tanh_rnn()
        inputs = brainstate.random.randn(8, 3)
        base = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: IODimVjpAlgorithm(m, decay_or_rank=0.9, vjp_method='multi-step'),
        )
        alias = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: pp_prop(m, decay_or_rank=0.9, vjp_method='multi-step'),
        )
        oracle.assert_param_gradients_close(alias, base, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize('decay_or_rank', [0.5, 0.9, 0.99, 5, 19])
    def test_gradients_finite_across_decay_and_rank(self, decay_or_rank):
        spec = om.tanh_rnn()
        inputs = brainstate.random.randn(6, 3)
        grads = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: IODimVjpAlgorithm(
                m, decay_or_rank=decay_or_rank, vjp_method='multi-step'),
        )
        for key in spec.etp_param_keys:
            assert bool(jnp.all(jnp.isfinite(jnp.asarray(grads[key]))))


# ---------------------------------------------------------------------------
# RNN-cell sweep — finite gradients across the public cell zoo
# ---------------------------------------------------------------------------

class TestRNNCells:

    @pytest.mark.parametrize('cls', RNN_CELLS)
    def test_cell_gradients_are_finite(self, cls):
        model = cls(4, 5)
        brainstate.nn.init_all_states(model)
        algo = IODimVjpAlgorithm(model, decay_or_rank=0.9)
        x = brainstate.random.rand(4)
        algo.compile_graph(x)
        algo.init_etrace_state()

        grads = brainstate.transform.grad(
            lambda inp: algo(inp).sum(),
            model.states(brainstate.ParamState),
        )(x)
        leaves = jax.tree.leaves(grads)
        assert leaves
        for leaf in leaves:
            assert bool(jnp.all(jnp.isfinite(u.get_mantissa(leaf))))


# ---------------------------------------------------------------------------
# Internal Batching() mode — elemwise eligibility trace regression
# ---------------------------------------------------------------------------

class TestElemwiseBatchingMode:
    """Regression: ``etp_elemwise`` under ``brainstate.mixin.Batching()`` for the
    IO-dimension (ES-D-RTRL) solver.

    See ``param_dim_vjp_test.TestElemwiseBatchingMode`` for the full description.
    The batched eligibility-trace gradient of a per-element ``element_wise`` weight
    must equal the batch-mean of the per-example unbatched gradients, with no
    leaked batch axis — the elemwise primitive is registered ``batched=False`` so
    the batch axis must be detected by shape and reduced in the solve stage.
    """

    def test_io_dim_batching_matches_per_example(self):
        # Imported lazily from the sibling test module to share the leaky-cell
        # harness and ground-truth comparison without duplicating it.
        from .param_dim_vjp_test import _assert_batching_matches_per_example
        _assert_batching_matches_per_example(
            lambda m, **kw: braintrace.IODimVjpAlgorithm(m, 0.9, **kw)
        )
