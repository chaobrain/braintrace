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

import math

import brainstate
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
import brainunit as u

import braintrace
from braintrace._algorithm import EligibilityTrace
from braintrace._testing import oracle
from braintrace._testing import oracle_models as om
from braintrace._testing.scenario_catalog import PartialPathRNN
from braintrace._algorithm.io_dim_vjp import (
    IODimVjpAlgorithm,
    _contract_hidden_jacobian,
    _expon_smooth,
    _f_trace_bias_correction,
    _format_decay_and_rank,
    _low_pass_filter,
)
from braintrace._algorithm.pp_prop import ES_D_RTRL, pp_prop

EXACT_MODELS = {
    'tanh_rnn': om.tanh_rnn,
    'leaky_linear': om.leaky_linear,
    'stacked_tanh_rnn': om.stacked_tanh_rnn,
    'two_state_rnn': om.two_state_rnn,
}

RNN_CELLS = [
    braintrace.nn.GRUCell,
    braintrace.nn.LSTMCell,
]

MIXED_PATH_RNN_CELLS = [
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


class _ElementWiseTraceNet(brainstate.nn.Module):

    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(jnp.array([0.1, -0.2], dtype=jnp.float32))
        self.h = brainstate.HiddenState(jnp.zeros(2, dtype=jnp.float32))

    def update(self, x):
        self.h.value = jnp.tanh(
            self.h.value + x + braintrace.element_wise(self.w.value)
        )
        return self.h.value


class _MixedOwnershipTraceNet(brainstate.nn.Module):

    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(jnp.ones(2, dtype=jnp.float32))
        self.h = brainstate.HiddenState(jnp.zeros(2, dtype=jnp.float32))

    def update(self, x):
        w = self.w.value
        self.h.value = jnp.tanh(
            self.h.value + x + braintrace.element_wise(w) + 2 * w
        )
        return self.h.value


class _DenseBiasTraceNet(brainstate.nn.Module):

    def __init__(self):
        super().__init__()
        self.w = brainstate.ParamState(jnp.eye(2, dtype=jnp.float32))
        self.b = brainstate.ParamState(jnp.array([0.1, 0.2], dtype=jnp.float32))
        self.h = brainstate.HiddenState(jnp.zeros(2, dtype=jnp.float32))

    def update(self, x):
        self.h.value = jnp.tanh(
            self.h.value + braintrace.matmul(x, self.w.value, self.b.value)
        )
        return self.h.value


class _TwoEmbeddingTraceNet(brainstate.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = brainstate.ParamState(
            jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10
        )
        self.b = brainstate.ParamState(
            jnp.arange(10, dtype=jnp.float32).reshape(5, 2) / 10
        )
        self.h = brainstate.HiddenState(jnp.zeros(2, dtype=jnp.float32))

    def update(self, token):
        self.h.value = jnp.tanh(
            self.h.value
            + braintrace.embedding(token, self.a.value)
            + braintrace.embedding(token, self.b.value)
        )
        return self.h.value


class _SharedRawInputTraceNet(brainstate.nn.Module):

    def __init__(self):
        super().__init__()
        self.a = brainstate.ParamState(jnp.eye(2, dtype=jnp.float32))
        self.b = brainstate.ParamState(jnp.eye(2, dtype=jnp.float32) * 0.5)
        self.h = brainstate.HiddenState(jnp.zeros(2, dtype=jnp.float32))

    def update(self, x):
        self.h.value = jnp.tanh(
            braintrace.matmul(x, self.a.value)
            + braintrace.matmul(x, self.b.value)
        )
        return self.h.value


class _MixedInputTraceNet(brainstate.nn.Module):

    def __init__(self):
        super().__init__()
        self.table = brainstate.ParamState(
            jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10
        )
        self.weight = brainstate.ParamState(jnp.eye(2, dtype=jnp.float32))
        self.h = brainstate.HiddenState(jnp.zeros(2, dtype=jnp.float32))

    def update(self, token, x):
        self.h.value = jnp.tanh(
            braintrace.embedding(token, self.table.value)
            + braintrace.matmul(x, self.weight.value)
        )
        return self.h.value


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

    @pytest.mark.parametrize('bad', [1.0, 1.5, -0.1])
    def test_float_out_of_range_raises(self, bad):
        with pytest.raises(ValueError):
            _format_decay_and_rank(bad)

    def test_zero_decay_is_admitted(self):
        """The bound is ``[0, 1)``, so ``temporal_recursion='none'`` is a float.

        This adds no numerical regime that was not already reachable: rank 1
        maps to decay 0 through the very same function, and the bias
        correction's exponent ``running_index + 1`` is always at least 1, so
        ``0 ** k`` is 0 and the correction factor is exactly 1 — no singularity.
        """
        assert _format_decay_and_rank(0.0) == (0.0, 1)
        assert _format_decay_and_rank(1) == (0.0, 1)

    @pytest.mark.parametrize('bad', [0, -1, -10])
    def test_nonpositive_rank_raises(self, bad):
        with pytest.raises(ValueError):
            _format_decay_and_rank(bad)

    @pytest.mark.parametrize('bad', ['0.9', None, (0.9,)])
    def test_invalid_type_raises(self, bad):
        with pytest.raises(TypeError):
            _format_decay_and_rank(bad)

    @pytest.mark.parametrize('bad', [True, False])
    def test_boolean_is_not_a_rank(self, bad):
        with pytest.raises(TypeError):
            _format_decay_and_rank(bad)

    @pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
    def test_non_finite_decay_raises(self, bad):
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

    @pytest.mark.parametrize('bad', [1.5, -0.2])
    def test_invalid_decay_float_raises(self, bad):
        with pytest.raises(ValueError):
            IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=bad)

    def test_zero_decay_is_admitted(self):
        """``[0, 1)``: decay 0 is the exact ``temporal_recursion='none'`` point."""
        algo = IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=0.0)
        assert algo.decay == 0.0
        assert algo.config.temporal_recursion == ('none', 'none')

    @pytest.mark.parametrize('bad', [0, -3])
    def test_invalid_rank_int_raises(self, bad):
        with pytest.raises(ValueError):
            IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=bad)

    def test_invalid_decay_type_raises(self):
        with pytest.raises(TypeError):
            IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank='bad')

    @pytest.mark.parametrize('method', ['single-step', 'multi-step'])
    def test_vjp_method_stored(self, method):
        algo = IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=0.9, vjp_method=method)
        assert algo.vjp_method == method

    def test_invalid_vjp_method_raises(self):
        with pytest.raises(ValueError):
            IODimVjpAlgorithm(_build(om.tanh_rnn), decay_or_rank=0.9, vjp_method='nope')

    def test_mixed_parameter_ownership_is_rejected(self):
        model = _MixedOwnershipTraceNet()
        brainstate.nn.init_all_states(model)
        algo = IODimVjpAlgorithm(model, decay_or_rank=0.9)
        with pytest.raises(
            braintrace.NotSupportedError,
            match='compiled ETP ownership.*unrepresented differentiable path',
        ):
            algo.compile_graph(jnp.ones(2))

    def test_partial_direct_and_indirect_relation_is_rejected(self):
        model = PartialPathRNN(3, 4)
        brainstate.nn.init_all_states(model)
        algo = IODimVjpAlgorithm(model, decay_or_rank=0.9)
        with pytest.raises(
            braintrace.NotSupportedError,
            match='both a direct path and an indirect path',
        ):
            algo.compile_graph(jnp.ones(3))
        assert not algo.is_compiled

    def test_independent_direct_relations_may_share_one_parameter(self):
        spec = om.tied_weight_rnn(n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = IODimVjpAlgorithm(model, decay_or_rank=0.9)
        algo.compile_graph(spec.make_inputs(1, 4)[0])
        assert algo.is_compiled
        assert len(algo.graph.hidden_param_op_relations) == 2

    def test_matching_config_decay_is_accepted(self):
        config = braintrace.ETraceConfig(
            trace_factorization='io_factorized', decay=(0.9, 0.5)
        )
        algo = IODimVjpAlgorithm(
            _build(om.tanh_rnn), decay_or_rank=(0.9, 0.5), config=config
        )
        assert (algo.decay_x, algo.decay_f) == (0.9, 0.5)

    def test_conflicting_config_decay_raises(self):
        config = braintrace.ETraceConfig(
            trace_factorization='io_factorized', decay=(0.9, 0.5)
        )
        with pytest.raises(ValueError, match='decay_or_rank.*config'):
            IODimVjpAlgorithm(
                _build(om.tanh_rnn), decay_or_rank=(0.9, 0.4), config=config
            )

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

    def test_get_etrace_of_elementwise_has_no_x_trace(self):
        model = _ElementWiseTraceNet()
        brainstate.nn.init_all_states(model)
        algo = _compiled(model, x=jnp.ones(2))
        xs, dfs = algo.get_etrace_of(model.w)
        assert xs == {}
        assert dfs

    def test_get_etrace_of_dense_bias_matches_weight_relation(self):
        model = _DenseBiasTraceNet()
        brainstate.nn.init_all_states(model)
        algo = _compiled(model, x=jnp.ones(2))
        weight_traces = algo.get_etrace_of(model.w)
        bias_traces = algo.get_etrace_of(model.b)
        path_traces = algo.get_etrace_of(('b',))
        assert weight_traces[0].keys() == bias_traces[0].keys()
        assert weight_traces[1].keys() == bias_traces[1].keys()
        assert path_traces[0].keys() == bias_traces[0].keys()
        assert path_traces[1].keys() == bias_traces[1].keys()

    def test_get_etrace_of_missing_path_raises_value_error(self):
        model = _DenseBiasTraceNet()
        brainstate.nn.init_all_states(model)
        algo = _compiled(model, x=jnp.ones(2))
        with pytest.raises(ValueError, match='No eligibility trace'):
            algo.get_etrace_of(('missing',))


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
        assert int(algo.running_index.value) == 6

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

    @pytest.mark.parametrize('cls', MIXED_PATH_RNN_CELLS)
    def test_mixed_path_cells_are_rejected(self, cls):
        model = cls(4, 5)
        brainstate.nn.init_all_states(model)
        algo = IODimVjpAlgorithm(model, decay_or_rank=0.9)
        with pytest.raises(
            braintrace.NotSupportedError,
            match='both a direct path and an indirect path',
        ):
            algo.compile_graph(brainstate.random.rand(4))
        assert not algo.is_compiled


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


class TestBiasCorrectionTimeIndex:

    def test_running_index_counts_trace_steps(self):
        spec = om.tanh_rnn(n_in=3, n_rec=4, seed=0)
        xs = spec.make_inputs(6, 3, seed=0)
        algo = _compiled(
            _build(lambda: spec), x=xs[0],
            decay_or_rank=0.9, vjp_method='multi-step',
        )
        algo(braintrace.MultiStepData(xs))
        assert int(algo.running_index.value) == 6

    def test_multistep_solver_receives_the_window_entry_trace_age(self):
        spec = om.nonzero_init_rnn(n_rec=2, h0=0.4, seed=0)
        xs = spec.make_inputs(6, 2, seed=0)
        model = _build(lambda: spec, batch_size=None)
        algo = _compiled(
            model, x=xs[0], decay_or_rank=0.9,
            vjp_method='multi-step',
        )
        weight_vals = {
            path: state.value for path, state in algo.param_states.items()
        }
        hidden_vals = {
            path: state.value for path, state in algo.hidden_states.items()
        }
        other_vals = {
            path: state.value for path, state in algo.other_states.items()
        }
        trace_vals = algo._get_etrace_data()

        first, first_res = algo._update_fn_fwd(
            (braintrace.MultiStepData(xs[:2]),),
            weight_vals, hidden_vals, other_vals, trace_vals, 2,
        )
        second, second_res = algo._update_fn_fwd(
            (braintrace.MultiStepData(xs[2:4]),),
            weight_vals, first[1], first[2], first[3], 4,
        )
        _, third_res = algo._update_fn_fwd(
            (braintrace.MultiStepData(xs[4:]),),
            weight_vals, second[1], second[2], second[3], 6,
        )

        assert [int(first_res[3]), int(second_res[3]), int(third_res[3])] == [0, 2, 4]

    def test_zero_age_bias_correction_is_neutral(self):
        assert float(_f_trace_bias_correction(0.9, 0)) == 1.0

    @pytest.mark.parametrize('decay', [0.0, 0.9, 0.9999])
    def test_trace_age_is_independent_of_window_partition(self, decay):
        spec = om.nonzero_init_rnn(n_rec=2, h0=0.4, seed=0)
        xs = spec.make_inputs(6, 2, seed=0)
        whole_model = _build(lambda: spec, batch_size=None)
        chunked_model = _build(lambda: spec, batch_size=None)
        whole = _compiled(
            whole_model, x=xs[0], decay_or_rank=decay,
            vjp_method='multi-step',
        )
        chunked = _compiled(
            chunked_model, x=xs[0], decay_or_rank=decay,
            vjp_method='multi-step',
        )

        whole(braintrace.MultiStepData(xs))
        chunked(braintrace.MultiStepData(xs[:2]))
        chunked(braintrace.MultiStepData(xs[2:4]))
        chunked(braintrace.MultiStepData(xs[4:]))

        whole_traces = [state.value for state in whole.etrace_xs.values()]
        whole_traces += [state.value for state in whole.etrace_dfs.values()]
        chunked_traces = [state.value for state in chunked.etrace_xs.values()]
        chunked_traces += [state.value for state in chunked.etrace_dfs.values()]
        assert int(whole.running_index.value) == 6
        assert int(chunked.running_index.value) == 6
        assert len(whole_traces) == len(chunked_traces) == 2
        for actual, expected in zip(chunked_traces, whole_traces):
            npt.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
        reference = 1.0 if decay == 0.0 else -math.expm1(6 * math.log(decay))
        npt.assert_allclose(
            _f_trace_bias_correction(decay, whole.running_index.value),
            reference,
            rtol=2e-6,
            atol=1e-12,
        )

    @pytest.mark.parametrize('decay', [0.0, 0.9, 0.9999])
    @pytest.mark.parametrize('completed_steps', [1, 6, 1000, 1001])
    def test_stable_correction_matches_reference(self, decay, completed_steps):
        expected = 1.0 if decay == 0.0 else -math.expm1(
            completed_steps * math.log(decay)
        )
        actual = _f_trace_bias_correction(decay, completed_steps)
        npt.assert_allclose(actual, expected, rtol=2e-6, atol=1e-12)

    def test_high_decay_has_no_step_1000_discontinuity(self):
        before = float(_f_trace_bias_correction(0.9999, 1000))
        after = float(_f_trace_bias_correction(0.9999, 1001))
        assert 1.0 < after / before < 1.002


class TestInputTraceKeys:

    def test_different_embedding_representations_do_not_collide(self):
        model = _TwoEmbeddingTraceNet()
        brainstate.nn.init_all_states(model)
        token = jnp.array(1, dtype=jnp.int32)
        algo = _compiled(model, x=token)
        out = algo(token)
        a_xs, _ = algo.get_etrace_of(model.a)
        b_xs, _ = algo.get_etrace_of(model.b)
        assert out.shape == (2,)
        assert {tuple(value.shape) for value in a_xs.values()} == {(3,)}
        assert {tuple(value.shape) for value in b_xs.values()} == {(5,)}

    def test_untransformed_shared_input_keeps_one_trace(self):
        model = _SharedRawInputTraceNet()
        brainstate.nn.init_all_states(model)
        algo = _compiled(model, x=jnp.ones(2))
        assert len(algo.etrace_xs) == 1

    def test_transformed_and_raw_trace_keys_share_one_comparable_type(self):
        model = _MixedInputTraceNet()
        brainstate.nn.init_all_states(model)
        token = jnp.array(1, dtype=jnp.int32)
        x = jnp.ones(2, dtype=jnp.float32)
        algo = IODimVjpAlgorithm(model, decay_or_rank=0.9)
        algo.compile_graph(token, x)
        algo.init_etrace_state()

        out = algo(token, x)

        assert out.shape == (2,)
        assert len(algo.etrace_xs) == 2
        assert all(isinstance(key, tuple) for key in algo.etrace_xs)


def _docstring_rnn():
    """The exact ``RNN`` model used in the ``IODimVjpAlgorithm`` docstring example."""

    class RNN(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
            self.out = braintrace.nn.Linear(20, 1)

        def update(self, x):
            return x >> self.cell >> self.out

    return RNN()


def test_docstring_compile_example_runs():
    """Verify the ``braintrace.compile`` example in ``IODimVjpAlgorithm``'s docstring.

    Exercises the integer-rank variant (``decay_or_rank=19``) flagged in the
    docstring comment, which is distinct from ``pp_prop_test``'s decay variant.
    """
    model = _docstring_rnn()
    x0 = brainstate.random.randn(1)
    learner = braintrace.compile(model, braintrace.pp_prop, x0, decay_or_rank=19)
    y = learner(x0)
    assert y.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert len(learner.graph.hidden_param_op_relations) >= 1


class TestContractHiddenJacobian:
    """The unrolled Jacobian contraction must agree with the einsum it replaced."""

    @pytest.mark.parametrize('varshape', [(1,), (4,), (3, 7), (2, 5, 3)])
    @pytest.mark.parametrize('num_state', [1, 2, 3, 5])
    def test_matches_einsum(self, varshape, num_state):
        key = jax.random.PRNGKey(0)
        jac_key, trace_key = jax.random.split(key)
        jac = jax.random.normal(jac_key, (*varshape, num_state, num_state))
        trace = jax.random.normal(trace_key, (*varshape, num_state))
        npt.assert_allclose(
            _contract_hidden_jacobian(jac, trace),
            jnp.einsum('...ij,...j->...i', jac, trace),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_identity_jacobian_returns_the_trace(self):
        trace = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
        jac = jnp.broadcast_to(jnp.eye(3), (6, 3, 3))
        npt.assert_allclose(_contract_hidden_jacobian(jac, trace), trace, atol=1e-6)

    def test_zero_jacobian_kills_the_trace(self):
        trace = jax.random.normal(jax.random.PRNGKey(2), (6, 3))
        jac = jnp.zeros((6, 3, 3))
        npt.assert_allclose(_contract_hidden_jacobian(jac, trace), 0.0, atol=1e-7)

    def test_row_ordering_is_output_major(self):
        jac = jnp.asarray([[[0.0, 1.0], [0.0, 0.0]]])
        trace = jnp.asarray([[3.0, 5.0]])
        npt.assert_allclose(_contract_hidden_jacobian(jac, trace), [[5.0, 0.0]])

    def test_is_jit_compatible(self):
        jac = jax.random.normal(jax.random.PRNGKey(3), (4, 2, 2))
        trace = jax.random.normal(jax.random.PRNGKey(4), (4, 2))
        npt.assert_allclose(
            jax.jit(_contract_hidden_jacobian)(jac, trace),
            _contract_hidden_jacobian(jac, trace),
            atol=1e-6,
        )
