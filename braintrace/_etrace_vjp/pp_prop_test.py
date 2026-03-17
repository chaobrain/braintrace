# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._etrace_model_test import (
    IF_Delta_Dense_Layer,
    LIF_ExpCo_Dense_Layer,
    ALIF_ExpCo_Dense_Layer,
    LIF_ExpCu_Dense_Layer,
    LIF_STDExpCu_Dense_Layer,
    LIF_STPExpCu_Dense_Layer,
    ALIF_ExpCu_Dense_Layer,
    ALIF_Delta_Dense_Layer,
    ALIF_STDExpCu_Dense_Layer,
    ALIF_STPExpCu_Dense_Layer,
)
from braintrace._etrace_vjp.pp_prop import (
    _format_decay_and_rank,
    _expon_smooth,
    _low_pass_filter,
    IODimVjpAlgorithm,
    ES_D_RTRL,
    pp_prop,
)


# ---------------------------------------------------------------------------
# Helper to create a compiled GRU-based algorithm for integration-style tests
# ---------------------------------------------------------------------------

def _make_compiled_algo(n_in=3, n_rec=4, decay_or_rank=0.9, vjp_method='single-step'):
    """Create a GRUCell model with an IODimVjpAlgorithm, compile it, and return both."""
    gru = braintrace.nn.GRUCell(n_in, n_rec)
    brainstate.nn.init_all_states(gru)
    algo = braintrace.IODimVjpAlgorithm(gru, decay_or_rank=decay_or_rank, vjp_method=vjp_method)
    sample_input = brainstate.random.rand(n_in)
    algo.compile_graph(sample_input)
    return gru, algo


# ===========================================================================
#  Tests for _format_decay_and_rank
# ===========================================================================

class TestFormatDecayAndRank:
    """Unit tests for _format_decay_and_rank."""

    # --- valid float inputs (decay) ---

    def test_float_returns_tuple(self):
        result = _format_decay_and_rank(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_float_decay_preserved(self):
        decay, _ = _format_decay_and_rank(0.9)
        assert decay == 0.9

    def test_float_rank_formula(self):
        """rank = round(2/(1-decay) - 1)"""
        decay_in = 0.9
        _, rank = _format_decay_and_rank(decay_in)
        expected_rank = round(2.0 / (1 - decay_in) - 1)
        assert rank == expected_rank

    def test_float_decay_0_5(self):
        decay, rank = _format_decay_and_rank(0.5)
        assert decay == 0.5
        # round(2/(1-0.5) - 1) = round(4 - 1) = 3
        assert rank == 3

    def test_float_decay_small(self):
        decay, rank = _format_decay_and_rank(0.1)
        assert decay == 0.1
        # round(2/0.9 - 1) = round(2.222.. - 1) = round(1.222..) = 1
        assert rank == round(2.0 / 0.9 - 1)

    def test_float_decay_large(self):
        decay, rank = _format_decay_and_rank(0.99)
        assert decay == 0.99
        # round(2/0.01 - 1) = round(199) = 199
        assert rank == 199

    # --- valid int inputs (rank) ---

    def test_int_returns_tuple(self):
        result = _format_decay_and_rank(5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_int_rank_preserved(self):
        _, rank = _format_decay_and_rank(5)
        assert rank == 5

    def test_int_decay_formula(self):
        """decay = (rank-1)/(rank+1)"""
        rank_in = 5
        decay, _ = _format_decay_and_rank(rank_in)
        expected_decay = (rank_in - 1) / (rank_in + 1)
        assert decay == expected_decay

    def test_int_rank_1(self):
        decay, rank = _format_decay_and_rank(1)
        assert rank == 1
        # (1-1)/(1+1) = 0.0
        assert decay == 0.0

    def test_int_rank_10(self):
        decay, rank = _format_decay_and_rank(10)
        assert rank == 10
        # (10-1)/(10+1) = 9/11
        assert decay == pytest.approx(9 / 11)

    def test_int_rank_large(self):
        decay, rank = _format_decay_and_rank(1000)
        assert rank == 1000
        assert decay == pytest.approx(999 / 1001)

    # --- invalid float inputs ---

    def test_float_zero_raises(self):
        with pytest.raises(AssertionError, match="decay should be in"):
            _format_decay_and_rank(0.0)

    def test_float_one_raises(self):
        with pytest.raises(AssertionError, match="decay should be in"):
            _format_decay_and_rank(1.0)

    def test_float_negative_raises(self):
        with pytest.raises(AssertionError, match="decay should be in"):
            _format_decay_and_rank(-0.5)

    def test_float_greater_than_one_raises(self):
        with pytest.raises(AssertionError, match="decay should be in"):
            _format_decay_and_rank(1.5)

    # --- invalid int inputs ---

    def test_int_zero_raises(self):
        with pytest.raises(AssertionError, match="num_rank should be greater than 0"):
            _format_decay_and_rank(0)

    def test_int_negative_raises(self):
        with pytest.raises(AssertionError, match="num_rank should be greater than 0"):
            _format_decay_and_rank(-3)

    # --- invalid types ---

    def test_string_raises(self):
        with pytest.raises(ValueError, match="num_rank.*int.*or.*decay.*float"):
            _format_decay_and_rank("0.5")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="num_rank.*int.*or.*decay.*float"):
            _format_decay_and_rank(None)

    def test_list_raises(self):
        with pytest.raises(ValueError, match="num_rank.*int.*or.*decay.*float"):
            _format_decay_and_rank([0.5])

    def test_bool_treated_as_int(self):
        # In Python, bool is a subclass of int, so True == 1 and False == 0
        # True (==1) should pass the int branch with rank=1
        decay, rank = _format_decay_and_rank(True)
        assert rank == 1
        assert decay == 0.0

    # --- round-trip consistency ---

    def test_float_to_int_roundtrip_approximate(self):
        """Verify that going float -> (decay, rank) -> decay_from_rank is close to original."""
        original_decay = 0.8
        decay, rank = _format_decay_and_rank(original_decay)
        reconstructed_decay = (rank - 1) / (rank + 1)
        # Because of the rounding in rank, we allow some tolerance
        assert abs(reconstructed_decay - original_decay) < 0.15

    def test_int_to_float_roundtrip_exact(self):
        """int -> (decay, rank) -> rank is always exact."""
        original_rank = 7
        decay, rank = _format_decay_and_rank(original_rank)
        assert rank == original_rank


# ===========================================================================
#  Tests for _expon_smooth
# ===========================================================================

class TestExponSmooth:
    """Unit tests for _expon_smooth."""

    def test_new_is_none_returns_decay_times_old(self):
        old = jnp.array(10.0)
        result = _expon_smooth(old, None, 0.9)
        np.testing.assert_allclose(result, 9.0)

    def test_new_is_none_decay_zero(self):
        old = jnp.array(5.0)
        result = _expon_smooth(old, None, 0.0)
        np.testing.assert_allclose(result, 0.0)

    def test_basic_smoothing(self):
        old = jnp.array(10.0)
        new = jnp.array(20.0)
        decay = 0.8
        # 0.8 * 10 + 0.2 * 20 = 8 + 4 = 12
        result = _expon_smooth(old, new, decay)
        np.testing.assert_allclose(result, 12.0)

    def test_decay_one_ignores_new(self):
        old = jnp.array(10.0)
        new = jnp.array(20.0)
        result = _expon_smooth(old, new, 1.0)
        np.testing.assert_allclose(result, 10.0)

    def test_decay_zero_takes_new(self):
        old = jnp.array(10.0)
        new = jnp.array(20.0)
        result = _expon_smooth(old, new, 0.0)
        np.testing.assert_allclose(result, 20.0)

    def test_array_inputs(self):
        old = jnp.array([1.0, 2.0, 3.0])
        new = jnp.array([10.0, 20.0, 30.0])
        decay = 0.5
        # 0.5 * [1,2,3] + 0.5 * [10,20,30] = [5.5, 11, 16.5]
        result = _expon_smooth(old, new, decay)
        np.testing.assert_allclose(result, jnp.array([5.5, 11.0, 16.5]))

    def test_2d_array(self):
        old = jnp.ones((2, 3))
        new = jnp.full((2, 3), 5.0)
        decay = 0.6
        # 0.6 * 1 + 0.4 * 5 = 0.6 + 2.0 = 2.6
        result = _expon_smooth(old, new, decay)
        np.testing.assert_allclose(result, jnp.full((2, 3), 2.6))

    def test_none_new_array_old(self):
        old = jnp.array([2.0, 4.0])
        decay = 0.7
        result = _expon_smooth(old, None, decay)
        np.testing.assert_allclose(result, jnp.array([1.4, 2.8]))


# ===========================================================================
#  Tests for _low_pass_filter
# ===========================================================================

class TestLowPassFilter:
    """Unit tests for _low_pass_filter."""

    def test_new_is_none_returns_alpha_times_old(self):
        old = jnp.array(10.0)
        result = _low_pass_filter(old, None, 0.9)
        np.testing.assert_allclose(result, 9.0)

    def test_new_is_none_alpha_zero(self):
        old = jnp.array(5.0)
        result = _low_pass_filter(old, None, 0.0)
        np.testing.assert_allclose(result, 0.0)

    def test_basic_filter(self):
        old = jnp.array(10.0)
        new = jnp.array(3.0)
        alpha = 0.8
        # 0.8 * 10 + 3 = 8 + 3 = 11
        result = _low_pass_filter(old, new, alpha)
        np.testing.assert_allclose(result, 11.0)

    def test_alpha_zero_takes_new(self):
        old = jnp.array(10.0)
        new = jnp.array(3.0)
        result = _low_pass_filter(old, new, 0.0)
        np.testing.assert_allclose(result, 3.0)

    def test_alpha_one(self):
        old = jnp.array(10.0)
        new = jnp.array(3.0)
        result = _low_pass_filter(old, new, 1.0)
        # 1.0 * 10 + 3 = 13
        np.testing.assert_allclose(result, 13.0)

    def test_array_inputs(self):
        old = jnp.array([1.0, 2.0, 3.0])
        new = jnp.array([10.0, 20.0, 30.0])
        alpha = 0.5
        # 0.5 * [1,2,3] + [10,20,30] = [10.5, 21, 31.5]
        result = _low_pass_filter(old, new, alpha)
        np.testing.assert_allclose(result, jnp.array([10.5, 21.0, 31.5]))

    def test_2d_array(self):
        old = jnp.ones((2, 3))
        new = jnp.full((2, 3), 5.0)
        alpha = 0.6
        # 0.6 * 1 + 5 = 5.6
        result = _low_pass_filter(old, new, alpha)
        np.testing.assert_allclose(result, jnp.full((2, 3), 5.6))

    def test_none_new_array_old(self):
        old = jnp.array([2.0, 4.0])
        alpha = 0.7
        result = _low_pass_filter(old, None, alpha)
        np.testing.assert_allclose(result, jnp.array([1.4, 2.8]))

    def test_difference_from_expon_smooth(self):
        """
        _low_pass_filter and _expon_smooth differ when new is not None:
          - expon_smooth: decay*old + (1-decay)*new
          - low_pass_filter: alpha*old + new
        """
        old = jnp.array(10.0)
        new = jnp.array(3.0)
        alpha = 0.8

        lpf_result = _low_pass_filter(old, new, alpha)
        es_result = _expon_smooth(old, new, alpha)

        # lpf: 0.8*10 + 3 = 11
        # es:  0.8*10 + 0.2*3 = 8.6
        np.testing.assert_allclose(lpf_result, 11.0)
        np.testing.assert_allclose(es_result, 8.6)
        assert float(lpf_result) != float(es_result)

    def test_both_none_case_identical(self):
        """When new is None, both functions behave identically: alpha * old."""
        old = jnp.array(10.0)
        alpha = 0.8

        lpf_result = _low_pass_filter(old, None, alpha)
        es_result = _expon_smooth(old, None, alpha)

        np.testing.assert_allclose(lpf_result, es_result)


# ===========================================================================
#  Tests for IODimVjpAlgorithm
# ===========================================================================

class TestIODimVjpAlgorithmInit:
    """Tests for IODimVjpAlgorithm.__init__."""

    def test_init_with_float_decay(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9)
        assert algo.decay == 0.9

    def test_init_with_int_rank(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=5)
        expected_decay = 4 / 6  # (5-1)/(5+1)
        assert algo.decay == pytest.approx(expected_decay)

    def test_init_default_vjp_method(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9)
        assert algo.vjp_method == 'single-step'

    def test_init_multi_step_vjp(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9, vjp_method='multi-step')
        assert algo.vjp_method == 'multi-step'

    def test_invalid_vjp_method_raises(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        with pytest.raises(AssertionError, match="single-step.*multi-step"):
            IODimVjpAlgorithm(gru, decay_or_rank=0.9, vjp_method='invalid')

    def test_init_with_name(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9, name='test_algo')
        assert algo.name == 'test_algo'

    def test_init_default_mode(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9)
        assert isinstance(algo.mode, brainstate.mixin.Mode)

    def test_init_custom_mode(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        mode = brainstate.mixin.Batching(8)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9, mode=mode)
        assert algo.mode is mode

    def test_not_compiled_initially(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9)
        assert algo.is_compiled is False


class TestIODimVjpAlgorithmModule:
    """Tests for __module__ attribute."""

    def test_module_is_braintrace(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9)
        assert algo.__module__ == 'braintrace'

    def test_class_module_is_braintrace(self):
        assert IODimVjpAlgorithm.__module__ == 'braintrace'


class TestIODimVjpAlgorithmCompileAndState:
    """Tests for compile_graph, init_etrace_state, and state management."""

    def test_compile_graph_sets_compiled_flag(self):
        _, algo = _make_compiled_algo()
        assert algo.is_compiled is True

    def test_compile_graph_creates_etrace_xs(self):
        _, algo = _make_compiled_algo()
        assert hasattr(algo, 'etrace_xs')
        assert isinstance(algo.etrace_xs, dict)

    def test_compile_graph_creates_etrace_dfs(self):
        _, algo = _make_compiled_algo()
        assert hasattr(algo, 'etrace_dfs')
        assert isinstance(algo.etrace_dfs, dict)

    def test_etrace_xs_are_eligibility_traces(self):
        _, algo = _make_compiled_algo()
        from braintrace._etrace_algorithms import EligibilityTrace
        for v in algo.etrace_xs.values():
            assert isinstance(v, EligibilityTrace)

    def test_etrace_dfs_are_eligibility_traces(self):
        _, algo = _make_compiled_algo()
        from braintrace._etrace_algorithms import EligibilityTrace
        for v in algo.etrace_dfs.values():
            assert isinstance(v, EligibilityTrace)

    def test_etrace_xs_initialized_to_zeros(self):
        _, algo = _make_compiled_algo()
        for v in algo.etrace_xs.values():
            np.testing.assert_array_equal(v.value, jnp.zeros_like(v.value))

    def test_etrace_dfs_initialized_to_zeros(self):
        _, algo = _make_compiled_algo()
        for v in algo.etrace_dfs.values():
            np.testing.assert_array_equal(v.value, jnp.zeros_like(v.value))

    def test_compile_graph_not_empty_etrace(self):
        """A GRU should produce non-empty etrace dicts."""
        _, algo = _make_compiled_algo()
        assert len(algo.etrace_xs) > 0 or len(algo.etrace_dfs) > 0

    def test_double_compile_is_idempotent(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9)
        sample = brainstate.random.rand(3)
        algo.compile_graph(sample)
        n_xs = len(algo.etrace_xs)
        n_dfs = len(algo.etrace_dfs)
        # Compile again -- should be a no-op
        algo.compile_graph(sample)
        assert len(algo.etrace_xs) == n_xs
        assert len(algo.etrace_dfs) == n_dfs


class TestIODimVjpAlgorithmResetState:
    """Tests for reset_state."""

    def test_reset_state_zeros_etrace_xs(self):
        _, algo = _make_compiled_algo()
        # Pollute the etrace values
        for v in algo.etrace_xs.values():
            v.value = jnp.ones_like(v.value) * 99.0
        algo.reset_state()
        for v in algo.etrace_xs.values():
            np.testing.assert_array_equal(v.value, jnp.zeros_like(v.value))

    def test_reset_state_zeros_etrace_dfs(self):
        _, algo = _make_compiled_algo()
        for v in algo.etrace_dfs.values():
            v.value = jnp.ones_like(v.value) * 99.0
        algo.reset_state()
        for v in algo.etrace_dfs.values():
            np.testing.assert_array_equal(v.value, jnp.zeros_like(v.value))

    def test_reset_state_resets_running_index(self):
        _, algo = _make_compiled_algo()
        algo.running_index.value = 42
        algo.reset_state()
        assert algo.running_index.value == 0


class TestGetAndAssignEtraceData:
    """Tests for _get_etrace_data and _assign_etrace_data round-trip."""

    def test_get_etrace_data_returns_tuple_of_two_dicts(self):
        _, algo = _make_compiled_algo()
        data = algo._get_etrace_data()
        assert isinstance(data, tuple)
        assert len(data) == 2
        xs, dfs = data
        assert isinstance(xs, dict)
        assert isinstance(dfs, dict)

    def test_get_etrace_data_keys_match_state_keys(self):
        _, algo = _make_compiled_algo()
        xs, dfs = algo._get_etrace_data()
        assert set(xs.keys()) == set(algo.etrace_xs.keys())
        assert set(dfs.keys()) == set(algo.etrace_dfs.keys())

    def test_get_etrace_data_returns_values_not_states(self):
        """Values returned should be jax arrays, not brainstate.State objects."""
        _, algo = _make_compiled_algo()
        xs, dfs = algo._get_etrace_data()
        for v in xs.values():
            assert not isinstance(v, brainstate.State)
        for v in dfs.values():
            assert not isinstance(v, brainstate.State)

    def test_round_trip_preserves_data(self):
        """get -> modify -> assign -> get should reflect the modification."""
        _, algo = _make_compiled_algo()

        # Get current data (zeros)
        xs, dfs = algo._get_etrace_data()

        # Modify
        new_xs = {k: v + 1.0 for k, v in xs.items()}
        new_dfs = {k: v + 2.0 for k, v in dfs.items()}

        # Assign
        algo._assign_etrace_data((new_xs, new_dfs))

        # Get again
        xs2, dfs2 = algo._get_etrace_data()

        for k in xs.keys():
            np.testing.assert_allclose(xs2[k], xs[k] + 1.0)
        for k in dfs.keys():
            np.testing.assert_allclose(dfs2[k], dfs[k] + 2.0)


class TestGetEtraceOf:
    """Tests for get_etrace_of."""

    def test_get_etrace_of_before_compile_raises(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=0.9)
        with pytest.raises(ValueError, match="not been compiled"):
            algo.get_etrace_of(list(gru.states(brainstate.ParamState).values())[0])

    def test_get_etrace_of_returns_dicts(self):
        gru, algo = _make_compiled_algo()
        param_states = list(gru.states(brainstate.ParamState).values())
        # Find a param that is tracked in the etrace graph
        found = False
        for ps in param_states:
            try:
                etrace_xs, etrace_dfs = algo.get_etrace_of(ps)
                assert isinstance(etrace_xs, dict)
                assert isinstance(etrace_dfs, dict)
                found = True
                break
            except ValueError:
                continue
        # At least one param should be trackable
        assert found, "No parameter found in the etrace graph"

    def test_get_etrace_of_nonexistent_weight_raises(self):
        _, algo = _make_compiled_algo()
        # Create a dummy ParamState that is not part of the model
        fake_param = brainstate.ParamState(jnp.zeros(10))
        with pytest.raises(ValueError, match="Do not the etrace"):
            algo.get_etrace_of(fake_param)


# ===========================================================================
#  Tests for aliases
# ===========================================================================

class TestAliases:
    """Verify that ES_D_RTRL and pp_prop are aliases for IODimVjpAlgorithm."""

    def test_es_d_rtrl_is_iodimvjpalgorithm(self):
        assert ES_D_RTRL is IODimVjpAlgorithm

    def test_pp_prop_is_iodimvjpalgorithm(self):
        assert pp_prop is IODimVjpAlgorithm

    def test_braintrace_es_d_rtrl(self):
        assert braintrace.ES_D_RTRL is IODimVjpAlgorithm

    def test_braintrace_pp_prop(self):
        assert braintrace.pp_prop is IODimVjpAlgorithm

    def test_alias_instance_is_same_class(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = ES_D_RTRL(gru, decay_or_rank=0.9)
        assert isinstance(algo, IODimVjpAlgorithm)

        algo2 = pp_prop(gru, decay_or_rank=0.9)
        assert isinstance(algo2, IODimVjpAlgorithm)


# ===========================================================================
#  Tests for different decay_or_rank values with the full algorithm
# ===========================================================================

class TestIODimVjpAlgorithmDecayVariations:
    """Test that different decay/rank values produce valid algorithms."""

    @pytest.mark.parametrize("decay", [0.1, 0.5, 0.9, 0.99])
    def test_float_decay_values(self, decay):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=decay)
        assert algo.decay == decay

    @pytest.mark.parametrize("rank", [1, 2, 5, 10, 50])
    def test_int_rank_values(self, rank):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = IODimVjpAlgorithm(gru, decay_or_rank=rank)
        expected_decay = (rank - 1) / (rank + 1)
        assert algo.decay == pytest.approx(expected_decay)


# ===========================================================================
#  Tests for forward pass execution
# ===========================================================================

class TestIODimVjpAlgorithmForwardPass:
    """Tests that the compiled algorithm can execute a forward pass."""

    def test_single_step_forward(self):
        n_in, n_rec = 3, 4
        _, algo = _make_compiled_algo(n_in, n_rec)
        inp = brainstate.random.rand(n_in)
        out = algo(inp)
        assert out.shape == (n_rec,)

    def test_multiple_steps_forward(self):
        n_in, n_rec, n_seq = 3, 4, 5
        _, algo = _make_compiled_algo(n_in, n_rec)
        inputs = brainstate.random.rand(n_seq, n_in)
        outs = brainstate.transform.for_loop(algo, inputs)
        assert outs.shape == (n_seq, n_rec)

    def test_running_index_increments(self):
        n_in, n_rec = 3, 4
        _, algo = _make_compiled_algo(n_in, n_rec)
        assert algo.running_index.value == 0
        inp = brainstate.random.rand(n_in)
        algo(inp)
        assert algo.running_index.value == 1
        algo(inp)
        assert algo.running_index.value == 2

    def test_etrace_states_change_after_forward(self):
        """After a forward step with non-zero input, etrace states should change."""
        n_in, n_rec = 3, 4
        _, algo = _make_compiled_algo(n_in, n_rec)

        # Get initial etrace data (should be zeros)
        xs0, dfs0 = algo._get_etrace_data()

        # Run one step with non-zero input
        inp = jnp.ones(n_in)
        algo(inp)

        # Get etrace data after one step
        xs1, dfs1 = algo._get_etrace_data()

        # At least some etrace state should have changed
        any_changed = False
        for k in xs0:
            if not jnp.allclose(xs0[k], xs1[k]):
                any_changed = True
                break
        if not any_changed:
            for k in dfs0:
                if not jnp.allclose(dfs0[k], dfs1[k]):
                    any_changed = True
                    break
        assert any_changed, "Etrace states should change after a forward step"

    def test_multi_step_vjp_method(self):
        n_in, n_rec, n_seq = 3, 4, 5
        _, algo = _make_compiled_algo(n_in, n_rec, vjp_method='multi-step')
        inputs = brainstate.random.rand(n_seq, n_in)
        out = algo(braintrace.MultiStepData(inputs))
        assert out.shape == (n_seq, n_rec)


# ===========================================================================
#  Tests for gradient computation
# ===========================================================================

class TestIODimVjpAlgorithmGradients:
    """Tests that gradients can be computed through the algorithm."""

    def test_grad_single_step(self):
        n_in, n_rec = 3, 4
        gru, algo = _make_compiled_algo(n_in, n_rec)

        @brainstate.transform.jit
        def compute_grad(inp):
            return brainstate.transform.grad(
                lambda inp: algo(inp).sum(),
                gru.states(brainstate.ParamState)
            )(inp)

        inp = brainstate.random.rand(n_in)
        grads = compute_grad(inp)
        assert isinstance(grads, dict)
        assert len(grads) > 0
        # All gradients should be finite
        for path, g in grads.items():
            leaves = jax.tree.leaves(g)
            for leaf in leaves:
                assert jnp.all(jnp.isfinite(leaf)), f"Non-finite gradient at {path}"

    def test_grad_two_consecutive_steps(self):
        """Gradients should be computable over multiple consecutive steps."""
        n_in, n_rec = 3, 4
        gru, algo = _make_compiled_algo(n_in, n_rec)

        @brainstate.transform.jit
        def compute_grad(inp):
            return brainstate.transform.grad(
                lambda inp: algo(inp).sum(),
                gru.states(brainstate.ParamState)
            )(inp)

        inp1 = brainstate.random.rand(n_in)
        inp2 = brainstate.random.rand(n_in)
        grads1 = compute_grad(inp1)
        grads2 = compute_grad(inp2)
        # Second step gradients should generally differ from first
        assert isinstance(grads2, dict)
        assert len(grads2) > 0


# ===========================================================================
#  Import needed for gradient leaf checking
# ===========================================================================
import jax


class TestDiagOn:

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_single_step_vjp(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        model = brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algorithm = braintrace.IODimVjpAlgorithm(model, decay_or_rank=0.9)
        algorithm.compile_graph(inputs[0])

        outs = brainstate.transform.for_loop(algorithm, inputs)
        print(outs.shape)

        @brainstate.transform.jit
        def grad_single_step_vjp(inp):
            return brainstate.transform.grad(
                lambda inp: algorithm(inp).sum(),
                model.states(brainstate.ParamState)
            )(inp)

        grads = grad_single_step_vjp(inputs[0])
        grads = grad_single_step_vjp(inputs[1])
        print(brainstate.util.PrettyDict(grads))

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_multi_step_vjp(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        model = brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algorithm = braintrace.IODimVjpAlgorithm(model, decay_or_rank=0.9, vjp_method='multi-step')
        algorithm.compile_graph(inputs[0])

        outs = algorithm(braintrace.MultiStepData(inputs))
        print(outs.shape)

        @brainstate.transform.jit
        def grad_single_step_vjp(inp):
            return brainstate.transform.grad(
                lambda inp: algorithm(braintrace.MultiStepData(inp)).sum(),
                model.states(brainstate.ParamState)
            )(inp)

        grads = grad_single_step_vjp(inputs[:1])
        print(brainstate.util.PrettyDict(grads))
        print()
        grads = grad_single_step_vjp(inputs[1:2])
        print(brainstate.util.PrettyDict(grads))

    @pytest.mark.parametrize(
        "cls",
        [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]
    )
    def test_snn_single_step_vjp(self, cls):
        with brainstate.environ.context(dt=0.1 * u.ms):
            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = brainstate.nn.init_all_states(model)

            inputs = brainstate.random.randn(n_seq, n_in)
            algorithm = braintrace.IODimVjpAlgorithm(model, decay_or_rank=0.9)
            algorithm.compile_graph(inputs[0])

            outs = brainstate.transform.for_loop(algorithm, inputs)
            print(outs.shape)

            @brainstate.transform.jit
            def grad_single_step_vjp(inp):
                return brainstate.transform.grad(
                    lambda inp: algorithm(inp).sum(),
                    model.states(brainstate.ParamState)
                )(inp)

            grads = grad_single_step_vjp(inputs[0])
            grads = grad_single_step_vjp(inputs[1])
            print(brainstate.util.PrettyDict(grads))

    @pytest.mark.parametrize(
        "cls",
        [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]
    )
    def test_snn_multi_step_vjp(self, cls):
        with brainstate.environ.context(dt=0.1 * u.ms):
            print(cls)

            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = brainstate.nn.init_all_states(model)

            inputs = brainstate.random.randn(n_seq, n_in)
            algorithm = braintrace.IODimVjpAlgorithm(model, decay_or_rank=0.9, vjp_method='multi-step')
            algorithm.compile_graph(inputs[0])

            outs = algorithm(braintrace.MultiStepData(inputs))
            print(outs.shape)

            @brainstate.transform.jit
            def grad_single_step_vjp(inp):
                return brainstate.transform.grad(
                    lambda inp: algorithm(braintrace.MultiStepData(inp)).sum(),
                    model.states(brainstate.ParamState)
                )(inp)

            grads = grad_single_step_vjp(inputs[:1])
            print(brainstate.util.PrettyDict(grads))
            print()
            grads = grad_single_step_vjp(inputs[1:2])
            print(brainstate.util.PrettyDict(grads))
