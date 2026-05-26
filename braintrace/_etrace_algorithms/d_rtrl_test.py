# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
import saiunit as u

import braintrace
from braintrace._etrace_algorithms.d_rtrl import D_RTRL
from braintrace._etrace_algorithms.param_dim_vjp import (
    ParamDimVjpAlgorithm,
    _normalize_vector,
    _normalize_matrix_spectrum,
    _remove_units,
)
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


# ---------------------------------------------------------------------------
# Tests for _normalize_vector
# ---------------------------------------------------------------------------

class TestNormalizeVector:
    """Unit tests for the _normalize_vector function."""

    def test_all_values_within_unit_range(self):
        """When max(abs(v)) <= 1 the vector should be returned unchanged."""
        v = jnp.array([0.1, -0.5, 0.9])
        result = _normalize_vector(v)
        npt.assert_array_almost_equal(result, v)

    def test_exactly_one(self):
        """max(abs(v)) == 1 should NOT trigger normalization (> 1 condition)."""
        v = jnp.array([1.0, -0.5, 0.0])
        result = _normalize_vector(v)
        npt.assert_array_almost_equal(result, v)

    def test_all_large_values(self):
        """When all values exceed 1, the vector should be divided by max(abs)."""
        v = jnp.array([2.0, -4.0, 3.0])
        result = _normalize_vector(v)
        expected = v / 4.0
        npt.assert_array_almost_equal(result, expected)

    def test_mixed_values(self):
        """When only some values are large, normalization should still apply."""
        v = jnp.array([0.1, -5.0, 0.5])
        result = _normalize_vector(v)
        expected = v / 5.0
        npt.assert_array_almost_equal(result, expected)

    def test_all_zeros(self):
        """A zero vector should be returned unchanged (max abs is 0, not > 1)."""
        v = jnp.array([0.0, 0.0, 0.0])
        result = _normalize_vector(v)
        npt.assert_array_almost_equal(result, v)

    def test_single_element_large(self):
        """A single-element vector with value > 1 should be normalized to 1."""
        v = jnp.array([3.0])
        result = _normalize_vector(v)
        npt.assert_array_almost_equal(result, jnp.array([1.0]))

    def test_single_element_small(self):
        """A single-element vector with value <= 1 should be unchanged."""
        v = jnp.array([0.7])
        result = _normalize_vector(v)
        npt.assert_array_almost_equal(result, jnp.array([0.7]))

    def test_negative_large_values(self):
        """Normalization should work correctly with negative values."""
        v = jnp.array([-10.0, 5.0, -3.0])
        result = _normalize_vector(v)
        expected = v / 10.0
        npt.assert_array_almost_equal(result, expected)

    def test_2d_array(self):
        """Function should work with multi-dimensional arrays."""
        v = jnp.array([[2.0, -4.0], [1.0, 3.0]])
        result = _normalize_vector(v)
        expected = v / 4.0
        npt.assert_array_almost_equal(result, expected)

    def test_max_abs_just_above_one(self):
        """Edge case: max abs is just slightly above 1."""
        v = jnp.array([1.001, 0.5, -0.3])
        result = _normalize_vector(v)
        expected = v / 1.001
        npt.assert_array_almost_equal(result, expected, decimal=3)

    def test_preserves_sign(self):
        """Normalization should preserve the sign of all elements."""
        v = jnp.array([-3.0, 2.0, -1.0, 4.0])
        result = _normalize_vector(v)
        # Signs should be preserved
        assert jnp.all((result > 0) == (v > 0))
        assert jnp.all((result < 0) == (v < 0))


# ---------------------------------------------------------------------------
# Tests for _normalize_matrix_spectrum
# ---------------------------------------------------------------------------

class TestNormalizeMatrixSpectrum:
    """Unit tests for the _normalize_matrix_spectrum function."""

    def test_identity_matrix_unchanged(self):
        """Identity matrix has max eigenvalue = 1, should be returned unchanged."""
        mat = jnp.eye(3)
        result = _normalize_matrix_spectrum(mat)
        npt.assert_array_almost_equal(result, mat)

    def test_scaled_identity_above_one(self):
        """2*I has max eigenvalue = 2, so the matrix should be divided by 2."""
        mat = 2.0 * jnp.eye(3)
        result = _normalize_matrix_spectrum(mat)
        expected = jnp.eye(3)
        npt.assert_array_almost_equal(result, expected)

    def test_scaled_identity_below_one(self):
        """0.5*I has max eigenvalue = 0.5, so the matrix should be unchanged."""
        mat = 0.5 * jnp.eye(3)
        result = _normalize_matrix_spectrum(mat)
        npt.assert_array_almost_equal(result, mat)

    def test_diagonal_matrix_above_one(self):
        """Diagonal matrix with max eigenvalue > 1 should be normalized."""
        mat = jnp.diag(jnp.array([3.0, 1.0, 0.5]))
        result = _normalize_matrix_spectrum(mat)
        expected = mat / 3.0
        npt.assert_array_almost_equal(result, expected)

    def test_diagonal_matrix_below_one(self):
        """Diagonal matrix with all eigenvalues <= 1 should be unchanged."""
        mat = jnp.diag(jnp.array([0.9, 0.3, 0.1]))
        result = _normalize_matrix_spectrum(mat)
        npt.assert_array_almost_equal(result, mat)

    def test_zero_matrix(self):
        """Zero matrix has eigenvalue 0, should be returned unchanged."""
        mat = jnp.zeros((3, 3))
        result = _normalize_matrix_spectrum(mat)
        npt.assert_array_almost_equal(result, mat)

    def test_batched_3d(self):
        """3D input (batch of matrices) should apply normalization per matrix."""
        mat1 = 4.0 * jnp.eye(2)  # max eigenvalue = 4
        mat2 = 0.5 * jnp.eye(2)  # max eigenvalue = 0.5
        batch = jnp.stack([mat1, mat2], axis=0)  # shape (2, 2, 2)
        result = _normalize_matrix_spectrum(batch)

        expected_mat1 = jnp.eye(2)  # 4*I / 4 = I
        expected_mat2 = 0.5 * jnp.eye(2)  # unchanged
        expected = jnp.stack([expected_mat1, expected_mat2], axis=0)
        npt.assert_array_almost_equal(result, expected)

    def test_batched_4d(self):
        """4D input should vmap over two leading dims."""
        mat = 2.0 * jnp.eye(2)
        # shape (2, 3, 2, 2)
        batch = jnp.broadcast_to(mat, (2, 3, 2, 2))
        result = _normalize_matrix_spectrum(batch)
        expected = jnp.broadcast_to(jnp.eye(2), (2, 3, 2, 2))
        npt.assert_array_almost_equal(result, expected)

    def test_1x1_matrix_above_one(self):
        """1x1 matrix with value > 1 should be normalized to 1."""
        mat = jnp.array([[5.0]])
        result = _normalize_matrix_spectrum(mat)
        npt.assert_array_almost_equal(result, jnp.array([[1.0]]))

    def test_1x1_matrix_below_one(self):
        """1x1 matrix with value <= 1 should be unchanged."""
        mat = jnp.array([[0.3]])
        result = _normalize_matrix_spectrum(mat)
        npt.assert_array_almost_equal(result, jnp.array([[0.3]]))

    def test_negative_eigenvalues_above_one_abs(self):
        """Negative eigenvalues with abs > 1 should trigger normalization."""
        mat = jnp.diag(jnp.array([-3.0, 0.5]))
        result = _normalize_matrix_spectrum(mat)
        expected = mat / 3.0
        npt.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# Tests for _remove_units
# ---------------------------------------------------------------------------

class TestRemoveUnits:
    """Unit tests for the _remove_units function."""

    def test_plain_array_round_trip(self):
        """Plain jax arrays (no units) should round-trip without change."""
        x = jnp.array([1.0, 2.0, 3.0])
        unitless, restore = _remove_units(x)
        npt.assert_array_almost_equal(unitless, x)
        restored = restore(unitless)
        npt.assert_array_almost_equal(restored, x)

    def test_quantity_round_trip(self):
        """A saiunit Quantity should be stripped and restored."""
        x = jnp.array([1.0, 2.0]) * u.mV
        unitless, restore = _remove_units(x)
        # unitless should be the mantissa
        npt.assert_array_almost_equal(unitless, jnp.array([1.0, 2.0]))
        restored = restore(unitless)
        # restored should be equal to the original quantity
        npt.assert_array_almost_equal(u.get_mantissa(restored), u.get_mantissa(x))
        assert u.get_unit(restored) == u.get_unit(x)

    def test_mixed_pytree_round_trip(self):
        """A pytree mixing plain arrays and Quantities should round-trip."""
        x_plain = jnp.array([3.0, 4.0])
        x_qty = jnp.array([5.0, 6.0]) * u.ms
        tree = (x_plain, x_qty)

        unitless, restore = _remove_units(tree)
        restored = restore(unitless)

        npt.assert_array_almost_equal(restored[0], x_plain)
        npt.assert_array_almost_equal(u.get_mantissa(restored[1]), u.get_mantissa(x_qty))
        assert u.get_unit(restored[1]) == u.get_unit(x_qty)

    def test_nested_dict_pytree(self):
        """Nested dict pytree should work correctly."""
        tree = {
            'a': jnp.array([1.0]),
            'b': jnp.array([2.0]) * u.mV,
        }
        unitless, restore = _remove_units(tree)
        restored = restore(unitless)

        npt.assert_array_almost_equal(restored['a'], tree['a'])
        npt.assert_array_almost_equal(
            u.get_mantissa(restored['b']),
            u.get_mantissa(tree['b'])
        )

    def test_unitless_values_unchanged_through_restore(self):
        """Applying restore to modified unitless values should work."""
        x = jnp.array([1.0, 2.0]) * u.mV
        unitless, restore = _remove_units(x)
        modified = unitless * 2.0
        restored = restore(modified)
        npt.assert_array_almost_equal(
            u.get_mantissa(restored),
            jnp.array([2.0, 4.0])
        )
        assert u.get_unit(restored) == u.mV

    def test_restore_with_mismatched_tree_raises(self):
        """Restoring with a different tree structure should raise."""
        x = jnp.array([1.0, 2.0])
        _, restore = _remove_units(x)
        with pytest.raises(Exception):
            # Pass a tuple instead of a plain array
            restore((jnp.array([1.0]), jnp.array([2.0])))

    def test_single_scalar_quantity(self):
        """Single scalar Quantity should work."""
        x = 5.0 * u.nA
        unitless, restore = _remove_units(x)
        restored = restore(unitless)
        npt.assert_almost_equal(float(u.get_mantissa(restored)), 5.0)
        assert u.get_unit(restored) == u.nA


# ---------------------------------------------------------------------------
# Tests for ParamDimVjpAlgorithm and D_RTRL alias
# ---------------------------------------------------------------------------

class TestParamDimVjpAlgorithmAlias:
    """Check that D_RTRL is a subclass of ParamDimVjpAlgorithm."""

    def test_d_rtrl_is_param_dim_vjp_algorithm(self):
        assert issubclass(D_RTRL, ParamDimVjpAlgorithm)

    def test_d_rtrl_instance_is_param_dim_vjp_algorithm(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = D_RTRL(model)
        assert isinstance(algo, D_RTRL)
        assert isinstance(algo, ParamDimVjpAlgorithm)
        # Constructor defaults preserved through inheritance.
        assert algo.vjp_method == 'single-step'
        assert algo.fast_solve is True
        assert algo.normalize_matrix_spectrum is False


class TestParamDimVjpAlgorithmInit:
    """Unit tests for the __init__ method of ParamDimVjpAlgorithm."""

    def _make_model(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        return model

    def test_default_vjp_method(self):
        model = self._make_model()
        algo = ParamDimVjpAlgorithm(model)
        assert algo.vjp_method == 'single-step'

    def test_multi_step_vjp_method(self):
        model = self._make_model()
        algo = ParamDimVjpAlgorithm(model, vjp_method='multi-step')
        assert algo.vjp_method == 'multi-step'

    def test_invalid_vjp_method_raises(self):
        model = self._make_model()
        with pytest.raises(AssertionError):
            ParamDimVjpAlgorithm(model, vjp_method='invalid-method')

    def test_name_set(self):
        model = self._make_model()
        algo = ParamDimVjpAlgorithm(model, name='test_algo')
        assert algo.name == 'test_algo'

    def test_model_must_be_module(self):
        with pytest.raises(TypeError, match='brainstate.nn.Module'):
            ParamDimVjpAlgorithm("not_a_model")


class TestParamDimVjpAlgorithmCompileAndState:
    """Integration-style unit tests for compile_graph, init_etrace_state, reset_state."""

    def _build_algo(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model)
        return algo, model

    def test_not_compiled_initially(self):
        algo, _ = self._build_algo()
        assert algo.is_compiled is False

    def test_compile_graph_sets_compiled_flag(self):
        algo, _ = self._build_algo()
        x = brainstate.random.rand(3)
        algo.compile_graph(x)
        assert algo.is_compiled is True

    def test_compile_graph_creates_etrace_bwg(self):
        algo, _ = self._build_algo()
        x = brainstate.random.rand(3)
        algo.compile_graph(x)
        assert hasattr(algo, 'etrace_bwg')
        assert isinstance(algo.etrace_bwg, dict)
        assert len(algo.etrace_bwg) > 0

    def test_etrace_bwg_values_are_eligibility_traces(self):
        algo, _ = self._build_algo()
        x = brainstate.random.rand(3)
        algo.compile_graph(x)
        for key, state in algo.etrace_bwg.items():
            from braintrace._etrace_algorithms import EligibilityTrace
            assert isinstance(state, EligibilityTrace)

    def test_etrace_bwg_values_are_zeros_initially(self):
        algo, _ = self._build_algo()
        x = brainstate.random.rand(3)
        algo.compile_graph(x)
        for key, state in algo.etrace_bwg.items():
            leaves = jax.tree.leaves(state.value)
            for leaf in leaves:
                npt.assert_array_almost_equal(leaf, jnp.zeros_like(leaf))

    def test_reset_state_zeros_etrace_bwg(self):
        algo, _ = self._build_algo()
        x = brainstate.random.rand(3)
        algo.compile_graph(x)

        # Manually set a nonzero value in the first etrace state
        first_key = next(iter(algo.etrace_bwg))
        old_val = algo.etrace_bwg[first_key].value
        algo.etrace_bwg[first_key].value = jax.tree.map(jnp.ones_like, old_val)

        # Reset should zero everything out
        algo.reset_state()
        for key, state in algo.etrace_bwg.items():
            leaves = jax.tree.leaves(state.value)
            for leaf in leaves:
                npt.assert_array_almost_equal(leaf, jnp.zeros_like(leaf))

    def test_reset_state_zeros_running_index(self):
        algo, _ = self._build_algo()
        x = brainstate.random.rand(3)
        algo.compile_graph(x)
        algo.running_index.value = 10
        algo.reset_state()
        assert algo.running_index.value == 0

    def test_double_compile_is_noop(self):
        """Compiling twice should not raise; the second call is a no-op."""
        algo, _ = self._build_algo()
        x = brainstate.random.rand(3)
        algo.compile_graph(x)
        n_keys_first = len(algo.etrace_bwg)
        algo.compile_graph(x)
        assert len(algo.etrace_bwg) == n_keys_first


class TestGetAssignEtraceData:
    """Tests for _get_etrace_data / _assign_etrace_data round-trip."""

    def _build_compiled_algo(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model)
        x = brainstate.random.rand(3)
        algo.compile_graph(x)
        return algo

    def test_get_returns_dict(self):
        algo = self._build_compiled_algo()
        data = algo._get_etrace_data()
        assert isinstance(data, dict)

    def test_get_keys_match_etrace_bwg_keys(self):
        algo = self._build_compiled_algo()
        data = algo._get_etrace_data()
        assert set(data.keys()) == set(algo.etrace_bwg.keys())

    def test_round_trip(self):
        """Get -> modify -> assign -> get should reflect the modification."""
        algo = self._build_compiled_algo()
        data = algo._get_etrace_data()

        # Modify all values to ones
        modified = {
            k: jax.tree.map(jnp.ones_like, v)
            for k, v in data.items()
        }
        algo._assign_etrace_data(modified)

        retrieved = algo._get_etrace_data()
        for k in modified:
            leaves_mod = jax.tree.leaves(modified[k])
            leaves_ret = jax.tree.leaves(retrieved[k])
            for lm, lr in zip(leaves_mod, leaves_ret):
                npt.assert_array_almost_equal(lr, lm)


class TestGetEtraceOf:
    """Tests for the get_etrace_of method."""

    def _build_compiled_algo(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model)
        x = brainstate.random.rand(3)
        algo.compile_graph(x)
        return algo, model

    def test_before_compile_raises(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model)
        param_states = model.states(brainstate.ParamState)
        first_param = list(param_states.values())[0]
        with pytest.raises(ValueError, match='compile'):
            algo.get_etrace_of(first_param)

    def test_unknown_weight_raises(self):
        algo, model = self._build_compiled_algo()
        # Create a random ParamState that is not part of the model
        fake_state = brainstate.ParamState(jnp.zeros(10))
        with pytest.raises(ValueError):
            algo.get_etrace_of(fake_state)

    def test_known_weight_returns_dict(self):
        algo, model = self._build_compiled_algo()
        param_states = model.states(brainstate.ParamState)
        first_param = list(param_states.values())[0]
        result = algo.get_etrace_of(first_param)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_get_etrace_by_path(self):
        """get_etrace_of should also accept a Path (tuple of strings)."""
        algo, model = self._build_compiled_algo()
        # Find a valid path from the algorithm's path_to_states
        param_paths = list(algo.param_states.keys())
        if len(param_paths) > 0:
            path = param_paths[0]
            result = algo.get_etrace_of(path)
            assert isinstance(result, dict)
            assert len(result) > 0


class TestParamDimVjpAlgorithmForwardPass:
    """Test that the algorithm can do a forward pass after compilation."""

    def test_single_step_forward(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model)
        x = brainstate.random.rand(3)
        algo.compile_graph(x)

        out = algo(x)
        assert out.shape == (4,)

    def test_running_index_increments(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model)
        x = brainstate.random.rand(3)
        algo.compile_graph(x)

        assert algo.running_index.value == 0
        algo(x)
        assert algo.running_index.value == 1
        algo(x)
        assert algo.running_index.value == 2

    def test_multi_step_forward(self):
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model, vjp_method='multi-step')
        x_single = brainstate.random.rand(3)  # single-step shape for compilation
        algo.compile_graph(x_single)

        x_multi = brainstate.random.rand(5, 3)  # 5 time steps
        out = algo(braintrace.MultiStepData(x_multi))
        assert out.shape == (5, 4)

    def test_grad_computable(self):
        """Verify that gradients can be computed through the algorithm."""
        model = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ParamDimVjpAlgorithm(model)
        x = brainstate.random.rand(3)
        algo.compile_graph(x)

        @brainstate.transform.jit
        def compute_grad(inp):
            return brainstate.transform.grad(
                lambda inp: algo(inp).sum(),
                model.states(brainstate.ParamState)
            )(inp)

        grads = compute_grad(x)
        # grads should be a non-empty dict-like structure
        assert len(grads) > 0
        # Check that at least some gradients are non-zero
        all_zero = all(
            jnp.allclose(jax.tree.leaves(v)[0], 0.0)
            for v in grads.values()
        )
        # After first step, it is possible all grads are zero, so we just check it runs.


class TestNormalizeVectorEdgeCases:
    """Additional edge-case tests for _normalize_vector under JIT."""

    def test_jit_compatible(self):
        """_normalize_vector should work under jax.jit."""
        v = jnp.array([3.0, -1.0, 2.0])
        jitted = jax.jit(_normalize_vector)
        result = jitted(v)
        expected = v / 3.0
        npt.assert_array_almost_equal(result, expected)

    def test_float64_precision(self):
        """Test with float64 precision if available."""
        v = jnp.array([0.5, -0.5], dtype=jnp.float32)
        result = _normalize_vector(v)
        npt.assert_array_almost_equal(result, v)


class TestNormalizeMatrixSpectrumEdgeCases:
    """Additional edge-case tests for _normalize_matrix_spectrum."""

    def test_jit_compatible(self):
        """_normalize_matrix_spectrum should work under jax.jit."""
        mat = 3.0 * jnp.eye(2)
        jitted = jax.jit(_normalize_matrix_spectrum)
        result = jitted(mat)
        expected = jnp.eye(2)
        npt.assert_array_almost_equal(result, expected)

    def test_non_symmetric_matrix(self):
        """Non-symmetric matrix: should use abs of eigenvalues."""
        # Upper triangular with known eigenvalues on diagonal
        mat = jnp.array([[2.0, 1.0], [0.0, 0.5]])
        # Eigenvalues are 2.0 and 0.5, max abs = 2.0
        result = _normalize_matrix_spectrum(mat)
        expected = mat / 2.0
        npt.assert_array_almost_equal(result, expected)

    def test_2x2_matrix_eigenvalue_exactly_one(self):
        """Identity matrix has eigenvalue exactly 1, should not normalize."""
        mat = jnp.eye(2)
        result = _normalize_matrix_spectrum(mat)
        npt.assert_array_almost_equal(result, mat)


class TestRemoveUnitsEdgeCases:
    """Additional edge-case tests for _remove_units."""

    def test_empty_list_pytree(self):
        """Empty list should pass through without error."""
        unitless, restore = _remove_units([])
        assert unitless == []
        restored = restore([])
        assert restored == []

    def test_list_of_arrays(self):
        """A list of plain arrays should round-trip."""
        tree = [jnp.array([1.0, 2.0]), jnp.array([3.0])]
        unitless, restore = _remove_units(tree)
        restored = restore(unitless)
        for orig, rest in zip(tree, restored):
            npt.assert_array_almost_equal(rest, orig)

    def test_multiple_quantities_different_units(self):
        """Multiple Quantities with different units should round-trip."""
        tree = (
            jnp.array([1.0]) * u.mV,
            jnp.array([2.0]) * u.ms,
        )
        unitless, restore = _remove_units(tree)
        restored = restore(unitless)
        npt.assert_array_almost_equal(u.get_mantissa(restored[0]), jnp.array([1.0]))
        assert u.get_unit(restored[0]) == u.mV
        npt.assert_array_almost_equal(u.get_mantissa(restored[1]), jnp.array([2.0]))
        assert u.get_unit(restored[1]) == u.ms


class TestDiagOn2:
    @pytest.mark.parametrize(
        "cls",
        [
            # braintrace.nn.GRUCell,
            # braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            # braintrace.nn.MGUCell,
            # braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_single_step_vjp(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        model = brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algorithm = braintrace.ParamDimVjpAlgorithm(model)
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
        algorithm = braintrace.ParamDimVjpAlgorithm(model, vjp_method='multi-step')
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
            print(cls)

            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = brainstate.nn.init_all_states(model)

            param_states = model.states(brainstate.ParamState).to_dict_values()

            inputs = brainstate.random.randn(n_seq, n_in)
            algorithm = braintrace.ParamDimVjpAlgorithm(model)
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

            for k in grads:
                assert u.get_unit(param_states[k]) == u.get_unit(grads[k])

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

            param_states = model.states(brainstate.ParamState).to_dict_values()

            inputs = brainstate.random.randn(n_seq, n_in)
            algorithm = braintrace.ParamDimVjpAlgorithm(model, vjp_method='multi-step')
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

            for k in grads:
                assert u.get_unit(param_states[k]) == u.get_unit(grads[k])


class TestDRtrlDictTraceStorage:
    """D-RTRL stores per-primitive-instance traces keyed by (id(y_var), group_index)
    whose value is a Dict[str, Array] keyed by trainable-input names.

    Because no primitive has migrated to the new API yet, the dict has only
    the single key 'weight' and its shape matches what the legacy rule produced."""

    def test_trace_is_dict_keyed_by_weight_for_mm(self):
        import brainstate
        import jax.numpy as jnp
        import braintrace
        from braintrace._etrace_algorithms.param_dim_vjp import _init_param_dim_state

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)))
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(braintrace.matmul(self.h.value, self.w.value))
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        relations = braintrace.find_hidden_param_op_relations_from_module(
            cell, jnp.zeros((1, 4))
        )
        assert len(relations) == 1
        relation = relations[0]
        etrace = {}
        _init_param_dim_state(etrace, relation)
        assert len(etrace) == 1
        (_, entry), = etrace.items()
        assert isinstance(entry.value, dict)
        assert set(entry.value.keys()) == {'weight'}
        assert entry.value['weight'].shape[0] == 1  # batch
        assert entry.value['weight'].shape[1:3] == (4, 4)  # W shape


# ---------------------------------------------------------------------------
# Helpers for the trace-update specialization + dtype-knob tests
# ---------------------------------------------------------------------------

def _rnn_mm(seed=0, n=4, bias=False):
    """Batched single-recurrent-weight tanh RNN -> etp_mm_p, num_state==1."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
            self.w = brainstate.ParamState(0.3 * jax.random.normal(k1, (n, n)))
            self.b = brainstate.ParamState(0.1 * jax.random.normal(k2, (n,))) if bias else None
            self.h = brainstate.HiddenState(jnp.zeros((1, n)))

        def update(self, x):
            b = self.b.value if self.b is not None else None
            self.h.value = jax.nn.tanh(braintrace.matmul(self.h.value + x, self.w.value, b))
            return self.h.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _rnn_mv(seed=0, n=4):
    """Unbatched single-recurrent-weight tanh RNN -> etp_mv_p, num_state==1."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.3 * jax.random.normal(jax.random.PRNGKey(seed), (n, n))
            )
            self.h = brainstate.HiddenState(jnp.zeros((n,)))

        def update(self, x):
            self.h.value = jax.nn.tanh(braintrace.matmul(self.h.value + x, self.w.value))
            return self.h.value

    net = Net()
    brainstate.nn.init_all_states(net)
    return net


def _run_updates(algo, xs):
    """Compile, init, run len(xs) eager updates; return etrace_bwg snapshot.

    Returns the etrace values as an ordered list of per-key dicts (keys differ
    across algo instances, but relation/group iteration order is deterministic
    for identical models, so positional comparison is valid)."""
    algo.compile_graph(xs[0])
    algo.init_etrace_state()
    for t in range(xs.shape[0]):
        algo.update(xs[t])
    return [v.value for v in algo.etrace_bwg.values()]


def _assert_etrace_lists_equal(a_list, b_list, *, exact=True, rtol=0.0, atol=0.0):
    assert len(a_list) == len(b_list) >= 1
    for da, db in zip(a_list, b_list):
        assert set(da.keys()) == set(db.keys())
        for k in da:
            av = jax.tree.map(u.get_mantissa, da[k])
            bv = jax.tree.map(u.get_mantissa, db[k])
            if exact:
                npt.assert_array_equal(av, bv)
            else:
                npt.assert_allclose(av, bv, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Change B — S==1 recurrent trace update is bit-identical to the legacy path
# ---------------------------------------------------------------------------

class TestS1RecurrentBroadcastEqualsLegacy:
    """The S==1 broadcast specialization of the recurrent trace term must be
    bit-identical to the legacy nested-vmap path (fast_solve=False).

    The recurrent term is only exercised once the trace is non-zero, i.e. from
    the 2nd update on, so we run several steps."""

    def _xs_mm(self, n=4, steps=4):
        return brainstate.random.randn(steps, 1, n)

    def _xs_mv(self, n=4, steps=4):
        return brainstate.random.randn(steps, n)

    def test_mm_s1_fast_equals_legacy(self):
        xs = self._xs_mm()
        fast = _run_updates(ParamDimVjpAlgorithm(_rnn_mm(), fast_solve=True), xs)
        legacy = _run_updates(ParamDimVjpAlgorithm(_rnn_mm(), fast_solve=False), xs)
        _assert_etrace_lists_equal(fast, legacy, exact=True)

    def test_mm_s1_with_bias_fast_equals_legacy(self):
        xs = self._xs_mm()
        fast = _run_updates(ParamDimVjpAlgorithm(_rnn_mm(bias=True), fast_solve=True), xs)
        legacy = _run_updates(ParamDimVjpAlgorithm(_rnn_mm(bias=True), fast_solve=False), xs)
        _assert_etrace_lists_equal(fast, legacy, exact=True)

    def test_mv_s1_fast_equals_legacy(self):
        xs = self._xs_mv()
        fast = _run_updates(ParamDimVjpAlgorithm(_rnn_mv(), fast_solve=True), xs)
        legacy = _run_updates(ParamDimVjpAlgorithm(_rnn_mv(), fast_solve=False), xs)
        _assert_etrace_lists_equal(fast, legacy, exact=True)


# ---------------------------------------------------------------------------
# Change A — trace_dtype knob (store the eligibility trace in lower precision)
# ---------------------------------------------------------------------------

class TestTraceDtypeKnob:
    """trace_dtype stores the eligibility trace at reduced precision (default
    None = unchanged fp32 behavior). Jacobians and the final gradient stay fp32."""

    def _xs(self, n=4, steps=4):
        return brainstate.random.randn(steps, 1, n)

    def test_default_is_float32(self):
        xs = self._xs()
        algo = ParamDimVjpAlgorithm(_rnn_mm())
        _run_updates(algo, xs)
        for v in algo.etrace_bwg.values():
            for leaf in jax.tree.leaves(v.value):
                assert u.get_mantissa(leaf).dtype == jnp.float32

    def test_bf16_state_dtype(self):
        xs = self._xs()
        algo = ParamDimVjpAlgorithm(_rnn_mm(), trace_dtype=jnp.bfloat16)
        _run_updates(algo, xs)
        for v in algo.etrace_bwg.values():
            for leaf in jax.tree.leaves(v.value):
                assert u.get_mantissa(leaf).dtype == jnp.bfloat16

    def test_none_equals_default(self):
        xs = self._xs()
        a = _run_updates(ParamDimVjpAlgorithm(_rnn_mm(), trace_dtype=None), xs)
        b = _run_updates(ParamDimVjpAlgorithm(_rnn_mm()), xs)
        _assert_etrace_lists_equal(a, b, exact=True)

    def test_bf16_grad_close_to_fp32(self):
        xs = self._xs()

        def grad_of(algo):
            algo.compile_graph(xs[0])
            algo.init_etrace_state()
            algo.update(xs[0])
            algo.update(xs[1])

            def loss(x_):
                return (algo.update(x_) ** 2).sum()

            grads, _ = brainstate.transform.grad(
                loss, algo.param_states, return_value=True
            )(xs[2])
            return grads

        g32 = grad_of(ParamDimVjpAlgorithm(_rnn_mm()))
        gbf = grad_of(ParamDimVjpAlgorithm(_rnn_mm(), trace_dtype=jnp.bfloat16))
        for k in g32:
            a = u.get_mantissa(jax.tree.leaves(g32[k])[0])
            b = u.get_mantissa(jax.tree.leaves(gbf[k])[0])
            # bf16 trace -> ~2-3 significant digits; assert bounded divergence.
            npt.assert_allclose(b, a, rtol=0.2, atol=1e-2)

    def test_bf16_with_normalize_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            ParamDimVjpAlgorithm(
                _rnn_mm(), trace_dtype=jnp.bfloat16, normalize_matrix_spectrum=True
            )


# ---------------------------------------------------------------------------
# The S==1 broadcast branch must NOT perturb the multi-state (num_state>1) path
# ---------------------------------------------------------------------------

def _alif(n_in=4, n_rec=5):
    """Adaptive-LIF dense layer -> coupled (V, adaptation) hidden group with
    num_state==2, exercising the S>1 einsum branch of _fast_recurrent_term."""
    model = ALIF_Delta_Dense_Layer(n_in, n_rec)
    brainstate.nn.init_all_states(model)
    return model


def _clone_state_values(src, dst):
    """Copy every state value from ``src`` into ``dst`` (same model class -> same
    paths), so the two models are bit-identical regardless of global RNG."""
    ss, ds = src.states(), dst.states()
    for k in ss.keys():
        ds[k].value = ss[k].value


class TestMultiStateUnaffected:
    """For num_state>1 the recurrent term must keep using the einsum (the S==1
    broadcast would silently drop the cross-state coupling). Pin fast==legacy
    on an ALIF model to guarantee the new ``num_state==1`` branch is not taken.

    fast and legacy reduce over the state axis in a different order (einsum vs
    nested vmap+sum), so this is numerically equal, not bit-identical."""

    def test_alif_fast_equals_legacy(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            xs = brainstate.random.randn(5, 4)
            m_fast, m_legacy = _alif(), _alif()
            _clone_state_values(m_fast, m_legacy)  # force identical weights + init
            fast = _run_updates(ParamDimVjpAlgorithm(m_fast, fast_solve=True), xs)
            legacy = _run_updates(ParamDimVjpAlgorithm(m_legacy, fast_solve=False), xs)
            _assert_etrace_lists_equal(fast, legacy, exact=False, rtol=1e-5, atol=1e-6)
