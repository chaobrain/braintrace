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
import numpy.testing as npt
import pytest

from braintrace._etrace_vjp.misc import (
    _batched_zeros_like,
    _reset_state_in_a_dict,
    _sum_dim,
    _update_dict,
    _zeros_like_batch_or_not,
)


# ---------------------------------------------------------------------------
# _zeros_like_batch_or_not
# ---------------------------------------------------------------------------

class TestZerosLikeBatchOrNot:

    def test_no_batch_preserves_shape(self):
        x = jnp.ones((3, 4), dtype=jnp.float32)
        result = _zeros_like_batch_or_not(None, x)
        assert result.shape == (3, 4)

    def test_no_batch_preserves_dtype(self):
        x = jnp.ones((2, 5), dtype=jnp.float32)
        result = _zeros_like_batch_or_not(None, x)
        assert result.dtype == jnp.float32

    def test_no_batch_all_zeros(self):
        x = jnp.ones((3, 4), dtype=jnp.float32)
        result = _zeros_like_batch_or_not(None, x)
        npt.assert_array_equal(result, jnp.zeros((3, 4), dtype=jnp.float32))

    def test_with_batch_replaces_first_dim(self):
        x = jnp.ones((10, 4), dtype=jnp.float32)
        result = _zeros_like_batch_or_not(8, x)
        assert result.shape == (8, 4)

    def test_with_batch_preserves_dtype(self):
        x = jnp.ones((2, 3), dtype=jnp.float32)
        result = _zeros_like_batch_or_not(5, x)
        assert result.dtype == jnp.float32

    def test_with_batch_all_zeros(self):
        x = jnp.ones((2, 3), dtype=jnp.float32)
        result = _zeros_like_batch_or_not(4, x)
        npt.assert_array_equal(result, jnp.zeros((4, 3), dtype=jnp.float32))

    def test_with_batch_higher_rank(self):
        x = jnp.ones((2, 3, 5, 7), dtype=jnp.float32)
        result = _zeros_like_batch_or_not(6, x)
        assert result.shape == (6, 3, 5, 7)

    def test_no_batch_scalar_like(self):
        x = jnp.array(1.0)
        result = _zeros_like_batch_or_not(None, x)
        assert result.shape == ()
        npt.assert_array_equal(result, jnp.array(0.0))

    def test_with_batch_1d_input(self):
        x = jnp.ones((5,), dtype=jnp.float32)
        result = _zeros_like_batch_or_not(3, x)
        # shape[1:] for 1D is () so result is (3,)
        assert result.shape == (3,)

    def test_batch_size_must_be_int(self):
        x = jnp.ones((2, 3))
        with pytest.raises(AssertionError, match="batch size should be an integer"):
            _zeros_like_batch_or_not(2.5, x)

    def test_no_batch_2d_int_dtype(self):
        x = jnp.ones((4, 4), dtype=jnp.int32)
        result = _zeros_like_batch_or_not(None, x)
        assert result.dtype == jnp.int32
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# _batched_zeros_like
# ---------------------------------------------------------------------------

class TestBatchedZerosLike:

    def test_no_batch_appends_num_state(self):
        x = jnp.ones((3, 4), dtype=jnp.float32)
        result = _batched_zeros_like(None, 7, x)
        assert result.shape == (3, 4, 7)

    def test_no_batch_all_zeros(self):
        x = jnp.ones((2, 3), dtype=jnp.float32)
        result = _batched_zeros_like(None, 5, x)
        npt.assert_array_equal(result, jnp.zeros((2, 3, 5), dtype=jnp.float32))

    def test_no_batch_preserves_dtype(self):
        x = jnp.ones((2,), dtype=jnp.float32)
        result = _batched_zeros_like(None, 3, x)
        assert result.dtype == jnp.float32

    def test_with_batch_prepends_batch_dim(self):
        x = jnp.ones((3, 4), dtype=jnp.float32)
        result = _batched_zeros_like(8, 5, x)
        assert result.shape == (8, 3, 4, 5)

    def test_with_batch_all_zeros(self):
        x = jnp.ones((2,), dtype=jnp.float32)
        result = _batched_zeros_like(4, 3, x)
        npt.assert_array_equal(result, jnp.zeros((4, 2, 3), dtype=jnp.float32))

    def test_with_batch_preserves_dtype(self):
        x = jnp.ones((5,), dtype=jnp.float32)
        result = _batched_zeros_like(2, 10, x)
        assert result.dtype == jnp.float32

    def test_num_state_one(self):
        x = jnp.ones((3,))
        result = _batched_zeros_like(None, 1, x)
        assert result.shape == (3, 1)

    def test_no_batch_scalar_input(self):
        x = jnp.array(1.0)
        result = _batched_zeros_like(None, 4, x)
        assert result.shape == (4,)

    def test_with_batch_scalar_input(self):
        x = jnp.array(1.0)
        result = _batched_zeros_like(2, 4, x)
        assert result.shape == (2, 4)

    def test_high_rank_no_batch(self):
        x = jnp.ones((2, 3, 4))
        result = _batched_zeros_like(None, 6, x)
        assert result.shape == (2, 3, 4, 6)

    def test_high_rank_with_batch(self):
        x = jnp.ones((2, 3, 4))
        result = _batched_zeros_like(5, 6, x)
        assert result.shape == (5, 2, 3, 4, 6)


# ---------------------------------------------------------------------------
# _reset_state_in_a_dict
# ---------------------------------------------------------------------------

class TestResetStateInADict:

    def test_empty_dict(self):
        state_dict = {}
        _reset_state_in_a_dict(state_dict, None)
        assert state_dict == {}

    def test_single_state_no_batch(self):
        s = brainstate.State(jnp.ones((3, 4)))
        state_dict = {"a": s}
        _reset_state_in_a_dict(state_dict, None)
        npt.assert_array_equal(s.value, jnp.zeros((3, 4)))

    def test_single_state_with_batch(self):
        s = brainstate.State(jnp.ones((2, 4)))
        state_dict = {"a": s}
        _reset_state_in_a_dict(state_dict, 8)
        assert s.value.shape == (8, 4)
        npt.assert_array_equal(s.value, jnp.zeros((8, 4)))

    def test_multiple_states_no_batch(self):
        s1 = brainstate.State(jnp.ones((3,)))
        s2 = brainstate.State(jnp.ones((5, 2)))
        state_dict = {"x": s1, "y": s2}
        _reset_state_in_a_dict(state_dict, None)
        npt.assert_array_equal(s1.value, jnp.zeros((3,)))
        npt.assert_array_equal(s2.value, jnp.zeros((5, 2)))

    def test_multiple_states_with_batch(self):
        s1 = brainstate.State(jnp.ones((4, 3)))
        s2 = brainstate.State(jnp.ones((4, 5)))
        state_dict = {"a": s1, "b": s2}
        _reset_state_in_a_dict(state_dict, 6)
        assert s1.value.shape == (6, 3)
        assert s2.value.shape == (6, 5)
        npt.assert_array_equal(s1.value, jnp.zeros((6, 3)))
        npt.assert_array_equal(s2.value, jnp.zeros((6, 5)))

    def test_preserves_dtype(self):
        s = brainstate.State(jnp.ones((2, 3), dtype=jnp.float32))
        state_dict = {"s": s}
        _reset_state_in_a_dict(state_dict, None)
        assert s.value.dtype == jnp.float32

    def test_dict_keys_unchanged(self):
        s1 = brainstate.State(jnp.ones((2,)))
        s2 = brainstate.State(jnp.ones((3,)))
        state_dict = {"alpha": s1, "beta": s2}
        _reset_state_in_a_dict(state_dict, None)
        assert set(state_dict.keys()) == {"alpha", "beta"}

    def test_modifies_in_place(self):
        s = brainstate.State(jnp.ones((2,)))
        state_dict = {"k": s}
        _reset_state_in_a_dict(state_dict, None)
        # The same State object is still in the dict
        assert state_dict["k"] is s


# ---------------------------------------------------------------------------
# _sum_dim
# ---------------------------------------------------------------------------

class TestSumDim:

    def test_single_array_default_axis(self):
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = _sum_dim(x)
        expected = jnp.array([6.0, 15.0])
        npt.assert_array_almost_equal(result, expected)

    def test_single_array_axis_zero(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = _sum_dim(x, axis=0)
        expected = jnp.array([4.0, 6.0])
        npt.assert_array_almost_equal(result, expected)

    def test_nested_pytree_list(self):
        xs = [
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            jnp.array([[10.0, 20.0]]),
        ]
        result = _sum_dim(xs)
        npt.assert_array_almost_equal(result[0], jnp.array([3.0, 7.0]))
        npt.assert_array_almost_equal(result[1], jnp.array([30.0]))

    def test_nested_pytree_dict(self):
        xs = {
            "a": jnp.array([[1.0, 2.0, 3.0]]),
            "b": jnp.array([[4.0, 5.0]]),
        }
        result = _sum_dim(xs)
        npt.assert_array_almost_equal(result["a"], jnp.array([6.0]))
        npt.assert_array_almost_equal(result["b"], jnp.array([9.0]))

    def test_1d_array_default_axis(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = _sum_dim(x)
        # sum over axis=-1 of 1D gives a scalar
        npt.assert_array_almost_equal(result, jnp.array(6.0))

    def test_3d_array_last_axis(self):
        x = jnp.ones((2, 3, 4))
        result = _sum_dim(x)
        assert result.shape == (2, 3)
        npt.assert_array_almost_equal(result, jnp.full((2, 3), 4.0))

    def test_custom_axis_middle(self):
        x = jnp.ones((2, 3, 4))
        result = _sum_dim(x, axis=1)
        assert result.shape == (2, 4)
        npt.assert_array_almost_equal(result, jnp.full((2, 4), 3.0))

    def test_preserves_pytree_structure(self):
        xs = {"key1": jnp.ones((2, 3)), "key2": jnp.ones((4, 5))}
        result = _sum_dim(xs)
        assert set(result.keys()) == {"key1", "key2"}
        assert result["key1"].shape == (2,)
        assert result["key2"].shape == (4,)

    def test_empty_pytree(self):
        xs = {}
        result = _sum_dim(xs)
        assert result == {}


# ---------------------------------------------------------------------------
# _update_dict
# ---------------------------------------------------------------------------

class TestUpdateDict:

    def test_new_key_inserted(self):
        d = {}
        val = jnp.array([1.0, 2.0])
        _update_dict(d, "a", val)
        assert "a" in d
        npt.assert_array_equal(d["a"], val)

    def test_existing_key_accumulated(self):
        d = {"a": jnp.array([1.0, 2.0])}
        _update_dict(d, "a", jnp.array([3.0, 4.0]))
        npt.assert_array_almost_equal(d["a"], jnp.array([4.0, 6.0]))

    def test_accumulate_multiple_times(self):
        d = {}
        _update_dict(d, "k", jnp.array([1.0]))
        _update_dict(d, "k", jnp.array([2.0]))
        _update_dict(d, "k", jnp.array([3.0]))
        npt.assert_array_almost_equal(d["k"], jnp.array([6.0]))

    def test_error_when_no_key_true_raises(self):
        d = {}
        with pytest.raises(ValueError, match="does not exist"):
            _update_dict(d, "missing", jnp.array([1.0]), error_when_no_key=True)

    def test_error_when_no_key_false_inserts(self):
        d = {}
        _update_dict(d, "new", jnp.array([5.0]), error_when_no_key=False)
        npt.assert_array_equal(d["new"], jnp.array([5.0]))

    def test_error_when_no_key_default_is_false(self):
        d = {}
        _update_dict(d, "k", jnp.array([1.0]))
        assert "k" in d

    def test_existing_key_with_error_flag_still_accumulates(self):
        d = {"x": jnp.array([10.0])}
        _update_dict(d, "x", jnp.array([5.0]), error_when_no_key=True)
        npt.assert_array_almost_equal(d["x"], jnp.array([15.0]))

    def test_pytree_value_accumulation(self):
        d = {"k": {"inner": jnp.array([1.0, 2.0])}}
        _update_dict(d, "k", {"inner": jnp.array([3.0, 4.0])})
        npt.assert_array_almost_equal(d["k"]["inner"], jnp.array([4.0, 6.0]))

    def test_multiple_keys_independent(self):
        d = {}
        _update_dict(d, "a", jnp.array([1.0]))
        _update_dict(d, "b", jnp.array([2.0]))
        _update_dict(d, "a", jnp.array([3.0]))
        npt.assert_array_almost_equal(d["a"], jnp.array([4.0]))
        npt.assert_array_almost_equal(d["b"], jnp.array([2.0]))

    def test_quantity_value_accumulation(self):
        q1 = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
        q2 = u.Quantity(jnp.array([3.0, 4.0]), unit=u.mV)
        d = {"q": q1}
        _update_dict(d, "q", q2)
        expected = jnp.array([4.0, 6.0])
        npt.assert_array_almost_equal(d["q"].mantissa, expected)

    def test_new_key_with_none_value_in_dict(self):
        # If the dict has a key mapped to None, it behaves as if key is absent
        d = {"k": None}
        _update_dict(d, "k", jnp.array([5.0]))
        npt.assert_array_equal(d["k"], jnp.array([5.0]))

    def test_integer_key(self):
        d = {}
        _update_dict(d, 42, jnp.array([7.0]))
        assert 42 in d
        npt.assert_array_equal(d[42], jnp.array([7.0]))

    def test_tuple_key(self):
        d = {}
        key = ("layer", "weight")
        _update_dict(d, key, jnp.array([1.0]))
        assert key in d
        npt.assert_array_equal(d[key], jnp.array([1.0]))
