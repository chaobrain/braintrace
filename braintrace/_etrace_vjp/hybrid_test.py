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

from unittest.mock import MagicMock

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._etrace_concepts import ETraceParam, ElemWiseParam, ETraceGrad
from braintrace._etrace_vjp.hybrid import (
    _numel,
    _is_weight_need_full_grad,
    HybridDimVjpAlgorithm,
)


# ---------------------------------------------------------------------------
# Helper: a lightweight wrapper that behaves like a JAX Var with .aval.shape
# but is also traversable by jax.tree as a pytree (so _numel works on it).
# ---------------------------------------------------------------------------

class _VarLike:
    """Wraps a JAX array so that it has .aval.shape AND is a valid pytree leaf."""

    def __init__(self, arr):
        self._arr = arr
        self.aval = MagicMock()
        self.aval.shape = arr.shape


# Register _VarLike once so jax.tree.leaves can unwrap the inner array.
jax.tree_util.register_pytree_node(
    _VarLike,
    lambda v: ([v._arr], None),
    lambda _, children: _VarLike(children[0]),
)


# ---------------------------------------------------------------------------
# Tests for _numel
# ---------------------------------------------------------------------------

class TestNumel:
    def test_single_1d_array(self):
        x = jnp.ones((5,))
        assert _numel(x) == 5

    def test_single_2d_array(self):
        x = jnp.ones((3, 4))
        assert _numel(x) == 12

    def test_single_3d_array(self):
        x = jnp.ones((2, 3, 4))
        assert _numel(x) == 24

    def test_scalar_array(self):
        x = jnp.array(1.0)
        assert _numel(x) == 1

    def test_nested_dict(self):
        pytree = {
            'a': jnp.ones((3, 4)),
            'b': jnp.ones((2,)),
        }
        assert _numel(pytree) == 14

    def test_nested_list(self):
        pytree = [jnp.ones((5,)), jnp.ones((3, 2))]
        assert _numel(pytree) == 11

    def test_deeply_nested(self):
        pytree = {
            'layer1': {
                'w': jnp.ones((4, 3)),
                'b': jnp.ones((3,)),
            },
            'layer2': [jnp.ones((2,))],
        }
        assert _numel(pytree) == 17

    def test_empty_pytree_dict(self):
        assert _numel({}) == 0

    def test_empty_pytree_list(self):
        assert _numel([]) == 0

    def test_single_element(self):
        x = jnp.array(42.0)
        assert _numel(x) == 1


# ---------------------------------------------------------------------------
# Tests for _is_weight_need_full_grad
# ---------------------------------------------------------------------------

class TestIsWeightNeedFullGrad:
    """Unit tests for _is_weight_need_full_grad using mocked relation objects."""

    # --- ETraceParam with gradient == ETraceGrad.full ---

    def test_etrace_param_full_grad_returns_true(self):
        weight = MagicMock(spec=ETraceParam)
        weight.gradient = ETraceGrad.full
        relation = MagicMock()
        relation.weight = weight
        mode = brainstate.mixin.Mode()
        assert _is_weight_need_full_grad(relation, mode) is True

    # --- ETraceParam with gradient == ETraceGrad.approx ---

    def test_etrace_param_approx_grad_returns_false(self):
        weight = MagicMock(spec=ETraceParam)
        weight.gradient = ETraceGrad.approx
        relation = MagicMock()
        relation.weight = weight
        mode = brainstate.mixin.Mode()
        assert _is_weight_need_full_grad(relation, mode) is False

    # --- ElemWiseParam always returns True ---

    def test_elemwise_param_returns_true_via_full_grad(self):
        weight = MagicMock(spec=ElemWiseParam)
        weight.gradient = ETraceGrad.full
        relation = MagicMock()
        relation.weight = weight
        mode = brainstate.mixin.Mode()
        # ElemWiseParam IS-A ETraceParam; gradient=full hits first check -> True
        assert _is_weight_need_full_grad(relation, mode) is True

    def test_elemwise_param_with_adaptive_gradient(self):
        """ElemWiseParam with adaptive gradient returns True via isinstance check."""
        weight = MagicMock(spec=ElemWiseParam)
        weight.gradient = ETraceGrad.adaptive
        relation = MagicMock()
        relation.weight = weight
        mode = brainstate.mixin.Mode()
        # Skips full/approx checks, then isinstance(weight, ElemWiseParam) -> True
        assert _is_weight_need_full_grad(relation, mode) is True

    # --- Adaptive gradient falls through to numel comparison ---

    def test_adaptive_grad_large_io_returns_true(self):
        """When numel(x) + numel(y) > batch_size * numel(weight), return True."""
        weight = MagicMock(spec=ETraceParam)
        weight.gradient = ETraceGrad.adaptive
        weight.value = jnp.ones((2,))
        relation = MagicMock()
        relation.weight = weight
        relation.x = jnp.zeros((10,))
        relation.y = jnp.zeros((10,))
        mode = brainstate.mixin.Mode()
        # numel(x)=10 + numel(y)=10 = 20 > 1 * 2 = 2 -> True
        assert _is_weight_need_full_grad(relation, mode) is True

    def test_adaptive_grad_small_io_returns_false(self):
        """When numel(x) + numel(y) <= batch_size * numel(weight), return False."""
        weight = MagicMock(spec=ETraceParam)
        weight.gradient = ETraceGrad.adaptive
        weight.value = jnp.ones((100, 100))
        relation = MagicMock()
        relation.weight = weight
        relation.x = jnp.zeros((5,))
        relation.y = jnp.zeros((5,))
        mode = brainstate.mixin.Mode()
        # 5+5=10 <= 1*10000 -> False
        assert _is_weight_need_full_grad(relation, mode) is False

    def test_adaptive_grad_with_batching_mode(self):
        """With Batching mode, batch_size comes from x.aval.shape[0]."""
        weight = MagicMock(spec=ETraceParam)
        weight.gradient = ETraceGrad.adaptive
        weight.value = jnp.ones((3,))
        relation = MagicMock()
        relation.weight = weight
        relation.x = _VarLike(jnp.zeros((8, 3)))
        relation.y = _VarLike(jnp.zeros((8, 3)))
        mode = brainstate.mixin.Batching()
        # numel(x)=24, numel(y)=24, total=48
        # batch_size=8, numel(weight)=3, product=24
        # 48 > 24 -> True
        assert _is_weight_need_full_grad(relation, mode) is True

    def test_adaptive_grad_with_batching_large_weight(self):
        """With Batching, large weight makes the comparison False."""
        weight = MagicMock(spec=ETraceParam)
        weight.gradient = ETraceGrad.adaptive
        weight.value = jnp.ones((50, 50))
        relation = MagicMock()
        relation.weight = weight
        relation.x = _VarLike(jnp.zeros((4, 5)))
        relation.y = _VarLike(jnp.zeros((4, 5)))
        mode = brainstate.mixin.Batching()
        # numel(x)=20, numel(y)=20, total=40
        # batch_size=4, numel(weight)=2500, product=10000
        # 40 <= 10000 -> False
        assert _is_weight_need_full_grad(relation, mode) is False

    def test_adaptive_grad_boundary_equal(self):
        """When numel(x) + numel(y) == batch_size * numel(weight), return False (not >)."""
        weight = MagicMock(spec=ETraceParam)
        weight.gradient = ETraceGrad.adaptive
        weight.value = jnp.ones((10,))
        relation = MagicMock()
        relation.weight = weight
        relation.x = jnp.zeros((5,))
        relation.y = jnp.zeros((5,))
        mode = brainstate.mixin.Mode()
        # 5+5=10 == 1*10 -> not > -> False
        assert _is_weight_need_full_grad(relation, mode) is False

    # --- Non-ETraceParam, non-ElemWiseParam weight (generic weight) ---

    def test_generic_weight_falls_through_to_numel(self):
        """A weight that is neither ETraceParam nor ElemWiseParam goes to numel comparison."""
        weight = MagicMock()
        weight.value = jnp.ones((2,))
        relation = MagicMock()
        relation.weight = weight
        relation.x = jnp.zeros((10,))
        relation.y = jnp.zeros((10,))
        mode = brainstate.mixin.Mode()
        # 10+10=20 > 1*2 -> True
        assert _is_weight_need_full_grad(relation, mode) is True


# ---------------------------------------------------------------------------
# Tests for HybridDimVjpAlgorithm
# ---------------------------------------------------------------------------

class TestHybridDimVjpAlgorithmInit:
    """Tests for __init__ of HybridDimVjpAlgorithm."""

    def test_init_with_float_decay(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        assert algo.decay == 0.9
        assert algo.vjp_method == 'single-step'

    def test_init_with_int_rank(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=5)
        assert isinstance(algo.decay, float)

    def test_init_with_custom_name(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9, name='my_algo')
        assert algo.name == 'my_algo'

    def test_init_with_custom_mode(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        mode = brainstate.mixin.Batching()
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9, mode=mode)
        assert algo.mode is mode

    def test_init_default_mode(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        assert isinstance(algo.mode, brainstate.mixin.Mode)

    def test_init_single_step_vjp(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9, vjp_method='single-step')
        assert algo.vjp_method == 'single-step'

    def test_init_multi_step_vjp(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9, vjp_method='multi-step')
        assert algo.vjp_method == 'multi-step'

    def test_init_invalid_vjp_method_raises(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        with pytest.raises(AssertionError):
            HybridDimVjpAlgorithm(gru, decay_or_rank=0.9, vjp_method='invalid')


class TestHybridDimVjpAlgorithmCompileAndState:
    """Tests for compile_graph, init_etrace_state, and reset_state."""

    def test_compile_graph_creates_etrace_dicts(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))
        assert hasattr(algo, 'etrace_xs')
        assert hasattr(algo, 'etrace_dfs')
        assert hasattr(algo, 'etrace_bwg')
        assert isinstance(algo.etrace_xs, dict)
        assert isinstance(algo.etrace_dfs, dict)
        assert isinstance(algo.etrace_bwg, dict)

    def test_compile_graph_is_compiled_flag(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        assert algo.is_compiled is False
        algo.compile_graph(brainstate.random.rand(3))
        assert algo.is_compiled is True

    def test_compile_graph_idempotent(self):
        """Calling compile_graph twice should not re-compile."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))
        xs_before = algo.etrace_xs
        algo.compile_graph(brainstate.random.rand(3))
        assert algo.etrace_xs is xs_before

    def test_etrace_states_have_values(self):
        """All etrace state dicts should contain brainstate.State objects with values."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        for key, st in algo.etrace_xs.items():
            assert isinstance(st, brainstate.State)
        for key, st in algo.etrace_dfs.items():
            assert isinstance(st, brainstate.State)
        for key, st in algo.etrace_bwg.items():
            assert isinstance(st, brainstate.State)

    def test_reset_state_zeros_etrace_xs(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        for key, st in algo.etrace_xs.items():
            st.value = jax.tree.map(lambda x: x + 1.0, st.value)

        algo.reset_state()

        for key, st in algo.etrace_xs.items():
            for leaf in jax.tree.leaves(st.value):
                assert jnp.allclose(leaf, 0.0), "etrace_xs should be zero after reset"

    def test_reset_state_zeros_etrace_dfs(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        for key, st in algo.etrace_dfs.items():
            st.value = jax.tree.map(lambda x: x + 1.0, st.value)

        algo.reset_state()

        for key, st in algo.etrace_dfs.items():
            for leaf in jax.tree.leaves(st.value):
                assert jnp.allclose(leaf, 0.0), "etrace_dfs should be zero after reset"

    def test_reset_state_zeros_etrace_bwg(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        for key, st in algo.etrace_bwg.items():
            st.value = jax.tree.map(lambda x: x + 1.0, st.value)

        algo.reset_state()

        for key, st in algo.etrace_bwg.items():
            for leaf in jax.tree.leaves(st.value):
                assert jnp.allclose(leaf, 0.0), "etrace_bwg should be zero after reset"

    def test_reset_state_zeros_running_index(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        algo.running_index.value = 5
        algo.reset_state()
        assert algo.running_index.value == 0


class TestHybridDimVjpAlgorithmEtraceData:
    """Tests for _get_etrace_data / _assign_etrace_data round-trip."""

    def test_get_etrace_data_returns_3_tuple(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        result = algo._get_etrace_data()
        assert isinstance(result, tuple)
        assert len(result) == 3
        etrace_xs, etrace_dfs, etrace_wgrads = result
        assert isinstance(etrace_xs, dict)
        assert isinstance(etrace_dfs, dict)
        assert isinstance(etrace_wgrads, dict)

    def test_get_etrace_data_keys_match(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        etrace_xs, etrace_dfs, etrace_wgrads = algo._get_etrace_data()
        assert set(etrace_xs.keys()) == set(algo.etrace_xs.keys())
        assert set(etrace_dfs.keys()) == set(algo.etrace_dfs.keys())
        assert set(etrace_wgrads.keys()) == set(algo.etrace_bwg.keys())

    def test_assign_get_round_trip(self):
        """_assign_etrace_data followed by _get_etrace_data should recover values."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        data = algo._get_etrace_data()

        modified_data = tuple(
            {k: jax.tree.map(lambda x: x + 1.0, v) for k, v in d.items()}
            for d in data
        )

        algo._assign_etrace_data(modified_data)

        recovered = algo._get_etrace_data()
        for orig_dict, recovered_dict in zip(modified_data, recovered):
            for key in orig_dict:
                assert key in recovered_dict
                orig_leaves = jax.tree.leaves(orig_dict[key])
                recov_leaves = jax.tree.leaves(recovered_dict[key])
                for o, r in zip(orig_leaves, recov_leaves):
                    assert jnp.allclose(o, r), "Round-trip assign/get mismatch"


class TestHybridDimVjpAlgorithmGetEtraceOf:
    """Tests for get_etrace_of."""

    def test_get_etrace_of_before_compile_raises(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        param_states = gru.states(brainstate.ParamState)
        first_weight = list(param_states.values())[0]
        with pytest.raises(ValueError, match='not been compiled'):
            algo.get_etrace_of(first_weight)

    def test_get_etrace_of_unknown_weight_raises(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        fake_weight = brainstate.ParamState(jnp.ones((10, 10)))
        with pytest.raises(ValueError, match='Do not the etrace'):
            algo.get_etrace_of(fake_weight)

    def test_get_etrace_of_returns_3_tuple_of_dicts(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        param_states = gru.states(brainstate.ParamState)
        first_weight = list(param_states.values())[0]

        result = algo.get_etrace_of(first_weight)
        assert isinstance(result, tuple)
        assert len(result) == 3
        etrace_xs, etrace_dfs, etrace_bws = result
        assert isinstance(etrace_xs, dict)
        assert isinstance(etrace_dfs, dict)
        assert isinstance(etrace_bws, dict)

    def test_get_etrace_of_all_tracked_weights(self):
        """Calling get_etrace_of for every tracked weight should not raise."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        algo.compile_graph(brainstate.random.rand(3))

        # Only query weights that are actually tracked in the etrace graph.
        # Some weights (e.g. Wr in GRU) may not be associated with hidden
        # states and therefore are not in the etrace relations.
        tracked_weights = set()
        for relation in algo.graph.hidden_param_op_relations:
            tracked_weights.add(id(relation.weight))

        param_states = gru.states(brainstate.ParamState)
        for path, weight in param_states.items():
            if id(weight) in tracked_weights:
                result = algo.get_etrace_of(weight)
                assert isinstance(result, tuple)
                assert len(result) == 3


# ---------------------------------------------------------------------------
# Integration tests using real models
# ---------------------------------------------------------------------------

class TestHybridDimVjpAlgorithmIntegration:
    """Integration tests running full forward and gradient passes."""

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_single_step_vjp(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algo = braintrace.HybridDimVjpAlgorithm(model, decay_or_rank=0.9)
        algo.compile_graph(inputs[0])

        outs = brainstate.transform.for_loop(algo, inputs)
        assert outs.shape == (n_seq, n_rec)

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_single_step_gradient(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algo = braintrace.HybridDimVjpAlgorithm(model, decay_or_rank=0.9)
        algo.compile_graph(inputs[0])

        @brainstate.transform.jit
        def grad_fn(inp):
            return brainstate.transform.grad(
                lambda inp: algo(inp).sum(),
                model.states(brainstate.ParamState)
            )(inp)

        grads = grad_fn(inputs[0])
        assert isinstance(grads, dict)
        assert len(grads) > 0
        for path, g in grads.items():
            assert g is not None

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_multi_step_vjp(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algo = braintrace.HybridDimVjpAlgorithm(
            model, decay_or_rank=0.9, vjp_method='multi-step'
        )
        algo.compile_graph(inputs[0])

        outs = algo(braintrace.MultiStepData(inputs))
        assert outs.shape == (n_seq, n_rec)

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_multi_step_gradient(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algo = braintrace.HybridDimVjpAlgorithm(
            model, decay_or_rank=0.9, vjp_method='multi-step'
        )
        algo.compile_graph(inputs[0])

        @brainstate.transform.jit
        def grad_fn(inp):
            return brainstate.transform.grad(
                lambda inp: algo(braintrace.MultiStepData(inp)).sum(),
                model.states(brainstate.ParamState)
            )(inp)

        grads = grad_fn(inputs[:2])
        assert isinstance(grads, dict)
        for path, g in grads.items():
            assert g is not None

    def test_lru_single_step_vjp(self):
        """LRUCell uses ElemWiseParam, so it exercises the O(n^2) code path."""
        model = braintrace.nn.LRUCell(4, 5)
        brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(10, 4)
        algo = braintrace.HybridDimVjpAlgorithm(model, decay_or_rank=0.9)
        algo.compile_graph(inputs[0])

        outs = brainstate.transform.for_loop(algo, inputs)
        assert outs.shape == (10, 4)

    def test_for_loop_produces_outputs_each_step(self):
        """Verify that for_loop produces correct output shape."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = braintrace.HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        inputs = brainstate.random.randn(8, 3)
        algo.compile_graph(inputs[0])
        outs = brainstate.transform.for_loop(algo, inputs)
        assert outs.shape == (8, 4)

    def test_reset_and_rerun(self):
        """After reset_state, running again should produce valid outputs."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = braintrace.HybridDimVjpAlgorithm(gru, decay_or_rank=0.9)
        inputs = brainstate.random.randn(5, 3)
        algo.compile_graph(inputs[0])

        outs1 = brainstate.transform.for_loop(algo, inputs)
        assert outs1.shape == (5, 4)

        algo.reset_state()
        gru.reset_state()
        outs2 = brainstate.transform.for_loop(algo, inputs)
        assert outs2.shape == (5, 4)

    def test_hybrid_with_integer_rank(self):
        """Using an integer rank instead of float decay should work."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        algo = braintrace.HybridDimVjpAlgorithm(gru, decay_or_rank=2)
        algo.compile_graph(brainstate.random.rand(3))

        inputs = brainstate.random.randn(5, 3)
        outs = brainstate.transform.for_loop(algo, inputs)
        assert outs.shape == (5, 4)
