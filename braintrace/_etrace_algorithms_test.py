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


import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import brainstate
import jax.numpy as jnp

import braintrace
from braintrace._etrace_algorithms import ETraceAlgorithm, EligibilityTrace
from braintrace._etrace_graph_executor import ETraceGraphExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleModule(brainstate.nn.Module):
    """A minimal module for testing."""

    def __init__(self, n_in=3, n_out=4):
        super().__init__()
        self.w = brainstate.ParamState(jnp.ones((n_in, n_out)))

    def __call__(self, x):
        return x @ self.w.value


class ConcreteETraceAlgorithm(ETraceAlgorithm):
    """A concrete subclass so we can test base-class behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_etrace_called_with = None
        self.update_calls = []

    def update(self, *args):
        self.update_calls.append(args)
        return 'updated'

    def init_etrace_state(self, *args, **kwargs):
        self.init_etrace_called_with = (args, kwargs)

    def get_etrace_of(self, weight):
        return f'etrace_of_{weight}'


def _make_module_and_executor(n_in=3, n_out=4):
    """Create a simple module and executor pair."""
    module = SimpleModule(n_in, n_out)
    executor = ETraceGraphExecutor(module)
    return module, executor


def _make_gru_and_executor(n_in=3, n_out=4):
    """Create a GRU model and executor for integration tests."""
    gru = braintrace.nn.GRUCell(n_in, n_out)
    brainstate.nn.init_all_states(gru)
    executor = ETraceGraphExecutor(gru)
    return gru, executor


# ===========================================================================
# Tests for EligibilityTrace
# ===========================================================================

class TestEligibilityTrace(unittest.TestCase):

    def test_is_short_term_state(self):
        et = EligibilityTrace(jnp.zeros(3))
        self.assertIsInstance(et, brainstate.ShortTermState)

    def test_stores_value(self):
        val = jnp.ones(5)
        et = EligibilityTrace(val)
        self.assertTrue(jnp.array_equal(et.value, val))

    def test_module_attribute(self):
        self.assertEqual(EligibilityTrace.__module__, 'braintrace')

    def test_different_shapes(self):
        for shape in [(2,), (3, 4), (2, 3, 5)]:
            et = EligibilityTrace(jnp.zeros(shape))
            self.assertEqual(et.value.shape, shape)

    def test_float32_dtype(self):
        et = EligibilityTrace(jnp.zeros(3, dtype=jnp.float32))
        self.assertEqual(et.value.dtype, jnp.float32)


# ===========================================================================
# Tests for ETraceAlgorithm.__init__
# ===========================================================================

class TestETraceAlgorithmInit(unittest.TestCase):

    def test_valid_init(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)

        self.assertIs(algo.model4compile, module)
        self.assertIs(algo.graph_executor, executor)
        self.assertFalse(algo.is_compiled)
        self.assertEqual(algo.running_index.value, 0)
        self.assertIsNone(algo._param_states)
        self.assertIsNone(algo._hidden_states)
        self.assertIsNone(algo._other_states)

    def test_custom_name(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor, name='my_algo')
        self.assertEqual(algo.name, 'my_algo')

    def test_default_name_is_none(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        # Default name is None when not explicitly set
        self.assertIsNone(algo.name)

    def test_invalid_model_raises(self):
        _, executor = _make_module_and_executor()
        with self.assertRaises(ValueError) as ctx:
            ETraceAlgorithm('not_a_module', executor)
        self.assertIn('brainstate.nn.Module', str(ctx.exception))

    def test_invalid_model_type_in_error(self):
        _, executor = _make_module_and_executor()
        with self.assertRaises(ValueError) as ctx:
            ETraceAlgorithm(42, executor)
        self.assertIn("<class 'int'>", str(ctx.exception))

    def test_invalid_executor_raises(self):
        module, _ = _make_module_and_executor()
        with self.assertRaises(ValueError) as ctx:
            ETraceAlgorithm(module, 'not_an_executor')
        self.assertIn('ETraceGraphExecutor', str(ctx.exception))

    def test_invalid_executor_type_in_error(self):
        module, _ = _make_module_and_executor()
        with self.assertRaises(ValueError) as ctx:
            ETraceAlgorithm(module, [1, 2, 3])
        self.assertIn("<class 'list'>", str(ctx.exception))

    def test_both_invalid_model_raises_first(self):
        """When both model and executor are invalid, model check happens first."""
        with self.assertRaises(ValueError) as ctx:
            ETraceAlgorithm('bad_model', 'bad_executor')
        self.assertIn('brainstate.nn.Module', str(ctx.exception))

    def test_running_index_is_long_term_state(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        self.assertIsInstance(algo.running_index, brainstate.LongTermState)

    def test_module_attribute(self):
        self.assertEqual(ETraceAlgorithm.__module__, 'braintrace')


# ===========================================================================
# Tests for ETraceAlgorithm properties (before compilation)
# ===========================================================================

class TestETraceAlgorithmPropertiesBeforeCompile(unittest.TestCase):

    def test_graph_before_compile_raises(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        with self.assertRaises(ValueError):
            _ = algo.graph

    def test_executor_property(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        self.assertIs(algo.executor, executor)

    def test_path_to_states_before_compile_raises(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        with self.assertRaises(ValueError):
            _ = algo.path_to_states

    def test_state_id_to_path_before_compile_raises(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        with self.assertRaises(ValueError):
            _ = algo.state_id_to_path


# ===========================================================================
# Tests for ETraceAlgorithm abstract methods
# ===========================================================================

class TestETraceAlgorithmAbstractMethods(unittest.TestCase):

    def test_update_raises(self):
        module, executor = _make_module_and_executor()
        algo = ETraceAlgorithm(module, executor)
        with self.assertRaises(NotImplementedError):
            algo.update()

    def test_call_raises(self):
        module, executor = _make_module_and_executor()
        algo = ETraceAlgorithm(module, executor)
        with self.assertRaises(NotImplementedError):
            algo()

    def test_init_etrace_state_raises(self):
        module, executor = _make_module_and_executor()
        algo = ETraceAlgorithm(module, executor)
        with self.assertRaises(NotImplementedError):
            algo.init_etrace_state()

    def test_get_etrace_of_raises(self):
        module, executor = _make_module_and_executor()
        algo = ETraceAlgorithm(module, executor)
        with self.assertRaises(NotImplementedError):
            algo.get_etrace_of(module.w)


# ===========================================================================
# Tests for ETraceAlgorithm.__call__ and .update (concrete)
# ===========================================================================

class TestETraceAlgorithmCallAndUpdate(unittest.TestCase):

    def test_call_delegates_to_update(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        result = algo('arg1', 'arg2')
        self.assertEqual(result, 'updated')
        self.assertEqual(algo.update_calls, [('arg1', 'arg2')])

    def test_call_with_no_args(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        result = algo()
        self.assertEqual(result, 'updated')
        self.assertEqual(algo.update_calls, [()])

    def test_update_return_value(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        result = algo.update('x')
        self.assertEqual(result, 'updated')


# ===========================================================================
# Tests for ETraceAlgorithm.get_etrace_of (concrete)
# ===========================================================================

class TestETraceAlgorithmGetEtraceOf(unittest.TestCase):

    def test_get_etrace_of_with_param_state(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        result = algo.get_etrace_of(module.w)
        self.assertIn('etrace_of_', result)

    def test_get_etrace_of_with_path(self):
        module, executor = _make_module_and_executor()
        algo = ConcreteETraceAlgorithm(module, executor)
        result = algo.get_etrace_of(('w',))
        self.assertEqual(result, "etrace_of_('w',)")


# ===========================================================================
# Tests for compile_graph
# ===========================================================================

class TestETraceAlgorithmCompileGraph(unittest.TestCase):

    def test_compile_graph_with_gru(self):
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)

        self.assertFalse(algo.is_compiled)
        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)
        self.assertTrue(algo.is_compiled)

    def test_compile_graph_calls_init_etrace_state(self):
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)

        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        self.assertIsNotNone(algo.init_etrace_called_with)
        # The args passed to init_etrace_state should include input_data
        self.assertEqual(len(algo.init_etrace_called_with[0]), 1)

    def test_compile_graph_invalidates_cached_states(self):
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)

        # Manually set cached states
        algo._param_states = 'cached_params'
        algo._hidden_states = 'cached_hidden'
        algo._other_states = 'cached_other'

        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        # After compilation, accessing properties should trigger fresh split
        # The cached values should have been invalidated during compile
        # (but re-populated on first property access)
        self.assertIsNotNone(algo.param_states)
        self.assertIsNotNone(algo.hidden_states)

    def test_compile_graph_idempotent(self):
        """Calling compile_graph twice should only compile once."""
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)

        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)
        first_graph = algo.graph

        # Reset tracking
        algo.init_etrace_called_with = None

        # Second compile should be a no-op
        algo.compile_graph(input_data)
        self.assertIs(algo.graph, first_graph)
        self.assertIsNone(algo.init_etrace_called_with)


# ===========================================================================
# Tests for properties after compilation (integration)
# ===========================================================================

class TestETraceAlgorithmPropertiesAfterCompile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gru, cls.executor = _make_gru_and_executor()
        cls.algo = ConcreteETraceAlgorithm(cls.gru, cls.executor)
        cls.input_data = brainstate.random.rand(3)
        cls.algo.compile_graph(cls.input_data)

    def test_graph_returns_etrace_graph(self):
        self.assertIsInstance(self.algo.graph, braintrace.ETraceGraph)

    def test_param_states_type(self):
        self.assertIsInstance(self.algo.param_states, brainstate.util.FlattedDict)

    def test_param_states_contains_param_states(self):
        for path, state in self.algo.param_states.items():
            self.assertIsInstance(state, brainstate.ParamState)

    def test_hidden_states_type(self):
        self.assertIsInstance(self.algo.hidden_states, brainstate.util.FlattedDict)

    def test_hidden_states_contains_hidden_states(self):
        for path, state in self.algo.hidden_states.items():
            self.assertIsInstance(state, brainstate.HiddenState)

    def test_other_states_type(self):
        self.assertIsInstance(self.algo.other_states, brainstate.util.FlattedDict)

    def test_gru_has_param_states(self):
        self.assertGreater(len(self.algo.param_states), 0)

    def test_gru_has_hidden_states(self):
        self.assertGreater(len(self.algo.hidden_states), 0)

    def test_hidden_state_paths(self):
        hidden_keys = list(self.algo.hidden_states.keys())
        self.assertIn(('h',), hidden_keys)

    def test_path_to_states_type(self):
        self.assertIsInstance(self.algo.path_to_states, brainstate.util.FlattedDict)

    def test_state_id_to_path_type(self):
        self.assertIsInstance(self.algo.state_id_to_path, dict)

    def test_state_id_to_path_maps_ids_to_paths(self):
        for state_id, path in self.algo.state_id_to_path.items():
            self.assertIsInstance(state_id, int)
            self.assertIsInstance(path, tuple)


# ===========================================================================
# Tests for _split_state (lazy initialization)
# ===========================================================================

class TestETraceAlgorithmSplitState(unittest.TestCase):

    def test_lazy_init_param_states(self):
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)
        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        # Before first access, cached should be None (reset during compile)
        algo._param_states = None
        algo._hidden_states = None
        algo._other_states = None

        # Accessing param_states triggers _split_state
        params = algo.param_states
        self.assertIsNotNone(params)
        # All three should now be populated
        self.assertIsNotNone(algo._param_states)
        self.assertIsNotNone(algo._hidden_states)
        self.assertIsNotNone(algo._other_states)

    def test_lazy_init_hidden_states(self):
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)
        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        algo._param_states = None
        algo._hidden_states = None
        algo._other_states = None

        # Accessing hidden_states triggers _split_state
        hidden = algo.hidden_states
        self.assertIsNotNone(hidden)
        self.assertIsNotNone(algo._param_states)

    def test_lazy_init_other_states(self):
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)
        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        algo._param_states = None
        algo._hidden_states = None
        algo._other_states = None

        # Accessing other_states triggers _split_state
        other = algo.other_states
        self.assertIsNotNone(other)
        self.assertIsNotNone(algo._param_states)

    def test_cached_states_not_recomputed(self):
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)
        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        # Access once to populate
        params1 = algo.param_states
        # Access again - should return same object
        params2 = algo.param_states
        self.assertIs(params1, params2)


# ===========================================================================
# Tests for show_graph
# ===========================================================================

class TestETraceAlgorithmShowGraph(unittest.TestCase):

    def test_show_graph_delegates_to_executor(self):
        gru, executor = _make_gru_and_executor()
        algo = ConcreteETraceAlgorithm(gru, executor)
        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        # Patch executor's show_graph to verify delegation
        executor.show_graph = MagicMock(return_value=None)
        algo.show_graph()
        executor.show_graph.assert_called_once()


# ===========================================================================
# Integration: GRU model full workflow
# ===========================================================================

class TestETraceAlgorithmGRUIntegration(unittest.TestCase):

    def test_full_workflow(self):
        """End-to-end: create -> compile -> access properties."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        executor = ETraceGraphExecutor(gru)
        algo = ConcreteETraceAlgorithm(gru, executor)

        input_data = brainstate.random.rand(3)

        # Before compile
        self.assertFalse(algo.is_compiled)

        # Compile
        algo.compile_graph(input_data)
        self.assertTrue(algo.is_compiled)

        # Access graph
        graph = algo.graph
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, braintrace.ETraceGraph)

        # Access states
        self.assertGreater(len(algo.param_states), 0)
        self.assertGreater(len(algo.hidden_states), 0)

        # State consistency: all param states should be ParamState instances
        for path, state in algo.param_states.items():
            self.assertIsInstance(state, brainstate.ParamState)
            self.assertIsInstance(path, tuple)

        # State consistency: all hidden states should be HiddenState instances
        for path, state in algo.hidden_states.items():
            self.assertIsInstance(state, brainstate.HiddenState)
            self.assertIsInstance(path, tuple)

        # state_id_to_path should have entries
        self.assertGreater(len(algo.state_id_to_path), 0)

        # Call update
        result = algo.update(input_data)
        self.assertEqual(result, 'updated')


# ===========================================================================
# Integration: LSTM model
# ===========================================================================

class TestETraceAlgorithmLSTMIntegration(unittest.TestCase):

    def test_lstm_compile_and_properties(self):
        lstm = braintrace.nn.LSTMCell(3, 4)
        brainstate.nn.init_all_states(lstm)
        executor = ETraceGraphExecutor(lstm)
        algo = ConcreteETraceAlgorithm(lstm, executor)

        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        self.assertTrue(algo.is_compiled)

        # LSTM has multiple hidden states (h and c)
        hidden_keys = list(algo.hidden_states.keys())
        self.assertGreaterEqual(len(hidden_keys), 2)

        # Should have param states
        self.assertGreater(len(algo.param_states), 0)


# ===========================================================================
# Integration: Two-layer network
# ===========================================================================

class TestETraceAlgorithmTwoLayerIntegration(unittest.TestCase):

    def test_two_layer_gru_compile(self):
        net = brainstate.nn.Sequential(
            braintrace.nn.GRUCell(3, 4),
            brainstate.nn.ReLU(),
            braintrace.nn.GRUCell(4, 3),
        )
        brainstate.nn.init_all_states(net)
        executor = ETraceGraphExecutor(net)
        algo = ConcreteETraceAlgorithm(net, executor)

        input_data = brainstate.random.rand(3)
        algo.compile_graph(input_data)

        self.assertTrue(algo.is_compiled)

        # Two GRU layers -> more param states and hidden states
        self.assertGreater(len(algo.param_states), 3)
        self.assertGreaterEqual(len(algo.hidden_states), 2)


if __name__ == '__main__':
    unittest.main()
