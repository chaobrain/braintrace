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
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._etrace_algorithms import ETraceAlgorithm
from braintrace._etrace_vjp.base import ETraceVjpAlgorithm
from braintrace._etrace_vjp.graph_executor import ETraceVjpGraphExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gru(in_size=3, out_size=4):
    """Create and initialize a GRU model for testing."""
    model = braintrace.nn.GRUCell(in_size, out_size)
    brainstate.nn.init_all_states(model)
    return model


class ConcreteVjpAlgorithm(ETraceVjpAlgorithm):
    """Minimal concrete subclass that implements all abstract protocol methods."""

    def __init__(self, model, name=None, vjp_method='single-step'):
        super().__init__(model, name=name, vjp_method=vjp_method)
        self._etrace_data = {}
        self._solve_weight_gradients_called = False
        self._update_etrace_data_called = False
        self._get_etrace_data_called = False
        self._assign_etrace_data_called = False

    def init_etrace_state(self, *args, **kwargs):
        """Initialize etrace states (no-op for testing)."""
        pass

    def _solve_weight_gradients(
        self,
        running_index,
        etrace_h2w_at_t,
        dl_to_hidden_groups,
        weight_vals,
        dl_to_nonetws_at_t,
        dl_to_etws_at_t,
    ):
        self._solve_weight_gradients_called = True
        return {k: jax.tree.map(jnp.zeros_like, v) for k, v in weight_vals.items()}

    def _update_etrace_data(
        self,
        running_index,
        etrace_vals_util_t_1,
        hid2weight_jac_single_or_multi_times,
        hid2hid_jac_single_or_multi_times,
        weight_vals,
        input_is_multi_step,
    ):
        self._update_etrace_data_called = True
        return etrace_vals_util_t_1

    def _get_etrace_data(self):
        self._get_etrace_data_called = True
        return self._etrace_data

    def _assign_etrace_data(self, etrace_vals):
        self._assign_etrace_data_called = True
        self._etrace_data = etrace_vals


# ---------------------------------------------------------------------------
# Tests: __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for ETraceVjpAlgorithm.__init__."""

    def test_default_vjp_method(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert algo.vjp_method == 'single-step'

    def test_single_step_vjp_method(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model, vjp_method='single-step')
        assert algo.vjp_method == 'single-step'

    def test_multi_step_vjp_method(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model, vjp_method='multi-step')
        assert algo.vjp_method == 'multi-step'

    def test_graph_executor_type(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert isinstance(algo.graph_executor, ETraceVjpGraphExecutor)

    def test_graph_executor_vjp_method_matches(self):
        model = _make_gru()
        for method in ('single-step', 'multi-step'):
            algo = ConcreteVjpAlgorithm(model, vjp_method=method)
            assert algo.graph_executor.vjp_method == method

    def test_custom_vjp_is_set(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert hasattr(algo, '_true_update_fun')
        # The custom_vjp wraps the _update_fn
        assert callable(algo._true_update_fun)

    def test_is_compiled_false_initially(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert algo.is_compiled is False

    def test_inherits_from_etrace_algorithm(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert isinstance(algo, ETraceAlgorithm)
        assert isinstance(algo, ETraceVjpAlgorithm)

    def test_name_parameter(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model, name='test_algo')
        assert algo.name == 'test_algo'

    def test_name_defaults_to_none(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        # When name is None, brainstate.nn.Module keeps it as None
        assert algo.name is None

    def test_model_stored(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert algo.model4compile is model


# ---------------------------------------------------------------------------
# Tests: vjp_method validation
# ---------------------------------------------------------------------------

class TestVjpMethodValidation:
    """Tests for vjp_method parameter validation."""

    def test_invalid_vjp_method_raises_assertion_error(self):
        model = _make_gru()
        with pytest.raises(AssertionError, match='single-step'):
            ConcreteVjpAlgorithm(model, vjp_method='invalid')

    def test_empty_string_raises_assertion_error(self):
        model = _make_gru()
        with pytest.raises(AssertionError):
            ConcreteVjpAlgorithm(model, vjp_method='')

    def test_none_vjp_method_raises_assertion_error(self):
        model = _make_gru()
        with pytest.raises(AssertionError):
            ConcreteVjpAlgorithm(model, vjp_method=None)

    def test_typo_single_raises_assertion_error(self):
        model = _make_gru()
        with pytest.raises(AssertionError):
            ConcreteVjpAlgorithm(model, vjp_method='singlestep')

    def test_typo_multi_raises_assertion_error(self):
        model = _make_gru()
        with pytest.raises(AssertionError):
            ConcreteVjpAlgorithm(model, vjp_method='multistep')

    def test_case_sensitive(self):
        model = _make_gru()
        with pytest.raises(AssertionError):
            ConcreteVjpAlgorithm(model, vjp_method='Single-Step')

    def test_case_sensitive_multi(self):
        model = _make_gru()
        with pytest.raises(AssertionError):
            ConcreteVjpAlgorithm(model, vjp_method='Multi-Step')

    def test_numeric_vjp_method_raises_assertion_error(self):
        model = _make_gru()
        with pytest.raises(AssertionError):
            ConcreteVjpAlgorithm(model, vjp_method=42)


# ---------------------------------------------------------------------------
# Tests: _assert_compiled
# ---------------------------------------------------------------------------

class TestAssertCompiled:
    """Tests for _assert_compiled method."""

    def test_raises_value_error_when_not_compiled(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert algo.is_compiled is False
        with pytest.raises(ValueError, match='compile_graph'):
            algo._assert_compiled()

    def test_no_error_when_compiled(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        algo.compile_graph(x)
        assert algo.is_compiled is True
        # Should not raise
        algo._assert_compiled()

    def test_error_message_content(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        with pytest.raises(ValueError) as exc_info:
            algo._assert_compiled()
        assert 'compile_graph()' in str(exc_info.value)
        assert 'not been compiled' in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: abstract protocol methods raise NotImplementedError
# ---------------------------------------------------------------------------

class TestAbstractProtocolMethods:
    """Tests that abstract protocol methods raise NotImplementedError on the base class."""

    def test_solve_weight_gradients_raises(self):
        model = _make_gru()
        algo = ETraceVjpAlgorithm(model)
        with pytest.raises(NotImplementedError):
            algo._solve_weight_gradients(
                running_index=0,
                etrace_h2w_at_t=None,
                dl_to_hidden_groups=[],
                weight_vals={},
                dl_to_nonetws_at_t=[],
                dl_to_etws_at_t=None,
            )

    def test_update_etrace_data_raises(self):
        model = _make_gru()
        algo = ETraceVjpAlgorithm(model)
        with pytest.raises(NotImplementedError):
            algo._update_etrace_data(
                running_index=0,
                etrace_vals_util_t_1={},
                hid2weight_jac_single_or_multi_times=({}, {}),
                hid2hid_jac_single_or_multi_times=[],
                weight_vals={},
                input_is_multi_step=False,
            )

    def test_get_etrace_data_raises(self):
        model = _make_gru()
        algo = ETraceVjpAlgorithm(model)
        with pytest.raises(NotImplementedError):
            algo._get_etrace_data()

    def test_assign_etrace_data_raises(self):
        model = _make_gru()
        algo = ETraceVjpAlgorithm(model)
        with pytest.raises(NotImplementedError):
            algo._assign_etrace_data({})


# ---------------------------------------------------------------------------
# Tests: __module__
# ---------------------------------------------------------------------------

class TestModuleAttribute:
    """Tests for the __module__ attribute."""

    def test_base_class_module(self):
        assert ETraceVjpAlgorithm.__module__ == 'braintrace'

    def test_instance_module_on_base(self):
        model = _make_gru()
        algo = ETraceVjpAlgorithm(model)
        assert algo.__class__.__module__ == 'braintrace'

    def test_concrete_subclass_module(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        # ConcreteVjpAlgorithm is defined in the test module, not 'braintrace'
        assert algo.__class__.__module__ != 'braintrace'


# ---------------------------------------------------------------------------
# Tests: update()
# ---------------------------------------------------------------------------

class TestUpdate:
    """Tests for the update method."""

    def test_update_raises_when_not_compiled(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        with pytest.raises(ValueError, match='compile_graph'):
            algo.update(x)

    def test_call_raises_when_not_compiled(self):
        """__call__ delegates to update, so it should also raise."""
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        with pytest.raises(ValueError, match='compile_graph'):
            algo(x)


# ---------------------------------------------------------------------------
# Tests: compile_graph
# ---------------------------------------------------------------------------

class TestCompileGraph:
    """Tests for compile_graph interaction."""

    def test_compile_graph_sets_is_compiled(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert algo.is_compiled is False
        x = jnp.ones((3,))
        algo.compile_graph(x)
        assert algo.is_compiled is True

    def test_compile_graph_idempotent(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        algo.compile_graph(x)
        assert algo.is_compiled is True
        # Calling compile_graph again should not fail
        algo.compile_graph(x)
        assert algo.is_compiled is True

    def test_graph_executor_has_compiled_graph(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        algo.compile_graph(x)
        assert algo.graph_executor._compiled_graph is not None

    def test_param_states_available_after_compile(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        algo.compile_graph(x)
        # param_states should be accessible
        assert algo.param_states is not None
        assert len(algo.param_states) > 0

    def test_hidden_states_available_after_compile(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        algo.compile_graph(x)
        assert algo.hidden_states is not None
        assert len(algo.hidden_states) > 0


# ---------------------------------------------------------------------------
# Tests: concrete subclass protocol method dispatch
# ---------------------------------------------------------------------------

class TestConcreteSubclassProtocol:
    """Tests that the concrete subclass protocol methods are properly callable."""

    def test_get_etrace_data_returns_dict(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        result = algo._get_etrace_data()
        assert isinstance(result, dict)
        assert algo._get_etrace_data_called is True

    def test_assign_etrace_data_stores_values(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        test_data = {'key': jnp.array([1.0, 2.0])}
        algo._assign_etrace_data(test_data)
        assert algo._assign_etrace_data_called is True
        assert algo._etrace_data is test_data

    def test_update_etrace_data_returns_input(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        etrace_vals = {'a': jnp.array([1.0])}
        result = algo._update_etrace_data(
            running_index=0,
            etrace_vals_util_t_1=etrace_vals,
            hid2weight_jac_single_or_multi_times=({}, {}),
            hid2hid_jac_single_or_multi_times=[],
            weight_vals={},
            input_is_multi_step=False,
        )
        assert algo._update_etrace_data_called is True
        assert result is etrace_vals

    def test_solve_weight_gradients_returns_zero_grads(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        weight_vals = {'w1': jnp.ones((3, 4))}
        result = algo._solve_weight_gradients(
            running_index=0,
            etrace_h2w_at_t=None,
            dl_to_hidden_groups=[],
            weight_vals=weight_vals,
            dl_to_nonetws_at_t=[],
            dl_to_etws_at_t=None,
        )
        assert algo._solve_weight_gradients_called is True
        assert 'w1' in result
        assert jnp.allclose(result['w1'], jnp.zeros((3, 4)))


# ---------------------------------------------------------------------------
# Tests: model validation in parent class
# ---------------------------------------------------------------------------

class TestModelValidation:
    """Tests for model type validation inherited from ETraceAlgorithm."""

    def test_non_module_model_raises_type_error(self):
        with pytest.raises(TypeError, match='brainstate.nn.Module'):
            ConcreteVjpAlgorithm(model="not_a_module")

    def test_none_model_raises_error(self):
        with pytest.raises((ValueError, TypeError)):
            ConcreteVjpAlgorithm(model=None)

    def test_callable_but_not_module_raises(self):
        with pytest.raises(TypeError):
            ConcreteVjpAlgorithm(model=lambda x: x)


# ---------------------------------------------------------------------------
# Tests: running_index initialization
# ---------------------------------------------------------------------------

class TestRunningIndex:
    """Tests for the running_index state."""

    def test_running_index_initial_value(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert algo.running_index.value == 0

    def test_running_index_is_long_term_state(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert isinstance(algo.running_index, brainstate.LongTermState)


# ---------------------------------------------------------------------------
# Tests: with different RNN cell types
# ---------------------------------------------------------------------------

class TestWithDifferentModels:
    """Tests that ETraceVjpAlgorithm works with different model types."""

    @pytest.mark.parametrize("cell_cls", [
        braintrace.nn.GRUCell,
        braintrace.nn.MGUCell,
        braintrace.nn.ValinaRNNCell,
    ])
    def test_init_with_different_cells(self, cell_cls):
        model = cell_cls(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ConcreteVjpAlgorithm(model)
        assert algo.vjp_method == 'single-step'
        assert algo.is_compiled is False

    @pytest.mark.parametrize("cell_cls", [
        braintrace.nn.GRUCell,
        braintrace.nn.MGUCell,
        braintrace.nn.ValinaRNNCell,
    ])
    def test_compile_with_different_cells(self, cell_cls):
        model = cell_cls(3, 4)
        brainstate.nn.init_all_states(model)
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        algo.compile_graph(x)
        assert algo.is_compiled is True

    @pytest.mark.parametrize("vjp_method", ['single-step', 'multi-step'])
    def test_compile_with_both_vjp_methods(self, vjp_method):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model, vjp_method=vjp_method)
        x = jnp.ones((3,))
        algo.compile_graph(x)
        assert algo.is_compiled is True
        assert algo.vjp_method == vjp_method


# ---------------------------------------------------------------------------
# Tests: _update_fn, _update_fn_fwd, _update_fn_bwd exist
# ---------------------------------------------------------------------------

class TestInternalMethods:
    """Tests that internal methods are properly defined on the class."""

    def test_update_fn_is_callable(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert callable(algo._update_fn)

    def test_update_fn_fwd_is_callable(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert callable(algo._update_fn_fwd)

    def test_update_fn_bwd_is_callable(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert callable(algo._update_fn_bwd)

    def test_true_update_fun_is_callable(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert callable(algo._true_update_fun)


# ---------------------------------------------------------------------------
# Tests: graph property
# ---------------------------------------------------------------------------

class TestGraphProperty:
    """Tests for the graph property accessor."""

    def test_graph_accessible_after_compile(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        x = jnp.ones((3,))
        algo.compile_graph(x)
        graph = algo.graph
        assert graph is not None

    def test_executor_property(self):
        model = _make_gru()
        algo = ConcreteVjpAlgorithm(model)
        assert algo.executor is algo.graph_executor
