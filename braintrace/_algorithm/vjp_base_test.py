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
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm import ETraceAlgorithm
from braintrace._algorithm.vjp_base import ETraceVjpAlgorithm
from braintrace._algorithm.vjp_graph_executor import ETraceVjpGraphExecutor


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

    def __init__(self, model, name=None, vjp_method='single-step', **kwargs):
        super().__init__(model, name=name, vjp_method=vjp_method, **kwargs)
        self._etrace_data = {}
        self._solve_weight_gradients_called = False
        self._update_etrace_data_called = False
        self._get_etrace_data_called = False
        self._assign_etrace_data_called = False

    def init_etrace_state(self, *args, **kwargs):
        """This subclass has no trace of its own; chain for the axis-side state."""
        super().init_etrace_state(*args, **kwargs)

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


# ---------------------------------------------------------------------------
# Tests: _compute_learning_signal hook
# ---------------------------------------------------------------------------


class TestComputeLearningSignalHook:
    """Tests for the overridable learning-signal hook on ETraceVjpAlgorithm."""

    def test_default_hook_is_identity(self):
        """`learning_signal='symmetric'` (the default) returns input unchanged.

        Built on a real instance rather than a ``__new__``'d shell: the hook now
        reads ``self.config`` to decide, so a shell that never ran ``__init__``
        would test nothing about the shipped default.
        """
        algo = ConcreteVjpAlgorithm(_make_gru())
        assert algo.config.learning_signal == 'symmetric'
        dl2h = [jnp.ones((2, 3)), jnp.zeros((2, 5))]
        out = algo._compute_learning_signal(dl2h, args=())
        assert isinstance(out, (list, tuple))
        assert len(out) == 2
        assert jnp.allclose(out[0], dl2h[0])
        assert jnp.allclose(out[1], dl2h[1])

    def test_random_feedback_without_allocation_raises(self):
        """A configuration that cannot be honoured must fail, not degrade.

        An unallocated feedback dict used to be indistinguishable from
        ``'symmetric'`` at this hook, so the algorithm would quietly compute a
        different learning rule than the one requested.
        """
        algo = ConcreteVjpAlgorithm(
            _make_gru(),
            config=braintrace.ETraceConfig(learning_signal='random_feedback'),
            random_feedback_key=jax.random.PRNGKey(0),
        )
        with pytest.raises(RuntimeError, match='silently fall back to symmetric'):
            algo._compute_learning_signal([jnp.ones((2, 3))], args=())

    def test_random_feedback_without_a_key_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match='random_feedback_key'):
            ConcreteVjpAlgorithm(
                _make_gru(),
                config=braintrace.ETraceConfig(learning_signal='random_feedback'),
            )

    def test_override_hook_replaces_learning_signal(self):
        """Subclass override is *used*, not merely called.

        Observing the call and a non-zero gradient is not enough: a base that
        invoked the hook and then went on using ``dl_autodiff`` would pass both.
        What pins it is that the parameter gradient is linear in the learning
        signal, so returning ``k * ones`` must scale the gradient by exactly
        ``k`` -- and must differ from the un-overridden reverse-AD signal.
        """
        from braintrace._algorithm.param_dim_vjp import ParamDimVjpAlgorithm

        class Mini(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((3, 3)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

            def update(self, x):
                self.h.value = jax.nn.tanh(
                    braintrace.matmul(self.h.value + x, self.w.value)
                )
                return self.h.value

        captured = {}

        def _w_grad(algo_factory):
            net = Mini()
            brainstate.nn.init_all_states(net, batch_size=1)
            algo = algo_factory(net)
            x0 = jnp.ones((1, 3))
            algo.compile_graph(x0)
            algo.init_etrace_state()
            grads, _ = brainstate.transform.grad(
                lambda x: (algo.update(x) ** 2).sum(),
                algo.param_states, return_value=True,
            )(x0)
            return np.asarray(u.get_mantissa(grads[next(iter(grads))]))

        def _constant(k):
            class ConstantSignalAlgo(ParamDimVjpAlgorithm):
                def _compute_learning_signal(self, dl_autodiff, args):
                    captured['autodiff'] = dl_autodiff
                    captured['args'] = args
                    return [jnp.full_like(a, k) for a in dl_autodiff]

            return ConstantSignalAlgo

        ones = _w_grad(_constant(1.0))
        assert 'autodiff' in captured, 'the hook was never invoked'
        assert np.any(ones != 0.0)

        twos = _w_grad(_constant(2.0))
        np.testing.assert_allclose(twos, 2.0 * ones, rtol=1e-5, atol=1e-7)

        autodiff = _w_grad(ParamDimVjpAlgorithm)
        assert not np.allclose(ones, autodiff, rtol=1e-3), (
            'the constant signal produced the reverse-AD gradient, so the '
            'override was called but its return value was discarded')

# ---------------------------------------------------------------------------
# P4: the modulator expansion contract, and the two-pass exit-cotangent hooks.
# ---------------------------------------------------------------------------

class TestExpandModulatorToGroup:
    """The ``learning_signal='modulatory'`` broadcasting contract."""

    def test_a_scalar_fills_the_whole_group_shape(self):
        from braintrace._algorithm.vjp_base import expand_modulator_to_group
        got = expand_modulator_to_group(0.25, (1, 4, 1), group_index=0)
        assert got.shape == (1, 4, 1)
        assert jnp.all(got == 0.25)

    def test_a_varshape_modulator_gains_the_trailing_state_axis(self):
        # The reason this helper exists: NumPy broadcasting aligns *trailing*
        # axes, so (1, 4) against (1, 4, 1) would raise -- or worse, when
        # num_state == n_rec, silently align the width against the state axis.
        from braintrace._algorithm.vjp_base import expand_modulator_to_group
        m = jnp.asarray([[0.0, 1.0, 2.0, 3.0]])
        got = expand_modulator_to_group(m, (1, 4, 1), group_index=0)
        assert got.shape == (1, 4, 1)
        assert jnp.allclose(got[..., 0], m)
        with pytest.raises(Exception):
            jnp.broadcast_to(m, (1, 4, 1))     # the trap, pinned

    def test_a_square_group_is_not_transposed_by_accident(self):
        # varshape (1, 3) with num_state 3 -> group shape (1, 3, 3). Bare
        # broadcasting would happily align the modulator's width against the
        # state axis here; ``expand_to`` must put it on the width axis.
        from braintrace._algorithm.vjp_base import expand_modulator_to_group
        m = jnp.asarray([[1.0, 2.0, 3.0]])
        got = expand_modulator_to_group(m, (1, 3, 3), group_index=0)
        assert got.shape == (1, 3, 3)
        for s in range(3):
            assert jnp.allclose(got[..., s], m)

    def test_a_fully_shaped_modulator_passes_through(self):
        from braintrace._algorithm.vjp_base import expand_modulator_to_group
        m = jnp.arange(4.0).reshape(1, 4, 1)
        got = expand_modulator_to_group(m, (1, 4, 1), group_index=0)
        assert jnp.array_equal(got, m)

    def test_units_survive_the_expansion(self):
        from braintrace._algorithm.vjp_base import expand_modulator_to_group
        got = expand_modulator_to_group(3.0 * u.mV, (1, 2, 1), group_index=0)
        assert u.get_unit(got) == u.get_unit(3.0 * u.mV)
        assert u.math.shape(got) == (1, 2, 1)

    def test_a_sequence_is_refused_with_the_anti_osttp_reason(self):
        from braintrace._algorithm.vjp_base import expand_modulator_to_group
        with pytest.raises(TypeError, match='per-group'):
            expand_modulator_to_group(
                [jnp.ones((1, 4, 1))] * 2, (1, 4, 1), group_index=0)

    def test_a_non_broadcastable_shape_names_the_group_and_both_shapes(self):
        from braintrace._algorithm.vjp_base import expand_modulator_to_group
        with pytest.raises(ValueError) as exc:
            expand_modulator_to_group(
                jnp.ones((3, 7)), (1, 4, 1), group_index=2,
                hidden_paths=[('h',)])
        msg = str(exc.value)
        assert '(3, 7)' in msg and '(1, 4, 1)' in msg
        assert 'group 2' in msg and "('h',)" in msg


class TestAddFutureForPlainPaths:
    """The pass-2 routing rule, in isolation."""

    def test_plain_paths_are_added_and_etp_paths_are_not(self):
        from braintrace._algorithm.vjp_base import _add_future_for_plain_paths
        base = {('w',): jnp.ones((2,)), ('win',): jnp.ones((2,))}
        future = {('w',): jnp.full((2,), 10.0), ('win',): jnp.full((2,), 10.0)}
        got = _add_future_for_plain_paths(base, future, {('w',)})
        assert jnp.array_equal(got[('w',)], jnp.ones((2,)))
        assert jnp.array_equal(got[('win',)], jnp.full((2,), 11.0))

    def test_an_absent_pass_is_a_no_op(self):
        from braintrace._algorithm.vjp_base import _add_future_for_plain_paths
        base = {('w',): jnp.ones((2,))}
        assert _add_future_for_plain_paths(base, None, set()) is base
        assert _add_future_for_plain_paths(base, {}, set()) is base

    def test_a_path_the_first_pass_lacks_raises_instead_of_vanishing(self):
        """The unreachable branch, kept loud on purpose.

        Both trees are unflattened from one ``in_tree``, so a path in ``future``
        and not in ``base`` cannot occur. Skipping it quietly would drop exactly
        the cross-window credit this helper delivers, and every DNI invariance
        test would still pass -- the failure shape that let F-34 through.
        """
        from braintrace._algorithm.vjp_base import _add_future_for_plain_paths
        base = {('w',): jnp.ones((2,))}
        future = {('w',): jnp.ones((2,)), ('ghost',): jnp.ones((2,))}
        with pytest.raises(KeyError, match='ghost'):
            _add_future_for_plain_paths(base, future, set())


class TestTheDefaultHooksAreInert:
    """Every non-DNI algorithm must pay nothing for the second pass."""

    def test_inject_exit_cotangent_defaults_to_none(self):
        algo = ConcreteVjpAlgorithm(_make_gru())
        assert algo._inject_exit_cotangent({}, None) is None

    def test_get_update_aux_defaults_to_none(self):
        algo = ConcreteVjpAlgorithm(_make_gru())
        assert algo._get_update_aux() is None

    def test_etp_routed_paths_reads_the_compiled_graph(self):
        from braintrace._testing import oracle_models as om
        spec = om.plain_and_etp_rnn(n_in=3, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='multi-step')
        algo.compile_graph(jnp.ones((1, 3)))
        # `w` goes through braintrace.matmul; `win`/`wout` are plain `@`.
        assert algo._etp_routed_paths() == {('w',)}


class TestTheExitCotangentTemplate:
    """``_exit_cotangent_grads``: zero everywhere but the exit hiddens."""

    def _algo(self):
        from braintrace._testing import oracle_models as om
        spec = om.plain_and_etp_rnn(n_in=3, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='multi-step')
        algo.compile_graph(jnp.ones((1, 3)))
        algo.init_etrace_state()
        return algo

    def _template(self, algo):
        return (jnp.ones((1, 2)),
                {p: jnp.ones((1, 4)) for p in algo.hidden_states},
                {})

    def test_everything_but_the_hiddens_is_zeroed(self):
        algo = self._algo()
        out, hid, oth = algo._exit_cotangent_grads(
            self._template(algo), {('h',): jnp.full((1, 4), 0.5)})
        assert jnp.all(out == 0.0)
        assert jnp.all(hid[('h',)] == 0.5)
        assert oth == {}

    def test_an_omitted_path_gets_zeros_not_the_template(self):
        algo = self._algo()
        _, hid, _ = algo._exit_cotangent_grads(self._template(algo), {})
        assert jnp.all(hid[('h',)] == 0.0)

    def test_an_unknown_path_is_refused(self):
        algo = self._algo()
        with pytest.raises(ValueError, match='not hidden states'):
            algo._exit_cotangent_grads(
                self._template(algo), {('nope',): jnp.zeros((1, 4))})

    def test_a_shape_mismatch_is_refused_naming_both_shapes(self):
        algo = self._algo()
        with pytest.raises(ValueError) as exc:
            algo._exit_cotangent_grads(
                self._template(algo), {('h',): jnp.zeros((1, 9))})
        assert '(1, 9)' in str(exc.value) and '(1, 4)' in str(exc.value)


# ---------------------------------------------------------------------------
# E-01: the hidden <-> gradient correspondence is checked, not asserted
#
# The backward pass hands back a collection of hidden-state cotangents that is
# then re-ordered onto the hidden groups by path and concatenated. Nothing
# downstream re-derives which hidden state a cotangent came from, so a
# mis-ordered or mis-shaped collection produces a *wrong gradient rather than an
# error*. The guards used to be three `assert` statements checking cardinalities
# and (in the multi-step branch) a key set -- never a correspondence -- and
# `python -O` stripped all three.
# ---------------------------------------------------------------------------

import os
import pathlib
import subprocess
import sys
import textwrap

from braintrace._algorithm.vjp_base import _check_hidden_gradient_correspondence


def _e01_groups(n_rec=4):
    """Real compiler output: the hidden groups of a one-layer GRU."""
    gru = braintrace.nn.GRUCell(3, n_rec)
    brainstate.nn.init_all_states(gru)
    groups, _ = braintrace.find_hidden_groups_from_module(
        gru, brainstate.random.randn(3))
    return list(groups)


def _e01_well_formed(groups):
    """One correctly shaped, correctly typed cotangent per hidden state."""
    return {
        path: jnp.zeros(
            tuple(state.varshape),
            dtype=u.get_mantissa(state.value).dtype,
        )
        for group in groups
        for path, state in zip(group.hidden_paths, group.hidden_states)
    }


class TestHiddenGradientCorrespondence:
    """The helper accepts what the compiler produces, and nothing else."""

    def test_a_well_formed_mapping_is_accepted(self):
        groups = _e01_groups()
        # returns None, raises nothing
        assert _check_hidden_gradient_correspondence(
            groups, _e01_well_formed(groups), source='unit test') is None

    def test_a_missing_path_is_refused_and_named(self):
        groups = _e01_groups()
        mapping = _e01_well_formed(groups)
        dropped = next(iter(mapping))
        del mapping[dropped]

        with pytest.raises(ValueError) as exc:
            _check_hidden_gradient_correspondence(
                groups, mapping, source='unit test')
        message = str(exc.value)
        assert str(dropped) in message
        assert 'unit test' in message

    def test_an_extra_path_is_refused_and_named(self):
        groups = _e01_groups()
        mapping = _e01_well_formed(groups)
        mapping[('a', 'stray', 'path')] = jnp.zeros((4,))

        with pytest.raises(ValueError) as exc:
            _check_hidden_gradient_correspondence(
                groups, mapping, source='unit test')
        message = str(exc.value)
        assert str(('a', 'stray', 'path')) in message
        assert 'no hidden group claims' in message

    def test_a_shape_mismatch_is_refused_naming_both_shapes(self):
        groups = _e01_groups(n_rec=4)
        mapping = _e01_well_formed(groups)
        path = groups[0].hidden_paths[0]
        mapping[path] = jnp.zeros((9,))

        with pytest.raises(ValueError) as exc:
            _check_hidden_gradient_correspondence(
                groups, mapping, source='unit test')
        message = str(exc.value)
        assert str(path) in message
        assert '(9,)' in message       # what arrived
        assert '(4,)' in message       # what the hidden state wants

    def test_a_dtype_mismatch_is_refused_naming_both_dtypes(self):
        groups = _e01_groups()
        mapping = _e01_well_formed(groups)
        path = groups[0].hidden_paths[0]
        state = groups[0].hidden_states[0]
        assert u.get_mantissa(state.value).dtype == jnp.float32
        mapping[path] = jnp.zeros(tuple(state.varshape), dtype=jnp.int32)

        with pytest.raises(ValueError) as exc:
            _check_hidden_gradient_correspondence(
                groups, mapping, source='unit test')
        message = str(exc.value)
        assert str(path) in message
        assert 'int32' in message
        assert 'float32' in message

    def test_the_message_names_the_group_the_position_and_the_source(self):
        groups = _e01_groups()
        mapping = _e01_well_formed(groups)
        path = groups[0].hidden_paths[0]
        mapping[path] = jnp.zeros((9,))

        with pytest.raises(ValueError) as exc:
            _check_hidden_gradient_correspondence(
                groups, mapping, source='multi-step last-hidden gradients')
        message = str(exc.value)
        assert f'hidden group {groups[0].index}' in message
        assert 'position 0' in message
        assert 'multi-step last-hidden gradients' in message


# The script the `-O` subprocess runs. It has to be self-contained: the point is
# that a *fresh optimised interpreter* still refuses a mis-shaped cotangent.
_E01_DASH_O_SCRIPT = textwrap.dedent(
    """
    import sys

    # Prove the interpreter really is optimised before proving anything else:
    # under -O, `__debug__` is False and `assert` compiles to nothing.
    if __debug__:
        print('NOT-OPTIMISED')
        sys.exit(2)
    assert False, 'this assert must have been stripped'

    import brainstate
    import brainunit as u
    import jax.numpy as jnp
    import braintrace
    from braintrace._algorithm.vjp_base import _check_hidden_gradient_correspondence

    gru = braintrace.nn.GRUCell(3, 4)
    brainstate.nn.init_all_states(gru)
    groups, _ = braintrace.find_hidden_groups_from_module(
        gru, brainstate.random.randn(3))
    groups = list(groups)

    mapping = {
        path: jnp.zeros(tuple(state.varshape),
                        dtype=u.get_mantissa(state.value).dtype)
        for group in groups
        for path, state in zip(group.hidden_paths, group.hidden_states)
    }

    # 1. the well-formed mapping is still accepted
    _check_hidden_gradient_correspondence(groups, mapping, source='dash-O probe')

    # 2. a mis-shaped cotangent is still refused
    mapping[groups[0].hidden_paths[0]] = jnp.zeros((9,))
    try:
        _check_hidden_gradient_correspondence(
            groups, mapping, source='dash-O probe')
    except ValueError as e:
        if 'not in correspondence' not in str(e):
            print('WRONG-MESSAGE:' + str(e))
            sys.exit(3)
    else:
        print('NOT-RAISED')
        sys.exit(4)

    # 3. and so is a short value list to concat_hidden -- the zip truncation
    try:
        groups[0].concat_hidden([])
    except ValueError:
        pass
    else:
        print('CONCAT-NOT-RAISED')
        sys.exit(5)

    print('GUARDS-SURVIVED-DASH-O')
    """
)


class TestTheGuardsSurviveDashO:
    """E-01's core complaint: `python -O` strips `assert`, so these must not be.

    Without this test the fix could silently regress to `assert` statements and
    every other test in this file would still pass, because pytest runs with
    assertions enabled.
    """

    def test_a_mis_shaped_cotangent_is_still_refused_under_dash_O(self):
        # Import the package this test imported, not whatever is installed:
        # `braintrace/__init__.py` lives one level below the path we want on
        # `sys.path`.
        package_root = str(pathlib.Path(braintrace.__file__).resolve().parent.parent)
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(
            [package_root] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
        )
        env.pop('PYTHONOPTIMIZE', None)

        proc = subprocess.run(
            [sys.executable, '-O', '-c', _E01_DASH_O_SCRIPT],
            capture_output=True, text=True, env=env, cwd=package_root, timeout=900,
        )

        assert proc.returncode == 0, (
            f'-O subprocess failed (returncode={proc.returncode})\n'
            f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}'
        )
        assert 'GUARDS-SURVIVED-DASH-O' in proc.stdout, proc.stdout


class TestGradientsAreUnchangedByTheGuards:
    """Positive control: the guards accept real compiler output, and the
    gradients they now stand in front of are the ones they always were."""

    def test_multistep_gradients_still_match_bptt(self):
        from braintrace._testing import oracle
        from braintrace._testing import oracle_models as om

        spec = om.tanh_rnn(n_in=3, n_rec=4)
        inputs = brainstate.random.randn(8, 3)
        bptt = oracle.bptt_param_gradients(spec.factory, inputs)
        got = oracle.online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        )
        oracle.assert_param_gradients_close(
            got, bptt, atol=1e-5, rtol=1e-5, keys=spec.etp_param_keys)

    @pytest.mark.parametrize('cls', [braintrace.nn.GRUCell, braintrace.nn.LSTMCell])
    def test_the_single_step_branch_accepts_the_compiled_graph(self, cls):
        """The single-step branch is the one that reads the perturbation vars.

        ``LSTMCell`` matters here: two hidden states, so the per-position
        correspondence has something to get wrong.
        """
        model = cls(3, 4)
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='single-step')
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        params = model.states(brainstate.ParamState)
        grads = brainstate.transform.grad(
            lambda inp: (algo(inp) ** 2).sum(), params)(x)

        leaves = jax.tree.leaves(grads)
        assert leaves
        assert all(bool(jnp.all(jnp.isfinite(u.get_mantissa(v)))) for v in leaves)
