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

import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

import braintrace
from braintrace._compiler.hid_param_op import PathClassification
from braintrace._testing.scenario_catalog import PartialPathRNN


class TestETraceVjpGraphExecutor(unittest.TestCase):

    @property
    def in_size(self):
        return 3

    def setUp(self):
        self.model = braintrace.nn.GRUCell(self.in_size, 4)
        brainstate.nn.init_all_states(self.model)
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_initialization(self):
        executor = braintrace.ETraceVjpGraphExecutor(self.model)
        self.assertEqual(executor.vjp_method, 'single-step')

        executor = braintrace.ETraceVjpGraphExecutor(self.model, vjp_method='multi-step')
        self.assertEqual(executor.vjp_method, 'multi-step')

    def test_invalid_vjp_method(self):
        with self.assertRaisesRegex(ValueError, 'single-step.*multi-step'):
            braintrace.ETraceVjpGraphExecutor(self.model, vjp_method='invalid')

    def test_is_single_step_vjp(self):
        executor = braintrace.ETraceVjpGraphExecutor(self.model)
        self.assertTrue(executor.is_single_step_vjp)
        self.assertFalse(executor.is_multi_step_vjp)

    def test_is_multi_step_vjp(self):
        executor = braintrace.ETraceVjpGraphExecutor(self.model, vjp_method='multi-step')
        self.assertFalse(executor.is_single_step_vjp)
        self.assertTrue(executor.is_multi_step_vjp)

    def test_compile_graph(self):
        executor = braintrace.ETraceVjpGraphExecutor(self.model)
        x = jnp.ones((self.in_size,))
        executor.compile_graph(x)
        self.assertIsNotNone(executor._compiled_graph)

    def test_mixed_relation_is_inspectable_but_not_executable(self):
        model = PartialPathRNN(3, 4)
        brainstate.nn.init_all_states(model)
        x = jnp.ones(3)
        graph = braintrace.compile_etrace_graph(model, x)
        by_path = {
            relation.trainable_paths['weight']: relation
            for relation in graph.hidden_param_op_relations
        }
        self.assertEqual(
            by_path[('w1',)].path_classification,
            {('h',): PathClassification.MIXED},
        )
        executor = braintrace.ETraceVjpGraphExecutor(model)
        with self.assertRaisesRegex(
            braintrace.NotSupportedError,
            'both a direct path and an indirect path',
        ):
            executor.compile_graph(x)
        with self.assertRaisesRegex(ValueError, 'not compiled'):
            _ = executor.graph

    def test_independent_direct_relations_may_share_one_parameter(self):
        from braintrace._testing import oracle_models as om

        spec = om.tied_weight_rnn(n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        executor = braintrace.ETraceVjpGraphExecutor(model)
        executor.compile_graph(spec.make_inputs(1, 4)[0])
        relations = executor.graph.hidden_param_op_relations
        self.assertEqual(len(relations), 2)
        self.assertTrue(all(
            relation.trainable_paths['weight'] == ('w',)
            and relation.path_classification == {
                ('h',): PathClassification.ALL_DIRECT
            }
            for relation in relations
        ))

    def test_solve_h2w_h2h_jacobian(self):
        executor = braintrace.ETraceVjpGraphExecutor(self.model)
        x = jnp.ones((self.in_size,))
        executor.compile_graph(x)

        outputs, etrace_vals, state_vals, h2w_jacobian, h2h_jacobian, final_etrace = (
            executor.solve_h2w_h2h_jacobian(x)
        )

        self.assertIsInstance(outputs, jax.Array)
        self.assertIsInstance(etrace_vals, dict)
        self.assertIsInstance(state_vals, dict)
        self.assertIsInstance(h2w_jacobian, tuple)
        self.assertIsInstance(h2h_jacobian, list)
        # No stepper passed -> legacy return, the fused-trace slot is None.
        self.assertIsNone(final_etrace)

    def test_single_step_vs_multi_step(self):
        single_step_executor = braintrace.ETraceVjpGraphExecutor(self.model, vjp_method='single-step')
        multi_step_executor = braintrace.ETraceVjpGraphExecutor(self.model, vjp_method='multi-step')

        x = jnp.ones((self.in_size,))
        single_step_executor.compile_graph(x)
        multi_step_executor.compile_graph(x)

        single_result = single_step_executor.solve_h2w_h2h_jacobian(x)
        multi_result = multi_step_executor.solve_h2w_h2h_jacobian(x)

        # Check that the outputs are the same
        np.testing.assert_allclose(single_result[0], multi_result[0])

        # Check that the etrace_vals and state_vals are the same
        self.assertEqual(set(single_result[1].keys()), set(multi_result[1].keys()))
        self.assertEqual(set(single_result[2].keys()), set(multi_result[2].keys()))

        # The Jacobians might differ due to the different methods
        # self.assertNotEqual(single_result[3], multi_result[3])
        # self.assertNotEqual(single_result[4], multi_result[4])


class TestFullJacobianFlag(unittest.TestCase):
    """The ``full_jacobian`` constructor flag added for ``random_projection``.

    The flag exists because a rank-1 estimator only pays for itself against the
    *full* within-group transition: rolling the per-position block diagonal
    instead converges onto the biased trace, at strictly more variance.
    """

    def _executor(self, **kwargs):
        from braintrace._testing import oracle_models as om
        from braintrace._algorithm.vjp_graph_executor import ETraceVjpGraphExecutor
        spec = om.stacked_tanh_rnn(n_in=4, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        return ETraceVjpGraphExecutor(model, vjp_method='multi-step', **kwargs), spec

    def test_the_flag_defaults_to_off(self):
        executor, _ = self._executor()
        self.assertFalse(executor.full_jacobian)

    def test_the_flag_changes_the_jacobian_shape(self):
        # Diagonal: one entry per hidden position. Full: the whole
        # (*V, S, *V, S) block, which is what the coupled recursion needs.
        diag, spec = self._executor(include_recurrent_mixing=True)
        full, _ = self._executor(include_recurrent_mixing=True, full_jacobian=True)
        x = spec.make_inputs(1, 4)[0]
        diag.compile_graph(x)
        full.compile_graph(x)
        d_jac = diag.solve_h2w_h2h_jacobian(x)[4]
        f_jac = full.solve_h2w_h2h_jacobian(x)[4]
        self.assertEqual(len(d_jac), len(f_jac))
        for group, d, f in zip(full.graph.hidden_groups, d_jac, f_jac):
            slab = tuple(group.varshape) + (group.num_state,)
            # Diagonal: one state-by-state block per position, `(*V, S, S)`.
            # Full: every position pair, `(*V, S, *V, S)`. The difference is
            # exactly the cross-position mixing a coupled recursion needs.
            self.assertEqual(u.math.shape(d), slab + (group.num_state,))
            self.assertEqual(u.math.shape(f), slab + slab)
            self.assertNotEqual(u.math.shape(d), u.math.shape(f))

    def test_coupled_compile_rejects_an_oversized_jacobian(self):
        executor, spec = self._executor(
            include_recurrent_mixing=True,
        )
        inputs = spec.make_inputs(1, 4)[0]
        executor.compile_graph(inputs)
        executor.snap_max_jacobian_elements = 1
        with self.assertRaisesRegex(braintrace.NotSupportedError, '1'):
            executor.compile_graph(inputs)
        with self.assertRaisesRegex(ValueError, 'not compiled'):
            _ = executor.graph

    def test_full_jacobian_compile_rejects_an_oversized_jacobian(self):
        executor, spec = self._executor(
            full_jacobian=True,
            snap_max_jacobian_elements=1,
        )
        with self.assertRaisesRegex(braintrace.NotSupportedError, '1'):
            executor.compile_graph(spec.make_inputs(1, 4)[0])
        with self.assertRaisesRegex(ValueError, 'not compiled'):
            _ = executor.graph

    def test_the_random_projection_engine_turns_the_flag_on(self):
        # The wiring that matters: the coordinate, not the caller, selects it.
        from braintrace._testing import oracle_models as om
        spec = om.tanh_rnn(n_in=3, n_rec=4)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        self.assertTrue(
            braintrace.UORO(model, vjp_method='multi-step')
            .graph_executor.full_jacobian)

        model2 = spec.factory()
        brainstate.nn.init_all_states(model2, batch_size=1)
        self.assertFalse(
            braintrace.D_RTRL(model2, vjp_method='multi-step')
            .graph_executor.full_jacobian)


if __name__ == '__main__':
    unittest.main()
