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
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braintrace


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
        with self.assertRaises(AssertionError):
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

    def test_solve_h2w_h2h_jacobian(self):
        executor = braintrace.ETraceVjpGraphExecutor(self.model)
        x = jnp.ones((self.in_size,))
        executor.compile_graph(x)

        outputs, etrace_vals, state_vals, h2w_jacobian, h2h_jacobian = executor.solve_h2w_h2h_jacobian(x)

        self.assertIsInstance(outputs, jax.Array)
        self.assertIsInstance(etrace_vals, dict)
        self.assertIsInstance(state_vals, dict)
        self.assertIsInstance(h2w_jacobian, tuple)
        self.assertIsInstance(h2h_jacobian, list)

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


if __name__ == '__main__':
    unittest.main()
