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

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._input_data import _count_update_steps


class TestEtraceInputData(unittest.TestCase):

    def test_jittable(self):
        @jax.jit
        def f(x):
            return x.data ** 2

        f(braintrace.SingleStepData(3))
        f(braintrace.MultiStepData(3))
        f(braintrace.SingleStepData(brainstate.random.rand(10)))
        f(braintrace.MultiStepData(brainstate.random.rand(10)))

    def test_grad(self):
        def f(x):
            return x.data ** 2

        y, grad = jax.value_and_grad(f)(braintrace.SingleStepData(3.))
        self.assertEqual(y, 9)
        self.assertEqual(grad.data, 6)

    def test_grad2(self):
        def f(x):
            return x.data ** 2

        y, grad = jax.value_and_grad(f)(braintrace.MultiStepData(3.))
        self.assertEqual(y, 9)
        self.assertEqual(grad.data, 6)


class TestCountUpdateSteps:

    def test_plain_input_is_one_step(self):
        assert _count_update_steps(jnp.ones((4,))) == 1

    def test_multistep_pytree_uses_leading_dimension(self):
        data = {'a': jnp.ones((6, 2)), 'b': (jnp.ones((6, 3)),)}
        assert _count_update_steps(braintrace.MultiStepData(data)) == 6

    def test_multiple_multistep_inputs_must_agree(self):
        with pytest.raises(ValueError, match='same sequence size'):
            _count_update_steps(
                braintrace.MultiStepData(jnp.ones((3, 2))),
                braintrace.MultiStepData(jnp.ones((4, 2))),
            )

    def test_empty_multistep_input_is_rejected(self):
        with pytest.raises(ValueError, match='at least one timestep'):
            _count_update_steps(braintrace.MultiStepData(jnp.ones((0, 2))))
