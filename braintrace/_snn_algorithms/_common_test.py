# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

from braintrace._snn_algorithms._common import PresynapticTrace


class TestPresynapticTrace(unittest.TestCase):
    def test_exponential_accumulation(self):
        trace = PresynapticTrace(jnp.zeros((2, 3)), leak=0.9)
        trace.update(jnp.ones((2, 3)))
        trace.update(jnp.ones((2, 3)))
        # After 2 ones: 0.9*(0.9*0 + 1) + 1 == 1.9
        assert jnp.allclose(trace.value, jnp.full((2, 3), 1.9))

    def test_reset_to_zero(self):
        trace = PresynapticTrace(jnp.ones((4,)), leak=0.5)
        trace.reset_state()
        assert jnp.allclose(trace.value, jnp.zeros((4,)))
