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

import braintrace
from braintrace._snn_algorithms.ostl import OSTL


def _tiny_rec_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w_rec = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.h.value = jax.nn.tanh(
                braintrace.matmul(self.h.value + x, self.w_rec.value)
            )
            return self.h.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


class TestOSTLConstruction(unittest.TestCase):
    def test_default_regime_is_with_h(self):
        algo = OSTL(_tiny_rec_net())
        assert algo.regime == 'with-H'

    def test_invalid_regime_raises(self):
        with self.assertRaises(ValueError):
            OSTL(_tiny_rec_net(), regime='bogus')

    def test_with_h_compiles_and_updates(self):
        net = _tiny_rec_net()
        algo = OSTL(net, regime='with-H')
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        out = algo.update(x)
        assert out.shape == (1, 3)

    def test_without_h_compiles_and_updates(self):
        net = _tiny_rec_net()
        algo = OSTL(net, regime='without-H')
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        out = algo.update(x)
        assert out.shape == (1, 3)
