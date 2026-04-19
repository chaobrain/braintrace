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
from braintrace._snn_algorithms.osttp import OSTTP


def _osttp_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.h.value = jax.nn.tanh(
                braintrace.matmul(self.h.value + x, self.w.value)
            )
            return self.h.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


class TestOSTTPConstruction(unittest.TestCase):
    def test_default_target_timing(self):
        B_list = [0.1 * jax.random.normal(jax.random.PRNGKey(1), (5, 3))]
        algo = OSTTP(_osttp_net(), B_list=B_list)
        assert algo.target_timing == 'per-step'

    def test_invalid_timing_raises(self):
        B_list = [0.1 * jax.random.normal(jax.random.PRNGKey(1), (5, 3))]
        with self.assertRaises(ValueError):
            OSTTP(_osttp_net(), B_list=B_list, target_timing='never')

    def test_missing_B_list_raises(self):
        with self.assertRaises(TypeError):
            OSTTP(_osttp_net())

    def test_B_list_wrong_length_raises_on_compile(self):
        B_list = [
            jax.random.normal(jax.random.PRNGKey(1), (5, 3)),
            jax.random.normal(jax.random.PRNGKey(2), (5, 3)),
        ]
        net = _osttp_net()
        algo = OSTTP(net, B_list=B_list)
        x = jnp.ones((1, 3))
        with self.assertRaises(ValueError):
            algo.compile_graph(x)

    def test_B_list_wrong_shape_raises_on_compile(self):
        B_list = [jax.random.normal(jax.random.PRNGKey(1), (5, 99))]
        net = _osttp_net()
        algo = OSTTP(net, B_list=B_list)
        x = jnp.ones((1, 3))
        with self.assertRaises(ValueError):
            algo.compile_graph(x)


class TestOSTTPTargetProjection(unittest.TestCase):
    def test_target_differs_from_symmetric(self):
        key = jax.random.PRNGKey(7)
        B = 0.1 * jax.random.normal(key, (4, 3))
        net = _osttp_net()
        algo = OSTTP(net, B_list=[B])
        x = jnp.ones((1, 3))
        y = jnp.ones((1, 4))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x_):
            out = algo.update(x_, y_target=y)
            return (out ** 2).sum()

        grads, _ = brainstate.augment.grad(
            loss, algo.param_states, return_value=True
        )(x)
        g_osttp = grads[next(iter(grads))]

        from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm
        net2 = _osttp_net()
        algo2 = ParamDimVjpAlgorithm(net2)
        algo2.compile_graph(x)
        algo2.init_etrace_state()

        def loss2(x_):
            out = algo2.update(x_)
            return (out ** 2).sum()

        grads2, _ = brainstate.augment.grad(
            loss2, algo2.param_states, return_value=True
        )(x)
        g_drtrl = grads2[next(iter(grads2))]
        assert not jnp.allclose(g_osttp, g_drtrl, atol=1e-4)
