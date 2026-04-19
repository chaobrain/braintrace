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
from braintrace._snn_algorithms.otpe import OTPE


class FakeLIF(brainstate.HiddenState):
    def __init__(self, init_value, leak):
        super().__init__(init_value)
        self.leak = leak


def _otpe_net_single_layer():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = FakeLIF(jnp.zeros((1, 3)), leak=0.9)

        def update(self, x):
            self.v.value = 0.9 * self.v.value + braintrace.matmul(x, self.w.value)
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


class TestOTPEConstruction(unittest.TestCase):
    def test_default_mode_full(self):
        algo = OTPE(_otpe_net_single_layer(), leak=0.9)
        assert algo.mode == 'full'

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            OTPE(_otpe_net_single_layer(), mode='bogus', leak=0.9)

    def test_leak_resolved_from_model(self):
        algo = OTPE(_otpe_net_single_layer())
        assert algo.leak == 0.9

    def test_compile_allocates_R_hat(self):
        algo = OTPE(_otpe_net_single_layer(), leak=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._R_hat) == len(algo.graph.hidden_param_op_relations)


class TestOTPESingleLayer(unittest.TestCase):
    def test_update_runs_and_produces_gradients(self):
        net = _otpe_net_single_layer()
        algo = OTPE(net, leak=0.9, mode='full')
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x_):
            return (algo.update(x_) ** 2).sum()

        grads, _ = brainstate.augment.grad(
            loss, algo.param_states, return_value=True
        )(x)
        g = grads[next(iter(grads))]
        assert g.shape == (3, 3)
        assert jnp.any(g != 0.0)


class TestOTPEApproxMode(unittest.TestCase):
    def test_approx_uses_factored_traces(self):
        net = _otpe_net_single_layer()
        algo = OTPE(net, mode='approx', leak=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._R_hat_x) == len(algo.graph.hidden_param_op_relations)
        assert len(algo._R_hat_g) == len(algo.graph.hidden_param_op_relations)

    def test_approx_runs(self):
        net = _otpe_net_single_layer()
        algo = OTPE(net, mode='approx', leak=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x_):
            return (algo.update(x_) ** 2).sum()

        grads, _ = brainstate.augment.grad(
            loss, algo.param_states, return_value=True
        )(x)
        g = grads[next(iter(grads))]
        assert g.shape == (3, 3)
