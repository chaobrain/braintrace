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
from braintrace._etrace_algorithms.ottt import OTTT


class FakeLIF(brainstate.HiddenState):
    """HiddenState with a `leak` attribute for _resolve_leak discovery."""

    def __init__(self, init_value, leak):
        super().__init__(init_value)
        self.leak = leak


def _ottt_net():
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


class TestOTTTConstruction(unittest.TestCase):
    def test_default_mode_is_A(self):
        algo = OTTT(_ottt_net(), leak=0.9)
        assert algo.mode == 'A'

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            OTTT(_ottt_net(), mode='bogus', leak=0.9)

    def test_leak_discovered_from_model(self):
        """_resolve_leak picks up FakeLIF.leak when not explicitly provided."""
        algo = OTTT(_ottt_net())
        assert algo.leak == 0.9

    def test_missing_leak_raises(self):
        class BareNet(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.eye(3))
                self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

            def update(self, x):
                self.h.value = braintrace.matmul(x, self.w.value)
                return self.h.value

        net = BareNet()
        brainstate.nn.init_all_states(net, batch_size=1)
        with self.assertRaises(ValueError):
            OTTT(net)

    def test_compile_allocates_presynaptic_traces(self):
        algo = OTTT(_ottt_net(), leak=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._pre_traces) == len(algo.graph.hidden_param_op_relations)


class TestOTTTEnd2End(unittest.TestCase):
    def test_update_runs_and_produces_gradients(self):
        net = _ottt_net()
        algo = OTTT(net, leak=0.9)
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

    def test_mode_O_differs_from_mode_A(self):
        def compute(mode):
            net = _ottt_net()
            algo = OTTT(net, mode=mode, leak=0.9)
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()
            algo.update(x)

            def loss(x_):
                return (algo.update(x_) ** 2).sum()

            grads, _ = brainstate.augment.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g_A = compute('A')
        g_O = compute('O')
        assert not jnp.allclose(g_A, g_O, atol=1e-4)
