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
import jax.numpy as jnp
import brainunit as u

import braintrace
from braintrace._etrace_model_test import (
    ALIF_STPExpCu_Dense_Layer,
)


class TestShowGraph(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        brainstate.environ.set(dt=0.1 * u.ms)

    def test_show_lstm_graph(self):
        cell = braintrace.nn.LSTMCell(10, 20, activation=jnp.tanh)
        brainstate.nn.init_all_states(cell, 16)

        graph = braintrace.ETraceGraphExecutor(cell)
        graph.compile_graph(jnp.zeros((16, 10)))
        graph.show_graph()

    def test_show_gru_graph(self):
        cell = braintrace.nn.GRUCell(10, 20, activation=jnp.tanh)
        brainstate.nn.init_all_states(cell, 16)

        graph = braintrace.ETraceGraphExecutor(cell)
        graph.compile_graph(jnp.zeros((16, 10)))
        graph.show_graph()

    def test_show_lru_graph(self):
        cell = braintrace.nn.LRUCell(10, 20)
        brainstate.nn.init_all_states(cell)

        graph = braintrace.ETraceGraphExecutor(cell)
        graph.compile_graph(jnp.zeros((10,)))
        graph.show_graph()

    def test_show_alig_stp_graph(self):
        n_in = 3
        n_rec = 4

        net = ALIF_STPExpCu_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        graph = braintrace.ETraceGraphExecutor(net)
        graph.compile_graph(brainstate.random.rand(n_in))
        graph.show_graph()


class TestControlFlowThreading(unittest.TestCase):
    def test_control_flow_policy_reaches_compiled_module_info(self):
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((3, 3)) * 0.1)
                self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

            def update(self, x):
                self.h.value = 0.9 * self.h.value + braintrace.matmul(
                    x.reshape(1, -1), self.w.value)
                return self.h.value

        policy = braintrace.ControlFlowPolicy(scan_unroll_limit=5)
        model = Net()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, control_flow=policy)
        algo.compile_graph(jnp.ones((3,), dtype='float32'))
        assert algo.graph.module_info.control_flow is policy
