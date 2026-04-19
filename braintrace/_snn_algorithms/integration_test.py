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

"""End-to-end smoke: each of the 5 algos drives a small toy task with decreasing loss."""

import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._snn_algorithms import EProp, OSTL, OSTTP, OTPE, OTTT


class FakeLIF(brainstate.HiddenState):
    def __init__(self, iv, leak):
        super().__init__(iv)
        self.leak = leak


def _toy_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = FakeLIF(jnp.zeros((1, 3)), leak=0.9)

        def update(self, x):
            self.v.value = jax.nn.tanh(
                0.9 * self.v.value + braintrace.matmul(x, self.w.value)
            )
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _run(algo, n_steps=10, lr=0.05, y_target=None, pass_y=False):
    x = jnp.ones((1, 3))
    algo.compile_graph(x)
    algo.init_etrace_state()

    losses = []
    for _ in range(n_steps):
        def loss_fn(x_):
            out = algo.update(x_, y_target=y_target) if pass_y else algo.update(x_)
            target = jnp.ones_like(out)
            return ((out - target) ** 2).mean()

        grads, loss_val = brainstate.augment.grad(
            loss_fn, algo.param_states, return_value=True
        )(x)
        for path, st in algo.param_states.items():
            st.value = st.value - lr * grads[path]
        losses.append(float(loss_val))
    return losses


class TestSmokeLossDecreases(unittest.TestCase):
    def test_eprop(self):
        losses = _run(EProp(_toy_net()))
        assert losses[-1] < losses[0]

    def test_ostl(self):
        losses = _run(OSTL(_toy_net()))
        assert losses[-1] < losses[0]

    def test_otpe(self):
        losses = _run(OTPE(_toy_net()))
        assert losses[-1] < losses[0]

    def test_ottt(self):
        losses = _run(OTTT(_toy_net()))
        assert losses[-1] < losses[0]

    def test_osttp(self):
        net = _toy_net()
        B = [0.1 * jax.random.normal(jax.random.PRNGKey(9), (3, 3))]
        algo = OSTTP(net, B_list=B)
        y = jnp.ones((1, 3))
        losses = _run(algo, y_target=y, pass_y=True)
        assert losses[-1] < losses[0]
