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
from braintrace._algorithm.ottt import OTTT


def _ottt_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            self.v.value = 0.9 * self.v.value + braintrace.matmul(x, self.w.value)
            return self.v.value

    net = Net()
    brainstate.nn.init_all_states(net, batch_size=1)
    return net


def _two_state_net():
    """Net whose two coupled hidden states form one group with num_state == 2."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))
            self.a = brainstate.HiddenState(jnp.zeros((1, 3)))

        def update(self, x):
            v, a = self.v.value, self.a.value
            self.v.value = 0.9 * v + braintrace.matmul(x, self.w.value) - 0.1 * a
            self.a.value = 0.95 * a + v
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

    def test_leak_is_required(self):
        """leak is never inferred from the model; omitting it is a TypeError."""
        with self.assertRaises(TypeError):
            OTTT(_ottt_net())

    def test_leak_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            OTTT(_ottt_net(), leak=1.5)

    def test_num_state_gt_one_raises(self):
        """Multi-state hidden groups have no theoretical basis for OTTT."""
        algo = OTTT(_two_state_net(), leak=0.9)
        with self.assertRaises(ValueError):
            algo.compile_graph(jnp.ones((1, 3)))

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

        grads, _ = brainstate.transform.grad(
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

            grads, _ = brainstate.transform.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g_A = compute('A')
        g_O = compute('O')
        assert not jnp.allclose(g_A, g_O, atol=1e-4)


def _toy_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = brainstate.HiddenState(jnp.zeros((1, 3)))

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

        grads, loss_val = brainstate.transform.grad(
            loss_fn, algo.param_states, return_value=True
        )(x)
        for path, st in algo.param_states.items():
            st.value = st.value - lr * grads[path]
        losses.append(float(loss_val))
    return losses


class TestSmokeLossDecreases(unittest.TestCase):
    def test_ottt(self):
        losses = _run(OTTT(_toy_net(), leak=0.9))
        assert losses[-1] < losses[0]


def _docstring_net():
    """The exact ``Net`` model used in the ``OTTT`` docstring example."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
            self.out = braintrace.nn.Linear(20, 1)

        def update(self, x):
            return x >> self.cell >> self.out

    return Net()


def test_docstring_compile_example_runs():
    """Verify the runnable ``braintrace.compile`` example in ``OTTT``'s docstring."""
    model = _docstring_net()
    x0 = brainstate.random.randn(1)
    learner = braintrace.compile(model, braintrace.OTTT, x0, mode='A', leak=0.9)
    y = learner(x0)
    assert y.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert len(learner.graph.hidden_param_op_relations) >= 1
