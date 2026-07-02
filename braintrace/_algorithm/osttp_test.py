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
from braintrace._algorithm.osttp import OSTTP


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

        grads, _ = brainstate.transform.grad(
            loss, algo.param_states, return_value=True
        )(x)
        g_osttp = grads[next(iter(grads))]

        from braintrace._algorithm.param_dim_vjp import ParamDimVjpAlgorithm
        net2 = _osttp_net()
        algo2 = ParamDimVjpAlgorithm(net2)
        algo2.compile_graph(x)
        algo2.init_etrace_state()

        def loss2(x_):
            out = algo2.update(x_)
            return (out ** 2).sum()

        grads2, _ = brainstate.transform.grad(
            loss2, algo2.param_states, return_value=True
        )(x)
        g_drtrl = grads2[next(iter(grads2))]
        assert not jnp.allclose(g_osttp, g_drtrl, atol=1e-4)


class TestOSTTPLearningSignalNonzero(unittest.TestCase):
    """Regression test for the zero-learning-signal bug (C4).

    OSTTP used to stash ``y_target`` on the instance during the forward
    ``update()`` and clear it in a ``finally`` block before returning. The
    learning-signal hook (``_compute_learning_signal``) only runs later,
    inside the ``jax.custom_vjp`` backward pass -- by then the stash was
    already ``None``, so the target-projected learning signal, and hence the
    weight gradient, was identically zero regardless of ``y_target``.
    """

    def _grad_for_target(self, y_target: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        net = _osttp_net()
        algo = OSTTP(net, B_list=[B])
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x_):
            out = algo.update(x_, y_target=y_target)
            return (out ** 2).sum()

        grads, _ = brainstate.transform.grad(
            loss, algo.param_states, return_value=True
        )(x)
        return grads[next(iter(grads))]

    def test_gradients_depend_on_target(self):
        B = 0.1 * jax.random.normal(jax.random.PRNGKey(11), (4, 3))
        y1 = jnp.ones((1, 4))
        y2 = -5.0 * jnp.ones((1, 4))

        g1 = self._grad_for_target(y1, B)
        g2 = self._grad_for_target(y2, B)

        # Different targets must yield different weight gradients -- if the
        # learning signal were identically zero (the bug), g1 == g2 (both
        # all-zero).
        assert not jnp.allclose(g1, g2, atol=1e-6)
        assert not jnp.allclose(g1, jnp.zeros_like(g1), atol=1e-8)
        assert not jnp.allclose(g2, jnp.zeros_like(g2), atol=1e-8)


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

        grads, loss_val = brainstate.transform.grad(
            loss_fn, algo.param_states, return_value=True
        )(x)
        for path, st in algo.param_states.items():
            st.value = st.value - lr * grads[path]
        losses.append(float(loss_val))
    return losses


class TestSmokeLossDecreases(unittest.TestCase):
    def test_osttp(self):
        net = _toy_net()
        B = [0.1 * jax.random.normal(jax.random.PRNGKey(9), (3, 3))]
        algo = OSTTP(net, B_list=B)
        y = jnp.ones((1, 3))
        losses = _run(algo, y_target=y, pass_y=True)
        assert losses[-1] < losses[0]


def _docstring_net():
    """The exact ``Net`` model used in the ``OSTTP`` docstring example."""

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
            self.out = braintrace.nn.Linear(20, 1)

        def update(self, x):
            return x >> self.cell >> self.out

    return Net()


def test_docstring_compile_example_runs():
    """Verify the runnable ``braintrace.compile`` example in ``OSTTP``'s docstring."""
    model = _docstring_net()
    x0 = brainstate.random.randn(1)
    # one (n_target, n_l) feedback matrix per HiddenGroup (here n_l = 20)
    B = jax.random.normal(jax.random.PRNGKey(0), (1, 20))
    learner = braintrace.compile(model, braintrace.OSTTP, x0, B_list=[B])
    y = learner.update(x0, y_target=brainstate.random.randn(1))
    assert y.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert len(learner.graph.hidden_param_op_relations) >= 1
