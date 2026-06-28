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
from braintrace._algorithm.param_dim_vjp import ParamDimVjpAlgorithm
from braintrace._algorithm.ostl import (
    OSTLFeedforward,
    OSTLRecurrent,
)
from braintrace._algorithm.pp_prop import pp_prop


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


def _drive(algo, n=3, x=None):
    """Compile, init, and run ``n`` update steps; return the algo and last out."""
    x = jnp.ones((1, 3)) if x is None else x
    algo.compile_graph(x)
    algo.init_etrace_state()
    out = None
    for _ in range(n):
        out = algo.update(x)
    return algo, out


def _train_loss(algo, n_steps=10, lr=0.05):
    """Online SGD on a fit-to-ones task; return the per-step loss list."""
    x = jnp.ones((1, 3))
    algo.compile_graph(x)
    algo.init_etrace_state()
    losses = []
    for _ in range(n_steps):
        def loss_fn(x_):
            out = algo.update(x_)
            return ((out - jnp.ones_like(out)) ** 2).mean()

        grads, loss_val = brainstate.augment.grad(
            loss_fn, algo.param_states, return_value=True
        )(x)
        for path, st in algo.param_states.items():
            st.value = st.value - lr * grads[path]
        losses.append(float(loss_val))
    return losses


class TestOSTLRecurrent(unittest.TestCase):
    def test_is_param_dim_subclass(self):
        algo = OSTLRecurrent(_tiny_rec_net())
        assert isinstance(algo, ParamDimVjpAlgorithm)
        assert algo.regime == 'with-H'

    def test_compiles_and_updates(self):
        algo, out = _drive(OSTLRecurrent(_tiny_rec_net()), n=1)
        assert out.shape == (1, 3)

    def test_reset_zeros_traces(self):
        algo, _ = _drive(OSTLRecurrent(_tiny_rec_net()), n=3)
        algo.reset_state(batch_size=1)
        for st in algo.etrace_bwg.values():
            for arr in jax.tree.leaves(st.value):
                assert jnp.allclose(arr, jnp.zeros_like(arr))
        assert algo.running_index.value == 0

    def test_forwards_kwargs_to_base(self):
        algo = OSTLRecurrent(
            _tiny_rec_net(),
            fast_solve=False,
            vjp_method='single-step',
        )
        assert algo.fast_solve is False
        assert algo.vjp_method == 'single-step'

    def test_get_etrace_of_named_weight(self):
        net = _tiny_rec_net()
        algo, _ = _drive(OSTLRecurrent(net), n=2)
        traces = algo.get_etrace_of(net.w_rec)
        assert len(traces) >= 1

    def test_training_decreases_loss(self):
        losses = _train_loss(OSTLRecurrent(_tiny_rec_net()))
        assert losses[-1] < losses[0]


class TestOSTLFeedforward(unittest.TestCase):
    def test_is_io_dim_subclass(self):
        algo = OSTLFeedforward(_tiny_rec_net())
        assert isinstance(algo, pp_prop)
        assert algo.regime == 'without-H'

    def test_default_decay_is_tiny(self):
        algo = OSTLFeedforward(_tiny_rec_net())
        assert algo.decay == 1e-6

    def test_compiles_and_updates(self):
        algo, out = _drive(OSTLFeedforward(_tiny_rec_net()), n=1)
        assert out.shape == (1, 3)

    def test_reset_zeros_traces(self):
        algo, _ = _drive(OSTLFeedforward(_tiny_rec_net()), n=3)
        algo.reset_state(batch_size=1)
        for st in algo.etrace_xs.values():
            assert jnp.allclose(st.value, jnp.zeros_like(st.value))
        for st in algo.etrace_dfs.values():
            assert jnp.allclose(st.value, jnp.zeros_like(st.value))
        assert algo.running_index.value == 0

    def test_custom_float_decay_honored(self):
        algo = OSTLFeedforward(_tiny_rec_net(), decay_or_rank=0.5)
        assert algo.decay == 0.5

    def test_int_rank_sets_decay(self):
        # rank r -> decay = (r-1)/(r+1); rank 3 -> 0.5
        algo = OSTLFeedforward(_tiny_rec_net(), decay_or_rank=3)
        assert abs(algo.decay - 0.5) < 1e-9

    def test_out_of_range_float_decay_rejected(self):
        with self.assertRaises(AssertionError):
            OSTLFeedforward(_tiny_rec_net(), decay_or_rank=1.5)

    def test_bad_decay_type_rejected(self):
        with self.assertRaises(ValueError):
            OSTLFeedforward(_tiny_rec_net(), decay_or_rank='nope')

    def test_training_decreases_loss(self):
        losses = _train_loss(OSTLFeedforward(_tiny_rec_net()))
        assert losses[-1] < losses[0]


class TestOSTLExports(unittest.TestCase):
    def test_exported_from_package(self):
        assert braintrace.OSTLRecurrent is OSTLRecurrent
        assert braintrace.OSTLFeedforward is OSTLFeedforward
        for name in ('OSTLRecurrent', 'OSTLFeedforward'):
            assert name in braintrace.__all__


class TestRegimesDiffer(unittest.TestCase):
    def test_recurrent_and_feedforward_disagree_on_gradient(self):
        """With recurrent dynamics, keeping vs dropping H must change the grad."""

        def final_grad(algo):
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()
            for _ in range(3):
                algo.update(x)

            def loss(x_):
                return (algo.update(x_) ** 2).sum()

            grads, _ = brainstate.augment.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g_with = final_grad(OSTLRecurrent(_tiny_rec_net()))
        g_without = final_grad(OSTLFeedforward(_tiny_rec_net()))
        assert not jnp.allclose(g_with, g_without, atol=1e-4)


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
    def test_ostl(self):
        losses = _run(OSTLRecurrent(_toy_net()))
        assert losses[-1] < losses[0]
