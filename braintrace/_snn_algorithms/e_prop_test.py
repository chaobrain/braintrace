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
from braintrace._snn_algorithms.e_prop import EProp


def _lsnn_like():
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


class TestEPropConstruction(unittest.TestCase):
    def test_default_feedback_and_kappa(self):
        algo = EProp(_lsnn_like())
        assert algo.feedback == 'symmetric'
        assert algo.kappa_filter_decay == 0.0

    def test_invalid_feedback_raises(self):
        with self.assertRaises(ValueError):
            EProp(_lsnn_like(), feedback='weird')

    def test_kappa_filter_allocated_when_nonzero(self):
        algo = EProp(_lsnn_like(), kappa_filter_decay=0.9)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._kappa_filters) == len(algo.graph.hidden_param_op_relations)

    def test_kappa_filter_skipped_when_zero(self):
        algo = EProp(_lsnn_like(), kappa_filter_decay=0.0)
        x = jnp.ones((1, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._kappa_filters) == 0


class TestEPropKappaApplied(unittest.TestCase):
    def test_kappa_zero_matches_d_rtrl(self):
        """κ=0 must reproduce D_RTRL gradients bit-for-bit on the same model."""
        from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm

        def compute(algo_cls, **extra):
            net = _lsnn_like()
            algo = algo_cls(net, **extra)
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x_):
                out = algo.update(x_)
                return (out ** 2).sum()

            grads, _ = brainstate.augment.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g_drtrl = compute(ParamDimVjpAlgorithm)
        g_eprop = compute(EProp, feedback='symmetric', kappa_filter_decay=0.0)
        assert jnp.allclose(g_drtrl, g_eprop, atol=1e-6)

    def test_kappa_nonzero_differs_from_zero(self):
        def compute(kappa):
            net = _lsnn_like()
            algo = EProp(net, kappa_filter_decay=kappa)
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()
            algo.update(x)

            def loss(x_):
                out = algo.update(x_)
                return (out ** 2).sum()

            grads, _ = brainstate.augment.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g0 = compute(0.0)
        g9 = compute(0.9)
        assert not jnp.allclose(g0, g9, atol=1e-4)


class TestEPropRandomFeedback(unittest.TestCase):
    def test_random_feedback_differs_from_symmetric(self):
        def compute(feedback, **extra):
            net = _lsnn_like()
            algo = EProp(net, feedback=feedback, **extra)
            x = jnp.ones((1, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x_):
                out = algo.update(x_)
                return (out ** 2).sum()

            grads, _ = brainstate.augment.grad(
                loss, algo.param_states, return_value=True
            )(x)
            return grads[next(iter(grads))]

        g_sym = compute('symmetric')
        g_rnd = compute('random', random_feedback_key=jax.random.PRNGKey(123))
        assert not jnp.allclose(g_sym, g_rnd, atol=1e-4)
