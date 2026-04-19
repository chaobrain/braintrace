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

"""Cross-class equivalence proofs.

Reduction identities from the spec:
- ``EProp(κ=0, symmetric)`` equals ``D_RTRL`` up to float tolerance.
"""

import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm
from braintrace._snn_algorithms import EProp


def _net():
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


def _grad(algo):
    x = jnp.ones((1, 3))
    algo.compile_graph(x)
    algo.init_etrace_state()

    def loss(x_):
        return (algo.update(x_) ** 2).sum()

    grads, _ = brainstate.augment.grad(
        loss, algo.param_states, return_value=True
    )(x)
    return grads[next(iter(grads))]


class TestCrossChecks(unittest.TestCase):
    def test_eprop_k0_matches_d_rtrl(self):
        g_eprop = _grad(EProp(_net(), feedback='symmetric', kappa_filter_decay=0.0))
        g_drtrl = _grad(ParamDimVjpAlgorithm(_net()))
        assert jnp.allclose(g_eprop, g_drtrl, atol=1e-6)
