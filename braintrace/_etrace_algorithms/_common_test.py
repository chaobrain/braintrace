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

import jax
import jax.numpy as jnp

from braintrace._etrace_algorithms._common import (
    FixedRandomFeedback,
    KappaFilter,
    PresynapticTrace,
    extract_y_target,
)


class TestPresynapticTrace(unittest.TestCase):
    def test_exponential_accumulation(self):
        trace = PresynapticTrace(jnp.zeros((2, 3)), leak=0.9)
        trace.update(jnp.ones((2, 3)))
        trace.update(jnp.ones((2, 3)))
        # After 2 ones: 0.9*(0.9*0 + 1) + 1 == 1.9
        assert jnp.allclose(trace.value, jnp.full((2, 3), 1.9))

    def test_reset_to_zero(self):
        trace = PresynapticTrace(jnp.ones((4,)), leak=0.5)
        trace.reset_state()
        assert jnp.allclose(trace.value, jnp.zeros((4,)))


class TestKappaFilter(unittest.TestCase):
    def test_low_pass(self):
        flt = KappaFilter(jnp.zeros((3,)), kappa=0.8)
        y1 = flt.update(jnp.ones((3,)))
        # (1 - 0.8)*1 + 0.8*0 == 0.2
        assert jnp.allclose(y1, jnp.full((3,), 0.2))
        y2 = flt.update(jnp.ones((3,)))
        # (1 - 0.8)*1 + 0.8*0.2 == 0.36
        assert jnp.allclose(y2, jnp.full((3,), 0.36))

    def test_kappa_zero_disables(self):
        flt = KappaFilter(jnp.zeros((3,)), kappa=0.0)
        y = flt.update(jnp.full((3,), 5.0))
        assert jnp.allclose(y, jnp.full((3,), 5.0))  # pass-through


class TestFixedRandomFeedback(unittest.TestCase):
    def test_shape_and_frozen(self):
        key = jax.random.PRNGKey(0)
        fb = FixedRandomFeedback(n_target=10, n_layer=200, key=key, init_scale=0.1)
        assert fb.B.shape == (10, 200)
        grad_fn = jax.grad(lambda y_target: (fb.project(y_target) ** 2).sum())
        y = jnp.ones((5, 10))
        g = grad_fn(y)
        assert g.shape == y.shape

    def test_project_shapes(self):
        fb = FixedRandomFeedback(n_target=4, n_layer=7, key=jax.random.PRNGKey(1))
        y_target = jnp.ones((3, 4))  # batched
        proj = fb.project(y_target)
        assert proj.shape == (3, 7)


class TestExtractYTarget(unittest.TestCase):
    def test_absent_returns_none(self):
        assert extract_y_target(()) is None

    def test_present_returns_value(self):
        y = jnp.ones((5,))
        assert extract_y_target((jnp.zeros(3), y), index=1) is y
