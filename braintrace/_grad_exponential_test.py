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

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace._grad_exponential import GradExpon


class TestGradExponInit:
    """Construction of :class:`GradExpon` and decay-factor resolution."""

    def test_float_decay_is_used_directly(self):
        acc = GradExpon(jnp.zeros((3,)), 0.9)
        assert acc.decay == 0.9

    def test_gradients_initialised_to_zeros(self):
        acc = GradExpon(jnp.ones((2, 4)), 0.5)
        assert acc.gradients.value.shape == (2, 4)
        assert jnp.allclose(acc.gradients.value, 0.0)

    def test_quantity_tau_resolves_to_exponential_decay(self):
        # decay = exp(-1 / (tau / dt)) = exp(-1 / 100) for tau=10ms, dt=0.1ms.
        with brainstate.environ.context(dt=0.1 * u.ms):
            acc = GradExpon(jnp.zeros((3,)), 10.0 * u.ms)
        assert 0.0 < float(acc.decay) < 1.0
        assert np.isclose(float(acc.decay), np.exp(-0.01))

    @pytest.mark.parametrize("bad_decay", [0.0, 1.0, 1.5, -0.2])
    def test_out_of_range_float_decay_raises(self, bad_decay):
        with pytest.raises(AssertionError):
            GradExpon(jnp.zeros((3,)), bad_decay)

    @pytest.mark.parametrize("bad_tau", [1, "0.9", None])
    def test_non_float_non_quantity_raises_type_error(self, bad_tau):
        with pytest.raises(TypeError):
            GradExpon(jnp.zeros((3,)), bad_tau)


class TestGradExponUpdate:
    """The exponential (leaky) accumulation rule g <- decay * g + grads."""

    def test_single_update_equals_input(self):
        acc = GradExpon(jnp.zeros((3,)), 0.9)
        acc.update(jnp.ones((3,)))
        assert jnp.allclose(acc.gradients.value, 1.0)

    def test_two_updates_match_docstring_example(self):
        acc = GradExpon(jnp.zeros((3,)), 0.9)
        acc.update(jnp.ones((3,)))
        acc.update(jnp.ones((3,)))
        # 0.9 * 1.0 + 1.0 = 1.9
        assert jnp.allclose(acc.gradients.value, 1.9)

    def test_decay_attenuates_old_gradient(self):
        acc = GradExpon(jnp.zeros((1,)), 0.5)
        acc.update(jnp.array([2.0]))      # -> 2.0
        acc.update(jnp.array([0.0]))      # -> 0.5 * 2.0 + 0.0 = 1.0
        assert jnp.allclose(acc.gradients.value, 1.0)

    def test_pytree_gradients_are_accumulated_leafwise(self):
        shape = {"a": jnp.zeros((2,)), "b": jnp.zeros((3,))}
        acc = GradExpon(shape, 0.5)
        acc.update({"a": jnp.ones((2,)), "b": jnp.full((3,), 4.0)})
        acc.update({"a": jnp.ones((2,)), "b": jnp.zeros((3,))})
        assert jnp.allclose(acc.gradients.value["a"], 1.5)   # 0.5*1 + 1
        assert jnp.allclose(acc.gradients.value["b"], 2.0)   # 0.5*4 + 0
