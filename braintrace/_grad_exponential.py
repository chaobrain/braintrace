# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
import jax.tree

__all__ = [
    'GradExpon',
]


class GradExpon(brainstate.nn.Module):
    r"""
    Accumulates gradients exponentially.

    Mathematically, the update rule is:

        $$
        g_{t+1} = \text{decay} \cdot g_t + \text{grads} \\
        $$

    where $g_t$ is the accumulated gradient at time $t$, $\text{grads}$ is the gradient at time
    $t$, and $\text{decay}$ is the decay factor.

    Args:
        grad_shape: The shape of the gradients.
        tau_or_decay: The decay time constant or the decay factor.
    """

    def __init__(
        self,
        grad_shape: brainstate.typing.PyTree,
        tau_or_decay: u.Quantity[u.second] | float,
    ):
        super().__init__()

        # gradients
        self.gradients = jax.tree.map(
            lambda x: jax.numpy.zeros_like(x), grad_shape
        )

        # decay time constant
        if isinstance(tau_or_decay, u.Quantity):
            tau = u.maybe_decimal(tau_or_decay / brainstate.environ.get_dt())
            decay = u.math.exp(-1.0 / tau)
        elif isinstance(tau_or_decay, float):
            assert 0.0 < tau_or_decay < 1.0, f"Decay must be between 0 and 1, but got {tau_or_decay}"
            decay = tau_or_decay
        else:
            raise TypeError(f"tau_or_decay must be a Quantity or a float, but got {tau_or_decay}")
        self.decay = decay

    def update(self, grads: brainstate.typing.PyTree):
        """
        Updates the accumulated gradients using the exponential decay rule.

        This method applies the update rule g_{t+1} = decay * g_t + grads, where g_t is the
        accumulated gradient at time t, grads is the new gradient, and decay is the decay factor.

        Args:
            grads (brainstate.typing.PyTree): The new gradients to be incorporated into the accumulated gradients.

        Returns:
            None. The method updates the `self.gradients` attribute in-place.
        """
        self.gradients = jax.tree.map(
            lambda x, y: x * self.decay + y,
            self.gradients,
            grads,
            is_leaf=u.math.is_quantity
        )
