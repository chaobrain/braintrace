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
import jax.tree
import saiunit as u

__all__ = [
    'GradExpon',
]


class GradExpon(brainstate.nn.Module):
    r"""Accumulate gradients with an exponential (leaky) running sum.

    Maintains a decaying accumulator over a pytree of gradients, useful for
    smoothing online-learning gradient signals across time steps.

    Parameters
    ----------
    grad_shape : brainstate.typing.PyTree
        A pytree whose leaves give the shape and dtype of the gradients to
        accumulate. The accumulator is initialised to zeros matching each
        leaf.
    tau_or_decay : saiunit.Quantity or float
        Either a decay time constant (as a :class:`~saiunit.Quantity`), from
        which the decay factor is computed as
        :math:`\exp(-1 / (\tau / \mathrm{dt}))`, or the decay factor itself
        (a ``float`` in the open interval :math:`(0, 1)`).

    Notes
    -----
    The update rule is

    .. math::

        g_{t+1} = \mathrm{decay} \cdot g_t + \mathrm{grads},

    where :math:`g_t` is the accumulated gradient at time :math:`t`,
    :math:`\mathrm{grads}` is the new gradient at time :math:`t`, and
    :math:`\mathrm{decay}` is the decay factor.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> acc = braintrace.GradExpon(jnp.zeros((3,)), 0.9)
        >>> acc.update(jnp.ones((3,)))
        >>> acc.update(jnp.ones((3,)))
        >>> print(acc.gradients.value)
        [1.9 1.9 1.9]
    """

    def __init__(
        self,
        grad_shape: brainstate.typing.PyTree,
        tau_or_decay: u.Quantity | float,
    ):
        super().__init__()

        # gradients (stored as LongTermState for proper JAX transform tracking)
        self.gradients = brainstate.LongTermState(
            jax.tree.map(lambda x: jax.numpy.zeros_like(x), grad_shape)
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
        r"""Update the accumulated gradients with the exponential decay rule.

        Applies :math:`g_{t+1} = \mathrm{decay} \cdot g_t + \mathrm{grads}`,
        where :math:`g_t` is the accumulated gradient, ``grads`` the new
        gradient, and :math:`\mathrm{decay}` the decay factor. The
        accumulator stored in ``self.gradients`` is updated in place.

        Parameters
        ----------
        grads : brainstate.typing.PyTree
            The new gradients to incorporate into the accumulated gradients.
            Must match the pytree structure of the accumulator.

        Returns
        -------
        None
            The ``self.gradients`` attribute is updated in place.
        """
        self.gradients.value = jax.tree.map(
            lambda x, y: x * self.decay + y,
            self.gradients.value,
            grads,
            is_leaf=u.math.is_quantity
        )
