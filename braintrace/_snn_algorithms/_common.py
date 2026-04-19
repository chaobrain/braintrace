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

"""Shared helpers for the SNN online-learning algorithms."""

from typing import Optional

import brainstate
import jax
import jax.numpy as jnp


__all__ = [
    'PresynapticTrace',
    'KappaFilter',
    'FixedRandomFeedback',
    'extract_y_target',
]


class PresynapticTrace(brainstate.ShortTermState):
    """Leaky accumulator ``â ← λ·â + x_t`` used by OTTT and OTPE-Approx.

    Parameters
    ----------
    init_value : jax.Array
        Initial value; also dictates shape and dtype.
    leak : float
        Decay factor λ in (0, 1). Pulled from the neuron's membrane leak in SNN usage.
    """

    __module__ = 'braintrace'

    def __init__(self, init_value, leak: float):
        super().__init__(init_value)
        if not (0.0 < leak < 1.0):
            raise ValueError(f'leak must be in (0, 1); got {leak}')
        self.leak = float(leak)
        self._init_shape = jnp.shape(init_value)
        self._init_dtype = init_value.dtype

    def update(self, x):
        """Apply one accumulation step: â ← λ·â + x."""
        self.value = self.leak * self.value + x
        return self.value

    def reset_state(self, batch_size: Optional[int] = None, **kwargs):
        if batch_size is None:
            shape = self._init_shape
        elif len(self._init_shape) == 0:
            shape = (batch_size,)
        else:
            shape = (batch_size, *self._init_shape[1:])
        self.value = jnp.zeros(shape, dtype=self._init_dtype)
