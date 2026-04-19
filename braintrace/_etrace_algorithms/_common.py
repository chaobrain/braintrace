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
    '_resolve_leak',
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


class KappaFilter(brainstate.ShortTermState):
    """Low-pass output-side filter ``x_filt ← (1-κ)·x + κ·x_filt`` used by EProp.

    Parameters
    ----------
    init_value : jax.Array
    kappa : float
        Decay factor in [0, 1). 0 disables filtering.
    """

    __module__ = 'braintrace'

    def __init__(self, init_value, kappa: float):
        super().__init__(init_value)
        if not (0.0 <= kappa < 1.0):
            raise ValueError(f'kappa must be in [0, 1); got {kappa}')
        self.kappa = float(kappa)
        self._init_shape = jnp.shape(init_value)
        self._init_dtype = init_value.dtype

    def update(self, x):
        new = (1.0 - self.kappa) * x + self.kappa * self.value
        self.value = new
        return new

    def reset_state(self, batch_size: Optional[int] = None, **kwargs):
        if batch_size is None:
            shape = self._init_shape
        elif len(self._init_shape) == 0:
            shape = (batch_size,)
        else:
            shape = (batch_size, *self._init_shape[1:])
        self.value = jnp.zeros(shape, dtype=self._init_dtype)


class FixedRandomFeedback:
    """Frozen random feedback matrix ``B ∈ ℝ^{n_target × n_layer}`` with stop_gradient guard.

    Used by OSTTP (per-HiddenGroup target projection) and EProp-random-feedback.
    """

    __module__ = 'braintrace'

    def __init__(self, n_target: int, n_layer: int, key, init_scale: float = 0.1):
        self.B = jax.lax.stop_gradient(
            init_scale * jax.random.normal(key, (n_target, n_layer))
        )
        self.n_target = int(n_target)
        self.n_layer = int(n_layer)

    def project(self, y_target):
        """Return ``y_target @ B`` with B frozen. Handles batched and unbatched y_target."""
        return y_target @ self.B


def extract_y_target(args: tuple, *, index: int = -1) -> Optional[jax.Array]:
    """Fetch the target tensor from a positional-args tuple.

    Returns ``None`` if ``args`` is empty. ``index`` defaults to the last position
    (OSTTP's convention: ``algo.update(x, y_target)``).
    """
    if not args:
        return None
    return args[index]


def _resolve_leak(model, explicit: Optional[float]) -> float:
    """Pick the leak factor λ for OTTT/OTPE.

    Priority:
    1. ``explicit`` argument (constructor-supplied) wins if not None.
    2. Walk ``model.states()``; first state whose object has a ``leak`` attribute wins.
    3. Raise ``ValueError`` if neither resolves.
    """
    if explicit is not None:
        return float(explicit)
    if model is not None:
        states = model.states()
        # model.states() may return a dict-like (FlattedDict) or an iterable;
        # unwrap via .values() when available.
        state_iter = states.values() if hasattr(states, 'values') else iter(states)
        for st in state_iter:
            if hasattr(st, 'leak'):
                return float(st.leak)
    raise ValueError(
        'Could not resolve the membrane leak factor. Provide `leak=<float>` at '
        'construction, or ensure the model has a state with a `leak` attribute.'
    )
