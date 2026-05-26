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

"""Deterministic toy models for the gradient oracle (test support)."""

from dataclasses import dataclass
from typing import Callable, Tuple

import brainstate
import jax
import jax.numpy as jnp

import braintrace


@dataclass(frozen=True)
class ModelSpec:
    """A zero-arg model factory plus metadata about its parameters.

    ``factory()`` returns a freshly constructed, *uninitialized* model with
    deterministic weights. Callers must call
    ``brainstate.nn.init_all_states(model, batch_size=...)`` themselves.
    """

    factory: Callable[[], brainstate.nn.Module]
    etp_param_keys: Tuple[tuple, ...]    # routed through an ETP primitive
    plain_param_keys: Tuple[tuple, ...]  # used via plain JAX ops (excluded from ETP)


def tanh_rnn(n_in: int = 3, n_rec: int = 4, seed: int = 0) -> ModelSpec:
    """Batched (batch=1) tanh RNN: recurrent ETP weight ``w``, plain input weight ``win``."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(seed), (n_rec, n_rec))
                )
                self.win = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(seed + 1), (n_in, n_rec))
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                inp = x @ self.win.value  # plain op -> excluded from ETP
                self.h.value = jax.nn.tanh(
                    inp + braintrace.matmul(self.h.value, self.w.value)
                )
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=(('win',),))
