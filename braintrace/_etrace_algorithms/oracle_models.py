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


def leaky_linear(n_in: int = 3, n_rec: int = 4, leak: float = 0.9, seed: int = 0) -> ModelSpec:
    """Pure leaky integrator with a trainable ETP *input* weight.

    The recurrence ``h_t = leak * h_{t-1} + matmul(x_t, w)`` has hidden-to-hidden
    Jacobian ``leak * I`` exactly (no off-diagonal recurrent term). This is the
    regime in which OTTT (which discards ``hid2hid_jac`` and assumes ``leak * I``)
    is exact. ``w`` reaches every future hidden state through the leaky carry, so
    it is a genuine ETP relation despite being an input projection.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(seed), (n_in, n_rec))
                )
                self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                drive = braintrace.matmul(x.reshape(1, -1), self.w.value)
                self.h.value = leak * self.h.value + drive
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=())


def stacked_tanh_rnn(n_in: int = 3, n_rec: int = 4, seed: int = 0) -> ModelSpec:
    """Two-layer tanh RNN with two trainable ETP recurrent weights.

    Layer 1: ``h1 = tanh(x @ win + matmul(h1, w1))``; layer 2:
    ``h2 = tanh(h1 @ wmid + matmul(h2, w2))``. ``w1``/``w2`` are ETP recurrent
    weights (two HiddenParamOp relations); ``win``/``wmid`` are plain projections
    (excluded from ETP). Exercises multi-relation D_RTRL == BPTT.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                k = jax.random.PRNGKey
                self.w1 = brainstate.ParamState(0.1 * jax.random.normal(k(seed), (n_rec, n_rec)))
                self.w2 = brainstate.ParamState(0.1 * jax.random.normal(k(seed + 1), (n_rec, n_rec)))
                self.win = brainstate.ParamState(0.1 * jax.random.normal(k(seed + 2), (n_in, n_rec)))
                self.wmid = brainstate.ParamState(0.1 * jax.random.normal(k(seed + 3), (n_rec, n_rec)))
                self.h1 = brainstate.HiddenState(jnp.zeros((1, n_rec)))
                self.h2 = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                self.h1.value = jax.nn.tanh(
                    x @ self.win.value + braintrace.matmul(self.h1.value, self.w1.value)
                )
                self.h2.value = jax.nn.tanh(
                    self.h1.value @ self.wmid.value + braintrace.matmul(self.h2.value, self.w2.value)
                )
                return self.h2.value

        return Net()

    return ModelSpec(
        factory=factory,
        etp_param_keys=(('w1',), ('w2',)),
        plain_param_keys=(('win',), ('wmid',)),
    )


def two_state_rnn(n_in: int = 3, n_rec: int = 3, seed: int = 0) -> ModelSpec:
    """Two coupled hidden states (v, a) that the compiler groups into ONE
    HiddenGroup with ``num_state == 2`` (an LIF+adaptation-like topology).

    ``v_t = 0.9 v + matmul(x, w) - 0.1 a``; ``a_t = 0.95 a + v``. ``w`` is the
    single trainable ETP input weight. D_RTRL handles this exactly; OTTT/OTPE
    reject it (their per-step rule assumes a single-state group).
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(seed), (n_in, n_rec))
                )
                self.v = brainstate.HiddenState(jnp.zeros((1, n_rec)))
                self.a = brainstate.HiddenState(jnp.zeros((1, n_rec)))

            def update(self, x):
                v, a = self.v.value, self.a.value
                self.v.value = 0.9 * v + braintrace.matmul(x.reshape(1, -1), self.w.value) - 0.1 * a
                self.a.value = 0.95 * a + v
                return self.v.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=())


def batched_tanh_rnn(n_in: int = 3, n_rec: int = 4, batch: int = 4, seed: int = 0) -> ModelSpec:
    """A tanh RNN whose hidden state carries an explicit leading batch axis of
    size ``batch``. The existing models hardcode a size-1 batch, so this one is
    used to exercise batch-axis invariance (batched gradient == sum of
    per-sequence gradients). ``w`` is the recurrent ETP weight; ``win`` is a
    plain input projection.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    0.5 * jax.random.normal(jax.random.PRNGKey(seed), (n_rec, n_rec))
                )
                self.win = brainstate.ParamState(
                    0.5 * jax.random.normal(jax.random.PRNGKey(seed + 1), (n_in, n_rec))
                )
                self.h = brainstate.HiddenState(jnp.zeros((batch, n_rec)))

            def update(self, x):
                self.h.value = jax.nn.tanh(
                    x @ self.win.value + braintrace.matmul(self.h.value, self.w.value)
                )
                return self.h.value

        return Net()

    return ModelSpec(factory=factory, etp_param_keys=(('w',),), plain_param_keys=(('win',),))
