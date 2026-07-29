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

"""Phase 3 while-hidden support: algorithm-level equivalence + negative pins.

**Primary equivalence.** ``while_settle_rnn`` and ``while_settle_twin_rnn``
(same seed) are the same math — the only differences are the compiler
machinery Phase 3 added for the while path: forward-mode instead of
reverse-mode hidden-Jacobian extraction, and the ``stop_gradient`` detach of
the loop inputs in the perturbed jaxpr. D_RTRL with the default single-step
VJP must therefore produce element-wise identical gradients on the pair over
a T=6 sequence; any mismatch is a Phase 3 bug, not an approximation.

**Ground-truth anchor.** The twin (no while, plain unrolled composition) is
registered in ``exact_correctness_test.py``'s ``_EXACT_CASES`` so its
multi-step D_RTRL / pp_prop_full gradients are checked against the BPTT
oracle. That anchors the primary assertion above to ground truth.

**Negative pins.**

- A weight used *through an ETP primitive* inside a while body is a hard
  compile error (``WEIGHT_IN_WHILE``).
- An ETP primitive inside a while body without a tracked weight invar is
  rejected by the relation-pass hardening under the default policy
  (``etp_in_control_flow='error'``).
- ``vjp_method='multi-step'`` on the while model hits JAX's structural
  reverse-through-``while_loop`` ``ValueError`` — the documented limitation;
  the single-step path is the supported one (the loop's own hidden group's
  learning signal comes exclusively from the perturbation cotangents, which
  the detach keeps exact).
- **Detach scope limitation (pinned as documented behavior):** the detach
  zeroes every same-step reverse path *through* the loop, so an upstream
  trainable layer whose only path to the loss crosses a while-hidden layer
  receives an exactly-zero learning signal — its gradient is zero while the
  twin's is not. Stacking trainable layers behind a while-hidden layer is
  therefore unsupported for upstream credit; the compile emits a
  WARNING-level ``CONTROL_FLOW_OPAQUE_FWD`` diagnostic for each detach.
"""

import warnings

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_param_gradients_close,
    online_param_gradients,
    online_param_gradients_singlestep_naive,
)
from braintrace._algorithm.oracle_models import (
    while_settle_rnn,
    while_settle_twin_rnn,
)

ATOL_EQUIV = 1e-5


def _inputs(T, n_in, seed=42):
    return jnp.asarray(np.random.RandomState(seed).randn(T, n_in).astype('float32'))


# --- twin contract -------------------------------------------------------------

def test_while_and_twin_share_weights_and_forward_values():
    """Same seed => identical weights, and the two updates produce identical
    hidden trajectories (the twin really is the same model minus the while)."""
    m_while = while_settle_rnn(seed=0).factory()
    m_twin = while_settle_twin_rnn(seed=0).factory()
    brainstate.nn.init_all_states(m_while, batch_size=1)
    brainstate.nn.init_all_states(m_twin, batch_size=1)
    assert bool(jnp.array_equal(m_while.win.value, m_twin.win.value))

    inputs = _inputs(4, 3)
    out_w = brainstate.transform.for_loop(m_while.update, inputs)
    out_t = brainstate.transform.for_loop(m_twin.update, inputs)
    np.testing.assert_allclose(np.asarray(out_w), np.asarray(out_t), atol=1e-6)


# --- primary: single-step D_RTRL while == twin ----------------------------------

def test_d_rtrl_singlestep_while_equals_twin():
    """D_RTRL (default single-step VJP) total gradient over T=6 on the while
    model equals the twin element-wise. Isolates the Phase 3 machinery:
    jacfwd-vs-jacrev extraction and the perturbation detach must be
    gradient-neutral."""
    inputs = _inputs(6, 3)
    g_while = online_param_gradients_singlestep_naive(
        while_settle_rnn(seed=0).factory, inputs, algo_factory=braintrace.D_RTRL
    )
    g_twin = online_param_gradients_singlestep_naive(
        while_settle_twin_rnn(seed=0).factory, inputs, algo_factory=braintrace.D_RTRL
    )
    assert_param_gradients_close(g_while, g_twin, atol=ATOL_EQUIV)


# --- negative: weight-in-while is a hard compile error --------------------------

class _WeightInWhileNet(brainstate.nn.Module):
    """Recurrent ETP matmul INSIDE the while body — must be rejected."""

    def __init__(self, n_rec: int = 4):
        super().__init__()
        with brainstate.random.seed_context(0):
            self.w = brainstate.ParamState(0.1 * brainstate.random.randn(n_rec, n_rec))
        self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

    def update(self, x):
        def body(s):
            i, h = s
            return i + 1, jnp.tanh(braintrace.matmul(h, self.w.value))

        _, h_new = jax.lax.while_loop(lambda s: s[0] < 3, body, (0, self.h.value))
        self.h.value = h_new
        return h_new


def test_weight_in_while_raises_at_compile():
    model = _WeightInWhileNet()
    brainstate.nn.init_all_states(model, batch_size=1)
    with pytest.raises(NotImplementedError, match='while'):
        braintrace.compile_etrace_graph(model, jnp.ones((3,), dtype='float32'))


# --- negative: ETP primitive in while without a weight invar --------------------

class _EtpConstInWhileNet(brainstate.nn.Module):
    """ETP matmul applied to a plain CONSTANT matrix inside the while body: no
    tracked weight invar enters the loop (so ``WEIGHT_IN_WHILE`` cannot fire),
    but the un-flattened ETP primitive must still be rejected by the
    relation-pass hardening under the default policy."""

    def __init__(self, n_rec: int = 4):
        super().__init__()
        with brainstate.random.seed_context(1):
            self.win = brainstate.ParamState(0.1 * brainstate.random.randn(3, n_rec))
        self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))
        self._const_w = 0.1 * jnp.ones((n_rec, n_rec))

    def update(self, x):
        pre = braintrace.matmul(x.reshape(1, -1), self.win.value)
        const_w = self._const_w

        def body(s):
            i, h = s
            return i + 1, jnp.tanh(pre + braintrace.matmul(h, const_w))

        _, h_new = jax.lax.while_loop(lambda s: s[0] < 3, body, (0, self.h.value))
        self.h.value = h_new
        return h_new


def test_etp_primitive_in_while_raises_under_default_policy():
    model = _EtpConstInWhileNet()
    brainstate.nn.init_all_states(model, batch_size=1)
    with pytest.raises(NotImplementedError, match='could not flatten'):
        braintrace.compile_etrace_graph(model, jnp.ones((3,), dtype='float32'))


# --- negative: multi-step VJP hits reverse-through-while (documented limit) -----

def test_multistep_vjp_on_while_model_raises_reverse_through_while():
    """Pin the current failure mode: the multi-step VJP path differentiates in
    reverse through the step function, and JAX structurally rejects
    reverse-mode through ``lax.while_loop``. The single-step path is the
    supported one for while-hidden models."""
    spec = while_settle_rnn(seed=0)
    inputs = _inputs(6, 3)
    with pytest.raises(ValueError, match='while_loop'):
        online_param_gradients(
            spec.factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        )


# --- documented limitation: upstream layer behind a while-hidden layer ----------

class _StackedWhileNet(brainstate.nn.Module):
    """Layer 1: tanh RNN (ETP ``w1``) feeding layer 2: while-settle (ETP
    ``w2``). ``while_layer=False`` replaces the loop with its hand-composed
    ``k``-fold twin; the same seed gives both variants identical weights."""

    def __init__(self, n_in: int = 3, n_rec: int = 4, k: int = 3,
                 decay: float = 0.8, while_layer: bool = True):
        super().__init__()
        self.k = k
        self.decay = decay
        self.while_layer = while_layer
        with brainstate.random.seed_context(2):
            self.w1 = brainstate.ParamState(0.1 * brainstate.random.randn(n_in, n_rec))
            self.w2 = brainstate.ParamState(0.1 * brainstate.random.randn(n_rec, n_rec))
        self.h1 = brainstate.HiddenState(jnp.zeros((1, n_rec)))
        self.h2 = brainstate.HiddenState(jnp.zeros((1, n_rec)))

    def update(self, x):
        x_row = x.reshape(1, -1)
        self.h1.value = jnp.tanh(
            braintrace.matmul(x_row, self.w1.value) + 0.5 * self.h1.value
        )
        pre = braintrace.matmul(self.h1.value, self.w2.value) + self.decay * self.h2.value
        h_prev = self.h2.value
        if self.while_layer:
            def body(s):
                i, h = s
                return i + 1, h + 0.5 * jnp.tanh(pre - h)

            _, h_new = jax.lax.while_loop(lambda s: s[0] < self.k, body, (0, h_prev))
        else:
            h_new = h_prev
            for _ in range(self.k):
                h_new = h_new + 0.5 * jnp.tanh(pre - h_new)
        self.h2.value = h_new
        return h_new


def test_upstream_layer_gradient_is_zero_behind_while_DOCUMENTED_LIMITATION():
    """The perturbation detach zeroes every same-step reverse path THROUGH the
    loop: the upstream layer's weight ``w1`` (whose only path to the loss
    crosses the while) gets an exactly-zero gradient, while the twin's is
    substantially nonzero. The while layer's own weight ``w2`` still matches
    the twin exactly. This is the documented while-hidden limitation, not an
    approximation — a WARNING-level CONTROL_FLOW_OPAQUE_FWD diagnostic is
    emitted at compile time for each detach."""
    inputs = _inputs(6, 3)

    def grads(while_layer):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return online_param_gradients_singlestep_naive(
                lambda: _StackedWhileNet(while_layer=while_layer),
                inputs,
                algo_factory=braintrace.D_RTRL,
            )

    g_while = grads(True)
    g_twin = grads(False)
    # the while layer's own weight: exact match with the twin
    assert_param_gradients_close(g_while, g_twin, atol=ATOL_EQUIV, keys=[('w2',)])
    # the upstream weight: exactly zero under the while model, nonzero in the twin
    assert bool(jnp.all(jnp.asarray(g_while[('w1',)]) == 0))
    assert float(jnp.max(jnp.abs(jnp.asarray(g_twin[('w1',)])))) > 1e-4
