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
  the single-step path is the supported one (its learning signal comes
  exclusively from the perturbation cotangents, which the detach keeps
  exact).
"""

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
