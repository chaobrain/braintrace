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

"""Tests for the LoRA ETP primitives and the :func:`lora_matmul` API.

LoRA factorises a dense weight into two low-rank factors
``B`` (in, rank) and ``A`` (rank, out), optionally scaled by
``alpha``. The ETP trace and gradient state are pytrees keyed by
``'B'`` / ``'A'``.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u

import braintrace
from braintrace._etrace_op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    etp_lora_mm_p,
    etp_lora_mv_p,
    lora_matmul,
)


_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

class TestForwardCorrectness:

    def test_unbatched_matches_reference(self):
        x = jnp.array([1.0, 2.0, 3.0])
        B = jnp.arange(6.0).reshape(3, 2)          # in=3, rank=2
        A = jnp.arange(8.0).reshape(2, 4)          # rank=2, out=4
        out = lora_matmul(x, B, A)
        ref = x @ B @ A
        np.testing.assert_allclose(out, ref)

    def test_batched_matches_reference(self):
        x = jnp.arange(6.0).reshape(2, 3)
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        out = lora_matmul(x, B, A)
        np.testing.assert_allclose(out, x @ B @ A)

    def test_alpha_scales_output(self):
        x = jnp.arange(6.0).reshape(2, 3)
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        out = lora_matmul(x, B, A, alpha=0.5)
        np.testing.assert_allclose(out, 0.5 * (x @ B @ A))

    def test_with_bias(self):
        x = jnp.arange(6.0).reshape(2, 3)
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        b = jnp.arange(4.0)
        out = lora_matmul(x, B, A, bias=b)
        np.testing.assert_allclose(out, x @ B @ A + b)


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

class TestAutoDispatch:

    def test_unbatched_uses_lora_mv(self):
        x = jnp.ones(3)
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        jaxpr = jax.make_jaxpr(
            lambda x, B, A: lora_matmul(x, B, A)
        )(x, B, A)
        assert any(eqn.primitive is etp_lora_mv_p for eqn in jaxpr.jaxpr.eqns)
        assert not any(eqn.primitive is etp_lora_mm_p for eqn in jaxpr.jaxpr.eqns)

    def test_batched_uses_lora_mm(self):
        x = jnp.ones((2, 3))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        jaxpr = jax.make_jaxpr(
            lambda x, B, A: lora_matmul(x, B, A)
        )(x, B, A)
        assert any(eqn.primitive is etp_lora_mm_p for eqn in jaxpr.jaxpr.eqns)
        assert not any(eqn.primitive is etp_lora_mv_p for eqn in jaxpr.jaxpr.eqns)


# ---------------------------------------------------------------------------
# Primitive static params
# ---------------------------------------------------------------------------

class TestPrimitiveParams:

    def test_alpha_propagates(self):
        x = jnp.ones((2, 3))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        jaxpr = jax.make_jaxpr(
            lambda x, B, A: lora_matmul(x, B, A, alpha=0.25)
        )(x, B, A)
        eqn = next(
            e for e in jaxpr.jaxpr.eqns if e.primitive is etp_lora_mm_p
        )
        assert eqn.params['alpha'] == 0.25

    def test_has_bias_true_when_bias_supplied(self):
        x = jnp.ones((2, 3))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        b = jnp.zeros(4)
        jaxpr = jax.make_jaxpr(
            lambda x, B, A, b: lora_matmul(x, B, A, bias=b)
        )(x, B, A, b)
        eqn = next(
            e for e in jaxpr.jaxpr.eqns if e.primitive is etp_lora_mm_p
        )
        assert eqn.params['has_bias'] is True


# ---------------------------------------------------------------------------
# saiunit
# ---------------------------------------------------------------------------

class TestSaiunit:

    def test_unitless(self):
        x = jnp.ones((2, 3))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        out = lora_matmul(x, B, A)
        assert not isinstance(out, u.Quantity)


# ---------------------------------------------------------------------------
# JAX rules
# ---------------------------------------------------------------------------

class TestJAXRules:

    def test_jit(self):
        x = jnp.ones((2, 3))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        out = jax.jit(lora_matmul)(x, B, A)
        np.testing.assert_allclose(out, x @ B @ A)

    def test_grad_wrt_B_and_A(self):
        x = jnp.arange(6.0).reshape(2, 3)
        B = jnp.arange(6.0).reshape(3, 2)
        A = jnp.arange(8.0).reshape(2, 4)

        gb = jax.grad(lambda B_: lora_matmul(x, B_, A).sum())(B)
        ga = jax.grad(lambda A_: lora_matmul(x, B, A_).sum())(A)

        gb_ref = jax.grad(lambda B_: (x @ B_ @ A).sum())(B)
        ga_ref = jax.grad(lambda A_: (x @ B @ A_).sum())(A)
        np.testing.assert_allclose(gb, gb_ref)
        np.testing.assert_allclose(ga, ga_ref)


# ---------------------------------------------------------------------------
# ETP rules
# ---------------------------------------------------------------------------

class TestLoraMmEtpRules:

    def test_yw_to_w_pytree_structure(self):
        """``yw_to_w`` broadcasts ``hidden`` over the row axis of
        ``trace['A']``. Driven directly here with shapes that broadcast
        cleanly; the runtime ``vmap`` chain in d_rtrl reshapes wrappers
        to land on this contract."""
        rule = ETP_RULES_YW_TO_W[etp_lora_mm_p]
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        trace_B = jnp.ones((3, 2))
        # trace_A shape (rank=4, k=1) — expand_dims hidden axis=1 → (4, 1)
        # broadcasts against (4, 1) cleanly.
        trace_A = jnp.ones((4, 1))
        out = rule(hidden, {'B': trace_B, 'A': trace_A})
        assert set(out.keys()) == {'B', 'A'}
        np.testing.assert_allclose(out['B'], trace_B)
        np.testing.assert_allclose(out['A'], hidden[:, None] * trace_A)

    def test_xy_to_dw_pytree_and_values(self):
        rule = ETP_RULES_XY_TO_DW[etp_lora_mm_p]
        x = jnp.ones((2, 3))
        B = jnp.arange(6.0).reshape(3, 2)
        A = jnp.arange(8.0).reshape(2, 4)
        hidden = jnp.ones((2, 4))
        d = rule(x, hidden, B, A, alpha=1.0)
        assert set(d.keys()) == {'B', 'A'}
        # Compare to pure JAX VJP.
        _, vjp_fn = jax.vjp(lambda B_, A_: x @ B_ @ A_, B, A)
        ref_dB, ref_dA = vjp_fn(hidden)
        np.testing.assert_allclose(d['B'], ref_dB)
        np.testing.assert_allclose(d['A'], ref_dA)

    def test_xy_to_dw_respects_alpha(self):
        rule = ETP_RULES_XY_TO_DW[etp_lora_mm_p]
        x = jnp.ones((2, 3))
        B = jnp.arange(6.0).reshape(3, 2)
        A = jnp.arange(8.0).reshape(2, 4)
        hidden = jnp.ones((2, 4))
        d1 = rule(x, hidden, B, A, alpha=1.0)
        d_half = rule(x, hidden, B, A, alpha=0.5)
        np.testing.assert_allclose(d_half['B'], d1['B'] * 0.5)
        np.testing.assert_allclose(d_half['A'], d1['A'] * 0.5)

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_lora_mm_p]
        x_var = _fake_var((2, 3))
        y_var = _fake_var((2, 4))
        w_var = _fake_var((3, 2))
        out = rule(x_var, y_var, w_var, num_hidden_state=5)
        assert out['B'].shape == (2, 3, 2, 5)
        assert out['A'].shape == (2, 2, 4, 5)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_lora_mm_p]
        x_var = _fake_var((2, 3))
        y_var = _fake_var((2, 4))
        w_var = _fake_var((3, 2))
        out = rule(x_var, y_var, w_var, num_hidden_state=5)
        assert out.shape == (2, 4, 5)


class TestLoraMvEtpRules:

    def test_yw_to_w_pytree_structure(self):
        """mv-variant: expand_dims axis=0 → broadcasts ``hidden`` over the
        column axis of ``trace['A']``."""
        rule = ETP_RULES_YW_TO_W[etp_lora_mv_p]
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        trace_B = jnp.ones((3, 2))
        trace_A = jnp.ones((2, 4))
        out = rule(hidden, {'B': trace_B, 'A': trace_A})
        assert set(out.keys()) == {'B', 'A'}
        np.testing.assert_allclose(out['B'], trace_B)
        np.testing.assert_allclose(out['A'], trace_A * hidden[None, :])

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_lora_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((4,))
        w_var = _fake_var((3, 2))
        out = rule(x_var, y_var, w_var, num_hidden_state=5)
        assert out['B'].shape == (3, 2, 5)
        assert out['A'].shape == (2, 4, 5)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_lora_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((4,))
        w_var = _fake_var((3, 2))
        out = rule(x_var, y_var, w_var, num_hidden_state=5)
        assert out.shape == (4, 5)


class TestPublicAPIRoundTrip:

    def test_public_alias(self):
        assert braintrace.lora_matmul is lora_matmul
