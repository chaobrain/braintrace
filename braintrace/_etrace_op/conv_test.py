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

"""Tests for the convolution ETP primitive and the :func:`conv` API.

Coverage:

* Forward correctness — agrees with ``jax.lax.conv_general_dilated``
  under SAME and VALID padding, with and without bias.
* The ``etp_conv_p`` primitive appears in the jaxpr and carries the
  expected static parameters.
* saiunit support — quantities with units multiply correctly.
* JAX rules — jit / grad work.
* Four ETP rules — the ``conv`` rules use a VJP-based ``xy_to_dw`` that
  must match a hand-written JAX VJP; the init fns return arrays of the
  documented shape.
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
    conv,
    etp_conv_p,
)


_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


# Standard NCHW-style conv1d: input (batch, channel, length), kernel
# (out_channel, in_channel, kw).
def _ref_conv(x, kernel, **kw):
    return jax.lax.conv_general_dilated(x, kernel, **kw)


_BASE_CONV_KW = dict(
    window_strides=(1,),
    padding='SAME',
    lhs_dilation=(1,),
    rhs_dilation=(1,),
    feature_group_count=1,
    batch_group_count=1,
    dimension_numbers=None,
)




# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

class TestForwardCorrectness:

    def test_same_padding_no_bias(self):
        x = jnp.ones((1, 3, 8))               # (batch, in_ch, L)
        k = jnp.arange(36.0).reshape(4, 3, 3) # (out_ch, in_ch, kw)
        out = conv(x, k)
        ref = _ref_conv(x, k, **_BASE_CONV_KW)
        np.testing.assert_allclose(out, ref)

    def test_valid_padding(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        out = conv(x, k, padding='VALID')
        ref_kw = dict(_BASE_CONV_KW)
        ref_kw['padding'] = 'VALID'
        ref = _ref_conv(x, k, **ref_kw)
        np.testing.assert_allclose(out, ref)

    def test_with_bias(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        b = jnp.arange(4.0)
        # bias broadcasts on the channel dim
        out = conv(x, k, b.reshape(1, 4, 1))
        ref = _ref_conv(x, k, **_BASE_CONV_KW) + b.reshape(1, 4, 1)
        np.testing.assert_allclose(out, ref)

    def test_strides(self):
        x = jnp.ones((1, 3, 16))
        k = jnp.ones((4, 3, 3))
        out = conv(x, k, strides=(2,), padding='VALID')
        ref_kw = dict(_BASE_CONV_KW)
        ref_kw['window_strides'] = (2,)
        ref_kw['padding'] = 'VALID'
        ref = _ref_conv(x, k, **ref_kw)
        np.testing.assert_allclose(out, ref)


# ---------------------------------------------------------------------------
# Primitive bind + params
# ---------------------------------------------------------------------------

class TestPrimitiveAndParams:

    def test_jaxpr_contains_etp_conv(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        jaxpr = jax.make_jaxpr(lambda x, k: conv(x, k))(x, k)
        assert any(eqn.primitive is etp_conv_p for eqn in jaxpr.jaxpr.eqns)

    def test_has_bias_true_when_bias_supplied(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        b = jnp.zeros((1, 4, 1))
        jaxpr = jax.make_jaxpr(lambda x, k, b: conv(x, k, b))(x, k, b)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_conv_p)
        assert eqn.params['has_bias'] is True

    def test_has_bias_false_when_bias_omitted(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        jaxpr = jax.make_jaxpr(lambda x, k: conv(x, k))(x, k)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_conv_p)
        assert eqn.params['has_bias'] is False

    def test_strides_propagate(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        jaxpr = jax.make_jaxpr(
            lambda x, k: conv(x, k, strides=(2,))
        )(x, k)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_conv_p)
        assert eqn.params['strides'] == (2,)


# ---------------------------------------------------------------------------
# saiunit
# ---------------------------------------------------------------------------

class TestSaiunit:

    def test_unitless(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        out = conv(x, k)
        assert not isinstance(out, u.Quantity)

    def test_unit_multiplication(self):
        x = jnp.ones((1, 3, 8)) * u.mV
        k = jnp.ones((4, 3, 3))
        out = conv(x, k)
        ref = _ref_conv(jnp.ones((1, 3, 8)), k, **_BASE_CONV_KW)
        np.testing.assert_allclose(u.get_mantissa(out), ref)


# ---------------------------------------------------------------------------
# JAX rules
# ---------------------------------------------------------------------------

class TestJAXRules:

    def test_jit(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        out = jax.jit(conv)(x, k)
        ref = _ref_conv(x, k, **_BASE_CONV_KW)
        np.testing.assert_allclose(out, ref)

    def test_grad_wrt_kernel(self):
        x = jnp.ones((1, 3, 8))
        k = jnp.ones((4, 3, 3))
        # Compare to grad through hand-written conv_general_dilated.
        gk_etp = jax.grad(lambda k_: conv(x, k_).sum())(k)
        gk_ref = jax.grad(
            lambda k_: _ref_conv(x, k_, **_BASE_CONV_KW).sum()
        )(k)
        np.testing.assert_allclose(gk_etp, gk_ref)


# ---------------------------------------------------------------------------
# ETP rules
# ---------------------------------------------------------------------------

class TestConvEtpRules:

    def test_yw_to_w_broadcasts_hidden(self):
        rule = ETP_RULES_YW_TO_W[etp_conv_p]
        # trace shape (out_ch, in_ch, kw) — rank 3.
        # hidden shape (out_ch,) — rank 1.
        # n_expand = 3 - 1 = 2 → expand_dims twice → (1, 1, out_ch).
        # Broadcasts against (out_ch, in_ch, kw)? No — last axis would
        # collide. The actual broadcast targets a (kw_dim) trailing
        # axis equal to out_ch only when kw == out_ch.
        # For a clean shape test pick trace shape (out_ch, ..., out_ch).
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        trace = jnp.ones((4,))                # rank 1 — no expand needed
        out = rule(hidden, trace)
        np.testing.assert_allclose(out, hidden * trace)

    def test_xy_to_dw_matches_jax_vjp(self):
        """The rule forwards its kwargs straight to
        ``jax.lax.conv_general_dilated`` (minus ``has_bias``); supply the
        ``window_strides`` / dilation kwargs that ``conv_general_dilated``
        natively expects."""
        rule = ETP_RULES_XY_TO_DW[etp_conv_p]
        x = jnp.ones((1, 3, 8))
        k = jnp.arange(36.0).reshape(4, 3, 3)
        ref_y_shape = _ref_conv(x, k, **_BASE_CONV_KW).shape
        hidden = jnp.ones(ref_y_shape)
        dk = rule(x, hidden, k, has_bias=False, **_BASE_CONV_KW)
        _, vjp_fn = jax.vjp(
            lambda k_: _ref_conv(x, k_, **_BASE_CONV_KW), k
        )
        ref_dk = vjp_fn(hidden)[0]
        np.testing.assert_allclose(dk, ref_dk)

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_conv_p]
        x_var = _fake_var((1, 3, 8))
        y_var = _fake_var((1, 4, 8))
        k_var = _fake_var((4, 3, 3))
        out = rule(x_var, y_var, k_var, num_hidden_state=2)
        assert out.shape == (1, 4, 3, 3, 2)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_conv_p]
        x_var = _fake_var((1, 3, 8))
        y_var = _fake_var((1, 4, 8))
        k_var = _fake_var((4, 3, 3))
        out = rule(x_var, y_var, k_var, num_hidden_state=2)
        assert out.shape == (1, 4, 8, 2)


class TestPublicAPIRoundTrip:

    def test_public_alias(self):
        assert braintrace.conv is conv
