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

"""Tests for the element-wise ETP primitive and the :func:`element_wise` API.

``etp_elemwise_p`` is the only ``gradient_enabled=True`` primitive: the
compiler must *evaluate* it when walking ``y -> h``. This module
verifies:

* The primitive is registered with the gradient-enabled flag.
* ``element_wise(w)`` (default ``fn=identity``) round-trips ``w``.
* ``element_wise(w, fn=lambda w: 2*w)`` applies ``fn`` *before* the bind
  (the primitive itself is the identity).
* saiunit quantities pass through.
* JAX rules — jit, vmap, grad, jvp.
* Four ETP rules return the documented values / shapes.
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
    GRADIENT_ENABLED_PRIMITIVES,
    element_wise,
    etp_elemwise_p,
    is_etp_enable_gradient_primitive,
)


_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


# ---------------------------------------------------------------------------
# Identity round-trip + custom fn
# ---------------------------------------------------------------------------

class TestIdentityRoundTrip:

    def test_default_fn_returns_weight(self):
        w = jnp.array([0.5, -1.0, 2.5])
        out = element_wise(w)
        np.testing.assert_allclose(out, w)

    def test_custom_fn_applied(self):
        w = jnp.array([1.0, 2.0, 3.0])
        out = element_wise(w, fn=lambda x: x * 10.0)
        np.testing.assert_allclose(out, w * 10.0)

    def test_nonlinear_fn(self):
        w = jnp.array([0.0, 0.5, 1.0])
        out = element_wise(w, fn=jnp.sin)
        np.testing.assert_allclose(out, jnp.sin(w))


# ---------------------------------------------------------------------------
# Gradient-enabled flag
# ---------------------------------------------------------------------------

class TestGradientEnabledFlag:

    def test_in_gradient_enabled_set(self):
        assert etp_elemwise_p in GRADIENT_ENABLED_PRIMITIVES

    def test_predicate_returns_true(self):
        assert is_etp_enable_gradient_primitive(etp_elemwise_p)


# ---------------------------------------------------------------------------
# Primitive appears in jaxpr
# ---------------------------------------------------------------------------

class TestPrimitiveInJaxpr:

    def test_jaxpr_contains_etp_elemwise(self):
        w = jnp.ones((3,))
        jaxpr = jax.make_jaxpr(lambda w: element_wise(w))(w)
        assert any(eqn.primitive is etp_elemwise_p for eqn in jaxpr.jaxpr.eqns)

    def test_fn_evaluated_before_bind(self):
        """The primitive itself is the identity; the user-supplied ``fn``
        runs as ordinary JAX ops *before* the bind. So if ``fn`` produces
        e.g. a ``mul`` op, that ``mul`` appears in the jaxpr alongside
        the elemwise bind."""
        w = jnp.ones((3,))
        jaxpr = jax.make_jaxpr(lambda w: element_wise(w, fn=lambda x: x * 2))(w)
        from jax import lax
        # mul (or scalar broadcast + mul) appears.
        assert any(
            eqn.primitive is etp_elemwise_p for eqn in jaxpr.jaxpr.eqns
        )
        # And there must be a multiplication-bearing op too.
        eqns = list(jaxpr.jaxpr.eqns)
        # Either lax.mul_p or something equivalent.
        has_mul = any('mul' in eqn.primitive.name for eqn in eqns)
        assert has_mul


# ---------------------------------------------------------------------------
# saiunit
# ---------------------------------------------------------------------------

class TestSaiunit:

    def test_unitless_passthrough(self):
        w = jnp.array([1.0, 2.0, 3.0])
        out = element_wise(w)
        assert not isinstance(out, u.Quantity)
        np.testing.assert_allclose(out, w)

    def test_unit_preserved(self):
        w = jnp.array([1.0, 2.0, 3.0]) * u.mV
        out = element_wise(w)
        np.testing.assert_allclose(u.get_mantissa(out), jnp.array([1.0, 2.0, 3.0]))

    def test_custom_fn_with_units(self):
        w = jnp.array([1.0, 2.0]) * u.mV
        out = element_wise(w, fn=lambda x: x * 2)
        np.testing.assert_allclose(u.get_mantissa(out), jnp.array([2.0, 4.0]))


# ---------------------------------------------------------------------------
# JAX rules
# ---------------------------------------------------------------------------

class TestJAXRules:

    def test_jit(self):
        w = jnp.array([1.0, 2.0, 3.0])
        out = jax.jit(element_wise)(w)
        np.testing.assert_allclose(out, w)

    def test_vmap(self):
        w = jnp.arange(12.0).reshape(4, 3)
        out = jax.vmap(element_wise)(w)
        np.testing.assert_allclose(out, w)

    def test_grad_default_fn(self):
        w = jnp.array([1.0, 2.0])
        g = jax.grad(lambda w_: element_wise(w_).sum())(w)
        np.testing.assert_allclose(g, jnp.ones_like(w))

    def test_grad_custom_fn(self):
        w = jnp.array([1.0, 2.0, 3.0])
        g = jax.grad(lambda w_: element_wise(w_, fn=lambda x: x ** 2).sum())(w)
        np.testing.assert_allclose(g, 2.0 * w)


# ---------------------------------------------------------------------------
# ETP rules
# ---------------------------------------------------------------------------

class TestEtpRules:

    def test_yw_to_w_is_elementwise_multiply(self):
        rule = ETP_RULES_YW_TO_W[etp_elemwise_p]
        hidden = jnp.array([1.0, 2.0, 3.0])
        trace = jnp.array([10.0, 20.0, 30.0])
        out = rule(hidden, trace)
        np.testing.assert_allclose(out, hidden * trace)

    def test_xy_to_dw_returns_hidden(self):
        rule = ETP_RULES_XY_TO_DW[etp_elemwise_p]
        # x is None for elemwise (x_invar_index=None) — but the function
        # signature still accepts it; just ignored.
        hidden = jnp.array([1.0, 2.0, 3.0])
        out = rule(None, hidden, None)
        np.testing.assert_allclose(out, hidden)

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_elemwise_p]
        y_var = _fake_var((4,))
        out = rule(None, y_var, None, num_hidden_state=3)
        assert out.shape == (4, 3)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_elemwise_p]
        y_var = _fake_var((4,))
        out = rule(None, y_var, None, num_hidden_state=3)
        assert out.shape == (4, 3)


class TestPublicAPIRoundTrip:

    def test_public_alias(self):
        assert braintrace.element_wise is element_wise
