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
* ``element_wise(w)`` (default ``weight_fn=None``) round-trips ``w``.
* ``element_wise(w, weight_fn=lambda w: 2*w)`` applies ``weight_fn`` *inside*
  the primitive.
* brainunit quantities pass through.
* JAX rules — jit, vmap, grad, jvp.
* Four ETP rules return the documented values / shapes.
"""



from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

import brainstate
import braintrace
from braintrace._op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    GRADIENT_ENABLED_PRIMITIVES,
    element_wise,
    etp_elemwise_p,
    get_fast_path_rules,
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
        out = element_wise(w, weight_fn=lambda x: x * 10.0)
        np.testing.assert_allclose(out, w * 10.0)

    def test_nonlinear_fn(self):
        w = jnp.array([0.0, 0.5, 1.0])
        out = element_wise(w, weight_fn=jnp.sin)
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

    def test_fn_carried_as_primitive_param(self):
        """weight_fn is now a primitive param, not a traced JAX op.
        The jaxpr contains exactly one etp_elemwise equation, and the
        weight_fn callable is stored inside its params dict."""
        w = jnp.ones((3,))
        jaxpr = jax.make_jaxpr(lambda w: element_wise(w, weight_fn=lambda x: x * 2))(w)
        eqns = list(jaxpr.jaxpr.eqns)
        assert len(eqns) == 1
        assert eqns[0].primitive is etp_elemwise_p
        assert eqns[0].params.get('weight_fn') is not None


# ---------------------------------------------------------------------------
# brainunit
# ---------------------------------------------------------------------------

class TestBrainunit:

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
        out = element_wise(w, weight_fn=lambda x: x * 2)
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
        g = jax.grad(lambda w_: element_wise(w_, weight_fn=lambda x: x ** 2).sum())(w)
        np.testing.assert_allclose(g, 2.0 * w)


# ---------------------------------------------------------------------------
# ETP rules
# ---------------------------------------------------------------------------

class TestEtpRules:

    def test_yw_to_w_is_elementwise_multiply(self):
        rule = ETP_RULES_YW_TO_W[etp_elemwise_p]
        hidden = jnp.array([1.0, 2.0, 3.0])
        trace = {'weight': jnp.array([10.0, 20.0, 30.0])}
        out = rule(hidden, trace)
        assert isinstance(out, dict)
        np.testing.assert_allclose(out['weight'], hidden * trace['weight'])

    def test_xy_to_dw_returns_hidden(self):
        rule = ETP_RULES_XY_TO_DW[etp_elemwise_p]
        # x is None for elemwise (x_invar_index=None) — but the function
        # signature still accepts it; just ignored.
        hidden = jnp.array([1.0, 2.0, 3.0])
        weights = {'weight': None}  # Not actually used in the body
        out = rule(None, hidden, weights)
        assert isinstance(out, dict)
        np.testing.assert_allclose(out['weight'], hidden)

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_elemwise_p]
        y_var = _fake_var((4,))
        out = rule(None, y_var, None, num_hidden_state=3)
        assert isinstance(out, dict)
        assert out['weight'].shape == (4, 3)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_elemwise_p]
        y_var = _fake_var((4,))
        out = rule(None, y_var, None, num_hidden_state=3)
        # init_pp returns a single array, not a dict
        assert out.shape == (4, 3)


class TestPublicAPIRoundTrip:

    def test_public_alias(self):
        assert braintrace.element_wise is element_wise


# ---------------------------------------------------------------------------
# Dict-based rule API
# ---------------------------------------------------------------------------

class TestElemwiseDictRules:

    def test_declares_trainable_invars(self):
        from braintrace._op import get_trainable_invars
        assert get_trainable_invars(etp_elemwise_p, {}) == {'weight': 0}

    def test_xy_to_dw_returns_dict(self):
        hidden_dim = jnp.ones((3, 4))
        weights = {'weight': jnp.ones((3, 4))}
        out = ETP_RULES_XY_TO_DW[etp_elemwise_p](None, hidden_dim, weights)
        assert isinstance(out, dict)
        assert set(out.keys()) == {'weight'}
        assert out['weight'].shape == (3, 4)

    def test_yw_to_w_returns_dict(self):
        hidden_dim = jnp.full((3, 4), 2.0)
        trace = {'weight': jnp.full((3, 4), 5.0)}
        out = ETP_RULES_YW_TO_W[etp_elemwise_p](hidden_dim, trace)
        assert isinstance(out, dict)
        assert out['weight'].shape == (3, 4)
        # 5 * 2 = 10
        assert float(out['weight'][0, 0]) == 10.0


class TestElemwiseWeightFnInside:

    def test_forward_applies_weight_fn(self):
        import brainstate
        w = brainstate.random.randn(5)
        out = element_wise(w, weight_fn=lambda x: x * 10.0)
        np.testing.assert_allclose(out, w * 10.0, atol=1e-5)

    def test_default_is_identity(self):
        import brainstate
        w = brainstate.random.randn(5)
        np.testing.assert_allclose(element_wise(w), w, atol=1e-6)

    def test_xy_to_dw_applies_fn_derivative(self):
        """xy_to_dw must return ∂h/∂w = vjp(weight_fn)(hidden) — NOT hidden itself."""
        import brainstate
        from braintrace._op import ETP_RULES_XY_TO_DW, etp_elemwise_p
        rule = ETP_RULES_XY_TO_DW[etp_elemwise_p]
        w = brainstate.random.randn(5)
        hidden = brainstate.random.randn(5)
        out = rule(None, hidden, {'weight': w}, weight_fn=lambda x: x ** 2)
        _, vjp = jax.vjp(lambda x: x ** 2, w)
        np.testing.assert_allclose(out['weight'], vjp(hidden)[0], atol=1e-5)

    def test_xy_to_dw_identity_returns_hidden(self):
        import brainstate
        from braintrace._op import ETP_RULES_XY_TO_DW, etp_elemwise_p
        rule = ETP_RULES_XY_TO_DW[etp_elemwise_p]
        hidden = brainstate.random.randn(5)
        out = rule(None, hidden, {'weight': brainstate.random.randn(5)}, weight_fn=None)
        np.testing.assert_allclose(out['weight'], hidden, atol=1e-6)

    def test_xy_to_dw_handles_batched_cotangent(self):
        """A ``hidden_dim`` with leading batch axis must broadcast against the
        unbatched per-element weight derivative.

        Under a batched hidden state the cotangent ``∂h/∂y`` acquires a leading
        batch axis ``(batch, n)`` while the weight stays ``(n,)``. Because the
        op is diagonal, ``∂h/∂w = hidden_dim ⊙ weight_fn'(w)`` broadcasts over
        the leading axes. The naive ``vjp_fn(hidden_dim)`` rejects the batched
        cotangent (output shape is ``(n,)``), so the rule must use the
        per-element derivative.
        """
        import brainstate
        from braintrace._op import ETP_RULES_XY_TO_DW, etp_elemwise_p
        rule = ETP_RULES_XY_TO_DW[etp_elemwise_p]
        w = brainstate.random.randn(4)
        hidden = brainstate.random.randn(2, 4)  # (batch, n) cotangent
        out = rule(None, hidden, {'weight': w}, weight_fn=lambda x: x ** 2)
        # f'(w) = 2w, broadcast over the batch axis.
        expected = hidden * (2.0 * w)
        assert out['weight'].shape == (2, 4)
        np.testing.assert_allclose(out['weight'], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Exactness gate: D_RTRL == BPTT when element_wise carries weight_fn=tanh
# ---------------------------------------------------------------------------

class TestElemwiseWeightFnExactness:
    """element_wise's own weight gradient must include weight_fn' (D_RTRL==BPTT)."""

    @staticmethod
    def _factory(weight_fn):
        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.nh = 4
                self.W = brainstate.ParamState(brainstate.random.randn(2 + 4, 4) * 0.2)
                self.alpha = brainstate.ParamState(brainstate.random.randn(4) * 0.5)

            def init_state(self, batch_size=None, **kw):
                size = (self.nh,) if batch_size is None else (batch_size, self.nh)
                self.h = brainstate.HiddenState(jnp.zeros(size))

            def update(self, x):
                a = braintrace.element_wise(self.alpha.value, weight_fn=weight_fn)
                xh = jnp.concatenate([x.reshape(1, -1), self.h.value], axis=-1)
                self.h.value = jnp.tanh(a * braintrace.matmul(xh, self.W.value))
                return self.h.value

        def factory():
            brainstate.random.seed(0)
            return Cell()

        return factory

    def test_d_rtrl_matches_bptt_with_elemwise_weight_fn(self):
        from braintrace._algorithm.oracle import (
            bptt_param_gradients, online_param_gradients, assert_param_gradients_close,
        )
        factory = self._factory(weight_fn=jnp.tanh)
        brainstate.random.seed(1)
        inputs = brainstate.random.randn(6, 2)
        bptt = bptt_param_gradients(factory, inputs)
        online = online_param_gradients(
            factory, inputs, algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step')
        )
        assert_param_gradients_close(online, bptt, atol=1e-4)


# ---------------------------------------------------------------------------
# Closed-form param-dim D-RTRL fast-path kernels (operator-layer bundle)
# ---------------------------------------------------------------------------

class TestElemwiseFastPath:
    """The diagonal closed-form fast-path bundle registered on
    ``etp_elemwise_p``.

    The elemwise op is diagonal and carries no ``x`` input and no bias, so
    its kernels operate on a single ``'weight'`` key only. The gate keys off
    ``weight_fn`` alone (no ``bias_fn`` for this op).
    """

    def test_fast_path_registered_on_elemwise(self):
        assert get_fast_path_rules(etp_elemwise_p) is not None

    def test_fast_applicable_true_without_weight_fn(self):
        rules = get_fast_path_rules(etp_elemwise_p)
        assert rules.applicable({'weight_fn': None}) is True

    def test_fast_applicable_false_with_weight_fn(self):
        rules = get_fast_path_rules(etp_elemwise_p)
        assert rules.applicable({'weight_fn': jnp.tanh}) is False

    def test_fast_instant_returns_df(self):
        rules = get_fast_path_rules(etp_elemwise_p)
        df = brainstate.random.randn(4, 3)
        out = rules.instant(None, df, False)
        assert set(out.keys()) == {'weight'}
        np.testing.assert_allclose(out['weight'], df)

    def test_fast_recurrent_num_state_1_equals_general_einsum(self):
        rules = get_fast_path_rules(etp_elemwise_p)
        # diag (*var, 1, 1); trace (*var, 1).
        var = 4
        diag = brainstate.random.randn(var, 1, 1)
        trace = {'weight': brainstate.random.randn(var, 1)}
        fast = rules.recurrent(diag, trace, 1)
        general = jnp.einsum('...ab,...b->...a', diag, trace['weight'])
        np.testing.assert_allclose(fast['weight'], general, atol=1e-6)

    def test_fast_solve_elemwise_fold_batch(self):
        rules = get_fast_path_rules(etp_elemwise_p)
        batch, var, n_state = 2, 4, 3
        diag_like = brainstate.random.randn(batch, var, n_state)
        etrace = {'weight': brainstate.random.randn(batch, var, n_state)}
        folded = rules.solve(diag_like, etrace, fold_batch=True)
        unfolded = rules.solve(diag_like, etrace, fold_batch=False)
        np.testing.assert_allclose(
            folded['weight'], unfolded['weight'].sum(axis=0), atol=1e-5
        )
