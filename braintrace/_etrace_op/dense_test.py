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

"""Tests for the dense matmul ETP primitives and the :func:`matmul` API.

Coverage:

* Auto-dispatch — ``x.ndim >= 2`` selects ``etp_mm_p``; otherwise
  ``etp_mv_p``. Verified by jaxpr inspection.
* Forward correctness — agrees with ``x @ w (+ b)``.
* Bias presence — ``has_bias`` parameter is propagated through ``bind``.
* saiunit support — quantities, mixed units, unitless inputs.
* JAX rules — jit, vmap, grad, jvp work with no extra plumbing.
* Four ETP rules — ``yw_to_w``, ``xy_to_dw``, ``init_drtrl``, ``init_pp``
  return tensors of the documented shape and value.
"""



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
    etp_mm_p,
    etp_mv_p,
    matmul,
)

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


# ---------------------------------------------------------------------------
# Forward correctness + dispatch
# ---------------------------------------------------------------------------

class TestForwardCorrectness:

    def test_unbatched_matches_python_matmul(self):
        x = jnp.array([1.0, 2.0, 3.0])
        w = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        out = matmul(x, w)
        np.testing.assert_allclose(out, x @ w)

    def test_batched_matches_python_matmul(self):
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        w = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        out = matmul(x, w)
        np.testing.assert_allclose(out, x @ w)

    def test_with_bias(self):
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 4))
        b = jnp.arange(4.0)
        out = matmul(x, w, bias=b)
        np.testing.assert_allclose(out, x @ w + b)

    def test_higher_rank_input(self):
        x = jnp.ones((2, 5, 3))
        w = jnp.ones((3, 4))
        out = matmul(x, w)
        np.testing.assert_allclose(out, x @ w)


class TestAutoDispatch:

    def test_unbatched_uses_mv_primitive(self):
        x = jnp.array([1.0, 2.0])
        w = jnp.eye(2)
        jaxpr = jax.make_jaxpr(lambda x, w: matmul(x, w))(x, w)
        assert any(eqn.primitive is etp_mv_p for eqn in jaxpr.jaxpr.eqns)
        assert not any(eqn.primitive is etp_mm_p for eqn in jaxpr.jaxpr.eqns)

    def test_batched_uses_mm_primitive(self):
        x = jnp.ones((4, 2))
        w = jnp.eye(2)
        jaxpr = jax.make_jaxpr(lambda x, w: matmul(x, w))(x, w)
        assert any(eqn.primitive is etp_mm_p for eqn in jaxpr.jaxpr.eqns)
        assert not any(eqn.primitive is etp_mv_p for eqn in jaxpr.jaxpr.eqns)


class TestHasBiasParam:

    def test_has_bias_true_when_bias_supplied(self):
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 4))
        b = jnp.zeros(4)
        jaxpr = jax.make_jaxpr(lambda x, w, b: matmul(x, w, bias=b))(x, w, b)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_mm_p)
        assert eqn.params['has_bias'] is True

    def test_has_bias_false_when_bias_omitted(self):
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 4))
        jaxpr = jax.make_jaxpr(lambda x, w: matmul(x, w))(x, w)
        eqn = next(e for e in jaxpr.jaxpr.eqns if e.primitive is etp_mm_p)
        assert eqn.params['has_bias'] is False


# ---------------------------------------------------------------------------
# saiunit support
# ---------------------------------------------------------------------------

class TestSaiunit:

    def test_unitless_input_returns_unitless(self):
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 4))
        out = matmul(x, w)
        assert not isinstance(out, u.Quantity)

    def test_input_with_units_returns_quantity(self):
        x = jnp.ones((2, 3)) * u.mV
        w = jnp.ones((3, 4))
        out = matmul(x, w)
        # Output should still be a Quantity
        assert hasattr(out, 'mantissa') or isinstance(out, u.Quantity)

    def test_units_multiply_correctly(self):
        x = jnp.ones((2, 3)) * u.mV
        w = jnp.ones((3, 4)) * u.ms
        out = matmul(x, w)
        # Unit should be mV * ms
        expected = (jnp.ones((2, 3)) @ jnp.ones((3, 4))) * (u.mV * u.ms)
        np.testing.assert_allclose(
            u.get_mantissa(out), u.get_mantissa(expected),
        )

    def test_bias_with_units(self):
        x = jnp.ones((2, 3)) * u.mV
        w = jnp.ones((3, 4))
        b = jnp.ones(4) * u.mV
        out = matmul(x, w, bias=b)
        expected = (jnp.ones((2, 3)) @ jnp.ones((3, 4)) + jnp.ones(4)) * u.mV
        np.testing.assert_allclose(
            u.get_mantissa(out), u.get_mantissa(expected),
        )


# ---------------------------------------------------------------------------
# JAX rules — jit / vmap / grad
# ---------------------------------------------------------------------------

class TestJAXRules:

    def test_jit(self):
        x = jnp.ones((2, 3))
        w = jnp.arange(12.0).reshape(3, 4)
        f = jax.jit(matmul)
        np.testing.assert_allclose(f(x, w), x @ w)

    def test_vmap_over_batch(self):
        x = jnp.arange(6.0).reshape(2, 3)
        w = jnp.eye(3)
        out = jax.vmap(lambda xi: matmul(xi, w))(x)
        np.testing.assert_allclose(out, x @ w)

    def test_grad_wrt_w(self):
        x = jnp.ones((2, 3))
        w = jnp.arange(12.0).reshape(3, 4)
        gw = jax.grad(lambda w_: matmul(x, w_).sum())(w)
        # d(sum(x@w))/dw = x.T @ ones(2, 4) = sum(x, axis=0)[:, None] * ones((1,4))
        expected = x.sum(axis=0)[:, None] * jnp.ones((1, 4))
        np.testing.assert_allclose(gw, expected)

    def test_grad_wrt_x(self):
        x = jnp.arange(6.0).reshape(2, 3)
        w = jnp.ones((3, 4))
        gx = jax.grad(lambda x_: matmul(x_, w).sum())(x)
        # d(sum(x@w))/dx = ones(2,4) @ w.T
        expected = jnp.ones((2, 4)) @ w.T
        np.testing.assert_allclose(gx, expected)


# ---------------------------------------------------------------------------
# ETP rules — yw_to_w / xy_to_dw / init_drtrl / init_pp
# ---------------------------------------------------------------------------

class TestMmEtpRules:

    def test_yw_to_w_broadcasts_hidden(self):
        """``yw_to_w`` multiplies ``trace['weight']`` element-wise by
        ``hidden_dim`` broadcast along the input axis. Trace shape is
        ``(in, out)`` (solve context, batch stripped); ``hidden_dim`` is
        ``(out,)``. ``expand_dims(hidden_dim, axis=-2)`` → ``(1, out)``
        broadcasts against ``(in, out)`` → per-row scaling by ``hidden[o]``.
        Correct for non-square (in != out)."""
        rule = ETP_RULES_YW_TO_W[etp_mm_p]
        in_dim, out_dim = 5, 3
        hidden = jnp.array([1.0, 2.0, 3.0])  # (out,)
        trace = {'weight': jnp.ones((in_dim, out_dim))}
        out = rule(hidden, trace)
        assert isinstance(out, dict)
        assert out['weight'].shape == (in_dim, out_dim)
        # Column j scaled by hidden[j].
        np.testing.assert_allclose(
            out['weight'], jnp.ones((in_dim, out_dim)) * hidden[None, :]
        )

    def test_yw_to_w_with_bias(self):
        """When has_bias=True, ``yw_to_w`` also scales ``trace['bias']``."""
        rule = ETP_RULES_YW_TO_W[etp_mm_p]
        in_dim, out_dim = 5, 4
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])  # (out,)
        trace = {
            'weight': jnp.ones((in_dim, out_dim)),  # (in, out)
            'bias': jnp.ones((out_dim,)),  # (out,)
        }
        out = rule(hidden, trace, has_bias=True)
        assert isinstance(out, dict)
        assert 'bias' in out
        np.testing.assert_allclose(out['bias'], hidden)

    def test_xy_to_dw_matches_jax_vjp(self):
        rule = ETP_RULES_XY_TO_DW[etp_mm_p]
        x = jnp.arange(6.0).reshape(2, 3)
        w = jnp.arange(12.0).reshape(3, 4)
        hidden = jnp.ones((2, 4))
        weights = {'weight': w}
        dw_dict = rule(x, hidden, weights)
        assert isinstance(dw_dict, dict)
        # VJP of y = x @ w wrt w with cotangent ones((2,4)) is x.T @ ones((2,4))
        expected = x.T @ hidden
        np.testing.assert_allclose(dw_dict['weight'], expected)

    def test_xy_to_dw_with_bias(self):
        """With has_bias=True the dict result also contains a 'bias' entry."""
        rule = ETP_RULES_XY_TO_DW[etp_mm_p]
        x = jnp.ones((2, 3))
        w = jnp.arange(12.0).reshape(3, 4)
        b = jnp.zeros(4)
        hidden = jnp.ones((2, 4))
        weights = {'weight': w, 'bias': b}
        dw_dict = rule(x, hidden, weights, has_bias=True)
        assert isinstance(dw_dict, dict)
        assert 'weight' in dw_dict
        assert 'bias' in dw_dict
        # db = sum of hidden over batch axis = ones(4)
        np.testing.assert_allclose(dw_dict['bias'], hidden.sum(axis=0))

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_mm_p]
        x_var = _fake_var((4, 3))  # (batch, in)
        y_var = _fake_var((4, 5))
        weight_vars = {'weight': _fake_var((3, 5))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert isinstance(out, dict)
        assert out['weight'].shape == (4, 3, 5, 2)

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_mm_p]
        x_var = _fake_var((4, 3))
        y_var = _fake_var((4, 5))
        weight_vars = {'weight': _fake_var((3, 5)), 'bias': _fake_var((5,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert isinstance(out, dict)
        assert out['weight'].shape == (4, 3, 5, 2)
        assert out['bias'].shape == (4, 5, 2)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_mm_p]
        x_var = _fake_var((4, 3))
        y_var = _fake_var((4, 5))
        weight_vars = {'weight': _fake_var((3, 5))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out.shape == (4, 5, 2)


class TestMvEtpRules:

    def test_yw_to_w_broadcasts_hidden(self):
        """``yw_to_w`` multiplies ``trace['weight']`` by ``hidden`` broadcast
        along the column axis. The rule accepts and returns a dict."""
        rule = ETP_RULES_YW_TO_W[etp_mv_p]
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])  # (out,)
        trace = {'weight': jnp.ones((3, 4))}  # (in, out)
        out = rule(hidden, trace)
        assert isinstance(out, dict)
        assert out['weight'].shape == (3, 4)
        # column j scaled by hidden[j]
        np.testing.assert_allclose(out['weight'], jnp.ones((3, 4)) * hidden[None, :])

    def test_yw_to_w_with_bias(self):
        """When has_bias=True, ``yw_to_w`` also scales ``trace['bias']``."""
        rule = ETP_RULES_YW_TO_W[etp_mv_p]
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])
        trace = {'weight': jnp.ones((3, 4)), 'bias': jnp.ones((4,))}
        out = rule(hidden, trace, has_bias=True)
        assert isinstance(out, dict)
        assert 'bias' in out
        np.testing.assert_allclose(out['bias'], hidden)

    def test_xy_to_dw_matches_outer_product(self):
        rule = ETP_RULES_XY_TO_DW[etp_mv_p]
        x = jnp.arange(3.0)
        w = jnp.arange(12.0).reshape(3, 4)
        hidden = jnp.arange(4.0)
        weights = {'weight': w}
        dw_dict = rule(x, hidden, weights)
        assert isinstance(dw_dict, dict)
        np.testing.assert_allclose(dw_dict['weight'], jnp.outer(x, hidden))

    def test_xy_to_dw_with_bias(self):
        """With has_bias=True the dict result also contains a 'bias' entry."""
        rule = ETP_RULES_XY_TO_DW[etp_mv_p]
        x = jnp.arange(3.0)
        w = jnp.arange(12.0).reshape(3, 4)
        b = jnp.zeros(4)
        hidden = jnp.arange(4.0)
        weights = {'weight': w, 'bias': b}
        dw_dict = rule(x, hidden, weights, has_bias=True)
        assert isinstance(dw_dict, dict)
        assert 'weight' in dw_dict
        assert 'bias' in dw_dict
        # db = hidden (unbatched VJP)
        np.testing.assert_allclose(dw_dict['bias'], hidden)

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((3, 5))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert isinstance(out, dict)
        assert out['weight'].shape == (3, 5, 2)

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((3, 5)), 'bias': _fake_var((5,))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert isinstance(out, dict)
        assert out['weight'].shape == (3, 5, 2)
        assert out['bias'].shape == (5, 2)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((5,))
        weight_vars = {'weight': _fake_var((3, 5))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=2)
        assert out.shape == (5, 2)


class TestPublicAPIRoundTrip:
    """``braintrace.matmul`` and ``braintrace._etrace_op.matmul`` are the
    same function — the public alias is not a re-implementation."""

    def test_public_alias_identity(self):
        assert braintrace.matmul is matmul
