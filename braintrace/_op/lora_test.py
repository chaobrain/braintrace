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
``alpha``. The ETP gradient state is a pytree keyed by ``'lora_b'`` /
``'lora_a'`` (and optionally ``'bias'``). Under param-dim D-RTRL the
``'lora_b'`` *trace* entry holds the dense-style effective-weight trace
for ``W_eff = alpha * b_fn(B) @ a_fn(A)`` (see ``lora.py``); the
gradient returned for ``'lora_b'`` is still ``(in, rank)``-shaped.

Note: :class:`braintrace.nn.LoRA` names its ``ParamState`` leaves the
other way round (its ``'lora_a'`` is the ``(in, rank)`` factor feeding
this primitive's ``lora_b`` operand); routing is by dataflow, not name.
"""



from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
import pytest

import braintrace
from braintrace._op import (
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
        B = jnp.arange(6.0).reshape(3, 2)  # in=3, rank=2
        A = jnp.arange(8.0).reshape(2, 4)  # rank=2, out=4
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
# Rank guard (M5) — mirrors the dense ``matmul`` guard: every ETP trace rule
# assumes a (batch, in) layout, so rank>2 ``x`` must be rejected rather than
# silently running through ``etp_lora_mm_p``.
# ---------------------------------------------------------------------------

class TestRankGuard:

    def test_rank3_input_raises_valueerror(self):
        x = jnp.ones((2, 5, 3))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        with pytest.raises(ValueError, match=r'ndim'):
            lora_matmul(x, B, A)

    def test_rank4_input_raises_valueerror(self):
        x = jnp.ones((2, 5, 6, 3))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        with pytest.raises(ValueError, match=r'ndim'):
            lora_matmul(x, B, A)

    def test_rank1_input_still_accepted(self):
        x = jnp.ones((3,))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        out = lora_matmul(x, B, A)
        np.testing.assert_allclose(out, x @ B @ A)

    def test_rank2_input_still_accepted(self):
        x = jnp.ones((2, 3))
        B = jnp.ones((3, 2))
        A = jnp.ones((2, 4))
        out = lora_matmul(x, B, A)
        np.testing.assert_allclose(out, x @ B @ A)


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
# brainunit
# ---------------------------------------------------------------------------

class TestBrainunit:

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
        """``yw_to_w`` broadcasts ``hidden`` across the leading matrix axis of
        BOTH traces via ``expand_dims(hidden, axis=-2)``: the ``'lora_b'``
        entry is the effective-weight trace ``(in, out)`` (dense-style
        recurrence), the ``'lora_a'`` entry the factor trace ``(rank, out)``.

        Two equivalent shapes are tested:
          * ``(out,)`` — per-slice context (batch stripped by the algorithm's vmap)
          * ``(batch, out)`` — as called from ``_update_param_dim_etrace_scan_fn``
        """
        rule = ETP_RULES_YW_TO_W[etp_lora_mm_p]

        # Test 1: slice context — hidden=(out=4,)
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])  # (out=4,)
        trace_W = jnp.ones((3, 4))  # effective-weight trace (in=3, out=4)
        trace_A = jnp.ones((2, 4))  # (rank=2, out=4)
        out = rule(hidden, {'lora_b': trace_W, 'lora_a': trace_A})
        assert set(out.keys()) == {'lora_b', 'lora_a'}
        # Both traces are scaled along the output axis (dense y -> W link).
        np.testing.assert_allclose(out['lora_b'], trace_W * hidden[None, :])
        np.testing.assert_allclose(out['lora_a'], trace_A * hidden[None, :])

        # Test 2: batched trace-update context — hidden=(batch=1, out=4)
        hidden_b = jnp.arange(1.0, 5.0).reshape(1, 4)  # (batch=1, out=4)
        trace_W_b = jnp.arange(12.0).reshape(1, 3, 4)  # (batch=1, in=3, out=4)
        trace_A_b = jnp.arange(8.0).reshape(1, 2, 4)  # (batch=1, rank=2, out=4)
        out_b = rule(hidden_b, {'lora_b': trace_W_b, 'lora_a': trace_A_b})
        np.testing.assert_allclose(out_b['lora_b'], trace_W_b * hidden_b[:, None, :])
        np.testing.assert_allclose(out_b['lora_a'], trace_A_b * hidden_b[:, None, :])

    def test_xy_to_dw_pytree_and_values(self):
        rule = ETP_RULES_XY_TO_DW[etp_lora_mm_p]
        x = jnp.ones((2, 3))
        B = jnp.arange(6.0).reshape(3, 2)
        A = jnp.arange(8.0).reshape(2, 4)
        hidden = jnp.ones((2, 4))
        weights = {'lora_b': B, 'lora_a': A}
        d = rule(x, hidden, weights, alpha=1.0)
        assert set(d.keys()) == {'lora_b', 'lora_a'}
        # Compare to pure JAX VJP.
        _, vjp_fn = jax.vjp(lambda B_, A_: x @ B_ @ A_, B, A)
        ref_dB, ref_dA = vjp_fn(hidden)
        np.testing.assert_allclose(d['lora_b'], ref_dB)
        np.testing.assert_allclose(d['lora_a'], ref_dA)

    def test_xy_to_dw_respects_alpha(self):
        rule = ETP_RULES_XY_TO_DW[etp_lora_mm_p]
        x = jnp.ones((2, 3))
        B = jnp.arange(6.0).reshape(3, 2)
        A = jnp.arange(8.0).reshape(2, 4)
        hidden = jnp.ones((2, 4))
        weights = {'lora_b': B, 'lora_a': A}
        d1 = rule(x, hidden, weights, alpha=1.0)
        d_half = rule(x, hidden, weights, alpha=0.5)
        np.testing.assert_allclose(d_half['lora_b'], d1['lora_b'] * 0.5)
        np.testing.assert_allclose(d_half['lora_a'], d1['lora_a'] * 0.5)

    def test_init_drtrl_shape(self):
        """``'lora_b'`` allocates the effective-weight trace
        ``(batch, in, out, n_state)`` — NOT the ``(in, rank)`` factor shape."""
        rule = ETP_RULES_INIT_DRTRL[etp_lora_mm_p]
        x_var = _fake_var((2, 3))
        y_var = _fake_var((2, 4))
        weight_vars = {'lora_b': _fake_var((3, 2)), 'lora_a': _fake_var((2, 4))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=5)
        assert out['lora_b'].shape == (2, 3, 4, 5)
        assert out['lora_a'].shape == (2, 2, 4, 5)

    def test_init_drtrl_shape_with_bias(self):
        rule = ETP_RULES_INIT_DRTRL[etp_lora_mm_p]
        x_var = _fake_var((2, 3))
        y_var = _fake_var((2, 4))
        weight_vars = {
            'lora_b': _fake_var((3, 2)),
            'lora_a': _fake_var((2, 4)),
            'bias': _fake_var((4,)),
        }
        out = rule(x_var, y_var, weight_vars, num_hidden_state=5)
        assert out['lora_b'].shape == (2, 3, 4, 5)
        assert out['lora_a'].shape == (2, 2, 4, 5)
        assert out['bias'].shape == (2, 4, 5)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_lora_mm_p]
        x_var = _fake_var((2, 3))
        y_var = _fake_var((2, 4))
        weight_vars = {'lora_b': _fake_var((3, 2)), 'lora_a': _fake_var((2, 4))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=5)
        assert out.shape == (2, 4, 5)


class TestLoraMvEtpRules:

    def test_yw_to_w_pytree_structure(self):
        """mv-variant: ``expand_dims(hidden, axis=-2)`` broadcasts ``hidden``
        along the output axis of BOTH the effective-weight trace ``(in, out)``
        and the ``(rank, out)`` A-trace."""
        rule = ETP_RULES_YW_TO_W[etp_lora_mv_p]
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])  # (out=4,)
        trace_W = jnp.ones((3, 4))  # effective-weight trace (in=3, out=4)
        trace_A = jnp.ones((2, 4))  # (rank=2, out=4)
        out = rule(hidden, {'lora_b': trace_W, 'lora_a': trace_A})
        assert set(out.keys()) == {'lora_b', 'lora_a'}
        np.testing.assert_allclose(out['lora_b'], trace_W * hidden[None, :])
        np.testing.assert_allclose(out['lora_a'], trace_A * hidden[None, :])

    def test_init_drtrl_shape(self):
        """``'lora_b'`` allocates the effective-weight trace
        ``(in, out, n_state)`` with no batch axis anywhere."""
        rule = ETP_RULES_INIT_DRTRL[etp_lora_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((4,))
        weight_vars = {'lora_b': _fake_var((3, 2)), 'lora_a': _fake_var((2, 4))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=5)
        assert out['lora_b'].shape == (3, 4, 5)
        assert out['lora_a'].shape == (2, 4, 5)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_lora_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((4,))
        weight_vars = {'lora_b': _fake_var((3, 2)), 'lora_a': _fake_var((2, 4))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=5)
        assert out.shape == (4, 5)


class TestLoraInstantSolveDrtrlRules:
    """The param-dim D-RTRL overrides: trace-structured instantaneous term
    and solve-time chaining of the effective-weight trace to the raw B."""

    def _weights(self, with_bias=False):
        B = jnp.arange(6.0).reshape(3, 2) * 0.1  # (in=3, rank=2)
        A = jnp.arange(8.0).reshape(2, 4) * 0.1  # (rank=2, out=4)
        w = {'lora_b': B, 'lora_a': A}
        if with_bias:
            w['bias'] = jnp.arange(4.0) * 0.1
        return w

    def test_registered_for_both_primitives(self):
        from braintrace._op._registries import (
            get_instant_drtrl_rule, get_solve_drtrl_rule,
        )
        for prim in (etp_lora_mm_p, etp_lora_mv_p):
            assert get_instant_drtrl_rule(prim) is not None
            assert get_solve_drtrl_rule(prim) is not None

    def test_instant_lora_b_is_outer_product_without_alpha(self):
        """The effective-weight increment is ``outer(x, df)`` — alpha and the
        factor transforms live inside W_eff and enter only at solve time."""
        from braintrace._op._registries import get_instant_drtrl_rule
        rule = get_instant_drtrl_rule(etp_lora_mm_p)
        x = jnp.array([1.0, 2.0, 3.0])  # (in=3,)
        df = jnp.array([0.5, -1.0, 2.0, 0.25])  # (out=4,)
        out = rule(x, df, self._weights(), alpha=0.5)
        assert out['lora_b'].shape == (3, 4)
        np.testing.assert_allclose(out['lora_b'], jnp.outer(x, df))

    def test_instant_lora_a_and_bias_match_xy_to_dw(self):
        """The A / bias entries reuse the exact param-shaped pullbacks."""
        from braintrace._op._registries import get_instant_drtrl_rule
        rule = get_instant_drtrl_rule(etp_lora_mm_p)
        xy = ETP_RULES_XY_TO_DW[etp_lora_mm_p]
        x = jnp.array([1.0, 2.0, 3.0])
        df = jnp.array([0.5, -1.0, 2.0, 0.25])
        w = self._weights(with_bias=True)
        params = dict(alpha=0.5, has_bias=True, b_fn=None, a_fn=jnp.tanh,
                      bias_fn=None)
        out = rule(x, df, w, **params)
        ref = xy(x, df, w, **params)
        np.testing.assert_allclose(out['lora_a'], ref['lora_a'])
        np.testing.assert_allclose(out['bias'], ref['bias'])

    def test_solve_chains_effective_weight_to_raw_B(self):
        """``lora_b`` gradient is ``alpha * (g * eps_W) @ a_fn(A)^T`` pulled
        back through ``b_fn``; ``lora_a`` / ``bias`` match the generic path."""
        from braintrace._op._registries import get_solve_drtrl_rule
        rule = get_solve_drtrl_rule(etp_lora_mm_p)
        w = self._weights(with_bias=True)
        dg = jnp.array([0.5, -1.0, 2.0, 0.25])  # (out=4,)
        trace = {
            'lora_b': jnp.arange(12.0).reshape(3, 4) * 0.1,  # eps_W (in, out)
            'lora_a': jnp.arange(8.0).reshape(2, 4) * 0.2,
            'bias': jnp.arange(4.0) * 0.3,
        }
        alpha = 2.0
        out = rule(dg, trace, w, alpha=alpha, has_bias=True)
        G = trace['lora_b'] * dg[None, :]
        np.testing.assert_allclose(out['lora_b'], alpha * (G @ w['lora_a'].T))
        np.testing.assert_allclose(out['lora_a'], trace['lora_a'] * dg[None, :])
        np.testing.assert_allclose(out['bias'], trace['bias'] * dg)
        assert out['lora_b'].shape == (3, 2)  # param-shaped, not trace-shaped

    def test_solve_pulls_back_through_b_fn_vjp(self):
        from braintrace._op._registries import get_solve_drtrl_rule
        rule = get_solve_drtrl_rule(etp_lora_mv_p)
        w = self._weights()
        dg = jnp.array([0.5, -1.0, 2.0, 0.25])
        trace = {
            'lora_b': jnp.arange(12.0).reshape(3, 4) * 0.1,
            'lora_a': jnp.arange(8.0).reshape(2, 4) * 0.2,
        }
        out = rule(dg, trace, w, alpha=1.0, b_fn=jnp.tanh, a_fn=jnp.tanh)
        G = trace['lora_b'] * dg[None, :]
        cot = G @ jnp.tanh(w['lora_a']).T
        # d tanh(B)/dB = 1 - tanh(B)^2, elementwise
        expected = (1.0 - jnp.tanh(w['lora_b']) ** 2) * cot
        np.testing.assert_allclose(out['lora_b'], expected, rtol=1e-6)

    def test_solve_collapses_leading_broadcast_axis_on_signal(self):
        """A batched hidden state feeding the unbatched mv primitive hands the
        solve rule a ``(1, out)`` signal; the ``lora_b`` gradient must still
        come back param-shaped (the leading axis is summed, exact by
        linearity), while ``lora_a`` keeps it for the algorithm's trailing
        reduction — mirroring the generic path."""
        from braintrace._op._registries import get_solve_drtrl_rule
        rule = get_solve_drtrl_rule(etp_lora_mv_p)
        w = self._weights()
        dg = jnp.array([[0.5, -1.0, 2.0, 0.25]])  # (1, out)
        trace = {
            'lora_b': jnp.arange(12.0).reshape(3, 4) * 0.1,
            'lora_a': jnp.arange(8.0).reshape(2, 4) * 0.2,
        }
        out = rule(dg, trace, w, alpha=1.0)
        ref = rule(dg[0], trace, w, alpha=1.0)
        assert out['lora_b'].shape == (3, 2)
        np.testing.assert_allclose(out['lora_b'], ref['lora_b'])
        assert out['lora_a'].shape == (1, 2, 4)
        np.testing.assert_allclose(out['lora_a'][0], ref['lora_a'])


class TestPublicAPIRoundTrip:

    def test_public_alias(self):
        assert braintrace.lora_matmul is lora_matmul


# ---------------------------------------------------------------------------
# End-to-end online learning: D-RTRL vs BPTT oracle (exactly-diagonal regime)
# ---------------------------------------------------------------------------

class TestLoRAOnlineLearningExact:
    """Single-step D-RTRL must reproduce BPTT exactly for every LoRA factor.

    The model is exactly diagonal (leaky-integrator dynamics
    ``h <- leak * h + drive``), so D-RTRL's diagonal approximation is exact
    and every parameter gradient must match a BPTT oracle element-wise
    (``rel < 1e-10`` in float64). This covers the audit finding that
    ``lora_b`` gradients were wrong even at T=1 (rel err ~4) because the
    B-factor trace was propagated unchanged through ``yw_to_w``.
    """

    LEAK = 0.5
    N_IN, N_REC, RANK = 3, 4, 2
    TOL = 1e-10

    def _make_factory(self, *, alpha=1.0, with_bias=False, b_fn=None,
                      batched=True, seed=0):
        import brainstate

        n_in, n_rec, rank = self.N_IN, self.N_REC, self.RANK
        leak = self.LEAK
        brainstate.random.seed(seed)
        B0 = 0.1 * brainstate.random.randn(n_in, rank)
        A0 = 0.1 * brainstate.random.randn(rank, n_rec)
        bias0 = 0.05 * brainstate.random.randn(n_rec) if with_bias else None
        w0 = 0.1 * brainstate.random.randn(n_in, n_rec)

        # The mv (unbatched) primitive carries no batch axis anywhere, so its
        # hidden state must be unbatched too; the mm variant pairs a batched
        # input with a (1, n_rec) hidden state.
        h_shape = (1, n_rec) if batched else (n_rec,)

        def factory():
            class Net(brainstate.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.B = brainstate.ParamState(B0)
                    self.A = brainstate.ParamState(A0)
                    if with_bias:
                        self.bias = brainstate.ParamState(bias0)
                    self.h = brainstate.HiddenState(jnp.zeros(h_shape))

                def update(self, x):
                    drive = braintrace.lora_matmul(
                        x, self.B.value, self.A.value,
                        alpha=alpha,
                        bias=self.bias.value if with_bias else None,
                        b_fn=b_fn,
                    ) + x @ w0
                    self.h.value = leak * self.h.value + drive
                    return self.h.value

            return Net()

        return factory

    @staticmethod
    def _rel_err(a, b):
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        denom = jnp.maximum(jnp.abs(a).max(), 1e-12)
        return float(jnp.abs(a - b).max() / denom)

    def _assert_exact(self, **factory_kwargs):
        import brainstate
        from braintrace._algorithm.oracle import (
            bptt_param_gradients,
            online_param_gradients_singlestep_naive,
        )

        batched = factory_kwargs.get('batched', True)
        with brainstate.environ.context(precision=64):
            for T in (1, 2, 4):
                factory = self._make_factory(**factory_kwargs)
                brainstate.random.seed(42)
                if batched:
                    xs = 0.3 * brainstate.random.randn(T, 1, self.N_IN)
                else:
                    xs = 0.3 * brainstate.random.randn(T, self.N_IN)
                g_bptt = bptt_param_gradients(factory, xs)
                g_online = online_param_gradients_singlestep_naive(
                    factory, xs,
                    algo_factory=lambda m: braintrace.D_RTRL(
                        m, vjp_method='single-step'
                    ),
                )
                for key in g_bptt:
                    rel = self._rel_err(g_bptt[key], g_online[key])
                    assert rel < self.TOL, (
                        f'D-RTRL diverges from BPTT for {key} at T={T}: '
                        f'max_rel_err={rel:.3e}'
                    )

    def test_lora_mm_exact_plain(self):
        """Batched (mm) variant: B and A gradients exact at T=1/2/4."""
        self._assert_exact(alpha=1.0, with_bias=False)

    def test_lora_mm_exact_alpha_and_bias(self):
        """alpha=2.0 scaling plus a trainable bias: all three factors exact."""
        self._assert_exact(alpha=2.0, with_bias=True)

    def test_lora_mm_exact_b_fn_tanh(self):
        """B-factor transform hook: gradient chained through tanh VJP exactly."""
        self._assert_exact(alpha=1.0, with_bias=False, b_fn=jnp.tanh)

    def test_lora_mv_exact_plain(self):
        """Unbatched (mv) variant: no batch axis anywhere in the trace."""
        self._assert_exact(alpha=1.0, with_bias=False, batched=False)


# ---------------------------------------------------------------------------
# Per-factor transform functions: b_fn / a_fn / bias_fn
# ---------------------------------------------------------------------------

class TestLoraFactorFns:

    def test_forward_applies_factor_fns(self):
        import brainstate
        x = brainstate.random.randn(4, 8)
        B = brainstate.random.randn(8, 2)
        A = brainstate.random.randn(2, 4)
        out = braintrace.lora_matmul(x, B, A, alpha=0.5,
                                     b_fn=lambda b: b ** 2, a_fn=jnp.tanh)
        ref = 0.5 * (x @ (B ** 2) @ jnp.tanh(A))
        np.testing.assert_allclose(out, ref, atol=1e-4)

    def test_xy_to_dw_matches_vjp_through_factor_fns(self):
        import brainstate
        from braintrace._op import ETP_RULES_XY_TO_DW, etp_lora_mm_p
        from braintrace._op.op_rule_oracle import assert_xy_to_dw_matches_vjp
        rule = ETP_RULES_XY_TO_DW[etp_lora_mm_p]
        x = brainstate.random.randn(4, 8)
        weights = {'lora_b': brainstate.random.randn(8, 2),
                   'lora_a': brainstate.random.randn(2, 4)}
        hidden = brainstate.random.randn(4, 4)
        params = {'alpha': 0.5, 'has_bias': False,
                  'b_fn': lambda b: b ** 2, 'a_fn': jnp.tanh, 'bias_fn': None}

        def impl(w):
            return 0.5 * (x @ (w['lora_b'] ** 2) @ jnp.tanh(w['lora_a']))

        assert_xy_to_dw_matches_vjp(rule=rule, impl=impl, x=x, hidden_dim=hidden,
                                    weights=weights, params=params, atol=1e-4)


# ---------------------------------------------------------------------------
# Audit Task 11 (T3): first-principles ``instant_drtrl`` / ``solve_drtrl``
# from ``jax.jacobian``
# ---------------------------------------------------------------------------

class TestInstantSolveDrtrlFirstPrinciplesFromJacobian:
    """Derive ``_lora_instant_drtrl``'s ``'lora_b'`` trace and
    ``_lora_solve_drtrl`` independently of their own source.

    The ``'lora_b'`` trace holds an effective-weight (dense-style) trace for
    :math:`W_{\\text{eff}} = \\alpha\\, b\\_fn(B)\\, a\\_fn(A)` (per
    ``lora.py``'s module docstring, "no alpha/b_fn/a_fn factor" enters the
    instantaneous term — those are chained back only at solve time). Since
    ``y = x @ W_eff`` is exactly the dense-matmul forward, its Jacobian
    ``dy_o/dW_eff[i, o']`` is diagonal in the two "out" indices — the same
    structural fact exploited by :mod:`dense`'s ``yw_to_w``. This test
    builds that Jacobian via ``jax.jacobian`` on the *unscaled, untransformed*
    effective weight, confirms the diagonal structure, and checks the
    rule's ``'lora_b'`` output against the raw-Jacobian contraction for a
    random cotangent.

    ``solve_drtrl`` has no new forward to differentiate (it contracts an
    already-built trace); it is checked by an independent reimplementation
    of the documented formula, built without reusing the rule's own code.
    """

    def test_instant_drtrl_lora_b_matches_jacobian_contraction(self):
        import brainstate
        from braintrace._op.lora import _lora_instant_drtrl
        brainstate.random.seed(901)
        n_in, n_out, rank = 4, 5, 2
        B0 = brainstate.random.randn(n_in, rank)
        A0 = brainstate.random.randn(rank, n_out)
        x = brainstate.random.randn(n_in)

        # Unscaled, untransformed effective weight -- matches the documented
        # convention that the instant 'lora_b' term carries no alpha/b_fn/a_fn.
        W_eff0 = B0 @ A0

        def fwd_weff(W):
            return x @ W

        J = jax.jacobian(fwd_weff)(W_eff0)  # (out, in, out)
        for o in range(n_out):
            for o2 in range(n_out):
                expected = x if o == o2 else jnp.zeros_like(x)
                np.testing.assert_allclose(
                    J[o, :, o2], expected, atol=1e-10,
                    err_msg=f'Jacobian not diagonal at o={o}, o2={o2}',
                )

        g = brainstate.random.randn(n_out)
        # Repeated-index-style contraction against the raw Jacobian (sum
        # over the y-side out-index `m`, keep the trace's own out-index):
        # built from J/g directly, never from the rule's own outer product.
        ref_lora_b = jnp.einsum('m,mio->io', g, J)

        out = _lora_instant_drtrl(
            x, g, {'lora_b': B0, 'lora_a': A0}, alpha=2.0, has_bias=False,
        )
        np.testing.assert_allclose(out['lora_b'], ref_lora_b, atol=1e-10)

    def test_solve_drtrl_matches_independent_reimplementation_with_transforms(self):
        import brainstate
        from braintrace._op.lora import _lora_solve_drtrl
        brainstate.random.seed(902)
        n_in, n_out, rank = 4, 5, 2
        alpha = 2.0
        B0 = brainstate.random.randn(n_in, rank)
        A0 = brainstate.random.randn(rank, n_out)
        b0 = brainstate.random.randn(n_out)

        trace_lora_b = brainstate.random.randn(n_in, n_out)  # W_eff-shaped
        trace_lora_a = brainstate.random.randn(rank, n_out)
        trace_bias = brainstate.random.randn(n_out)
        dg_hidden = brainstate.random.randn(n_out)

        b_fn = lambda B: jnp.tanh(B)
        a_fn = lambda A: 1.5 * A

        # Independent reimplementation of the documented solve-time formula
        # (never calls `_lora_solve_drtrl`'s own code):
        g = dg_hidden[None, :]
        G = trace_lora_b * g
        A_eff = a_fn(A0)
        dB_eff = alpha * (G @ A_eff.T)
        _, vjp_b = jax.vjp(b_fn, B0)
        ref_lora_b = vjp_b(dB_eff)[0]
        ref_lora_a = trace_lora_a * g
        ref_bias = trace_bias * dg_hidden

        out = _lora_solve_drtrl(
            dg_hidden,
            {'lora_b': trace_lora_b, 'lora_a': trace_lora_a, 'bias': trace_bias},
            {'lora_b': B0, 'lora_a': A0, 'bias': b0},
            alpha=alpha, has_bias=True, b_fn=b_fn, a_fn=a_fn,
        )
        np.testing.assert_allclose(out['lora_b'], ref_lora_b, atol=1e-8)
        np.testing.assert_allclose(out['lora_a'], ref_lora_a, atol=1e-10)
        np.testing.assert_allclose(out['bias'], ref_bias, atol=1e-10)
