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
``'lora_b'`` / ``'lora_a'`` (and optionally ``'bias'``), matching the
``ParamState`` pytree structure used by ``braintrace.nn.LoRALinear``.
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
        """``yw_to_w`` broadcasts ``hidden`` across the ``rank`` axis of
        ``trace['lora_a']`` using ``expand_dims(hidden, axis=-2)``.

        Two equivalent shapes are tested:
          * ``(out,)`` — as called from ``_solve_param_dim_weight_gradients``
          * ``(batch, out)`` — as called from ``_update_param_dim_etrace_scan_fn``
        """
        rule = ETP_RULES_YW_TO_W[etp_lora_mm_p]

        # Test 1: unbatched gradient-solve context — hidden=(out=4,), trace_A=(rank=2, out=4)
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])  # (out=4,)
        trace_B = jnp.ones((4, 2))  # (in=4, rank=2)
        trace_A = jnp.ones((2, 4))  # (rank=2, out=4)
        out = rule(hidden, {'lora_b': trace_B, 'lora_a': trace_A})
        assert set(out.keys()) == {'lora_b', 'lora_a'}
        np.testing.assert_allclose(out['lora_b'], trace_B)
        # expected: trace_A * hidden[None, :] = (2, 4) * (1, 4) = (2, 4)
        np.testing.assert_allclose(out['lora_a'], trace_A * hidden[None, :])

        # Test 2: batched trace-update context — hidden=(batch=1, out=4), trace_A=(batch=1, rank=2, out=4)
        hidden_b = jnp.ones((1, 4))  # (batch=1, out=4)
        trace_A_b = jnp.arange(8.0).reshape(1, 2, 4)  # (batch=1, rank=2, out=4)
        out_b = rule(hidden_b, {'lora_b': jnp.ones((1, 4, 2)), 'lora_a': trace_A_b})
        # expected: trace_A_b * hidden_b[:, None, :] = (1, 2, 4) * (1, 1, 4) = (1, 2, 4)
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
        rule = ETP_RULES_INIT_DRTRL[etp_lora_mm_p]
        x_var = _fake_var((2, 3))
        y_var = _fake_var((2, 4))
        weight_vars = {'lora_b': _fake_var((3, 2)), 'lora_a': _fake_var((2, 4))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=5)
        assert out['lora_b'].shape == (2, 3, 2, 5)
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
        assert out['lora_b'].shape == (2, 3, 2, 5)
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
        """mv-variant: expand_dims axis=0 → broadcasts ``hidden`` across the
        rank axis of ``trace['lora_a']``."""
        rule = ETP_RULES_YW_TO_W[etp_lora_mv_p]
        hidden = jnp.array([1.0, 2.0, 3.0, 4.0])  # (out=4,)
        trace_B = jnp.ones((3, 2))  # (in=3, rank=2)
        trace_A = jnp.ones((2, 4))  # (rank=2, out=4)
        out = rule(hidden, {'lora_b': trace_B, 'lora_a': trace_A})
        assert set(out.keys()) == {'lora_b', 'lora_a'}
        np.testing.assert_allclose(out['lora_b'], trace_B)
        # expand_dims(hidden, axis=0) = (1, 4); (2, 4) * (1, 4) = (2, 4)
        np.testing.assert_allclose(out['lora_a'], trace_A * hidden[None, :])

    def test_init_drtrl_shape(self):
        rule = ETP_RULES_INIT_DRTRL[etp_lora_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((4,))
        weight_vars = {'lora_b': _fake_var((3, 2)), 'lora_a': _fake_var((2, 4))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=5)
        assert out['lora_b'].shape == (3, 2, 5)
        assert out['lora_a'].shape == (2, 4, 5)

    def test_init_pp_shape(self):
        rule = ETP_RULES_INIT_PP[etp_lora_mv_p]
        x_var = _fake_var((3,))
        y_var = _fake_var((4,))
        weight_vars = {'lora_b': _fake_var((3, 2)), 'lora_a': _fake_var((2, 4))}
        out = rule(x_var, y_var, weight_vars, num_hidden_state=5)
        assert out.shape == (4, 5)


class TestPublicAPIRoundTrip:

    def test_public_alias(self):
        assert braintrace.lora_matmul is lora_matmul


# ---------------------------------------------------------------------------
# End-to-end online learning: D-RTRL vs BPTT
# ---------------------------------------------------------------------------

class TestLoRAOnlineLearning:
    """D-RTRL gradient correctness for etp_lora_mm_p with B, A, and bias."""

    def test_drtrl_lora_mm_grad_matches_bptt(self):
        import brainstate
        import jax
        import jax.numpy as jnp
        import numpy.testing as npt
        import braintrace

        rank = 2
        B_init = jnp.ones((4, rank)) * 0.1
        A_init = jnp.ones((rank, 4)) * 0.1
        bias_init = jnp.ones((4,)) * 0.05

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState({
                    'lora_b': B_init,
                    'lora_a': A_init,
                    'bias': bias_init,
                })
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                p = self.p.value
                y = braintrace.lora_matmul(
                    self.h.value, p['lora_b'], p['lora_a'],
                    alpha=1.0, bias=p['bias'],
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)

        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 4)))

        x = jnp.ones((1, 4)) * 0.3

        # --- ETP gradient via D-RTRL ---
        @brainstate.transform.jit
        def etrace_grad_step(inp):
            return brainstate.transform.grad(
                lambda inp: alg(inp).sum(),
                cell.states(brainstate.ParamState),
            )(inp)

        grads_etrace = etrace_grad_step(x)

        # grads_etrace is keyed by path; cell.p is a dict-valued ParamState
        grad_p = list(grads_etrace.values())[0]
        assert isinstance(grad_p, dict), (
            f'Expected dict gradient for merged ParamState, got {type(grad_p)}'
        )

        # --- BPTT reference ---
        def bptt_loss(params):
            h = jnp.zeros((1, 4))
            y = 1.0 * (h @ params['lora_b'] @ params['lora_a']) + params['bias']
            h = jnp.tanh(x + y)
            return h.sum()

        bptt = jax.grad(bptt_loss)({
            'lora_b': cell.p.value['lora_b'],
            'lora_a': cell.p.value['lora_a'],
            'bias': cell.p.value['bias'],
        })

        # Non-zero sanity check on bias gradient
        assert jnp.abs(bptt['bias']).max() > 1e-3, (
            f'BPTT bias gradient is unexpectedly near-zero: {bptt["bias"]}'
        )

        # Compare all three: lora_b, lora_a, bias
        npt.assert_allclose(grad_p['lora_b'], bptt['lora_b'], atol=1e-5,
                            err_msg='D-RTRL d(lora_b) does not match BPTT')
        npt.assert_allclose(grad_p['lora_a'], bptt['lora_a'], atol=1e-5,
                            err_msg='D-RTRL d(lora_a) does not match BPTT')
        npt.assert_allclose(grad_p['bias'], bptt['bias'], atol=1e-5,
                            err_msg='D-RTRL d(bias) does not match BPTT (was it zero?)')
