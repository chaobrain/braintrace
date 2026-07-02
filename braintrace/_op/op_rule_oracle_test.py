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

"""Operator-rule correctness: each primitive's xy_to_dw must equal jax.vjp of its
impl; init_drtrl/init_pp shape contract; lora effective-weight trace recurrence;
multi-primitive composition gradients. Fills gaps left by dense_test/elemwise_test."""

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_DT_TO_T,
)
from braintrace._op.dense import etp_mm_p, _etp_matmul_impl
from braintrace._op.conv import etp_conv_p, _etp_conv_impl
from braintrace._op.sparse import etp_sp_mm_p, _etp_sp_matmul_impl
from braintrace._op.lora import etp_lora_mm_p, _etp_lora_impl
from braintrace._op.op_rule_oracle import assert_xy_to_dw_matches_vjp, xy_to_dw_and_vjp
from braintrace._op.sparse_test import _StubSparseMat

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


# --- Task 1: dense mm xy_to_dw == jax.vjp ------------------------------------

def test_mm_xy_to_dw_matches_vjp():
    x = jnp.arange(6.0).reshape(2, 3)
    w = jnp.arange(12.0).reshape(3, 4)
    hidden = jnp.ones((2, 4))
    assert_xy_to_dw_matches_vjp(
        rule=ETP_RULES_XY_TO_DW[etp_mm_p],
        impl=lambda weights: _etp_matmul_impl(x, weights['weight'], has_bias=False),
        x=x, hidden_dim=hidden, weights={'weight': w}, params={'has_bias': False},
    )


# --- Task 2: conv xy_to_dw == jax.vjp (with bias) ----------------------------

def test_conv_xy_to_dw_matches_vjp_with_bias():
    # NCW layout: x=(batch, in_ch, L), kernel=(out_ch, in_ch, width)
    x = jnp.asarray(np.random.RandomState(0).randn(1, 2, 8).astype('float32'))
    kernel = jnp.asarray(np.random.RandomState(1).randn(4, 2, 3).astype('float32'))
    bias = jnp.zeros((1, 4, 1), dtype='float32')
    conv_params = dict(has_bias=True, strides=(1,), padding='SAME', dimension_numbers=None)
    y = _etp_conv_impl(x, kernel, bias, **conv_params)
    hidden = jnp.ones_like(y)
    impl = lambda w: _etp_conv_impl(x, w['weight'], w['bias'], **conv_params)
    weights = {'weight': kernel, 'bias': bias}
    # The kernel gradient must match jax.vjp exactly.
    assert_xy_to_dw_matches_vjp(
        rule=ETP_RULES_XY_TO_DW[etp_conv_p], impl=impl,
        x=x, hidden_dim=hidden, weights=weights, params=conv_params,
        atol=1e-4, keys=['weight'],
    )
    # The bias gradient is DEFERRED: the rule returns the un-summed cotangent
    # (batch, out_ch, L); summing over the spatial axis recovers the jax.vjp bias.
    rule_dw, vjp_dw = xy_to_dw_and_vjp(
        rule=ETP_RULES_XY_TO_DW[etp_conv_p], impl=impl,
        x=x, hidden_dim=hidden, weights=weights, params=conv_params,
    )
    summed_bias = jnp.asarray(rule_dw['bias']).sum(axis=2, keepdims=True)
    np.testing.assert_allclose(summed_bias, jnp.asarray(vjp_dw['bias']), atol=1e-4)


# --- Task 3: sparse xy_to_dw == jax.vjp --------------------------------------

def test_sparse_xy_to_dw_matches_vjp():
    in_dim, out_dim = 3, 4
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randn(2, in_dim).astype('float32'))
    weight_data = jnp.asarray(rng.randn(in_dim * out_dim).astype('float32'))
    sparse_mat = _StubSparseMat(jnp.zeros((in_dim, out_dim)))
    sp_params = dict(sparse_mat=sparse_mat, has_bias=False)
    y = _etp_sp_matmul_impl(x, weight_data, **sp_params)
    hidden = jnp.ones_like(y)
    assert_xy_to_dw_matches_vjp(
        rule=ETP_RULES_XY_TO_DW[etp_sp_mm_p],
        impl=lambda w: _etp_sp_matmul_impl(x, w['weight'], **sp_params),
        x=x, hidden_dim=hidden, weights={'weight': weight_data},
        params=sp_params, atol=1e-4,
    )


# --- Task 4: lora xy_to_dw == jax.vjp + partial propagation ------------------

def test_lora_xy_to_dw_matches_vjp():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randn(2, 6).astype('float32'))
    B = jnp.asarray(rng.randn(6, 3).astype('float32'))   # (in, rank)
    A = jnp.asarray(rng.randn(3, 4).astype('float32'))   # (rank, out)
    lora_params = dict(alpha=1.0, has_bias=False)
    y = _etp_lora_impl(x, B, A, **lora_params)
    hidden = jnp.ones_like(y)
    assert_xy_to_dw_matches_vjp(
        rule=ETP_RULES_XY_TO_DW[etp_lora_mm_p],
        impl=lambda w: _etp_lora_impl(x, w['lora_b'], w['lora_a'], **lora_params),
        x=x, hidden_dim=hidden, weights={'lora_b': B, 'lora_a': A},
        params=lora_params, atol=1e-4,
    )


def test_lora_dt_to_t_scales_both_traces_along_output_axis():
    """dt_to_t applies the dense ``y -> W`` link to BOTH traces: the
    ``'lora_b'`` entry is the effective-weight trace ``(in, out)`` for
    ``W_eff = alpha * b_fn(B) @ a_fn(A)`` and follows the same recurrence as
    the dense ``mm`` rule (a raw B-shaped trace cannot be discounted along
    the output axis it lacks — the old pass-through made ``lora_b``
    gradients wrong even at T=1)."""
    rule = ETP_RULES_DT_TO_T[etp_lora_mm_p]
    in_dim, rank, out_dim = 6, 3, 4
    hidden = jnp.arange(1.0, out_dim + 1.0)            # (out,)
    trace = {'lora_b': jnp.ones((in_dim, out_dim)), 'lora_a': jnp.ones((rank, out_dim))}
    out = rule(hidden, trace)
    np.testing.assert_allclose(out['lora_b'], trace['lora_b'] * hidden[None, :])
    np.testing.assert_allclose(out['lora_a'], trace['lora_a'] * hidden[None, :])


# --- Task 5: init_drtrl / init_pp shape contract -----------------------------

def test_init_drtrl_and_pp_shapes_consistent_for_mm():
    n_state = 2
    x_var = _fake_var((4, 3))           # (batch, in)
    y_var = _fake_var((4, 5))           # (batch, out)
    weight_vars = {'weight': _fake_var((3, 5))}
    drtrl = ETP_RULES_INIT_DRTRL[etp_mm_p](x_var, y_var, weight_vars, num_hidden_state=n_state)
    pp = ETP_RULES_INIT_PP[etp_mm_p](x_var, y_var, weight_vars, num_hidden_state=n_state)
    assert drtrl['weight'].shape == (4, 3, 5, n_state)   # batch + weight + state
    assert pp.shape == (4, 5, n_state)                   # y + state


# --- Task 6: multi-primitive composition gradients ---------------------------

def test_matmul_then_elementwise_grad_flows_through_both():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randn(2, 3).astype('float32'))
    w = jnp.asarray(rng.randn(3, 4).astype('float32'))
    scale = jnp.asarray(rng.randn(4).astype('float32'))

    def f(w_, scale_):
        y = braintrace.matmul(x, w_)                     # etp_mm_p
        z = braintrace.element_wise(scale_, weight_fn=lambda s: s) * y  # etp_elemwise_p marker
        return (z ** 2).sum()

    def f_ref(w_, scale_):
        y = x @ w_
        z = scale_ * y
        return (z ** 2).sum()

    gw, gs = jax.grad(f, argnums=(0, 1))(w, scale)
    gw_ref, gs_ref = jax.grad(f_ref, argnums=(0, 1))(w, scale)
    np.testing.assert_allclose(gw, gw_ref, atol=1e-4)
    np.testing.assert_allclose(gs, gs_ref, atol=1e-4)
