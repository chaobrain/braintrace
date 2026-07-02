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

"""Tests for the grouped (block-diagonal) matmul ETP primitives and
:func:`grouped_matmul` API."""

from collections import namedtuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainunit as u

import braintrace
from braintrace._op.grouped import etp_gmm_p, etp_gmv_p, grouped_matmul

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


def _loop_reference(x, w, b=None):
    """Independent per-block reference: y[..., g, :] = x[..., g, :] @ w[g]."""
    y = jnp.stack([x[..., g, :] @ w[g] for g in range(w.shape[0])], axis=-2)
    return y if b is None else y + b


class TestForwardCorrectness:

    def test_batched_matches_per_block_loop(self):
        brainstate.random.seed(0)
        x = brainstate.random.randn(5, 2, 3)
        w = brainstate.random.randn(2, 3, 4)
        np.testing.assert_allclose(grouped_matmul(x, w), _loop_reference(x, w), atol=1e-6)

    def test_unbatched_matches_per_block_loop(self):
        brainstate.random.seed(1)
        x = brainstate.random.randn(2, 3)
        w = brainstate.random.randn(2, 3, 4)
        np.testing.assert_allclose(grouped_matmul(x, w), _loop_reference(x, w), atol=1e-6)

    def test_with_bias(self):
        brainstate.random.seed(2)
        x = brainstate.random.randn(5, 2, 3)
        w = brainstate.random.randn(2, 3, 4)
        b = brainstate.random.randn(2, 4)
        np.testing.assert_allclose(grouped_matmul(x, w, b), _loop_reference(x, w, b), atol=1e-6)

    def test_weight_fn_applied_inside(self):
        x = jnp.ones((5, 2, 3))
        w = brainstate.random.randn(2, 3, 4)
        got = grouped_matmul(x, w, weight_fn=lambda ww: ww ** 2)
        np.testing.assert_allclose(got, _loop_reference(x, w ** 2), atol=1e-6)

    def test_equals_dense_block_diagonal(self):
        """grouped_matmul(x, w) == dense matmul against block_diag(w[0], ..., w[G-1])."""
        G, K, N = 2, 3, 4
        brainstate.random.seed(3)
        x = brainstate.random.randn(5, G, K)
        w = brainstate.random.randn(G, K, N)
        w_dense = jax.scipy.linalg.block_diag(*[w[g] for g in range(G)])
        want = (x.reshape(5, G * K) @ w_dense).reshape(5, G, N)
        np.testing.assert_allclose(grouped_matmul(x, w), want, atol=1e-5)

    def test_rejects_bad_ranks(self):
        with pytest.raises(ValueError, match=r'ndim'):
            grouped_matmul(jnp.ones((3,)), jnp.ones((2, 3, 4)))          # x rank < 2
        with pytest.raises(ValueError, match=r'ndim'):
            grouped_matmul(jnp.ones((6, 5, 2, 3)), jnp.ones((2, 3, 4)))  # x rank > 3
        with pytest.raises(ValueError):
            grouped_matmul(jnp.ones((5, 2, 3)), jnp.ones((3, 4)))        # weight rank != 3


class TestAutoDispatch:

    def test_unbatched_uses_gmv_primitive(self):
        jaxpr = jax.make_jaxpr(lambda x, w: grouped_matmul(x, w))(
            jnp.ones((2, 3)), jnp.ones((2, 3, 4)))
        prims = [e.primitive for e in jaxpr.jaxpr.eqns]
        assert etp_gmv_p in prims and etp_gmm_p not in prims

    def test_batched_uses_gmm_primitive(self):
        jaxpr = jax.make_jaxpr(lambda x, w: grouped_matmul(x, w))(
            jnp.ones((5, 2, 3)), jnp.ones((2, 3, 4)))
        prims = [e.primitive for e in jaxpr.jaxpr.eqns]
        assert etp_gmm_p in prims and etp_gmv_p not in prims


class TestBrainunit:

    def test_units_multiply_correctly(self):
        x = jnp.ones((5, 2, 3)) * u.mV
        w = jnp.ones((2, 3, 4)) * u.siemens
        y = grouped_matmul(x, w)
        assert isinstance(y, u.Quantity)
        expected = _loop_reference(jnp.ones((5, 2, 3)), jnp.ones((2, 3, 4))) * (u.mV * u.siemens)
        np.testing.assert_allclose(y.to_decimal(u.mV * u.siemens),
                                   expected.to_decimal(u.mV * u.siemens), atol=1e-6)

    def test_unitless_returns_plain_array(self):
        y = grouped_matmul(jnp.ones((5, 2, 3)), jnp.ones((2, 3, 4)))
        assert not isinstance(y, u.Quantity)


class TestJAXRules:

    def test_jit(self):
        f = jax.jit(lambda x, w: grouped_matmul(x, w))
        x, w = jnp.ones((5, 2, 3)), jnp.ones((2, 3, 4))
        np.testing.assert_allclose(f(x, w), _loop_reference(x, w), atol=1e-6)

    def test_vmap_over_batch(self):
        x = jnp.ones((7, 2, 3))
        w = jnp.ones((2, 3, 4))
        got = jax.vmap(lambda xi: grouped_matmul(xi, w))(x)
        np.testing.assert_allclose(got, _loop_reference(x, w), atol=1e-6)

    def test_vmap_promotes_gmv_to_gmm(self):
        """Identity-preserving promotion via register_batched_counterpart."""
        w = jnp.ones((2, 3, 4))
        jaxpr = jax.make_jaxpr(jax.vmap(lambda xi: grouped_matmul(xi, w)))(
            jnp.ones((7, 2, 3)))
        prims = [e.primitive for e in jaxpr.jaxpr.eqns]
        assert etp_gmm_p in prims and etp_gmv_p not in prims

    def test_grad_wrt_w(self):
        x = jnp.ones((5, 2, 3))
        w = jnp.ones((2, 3, 4))
        g = jax.grad(lambda ww: grouped_matmul(x, ww).sum())(w)
        # d/dw[g,k,n] sum(y) = sum_b x[b,g,k] = 5
        np.testing.assert_allclose(g, jnp.full((2, 3, 4), 5.0), atol=1e-6)
