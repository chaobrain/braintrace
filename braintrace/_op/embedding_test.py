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

"""Tests for the embedding (trainable gather) ETP primitives and
:func:`embedding` API."""

from collections import namedtuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainunit as u

import braintrace
from braintrace._op.embedding import embedding, etp_emb_p, etp_emb_v_p

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


class TestForwardCorrectness:

    def test_batched_lookup(self):
        table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        idx = jnp.array([0, 2, 2], dtype=jnp.int32)
        np.testing.assert_allclose(embedding(idx, table), table[idx])

    def test_scalar_lookup(self):
        table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        out = embedding(jnp.int32(1), table)
        assert out.shape == (3,)
        np.testing.assert_allclose(out, table[1])

    def test_weight_fn_applied_inside(self):
        table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        idx = jnp.array([1, 3], dtype=jnp.int32)
        got = embedding(idx, table, weight_fn=lambda w: w ** 2)
        np.testing.assert_allclose(got, (table ** 2)[idx])

    def test_rejects_float_indices(self):
        with pytest.raises(TypeError):
            embedding(jnp.array([0.0, 1.0]), jnp.ones((4, 3)))

    def test_rejects_rank2_indices(self):
        with pytest.raises(NotImplementedError):
            embedding(jnp.zeros((2, 3), dtype=jnp.int32), jnp.ones((4, 3)))

    def test_rejects_non_matrix_table(self):
        with pytest.raises(ValueError):
            embedding(jnp.array([0], dtype=jnp.int32), jnp.ones((4, 3, 2)))


class TestAutoDispatch:

    def test_batched_uses_emb_primitive(self):
        jaxpr = jax.make_jaxpr(lambda i, t: embedding(i, t))(
            jnp.array([0, 1], dtype=jnp.int32), jnp.ones((4, 3)))
        prims = [e.primitive for e in jaxpr.jaxpr.eqns]
        assert etp_emb_p in prims and etp_emb_v_p not in prims

    def test_scalar_uses_emb_v_primitive(self):
        jaxpr = jax.make_jaxpr(lambda i, t: embedding(i, t))(
            jnp.int32(0), jnp.ones((4, 3)))
        prims = [e.primitive for e in jaxpr.jaxpr.eqns]
        assert etp_emb_v_p in prims and etp_emb_p not in prims


class TestBrainunit:

    def test_table_units_propagate(self):
        table_val = jnp.ones((4, 3))
        table = table_val * u.mV
        idx = jnp.array([0, 1], dtype=jnp.int32)
        y = embedding(idx, table)
        assert isinstance(y, u.Quantity)
        np.testing.assert_allclose(y.to_decimal(u.mV), table_val[idx], atol=1e-6)


class TestJAXRules:

    def test_jit(self):
        table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        idx = jnp.array([3, 0], dtype=jnp.int32)
        f = jax.jit(lambda i, t: embedding(i, t))
        np.testing.assert_allclose(f(idx, table), table[idx])

    def test_grad_wrt_table_is_scatter_add(self):
        table = jnp.zeros((4, 3))
        idx = jnp.array([1, 1, 2], dtype=jnp.int32)
        g = jax.grad(lambda t: embedding(idx, t).sum())(table)
        want = jnp.zeros((4, 3)).at[idx].add(1.0)
        np.testing.assert_allclose(g, want)

    def test_vmap_over_batch_of_index_vectors(self):
        table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        idxs = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        got = jax.vmap(lambda i: embedding(i, table))(idxs)
        np.testing.assert_allclose(got, table[idxs])

    def test_vmap_promotes_emb_v_to_emb(self):
        """Identity-preserving promotion via register_batched_counterpart:
        vmap over scalar indices re-binds the batched primitive."""
        table = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        idx = jnp.array([0, 2, 1], dtype=jnp.int32)
        jaxpr = jax.make_jaxpr(jax.vmap(lambda i: embedding(i, table)))(idx)
        prims = [e.primitive for e in jaxpr.jaxpr.eqns]
        assert etp_emb_p in prims and etp_emb_v_p not in prims


from braintrace._op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
)
from braintrace._op.op_rule_oracle import assert_xy_to_dw_matches_vjp


class TestEmbEtpRules:
    B, V, D, A = 3, 5, 4, 2

    def test_yw_to_w_broadcasts_over_vocab(self):
        brainstate.random.seed(40)
        hd = brainstate.random.randn(self.B, self.D)
        tr = {'weight': brainstate.random.randn(self.B, self.V, self.D)}
        out = ETP_RULES_YW_TO_W[etp_emb_p](hd, tr)
        np.testing.assert_allclose(out['weight'], tr['weight'] * hd[:, None, :], atol=1e-6)

    def test_yw_to_w_grad_context(self):
        brainstate.random.seed(41)
        hd = brainstate.random.randn(self.D)
        tr = {'weight': brainstate.random.randn(self.V, self.D)}
        out = ETP_RULES_YW_TO_W[etp_emb_p](hd, tr)
        np.testing.assert_allclose(out['weight'], tr['weight'] * hd[None, :], atol=1e-6)

    def test_xy_to_dw_matches_vjp_scatter_add(self):
        brainstate.random.seed(42)
        idx = jnp.array([0, 2, 2], dtype=jnp.int32)
        hd = brainstate.random.randn(3, self.D)
        weights = {'weight': brainstate.random.randn(self.V, self.D)}
        assert_xy_to_dw_matches_vjp(
            rule=ETP_RULES_XY_TO_DW[etp_emb_p],
            impl=lambda wd: jnp.take(wd['weight'], idx, axis=0),
            x=idx, hidden_dim=hd, weights=weights,
        )

    def test_xy_to_dw_with_weight_fn(self):
        brainstate.random.seed(43)
        idx = jnp.array([1, 3], dtype=jnp.int32)
        hd = brainstate.random.randn(2, self.D)
        weights = {'weight': brainstate.random.randn(self.V, self.D)}
        fn = lambda w: jnp.tanh(w)
        assert_xy_to_dw_matches_vjp(
            rule=ETP_RULES_XY_TO_DW[etp_emb_p],
            impl=lambda wd: jnp.take(fn(wd['weight']), idx, axis=0),
            x=idx, hidden_dim=hd, weights=weights, params={'weight_fn': fn},
        )

    def test_init_shapes_and_dtypes(self):
        drtrl = ETP_RULES_INIT_DRTRL[etp_emb_p](
            _fake_var((self.B,), jnp.int32),
            _fake_var((self.B, self.D), jnp.bfloat16),
            {'weight': _fake_var((self.V, self.D), jnp.bfloat16)}, self.A)
        assert drtrl['weight'].shape == (self.B, self.V, self.D, self.A)
        # int32 indices don't affect jnp.result_type(int32, bf16, bf16)
        assert drtrl['weight'].dtype == jnp.bfloat16
        pp = ETP_RULES_INIT_PP[etp_emb_p](
            _fake_var((self.B,), jnp.int32), _fake_var((self.B, self.D), jnp.bfloat16),
            {'weight': _fake_var((self.V, self.D))}, self.A)
        assert pp.shape == (self.B, self.D, self.A)
        assert pp.dtype == jnp.bfloat16

    def test_unbatched_rules(self):
        brainstate.random.seed(44)
        hd = brainstate.random.randn(self.D)
        tr = {'weight': brainstate.random.randn(self.V, self.D)}
        out = ETP_RULES_YW_TO_W[etp_emb_v_p](hd, tr)
        np.testing.assert_allclose(out['weight'], tr['weight'] * hd[None, :], atol=1e-6)

        idx = jnp.int32(2)
        assert_xy_to_dw_matches_vjp(
            rule=ETP_RULES_XY_TO_DW[etp_emb_v_p],
            impl=lambda wd: jnp.take(wd['weight'], idx, axis=0),
            x=idx, hidden_dim=hd, weights=tr,
        )

        drtrl = ETP_RULES_INIT_DRTRL[etp_emb_v_p](
            _fake_var((), jnp.int32), _fake_var((self.D,), jnp.float16),
            {'weight': _fake_var((self.V, self.D), jnp.float16)}, self.A)
        assert drtrl['weight'].shape == (self.V, self.D, self.A)
        assert drtrl['weight'].dtype == jnp.float16
