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

"""Tests for braintrace.nn.Embedding."""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace


class TestEmbedding:

    def test_lookup_matches_table(self):
        layer = braintrace.nn.Embedding(10, 4)
        idx = jnp.array([0, 3, 3], dtype=jnp.int32)
        np.testing.assert_allclose(layer(idx), layer.weight.value[idx], atol=1e-6)

    def test_scalar_lookup(self):
        layer = braintrace.nn.Embedding(10, 4)
        out = layer(jnp.int32(7))
        assert out.shape == (4,)
        np.testing.assert_allclose(out, layer.weight.value[7], atol=1e-6)

    def test_folds_extra_leading_axes(self):
        layer = braintrace.nn.Embedding(10, 4)
        idx = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        out = layer(idx)
        assert out.shape == (2, 3, 4)
        np.testing.assert_allclose(out, layer.weight.value[idx], atol=1e-6)

    def test_uses_etp_primitive(self):
        layer = braintrace.nn.Embedding(10, 4)
        jp = jax.make_jaxpr(lambda i: layer(i))(jnp.array([0, 1], dtype=jnp.int32))
        prims = {str(e.primitive) for e in jp.jaxpr.eqns}
        assert 'etp_emb' in prims
        assert 'gather' not in prims

    def test_unsupported_features_rejected(self):
        idx = jnp.array([0], dtype=jnp.int32)
        with pytest.raises(NotImplementedError):
            braintrace.nn.Embedding(10, 4, max_norm=1.0)(idx)
        with pytest.raises(NotImplementedError):
            braintrace.nn.Embedding(10, 4, freeze=True)(idx)
        with pytest.raises(NotImplementedError):
            braintrace.nn.Embedding(10, 4, scale_grad_by_freq=True)(idx)
        with pytest.raises(NotImplementedError):
            braintrace.nn.Embedding(10, 4, padding_idx=0)(idx)

    def test_gradient_flows_to_table(self):
        layer = braintrace.nn.Embedding(10, 4)
        idx = jnp.array([2, 2], dtype=jnp.int32)
        g = brainstate.transform.grad(
            lambda: layer(idx).sum(), layer.states(brainstate.ParamState))()
        (gval,) = g.values()
        want = jnp.zeros((10, 4)).at[jnp.array([2, 2])].add(1.0)
        np.testing.assert_allclose(gval, want, atol=1e-6)

    def test_exported_from_nn(self):
        assert 'Embedding' in braintrace.nn.__all__
