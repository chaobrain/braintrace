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

"""Tests for the general linear-contraction ETP primitive and
:func:`einsum` API."""

from collections import namedtuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainunit as u

import braintrace
from braintrace._op.einsum import EinsumSpec, parse_etp_einsum

_FakeVar = namedtuple('_FakeVar', ['aval'])
_FakeAval = namedtuple('_FakeAval', ['shape', 'dtype'])


def _fake_var(shape, dtype=jnp.float32):
    return _FakeVar(aval=_FakeAval(shape=shape, dtype=dtype))


class TestParser:

    def test_dense_equation(self):
        s = parse_etp_einsum('bk,kn->bn')
        assert s == EinsumSpec('bk', 'kn', 'bn', 'b', diagonal='n',
                               contracted='k', shared='')

    def test_grouped_equation(self):
        s = parse_etp_einsum('bgk,gkn->bgn')
        assert s.batch == 'b'
        assert s.diagonal == 'gn'
        assert s.contracted == 'k'
        assert s.shared == ''

    def test_per_head_equation(self):
        s = parse_etp_einsum('bhd,hde->bhe')
        assert s.diagonal == 'he'
        assert s.contracted == 'd'
        assert s.shared == ''

    def test_shared_axis_equation_classified(self):
        s = parse_etp_einsum('btk,kn->btn')
        assert s.diagonal == 'n'
        assert s.contracted == 'k'
        assert s.shared == 't'

    def test_spaces_normalized(self):
        assert parse_etp_einsum(' bk , kn -> bn ') == parse_etp_einsum('bk,kn->bn')

    @pytest.mark.parametrize('bad', [
        'bk,kn',            # no explicit output
        'bk->b',            # one operand
        'bk,kn,nm->bm',     # three operands
        'bk,kn->bnz',       # output letter from nowhere
        'Bk,kn->Bn',        # uppercase
        'b...k,kn->b...n',  # ellipsis
        'bkk,kn->bn',       # repeated letter within a spec
        'bk,bn->bn',        # batch letter inside weight spec
        'kb,kn->bn',        # x does not lead with the batch letter
        'bk,kv->bn',        # weight letter v in neither x nor output
    ])
    def test_rejections(self, bad):
        with pytest.raises(ValueError):
            parse_etp_einsum(bad)


from braintrace._op.einsum import einsum, etp_einsum_p


class TestForwardCorrectness:

    @pytest.mark.parametrize('eq,x_shape,w_shape', [
        ('bk,kn->bn', (5, 3), (3, 4)),
        ('bgk,gkn->bgn', (5, 2, 3), (2, 3, 4)),
        ('bhd,hde->bhe', (5, 2, 3), (2, 3, 4)),
        ('bd,d->bd', (5, 3), (3,)),
    ])
    def test_matches_jnp_einsum(self, eq, x_shape, w_shape):
        brainstate.random.seed(0)
        x = brainstate.random.randn(*x_shape)
        w = brainstate.random.randn(*w_shape)
        np.testing.assert_allclose(
            einsum(eq, x, w), jnp.einsum(eq, x, w), atol=1e-5)

    def test_weight_fn_applied_inside(self):
        brainstate.random.seed(1)
        x = brainstate.random.randn(5, 3)
        w = brainstate.random.randn(3, 4)
        got = einsum('bk,kn->bn', x, w, weight_fn=lambda ww: ww ** 2)
        np.testing.assert_allclose(got, jnp.einsum('bk,kn->bn', x, w ** 2), atol=1e-5)

    def test_shared_axis_equation_gated(self):
        with pytest.raises(NotImplementedError):
            einsum('btk,kn->btn', jnp.ones((5, 2, 3)), jnp.ones((3, 4)))

    def test_rank_mismatch_rejected(self):
        with pytest.raises(ValueError):
            einsum('bk,kn->bn', jnp.ones((5, 3, 2)), jnp.ones((3, 4)))
        with pytest.raises(ValueError):
            einsum('bk,kn->bn', jnp.ones((5, 3)), jnp.ones((3,)))

    def test_uses_einsum_primitive(self):
        jaxpr = jax.make_jaxpr(
            lambda x, w: einsum('bk,kn->bn', x, w))(jnp.ones((5, 3)), jnp.ones((3, 4)))
        assert etp_einsum_p in [e.primitive for e in jaxpr.jaxpr.eqns]


class TestBrainunit:

    def test_units_multiply(self):
        xv = jnp.ones((5, 3))
        wv = jnp.ones((3, 4))
        y = einsum('bk,kn->bn', xv * u.mV, wv * u.siemens)
        assert isinstance(y, u.Quantity)
        want = jnp.einsum('bk,kn->bn', xv, wv)
        np.testing.assert_allclose(y.to_decimal(u.mV * u.siemens), want, atol=1e-5)


class TestJAXRules:

    def test_jit_vmap_grad(self):
        brainstate.random.seed(2)
        x = brainstate.random.randn(5, 3)
        w = brainstate.random.randn(3, 4)
        f = jax.jit(lambda xx, ww: einsum('bk,kn->bn', xx, ww))
        np.testing.assert_allclose(f(x, w), x @ w, atol=1e-5)
        g = jax.grad(lambda ww: einsum('bk,kn->bn', x, ww).sum())(w)
        np.testing.assert_allclose(g, jnp.tile(x.sum(0)[:, None], (1, 4)), atol=1e-5)
