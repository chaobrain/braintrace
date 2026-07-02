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


from braintrace._op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
)
from braintrace._op.dense import etp_mm_p
from braintrace._op.grouped import etp_gmm_p
from braintrace._op.op_rule_oracle import assert_xy_to_dw_matches_vjp


class TestEinsumEtpRules:

    def test_yw_to_w_matches_dense_rule(self):
        brainstate.random.seed(10)
        hd = brainstate.random.randn(5, 4)
        tr = {'weight': brainstate.random.randn(5, 3, 4)}
        got = ETP_RULES_YW_TO_W[etp_einsum_p](hd, tr, equation='bk,kn->bn')
        want = ETP_RULES_YW_TO_W[etp_mm_p](hd, dict(tr), has_bias=False)
        np.testing.assert_allclose(got['weight'], want['weight'], atol=1e-6)

    def test_yw_to_w_matches_grouped_rule(self):
        brainstate.random.seed(11)
        hd = brainstate.random.randn(5, 2, 4)
        tr = {'weight': brainstate.random.randn(5, 2, 3, 4)}
        got = ETP_RULES_YW_TO_W[etp_einsum_p](hd, tr, equation='bgk,gkn->bgn')
        want = ETP_RULES_YW_TO_W[etp_gmm_p](hd, dict(tr), has_bias=False)
        np.testing.assert_allclose(got['weight'], want['weight'], atol=1e-6)

    def test_yw_to_w_grad_context_batch_stripped(self):
        brainstate.random.seed(12)
        hd = brainstate.random.randn(4)                    # (n,) — batch stripped
        tr = {'weight': brainstate.random.randn(3, 4)}     # (k, n)
        got = ETP_RULES_YW_TO_W[etp_einsum_p](hd, tr, equation='bk,kn->bn')
        np.testing.assert_allclose(got['weight'], tr['weight'] * hd[None, :], atol=1e-6)

    def test_yw_to_w_per_head_diagonal_axes(self):
        brainstate.random.seed(13)
        hd = brainstate.random.randn(5, 2, 4)              # (b, h, e)
        tr = {'weight': brainstate.random.randn(5, 2, 3, 4)}  # (b, h, d, e)
        got = ETP_RULES_YW_TO_W[etp_einsum_p](hd, tr, equation='bhd,hde->bhe')
        want = tr['weight'] * hd[:, :, None, :]
        np.testing.assert_allclose(got['weight'], want, atol=1e-6)

    def test_yw_to_w_shared_axis_sums_hidden(self):
        """Rule-level contract for shared axes (API gate notwithstanding):
        hd is summed over 't' then broadcast — the pre-audit conv scheme,
        pinned here as the rule's defined behaviour while the gate is closed."""
        brainstate.random.seed(14)
        hd = brainstate.random.randn(5, 2, 4)              # (b, t, n)
        tr = {'weight': brainstate.random.randn(5, 3, 4)}  # (b, k, n)
        got = ETP_RULES_YW_TO_W[etp_einsum_p](hd, tr, equation='btk,kn->btn')
        want = tr['weight'] * hd.sum(axis=1)[:, None, :]
        np.testing.assert_allclose(got['weight'], want, atol=1e-5)

    @pytest.mark.parametrize('eq,x_shape,w_shape', [
        ('bk,kn->bn', (5, 3), (3, 4)),
        ('bhd,hde->bhe', (5, 2, 3), (2, 3, 4)),
        ('bd,d->bd', (5, 3), (3,)),
    ])
    def test_xy_to_dw_matches_vjp(self, eq, x_shape, w_shape):
        brainstate.random.seed(15)
        x = brainstate.random.randn(*x_shape)
        w = {'weight': brainstate.random.randn(*w_shape)}
        y = jnp.einsum(eq, x, w['weight'])
        hd = brainstate.random.randn(*y.shape)
        assert_xy_to_dw_matches_vjp(
            rule=ETP_RULES_XY_TO_DW[etp_einsum_p],
            impl=lambda wd: jnp.einsum(eq, x, wd['weight']),
            x=x, hidden_dim=hd, weights=w, params={'equation': eq},
        )

    def test_xy_to_dw_with_weight_fn(self):
        brainstate.random.seed(16)
        x = brainstate.random.randn(5, 3)
        w = {'weight': brainstate.random.randn(3, 4)}
        hd = brainstate.random.randn(5, 4)
        fn = lambda ww: jnp.tanh(ww)
        assert_xy_to_dw_matches_vjp(
            rule=ETP_RULES_XY_TO_DW[etp_einsum_p],
            impl=lambda wd: jnp.einsum('bk,kn->bn', x, fn(wd['weight'])),
            x=x, hidden_dim=hd, weights=w,
            params={'equation': 'bk,kn->bn', 'weight_fn': fn},
        )

    def test_init_shapes_and_dtypes(self):
        drtrl = ETP_RULES_INIT_DRTRL[etp_einsum_p](
            _fake_var((5, 2, 3), jnp.bfloat16), _fake_var((5, 2, 4), jnp.bfloat16),
            {'weight': _fake_var((2, 3, 4), jnp.bfloat16)}, 2)
        assert drtrl['weight'].shape == (5, 2, 3, 4, 2)
        assert drtrl['weight'].dtype == jnp.bfloat16
        pp = ETP_RULES_INIT_PP[etp_einsum_p](
            _fake_var((5, 2, 3)), _fake_var((5, 2, 4), jnp.bfloat16),
            {'weight': _fake_var((2, 3, 4))}, 2)
        assert pp.shape == (5, 2, 4, 2)
        assert pp.dtype == jnp.bfloat16


class TestPublicExports:

    def test_top_level_exports(self):
        assert braintrace.einsum is einsum
        assert 'einsum' in braintrace.__all__

    def test_op_package_exports(self):
        import braintrace._op as op
        for name in ('etp_einsum_p', 'einsum'):
            assert name in op.__all__

    def test_matmul_rank_guard_points_to_einsum(self):
        with pytest.raises(ValueError, match='einsum'):
            braintrace.matmul(jnp.ones((5, 2, 3)), jnp.ones((3, 4)))


from braintrace._algorithm.oracle import (
    assert_direction_aligned,
    assert_param_gradients_close,
    bptt_param_gradients,
    online_param_gradients,
)


def _per_head_rnn_factory(H=2, E=3, n_in=3, seed=0):
    """tanh RNN whose recurrence is a per-head contraction 'bhd,hde->bhe'
    (each head has its own E×E mixing matrix).

    The hidden state is kept rank-3 ``(1, H, E)``: the compiler requires the
    einsum output to be broadcast-compatible with the hidden state, so a flat
    ``(1, H*E)`` hidden fed by a reshaped einsum output silently severs the
    relation (the weight degrades to non-temporal and the oracle comparison
    becomes vacuous)."""

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                k = jax.random.PRNGKey
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(k(seed), (H, E, E)))
                self.win = brainstate.ParamState(
                    0.1 * jax.random.normal(k(seed + 1), (n_in, H * E)))
                self.h = brainstate.HiddenState(jnp.zeros((1, H, E)))

            def update(self, x):
                rec = braintrace.einsum('bhd,hde->bhe', self.h.value, self.w.value)
                inp = (x @ self.win.value).reshape(1, H, E)
                self.h.value = jax.nn.tanh(inp + rec)
                return self.h.value

        return Net()

    return factory


def _seq_inputs(T=6, n_in=3, seed=42):
    brainstate.random.seed(seed)
    return brainstate.random.randn(T, n_in)


class TestEinsumOracle:

    def test_per_head_model_produces_einsum_relation(self):
        """Guard against vacuous oracle passes: the per-head model must
        actually compile to an etp_einsum relation (a broadcast-incompatible
        hidden state silently drops it to a non-temporal parameter)."""
        net = _per_head_rnn_factory()()
        rels = braintrace.find_hidden_param_op_relations_from_module(
            net, _seq_inputs()[0])
        assert [r.primitive.name for r in rels].count('etp_einsum') == 1

    def test_d_rtrl_multistep_matches_bptt_per_head(self):
        factory = _per_head_rnn_factory()
        inputs = _seq_inputs()
        bptt = bptt_param_gradients(factory, inputs)
        online = online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        assert_param_gradients_close(online, bptt, atol=1e-4)

    def test_d_rtrl_dense_equation_matches_bespoke_matmul_model(self):
        """The same tanh RNN built once with einsum('bk,kn->bn') and once
        with braintrace.matmul must produce identical online gradients."""
        n_rec, n_in, seed = 4, 3, 0

        def make_factory(use_einsum):
            def factory():
                class Net(brainstate.nn.Module):
                    def __init__(self):
                        super().__init__()
                        k = jax.random.PRNGKey
                        self.w = brainstate.ParamState(
                            0.1 * jax.random.normal(k(seed), (n_rec, n_rec)))
                        self.win = brainstate.ParamState(
                            0.1 * jax.random.normal(k(seed + 1), (n_in, n_rec)))
                        self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

                    def update(self, x):
                        if use_einsum:
                            rec = braintrace.einsum('bk,kn->bn', self.h.value, self.w.value)
                        else:
                            rec = braintrace.matmul(self.h.value, self.w.value)
                        self.h.value = jax.nn.tanh(x @ self.win.value + rec)
                        return self.h.value

                return Net()
            return factory

        inputs = _seq_inputs(seed=43)
        g_einsum = online_param_gradients(
            make_factory(True), inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        g_matmul = online_param_gradients(
            make_factory(False), inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        assert_param_gradients_close(g_einsum, g_matmul, atol=1e-6)

    def test_pp_prop_direction_aligned(self):
        factory = _per_head_rnn_factory()
        inputs = _seq_inputs(seed=44)
        bptt = bptt_param_gradients(factory, inputs)
        approx = online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.pp_prop(
                m, decay_or_rank=0.9, vjp_method='multi-step'))
        assert_direction_aligned(
            approx, bptt, min_cosine=0.9, min_sign_agreement=0.8, keys=[('w',)])
