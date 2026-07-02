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


from braintrace._op import (
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_DT_TO_T,
)
from braintrace._op.op_rule_oracle import assert_xy_to_dw_matches_vjp


class TestGmmEtpRules:
    B, G, K, N, A = 5, 2, 3, 4, 2   # batch, groups, in, out, n_state

    def test_dt_to_t_broadcasts_hidden_over_in_axis(self):
        brainstate.random.seed(10)
        hd = brainstate.random.randn(self.B, self.G, self.N)
        tr = {'weight': brainstate.random.randn(self.B, self.G, self.K, self.N),
              'bias': brainstate.random.randn(self.B, self.G, self.N)}
        out = ETP_RULES_DT_TO_T[etp_gmm_p](hd, tr, has_bias=True)
        np.testing.assert_allclose(out['weight'], tr['weight'] * hd[:, :, None, :], atol=1e-6)
        np.testing.assert_allclose(out['bias'], tr['bias'] * hd, atol=1e-6)

    def test_dt_to_t_grad_context_batch_stripped(self):
        brainstate.random.seed(11)
        hd = brainstate.random.randn(self.G, self.N)
        tr = {'weight': brainstate.random.randn(self.G, self.K, self.N)}
        out = ETP_RULES_DT_TO_T[etp_gmm_p](hd, tr, has_bias=False)
        np.testing.assert_allclose(out['weight'], tr['weight'] * hd[:, None, :], atol=1e-6)

    def test_xy_to_dw_matches_vjp_plain(self):
        brainstate.random.seed(12)
        x = brainstate.random.randn(self.B, self.G, self.K)
        hd = brainstate.random.randn(self.B, self.G, self.N)
        weights = {'weight': brainstate.random.randn(self.G, self.K, self.N),
                   'bias': brainstate.random.randn(self.G, self.N)}
        assert_xy_to_dw_matches_vjp(
            rule=ETP_RULES_XY_TO_DW[etp_gmm_p],
            impl=lambda wd: jnp.einsum('bgk,gkn->bgn', x, wd['weight']) + wd['bias'],
            x=x, hidden_dim=hd, weights=weights, params={'has_bias': True},
        )

    def test_xy_to_dw_matches_vjp_with_weight_fn(self):
        brainstate.random.seed(13)
        x = brainstate.random.randn(self.B, self.G, self.K)
        hd = brainstate.random.randn(self.B, self.G, self.N)
        weights = {'weight': brainstate.random.randn(self.G, self.K, self.N)}
        fn = lambda ww: jnp.tanh(ww)
        assert_xy_to_dw_matches_vjp(
            rule=ETP_RULES_XY_TO_DW[etp_gmm_p],
            impl=lambda wd: jnp.einsum('bgk,gkn->bgn', x, fn(wd['weight'])),
            x=x, hidden_dim=hd, weights=weights,
            params={'has_bias': False, 'weight_fn': fn},
        )

    def test_init_drtrl_shapes_and_dtype(self):
        out = ETP_RULES_INIT_DRTRL[etp_gmm_p](
            _fake_var((self.B, self.G, self.K), jnp.bfloat16),
            _fake_var((self.B, self.G, self.N), jnp.bfloat16),
            {'weight': _fake_var((self.G, self.K, self.N), jnp.bfloat16),
             'bias': _fake_var((self.G, self.N), jnp.bfloat16)},
            self.A,
        )
        assert out['weight'].shape == (self.B, self.G, self.K, self.N, self.A)
        assert out['weight'].dtype == jnp.bfloat16
        assert out['bias'].shape == (self.B, self.G, self.N, self.A)
        assert out['bias'].dtype == jnp.bfloat16

    def test_init_drtrl_dtype_promotes_across_operands(self):
        """Trace dtype is jnp.result_type over x/y/weight avals, not just weight."""
        out = ETP_RULES_INIT_DRTRL[etp_gmm_p](
            _fake_var((self.B, self.G, self.K), jnp.float32),
            _fake_var((self.B, self.G, self.N), jnp.float32),
            {'weight': _fake_var((self.G, self.K, self.N), jnp.bfloat16)},
            self.A,
        )
        assert out['weight'].dtype == jnp.float32

    def test_init_pp_shape_and_dtype(self):
        out = ETP_RULES_INIT_PP[etp_gmm_p](
            _fake_var((self.B, self.G, self.K)),
            _fake_var((self.B, self.G, self.N), jnp.bfloat16),
            {'weight': _fake_var((self.G, self.K, self.N))},
            self.A,
        )
        assert out.shape == (self.B, self.G, self.N, self.A)
        assert out.dtype == jnp.bfloat16


class TestGmvEtpRules:
    G, K, N, A = 2, 3, 4, 2

    def test_dt_to_t(self):
        brainstate.random.seed(20)
        hd = brainstate.random.randn(self.G, self.N)
        tr = {'weight': brainstate.random.randn(self.G, self.K, self.N),
              'bias': brainstate.random.randn(self.G, self.N)}
        out = ETP_RULES_DT_TO_T[etp_gmv_p](hd, tr, has_bias=True)
        np.testing.assert_allclose(out['weight'], tr['weight'] * hd[:, None, :], atol=1e-6)
        np.testing.assert_allclose(out['bias'], tr['bias'] * hd, atol=1e-6)

    def test_xy_to_dw_matches_vjp(self):
        brainstate.random.seed(21)
        x = brainstate.random.randn(self.G, self.K)
        hd = brainstate.random.randn(self.G, self.N)
        weights = {'weight': brainstate.random.randn(self.G, self.K, self.N)}
        assert_xy_to_dw_matches_vjp(
            rule=ETP_RULES_XY_TO_DW[etp_gmv_p],
            impl=lambda wd: jnp.einsum('gk,gkn->gn', x, wd['weight']),
            x=x, hidden_dim=hd, weights=weights, params={'has_bias': False},
        )

    def test_init_shapes_and_dtypes(self):
        drtrl = ETP_RULES_INIT_DRTRL[etp_gmv_p](
            _fake_var((self.G, self.K), jnp.float16),
            _fake_var((self.G, self.N), jnp.float16),
            {'weight': _fake_var((self.G, self.K, self.N), jnp.float16)}, self.A)
        assert drtrl['weight'].shape == (self.G, self.K, self.N, self.A)
        assert drtrl['weight'].dtype == jnp.float16
        pp = ETP_RULES_INIT_PP[etp_gmv_p](
            _fake_var((self.G, self.K)), _fake_var((self.G, self.N), jnp.float16),
            {'weight': _fake_var((self.G, self.K, self.N))}, self.A)
        assert pp.shape == (self.G, self.N, self.A)
        assert pp.dtype == jnp.float16


from braintrace._op import get_fast_path_rules


class TestGroupedFastPath:
    B, G, K, N, A = 5, 2, 3, 4, 2

    def test_registered_and_shared(self):
        rules = get_fast_path_rules(etp_gmm_p)
        assert rules is not None
        assert rules is get_fast_path_rules(etp_gmv_p)

    def test_applicable_gate(self):
        rules = get_fast_path_rules(etp_gmm_p)
        assert rules.applicable({'weight_fn': None, 'bias_fn': None}) is True
        assert rules.applicable({'weight_fn': jnp.tanh, 'bias_fn': None}) is False
        assert rules.applicable({'weight_fn': None, 'bias_fn': jnp.tanh}) is False

    def test_instant(self):
        brainstate.random.seed(30)
        x = brainstate.random.randn(self.B, self.G, self.K)
        df = brainstate.random.randn(self.B, self.G, self.N, self.A)
        out = get_fast_path_rules(etp_gmm_p).instant(x, df, True)
        want = jnp.einsum('bgk,bgna->bgkna', x, df)
        np.testing.assert_allclose(out['weight'], want, atol=1e-6)
        np.testing.assert_allclose(out['bias'], df, atol=1e-6)

    def test_recurrent_general_vs_manual(self):
        brainstate.random.seed(31)
        diag = brainstate.random.randn(self.B, self.G, self.N, self.A, self.A)
        bwg = {'weight': brainstate.random.randn(self.B, self.G, self.K, self.N, self.A),
               'bias': brainstate.random.randn(self.B, self.G, self.N, self.A)}
        out = get_fast_path_rules(etp_gmm_p).recurrent(diag, bwg, self.A)
        # reference einsum: x = new-state axis, y = contracted old-state axis
        # (cannot reuse 'b' for a state axis — it is the batch label here)
        want_w = jnp.einsum('bgnxy,bgkny->bgknx', diag, bwg['weight'])
        want_b = jnp.einsum('bgnxy,bgny->bgnx', diag, bwg['bias'])
        np.testing.assert_allclose(out['weight'], want_w, atol=1e-5)
        np.testing.assert_allclose(out['bias'], want_b, atol=1e-5)

    def test_recurrent_single_state_shortcut_equals_einsum(self):
        brainstate.random.seed(32)
        diag = brainstate.random.randn(self.B, self.G, self.N, 1, 1)
        bwg = {'weight': brainstate.random.randn(self.B, self.G, self.K, self.N, 1)}
        out = get_fast_path_rules(etp_gmm_p).recurrent(diag, bwg, 1)
        want = jnp.einsum('bgnxy,bgkny->bgknx', diag, bwg['weight'])
        np.testing.assert_allclose(out['weight'], want, atol=1e-6)

    def test_solve_with_and_without_fold_batch(self):
        brainstate.random.seed(33)
        dl = brainstate.random.randn(self.B, self.G, self.N, self.A)
        tr = {'weight': brainstate.random.randn(self.B, self.G, self.K, self.N, self.A),
              'bias': brainstate.random.randn(self.B, self.G, self.N, self.A)}
        rules = get_fast_path_rules(etp_gmm_p)
        out = rules.solve(dl, tr, fold_batch=False)
        np.testing.assert_allclose(
            out['weight'], jnp.einsum('bgna,bgkna->bgkn', dl, tr['weight']), atol=1e-5)
        folded = rules.solve(dl, tr, fold_batch=True)
        np.testing.assert_allclose(
            folded['weight'], jnp.einsum('bgna,bgkna->gkn', dl, tr['weight']), atol=1e-5)
        np.testing.assert_allclose(
            folded['bias'], jnp.einsum('bgna,bgna->gn', dl, tr['bias']), atol=1e-5)


class TestPublicExports:

    def test_top_level_exports(self):
        assert braintrace.grouped_matmul is grouped_matmul
        assert 'grouped_matmul' in braintrace.__all__

    def test_op_package_exports(self):
        import braintrace._op as op
        assert op.grouped_matmul is grouped_matmul
        for name in ('etp_gmm_p', 'etp_gmv_p', 'grouped_matmul'):
            assert name in op.__all__


from braintrace._algorithm.oracle import (
    assert_direction_aligned,
    assert_param_gradients_close,
    bptt_param_gradients,
    online_param_gradients,
)


def _grouped_tanh_rnn_factory(G=2, K=3, n_in=3, seed=0):
    """tanh RNN whose recurrence is a block-diagonal grouped_matmul.

    ``w (G, K, K)`` is the ETP recurrent weight; ``win`` is a plain input
    projection (excluded from ETP). Hidden state is flat ``(1, G*K)``;
    reshapes around the grouped op are plain JAX ops on the y→h path.
    """

    def factory():
        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                k = jax.random.PRNGKey
                self.w = brainstate.ParamState(
                    0.1 * jax.random.normal(k(seed), (G, K, K)))
                self.win = brainstate.ParamState(
                    0.1 * jax.random.normal(k(seed + 1), (n_in, G * K)))
                self.h = brainstate.HiddenState(jnp.zeros((1, G * K)))

            def update(self, x):
                inp = x @ self.win.value
                rec = braintrace.grouped_matmul(
                    self.h.value.reshape(1, G, K), self.w.value)
                self.h.value = jax.nn.tanh(inp + rec.reshape(1, G * K))
                return self.h.value

        return Net()

    return factory


def _seq_inputs(T=6, n_in=3, seed=42):
    brainstate.random.seed(seed)
    return brainstate.random.randn(T, n_in)


class TestGroupedOracle:

    def test_d_rtrl_multistep_matches_bptt(self):
        factory = _grouped_tanh_rnn_factory()
        inputs = _seq_inputs()
        bptt = bptt_param_gradients(factory, inputs)
        online = online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        )
        assert_param_gradients_close(online, bptt, atol=1e-4)

    def test_d_rtrl_multistep_matches_bptt_with_bias(self):
        G, K, n_in = 2, 3, 3

        def factory():
            class Net(brainstate.nn.Module):
                def __init__(self):
                    super().__init__()
                    k = jax.random.PRNGKey
                    self.w = brainstate.ParamState(
                        0.1 * jax.random.normal(k(0), (G, K, K)))
                    self.b = brainstate.ParamState(
                        0.1 * jax.random.normal(k(1), (G, K)))
                    self.win = brainstate.ParamState(
                        0.1 * jax.random.normal(k(2), (n_in, G * K)))
                    self.h = brainstate.HiddenState(jnp.zeros((1, G * K)))

                def update(self, x):
                    rec = braintrace.grouped_matmul(
                        self.h.value.reshape(1, G, K), self.w.value, self.b.value)
                    self.h.value = jax.nn.tanh(x @ self.win.value + rec.reshape(1, G * K))
                    return self.h.value

            return Net()

        inputs = _seq_inputs(seed=43)
        bptt = bptt_param_gradients(factory, inputs)
        online = online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        )
        assert_param_gradients_close(online, bptt, atol=1e-4)

    def test_d_rtrl_with_weight_fn_matches_bptt(self):
        """weight_fn forces the generic rule path (fast path gated off)."""
        G, K, n_in = 2, 3, 3

        def factory():
            class Net(brainstate.nn.Module):
                def __init__(self):
                    super().__init__()
                    k = jax.random.PRNGKey
                    self.w = brainstate.ParamState(
                        0.1 * jax.random.normal(k(0), (G, K, K)))
                    self.win = brainstate.ParamState(
                        0.1 * jax.random.normal(k(1), (n_in, G * K)))
                    self.h = brainstate.HiddenState(jnp.zeros((1, G * K)))

                def update(self, x):
                    rec = braintrace.grouped_matmul(
                        self.h.value.reshape(1, G, K), self.w.value,
                        weight_fn=jnp.tanh)
                    self.h.value = jax.nn.tanh(x @ self.win.value + rec.reshape(1, G * K))
                    return self.h.value

            return Net()

        inputs = _seq_inputs(seed=44)
        bptt = bptt_param_gradients(factory, inputs)
        online = online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        )
        assert_param_gradients_close(online, bptt, atol=1e-4)

    def test_fast_and_legacy_solve_parity(self):
        factory = _grouped_tanh_rnn_factory()
        inputs = _seq_inputs()
        fast = online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(
                m, vjp_method='multi-step', fast_solve=True))
        legacy = online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.D_RTRL(
                m, vjp_method='multi-step', fast_solve=False))
        assert_param_gradients_close(fast, legacy, atol=1e-5)

    def test_pp_prop_direction_aligned_with_bptt(self):
        factory = _grouped_tanh_rnn_factory()
        inputs = _seq_inputs()
        bptt = bptt_param_gradients(factory, inputs)
        approx = online_param_gradients(
            factory, inputs,
            algo_factory=lambda m: braintrace.pp_prop(
                m, decay_or_rank=0.9, vjp_method='multi-step'),
        )
        assert_direction_aligned(
            approx, bptt, min_cosine=0.9, min_sign_agreement=0.8,
            keys=[('w',)],
        )
