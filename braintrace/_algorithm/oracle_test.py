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

"""Tests for the gradient oracle: self-validation (BPTT vs finite-difference),
the headline exact-correctness proof (multi-step D_RTRL == BPTT), and the
single-step naive recipe asserted as directionally aligned with BPTT (the
former F-SINGLESTEP finding)."""

import brainevent
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_direction_aligned,
    assert_param_gradients_close,
    bptt_param_gradients,
    finite_difference_param_gradients,
    online_param_gradients,
    online_param_gradients_singlestep_naive,
)
from braintrace._algorithm.oracle_models import ModelSpec, tanh_rnn


def _inputs(T, n_in, seed=42):
    return jnp.asarray(np.random.RandomState(seed).randn(T, n_in).astype('float32'))


# --- Task 1: model factory ---------------------------------------------------

def test_tanh_rnn_factory_builds_runnable_model():
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    assert isinstance(spec, ModelSpec)
    assert spec.etp_param_keys == (('w',),)
    assert spec.plain_param_keys == (('win',),)

    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    keys = set(model.states(brainstate.ParamState).keys())
    assert keys == {('w',), ('win',)}

    y = model(jnp.ones((3,), dtype='float32'))
    assert y.shape == (1, 4)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_tanh_rnn_factory_is_deterministic():
    m1 = tanh_rnn(seed=0).factory(); brainstate.nn.init_all_states(m1, batch_size=1)
    m2 = tanh_rnn(seed=0).factory(); brainstate.nn.init_all_states(m2, batch_size=1)
    w1 = m1.states(brainstate.ParamState)[('w',)].value
    w2 = m2.states(brainstate.ParamState)[('w',)].value
    assert bool(jnp.allclose(w1, w2))


# --- Task 2: BPTT reference --------------------------------------------------

def test_bptt_param_gradients_shapes_and_finiteness():
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    grads = bptt_param_gradients(spec.factory, _inputs(6, 3))
    assert set(grads.keys()) == {('w',), ('win',)}
    assert grads[('w',)].shape == (4, 4)
    assert grads[('win',)].shape == (3, 4)
    for v in grads.values():
        assert bool(jnp.all(jnp.isfinite(v)))
    # win is upstream of the loss every step -> its gradient is non-trivial
    assert float(jnp.abs(grads[('win',)]).sum()) > 1e-3


# --- Task 3: finite-difference arbiter (validates BPTT) ----------------------

def test_finite_difference_matches_bptt():
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_fd = finite_difference_param_gradients(spec.factory, inputs, eps=1e-3)
    for key in g_bptt:
        diff = float(jnp.max(jnp.abs(jnp.asarray(g_bptt[key]) - jnp.asarray(g_fd[key]))))
        assert diff < 1e-3, f"{key}: BPTT vs FD maxdiff={diff:.3e}"


# --- Task 4: multi-step online gradients -------------------------------------

def test_online_multistep_gradients_shapes():
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    grads = online_param_gradients(
        spec.factory, _inputs(6, 3),
        algo_factory=lambda m: braintrace.ParamDimVjpAlgorithm(m, vjp_method='multi-step'),
    )
    assert set(grads.keys()) == {('w',), ('win',)}
    assert grads[('w',)].shape == (4, 4)
    for v in grads.values():
        assert bool(jnp.all(jnp.isfinite(v)))


# --- Task 5: comparison assertion helper -------------------------------------

def test_assert_close_passes_for_equal_trees():
    a = {('w',): jnp.ones((2, 2))}
    b = {('w',): jnp.ones((2, 2)) + 1e-7}
    assert_param_gradients_close(a, b, atol=1e-4)  # must not raise


def test_assert_close_reports_offending_key():
    a = {('w',): jnp.zeros((2, 2)), ('v',): jnp.zeros((2, 2))}
    b = {('w',): jnp.zeros((2, 2)), ('v',): jnp.ones((2, 2))}
    with pytest.raises(AssertionError, match=r"\('v',\)"):
        assert_param_gradients_close(a, b, atol=1e-4)


def test_assert_close_can_restrict_to_subset_of_keys():
    a = {('w',): jnp.zeros((2, 2)), ('v',): jnp.zeros((2, 2))}
    b = {('w',): jnp.zeros((2, 2)), ('v',): jnp.ones((2, 2))}
    assert_param_gradients_close(a, b, atol=1e-4, keys=[('w',)])  # ('v',) ignored


# --- Task 6: HEADLINE — multi-step D_RTRL == BPTT ----------------------------

def test_d_rtrl_multistep_matches_bptt():
    """Exact algorithm: multi-step D_RTRL must reproduce the BPTT gradient exactly."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_online = online_param_gradients(
        spec.factory, inputs,
        algo_factory=lambda m: braintrace.ParamDimVjpAlgorithm(m, vjp_method='multi-step'),
    )
    # multi-step reproduces BPTT for ALL params (observed maxdiff 0.0 in the spike)
    assert_param_gradients_close(g_online, g_bptt, atol=1e-4)


# --- Task 7: former F-SINGLESTEP — single-step naive is directionally aligned -

def test_singlestep_naive_directionally_aligned_with_bptt():
    """Approximate recipe: naive single-step per-step-grad summation does NOT
    equal BPTT element-wise — only the multi-step VJP path is exact (see
    ``test_d_rtrl_multistep_matches_bptt``, observed maxdiff 0.0).

    Per the algorithm taxonomy, the *guaranteed* property of this approximate
    single-step recipe is directional, not element-wise: it stays strongly
    aligned with BPTT (high cosine, consistent signs) with a bounded magnitude
    bias from the single-step diagonal approximation. This finding was formerly
    F-SINGLESTEP, pinned as a strict xfail against an (unattainable) element-wise
    match; it is now asserted positively as the property that actually holds.
    Observed at T=6, seed=0 for the ETP weight: cosine 0.9955, sign agreement
    1.0, relmag 1.066."""
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_naive = online_param_gradients_singlestep_naive(
        spec.factory, inputs,
        algo_factory=lambda m: braintrace.ParamDimVjpAlgorithm(m, vjp_method='single-step'),
    )
    # ETP weight: not element-wise equal to BPTT, but strongly direction-aligned
    # with bounded magnitude bias. Thresholds are set with margin below/around the
    # observed values (cos 0.9955, sign 1.0, relmag 1.066).
    assert_direction_aligned(
        g_naive, g_bptt,
        min_cosine=0.99,
        min_sign_agreement=0.9,
        mag_bounds=(0.8, 1.3),
        keys=list(spec.etp_param_keys),
    )


# =============================================================================
# Audit Task 11: cross-family single-step oracle suite (T1, T3)
# =============================================================================
#
# Every family below is an *exactly-diagonal* leaky-integrator model
# (``h <- leak * h + drive``), so single-step D-RTRL's diagonal approximation
# is exact and must reproduce a BPTT oracle element-wise for every parameter,
# at every T. This is the "real test" the audit's T1 finding asked for: prior
# coverage of the conv-kernel (Task 7) and LoRA-B (Task 6) fixes either used
# an all-zero hidden state as the op's own *input* (making the weight/kernel
# gradient trivially zero on both sides of the comparison) or never drove
# the op with genuinely random, nonzero data at all. The factories here feed
# real ``brainstate.random`` data through every op family, so the kernel and
# weight gradients are actually exercised.
#
# ``pp_prop`` (ES-D-RTRL, an *approximate* algorithm) is exact only at T=1
# (no history to factor/decay yet); for T>1 it is expected to diverge from
# BPTT, so only a loose, structural-break-catching bound is asserted there
# (rel < 1.0), never element-wise equality.

LEAK = 0.5
_TOL = 1e-10


def _rel_err(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    denom = float(jnp.maximum(jnp.abs(a).max(), 1e-12))
    return float(jnp.abs(a - b).max() / denom)


def _dense_mm_factory():
    """Batched dense ``matmul`` (+bias) leaky-integrator, ``h`` shape ``(1, n_out)``."""
    brainstate.random.seed(11)
    n_in, n_out = 3, 4
    w0 = 0.1 * brainstate.random.randn(n_in, n_out)
    b0 = 0.05 * brainstate.random.randn(n_out)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(w0)
            self.b = brainstate.ParamState(b0)
            self.h = brainstate.HiddenState(jnp.zeros((1, n_out)))

        def update(self, x):
            drive = braintrace.matmul(x, self.w.value, bias=self.b.value)
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def _dense_mv_factory():
    """Unbatched dense ``matmul`` leaky-integrator, ``h`` shape ``(n_out,)``."""
    brainstate.random.seed(12)
    n_in, n_out = 3, 4
    w0 = 0.1 * brainstate.random.randn(n_in, n_out)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(w0)
            self.h = brainstate.HiddenState(jnp.zeros((n_out,)))

        def update(self, x):
            drive = braintrace.matmul(x, self.w.value)
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def _elemwise_factory():
    """Elementwise-scaled input leaky-integrator, ``h`` shape ``(n,)``."""
    brainstate.random.seed(13)
    n = 4
    w0 = 0.5 + 0.1 * brainstate.random.randn(n)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(w0)
            self.h = brainstate.HiddenState(jnp.zeros((n,)))

        def update(self, x):
            drive = x * braintrace.element_wise(self.w.value)
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def _conv_default_factory():
    """1-D conv, JAX-default (NCH/OIH) layout, kernel width 3 (spatial extent > 1)."""
    brainstate.random.seed(14)
    in_ch, out_ch, kw, length = 2, 3, 3, 8
    k0 = 0.1 * brainstate.random.randn(out_ch, in_ch, kw)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.k = brainstate.ParamState(k0)
            self.h = brainstate.HiddenState(jnp.zeros((1, out_ch, length)))

        def update(self, x):
            drive = braintrace.conv(x, self.k.value, strides=(1,), padding='SAME')
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def _conv_nwc_bias_factory():
    """1-D conv, channel-last (NWC/WIO) layout with a trainable bias."""
    brainstate.random.seed(15)
    in_ch, out_ch, kw, length = 2, 3, 3, 8
    k0 = 0.1 * brainstate.random.randn(kw, in_ch, out_ch)
    b0 = 0.05 * brainstate.random.randn(out_ch)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.k = brainstate.ParamState(k0)
            self.b = brainstate.ParamState(b0)
            self.h = brainstate.HiddenState(jnp.zeros((1, length, out_ch)))

        def update(self, x):
            drive = braintrace.conv(
                x, self.k.value, self.b.value,
                strides=(1,), padding='SAME',
                dimension_numbers=('NWC', 'WIO', 'NWC'),
            )
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def _sparse_csr():
    dense_mask = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ], dtype=bool)
    return brainevent.CSR.fromdense(jnp.asarray(dense_mask, dtype=jnp.float64)), dense_mask.shape[1]


def _sparse_unbatched_factory():
    """Unbatched sparse ``matmul`` (real ``brainevent.CSR``), ``h`` shape ``(n_rec,)``."""
    brainstate.random.seed(16)
    csr, n_rec = _sparse_csr()
    w0 = 0.1 * brainstate.random.randn(csr.data.shape[0])

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(w0)
            self.h = brainstate.HiddenState(jnp.zeros((n_rec,)))

        def update(self, x):
            drive = braintrace.sparse_matmul(x, self.w.value, sparse_mat=csr)
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def _sparse_batched_factory():
    """Batched (batch=2) sparse ``matmul``, ``h`` shape ``(2, n_rec)``."""
    brainstate.random.seed(17)
    csr, n_rec = _sparse_csr()
    batch = 2
    w0 = 0.1 * brainstate.random.randn(csr.data.shape[0])

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(w0)
            self.h = brainstate.HiddenState(jnp.zeros((batch, n_rec)))

        def update(self, x):
            drive = braintrace.sparse_matmul(x, self.w.value, sparse_mat=csr)
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def _lora_factory():
    """Batched LoRA ``lora_matmul`` with a trainable bias and ``alpha != 1``."""
    brainstate.random.seed(18)
    n_in, n_rec, rank = 3, 4, 2
    b0_ = 0.1 * brainstate.random.randn(n_in, rank)
    a0_ = 0.1 * brainstate.random.randn(rank, n_rec)
    bias0 = 0.05 * brainstate.random.randn(n_rec)

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.B = brainstate.ParamState(b0_)
            self.A = brainstate.ParamState(a0_)
            self.bias = brainstate.ParamState(bias0)
            self.h = brainstate.HiddenState(jnp.zeros((1, n_rec)))

        def update(self, x):
            drive = braintrace.lora_matmul(
                x, self.B.value, self.A.value, alpha=2.0, bias=self.bias.value,
            )
            self.h.value = LEAK * self.h.value + drive
            return self.h.value

    return Net()


def _xs_for(name, T, seed):
    brainstate.random.seed(seed)
    shapes = {
        'dense_mm': (T, 1, 3),
        'dense_mv': (T, 3),
        'elemwise': (T, 4),
        'conv_default': (T, 1, 2, 8),
        'conv_nwc_bias': (T, 1, 8, 2),
        'sparse_unbatched': (T, 3),
        'sparse_batched': (T, 2, 3),
        'lora': (T, 1, 3),
    }
    return 0.3 * brainstate.random.randn(*shapes[name])


# name -> (factory, xs seed)
_FAMILIES = {
    'dense_mm': (_dense_mm_factory, 101),
    'dense_mv': (_dense_mv_factory, 102),
    'elemwise': (_elemwise_factory, 103),
    'conv_default': (_conv_default_factory, 104),
    'conv_nwc_bias': (_conv_nwc_bias_factory, 105),
    'sparse_unbatched': (_sparse_unbatched_factory, 106),
    'sparse_batched': (_sparse_batched_factory, 107),
    'lora': (_lora_factory, 108),
}


@pytest.mark.parametrize('name', sorted(_FAMILIES))
@pytest.mark.parametrize('T', [1, 4])
def test_d_rtrl_singlestep_matches_bptt_across_families(name, T):
    """D_RTRL (param-dim, single-step) is an *exact* algorithm: for every op
    family, driven by real nonzero random input, it must reproduce the BPTT
    gradient element-wise for every trainable factor -- including the conv
    kernel at spatial extent > 1 (Task 7) and ``lora_b`` (Task 6), neither of
    which any pre-existing test exercised with a nonzero op input."""
    factory, seed = _FAMILIES[name]
    with brainstate.environ.context(precision=64):
        xs = _xs_for(name, T, seed)
        g_bptt = bptt_param_gradients(factory, xs)
        g_online = online_param_gradients_singlestep_naive(
            factory, xs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='single-step'),
        )
        for key in g_bptt:
            rel = _rel_err(g_bptt[key], g_online[key])
            assert rel < _TOL, f'{name} T={T} {key}: D_RTRL vs BPTT rel={rel:.3e}'


@pytest.mark.parametrize('name', sorted(_FAMILIES))
def test_pp_prop_singlestep_exact_at_t1_across_families(name):
    """pp_prop (ES-D-RTRL, IO-dim, approximate) has no history to factor or
    decay at T=1, so it must also match BPTT exactly there, for every family.
    """
    factory, seed = _FAMILIES[name]
    with brainstate.environ.context(precision=64):
        xs = _xs_for(name, 1, seed)
        g_bptt = bptt_param_gradients(factory, xs)
        g_online = online_param_gradients_singlestep_naive(
            factory, xs,
            algo_factory=lambda m: braintrace.pp_prop(m, decay_or_rank=0.9, vjp_method='single-step'),
        )
        for key in g_bptt:
            rel = _rel_err(g_bptt[key], g_online[key])
            assert rel < _TOL, f'{name} T=1 {key}: pp_prop vs BPTT rel={rel:.3e}'


@pytest.mark.parametrize('name', sorted(_FAMILIES))
def test_pp_prop_singlestep_bounded_at_t2_across_families(name):
    """pp_prop is an *approximate* algorithm beyond T=1: at T=2 it factors /
    decays history and is **not** expected to match BPTT element-wise. This
    only asserts a loose bound (rel < 1.0) to catch structural breaks (NaNs,
    blow-ups, shape errors) without codifying the approximation's magnitude
    as a spec.
    """
    factory, seed = _FAMILIES[name]
    with brainstate.environ.context(precision=64):
        xs = _xs_for(name, 2, seed)
        g_bptt = bptt_param_gradients(factory, xs)
        g_online = online_param_gradients_singlestep_naive(
            factory, xs,
            algo_factory=lambda m: braintrace.pp_prop(m, decay_or_rank=0.9, vjp_method='single-step'),
        )
        for key in g_bptt:
            rel = _rel_err(g_bptt[key], g_online[key])
            assert np.isfinite(rel) and rel < 1.0, (
                f'{name} T=2 {key}: pp_prop vs BPTT rel={rel:.3e} (expected bounded, not exact)'
            )


def test_pp_prop_conv_bias_matches_bptt():
    """Formerly ``test_pp_prop_conv_bias_known_limitation`` (finding F-26).

    ``_conv_xy_to_dw`` returns the bias Jacobian per output position by design;
    the param-dim path reduces it in ``_conv_dt_to_t``, but the IO-dim solver
    calls ``xy_to_dw`` directly at solve time and used to hand the un-reduced
    ``(batch, *spatial, out_ch)`` array back to ``custom_vjp``, which rejected it
    against the bias's own ``(out_ch,)``. The IO-dim solver now reduces every
    produced leaf to its parameter's shape. At T=1 there is no history to factor,
    so pp_prop must match BPTT exactly.
    """
    factory, seed = _FAMILIES['conv_nwc_bias']
    with brainstate.environ.context(precision=64):
        xs = _xs_for('conv_nwc_bias', 1, seed)
        g_bptt = bptt_param_gradients(factory, xs)
        g_online = online_param_gradients_singlestep_naive(
            factory, xs,
            algo_factory=lambda m: braintrace.pp_prop(
                m, decay_or_rank=0.9, vjp_method='single-step'),
        )
        for key in g_bptt:
            rel = _rel_err(g_bptt[key], g_online[key])
            assert rel < _TOL, f'conv_nwc_bias T=1 {key}: pp_prop vs BPTT rel={rel:.3e}'


# --- P1: negative-control helpers -------------------------------------------

def test_flat_gradient_leaves_handles_nested_and_units():
    """Gradient trees are nested dicts and may carry units; the flattener must
    yield plain arrays keyed by a stable label."""
    import brainunit as u
    from braintrace._algorithm.oracle import flat_gradient_leaves
    tree = {
        ('syn', 'comm', 'weight'): {'weight': jnp.ones((2, 3)) * u.mS,
                                    'bias': jnp.zeros((3,)) * u.mS},
        ('w',): jnp.arange(4.0),
    }
    flat = flat_gradient_leaves(tree)
    assert len(flat) == 3
    for arr in flat.values():
        assert not isinstance(arr, u.Quantity)
    assert sorted(k.split('|')[0] for k in flat) == ['syn/comm/weight',
                                                    'syn/comm/weight', 'w']


def test_gradient_norm_and_relative_deviation():
    from braintrace._algorithm.oracle import gradient_norm, relative_deviation
    a = {('w',): jnp.array([3.0, 4.0])}
    b = {('w',): jnp.array([3.0, 4.0])}
    assert gradient_norm(a) == pytest.approx(5.0, abs=1e-6)
    assert relative_deviation(a, b) == pytest.approx(0.0, abs=1e-12)
    # relative_deviation(actual, expected) normalises by ||expected||:
    # ||[3,4] - [0,4]|| / ||[0,4]|| == 3 / 4.
    c = {('w',): jnp.array([0.0, 4.0])}
    assert relative_deviation(a, c) == pytest.approx(3.0 / 4.0, abs=1e-6)
    # all-zero reference: infinite deviation, not a silent zero.
    zero = {('w',): jnp.zeros(2)}
    assert relative_deviation(a, zero) == float('inf')
    assert relative_deviation(zero, zero) == 0.0


def test_assert_model_is_live_passes_on_live_model():
    from braintrace._algorithm.oracle import assert_model_is_live
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    xs = _inputs(4, 3)
    norm = assert_model_is_live(spec.factory, xs)
    assert norm > 0.0


def test_assert_model_is_live_rejects_a_dead_model():
    """A model whose output is detached from its parameter has a zero gradient,
    so any comparison against it asserts nothing. The guard must reject it.

    The parameter is still *used* — an entirely unused ``ParamState`` makes
    ``brainstate``'s gradient transform raise before the oracle sees it — but
    ``stop_gradient`` severs the derivative, which is exactly the silent-SNN
    situation F-25 describes: a live forward pass with a zero gradient.
    """
    from braintrace._algorithm.oracle import assert_model_is_live

    def factory():
        class Dead(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((3, 3)))
                self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

            def update(self, x):
                self.h.value = jax.lax.stop_gradient(x @ self.w.value)
                return self.h.value

        return Dead()

    xs = jnp.ones((3, 1, 3))
    with pytest.raises(AssertionError, match='gradient norm'):
        assert_model_is_live(factory, xs)


def test_assert_gradients_differ_flags_a_dead_knob():
    from braintrace._algorithm.oracle import assert_gradients_differ
    a = {('w',): jnp.array([1.0, 2.0])}
    assert_gradients_differ(a, {('w',): jnp.array([1.0, 5.0])})
    with pytest.raises(AssertionError, match='indistinguishable'):
        assert_gradients_differ(a, {('w',): jnp.array([1.0, 2.0])})


def test_assert_param_gradients_close_supports_nested_unit_trees():
    """The pre-existing helper only handled flat, unitless dicts. SNN models
    have nested weight dicts carrying units."""
    import brainunit as u
    a = {('syn',): {'weight': jnp.ones((2, 2)) * u.mS, 'bias': jnp.zeros(2) * u.mS}}
    b = {('syn',): {'weight': jnp.ones((2, 2)) * u.mS, 'bias': jnp.zeros(2) * u.mS}}
    assert_param_gradients_close(a, b, atol=1e-6)
    c = {('syn',): {'weight': jnp.full((2, 2), 2.0) * u.mS, 'bias': jnp.zeros(2) * u.mS}}
    with pytest.raises(AssertionError, match='maxabsdiff'):
        assert_param_gradients_close(a, c, atol=1e-6)


# --- P4: statistical acceptance infrastructure -------------------------------
#
# The roadmap flags this as "statistical test infrastructure the repo does not
# have". Its job is to accept an *unbiased but noisy* estimator (UORO) and reject
# a *biased* one, which a decay bound like `deviation <= 4/sqrt(N)` cannot do: a
# fixed 10% bias satisfies that at every N a test can afford. So the helpers are
# tested here against estimators whose bias is known by construction.

from braintrace._algorithm.oracle import (  # noqa: E402
    assert_unbiased_estimator,
    fixed_gradient_directions,
    project_gradient,
)

_REF_TREE = {
    ('a',): {'weight': jnp.asarray([[1.0, -2.0], [0.5, 3.0]]), 'bias': jnp.asarray([1.0, -1.0])},
    ('b',): jnp.asarray([2.0, 0.25, -0.5]),
}
_A_KEYS = ["a|['weight']", "a|['bias']"]
_B_KEYS = ['b|']


def _flat_labels(tree):
    from braintrace._algorithm.oracle import flat_gradient_leaves
    return set(flat_gradient_leaves(tree))


def _noisy_samples(n, *, bias=0.0, scale=0.3, seed=0, tree=None):
    """`n` gradient trees equal to the reference times `(1 + bias)` plus noise."""
    tree = _REF_TREE if tree is None else tree
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        out.append(jax.tree.map(
            lambda a: jnp.asarray(
                np.asarray(a) * (1.0 + bias)
                + scale * np.abs(np.asarray(a)).max() * rng.randn(*np.shape(a))),
            tree))
    return out


class TestFixedGradientDirections:

    def test_directions_match_the_tree_structure_and_are_unit_norm(self):
        dirs = fixed_gradient_directions(_REF_TREE, 4, seed=0)
        assert len(dirs) == 4
        for d in dirs:
            assert set(d) == _flat_labels(_REF_TREE)
            norm = np.sqrt(sum(float((np.asarray(v) ** 2).sum()) for v in d.values()))
            np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_directions_are_deterministic_in_the_seed(self):
        a = fixed_gradient_directions(_REF_TREE, 3, seed=7)
        b = fixed_gradient_directions(_REF_TREE, 3, seed=7)
        c = fixed_gradient_directions(_REF_TREE, 3, seed=8)
        for x, y in zip(a, b):
            for k in x:
                np.testing.assert_array_equal(np.asarray(x[k]), np.asarray(y[k]))
        assert any(not np.allclose(np.asarray(a[0][k]), np.asarray(c[0][k]))
                   for k in a[0])

    def test_keys_restricts_the_support(self):
        dirs = fixed_gradient_directions(_REF_TREE, 2, seed=0, keys=_B_KEYS)
        for d in dirs:
            assert set(d) == set(_B_KEYS)

    def test_projection_is_linear(self):
        d = fixed_gradient_directions(_REF_TREE, 1, seed=0)[0]
        doubled = jax.tree.map(lambda a: a * 2.0, _REF_TREE)
        np.testing.assert_allclose(
            project_gradient(doubled, d), 2.0 * project_gradient(_REF_TREE, d),
            rtol=1e-6)


def _flat_labels(tree):
    from braintrace._algorithm.oracle import flat_gradient_leaves
    return set(flat_gradient_leaves(tree))


class TestAssertUnbiasedEstimator:

    def test_accepts_an_unbiased_noisy_estimator(self):
        samples = _noisy_samples(256, bias=0.0, scale=0.3, seed=1)
        assert_unbiased_estimator(samples, _REF_TREE, seed=0)

    def test_rejects_a_small_multiplicative_bias(self):
        # The failure mode `4/sqrt(N)` cannot see. 8% of the reference, with
        # noise an order of magnitude larger than the bias per sample.
        samples = _noisy_samples(256, bias=0.08, scale=0.3, seed=1)
        with pytest.raises(AssertionError, match='outside its confidence interval'):
            assert_unbiased_estimator(samples, _REF_TREE, seed=0)

    def test_rejects_an_unbiased_but_uselessly_noisy_estimator(self):
        # Unbiased, but the interval is so wide that passing means nothing; the
        # tightness clause must catch it or the test is vacuous.
        samples = _noisy_samples(24, bias=0.0, scale=40.0, seed=2)
        with pytest.raises(AssertionError, match='too wide'):
            assert_unbiased_estimator(samples, _REF_TREE, seed=0)

    def test_the_failure_message_prints_every_direction(self):
        samples = _noisy_samples(128, bias=0.5, scale=0.2, seed=3)
        with pytest.raises(AssertionError) as info:
            assert_unbiased_estimator(samples, _REF_TREE, num_directions=5, seed=0)
        text = str(info.value)
        # a statistical failure that prints one number is unactionable
        assert text.count('direction') >= 5

    def test_a_single_sample_is_rejected_rather_than_dividing_by_zero(self):
        with pytest.raises(AssertionError, match='at least 2'):
            assert_unbiased_estimator(_noisy_samples(1), _REF_TREE, seed=0)

    def test_keys_restricts_the_comparison(self):
        # Bias only the ('b',) leaf; restricting to ('a',) must then pass and
        # restricting to ('b',) must fail. This is the guard against a whole-tree
        # comparison being dominated by leaves the axis does not touch.
        rng = np.random.RandomState(4)
        samples = []
        for _ in range(256):
            s = {
                ('a',): jax.tree.map(
                    lambda x: jnp.asarray(np.asarray(x) + 0.3 * rng.randn(*np.shape(x))),
                    _REF_TREE[('a',)]),
                ('b',): jnp.asarray(np.asarray(_REF_TREE[('b',)]) * 1.4
                                    + 0.3 * rng.randn(3)),
            }
            samples.append(s)
        assert_unbiased_estimator(samples, _REF_TREE, seed=0, keys=_A_KEYS)
        with pytest.raises(AssertionError, match='outside its confidence interval'):
            assert_unbiased_estimator(samples, _REF_TREE, seed=0, keys=_B_KEYS)

    def test_an_exact_deterministic_estimator_passes(self):
        # Zero variance, zero bias: the interval is degenerate but the test must
        # not divide by zero or reject it.
        assert_unbiased_estimator([_REF_TREE] * 8, _REF_TREE, seed=0)


class TestFutureHiddenGradients:
    """The DNI target oracle: ``d(sum_{t >= b} L_t) / d h^b``."""

    def test_matches_a_hand_rolled_suffix_gradient(self):
        from braintrace._algorithm.oracle import future_hidden_gradients
        spec = tanh_rnn(n_in=3, n_rec=4)
        inputs = _inputs(5, 3)
        got = future_hidden_gradients(spec.factory, inputs, [1, 3])

        # Independent reference: re-derive the recurrence as a *pure* function of
        # (h, x) with the model's own weights, and differentiate that. This shares
        # no machinery with the helper -- not the state snapshotting, not the
        # brainstate gradient route -- so agreement is evidence, not a tautology.
        probe = spec.factory()
        brainstate.nn.init_all_states(probe, batch_size=1)
        params = probe.states(brainstate.ParamState)
        w = np.asarray(params[('w',)].value)
        win = np.asarray(params[('win',)].value)
        h0 = jnp.zeros_like(probe.states(brainstate.HiddenState)[('h',)].value)

        def step(h, x):
            return jax.nn.tanh(x @ win + h @ w)

        def suffix(h, b):
            total = 0.0
            for t in range(b, inputs.shape[0]):
                h = step(h, inputs[t])
                total = total + (h ** 2).sum()
            return total

        for idx, b in enumerate([1, 3]):
            h_at_b = h0
            for t in range(b):
                h_at_b = step(h_at_b, inputs[t])
            want = jax.grad(suffix)(h_at_b, b)
            np.testing.assert_allclose(
                np.asarray(got[idx][('h',)]), np.asarray(want), atol=1e-5)

    def test_the_last_boundary_has_no_future_loss(self):
        from braintrace._algorithm.oracle import future_hidden_gradients
        spec = tanh_rnn(n_in=3, n_rec=4)
        inputs = _inputs(4, 3)
        got = future_hidden_gradients(spec.factory, inputs, [4])
        np.testing.assert_allclose(np.asarray(got[0][('h',)]), 0.0, atol=0.0)

    def test_the_boundary_is_half_open(self):
        # The step that writes h^b belongs to the window *before* b, so its loss
        # must be excluded here. If it were included, boundary b and boundary
        # b - 1 would both count L_{b-1} and every DNI target would be wrong by
        # one loss.
        from braintrace._algorithm.oracle import future_hidden_gradients
        spec = tanh_rnn(n_in=3, n_rec=4)
        inputs = _inputs(4, 3)
        g3, g4 = future_hidden_gradients(spec.factory, inputs, [3, 4])
        assert float(np.abs(np.asarray(g3[('h',)])).max()) > 1e-6  # L_3 is future
        np.testing.assert_allclose(np.asarray(g4[('h',)]), 0.0, atol=0.0)

    def test_it_leaves_a_live_model_untouched(self):
        # It rolls the model forward internally; leaking that into the states
        # under test would corrupt every later assertion.
        from braintrace._algorithm.oracle import future_hidden_gradients
        spec = tanh_rnn(n_in=3, n_rec=4)
        inputs = _inputs(4, 3)
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        before = np.asarray(model.states(brainstate.HiddenState)[('h',)].value).copy()
        future_hidden_gradients(spec.factory, inputs, [1, 2])
        after = np.asarray(model.states(brainstate.HiddenState)[('h',)].value)
        np.testing.assert_array_equal(before, after)
