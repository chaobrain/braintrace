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


@pytest.mark.parametrize('name', sorted(set(_FAMILIES) - {'conv_nwc_bias'}))
def test_pp_prop_singlestep_exact_at_t1_across_families(name):
    """pp_prop (ES-D-RTRL, IO-dim, approximate) has no history to factor or
    decay at T=1, so it must also match BPTT exactly there, for every family.

    ``conv_nwc_bias`` is excluded here and covered separately by
    ``test_pp_prop_conv_bias_known_limitation`` -- see that test's docstring
    for the newly-discovered (pre-existing, out-of-scope-for-this-task) gap.
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


@pytest.mark.parametrize('name', sorted(set(_FAMILIES) - {'conv_nwc_bias'}))
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


def test_pp_prop_conv_bias_known_limitation():
    """Documents a newly-discovered gap found while building this suite:
    ``pp_prop``/``IODimVjpAlgorithm`` raises when a conv layer has a
    trainable bias, regardless of layout (NCH or NWC) -- the custom-VJP bwd
    rule returns the bias cotangent still shaped per-position
    ``(batch, *spatial, out_ch)`` instead of reduced to the bias's own shape
    ``(out_ch,)``.

    This is unrelated to the Task 6/7 fixes under audit: ``D_RTRL``
    (param-dim) handles conv+bias exactly (see
    ``test_d_rtrl_singlestep_matches_bptt_across_families['conv_nwc_bias']``);
    only the IO-dim path used by ``pp_prop`` is affected. Fixing it would
    require touching ``io_dim_vjp.py``, which is out of scope for this task
    (see module docstring / architecture notes: io_dim_vjp.py's core logic is
    never modified as part of this audit). This test pins the *current*
    behavior with ``xfail(strict=True)`` so that a future fix is caught (the
    xfail will start failing as an unexpected pass) rather than silently
    going unnoticed.
    """
    factory, seed = _FAMILIES['conv_nwc_bias']
    with brainstate.environ.context(precision=64):
        xs = _xs_for('conv_nwc_bias', 1, seed)
        with pytest.raises(ValueError, match='Custom VJP bwd rule'):
            online_param_gradients_singlestep_naive(
                factory, xs,
                algo_factory=lambda m: braintrace.pp_prop(m, decay_or_rank=0.9, vjp_method='single-step'),
            )
