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
F-SINGLESTEP finding encoded as a strict xfail."""

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.oracle import (
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


# --- Task 7: finding F-SINGLESTEP encoded as strict xfail --------------------

@pytest.mark.xfail(
    reason="F-SINGLESTEP: naive single-step per-step-grad summation diverges from "
           "BPTT (ETP weight ~1.65e-2 off at T=6) though multi-step is exact; "
           "accumulation recipe / single-step semantics need investigation",
    strict=True,
)
def test_singlestep_naive_matches_bptt_KNOWN_DIVERGENCE():
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(6, 3)
    g_bptt = bptt_param_gradients(spec.factory, inputs)
    g_naive = online_param_gradients_singlestep_naive(
        spec.factory, inputs,
        algo_factory=lambda m: braintrace.ParamDimVjpAlgorithm(m, vjp_method='single-step'),
    )
    # Compare only the ETP weight; this is expected to FAIL (hence xfail strict).
    assert_param_gradients_close(g_naive, g_bptt, atol=1e-4, keys=list(spec.etp_param_keys))
