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

"""Diagnostic layer (test-strategy design sections 8.2 + 11): hypothesis random
exploration.

The curated correctness matrices (oracle_test / exact_correctness_test /
transform_correctness_test) pin behavior at *fixed* dimensions. This module
broadens hidden_size / seq_len / layers / batch into a generated range and
asserts the structural invariants we already trust at fixed dims:

  * exact-class ``D_RTRL`` via the multi-step VJP path equals BPTT element-wise,
    for a single-relation rate RNN, a 2-layer stacked RNN (two relations), and a
    ``num_state == 2`` coupled-state group;
  * batch invariance: a batched multi-step gradient equals the sum of the
    per-sequence single-batch gradients.

Every test carries ``@pytest.mark.diagnostic``. That marker is deselected by the
default ``addopts`` (``-m 'not diagnostic'``) so this exploration never gates CI
red/green; run it on demand with ``pytest braintrace/ -m diagnostic``. Failing
examples auto-shrink and their seed is recorded in the gitignored ``.hypothesis/``
database for replay. Verified bit-exact (batch within float32 round-off) on
2026-05-27; see ``temp/spike_p8.py``.
"""

import importlib.util
import warnings

import pytest

import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

import brainstate
import braintrace
from braintrace._etrace_algorithms.oracle import (
    assert_param_gradients_close,
    bptt_param_gradients,
    online_param_gradients,
)
from braintrace._etrace_algorithms.oracle_models import (
    batched_tanh_rnn,
    stacked_tanh_rnn,
    tanh_rnn,
    two_state_rnn,
)

pytestmark = pytest.mark.diagnostic

# Exact comparisons are bit-exact; batch invariance is float32 round-off
# (worst 4.3e-6 in the spike). 1e-4 leaves head-room without hiding a real bug.
ATOL = 1e-4

_EXACT_SETTINGS = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
_BATCH_SETTINGS = settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _drtrl(model):
    return braintrace.D_RTRL(model, vjp_method='multi-step')


def _assert_exact_equals_bptt(spec, inputs):
    """D_RTRL multi-step gradient == BPTT gradient for every ParamState."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = bptt_param_gradients(spec.factory, inputs)
        actual = online_param_gradients(spec.factory, inputs, algo_factory=_drtrl)
    assert_param_gradients_close(actual, expected, atol=ATOL)


@given(
    n_in=st.integers(min_value=1, max_value=5),
    n_rec=st.integers(min_value=2, max_value=9),
    seq_len=st.integers(min_value=2, max_value=15),
    seed=st.integers(min_value=0, max_value=10_000),
)
@_EXACT_SETTINGS
def test_exact_single_relation_matches_bptt_over_dims(n_in, n_rec, seq_len, seed):
    """Single-relation tanh RNN: exact online gradient == BPTT across generated
    hidden_size / seq_len."""
    spec = tanh_rnn(n_in=n_in, n_rec=n_rec, seed=seed)
    inputs = jnp.asarray(
        np.random.RandomState(seed).randn(seq_len, n_in).astype('float32'))
    _assert_exact_equals_bptt(spec, inputs)


@given(
    n_in=st.integers(min_value=1, max_value=5),
    n_rec=st.integers(min_value=2, max_value=9),
    seq_len=st.integers(min_value=2, max_value=15),
    seed=st.integers(min_value=0, max_value=10_000),
)
@_EXACT_SETTINGS
def test_exact_stacked_two_layer_matches_bptt_over_dims(n_in, n_rec, seq_len, seed):
    """Two-layer stacked RNN (two ETP relations): exact online gradient == BPTT
    across generated dims. Broadens the 'layers' axis to depth 2."""
    spec = stacked_tanh_rnn(n_in=n_in, n_rec=n_rec, seed=seed)
    inputs = jnp.asarray(
        np.random.RandomState(seed).randn(seq_len, n_in).astype('float32'))
    _assert_exact_equals_bptt(spec, inputs)


@given(
    n_in=st.integers(min_value=1, max_value=5),
    n_rec=st.integers(min_value=2, max_value=9),
    seq_len=st.integers(min_value=2, max_value=15),
    seed=st.integers(min_value=0, max_value=10_000),
)
@_EXACT_SETTINGS
def test_exact_two_state_group_matches_bptt_over_dims(n_in, n_rec, seq_len, seed):
    """Coupled (v, a) group with num_state == 2: exact online gradient == BPTT
    across generated dims. Broadens the SNN-like multi-state axis."""
    spec = two_state_rnn(n_in=n_in, n_rec=n_rec, seed=seed)
    inputs = jnp.asarray(
        np.random.RandomState(seed).randn(seq_len, n_in).astype('float32'))
    _assert_exact_equals_bptt(spec, inputs)


def _batched_multistep_grad(n_in, n_rec, batch, seq, seed):
    spec = batched_tanh_rnn(n_in=n_in, n_rec=n_rec, batch=batch, seed=seed)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=batch)
    algo = braintrace.D_RTRL(model, vjp_method='multi-step')
    algo.compile_graph(seq[0])
    algo.init_etrace_state()
    params = model.states(brainstate.ParamState)
    return brainstate.transform.grad(
        lambda s: (algo(braintrace.MultiStepData(s)) ** 2).sum(), params)(seq)


@given(
    n_in=st.integers(min_value=1, max_value=4),
    n_rec=st.integers(min_value=2, max_value=7),
    batch=st.integers(min_value=2, max_value=5),
    seq_len=st.integers(min_value=2, max_value=9),
    seed=st.integers(min_value=0, max_value=10_000),
)
@_BATCH_SETTINGS
def test_batch_invariance_over_dims(n_in, n_rec, batch, seq_len, seed):
    """A batched multi-step gradient equals the sum of per-sequence single-batch
    gradients, across generated batch / hidden_size / seq_len. Linearity of the
    summed per-step SSE loss over the batch axis."""
    seq = jnp.asarray(
        np.random.RandomState(seed).randn(seq_len, batch, n_in).astype('float32'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        batched = _batched_multistep_grad(n_in, n_rec, batch, seq, seed)
        summed = None
        for b in range(batch):
            sub = seq[:, b:b + 1, :]
            g = _batched_multistep_grad(n_in, n_rec, 1, sub, seed)
            summed = g if summed is None else {k: summed[k] + g[k] for k in g}
    assert_param_gradients_close(batched, summed, atol=ATOL)
