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

"""F-23: which oracle paths can see a learning-rule axis, and which cannot.

A full-sequence ``MultiStepData`` call makes the within-call gradient exact
reverse-mode, so the eligibility trace enters only at a sequence boundary that
does not exist. Every algorithm then returns BPTT. This module pins that in both
directions, so a future test cannot silently assert an approximation's behaviour
on a path that cannot observe it -- which is how F-21 came to attribute the
effect to the model instead of the harness.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._testing.oracle import (
    assert_gradients_differ,
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
    online_param_gradients,
    relative_deviation,
)
from braintrace._testing.oracle_models import tanh_rnn

T = 8
CHUNK = 2


def _spec():
    return tanh_rnn(n_in=3, n_rec=4, seed=0)


def _inputs():
    return jnp.asarray(np.random.RandomState(0).randn(T, 3).astype('float32'))


# Pairs of configurations that differ ONLY in a learning-rule axis value.
AXIS_PAIRS = {
    # Axis 1/2: IO-dim trace factorization strength (decay).
    'pp_prop decay': (
        lambda m: braintrace.pp_prop(m, decay_or_rank=0.99, vjp_method='multi-step'),
        lambda m: braintrace.pp_prop(m, decay_or_rank=0.01, vjp_method='multi-step'),
    ),
    # Axis 5: trace filter (kappa).
    'EProp kappa': (
        lambda m: braintrace.EProp(m, kappa_filter_decay=0.0, vjp_method='multi-step'),
        lambda m: braintrace.EProp(m, kappa_filter_decay=0.95, vjp_method='multi-step'),
    ),
    # Axis 3: recurrence scope (diagonal vs coupled).
    'recurrence scope': (
        lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        lambda m: braintrace.OSTLRecurrent(m, vjp_method='multi-step'),
    ),
}


@pytest.mark.parametrize('axis', sorted(AXIS_PAIRS))
def test_full_window_multistep_cannot_see_any_axis(axis):
    """The full-window path collapses axis-distinct configs to identical
    gradients. Pinned so the semantics are understood rather than accidental --
    if this ever starts failing, the window semantics changed and every
    assertion written against this path needs review."""
    spec, xs = _spec(), _inputs()
    lo, hi = AXIS_PAIRS[axis]
    g_lo = online_param_gradients(spec.factory, xs, algo_factory=lo)
    g_hi = online_param_gradients(spec.factory, xs, algo_factory=hi)
    rel = relative_deviation(g_lo, g_hi)
    assert rel == 0.0, (
        f'{axis}: full-window multi-step distinguished two configurations '
        f'(rel={rel:.3e}); F-23 assumed it cannot. Re-check the oracle docs.'
    )


@pytest.mark.parametrize('axis', sorted(AXIS_PAIRS))
def test_finite_window_does_see_the_axis(axis):
    """A chunked window makes the trace enter at every chunk boundary, so the
    axis becomes observable. This is the path axis assertions must use."""
    spec, xs = _spec(), _inputs()
    lo, hi = AXIS_PAIRS[axis]
    g_lo = chunked_online_param_gradients(
        spec.factory, xs, algo_factory=lo, chunk_size=CHUNK)
    g_hi = chunked_online_param_gradients(
        spec.factory, xs, algo_factory=hi, chunk_size=CHUNK)
    assert_gradients_differ(g_lo, g_hi, min_rel=1e-6)


def test_full_window_still_reproduces_bptt_for_an_exact_algorithm():
    """The corollary that makes the full-window path useful: it is the right
    instrument for asserting an exact algorithm matches BPTT."""
    spec, xs = _spec(), _inputs()
    g_bptt = bptt_param_gradients(spec.factory, xs)
    g_online = online_param_gradients(
        spec.factory, xs,
        algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    assert_param_gradients_close(g_online, g_bptt, atol=1e-4)
