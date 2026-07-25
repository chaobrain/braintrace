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

"""The SNN specs must be deterministic (F-24) and live (F-25) before any
gradient assertion built on them means anything."""

import brainstate
import brainunit as u
import jax.numpy as jnp
import pytest

from braintrace._algorithm.oracle import (
    assert_model_is_live,
    bptt_param_gradients,
    flat_gradient_leaves,
    gradient_norm,
)
from braintrace._algorithm.oracle_models import SNN_SPECS


@pytest.mark.parametrize('name', sorted(SNN_SPECS))
def test_snn_spec_construction_is_deterministic(name):
    """F-24: the underlying layer classes seed from the global RNG, so two
    factory() calls must be pinned to produce identical weights."""
    spec = SNN_SPECS[name]()
    with brainstate.environ.context(dt=0.1 * u.ms):
        w1 = flat_gradient_leaves(
            {k: v.value for k, v in spec.factory().states(brainstate.ParamState).items()})
        w2 = flat_gradient_leaves(
            {k: v.value for k, v in spec.factory().states(brainstate.ParamState).items()})
    assert set(w1) == set(w2)
    for key in w1:
        assert bool(jnp.allclose(w1[key], w2[key])), f'{name}: {key} differs across calls'


@pytest.mark.parametrize('name', sorted(SNN_SPECS))
def test_snn_spec_is_live(name):
    """F-25: at the default input scale these networks never spike, so their
    gradients are identically zero and every comparison is vacuous. Each spec
    records a scale that produces a non-trivial gradient."""
    spec = SNN_SPECS[name]()
    with brainstate.environ.context(dt=0.1 * u.ms):
        xs = spec.make_inputs(6, 4)
        norm = assert_model_is_live(spec.factory, xs, min_norm=1e-6)
    assert norm > 1e-6


def test_underdriven_input_scale_is_dead():
    """The counterpart of the above: pins *why* the scale field exists. At
    scale 1.0 a conductance-based model never reaches threshold and its gradient
    is exactly zero, so any comparison on it would be vacuous."""
    spec = SNN_SPECS['lif_expcu']()
    with brainstate.environ.context(dt=0.1 * u.ms):
        dead_xs = spec.make_inputs(6, 4) / spec.input_scale  # undo the scaling
        assert gradient_norm(bptt_param_gradients(spec.factory, dead_xs)) == 0.0


def test_overdriven_input_scale_is_also_dead_while_still_spiking():
    """The live window is bounded *above* as well as below, and this is the half
    that is easy to miss.

    Driven hard, ``ALIF_Delta`` keeps spiking (rate 0.60) but the surrogate
    derivative saturates and the BPTT gradient returns to exactly zero. A
    liveness check keyed on spike rate would pass here and the comparison would
    still assert nothing -- which is why ``assert_model_is_live`` keys on the
    gradient norm instead. See F-25.
    """
    spec = SNN_SPECS['alif_delta']()
    with brainstate.environ.context(dt=0.1 * u.ms):
        live_xs = spec.make_inputs(6, 4)
        over_xs = live_xs * 20.0

        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        outs = brainstate.transform.for_loop(lambda x: model(x), over_xs)
        spike_rate = float(jnp.mean(jnp.asarray(u.get_mantissa(outs)) > 0.0))

        assert gradient_norm(bptt_param_gradients(spec.factory, live_xs)) > 1e-6
        assert spike_rate > 0.1, 'the over-driven network must still be spiking'
        assert gradient_norm(bptt_param_gradients(spec.factory, over_xs)) == 0.0
