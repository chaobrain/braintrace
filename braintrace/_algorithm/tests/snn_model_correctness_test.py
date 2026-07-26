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

"""Gradient correctness on realistic SNN models: multi-timescale synapses,
per-neuron heterogeneous leaks, multi-state HiddenGroups, and E/I populations.

These are the claims ``AGENTS.md`` carried only as prose ("heterogeneous-
population leak resolution", "multi-state HiddenGroups"). They are discharged
here as passing tests rather than as fixes -- the compiler already handles them.

Assertion paths, per F-23: an *exact* algorithm versus BPTT uses the full-window
multi-step path, whose subject is the compiler and the ETP per-primitive rules
and which is the right instrument for that. Anything comparing *across*
algorithms uses a finite window.
"""

import brainstate
import brainunit as u
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_gradients_differ,
    assert_model_is_live,
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
    online_param_gradients,
)
from braintrace._algorithm.oracle_models import SNN_SPECS

T = 6
N_IN = 4
N_REC = 5
ATOL = 1e-4

_ALL_SPECS = sorted(SNN_SPECS)


def _setup(name):
    spec = SNN_SPECS[name]()
    xs = spec.make_inputs(T, N_IN)
    return spec, xs


@pytest.mark.parametrize('name', _ALL_SPECS)
def test_d_rtrl_matches_bptt_on_snn_models(name):
    """D_RTRL is exact, so it must reproduce BPTT on every realistic model:
    multi-timescale synapses, heterogeneous leaks, E/I populations and
    HiddenGroups with num_state from 1 to 5."""
    spec, xs = _setup(name)
    with brainstate.environ.context(dt=0.1 * u.ms):
        assert_model_is_live(spec.factory, xs, min_norm=1e-6)
        g_bptt = bptt_param_gradients(spec.factory, xs)
        g_online = online_param_gradients(
            spec.factory, xs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        assert_param_gradients_close(g_online, g_bptt, atol=ATOL)


@pytest.mark.parametrize('name', ['lif_expcu_heterogeneous',
                                  'alif_expcu_heterogeneous'])
def test_heterogeneous_leaks_do_not_break_exactness(name):
    """A per-neuron time constant leaves no single global leak for the
    transition to factor out. The compiler takes a true Jacobian, so exactness
    survives -- this is what retires the AGENTS.md prose item."""
    spec, xs = _setup(name)
    with brainstate.environ.context(dt=0.1 * u.ms):
        assert_model_is_live(spec.factory, xs, min_norm=1e-6)
        g_bptt = bptt_param_gradients(spec.factory, xs)
        g_online = online_param_gradients(
            spec.factory, xs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        assert_param_gradients_close(g_online, g_bptt, atol=ATOL)


@pytest.mark.parametrize('name,expected_min_state', [
    ('if_delta', 1),
    ('lif_expcu', 2),
    ('alif_expcu', 3),
    ('alif_expco_ei', 5),
])
def test_multi_state_hidden_groups_are_discovered(name, expected_min_state):
    """Pins the structural facts the exactness tests above rest on: these models
    really do form multi-state HiddenGroups, so the per-state axis is exercised
    rather than assumed."""
    spec, xs = _setup(name)
    with brainstate.environ.context(dt=0.1 * u.ms):
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='multi-step')
        algo.compile_graph(xs[0])
        assert len(algo.graph.hidden_groups) >= 1
        assert max(hg.num_state for hg in algo.graph.hidden_groups) >= expected_min_state
        assert len(algo.graph.hidden_param_op_relations) >= 1


def test_ei_population_split_yields_multiple_relations():
    """The E/I model routes separate excitatory and inhibitory projections into
    one hidden group, so the compiler must record more than one ETP relation."""
    spec, xs = _setup('alif_expco_ei')
    with brainstate.environ.context(dt=0.1 * u.ms):
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='multi-step')
        algo.compile_graph(xs[0])
        assert len(algo.graph.hidden_param_op_relations) >= 2


@pytest.mark.parametrize('name', ['lif_expcu', 'alif_expco_ei'])
def test_approximation_is_measurable_on_snn_models(name):
    """The genuinely approximate configuration must be *distinguishable* from
    the exact one on a realistic model -- via a finite window, which is the only
    path that can see it (F-23). This is what F-22 was really asking for."""
    spec, xs = _setup(name)
    with brainstate.environ.context(dt=0.1 * u.ms):
        g_exact = chunked_online_param_gradients(
            spec.factory, xs, chunk_size=2,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        g_approx = chunked_online_param_gradients(
            spec.factory, xs, chunk_size=2,
            algo_factory=lambda m: braintrace.pp_prop(
                m, decay_or_rank=0.5, vjp_method='multi-step'))
        assert_gradients_differ(g_exact, g_approx, min_rel=1e-6)
