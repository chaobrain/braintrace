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

"""P2 regression guard: the axis refactor must not move any preset's gradients.

The golden values in ``data/p2_golden.npz`` were captured from commit
``156d058`` — the tree *before* the P2 axis decomposition — and are compared
against the live code here. See
``docs/specs/2026-07-25-p2-axis-decomposition.md`` § Test plan.

Three properties make this guard non-vacuous, and all are asserted rather than
assumed:

- **The oracle path can see a learning-rule axis.** Golden values are captured
  through :func:`chunked_online_param_gradients` with ``chunk_size < T``. The
  full-window multi-step path returns BPTT for every algorithm at every
  hyperparameter (F-23), so golden values taken there would be identical across
  all six cases and would guard nothing.
  ``test_axis_distinctness_is_as_measured`` pins that the captured cases really
  are distinguishable on this path.
- **The model is live.** ``test_models_are_live`` pins a non-trivial BPTT
  gradient, so no comparison is being made against zeros (F-25 / lesson 4).
- **Every trace path is covered.** The engines reach the trace three different
  ways, and a golden set that only exercised the defaults would leave two of
  them free to move. :data:`PATH_CASES` adds the missing two — see the table
  there.

Comparison is **per leaf**, not over a joint norm: the parameter trees here have
one leaf carrying ~30x the norm of the other, so a joint norm would let a
complete corruption of the small leaf pass under the threshold.

Regenerate after a *deliberate* numerical change with::

    python -m braintrace._algorithm.tests.axis_golden_test
"""

from __future__ import annotations

import os

import jax
import numpy as np
import pytest

import braintrace
from braintrace._testing.oracle import (
    assert_gradients_differ,
    assert_model_is_live,
    bptt_param_gradients,
    chunked_online_param_gradients,
    flat_gradient_leaves,
    online_param_gradients_singlestep_naive,
    relative_deviation,
)
from braintrace._testing import oracle_models

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'data', 'p2_golden.npz')

T = 8

#: ``name -> (spec factory, n_in, chunk_size)``. ``chunk_size`` must stay below
#: ``T`` or the path stops seeing the trace (F-23).
MODELS = {
    'tanh_rnn': (lambda: oracle_models.tanh_rnn(n_in=3, n_rec=4, seed=0), 3, 2),
    'two_state_rnn': (lambda: oracle_models.two_state_rnn(n_in=3, n_rec=3, seed=0), 3, 3),
}

#: The five surviving presets, plus ``EProp(feedback='random')`` — the
#: random-feedback code moves between classes in P2, so it needs its own
#: frozen reference.
PRESETS = {
    'd_rtrl': lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
    'ostl_recurrent': lambda m: braintrace.OSTLRecurrent(m, vjp_method='multi-step'),
    'eprop_kappa': lambda m: braintrace.EProp(
        m, kappa_filter_decay=0.9, vjp_method='multi-step'),
    'eprop_random': lambda m: braintrace.EProp(
        m, feedback='random', random_feedback_key=jax.random.PRNGKey(7),
        vjp_method='multi-step'),
    'pp_prop': lambda m: braintrace.pp_prop(
        m, decay_or_rank=0.9, vjp_method='multi-step'),
    'ostl_feedforward': lambda m: braintrace.OSTLFeedforward(
        m, vjp_method='multi-step'),
}

#: Cases that exist for *trace-path* coverage rather than axis coverage, as
#: ``name -> (algo factory, driver)``. The engines reach the trace three ways,
#: and P2 must substitute the hidden-to-hidden Jacobian exactly once on each:
#:
#: ===================================== ================== ==================
#: path                                  reached by         covered by
#: ===================================== ================== ==================
#: multi-step, ``chunked_trace=True``    ``_update_etrace_  :data:`PRESETS`
#: (the ParamDim default)                data`` + chunk     (all six)
#:                                       factorisation
#: multi-step, ``chunked_trace=False``;  the executor's     ``d_rtrl_
#: every IO-dim coordinate               fused in-loop      unchunked``
#:                                       stepper
#: single-step                           ``_update_etrace_  ``d_rtrl_
#:                                       data``, one step   singlestep``
#: ===================================== ================== ==================
#:
#: ``d_rtrl_singlestep`` needs its own driver: the single-step VJP rejects
#: ``MultiStepData`` outright, so the chunked oracle cannot reach it.
PATH_CASES = {
    'd_rtrl_unchunked': (
        lambda m: braintrace.D_RTRL(
            m, vjp_method='multi-step', chunked_trace=False),
        'chunked',
    ),
    'd_rtrl_singlestep': (
        lambda m: braintrace.D_RTRL(m, vjp_method='single-step'),
        'singlestep',
    ),
}

CASES = [(model, preset) for model in MODELS for preset in PRESETS]
ALL_CASES = CASES + [(model, case) for model in MODELS for case in PATH_CASES]

#: ``(model, preset) -> reason`` for the pairs that provably *collapse* onto
#: ``d_rtrl`` on that model. Measured, not assumed. An axis is a property of a
#: (rule, model) pair: naming a different coordinate does not guarantee a
#: different gradient if the model gives that coordinate nothing to act on.
#: Listing these keeps the distinctness check honest in both directions — a pair
#: that starts differing is as much a regression as one that stops.
DEGENERATE = {
    ('tanh_rnn', 'ostl_feedforward'): (
        "F-29: this model's only ETP relation is the *recurrent* weight, so the "
        "IO-dim input factor is the hidden state itself; at decay 1e-6 the "
        "rank-1 product reproduces the exact per-parameter trace to round-off."
    ),
    ('two_state_rnn', 'ostl_recurrent'): (
        "the v/a coupling is hand-written arithmetic, not an ETP op, so there is "
        "no recurrent mixing primitive for `coupled` to trace into the "
        "transition; both recurrence scopes compile to the same Jacobian."
    ),
}


def _inputs(model_name: str):
    spec_fn, n_in, _ = MODELS[model_name]
    return spec_fn().make_inputs(T, n_in, seed=0)


def compute_case(model_name: str, case_name: str) -> dict:
    """Finite-window gradient of one (model, case) pair, as flat labelled leaves."""
    spec_fn, _, chunk = MODELS[model_name]
    spec = spec_fn()
    xs = _inputs(model_name)
    if case_name in PRESETS:
        algo_factory, driver = PRESETS[case_name], 'chunked'
    else:
        algo_factory, driver = PATH_CASES[case_name]
    if driver == 'chunked':
        grads = chunked_online_param_gradients(
            spec.factory, xs, algo_factory=algo_factory, chunk_size=chunk)
    else:
        grads = online_param_gradients_singlestep_naive(
            spec.factory, xs, algo_factory=algo_factory)
    return flat_gradient_leaves(grads)


def _leaf_deviations(actual: dict, golden: dict) -> dict:
    """Per-leaf ``|actual - golden| / |golden|``, keyed by leaf label.

    Reduced per leaf rather than jointly: these parameter trees are badly scaled
    against each other (the readout gradient outweighs the recurrent one by ~30x
    on ``tanh_rnn``), so a joint norm would absorb an arbitrarily large error in
    the smaller leaf.

    A leaf whose golden norm is zero is compared absolutely — a relative measure
    is undefined there, and silently skipping it would hide a leaf that gained a
    spurious gradient.
    """
    if set(actual) != set(golden):
        raise AssertionError(
            f'leaf labels changed: {sorted(set(actual) ^ set(golden))}')
    if all(not np.any(np.asarray(v)) for v in golden.values()):
        raise AssertionError(
            'the golden gradients are all zero — this comparison is vacuous. '
            'Regenerate them against a live model.')
    out = {}
    for label in golden:
        ref = np.asarray(golden[label], dtype=np.float64)
        got = np.asarray(actual[label], dtype=np.float64)
        if got.shape != ref.shape:
            raise AssertionError(
                f'{label}: shape changed {ref.shape} -> {got.shape}')
        den = float(np.sqrt((ref ** 2).sum()))
        num = float(np.sqrt(((got - ref) ** 2).sum()))
        out[label] = num / den if den > 0.0 else num
    return out


def _load_golden() -> dict:
    if not os.path.exists(GOLDEN_PATH):
        raise AssertionError(
            f'missing golden reference {GOLDEN_PATH}; regenerate with '
            '`python -m braintrace._algorithm.tests.axis_golden_test`')
    with np.load(GOLDEN_PATH) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def regenerate() -> None:
    """Recompute every case and overwrite the golden ``.npz``."""
    payload = {}
    for model_name, case_name in ALL_CASES:
        for label, arr in compute_case(model_name, case_name).items():
            payload[f'{model_name}::{case_name}::{label}'] = np.asarray(arr)
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    np.savez_compressed(GOLDEN_PATH, **payload)
    print(f'wrote {len(payload)} arrays to {GOLDEN_PATH}')


@pytest.mark.parametrize('model_name', sorted(MODELS))
def test_models_are_live(model_name):
    """A golden comparison against an all-zero gradient asserts nothing."""
    spec_fn, _, _ = MODELS[model_name]
    assert_model_is_live(spec_fn().factory, _inputs(model_name), min_norm=1e-6)


@pytest.mark.parametrize('model_name', sorted(MODELS))
def test_axis_distinctness_is_as_measured(model_name):
    """The captured path must distinguish the presets it is expected to (F-23).

    If it distinguished none of them, every golden comparison below would pass
    for the wrong reason: all six cases would hold the same numbers and the
    refactor could change any axis without being noticed. The pairs listed in
    :data:`DEGENERATE` are asserted to *stay* collapsed, for the reason recorded
    there.
    """
    grads = {name: compute_case(model_name, name) for name in PRESETS}
    for other in PRESETS:
        if other == 'd_rtrl':
            continue
        reason = DEGENERATE.get((model_name, other))
        rel = relative_deviation(grads['d_rtrl'], grads[other])
        if reason is None:
            assert_gradients_differ(grads['d_rtrl'], grads[other], min_rel=1e-6)
        else:
            # Round-off, not a difference: below float32 noise on these trees.
            assert rel < 1e-8, (
                f'{model_name}/{other} was expected to collapse onto d_rtrl '
                f'({reason}) but deviates by {rel:.3e}.')


def test_every_axis_is_live_somewhere():
    """Each preset must be distinguishable from ``d_rtrl`` on at least one model.

    :data:`DEGENERATE` records per-model collapses; this pins that no preset is
    degenerate on *every* model in the set, which would leave its axis entirely
    unguarded by this module.
    """
    for preset in PRESETS:
        if preset == 'd_rtrl':
            continue
        live = [m for m in MODELS if (m, preset) not in DEGENERATE]
        assert live, (
            f'{preset} collapses onto d_rtrl on every model in the golden set, '
            'so its axis is not guarded here. Add a model that separates it.')


@pytest.mark.parametrize('model_name,case_name', ALL_CASES)
def test_matches_golden(model_name, case_name):
    """Per-leaf equality with the pre-refactor reference."""
    golden_all = _load_golden()
    prefix = f'{model_name}::{case_name}::'
    golden = {k[len(prefix):]: v for k, v in golden_all.items()
              if k.startswith(prefix)}
    assert golden, (
        f'no golden entries for {prefix}; regenerate with '
        '`python -m braintrace._algorithm.tests.axis_golden_test`')
    deviations = _leaf_deviations(compute_case(model_name, case_name), golden)
    # 1e-6 sits two orders above float32 round-off on these trees (~1e-8) and
    # two-to-five orders below the deviations the presets show from each other
    # (see test_axis_distinctness_is_as_measured), so it separates "unchanged"
    # from "changed" without tripping on reassociation (lesson 5).
    moved = {k: v for k, v in deviations.items() if v > 1e-6}
    assert not moved, (
        f'{model_name}/{case_name}: gradients moved relative to the pre-P2 '
        f'reference on ' + ', '.join(f'{k} ({v:.3e})' for k, v in moved.items())
        + '. The axis refactor must be numerically inert.')


@pytest.mark.parametrize('model_name', sorted(MODELS))
def test_chunked_trace_is_an_execution_option_not_an_axis(model_name):
    """``chunked_trace`` picks a code path; it must not pick a different answer.

    Measured bitwise-identical on both models before the refactor. This is the
    only guard that the executor's fused in-loop stepper and the chunk-factorised
    ``_update_etrace_data`` roll agree, so P2's Jacobian substitution — which
    has to be installed separately on each — cannot drift them apart unnoticed.
    """
    chunked = compute_case(model_name, 'd_rtrl')
    fused = compute_case(model_name, 'd_rtrl_unchunked')
    for label, deviation in _leaf_deviations(fused, chunked).items():
        # Both models measure exactly 0.0 here, but the guarantee is agreement
        # to round-off, not bitwise identity: the two paths sum the same terms
        # in different orders, and `leaky_linear` (not in this set) already
        # shows 3.3e-08 from that reassociation alone.
        assert deviation < 1e-6, (
            f'{model_name}/{label}: chunked_trace=False deviates from the '
            f'default by {deviation:.3e}, far above reassociation noise. '
            'It selects a code path, not a different answer.')


def test_bptt_reference_is_reproducible():
    """Guards the harness itself: the model factories must be deterministic.

    A factory drawing from the unseeded global RNG returns a different network
    per call, which silently turns every comparison in this module into noise
    (lesson 4).
    """
    for model_name in MODELS:
        spec_fn, _, _ = MODELS[model_name]
        xs = _inputs(model_name)
        a = flat_gradient_leaves(bptt_param_gradients(spec_fn().factory, xs))
        b = flat_gradient_leaves(bptt_param_gradients(spec_fn().factory, xs))
        for label in a:
            np.testing.assert_array_equal(
                np.asarray(a[label]), np.asarray(b[label]),
                err_msg=f'{model_name}/{label} is not reproducible')


if __name__ == '__main__':
    regenerate()
