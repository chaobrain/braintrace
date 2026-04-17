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

"""Property tests for the ETrace compiler.

Hypothesis-driven tests that pin down the *invariants* the compiler must
satisfy across a wide parameter space, rather than enumerating examples.

Properties under test
---------------------

* **Idempotence** — Compiling the same model twice produces relations in
  exactly the same order, with exactly the same path classifications and
  diagnostic kinds.
* **Stacked layer scoping** — A depth-``k`` ``StackedDeepRNN`` must
  produce exactly ``k`` relations, each scoped to its own layer's hidden
  state. No cross-layer leakage.
* **Tied-weight call-site multiplicity** — A weight tied across ``k``
  ETP call sites that all reach the home hidden state must produce
  exactly ``k`` relations, all pointing at the same weight path.
* **W -> W -> h exclusion (chain depth)** — In a chain
  ``mid_k = matmul(mid_{k-1}, w_k)``, only the *last* weight may
  register; every earlier weight must emit
  ``RELATION_EXCLUDED_WEIGHT_TO_WEIGHT``.
* **PartialPath classification** — Across random shape choices,
  ``PartialPathRNN``'s ``w1`` is always ``MIXED`` and ``w2`` is always
  ``ALL_DIRECT``.
"""

from __future__ import annotations

import warnings

import brainstate
import jax.numpy as jnp
import pytest

from hypothesis import HealthCheck, given, settings, strategies as st

import braintrace
from braintrace import (
    DiagnosticKind,
    compile_etrace_graph,
)
from braintrace._etrace_compiler.hid_param_op import PathClassification
from braintrace._etrace_compiler.scenario_catalog import (
    PartialPathRNN,
    SharedTiedWeightRNN,
    StackedDeepRNN,
    UnbatchedMvRNN,
)


_HYPOTHESIS_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _silent_compile(model, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return compile_etrace_graph(model, *args, include_hidden_perturb=False)


def _summary(graph):
    """Order-preserving fingerprint of a compilation result.

    Captures everything an oracle can compare across compiles: relation
    weight paths, hidden paths, primitive identity, path classification,
    and diagnostic kinds in emit order.
    """
    return (
        tuple(
            (
                r.weight_path,
                tuple(r.connected_hidden_paths),
                r.primitive,
                tuple(sorted(r.path_classification.items())),
            )
            for r in graph.hidden_param_op_relations
        ),
        tuple(d.kind for d in graph.diagnostics),
    )


# ---------------------------------------------------------------------------
# Property 1 — Idempotence
# ---------------------------------------------------------------------------

class TestIdempotence:
    """Compiling a freshly-built model twice with the same inputs yields
    identical fingerprints. This is the strongest determinism statement
    we can make without inspecting the compiler's internal state."""

    @given(
        n_in=st.integers(min_value=1, max_value=8),
        n_out=st.integers(min_value=1, max_value=8),
    )
    @_HYPOTHESIS_SETTINGS
    def test_unbatched_mv_rnn_is_idempotent(self, n_in, n_out):
        inp = jnp.zeros(n_in)

        m1 = UnbatchedMvRNN(n_in, n_out)
        brainstate.nn.init_all_states(m1)
        m2 = UnbatchedMvRNN(n_in, n_out)
        brainstate.nn.init_all_states(m2)

        s1 = _summary(_silent_compile(m1, inp))
        s2 = _summary(_silent_compile(m2, inp))
        assert s1 == s2

    @given(
        n=st.integers(min_value=1, max_value=6),
    )
    @_HYPOTHESIS_SETTINGS
    def test_partial_path_rnn_is_idempotent(self, n):
        inp = jnp.zeros(n)

        m1 = PartialPathRNN(n, n)
        brainstate.nn.init_all_states(m1)
        m2 = PartialPathRNN(n, n)
        brainstate.nn.init_all_states(m2)

        s1 = _summary(_silent_compile(m1, inp))
        s2 = _summary(_silent_compile(m2, inp))
        assert s1 == s2


# ---------------------------------------------------------------------------
# Property 2 — Stacked layer scoping
# ---------------------------------------------------------------------------

class TestStackedScoping:
    """A depth-``k`` stacked RNN must produce exactly ``k`` relations,
    each connected to exactly one hidden state — its own layer's. This
    is the home-group restriction in action."""

    @given(
        depth=st.integers(min_value=1, max_value=5),
        n_in=st.integers(min_value=1, max_value=4),
        n_out=st.integers(min_value=1, max_value=4),
    )
    @_HYPOTHESIS_SETTINGS
    def test_one_relation_per_layer_scoped_to_own_h(self, depth, n_in, n_out):
        model = StackedDeepRNN(n_in, n_out, depth=depth)
        brainstate.nn.init_all_states(model)
        inp = jnp.zeros(n_in)

        graph = _silent_compile(model, inp)
        rels = graph.hidden_param_op_relations

        assert len(rels) == depth
        for i, r in enumerate(rels):
            assert r.connected_hidden_paths == [(f'cell{i}', 'h')], (
                f'Layer {i} weight must reach only its own hidden state; '
                f'got {r.connected_hidden_paths}'
            )


# ---------------------------------------------------------------------------
# Property 3 — Tied-weight call-site multiplicity
# ---------------------------------------------------------------------------

class TestTiedWeightMultiplicity:
    """``SharedTiedWeightRNN`` registers two relations for the same
    ParamState because there are two call sites. The property generalises
    to any ``k``: the compiler is per-call-site, not per-ParamState."""

    @given(n=st.integers(min_value=1, max_value=8))
    @_HYPOTHESIS_SETTINGS
    def test_two_call_sites_yield_two_relations(self, n):
        model = SharedTiedWeightRNN(n, n)
        brainstate.nn.init_all_states(model)
        inp = jnp.zeros(n)

        graph = _silent_compile(model, inp)
        rels = graph.hidden_param_op_relations

        assert len(rels) == 2
        for r in rels:
            assert r.weight_path == ('w',)
            assert r.connected_hidden_paths == [('h',)]


# ---------------------------------------------------------------------------
# Property 4 — W -> W -> h exclusion across chain depth
# ---------------------------------------------------------------------------

def _make_chain_rnn(chain_len: int, n: int) -> brainstate.nn.Module:
    """Return a model with ``chain_len`` matmuls in series feeding ``h``.

    ``mid_0 = matmul(xh, w_1); mid_1 = matmul(mid_0, w_2); ...
    h = tanh(mid_{chain_len-1})``.

    Only ``w_{chain_len}`` (the final matmul) must be included in
    relations; all earlier weights are W -> W -> h excluded.
    """
    class _Chain(brainstate.nn.Module):
        def __init__(self_inner):
            super().__init__()
            # First weight takes concat(x, h_prev) of length 2n.
            shape = (2 * n, n)
            self_inner._chain_len = chain_len
            for i in range(chain_len):
                setattr(
                    self_inner, f'w{i}',
                    brainstate.ParamState(
                        brainstate.random.randn(*shape) * 0.1
                    ),
                )
                shape = (n, n)
            self_inner.h = brainstate.HiddenState(jnp.zeros(n))

        def init_state(self_inner, *a, **k):
            self_inner.h.value = jnp.zeros_like(self_inner.h.value)

        def update(self_inner, x):
            mid = jnp.concatenate([x, self_inner.h.value])
            for i in range(self_inner._chain_len):
                ps = getattr(self_inner, f'w{i}')
                mid = braintrace.matmul(mid, ps.value)
            self_inner.h.value = jnp.tanh(mid)
            return self_inner.h.value

    return _Chain()


class TestChainExclusion:
    """For a chain of ``k`` matmuls, exactly the last weight (``w_{k-1}``)
    is included; the other ``k-1`` weights produce
    ``RELATION_EXCLUDED_WEIGHT_TO_WEIGHT`` records."""

    @given(
        chain_len=st.integers(min_value=1, max_value=5),
        n=st.integers(min_value=1, max_value=4),
    )
    @_HYPOTHESIS_SETTINGS
    def test_only_last_weight_registers(self, chain_len, n):
        model = _make_chain_rnn(chain_len, n)
        brainstate.nn.init_all_states(model)
        inp = jnp.zeros(n)

        graph = _silent_compile(model, inp)
        rels = graph.hidden_param_op_relations
        included_paths = {r.weight_path for r in rels}
        last = (f'w{chain_len - 1}',)

        assert included_paths == {last}, (
            f'For chain length {chain_len} only the last weight should be '
            f'included; got {included_paths}'
        )

        excluded = graph.explain(
            kind=DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT,
        )
        assert len(excluded) == chain_len - 1, (
            f'Expected {chain_len - 1} W->W->h exclusions; got {len(excluded)}'
        )


# ---------------------------------------------------------------------------
# Property 5 — Partial-path classification stability
# ---------------------------------------------------------------------------

class TestPartialPathStability:
    """Across random shape choices, ``PartialPathRNN`` always classifies
    ``w1`` as ``MIXED`` and ``w2`` as ``ALL_DIRECT``."""

    @given(
        n_in=st.integers(min_value=1, max_value=4),
        n_out=st.integers(min_value=1, max_value=4),
    )
    @_HYPOTHESIS_SETTINGS
    def test_classification_is_shape_invariant(self, n_in, n_out):
        model = PartialPathRNN(n_in, n_out)
        brainstate.nn.init_all_states(model)
        inp = jnp.zeros(n_in)

        graph = _silent_compile(model, inp)
        by_path = {r.weight_path: r for r in graph.hidden_param_op_relations}

        assert by_path[('w1',)].path_classification == {
            ('h',): PathClassification.MIXED,
        }
        assert by_path[('w2',)].path_classification == {
            ('h',): PathClassification.ALL_DIRECT,
        }
