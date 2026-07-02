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

"""Tests that the compiler emits structured :class:`CompilationRecord` entries
for every decision it makes, and that ``ETraceGraph.explain()`` lets tests
assert on those records rather than parsing warning strings.

These tests encode the three core principles:

    1. Primitive-type identity (not name matching)
    2. Only gradient-enabled ETP primitives may be traversed on the tail
       from ``y`` to ``h``
    3. ``W -> non-gradient-enabled W -> h`` excludes the preceding weight

Each principle maps to a specific :class:`DiagnosticKind` that must appear in
``graph.explain(...)`` under the corresponding scenario.
"""

import brainstate
import pytest

import braintrace
from braintrace import (
    CompilationRecord,
    DiagnosticKind,
    DiagnosticLevel,
    compile_etrace_graph,
)
from braintrace._op import etp_mm_p, etp_mv_p


class TestDiagnosticsBasics:

    def test_phase3_control_flow_kinds_exist(self):
        """The Phase 3 while/opaque-forward decisions each have a dedicated
        machine-readable kind, so tests and users can assert on
        ``CompilationRecord.kind`` instead of parsing messages."""
        assert DiagnosticKind.WEIGHT_IN_WHILE.value == 'weight_in_while'
        assert (DiagnosticKind.CONTROL_FLOW_OPAQUE_FWD.value
                == 'control_flow_opaque_fwd')
        assert (DiagnosticKind.CONTROL_FLOW_RECURRENT_MIXING.value
                == 'control_flow_recurrent_mixing')

    def test_graph_has_diagnostics_field(self):
        """``ETraceGraph`` carries a ``diagnostics`` field populated by the
        compiler. Each entry is a :class:`CompilationRecord`."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inp = brainstate.random.rand(3)

        with pytest.warns(UserWarning):
            graph = compile_etrace_graph(gru, inp, include_hidden_perturb=False)

        assert isinstance(graph.diagnostics, tuple)
        assert len(graph.diagnostics) > 0
        for record in graph.diagnostics:
            assert isinstance(record, CompilationRecord)
            assert isinstance(record.kind, DiagnosticKind)
            assert isinstance(record.level, DiagnosticLevel)
            assert isinstance(record.message, str) and record.message

    def test_explain_filters_by_kind(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inp = brainstate.random.rand(3)

        with pytest.warns(UserWarning):
            graph = compile_etrace_graph(gru, inp, include_hidden_perturb=False)

        included = graph.explain(kind=DiagnosticKind.RELATION_INCLUDED)
        assert len(included) == len(graph.hidden_param_op_relations) == 2
        for record in included:
            assert record.kind is DiagnosticKind.RELATION_INCLUDED
            assert record.level is DiagnosticLevel.INFO


class TestPrincipleWeightToWeightExclusion:
    """Principle 3: ``Wr`` in a GRU must emit
    :attr:`DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT` — NOT
    :attr:`RELATION_EXCLUDED_NON_TEMPORAL`. The distinction matters because
    the two have different remediation: the former requires architectural
    change, the latter means the weight is genuinely not part of the
    recurrence."""

    def test_gru_wr_emits_weight_to_weight_record(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inp = brainstate.random.rand(3)

        with pytest.warns(UserWarning, match='trainable ETP primitive'):
            graph = compile_etrace_graph(gru, inp, include_hidden_perturb=False)

        w2w = graph.explain(kind=DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT)
        assert len(w2w) == 1, (
            f'Expected exactly one W->W->h exclusion (GRU Wr) but got {len(w2w)}: '
            f'{[r.message for r in w2w]}'
        )
        record = w2w[0]
        assert record.level is DiagnosticLevel.WARNING
        # Blocking primitive should be etp_mv_p (Wh's matmul, unbatched here).
        blocking = record.context['blocking_primitives']
        assert etp_mv_p in blocking or etp_mm_p in blocking

    def test_gru_emits_no_non_temporal_record(self):
        """All of GRU's three weights either are included or are blocked by
        W->W->h. None is truly non-temporal, so no ``NON_TEMPORAL`` record
        should appear."""
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inp = brainstate.random.rand(3)

        with pytest.warns(UserWarning):
            graph = compile_etrace_graph(gru, inp, include_hidden_perturb=False)

        non_temporal = graph.explain(
            kind=DiagnosticKind.RELATION_EXCLUDED_NON_TEMPORAL,
        )
        assert len(non_temporal) == 0, (
            f'GRU should not emit non-temporal records; got {non_temporal}'
        )


class TestPrincipleTypeIdentityDispatch:
    """Principle 1: Records carry the actual :class:`ETPPrimitive` instance,
    not a name string. Tests can compare with ``is`` (type identity)."""

    def test_record_primitive_is_etppr(self):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inp = brainstate.random.rand(3)

        with pytest.warns(UserWarning):
            graph = compile_etrace_graph(gru, inp, include_hidden_perturb=False)

        included = graph.explain(kind=DiagnosticKind.RELATION_INCLUDED)
        # GRUCell input is 1-D, so each Linear uses etp_mv_p (unbatched).
        for record in included:
            assert record.primitive is etp_mv_p, (
                f'Expected primitive identity etp_mv_p, got {record.primitive}'
            )


class TestTrainableInvarNotParamState:
    """When a trainable invar (e.g. a constant bias) does not trace back to
    any ParamState, the compiler must emit
    :attr:`DiagnosticKind.TRAINABLE_INVAR_NOT_PARAMSTATE` at INFO level.

    The relation itself is still registered (the weight ParamState is still
    found), but the bias key is silently dropped from ``trainable_vars`` /
    ``trainable_paths`` and an INFO diagnostic is emitted so users know
    that wrapping the bias in a ParamState is required to train it.
    """

    def test_constant_bias_emits_diagnostic(self):
        import jax.numpy as jnp

        bias_const = jnp.ones((4,))  # NOT wrapped in a ParamState

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)))
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value, bias_const)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        graph = compile_etrace_graph(
            cell, jnp.zeros((1, 4)), include_hidden_perturb=False
        )

        kinds = [d.kind for d in graph.diagnostics]
        assert DiagnosticKind.TRAINABLE_INVAR_NOT_PARAMSTATE in kinds, (
            f'Expected TRAINABLE_INVAR_NOT_PARAMSTATE in diagnostics; got {kinds}'
        )

    def test_constant_bias_diagnostic_names_key(self):
        """The diagnostic message includes the key name ('bias')."""
        import jax.numpy as jnp

        bias_const = jnp.ones((4,))

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)))
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value, bias_const)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        graph = compile_etrace_graph(
            cell, jnp.zeros((1, 4)), include_hidden_perturb=False
        )

        records = graph.explain(kind=DiagnosticKind.TRAINABLE_INVAR_NOT_PARAMSTATE)
        assert len(records) >= 1
        record = records[0]
        assert record.level is DiagnosticLevel.INFO
        assert record.context is not None
        assert record.context.get('key') == 'bias'

    def test_paramstate_bias_does_not_emit_diagnostic(self):
        """When bias IS a ParamState, the diagnostic must NOT be emitted."""
        import jax.numpy as jnp

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)))
                self.b = brainstate.ParamState(jnp.zeros((4,)))
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value, self.b.value)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        graph = compile_etrace_graph(
            cell, jnp.zeros((1, 4)), include_hidden_perturb=False
        )

        records = graph.explain(kind=DiagnosticKind.TRAINABLE_INVAR_NOT_PARAMSTATE)
        assert len(records) == 0, (
            f'Expected no TRAINABLE_INVAR_NOT_PARAMSTATE when bias is a ParamState; '
            f'got {records}'
        )
