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

"""Structured diagnostics for the ETrace compiler.

Every compilation decision (a weight included as a relation, excluded because
its tail crosses another trainable ETP primitive, excluded because its shape
does not broadcast with any hidden state, and so on) emits a
:class:`CompilationRecord`. Records are collected into an
:class:`ETraceGraph`'s ``diagnostics`` field so users and tests can query
*why* the compiler made each call — rather than parsing warning strings.

Activation is scoped by :func:`diagnostic_context`; outside that context the
helpers fall back to ``warnings.warn`` so isolated compiler usage still
surfaces issues.
"""

import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

__all__ = [
    'DiagnosticLevel',
    'DiagnosticKind',
    'CompilationRecord',
    'DiagnosticReporter',
    'diagnostic_context',
    'emit',
    'get_reporter',
]


class DiagnosticLevel(str, Enum):
    """Severity of a :class:`CompilationRecord`.

    A string-valued enumeration ordering compiler diagnostics by severity.

    Attributes
    ----------
    INFO
        Informational record; not surfaced through :func:`warnings.warn`.
    WARNING
        A potential problem (e.g. an excluded relation) that is also emitted
        as a Python warning.
    ERROR
        A serious problem that is also emitted as a Python warning.
    """

    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'


class DiagnosticKind(str, Enum):
    """Machine-readable reason for a :class:`CompilationRecord`.

    A string-valued enumeration naming the exact decision the ETrace compiler
    made. Every decision maps to exactly one ``DiagnosticKind`` so tests can
    assert on ``CompilationRecord.kind`` rather than parsing message strings.

    Notes
    -----
    The members fall into a few families: an inclusion marker
    (``RELATION_INCLUDED``), exclusion reasons (the ``RELATION_EXCLUDED_*``
    members), a path-classification marker (``RELATION_PARTIAL_PATH``), and a
    set of structural observations about the jaxpr (nested ``jit``, control
    flow, multi-output primitives, state mismatches, and so on).
    """

    # Inclusion
    RELATION_INCLUDED = 'relation_included'

    # Exclusion reasons (relation not recorded)
    RELATION_EXCLUDED_NO_PARAMSTATE = 'relation_excluded_no_paramstate'
    RELATION_EXCLUDED_NON_TEMPORAL = 'relation_excluded_non_temporal'
    RELATION_EXCLUDED_SHAPE_MISMATCH = 'relation_excluded_shape_mismatch'
    RELATION_EXCLUDED_WEIGHT_TO_WEIGHT = 'relation_excluded_weight_to_weight'

    # Path classification (informational; relation still included)
    RELATION_PARTIAL_PATH = 'relation_partial_path'

    # Trainable invar did not trace back to any ParamState (e.g. a constant bias)
    TRAINABLE_INVAR_NOT_PARAMSTATE = 'trainable_invar_not_paramstate'

    # Control-flow canonicalization (see _compiler/canonicalize.py)
    COND_IF_CONVERTED = 'cond_if_converted'
    COND_CONVERSION_SKIPPED = 'cond_conversion_skipped'
    SCAN_UNROLLED = 'scan_unrolled'
    SCAN_UNROLL_SKIPPED = 'scan_unroll_skipped'
    RELATION_EXCLUDED_SLICED_WEIGHT = 'relation_excluded_sliced_weight'

    # Structured scan descent (Phase 4; see _compiler/scan_descent.py)
    #
    # ``SCAN_DESCENT_APPLIED``: an ETP-relevant scan too long to unroll was
    # rewritten for structured descent — relations/hidden groups discovered
    # inside its body, stacked per-substep values emitted as extra ys (INFO).
    # ``SCAN_DESCENT_SKIPPED``: descent was requested (policy ``'auto'``) but
    # a v1 restriction blocked it; the existing control-flow restrictions
    # apply (WARNING).
    # ``SCAN_DESCENT_NO_RELATIONS``: a descended scan consumes weight
    # state(s) that produce no ETP relation (plain-op usage inside the
    # body), so those weights do not learn online. Pre-descent this scan
    # was a hard error, so the exclusion is surfaced loudly (WARNING).
    SCAN_DESCENT_APPLIED = 'scan_descent_applied'
    SCAN_DESCENT_SKIPPED = 'scan_descent_skipped'
    SCAN_DESCENT_NO_RELATIONS = 'scan_descent_no_relations'

    # Opaque control flow touching weights / hidden states (Phase 3)
    #
    # ``WEIGHT_IN_WHILE``: a tracked weight invar is consumed by a ``while``
    # equation — always an ERROR (a data-dependent trip count admits no fixed
    # per-iteration hoisting, so the weight cannot participate in online
    # learning).
    # ``CONTROL_FLOW_OPAQUE_FWD``: a weight-free opaque ``scan``/``while``/
    # ``cond`` produces a hidden state and is kept as an opaque forward node
    # (INFO; see ``ControlFlowPolicy.while_hidden``).
    # ``CONTROL_FLOW_RECURRENT_MIXING``: an opaque control-flow body applies a
    # recurrent weight-mixing primitive to the carried hidden state, so the
    # equation is excluded from the hidden-to-hidden transition in the default
    # ("without recurrence") grouping mode (WARNING).
    WEIGHT_IN_WHILE = 'weight_in_while'
    CONTROL_FLOW_OPAQUE_FWD = 'control_flow_opaque_fwd'
    CONTROL_FLOW_RECURRENT_MIXING = 'control_flow_recurrent_mixing'

    # Structural observations (informational / partial)
    PRIMITIVE_INSIDE_NESTED_JIT = 'primitive_inside_nested_jit'
    PRIMITIVE_INSIDE_CONTROL_FLOW = 'primitive_inside_control_flow'
    MULTI_OUTPUT_PRIMITIVE_DETECTED = 'multi_output_primitive_detected'
    PYTREE_WEIGHT_LEAF_AMBIGUOUS = 'pytree_weight_leaf_ambiguous'
    TRANSITION_TAIL_BOUNDED = 'transition_tail_bounded'
    HIDDEN_GROUP_MERGED = 'hidden_group_merged'
    STATE_MISMATCH = 'state_mismatch'
    WEIGHT_IN_CONTROL_FLOW = 'weight_in_control_flow'


@dataclass(frozen=True)
class CompilationRecord:
    """A single compiler decision, captured with structured context.

    A frozen dataclass recording one decision made by the ETrace compiler,
    together with enough structured context to query *why* the decision was
    made without parsing the human-readable ``message``.

    Parameters
    ----------
    kind : DiagnosticKind
        Machine-readable reason for the record.
    level : DiagnosticLevel
        Severity of the record.
    message : str
        Human-readable description of the decision.
    primitive : object or None, optional
        The JAX primitive the decision concerns, if any. Default ``None``.
    weight_path : tuple of object or None, optional
        Module path of the weight ``ParamState`` the decision concerns, if
        any. Default ``None``.
    hidden_paths : tuple of tuple of object, optional
        Module paths of the hidden states the decision concerns. Default
        ``()``.
    context : dict or None, optional
        Open dict of extra context keyed by the emitting site; see the
        :class:`DiagnosticKind` documentation for the schema of each kind.
        Default ``None``.
    """

    kind: DiagnosticKind
    level: DiagnosticLevel
    message: str
    primitive: Optional[Any] = None
    weight_path: Optional[Tuple[Any, ...]] = None
    hidden_paths: Tuple[Tuple[Any, ...], ...] = ()
    context: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        parts = [f'kind={self.kind.value}', f'level={self.level.value}']
        if self.primitive is not None:
            parts.append(
                f'primitive={getattr(self.primitive, "name", self.primitive)!r}'
            )
        if self.weight_path is not None:
            parts.append(f'weight_path={self.weight_path}')
        if self.hidden_paths:
            parts.append(f'hidden_paths={list(self.hidden_paths)}')
        parts.append(f'message={self.message!r}')
        if self.context:
            parts.append(f'context={self.context}')
        return f'CompilationRecord({", ".join(parts)})'


class DiagnosticReporter:
    """Collects :class:`CompilationRecord` instances during a compilation pass."""

    def __init__(self) -> None:
        self._records: List[CompilationRecord] = []

    def append(self, record: CompilationRecord) -> None:
        self._records.append(record)

    def records(self) -> Tuple[CompilationRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)


_CURRENT = threading.local()


def get_reporter() -> Optional[DiagnosticReporter]:
    """Return the reporter active for the current thread, or ``None``."""
    return getattr(_CURRENT, 'reporter', None)


@contextmanager
def diagnostic_context() -> Iterator[DiagnosticReporter]:
    """Activate a :class:`DiagnosticReporter` for the current thread.

    Nested contexts are honoured: inner ``emit()`` calls land in the innermost
    reporter, and the previous reporter is restored on exit.
    """
    prev = getattr(_CURRENT, 'reporter', None)
    reporter = DiagnosticReporter()
    _CURRENT.reporter = reporter
    try:
        yield reporter
    finally:
        _CURRENT.reporter = prev


def emit(
    kind: DiagnosticKind,
    level: DiagnosticLevel,
    message: str,
    *,
    primitive: Any = None,
    weight_path: Optional[Tuple[Any, ...]] = None,
    hidden_paths: Tuple[Tuple[Any, ...], ...] = (),
    context: Optional[Dict[str, Any]] = None,
    also_warn: bool = True,
    stacklevel: int = 3,
) -> CompilationRecord:
    """Emit a :class:`CompilationRecord` to the active reporter.

    Always calls :func:`warnings.warn` with ``message`` when ``level`` is
    :attr:`DiagnosticLevel.WARNING` or :attr:`DiagnosticLevel.ERROR` and
    ``also_warn`` is ``True``, so non-structured consumers still see the
    message.  Returns the record for convenience in tests.
    """
    record = CompilationRecord(
        kind=kind,
        level=level,
        message=message,
        primitive=primitive,
        weight_path=weight_path,
        hidden_paths=hidden_paths,
        context=context,
    )
    reporter = get_reporter()
    if reporter is not None:
        reporter.append(record)
    if also_warn and level is not DiagnosticLevel.INFO:
        warnings.warn(message, stacklevel=stacklevel)
    return record
