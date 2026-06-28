# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================

"""Read-only, human- and machine-readable view over a compiled ``ETraceGraph``.

:class:`CompilationReport` is the single source of truth for rendering the
eligibility-trace compilation result. It is a pure view: it owns no state beyond
the graph it wraps, and it is safe to build lazily whenever a graph is available.
"""

from typing import Dict, List, Optional, Tuple

import brainstate

from .diagnostics import CompilationRecord, DiagnosticKind, DiagnosticLevel
from .graph import ETraceGraph
from .._typing import Path

__all__ = ['CompilationReport']

# Exclusion DiagnosticKinds, used to annotate why a ParamState is not an
# eligibility-trace weight.
_EXCLUSION_KINDS = (
    DiagnosticKind.RELATION_EXCLUDED_NO_PARAMSTATE,
    DiagnosticKind.RELATION_EXCLUDED_NON_TEMPORAL,
    DiagnosticKind.RELATION_EXCLUDED_SHAPE_MISMATCH,
    DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT,
)


class CompilationReport:
    """A read-only view over a compiled :class:`ETraceGraph`.

    Parameters
    ----------
    graph : ETraceGraph
        The compiled eligibility-trace graph to summarise.

    Notes
    -----
    All properties are derived on access from ``graph``; the report stores no
    independent state. ``to_str(1)`` renders the full structural summary (hidden
    groups, dynamic states, eligibility-trace weights, excluded weights);
    ``to_str(2)`` additionally appends WARNING/ERROR compiler diagnostics.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate, jax.numpy as jnp, braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> graph = braintrace.compile_etrace_graph(gru, jnp.ones((3,)))
        >>> report = braintrace.CompilationReport(graph)
        >>> report.counts['hidden_groups'] >= 1
        True
    """
    __module__ = 'braintrace'

    def __init__(self, graph: ETraceGraph):
        self._graph = graph

    @property
    def graph(self) -> ETraceGraph:
        """The wrapped :class:`ETraceGraph`."""
        return self._graph

    # --- data views ------------------------------------------------------

    @property
    def hidden_groups(self) -> List[Tuple[int, List[Path]]]:
        """List of ``(group_index, hidden_paths)`` for each hidden group."""
        return [(g.index, list(g.hidden_paths)) for g in self._graph.hidden_groups]

    def _etrace_paths(self) -> set:
        paths = set()
        for rel in self._graph.hidden_param_op_relations:
            paths.update(rel.trainable_paths.values())
        return paths

    @property
    def etrace_weights(self) -> List[Tuple[Path, List[int]]]:
        """List of ``(weight_path, [group_index, ...])`` for ETP-routed weights."""
        group_index = {id(g): g.index for g in self._graph.hidden_groups}
        out: List[Tuple[Path, List[int]]] = []
        for rel in self._graph.hidden_param_op_relations:
            idxs = [group_index[id(g)] for g in rel.hidden_groups]
            for path in rel.trainable_paths.values():
                out.append((path, idxs))
        return out

    def _exclusion_reasons(self) -> Dict[Path, str]:
        reasons: Dict[Path, str] = {}
        for rec in self._graph.diagnostics:
            if rec.kind in _EXCLUSION_KINDS and rec.weight_path is not None:
                reasons.setdefault(rec.weight_path, rec.kind.value)
        return reasons

    @property
    def excluded_weights(self) -> List[Tuple[Path, Optional[str]]]:
        """``ParamState`` paths not routed through any ETP op, with a reason if known."""
        states = self._graph.module_info.retrieved_model_states
        param_states = states.filter(brainstate.ParamState)
        etrace = self._etrace_paths()
        reasons = self._exclusion_reasons()
        return [
            (path, reasons.get(path))
            for path in param_states.keys()
            if path not in etrace
        ]

    @property
    def dynamic_states(self) -> List[Path]:
        """``ShortTermState`` paths that are not part of any hidden group."""
        group_paths = set()
        for g in self._graph.hidden_groups:
            group_paths.update(g.hidden_paths)
        states = self._graph.module_info.retrieved_model_states
        short = states.filter(brainstate.ShortTermState)
        return [p for p in short.keys() if p not in group_paths]

    @property
    def diagnostics(self) -> Tuple[CompilationRecord, ...]:
        """The structured compiler diagnostics, in emission order."""
        return tuple(self._graph.diagnostics)

    @property
    def counts(self) -> Dict[str, int]:
        """Summary counts of groups, weights, and diagnostic severities."""
        diags = self._graph.diagnostics
        return {
            'hidden_groups': len(self._graph.hidden_groups),
            'etrace_weights': len(self.etrace_weights),
            'excluded_weights': len(self.excluded_weights),
            'warnings': sum(1 for d in diags if d.level is DiagnosticLevel.WARNING),
            'errors': sum(1 for d in diags if d.level is DiagnosticLevel.ERROR),
        }

    # --- rendering -------------------------------------------------------

    def to_str(self, level: int = 1) -> str:
        """Render the report as text.

        Parameters
        ----------
        level : int, optional
            ``1`` (default) renders the full structural summary; ``2`` also
            appends WARNING/ERROR compiler diagnostics.

        Returns
        -------
        str
            The rendered report.
        """
        g = self._graph
        group_index = {id(grp): grp.index for grp in g.hidden_groups}

        msg = '===' * 40 + '\n'
        msg += 'The hidden groups are:\n\n'
        for grp in g.hidden_groups:
            msg += f'   Group {grp.index}: {grp.hidden_paths}\n'
        msg += '\n\n'

        dyn = self.dynamic_states
        if len(dyn):
            msg += 'The dynamic (non-hidden) states are:\n\n'
            for i, path in enumerate(dyn):
                msg += f'   Dynamic state {i}: {path}\n'
            msg += '\n\n'

        if len(g.hidden_param_op_relations):
            msg += 'The weight parameters which are associated with the hidden states are:\n\n'
            for i, rel in enumerate(g.hidden_param_op_relations):
                idxs = [group_index[id(grp)] for grp in rel.hidden_groups]
                if len(idxs) == 1:
                    msg += f'   Weight {i}: {rel.path}  is associated with hidden group {idxs[0]}\n'
                else:
                    msg += f'   Weight {i}: {rel.path}  is associated with hidden groups {idxs}\n'
            msg += '\n\n'

        excluded = self.excluded_weights
        if len(excluded):
            msg += 'The non-etrace weight parameters are:\n\n'
            for i, (path, reason) in enumerate(excluded):
                if reason is None:
                    msg += f'   Weight {i}: {path}\n'
                else:
                    msg += f'   Weight {i}: {path}  (excluded: {reason})\n'
            msg += '\n\n'

        if level >= 2:
            sev = (DiagnosticLevel.WARNING, DiagnosticLevel.ERROR)
            diags = [d for d in g.diagnostics if d.level in sev]
            if len(diags):
                msg += 'Compiler diagnostics (warnings / errors):\n\n'
                for i, d in enumerate(diags):
                    msg += f'   [{d.level.value}] {d.kind.value}: {d.message}\n'
                msg += '\n\n'

        return msg

    def show(self, level: int = 1, *, file=None) -> None:
        """Print :meth:`to_str` to ``file`` (stdout by default).

        Parameters
        ----------
        level : int, optional
            Verbosity passed to :meth:`to_str`: ``1`` (default) renders the
            full structural summary; ``2`` also appends WARNING/ERROR compiler
            diagnostics.
        file : file-like, optional
            Destination forwarded to :func:`print`. Defaults to ``None``
            (standard output).
        """
        print(self.to_str(level), file=file)

    def __repr__(self) -> str:
        c = self.counts
        return (
            f"CompilationReport(hidden_groups={c['hidden_groups']}, "
            f"etrace_weights={c['etrace_weights']}, "
            f"excluded_weights={c['excluded_weights']}, "
            f"warnings={c['warnings']}, errors={c['errors']})"
        )


CompilationReport.__module__ = 'braintrace'
