# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Primitive-based weight-to-hidden-state relation discovery.

Replaces the old name-matching approach with direct primitive type checking
in the Jaxpr. The compiler:

1. Walks the Jaxpr (descending into ``jit``/``pjit``, ``scan``, ``while``,
   ``cond`` bodies) and collects equations whose primitive is registered
   as ETP (``eqn.primitive in ETP_PRIMITIVES``).
2. For each such equation, traces the weight invar backward to the
   originating ``ParamState`` (handling pytree leaves, masks, weight_fn
   chains).
3. Traces forward from the primitive output to find reachable hidden-state
   outvars and classifies each (weight, hidden) candidate path as
   ``ALL_DIRECT`` / ``ALL_THROUGH_OTHER_ETP`` / ``MIXED``.
4. Builds a transition Jaxpr ``y -> h`` for each connected hidden group,
   stopping at non-gradient-enabled ETP boundaries so the tail is
   non-parametric.

The algorithm is deterministic: every set-typed traversal is replaced
with insertion-ordered ``dict`` so the relation order is stable across
runs of the same model.
"""

from collections import deque
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import brainstate
import jax

from braintrace._compatible_imports import (
    Primitive,
    Var,
    Literal,
    JaxprEqn,
    Jaxpr,
    is_jit_primitive,
    is_scan_primitive,
    is_while_primitive,
    is_cond_primitive,
)
from braintrace._etrace_op import (
    ETP_PRIMITIVES,
    get_primitive_spec,
    is_etp_primitive,
    is_etp_enable_gradient_primitive,
)
from braintrace._misc import git_issue_addr
from braintrace._typing import (
    Path,
    HiddenOutVar,
)
from .diagnostics import (
    DiagnosticKind,
    DiagnosticLevel,
    emit,
)
from .hidden_group import HiddenGroup, find_hidden_groups_from_minfo
from .module_info import ModuleInfo, extract_module_info

__all__ = [
    'HiddenParamOpRelation',
    'PathClassification',
    'find_hidden_param_op_relations_from_minfo',
    'find_hidden_param_op_relations_from_module',
]


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

class PathClassification:
    """Three-way classification of paths from ``y`` to a hidden outvar.

    Used to decide whether a (weight, hidden) candidate is registered as a
    relation and which structured diagnostic to emit.
    """

    #: Every path from ``y`` to ``h`` avoids any other non-gradient-enabled
    #: ETP primitive. Relation included; emit ``RELATION_INCLUDED``.
    ALL_DIRECT = 'all_direct'

    #: Every path from ``y`` to ``h`` traverses another non-gradient-enabled
    #: ETP primitive. Relation excluded; emit
    #: ``RELATION_EXCLUDED_WEIGHT_TO_WEIGHT``.
    ALL_THROUGH_OTHER_ETP = 'all_through_other_etp'

    #: Some paths are direct and some traverse another ETP primitive. The
    #: relation is still registered (preserving prior behavior) but
    #: ``RELATION_PARTIAL_PATH`` is emitted at INFO level so callers know
    #: the gradient bookkeeping is only partially captured by ETP.
    MIXED = 'mixed'


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

class HiddenParamOpRelation(NamedTuple):
    r"""Connection between an ETP primitive, its trainable parameters, and hidden states.

    Records the structural relationship:

    .. math::
        h^t = f(y), \quad y = \text{primitive}(x, \theta)

    Attributes:
        primitive: The JAX primitive (``etp_mm_p``, ``etp_mv_p``, etc.).
        x_var: Jaxpr ``Var`` for the input (``None`` for element-wise ops).
        y_var: Jaxpr ``Var`` for the primitive output.
        hidden_groups: Hidden groups that this op feeds into.
        y_to_hidden_group_jaxprs: Transition Jaxpr from *y* to each hidden group.
        connected_hidden_paths: Hidden-state paths connected to this op.
        eqn_params: Static parameters of the primitive equation.
        path_classification: ``{hidden_path: PathClassification.*}`` for each
            connected hidden state. Populated by the path-classification pass.
        trainable_vars: Per-key dict mapping a primitive-chosen key name
            (e.g. ``'weight'``, ``'bias'``, ``'lora_b'``, ``'lora_a'``) to its
            jaxpr ``Var``. Populated by the compiler with one entry per declared
            trainable input.
        trainable_paths: Per-key dict mapping each key to the owning
            ``ParamState``'s module path. When the primitive has two keys
            whose invars trace to the same ``ParamState`` (e.g. merged
            ``{weight, bias}`` Linear), the entries share a path.
        trainable_leaf_indices: Per-key dict mapping each key to the leaf
            index in ``jax.tree.leaves`` of the owning ``ParamState``.
        trainable_param_states: Per-key dict mapping each key to the actual
            ``ParamState`` object.
        trainable_processing_chains: Per-key dict mapping each key to the
            backward-trace processing chain (primitives traversed from the
            trainable invar back to the originating ``ParamState`` invar).
    """
    primitive: Primitive
    x_var: Optional[Var]
    y_var: Var
    hidden_groups: List[HiddenGroup]
    y_to_hidden_group_jaxprs: List[Jaxpr]
    connected_hidden_paths: List[Path]
    eqn_params: dict
    path_classification: Dict[Path, str] = {}
    trainable_vars: Dict[str, Var] = {}
    trainable_paths: Dict[str, Path] = {}
    trainable_leaf_indices: Dict[str, int] = {}
    trainable_param_states: Dict[str, brainstate.ParamState] = {}
    trainable_processing_chains: Dict[str, Tuple[Primitive, ...]] = {}

    # backward compat aliases
    @property
    def x(self):
        return self.x_var

    @property
    def y(self):
        return self.y_var

    @property
    def path(self):
        return next(iter(self.trainable_paths.values()), None)

    def y_to_hidden_groups(self, y_val, const_vals, concat_hidden_vals=True):
        """Evaluate transition jaxprs: y -> hidden group values."""
        vals_of_hidden_groups = []
        for jaxpr, group in zip(self.y_to_hidden_group_jaxprs, self.hidden_groups):
            consts = [const_vals[var] for var in jaxpr.constvars]
            hidden_vals = jax.core.eval_jaxpr(jaxpr, consts, y_val)
            if concat_hidden_vals:
                hidden_vals = group.concat_hidden(hidden_vals)
            vals_of_hidden_groups.append(hidden_vals)
        return vals_of_hidden_groups

    def dict(self) -> Dict[str, Any]:
        return self._asdict()

    def __repr__(self) -> str:
        return repr(
            brainstate.util.PrettyMapping(
                self._asdict(), type_name=self.__class__.__name__
            )
        )


HiddenParamOpRelation.__module__ = 'braintrace'


# ---------------------------------------------------------------------------
# Internal helpers — producer / consumer maps
# ---------------------------------------------------------------------------

def _build_producer_map(jaxpr: Jaxpr) -> Dict[Var, JaxprEqn]:
    """Map each variable to the equation that produces it."""
    producers: Dict[Var, JaxprEqn] = {}
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            producers[v] = eqn
    return producers


def _build_consumer_map(jaxpr: Jaxpr) -> Dict[Var, List[JaxprEqn]]:
    """Map each variable to the equations that consume it."""
    consumers: Dict[Var, List[JaxprEqn]] = {}
    for eqn in jaxpr.eqns:
        for v in eqn.invars:
            if isinstance(v, Var):
                consumers.setdefault(v, []).append(eqn)
    return consumers


# ---------------------------------------------------------------------------
# Spec dispatch — locate weight / x / y vars on a primitive equation
# ---------------------------------------------------------------------------

def _resolve_eqn_trainable_invars(
    eqn: JaxprEqn,
) -> Dict[str, Var]:
    """Return ``{key: invar_var}`` for every trainable input of *eqn*."""
    primitive = eqn.primitive
    spec = get_primitive_spec(primitive)
    if spec is None:
        raise RuntimeError(
            f'ETP primitive {primitive.name} has no registered spec'
        )
    key_to_idx = spec.resolve_trainable_invars(eqn.params)
    return {k: eqn.invars[i] for k, i in key_to_idx.items()}


def _resolve_eqn_vars(
    eqn: JaxprEqn,
) -> Tuple[Optional[Var], Var]:
    """Return ``(x_var, y_var)`` for an ETP primitive equation."""
    primitive = eqn.primitive
    spec = get_primitive_spec(primitive)
    if spec is None:
        raise RuntimeError(
            f'ETP primitive {primitive.name} has no registered spec'
        )
    if spec.x_invar_index is None:
        x_var = None
    else:
        candidate = eqn.invars[spec.x_invar_index]
        x_var = candidate if isinstance(candidate, Var) else None
    y_var = eqn.outvars[spec.y_outvar_index]
    if len(eqn.outvars) > 1:
        emit(
            kind=DiagnosticKind.MULTI_OUTPUT_PRIMITIVE_DETECTED,
            level=DiagnosticLevel.INFO,
            message=(
                f'ETP primitive {primitive.name} has '
                f'{len(eqn.outvars)} outputs; using outvar index '
                f'{spec.y_outvar_index} as y per spec.'
            ),
            primitive=primitive,
            context={'num_outvars': len(eqn.outvars),
                     'y_outvar_index': spec.y_outvar_index},
        )
    return x_var, y_var


# ---------------------------------------------------------------------------
# Weight backward trace (ParamState resolution)
# ---------------------------------------------------------------------------

def _trace_var_to_param(
    var: Var,
    producers: Dict[Var, JaxprEqn],
    invar_to_weight_path: Dict[Var, Path],
    cache: Optional[Dict[Var, Optional[Tuple[Path, Tuple[Primitive, ...]]]]] = None,
) -> Tuple[Optional[Path], Tuple[Primitive, ...]]:
    """Trace *var* backward through the Jaxpr to its originating ``ParamState``.

    Returns ``(path, processing_chain)`` where ``processing_chain`` is the
    deduplicated, insertion-ordered tuple of intermediate primitive types
    traversed (mask multiplication, weight_fn, etc.). When the var is the
    raw ParamState invar, the chain is empty.
    """
    if cache is not None and var in cache:
        cached = cache[var]
        if cached is None:
            return None, ()
        return cached

    frontier: deque = deque([var])
    visited: Set[Var] = set()
    chain: Dict[Primitive, None] = {}
    found_path: Optional[Path] = None
    while frontier:
        v = frontier.popleft()
        if v in visited:
            continue
        visited.add(v)
        path = invar_to_weight_path.get(v)
        if path is not None:
            found_path = path
            break
        eqn = producers.get(v)
        if eqn is not None:
            chain[eqn.primitive] = None
            for iv in eqn.invars:
                if isinstance(iv, Var) and iv not in visited:
                    frontier.append(iv)
    chain_tuple = tuple(chain)
    if cache is not None:
        cache[var] = (found_path, chain_tuple) if found_path is not None else None
    return found_path, chain_tuple


def _resolve_weight_leaf_idx(
    weight_var: Var,
    weight_path: Path,
    producers: Dict[Var, JaxprEqn],
    weight_path_to_invars: Optional[Dict[Path, List[Var]]],
) -> int:
    """Find which leaf of the owning ``ParamState`` produced *weight_var*.

    Backtraces *weight_var* through any ``mask``/``weight_fn`` chain to the
    raw ParamState invar and returns its position in the ParamState's leaf
    list. Falls back to ``0`` when the ParamState only has one leaf.
    """
    if weight_path_to_invars is None:
        return 0
    invars_list = weight_path_to_invars.get(weight_path, [])
    if len(invars_list) <= 1:
        return 0
    source_var = weight_var
    frontier: List[Var] = [weight_var]
    visited: Set[Var] = set()
    while frontier:
        v = frontier.pop()
        if v in visited:
            continue
        visited.add(v)
        if v in invars_list:
            source_var = v
            break
        eqn = producers.get(v)
        if eqn is not None:
            for iv in eqn.invars:
                if isinstance(iv, Var) and iv not in visited:
                    frontier.append(iv)
    try:
        return invars_list.index(source_var)
    except ValueError:
        emit(
            kind=DiagnosticKind.PYTREE_WEIGHT_LEAF_AMBIGUOUS,
            level=DiagnosticLevel.WARNING,
            message=(
                f'Could not resolve weight leaf for ParamState at {weight_path}; '
                f'falling back to leaf index 0.'
            ),
            weight_path=weight_path,
            context={'weight_var': weight_var,
                     'num_leaves': len(invars_list)},
        )
        return 0


# ---------------------------------------------------------------------------
# Forward reachability + path classification
# ---------------------------------------------------------------------------

def _bfs_forward(
    start_var: Var,
    consumer_map: Dict[Var, List[JaxprEqn]],
    hidden_outvar_set: Set[Var],
    *,
    stop_at_non_grad_etp: bool,
    outvar_to_group_index: Optional[Dict[Var, int]] = None,
) -> Tuple[Dict[Var, None], Tuple[JaxprEqn, ...]]:
    """Forward BFS from *start_var* to hidden outvars.

    Returns ``(reachable_hvars, blocking_eqns)``. ``reachable_hvars`` is an
    insertion-ordered dict (used as an ordered set) so iteration follows
    BFS encounter order — a plain ``set`` yields hash-ordered iteration,
    which makes the compiler's relation output non-deterministic.

    When ``stop_at_non_grad_etp`` is True, the BFS does not cross
    non-gradient-enabled ETP primitives (preserves the historical
    "restricted" semantics). When False, it crosses all equations and
    returns the full reachability set (used by path classification).

    When ``outvar_to_group_index`` is provided, the search restricts itself
    to hidden outvars in the *closest* hidden group — outvars from different
    groups are pruned so the relation only tracks the recurrence of the
    layer this primitive actually feeds.
    """
    reachable: Dict[Var, None] = {}
    home_group_indices: Set[int] = set()
    frontier: deque = deque([start_var])
    visited: Set[Var] = set()
    blocking_eqns: List[JaxprEqn] = []
    blocking_seen: Set[int] = set()
    while frontier:
        v = frontier.popleft()
        if v in visited:
            continue
        visited.add(v)
        if v in hidden_outvar_set:
            if outvar_to_group_index is not None:
                g = outvar_to_group_index.get(v)
                if not home_group_indices:
                    home_group_indices.add(g)
                if g in home_group_indices:
                    reachable[v] = None
                else:
                    continue
            else:
                reachable[v] = None
        for eqn in consumer_map.get(v, []):
            if (
                stop_at_non_grad_etp
                and is_etp_primitive(eqn.primitive)
                and not is_etp_enable_gradient_primitive(eqn.primitive)
            ):
                key = id(eqn)
                if key not in blocking_seen:
                    blocking_seen.add(key)
                    blocking_eqns.append(eqn)
                continue
            for ov in eqn.outvars:
                if ov not in visited:
                    frontier.append(ov)
    return reachable, tuple(blocking_eqns)


def _classify_path(
    y_var: Var,
    hidden_outvar: Var,
    consumer_map: Dict[Var, List[JaxprEqn]],
    producer_map: Dict[Var, JaxprEqn],
    self_eqn: JaxprEqn,
) -> str:
    """Classify the set of paths from ``y_var`` to ``hidden_outvar``.

    Returns one of :class:`PathClassification` constants.

    Algorithm:
      * ``forward`` = vars reachable from y_var without restriction.
      * ``backward`` = vars that can reach hidden_outvar by following
        consumers in reverse (i.e. equations that produce hidden_outvar).
      * ``mid`` = forward & backward = vars on at least one path y -> h.
      * If no equation in ``mid`` is a non-gradient-enabled ETP primitive
        (other than ``self_eqn``), classification is ``ALL_DIRECT``.
      * Otherwise, run a restricted BFS that severs non-grad-ETP edges and
        check if hidden_outvar is still reachable. Yes -> ``MIXED``;
        No -> ``ALL_THROUGH_OTHER_ETP``.
    """
    # Forward reachability (unrestricted).
    forward: Set[Var] = set()
    frontier: deque = deque([y_var])
    while frontier:
        v = frontier.popleft()
        if v in forward:
            continue
        forward.add(v)
        for eqn in consumer_map.get(v, []):
            for ov in eqn.outvars:
                if ov not in forward:
                    frontier.append(ov)

    if hidden_outvar not in forward:
        # Should not happen — caller already established reachability.
        return PathClassification.ALL_THROUGH_OTHER_ETP

    # Backward reachability: walk up from hidden_outvar via producers.
    backward: Set[Var] = set()
    bfrontier: deque = deque([hidden_outvar])
    while bfrontier:
        v = bfrontier.popleft()
        if v in backward:
            continue
        backward.add(v)
        eqn = producer_map.get(v)
        if eqn is None:
            continue
        for iv in eqn.invars:
            if isinstance(iv, Var) and iv not in backward:
                bfrontier.append(iv)

    # Eqns on at least one path.
    mid_eqns: List[JaxprEqn] = []
    for v, eqn in producer_map.items():
        if v in backward and v in forward and eqn is not self_eqn:
            if eqn not in mid_eqns:
                mid_eqns.append(eqn)

    has_blocking = any(
        is_etp_primitive(e.primitive)
        and not is_etp_enable_gradient_primitive(e.primitive)
        for e in mid_eqns
    )
    if not has_blocking:
        return PathClassification.ALL_DIRECT

    # Run restricted BFS to see if a direct path also exists.
    restricted_reach, _ = _bfs_forward(
        y_var, consumer_map,
        hidden_outvar_set={hidden_outvar},
        stop_at_non_grad_etp=True,
    )
    if hidden_outvar in restricted_reach:
        return PathClassification.MIXED
    return PathClassification.ALL_THROUGH_OTHER_ETP


# ---------------------------------------------------------------------------
# Transition jaxpr construction
# ---------------------------------------------------------------------------

def _build_transition_jaxpr(
    y_var: Var,
    group: HiddenGroup,
    jaxpr: Jaxpr,
) -> Jaxpr:
    """Build the sub-Jaxpr mapping y_var -> hidden group outputs.

    Collects all equations that backward-contribute to ``group.hidden_outvars``.
    Vars referenced by these equations but not produced by them (and not
    ``y_var``) become constvars, whose values must be supplied at evaluation
    time. Hidden outvars that do not depend on ``y_var`` still get computed
    from their constvar dependencies — their jvp tangent with respect to
    ``y_var`` is then zero, as expected.

    Equations that belong to another ETP primitive (not gradient-enabled) are
    treated as constvar boundaries: their outputs are supplied externally and
    their internal computation (and trainable weights) is *not* pulled into
    the transition jaxpr. This keeps ``dh/dy`` strictly through the
    non-parametric tail.
    """
    selected_rev = []
    all_needed_vars: Set[Var] = set(group.hidden_outvars)
    for eqn in reversed(jaxpr.eqns):
        if any(ov in all_needed_vars for ov in eqn.outvars):
            # Skip the equation that produces ``y_var`` — its value is
            # supplied as the transition jaxpr's *invar*. Otherwise the
            # backward walk would re-include the ETP primitive itself
            # and ``dh/dy`` would silently evaluate to zero.
            if y_var in eqn.outvars:
                continue
            if (
                is_etp_primitive(eqn.primitive)
                and not is_etp_enable_gradient_primitive(eqn.primitive)
            ):
                # Other ETP primitive on the tail -> stop, output becomes
                # a constvar of the transition jaxpr.
                continue
            selected_rev.append(eqn)
            for iv in eqn.invars:
                if isinstance(iv, Var):
                    all_needed_vars.add(iv)
    selected = list(reversed(selected_rev))

    produced: Set[Var] = {y_var}
    for eqn in selected:
        for ov in eqn.outvars:
            produced.add(ov)
    invars_needed: Dict[Var, None] = {}  # ordered set
    for eqn in selected:
        for iv in eqn.invars:
            if isinstance(iv, Var) and iv not in produced:
                invars_needed[iv] = None
    constvars = list(invars_needed)

    return Jaxpr(
        constvars=constvars,
        invars=[y_var],
        outvars=list(group.hidden_outvars),
        eqns=selected,
    )


# ---------------------------------------------------------------------------
# Jaxpr scanning (with control-flow descent)
# ---------------------------------------------------------------------------

def _scan_jaxpr_for_etp_eqns(
    jaxpr: Jaxpr,
    *,
    inside_control_flow: bool = False,
) -> List[JaxprEqn]:
    """Walk the Jaxpr and return all equations whose primitive is ETP.

    Descends into ``jit``/``pjit`` (transparently) and emits diagnostics
    when ETP primitives are found inside control-flow primitives
    (``scan``/``while``/``cond``). Control-flow ETP primitives are *not*
    returned to the main relation pass — their semantics (carry vars,
    branch unification) are not yet fully supported and a structured
    diagnostic is emitted so users can locate them.
    """
    etp_eqns: List[JaxprEqn] = []
    for eqn in jaxpr.eqns:
        if is_etp_primitive(eqn.primitive):
            if inside_control_flow:
                emit(
                    kind=DiagnosticKind.PRIMITIVE_INSIDE_CONTROL_FLOW,
                    level=DiagnosticLevel.WARNING,
                    message=(
                        f'ETP primitive {eqn.primitive.name} found inside a '
                        f'scan/while/cond body. Such primitives are not '
                        f'currently registered as relations because the '
                        f'compiler cannot yet expose carry-variable lineage '
                        f'across the control-flow boundary. The weight will '
                        f'not participate in ETP. Lift it out of the body or '
                        f'use BPTT.'
                    ),
                    primitive=eqn.primitive,
                )
            else:
                etp_eqns.append(eqn)
        elif is_jit_primitive(eqn) and 'jaxpr' in eqn.params:
            inner_jaxpr = eqn.params['jaxpr'].jaxpr
            inner_etp = _scan_jaxpr_for_etp_eqns(
                inner_jaxpr, inside_control_flow=inside_control_flow,
            )
            if inner_etp:
                emit(
                    kind=DiagnosticKind.PRIMITIVE_INSIDE_NESTED_JIT,
                    level=DiagnosticLevel.WARNING,
                    message=(
                        'Found ETP primitives inside a nested jit/pjit. '
                        'This is currently handled by tracing through the '
                        'outer jaxpr. If you see incorrect results, please '
                        'avoid wrapping individual ETP calls in jax.jit. '
                        f'Report issues at {git_issue_addr}.'
                    ),
                    context={'inner_primitives': tuple(
                        e.primitive for e in inner_etp
                    )},
                )
        elif is_scan_primitive(eqn) or is_while_primitive(eqn) or is_cond_primitive(eqn):
            for sub_jaxpr in _control_flow_subjaxprs(eqn):
                _scan_jaxpr_for_etp_eqns(
                    sub_jaxpr, inside_control_flow=True,
                )
    return etp_eqns


def _control_flow_subjaxprs(eqn: JaxprEqn) -> Iterable[Jaxpr]:
    """Yield every sub-Jaxpr stored on a control-flow equation's params.

    ``scan`` exposes ``jaxpr`` (a ``ClosedJaxpr``); ``while`` exposes
    ``cond_jaxpr`` and ``body_jaxpr``; ``cond`` exposes ``branches`` (a
    sequence of ``ClosedJaxpr``).
    """
    params = eqn.params
    for key in ('jaxpr', 'cond_jaxpr', 'body_jaxpr'):
        sub = params.get(key)
        if sub is not None:
            yield getattr(sub, 'jaxpr', sub)
    branches = params.get('branches')
    if branches is not None:
        for b in branches:
            yield getattr(b, 'jaxpr', b)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def find_hidden_param_op_relations_from_jaxpr(
    jaxpr: Jaxpr,
    invar_to_weight_path: Dict[Var, Path],
    path_to_state: Dict[Path, brainstate.State],
    outvar_to_hidden_path: Dict[HiddenOutVar, Path],
    hid_path_to_group: Dict[Path, HiddenGroup],
    weight_path_to_invars: Optional[Dict[Path, List[Var]]] = None,
    **_ignored,
) -> Sequence[HiddenParamOpRelation]:
    """Find all ETP-primitive-to-hidden-state relations in *jaxpr*."""
    producers = _build_producer_map(jaxpr)
    consumers = _build_consumer_map(jaxpr)
    hidden_outvar_set: Set[Var] = set(outvar_to_hidden_path.keys())
    weight_trace_cache: Dict[Var, Optional[Tuple[Path, Tuple[Primitive, ...]]]] = {}

    outvar_to_group_index: Dict[Var, int] = {
        ov: hid_path_to_group[p].index
        for ov, p in outvar_to_hidden_path.items()
        if p in hid_path_to_group
    }

    etp_eqns = _scan_jaxpr_for_etp_eqns(jaxpr)
    relations: List[HiddenParamOpRelation] = []

    for eqn in etp_eqns:
        primitive = eqn.primitive
        x_var, y_var = _resolve_eqn_vars(eqn)

        # --- Resolve every trainable invar declared by the primitive ---
        spec = get_primitive_spec(primitive)
        key_to_idx = spec.resolve_trainable_invars(eqn.params)
        trainable_invars_map = {k: eqn.invars[i] for k, i in key_to_idx.items()}
        trainable_vars: Dict[str, Var] = {}
        trainable_paths: Dict[str, Path] = {}
        trainable_leaf_indices: Dict[str, int] = {}
        trainable_param_states: Dict[str, brainstate.ParamState] = {}
        trainable_processing_chains: Dict[str, Tuple[Primitive, ...]] = {}

        # Use the first trainable key for the primary weight trace (needed for
        # diagnostics and shape-mismatch error messages below).
        first_key = next(iter(trainable_invars_map)) if trainable_invars_map else None
        weight_path: Optional[Path] = None
        if first_key is not None:
            primary_invar = trainable_invars_map[first_key]
            weight_path, _ = _trace_var_to_param(
                primary_invar, producers, invar_to_weight_path,
                cache=weight_trace_cache,
            )

        if weight_path is None:
            first_invar_repr = trainable_invars_map[first_key]
            emit(
                kind=DiagnosticKind.RELATION_EXCLUDED_NO_PARAMSTATE,
                level=DiagnosticLevel.WARNING,
                message=(
                    f'ETP primitive {primitive.name} at {eqn} has a trainable input '
                    f'({first_key}) that could not be traced back to any ParamState. Skipping.'
                ),
                primitive=primitive,
                context={'trainable_var': first_invar_repr, 'key': first_key},
            )
            continue

        for key, invar in trainable_invars_map.items():
            t_path, t_chain = _trace_var_to_param(
                invar, producers, invar_to_weight_path,
                cache=weight_trace_cache,
            )
            if t_path is None:
                # Trainable invar doesn't trace to any ParamState (e.g. a
                # constant bias passed directly as a jnp.array). Emit an INFO
                # diagnostic so users know no gradient will be produced for
                # this input, then skip it.
                emit(
                    kind=DiagnosticKind.TRAINABLE_INVAR_NOT_PARAMSTATE,
                    level=DiagnosticLevel.INFO,
                    message=(
                        f"ETP primitive {eqn.primitive.name}: trainable input "
                        f"'{key}' at invar index {key_to_idx[key]} does not trace to any "
                        f"ParamState. No online gradient will be produced for this input."
                    ),
                    primitive=eqn.primitive,
                    context={'key': key, 'invar_index': key_to_idx[key]},
                )
                continue
            t_leaf = _resolve_weight_leaf_idx(
                invar, t_path, producers, weight_path_to_invars,
            )
            t_state = path_to_state.get(t_path)
            trainable_vars[key] = invar
            trainable_paths[key] = t_path
            trainable_leaf_indices[key] = t_leaf
            trainable_param_states[key] = t_state
            trainable_processing_chains[key] = t_chain

        # --- Restricted reachability for relation registration ---
        reachable_hvars, blocking_eqns = _bfs_forward(
            y_var, consumers, hidden_outvar_set,
            stop_at_non_grad_etp=True,
            outvar_to_group_index=outvar_to_group_index,
        )

        # --- Filter by shape compatibility ---
        connected_paths: List[Path] = []
        path_class: Dict[Path, str] = {}
        for hvar in list(reachable_hvars):
            try:
                jax.numpy.broadcast_shapes(y_var.aval.shape, hvar.aval.shape)
            except ValueError:
                emit(
                    kind=DiagnosticKind.RELATION_EXCLUDED_SHAPE_MISMATCH,
                    level=DiagnosticLevel.WARNING,
                    message=(
                        f'ETP op {primitive.name}: weight={weight_path}, '
                        f'y shape={y_var.aval.shape} not broadcastable with '
                        f'hidden shape={hvar.aval.shape} at '
                        f'{outvar_to_hidden_path[hvar]}. Removing connection.'
                    ),
                    primitive=primitive,
                    weight_path=weight_path,
                    hidden_paths=(outvar_to_hidden_path[hvar],),
                    context={
                        'y_shape': tuple(y_var.aval.shape),
                        'hidden_shape': tuple(hvar.aval.shape),
                    },
                )
                reachable_hvars.pop(hvar, None)
                continue
            hpath = outvar_to_hidden_path[hvar]
            connected_paths.append(hpath)
            cls = _classify_path(
                y_var, hvar, consumers, producers, self_eqn=eqn,
            )
            path_class[hpath] = cls

        if not connected_paths:
            _emit_no_relation_diag(
                primitive, weight_path, blocking_eqns,
                producers, invar_to_weight_path, weight_trace_cache,
            )
            continue

        # MIXED paths: still register but emit info-level diagnostic so callers
        # know the gradient bookkeeping is only partially captured.
        for hpath, cls in path_class.items():
            if cls == PathClassification.MIXED:
                emit(
                    kind=DiagnosticKind.RELATION_PARTIAL_PATH,
                    level=DiagnosticLevel.WARNING,
                    message=(
                        f'ETP primitive {primitive.name} (weight={weight_path}) '
                        f'reaches hidden state {hpath} through *both* a direct '
                        f'tail and another trainable ETP primitive. The relation '
                        f'is still registered (preserving prior behavior) but '
                        f'the indirect path is not captured by ETP. The gradient '
                        f'contribution through the indirect path will be missing.'
                    ),
                    primitive=primitive,
                    weight_path=weight_path,
                    hidden_paths=(hpath,),
                    context={'classification': cls},
                )

        # --- Group by hidden group ---
        group_ids_seen: Set[int] = set()
        connected_groups: List[HiddenGroup] = []
        for p in connected_paths:
            g = hid_path_to_group[p]
            if g.index not in group_ids_seen:
                group_ids_seen.add(g.index)
                connected_groups.append(g)

        # --- Build transition Jaxprs ---
        y_to_hid_jaxprs = [
            _build_transition_jaxpr(y_var, g, jaxpr)
            for g in connected_groups
        ]

        relations.append(HiddenParamOpRelation(
            primitive=primitive,
            x_var=x_var,
            y_var=y_var,
            hidden_groups=connected_groups,
            y_to_hidden_group_jaxprs=y_to_hid_jaxprs,
            connected_hidden_paths=connected_paths,
            eqn_params=dict(eqn.params),
            path_classification=dict(path_class),
            trainable_vars=dict(trainable_vars),
            trainable_paths=dict(trainable_paths),
            trainable_leaf_indices=dict(trainable_leaf_indices),
            trainable_param_states=dict(trainable_param_states),
            trainable_processing_chains=dict(trainable_processing_chains),
        ))
        emit(
            kind=DiagnosticKind.RELATION_INCLUDED,
            level=DiagnosticLevel.INFO,
            message=(
                f'{primitive.name}({weight_path}) -> '
                f'{[g.index for g in connected_groups]}'
            ),
            primitive=primitive,
            weight_path=weight_path,
            hidden_paths=tuple(connected_paths),
            context={
                'hidden_group_indices': tuple(g.index for g in connected_groups),
                'path_classification': dict(path_class),
            },
        )

    return tuple(relations)


def _emit_no_relation_diag(
    primitive: Primitive,
    weight_path: Path,
    blocking_eqns: Tuple[JaxprEqn, ...],
    producers: Dict[Var, JaxprEqn],
    invar_to_weight_path: Dict[Var, Path],
    cache: Dict[Var, Optional[Tuple[Path, Tuple[Primitive, ...]]]],
) -> None:
    """Emit either a WEIGHT_TO_WEIGHT or NON_TEMPORAL diagnostic.

    Distinguishes a W -> W -> h exclusion (the blocking eqn at the tail
    was another non-gradient-enabled ETP primitive) from a truly
    non-temporal weight (no ETP op blocks the path; hidden states just
    don't depend on this weight).
    """
    if blocking_eqns:
        blocking_primitives = tuple(e.primitive for e in blocking_eqns)
        blocking_paths: List[Optional[Path]] = []
        for be in blocking_eqns:
            # Use the first trainable invar for tracing back to the ParamState.
            trainable_map = _resolve_eqn_trainable_invars(be)
            first_invar = next(iter(trainable_map.values())) if trainable_map else None
            if first_invar is not None:
                bp, _ = _trace_var_to_param(
                    first_invar, producers, invar_to_weight_path, cache=cache,
                )
            else:
                bp = None
            blocking_paths.append(bp)
        emit(
            kind=DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT,
            level=DiagnosticLevel.WARNING,
            message=(
                f'ETP primitive {primitive.name} (weight={weight_path}) '
                f'reaches a hidden state only through another trainable '
                f'ETP primitive ({", ".join(p.name for p in blocking_primitives)}). '
                f'Per the non-parametric-tail invariant this weight is '
                f'excluded from ETP; learn it by BPTT or rewire the '
                f'architecture so its output flows directly into a hidden '
                f'state.'
            ),
            primitive=primitive,
            weight_path=weight_path,
            context={
                'blocking_primitives': blocking_primitives,
                'blocking_weight_paths': tuple(blocking_paths),
            },
        )
    else:
        emit(
            kind=DiagnosticKind.RELATION_EXCLUDED_NON_TEMPORAL,
            level=DiagnosticLevel.WARNING,
            message=(
                f'ETP primitive {primitive.name} (weight={weight_path}) '
                f'has no connected hidden states. It will be treated as '
                f'a non-temporal parameter.'
            ),
            primitive=primitive,
            weight_path=weight_path,
        )


def find_hidden_param_op_relations_from_minfo(
    minfo: ModuleInfo,
    hid_path_to_group: Dict[Path, HiddenGroup],
) -> Sequence[HiddenParamOpRelation]:
    """Find ETP relations from a ``ModuleInfo``.

    Builds a mapping from ALL ``brainstate.ParamState`` invars so that
    plain ``ParamState`` weights used with ETP primitives are recognised.
    """
    invar_to_weight_path: Dict[Var, Path] = {}
    weight_path_to_invars: Dict[Path, List[Var]] = {}
    for invar_tree, st in zip(
        minfo.state_tree_invars, minfo.compiled_model_states
    ):
        if isinstance(st, brainstate.ParamState):
            path = minfo.state_id_to_path[id(st)]
            leaf_invars = [
                v for v in jax.tree.leaves(invar_tree) if isinstance(v, Var)
            ]
            weight_path_to_invars.setdefault(path, leaf_invars)
            for v in leaf_invars:
                invar_to_weight_path[v] = path

    for v, p in minfo.invar_to_weight_path.items():
        invar_to_weight_path.setdefault(v, p)

    return find_hidden_param_op_relations_from_jaxpr(
        jaxpr=minfo.jaxpr,
        invar_to_weight_path=invar_to_weight_path,
        path_to_state=minfo.retrieved_model_states,
        outvar_to_hidden_path=minfo.outvar_to_hidden_path,
        hid_path_to_group=hid_path_to_group,
        weight_path_to_invars=weight_path_to_invars,
    )


def find_hidden_param_op_relations_from_module(
    model: brainstate.nn.Module,
    *model_args,
    **model_kwargs,
) -> Sequence[HiddenParamOpRelation]:
    """Find ETP relations from a model."""
    minfo = extract_module_info(model, *model_args, **model_kwargs)
    hidden_groups, hid_path_to_group = find_hidden_groups_from_minfo(minfo)
    return find_hidden_param_op_relations_from_minfo(minfo, hid_path_to_group)
