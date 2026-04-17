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

"""
Primitive-based weight-to-hidden-state relation discovery.

Replaces the old name-matching approach (``JaxprEvalForWeightOpHiddenRelation``)
with direct primitive type checking in the Jaxpr. The compiler:

1. Walks the Jaxpr and finds ETP primitives (``eqn.primitive in ETP_PRIMITIVES``).
2. For each primitive, traces the weight invar backward to find its ``ParamState``.
3. Traces forward from the primitive output to find reachable hidden-state outvars.
4. Builds transition Jaxprs (y → h) for each connected hidden group.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Any

import brainstate
import jax

from braintrace._compatible_imports import (
    Primitive,
    Var,
    Literal,
    JaxprEqn,
    Jaxpr,
    is_jit_primitive,
)
from braintrace._etrace_operators import (
    ETP_PRIMITIVES,
    etp_elemwise_p,
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
    'find_hidden_param_op_relations_from_minfo',
    'find_hidden_param_op_relations_from_module',
]


class HiddenParamOpRelation(NamedTuple):
    r"""
    Connection between an ETP primitive, its weight parameter, and hidden states.

    Records the structural relationship:

    .. math::
        h^t = f(y), \quad y = \text{primitive}(x, \theta)

    Attributes:
        primitive: The JAX primitive (``etp_mm_p``, ``etp_mv_p``, etc.).
        weight: The ``ParamState`` object.
        weight_path: Path to the ``ParamState`` in the module hierarchy.
        weight_var: Jaxpr ``Var`` for the weight input of the primitive. When
            the ``ParamState`` stores a PyTree (e.g. ``{'weight': W, 'bias':
            b}``), this identifies which leaf is the weight.
        weight_leaf_idx: Index of ``weight_var`` among ``jax.tree.leaves`` of
            the owning ``ParamState``'s value. Used at runtime to extract the
            weight array from a pytree-valued ``ParamState``.
        x_var: Jaxpr ``Var`` for the input (``None`` for element-wise ops).
        y_var: Jaxpr ``Var`` for the primitive output.
        hidden_groups: Hidden groups that this op feeds into.
        y_to_hidden_group_jaxprs: Transition Jaxpr from *y* to each hidden group.
        connected_hidden_paths: Hidden-state paths connected to this op.
        eqn_params: Static parameters of the primitive equation.
    """
    primitive: Primitive
    weight: brainstate.ParamState
    weight_path: Path
    weight_var: Var
    weight_leaf_idx: int
    x_var: Optional[Var]
    y_var: Var
    hidden_groups: List[HiddenGroup]
    y_to_hidden_group_jaxprs: List[Jaxpr]
    connected_hidden_paths: List[Path]
    eqn_params: dict

    # backward compat aliases
    @property
    def x(self):
        return self.x_var

    @property
    def y(self):
        return self.y_var

    @property
    def path(self):
        return self.weight_path

    def y_to_hidden_groups(self, y_val, const_vals, concat_hidden_vals=True):
        """Evaluate transition jaxprs: y → hidden group values."""
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
# Internal helpers
# ---------------------------------------------------------------------------

def _build_producer_map(jaxpr: Jaxpr) -> Dict[Var, JaxprEqn]:
    """Map each variable to the equation that produces it."""
    producers = {}
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


def _trace_var_to_param(
    var: Var,
    producers: Dict[Var, JaxprEqn],
    invar_to_weight_path: Dict[Var, Path],
) -> Optional[Path]:
    """Trace *var* backward through the Jaxpr to find its originating ``ParamState``.

    Handles weight_fn/mask applied before the primitive — the processed
    weight is traced back to the raw ``ParamState`` invar.
    """
    frontier = [var]
    visited: Set[Var] = set()
    while frontier:
        v = frontier.pop()
        if v in visited:
            continue
        visited.add(v)
        path = invar_to_weight_path.get(v)
        if path is not None:
            return path
        eqn = producers.get(v)
        if eqn is not None:
            for iv in eqn.invars:
                if isinstance(iv, Var) and iv not in visited:
                    frontier.append(iv)
    return None


def _find_reachable_hidden_outvars(
    start_var: Var,
    consumer_map: Dict[Var, List[JaxprEqn]],
    hidden_outvar_set: Set[Var],
    outvar_to_group_index: Optional[Dict[Var, int]] = None,
) -> Tuple[Set[Var], Tuple[JaxprEqn, ...]]:
    """Forward BFS from *start_var* to find reachable hidden-state outvars.

    Returns ``(reachable_hvars, blocking_eqns)`` where ``blocking_eqns`` is the
    tuple (in BFS-encounter order, deduplicated) of consumer equations that
    were NOT crossed because they are non-gradient-enabled ETP primitives.
    The caller uses ``blocking_eqns`` to distinguish a "truly non-temporal"
    weight from one excluded by the ``weight -> weight -> hidden`` rule.

    When ``outvar_to_group_index`` is provided, the search restricts itself to
    hidden outvars in the *closest* hidden group — the first one encountered.
    Outvars from different groups (e.g. hidden states of a downstream layer in
    a stacked model) are pruned so the relation only tracks the recurrence of
    the layer that this primitive actually feeds.

    The BFS stops at any other ETP primitive that is not gradient-enabled:
    crossing such a primitive would encode a ``weight -> weight -> hidden``
    pathway, which ETP does not decompose correctly (the downstream primitive
    already owns the gradient of its input).
    """
    from collections import deque

    reachable: Set[Var] = set()
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
                    # First hidden outvar reached — fix the home group(s).
                    home_group_indices.add(g)
                if g in home_group_indices:
                    reachable.add(v)
                else:
                    # Different hidden group → do not cross further.
                    continue
            else:
                reachable.add(v)
        for eqn in consumer_map.get(v, []):
            # Do not cross another ETP primitive unless it is gradient-enabled
            # (e.g. ``etp_elemwise_p``). This prevents the "weight -> weight
            # -> hidden" pathway from registering a spurious relation — the
            # downstream primitive already owns the gradient of its input.
            if (
                is_etp_primitive(eqn.primitive)
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


def _build_transition_jaxpr(
    y_var: Var,
    group: HiddenGroup,
    jaxpr: Jaxpr,
    consumer_map: Dict[Var, List[JaxprEqn]],
) -> Jaxpr:
    """Build the sub-Jaxpr mapping y_var → hidden group outputs.

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
    # Backward pass: find all equations contributing to hidden outvars
    selected_rev = []
    all_needed_vars = set(group.hidden_outvars)
    for eqn in reversed(jaxpr.eqns):
        if any(ov in all_needed_vars for ov in eqn.outvars):
            if (
                is_etp_primitive(eqn.primitive)
                and not is_etp_enable_gradient_primitive(eqn.primitive)
            ):
                # Another ETP primitive on the tail — stop here. Its output
                # becomes a constvar of this transition jaxpr.
                continue
            selected_rev.append(eqn)
            for iv in eqn.invars:
                if isinstance(iv, Var):
                    all_needed_vars.add(iv)
    selected = list(reversed(selected_rev))

    # Determine const vars: vars used by selected eqns but not produced by
    # them and not the jaxpr's invar (y_var).
    produced = {y_var}
    for eqn in selected:
        for ov in eqn.outvars:
            produced.add(ov)
    all_invars_needed = set()
    for eqn in selected:
        for iv in eqn.invars:
            if isinstance(iv, Var):
                all_invars_needed.add(iv)
    constvars = list(all_invars_needed - produced)

    return Jaxpr(
        constvars=constvars,
        invars=[y_var],
        outvars=list(group.hidden_outvars),
        eqns=selected,
    )


def _scan_jaxpr_for_etp_eqns(jaxpr: Jaxpr) -> List[JaxprEqn]:
    """Walk the Jaxpr and return all equations whose primitive is ETP."""
    etp_eqns: List[JaxprEqn] = []
    for eqn in jaxpr.eqns:
        if is_etp_primitive(eqn.primitive):
            etp_eqns.append(eqn)
        elif is_jit_primitive(eqn) and 'jaxpr' in eqn.params:
            inner_jaxpr = eqn.params['jaxpr'].jaxpr
            inner_etp = _scan_jaxpr_for_etp_eqns(inner_jaxpr)
            if inner_etp:
                emit(
                    kind=DiagnosticKind.PRIMITIVE_INSIDE_NESTED_JIT,
                    level=DiagnosticLevel.WARNING,
                    message=(
                        'Found ETP primitives inside a nested jit/pjit. '
                        'This is currently handled by tracing through the outer jaxpr. '
                        'If you see incorrect results, please avoid wrapping individual '
                        f'ETP calls in jax.jit. Report issues at {git_issue_addr}.'
                    ),
                    context={'inner_primitives': tuple(e.primitive for e in inner_etp)},
                )
    return etp_eqns


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
    """Find all ETP-primitive-to-hidden-state relations in *jaxpr*.

    This replaces ``JaxprEvalForWeightOpHiddenRelation`` with direct
    primitive type checking (~200 lines vs ~400 lines).
    """
    producers = _build_producer_map(jaxpr)
    consumers = _build_consumer_map(jaxpr)
    hidden_outvar_set = set(outvar_to_hidden_path.keys())

    # Build an outvar → group-index lookup so the forward BFS can stop at
    # hidden outvars that belong to a different hidden group (e.g. downstream
    # layer in a stacked model). This keeps each relation scoped to the
    # recurrence of the layer it actually feeds.
    outvar_to_group_index: Dict[Var, int] = {
        ov: hid_path_to_group[p].index
        for ov, p in outvar_to_hidden_path.items()
        if p in hid_path_to_group
    }

    etp_eqns = _scan_jaxpr_for_etp_eqns(jaxpr)
    relations: List[HiddenParamOpRelation] = []

    for eqn in etp_eqns:
        primitive = eqn.primitive
        y_var = eqn.outvars[0]

        # --- Identify weight and x variables from the primitive's spec ---
        # Spec-driven dispatch lets third-party primitives declare their own
        # invar layout without touching the compiler. Legacy primitives
        # registered through the old API fall back to the historical
        # convention (weight at invar[1] unless the primitive is
        # ``etp_elemwise_p``).
        spec = get_primitive_spec(primitive)
        if spec is not None:
            weight_var = eqn.invars[spec.weight_invar_index]
            if spec.x_invar_index is None:
                x_var = None
            else:
                candidate = eqn.invars[spec.x_invar_index]
                x_var = candidate if isinstance(candidate, Var) else None
        elif primitive is etp_elemwise_p:
            weight_var = eqn.invars[0]
            x_var = None
        else:
            x_var = eqn.invars[0] if isinstance(eqn.invars[0], Var) else None
            weight_var = eqn.invars[1]

        # --- Trace weight back to ParamState ---
        weight_path = _trace_var_to_param(
            weight_var, producers, invar_to_weight_path,
        )
        if weight_path is None:
            emit(
                kind=DiagnosticKind.RELATION_EXCLUDED_NO_PARAMSTATE,
                level=DiagnosticLevel.WARNING,
                message=(
                    f'ETP primitive {primitive.name} at {eqn} has a weight input '
                    f'that could not be traced back to any ParamState. Skipping.'
                ),
                primitive=primitive,
                context={'weight_var': weight_var},
            )
            continue

        weight_state = path_to_state[weight_path]

        # --- Find weight_leaf_idx: position of weight_var among the owning
        # ParamState's invar leaves. Required at runtime so callers can pick
        # the correct leaf out of a pytree-valued ParamState.
        weight_leaf_idx = 0
        if weight_path_to_invars is not None:
            invars_list = weight_path_to_invars.get(weight_path, [])
            # The actual weight_var may have been produced by an upstream
            # equation (mask/weight_fn) — trace it back to the ParamState's
            # original invar that feeds the primitive chain.
            source_var = weight_var
            frontier = [weight_var]
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
                weight_leaf_idx = invars_list.index(source_var)
            except ValueError:
                weight_leaf_idx = 0

        # --- Find connected hidden states ---
        reachable_hvars, blocking_eqns = _find_reachable_hidden_outvars(
            y_var, consumers, hidden_outvar_set,
            outvar_to_group_index=outvar_to_group_index,
        )

        # Filter by shape compatibility
        connected_paths: List[Path] = []
        for hvar in list(reachable_hvars):
            try:
                jax.numpy.broadcast_shapes(y_var.aval.shape, hvar.aval.shape)
                connected_paths.append(outvar_to_hidden_path[hvar])
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
                reachable_hvars.discard(hvar)

        if not connected_paths:
            # Distinguish W -> W -> h exclusion (the blocking eqn at the tail
            # was another non-gradient-enabled ETP primitive) from a truly
            # non-temporal weight (no ETP op blocks the path; hidden states
            # just don't depend on this weight).
            if blocking_eqns:
                blocking_primitives = tuple(e.primitive for e in blocking_eqns)
                # Try to recover the *paths* of the blocking weights so the
                # diagnostic names which weight stood in the way. Any that
                # fail to resolve are dropped from the context.
                blocking_paths: List[Optional[Path]] = []
                for be in blocking_eqns:
                    be_spec = get_primitive_spec(be.primitive)
                    if be_spec is not None:
                        wvar = be.invars[be_spec.weight_invar_index]
                    elif be.primitive is etp_elemwise_p:
                        wvar = be.invars[0]
                    else:
                        wvar = be.invars[1]
                    bp = _trace_var_to_param(
                        wvar, producers, invar_to_weight_path,
                    )
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
            continue

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
            _build_transition_jaxpr(y_var, g, jaxpr, consumers)
            for g in connected_groups
        ]

        relations.append(HiddenParamOpRelation(
            primitive=primitive,
            weight=weight_state,
            weight_path=weight_path,
            weight_var=weight_var,
            weight_leaf_idx=weight_leaf_idx,
            x_var=x_var,
            y_var=y_var,
            hidden_groups=connected_groups,
            y_to_hidden_group_jaxprs=y_to_hid_jaxprs,
            connected_hidden_paths=connected_paths,
            eqn_params=dict(eqn.params),
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
            },
        )

    return tuple(relations)


def find_hidden_param_op_relations_from_minfo(
    minfo: ModuleInfo,
    hid_path_to_group: Dict[Path, HiddenGroup],
) -> Sequence[HiddenParamOpRelation]:
    """Find ETP relations from a ``ModuleInfo``.

    Builds a mapping from ALL ``brainstate.ParamState`` invars so that
    plain ``ParamState`` weights used with ETP primitives are recognised.
    """

    # Build invar → weight_path for ALL ParamState, plus an ordered list
    # of leaf invars per path (needed to recover weight_leaf_idx so callers
    # can pick the weight leaf out of a pytree-valued ParamState at runtime).
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

    # Merge with the existing mapping
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
