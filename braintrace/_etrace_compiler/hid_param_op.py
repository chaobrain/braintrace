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

import warnings
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
    is_etp_primitive,
)
from braintrace._misc import git_issue_addr
from braintrace._typing import (
    Path,
    HiddenOutVar,
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
) -> Set[Var]:
    """Forward BFS from *start_var* to find reachable hidden-state outvars."""
    reachable: Set[Var] = set()
    frontier = [start_var]
    visited: Set[Var] = set()
    while frontier:
        v = frontier.pop()
        if v in visited:
            continue
        visited.add(v)
        if v in hidden_outvar_set:
            reachable.add(v)
        for eqn in consumer_map.get(v, []):
            for ov in eqn.outvars:
                if ov not in visited:
                    frontier.append(ov)
    return reachable


def _build_transition_jaxpr(
    y_var: Var,
    group: HiddenGroup,
    jaxpr: Jaxpr,
    consumer_map: Dict[Var, List[JaxprEqn]],
) -> Jaxpr:
    """Build the sub-Jaxpr mapping y_var → hidden group outputs."""
    # Backward pass: find equations contributing to hidden outvars
    eqns_needed = []
    all_needed_vars = set(group.hidden_outvars)
    for eqn in reversed(jaxpr.eqns):
        if any(ov in all_needed_vars for ov in eqn.outvars):
            eqns_needed.append(eqn)
            for iv in eqn.invars:
                if isinstance(iv, Var):
                    all_needed_vars.add(iv)
    eqns_needed.reverse()

    # Filter: only keep equations reachable from y_var
    reachable_from_y = {y_var}
    selected = []
    for eqn in eqns_needed:
        if any(
            (isinstance(iv, Var) and iv in reachable_from_y)
            for iv in eqn.invars
        ):
            selected.append(eqn)
            for ov in eqn.outvars:
                reachable_from_y.add(ov)

    # Determine const vars
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
                warnings.warn(
                    f'Found ETP primitives inside a nested jit/pjit. '
                    f'This is currently handled by tracing through the outer jaxpr. '
                    f'If you see incorrect results, please avoid wrapping individual '
                    f'ETP calls in jax.jit. Report issues at {git_issue_addr}.',
                    stacklevel=3,
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
    **_ignored,
) -> Sequence[HiddenParamOpRelation]:
    """Find all ETP-primitive-to-hidden-state relations in *jaxpr*.

    This replaces ``JaxprEvalForWeightOpHiddenRelation`` with direct
    primitive type checking (~200 lines vs ~400 lines).
    """
    producers = _build_producer_map(jaxpr)
    consumers = _build_consumer_map(jaxpr)
    hidden_outvar_set = set(outvar_to_hidden_path.keys())

    etp_eqns = _scan_jaxpr_for_etp_eqns(jaxpr)
    relations: List[HiddenParamOpRelation] = []

    for eqn in etp_eqns:
        primitive = eqn.primitive
        y_var = eqn.outvars[0]

        # --- Identify weight variable ---
        if primitive is etp_elemwise_p:
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
            warnings.warn(
                f'ETP primitive {primitive.name} at {eqn} has a weight input '
                f'that could not be traced back to any ParamState. Skipping.',
                stacklevel=2,
            )
            continue

        weight_state = path_to_state[weight_path]

        # --- Find connected hidden states ---
        reachable_hvars = _find_reachable_hidden_outvars(
            y_var, consumers, hidden_outvar_set,
        )

        # Filter by shape compatibility
        connected_paths: List[Path] = []
        for hvar in list(reachable_hvars):
            try:
                jax.numpy.broadcast_shapes(y_var.aval.shape, hvar.aval.shape)
                connected_paths.append(outvar_to_hidden_path[hvar])
            except ValueError:
                warnings.warn(
                    f'ETP op {primitive.name}: weight={weight_path}, '
                    f'y shape={y_var.aval.shape} not broadcastable with '
                    f'hidden shape={hvar.aval.shape} at '
                    f'{outvar_to_hidden_path[hvar]}. Removing connection.',
                    stacklevel=2,
                )
                reachable_hvars.discard(hvar)

        if not connected_paths:
            warnings.warn(
                f'ETP primitive {primitive.name} (weight={weight_path}) '
                f'has no connected hidden states. It will be treated as '
                f'a non-temporal parameter.',
                stacklevel=2,
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
            x_var=x_var,
            y_var=y_var,
            hidden_groups=connected_groups,
            y_to_hidden_group_jaxprs=y_to_hid_jaxprs,
            connected_hidden_paths=connected_paths,
            eqn_params=dict(eqn.params),
        ))

    return tuple(relations)


def find_hidden_param_op_relations_from_minfo(
    minfo: ModuleInfo,
    hid_path_to_group: Dict[Path, HiddenGroup],
) -> Sequence[HiddenParamOpRelation]:
    """Find ETP relations from a ``ModuleInfo``.

    Builds a mapping from ALL ``brainstate.ParamState`` invars so that
    plain ``ParamState`` weights used with ETP primitives are recognised.
    """

    # Build invar → weight_path for ALL ParamState
    invar_to_weight_path: Dict[Var, Path] = {}
    for invar_tree, st in zip(
        minfo.state_tree_invars, minfo.compiled_model_states
    ):
        if isinstance(st, brainstate.ParamState):
            path = minfo.state_id_to_path[id(st)]
            for v in jax.tree.leaves(invar_tree):
                if isinstance(v, Var):
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
