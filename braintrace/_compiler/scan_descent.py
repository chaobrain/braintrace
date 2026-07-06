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

"""Structured scan descent (control-flow support, Phase 4).

An ETP-relevant inner ``scan`` whose static length exceeds
``ControlFlowPolicy.scan_unroll_limit`` used to be a dead end: the unroller
skipped it and the downstream walkers raised ``NotImplementedError``. This
pass gives such scans a third compile path — *descent*: the scan body is
analyzed in place with the same hidden-group / relation finders that run on
the flat top-level jaxpr, and the scan equation is rebuilt so the body emits
every value those analyses need (relation inputs, pre-activation ``y``
values, substep-entry hidden states, transition constants) as extra stacked
``ys`` with a leading substep axis. The scan itself stays a single equation
— compile size is independent of its length.

Semantically, descent applies the eligibility-trace recurrence at *substep*
granularity ("unrolling in time"): each substep contributes an injection
``x_tau (x) df_tau`` and a body-scoped diagonal Jacobian ``D_tau``, and the
algorithm folds ``eps <- D_tau * eps + x_tau (x) df_tau`` over the stacked
axis instead of once per outer step. For bodies whose hidden-to-hidden path
is elementwise (the SNN class), the folded trace equals the sum of the
per-relation traces the unroll path would produce — exactly. Hidden groups
produced here face outward through the scan's carry variables, so
perturbation, seam checks, and learning signals are untouched; only the
Jacobian extraction and the trace update see the substep axis.

.. note:: Single-step (perturbation) limitation — the per-step hidden
   perturbation is added to the descended scan's *carry* outvar. A loss
   that reads the hidden state through the scan's stacked ys instead
   (e.g. ``for_loop(...)[-1]``) bypasses the perturbation, so its
   same-step reverse credit is dropped from the learning signal (the
   one-step ETP gradient is zero). Read the state after the loop
   (``self.h.value``) to route the output through the perturbed carry.
   Multi-step VJP is unaffected. This parallels the Phase-3 while-hidden
   same-step limitation.

.. note:: Body analysis always runs in the default "without recurrence"
   grouping mode (``include_recurrent_mixing=False``), independent of the
   outer ``compile_etrace_graph(include_recurrent_mixing=...)`` flag: the
   per-substep trace fold consumes diagonal Jacobians, so tracing recurrent
   ETP mixing into a body transition has no consumer.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from braintrace._compatible_imports import (
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Var,
    is_cond_primitive,
    is_scan_primitive,
    is_while_primitive,
    new_var,
)
from braintrace._op import is_etp_primitive
from braintrace._typing import Path
from .canonicalize import ControlFlowPolicy
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit
from .jaxpr_graph import _sub_closed_jaxprs

if TYPE_CHECKING:
    from .hid_param_op import HiddenParamOpRelation
    from .hidden_group import HiddenGroup
    from .module_info import ModuleInfo

__all__ = [
    'GroupDescent',
    'RelationDescent',
    'ScanDescentBundle',
    'ScanDescentInfo',
    'add_scan_ys',
    'analyze_and_rewrite_scan',
    'apply_scan_descent',
]


class ScanDescentInfo(NamedTuple):
    """Compile-time context for one descended scan equation.

    All ``Var``s in ``stacked_var_map`` keys (and ``body_jaxpr``) are
    **body**-scoped; the map's values are fresh **outer**-scope vars produced
    by :func:`add_scan_ys`, hoisted as jaxpr outputs so the executor can read
    their runtime values (leading substep axis ``length``).
    """

    length: int
    """Static trip count ``L`` of the scan."""

    num_consts: int
    """Number of const entries at the head of the scan invars."""

    num_carry: int
    """Number of carry entries following the consts."""

    body_jaxpr: Jaxpr
    """The (jit-inlined) scan body; ``invars = [*consts, *carry, *xs]``."""

    stacked_var_map: Dict[Var, Var]
    """Body var -> outer stacked var of aval ``(length, *shape)``;
    insertion-ordered."""

    scan_eqn_id: int
    """``id()`` of the REWRITTEN scan eqn — the key the outer walkers use to
    exempt this equation."""


class GroupDescent(NamedTuple):
    """Descent context attached to a :class:`~.hidden_group.HiddenGroup`.

    The group's ``transition_jaxpr``/``transition_jaxpr_constvars`` are
    **body**-scoped (one substep) while its ``hidden_invars``/
    ``hidden_outvars`` are re-scoped to the **outer** scan carry vars;
    ``body_hidden_invars`` preserves the body-scope carry invars (substep-
    entry hidden values), matching ``transition_jaxpr.invars``.
    """

    scan: ScanDescentInfo
    body_hidden_invars: List[Var]


class RelationDescent(NamedTuple):
    """Descent context attached to a
    :class:`~.hid_param_op.HiddenParamOpRelation`.

    The relation's ``x_var``/``y_var``/``y_to_hidden_group_jaxprs`` are
    **body**-scoped; runtime values are read through
    ``scan.stacked_var_map`` (stacked over the substep axis).
    """

    scan: ScanDescentInfo


def _body_has_etp(jaxpr: Jaxpr) -> bool:
    """``True`` when an ETP primitive appears in ``jaxpr`` or any nested
    control-flow body inside it."""
    for eqn in jaxpr.eqns:
        if is_etp_primitive(eqn.primitive):
            return True
        for sub in _sub_closed_jaxprs(eqn):
            if _body_has_etp(sub.jaxpr):
                return True
    return False


def _is_etp_relevant(body_jaxpr: Jaxpr, eqn: JaxprEqn, weight_invars: Set[Var]) -> bool:
    """Whether a scan matters to descent at all.

    Mirrors the unroller's relevance rule (:mod:`.canonicalize`): a scan is
    ETP-relevant iff it consumes a trainable weight invar or its body holds
    an ETP primitive.

    Parameters
    ----------
    body_jaxpr : Jaxpr
        The scan's body jaxpr (``eqn.params['jaxpr'].jaxpr``).
    eqn : JaxprEqn
        The scan equation.
    weight_invars : set of Var
        Top-level weight invars of the enclosing module jaxpr.

    Returns
    -------
    bool
        ``True`` when descent (or unrolling) should consider this scan.
    """
    if any(isinstance(v, Var) and v in weight_invars for v in eqn.invars):
        return True
    return _body_has_etp(body_jaxpr)


def _descent_blockers(
    eqn: JaxprEqn,
    policy: ControlFlowPolicy,
    weight_invars: Set[Var],
) -> Optional[str]:
    """Check the v1 descendability restrictions for one scan equation.

    Parameters
    ----------
    eqn : JaxprEqn
        The (ETP-relevant) scan equation.
    policy : ControlFlowPolicy
        The active control-flow policy.
    weight_invars : set of Var
        Top-level weight invars of the enclosing module jaxpr.

    Returns
    -------
    str or None
        ``None`` when the scan is descendable; otherwise a human-readable
        reason suitable for a ``SCAN_DESCENT_SKIPPED`` diagnostic message.
    """
    if policy.scan_descent != 'auto':
        return "ControlFlowPolicy.scan_descent is 'off'"
    length = eqn.params['length']
    if length <= policy.scan_unroll_limit:
        return (f'its length {length} is within scan_unroll_limit='
                f'{policy.scan_unroll_limit}; the unroll path handles it')
    if eqn.params.get('reverse', False):
        return 'reverse scans are not supported by structured descent (v1)'
    body = eqn.params['jaxpr'].jaxpr
    for e in body.eqns:
        if is_scan_primitive(e) or is_while_primitive(e) or is_cond_primitive(e):
            return ('its body contains nested control flow '
                    f'({e.primitive.name}); not supported by descent (v1)')
    num_consts = eqn.params['num_consts']
    num_carry = eqn.params['num_carry']
    for v in eqn.invars[num_consts + num_carry:]:
        if isinstance(v, Var) and v in weight_invars:
            return ('a trainable weight is scanned over as xs (per-iteration '
                    'weight slices); not supported by descent')
    return None


def add_scan_ys(
    eqn: JaxprEqn,
    extra_body_outvars: Sequence[Var],
) -> Tuple[JaxprEqn, Dict[Var, Var]]:
    """Rebuild a scan eqn so its body also emits ``extra_body_outvars`` as ys.

    Parameters
    ----------
    eqn : JaxprEqn
        A ``scan`` equation (``is_scan_primitive(eqn)`` must hold).
    extra_body_outvars : Sequence[Var]
        Body-scope vars (computed vars or body invars — passthrough ys are
        valid jaxpr) to stack over the trip axis. Deduplicated preserving
        order; vars already among the body outvars are still appended as new
        ys so the stacked outer var is dedicated.

    Returns
    -------
    tuple
        ``(new_eqn, stacked_var_map)`` where ``stacked_var_map[body_var]`` is
        a fresh outer ``Var`` of aval ``(length, *body_var.aval.shape)``. The
        new eqn keeps the original outvars (by identity) followed by the
        stacked vars; ``num_consts``/``num_carry``/``length``/``linear`` are
        unchanged, and the body's eqns/invars keep their identity.
    """
    body_closed = eqn.params['jaxpr']
    body = body_closed.jaxpr
    length = eqn.params['length']
    extra = list(dict.fromkeys(extra_body_outvars))
    new_body = Jaxpr(
        constvars=body.constvars,
        invars=body.invars,
        outvars=list(body.outvars) + extra,
        eqns=body.eqns,
        effects=body.effects,
        debug_info=body.debug_info,
    )
    new_closed = ClosedJaxpr(new_body, body_closed.consts)
    stacked = {
        v: new_var('', v.aval.update(shape=(length, *v.aval.shape)))
        for v in extra
    }
    new_eqn = eqn.replace(
        params={**eqn.params, 'jaxpr': new_closed},
        outvars=list(eqn.outvars) + [stacked[v] for v in extra],
    )
    return new_eqn, stacked


class ScanDescentBundle(NamedTuple):
    """Everything :func:`analyze_and_rewrite_scan` produced for one scan.

    ``groups``/``relations`` carry their descent context
    (:class:`GroupDescent`/:class:`RelationDescent`); group ``index`` values
    are body-local (0-based) and are re-assigned when the bundle is merged
    into the outer graph.
    """

    info: ScanDescentInfo
    new_eqn: JaxprEqn
    groups: List['HiddenGroup']
    relations: List['HiddenParamOpRelation']
    stacked_outer_vars: List[Var]
    """Ordered values of ``info.stacked_var_map`` — the fresh outer vars to
    hoist as jaxpr outputs."""


def analyze_and_rewrite_scan(eqn: JaxprEqn, minfo) -> Optional[ScanDescentBundle]:
    """Analyze one descendable scan and rebuild it with stacked ys.

    Runs the flat hidden-group / relation finders on the scan *body* (with
    body-scoped hidden/weight maps derived from the carry/const positions),
    hoists every body value the executor will need as extra stacked ys via
    :func:`add_scan_ys`, and re-scopes the discovered groups so their
    ``hidden_invars``/``hidden_outvars`` are the scan's outer carry vars.

    Parameters
    ----------
    eqn : JaxprEqn
        A scan equation for which :func:`_descent_blockers` returned ``None``.
    minfo : ModuleInfo
        The enclosing module info (supplies the outer hidden/weight maps).

    Returns
    -------
    ScanDescentBundle or None
        ``None`` when the scan carries no hidden state (nothing to descend).
    """
    from .hid_param_op import find_hidden_param_op_relations_from_jaxpr
    from .hidden_group import find_hidden_groups_from_jaxpr
    from .jaxpr_graph import inline_jit_calls

    policy = minfo.control_flow
    body_closed = inline_jit_calls(eqn.params['jaxpr'])
    body = body_closed.jaxpr
    num_consts = eqn.params['num_consts']
    num_carry = eqn.params['num_carry']

    # ---- outer<->body position maps ---------------------------------------
    # scan invars [*consts, *carry, *xs] are positionally identical to
    # body.invars; outvars are [*carry, *ys].
    carry_hidden: Dict[int, Path] = {}
    for c in range(num_carry):
        path = minfo.outvar_to_hidden_path.get(eqn.outvars[c])
        if path is not None:
            carry_hidden[c] = path
    if not carry_hidden:
        return None

    # v1 guard: every hidden carry must be initialized from the *pristine*
    # step-entry hidden invar. If the model transforms the hidden state
    # between step entry and the scan (e.g. ``h.value = h.value * 0.5``
    # before the loop), that outer segment of the per-step transition is
    # invisible to the substep fold and the folded eligibility trace would
    # be silently wrong. Skip descent; the scan then hits the existing loud
    # control-flow restrictions.
    for c, path in carry_hidden.items():
        init_invar = eqn.invars[num_consts + c]
        if not (
            isinstance(init_invar, Var)
            and minfo.invar_to_hidden_path.get(init_invar) == path
        ):
            emit(
                kind=DiagnosticKind.SCAN_DESCENT_SKIPPED,
                level=DiagnosticLevel.WARNING,
                message=(
                    f'Structured scan descent was skipped for a scan '
                    f'carrying hidden state {path}: the carry is initialized '
                    f'from a transformed value, not the step-entry hidden '
                    f'state, so part of the per-step transition lies outside '
                    f'the scan and the folded eligibility trace would be '
                    f'wrong. Move the pre-scan update into the scan body, or '
                    f"set ControlFlowPolicy(scan_descent='off')."
                ),
                hidden_paths=(path,),
            )
            return None

    # ---- body-scope maps ---------------------------------------------------
    body_invar_to_hidden_path = {
        body.invars[num_consts + c]: p for c, p in carry_hidden.items()
    }
    body_outvar_to_hidden_path = {
        body.outvars[c]: p for c, p in carry_hidden.items()
    }
    body_hidden_outvar_to_invar = {
        body.outvars[c]: body.invars[num_consts + c]
        for c in carry_hidden
    }
    body_invar_to_weight_path: Dict[Var, Path] = {}
    body_weight_path_to_invars: Dict[Path, List[Var]] = {}
    for bv, ov in zip(body.invars, eqn.invars):
        if not isinstance(ov, Var):
            continue
        wp = minfo.invar_to_weight_path.get(ov)
        if wp is not None:
            body_invar_to_weight_path[bv] = wp
            body_weight_path_to_invars.setdefault(wp, []).append(bv)
    path_to_state = minfo.retrieved_model_states

    # ---- reuse the flat finders on the body --------------------------------
    body_groups, body_hid_path_to_group = find_hidden_groups_from_jaxpr(
        body,
        hidden_outvar_to_invar=body_hidden_outvar_to_invar,
        weight_invars=set(body_invar_to_weight_path),
        invar_to_hidden_path=body_invar_to_hidden_path,
        outvar_to_hidden_path=body_outvar_to_hidden_path,
        path_to_state=path_to_state,
        include_recurrent_mixing=False,
        control_flow=policy,
    )
    body_relations = find_hidden_param_op_relations_from_jaxpr(
        body,
        invar_to_weight_path=body_invar_to_weight_path,
        path_to_state=path_to_state,
        outvar_to_hidden_path=body_outvar_to_hidden_path,
        hid_path_to_group=body_hid_path_to_group,
        weight_path_to_invars=body_weight_path_to_invars,
        control_flow=policy,
    )

    # A weight consumed by this scan but routed through no ETP primitive
    # (plain-op usage) does not learn online — the primitive-based selection
    # principle, same as at the flat level. Pre-descent this whole scan was
    # a hard error, so surface the flip loudly instead of silently.
    covered_paths = {
        wp for r in body_relations for wp in r.trainable_paths.values()
    }
    missing = [wp for wp in body_weight_path_to_invars
               if wp not in covered_paths]
    if missing:
        emit(
            kind=DiagnosticKind.SCAN_DESCENT_NO_RELATIONS,
            level=DiagnosticLevel.WARNING,
            message=(
                f'Weight state(s) {missing} are consumed by a descended '
                f'scan but produce no ETP relation (they are used through '
                f'plain ops, not ETP primitives), so they will NOT '
                f'participate in online learning. Route them through an ETP '
                f'op (e.g. braintrace.matmul) inside the scan body if they '
                f'should learn, or ignore this warning if the exclusion is '
                f'deliberate.'
            ),
            context={'weight_paths': tuple(missing)},
        )

    # ---- assemble the hoist list (ordered dedup, body vars) ----------------
    hoist_seen: Dict[Var, None] = {}
    for r in body_relations:
        if r.x_var is not None:
            hoist_seen[r.x_var] = None
        for j in r.y_to_hidden_group_jaxprs:
            for v in list(j.invars) + list(j.constvars):
                hoist_seen[v] = None
    for g in body_groups:
        for v in list(g.hidden_invars) + list(g.transition_jaxpr_constvars):
            hoist_seen[v] = None
    hoist: List[Var] = list(hoist_seen)

    # ---- rewrite the eqn ----------------------------------------------------
    inlined_eqn = eqn if body_closed is eqn.params['jaxpr'] else eqn.replace(
        params={**eqn.params, 'jaxpr': body_closed})
    new_eqn, stacked = add_scan_ys(inlined_eqn, hoist)
    info = ScanDescentInfo(
        length=eqn.params['length'],
        num_consts=num_consts,
        num_carry=num_carry,
        body_jaxpr=body,
        stacked_var_map=stacked,
        scan_eqn_id=id(new_eqn),
    )

    # ---- re-scope groups to outer hidden vars -------------------------------
    path_to_carry = {p: c for c, p in carry_hidden.items()}
    final_groups = []
    for g in body_groups:
        outer_invars = [eqn.invars[num_consts + path_to_carry[p]]
                        for p in g.hidden_paths]
        outer_outvars = [eqn.outvars[path_to_carry[p]] for p in g.hidden_paths]
        final_groups.append(g._replace(
            hidden_invars=outer_invars,
            hidden_outvars=outer_outvars,
            descent=GroupDescent(scan=info,
                                 body_hidden_invars=list(g.hidden_invars)),
        ))

    # ---- patch relations: point at final groups + attach context ------------
    by_paths = {tuple(g.hidden_paths): g for g in final_groups}
    final_relations = [
        r._replace(
            hidden_groups=[by_paths[tuple(g.hidden_paths)]
                           for g in r.hidden_groups],
            control_flow_context=RelationDescent(scan=info),
        )
        for r in body_relations
    ]

    return ScanDescentBundle(
        info=info,
        new_eqn=new_eqn,
        groups=final_groups,
        relations=final_relations,
        stacked_outer_vars=[stacked[v] for v in hoist],
    )


def apply_scan_descent(minfo) -> Tuple['ModuleInfo', List[ScanDescentBundle]]:
    """Descend every eligible top-level scan in ``minfo`` and rewrite its jaxpr.

    Walks the top-level equations of ``minfo.jaxpr``; for each ETP-relevant
    ``scan`` that passes the v1 restrictions (:func:`_descent_blockers`), the
    equation is analyzed and rewritten via :func:`analyze_and_rewrite_scan`
    and a ``SCAN_DESCENT_APPLIED`` diagnostic is emitted. Scans that are
    blocked *and* too long to unroll get a ``SCAN_DESCENT_SKIPPED`` warning
    (short scans are the unroller's business; no diagnostic here).

    Parameters
    ----------
    minfo : ModuleInfo
        The module info produced by ``extract_module_info``.

    Returns
    -------
    tuple
        ``(minfo, bundles)``. When ``minfo.control_flow.scan_descent`` is not
        ``'auto'`` or nothing was descended, the *original* ``minfo`` is
        returned unchanged with an empty bundle list. Otherwise the returned
        ``ModuleInfo`` holds a rebuilt ``closed_jaxpr`` whose descended scan
        equations are the bundles' ``new_eqn`` objects (every other eqn and
        every pre-existing ``Var`` keeps its identity).
    """
    policy: ControlFlowPolicy = minfo.control_flow
    if policy.scan_descent != 'auto':
        return minfo, []

    weight_invars = set(minfo.weight_invars)
    bundles: List[ScanDescentBundle] = []
    new_eqns: List[JaxprEqn] = []
    for eqn in minfo.jaxpr.eqns:
        if not (
            is_scan_primitive(eqn)
            and _is_etp_relevant(eqn.params['jaxpr'].jaxpr, eqn, weight_invars)
        ):
            new_eqns.append(eqn)
            continue
        blocker = _descent_blockers(eqn, policy, weight_invars)
        if blocker is not None:
            if eqn.params['length'] > policy.scan_unroll_limit:
                emit(
                    kind=DiagnosticKind.SCAN_DESCENT_SKIPPED,
                    level=DiagnosticLevel.WARNING,
                    message=(
                        f'Structured scan descent was requested '
                        f"(scan_descent='auto') but this scan cannot be "
                        f'descended: {blocker}. The existing control-flow '
                        f'restrictions apply to it.'
                    ),
                    context={'blocker': blocker,
                             'length': eqn.params['length']},
                )
            new_eqns.append(eqn)
            continue
        bundle = analyze_and_rewrite_scan(eqn, minfo)
        if bundle is None:
            # no hidden state in the carry: nothing to descend; the walkers
            # keep their existing behavior for this equation.
            new_eqns.append(eqn)
            continue
        bundles.append(bundle)
        new_eqns.append(bundle.new_eqn)
        emit(
            kind=DiagnosticKind.SCAN_DESCENT_APPLIED,
            level=DiagnosticLevel.INFO,
            message=(
                f'Applied structured descent to a length-'
                f'{bundle.info.length} scan: {len(bundle.relations)} '
                f'relation(s) and {len(bundle.groups)} hidden group(s) '
                f'registered from its body.'
            ),
            hidden_paths=tuple(
                p for g in bundle.groups for p in g.hidden_paths
            ),
            context={
                'length': bundle.info.length,
                'weight_paths': tuple(sorted({
                    wp for r in bundle.relations
                    for wp in r.trainable_paths.values()
                })),
            },
        )

    if not bundles:
        return minfo, []
    old = minfo.jaxpr
    new_jaxpr = Jaxpr(
        constvars=old.constvars,
        invars=old.invars,
        outvars=old.outvars,
        eqns=new_eqns,
        effects=old.effects,
        debug_info=old.debug_info,
    )
    new_closed = ClosedJaxpr(new_jaxpr, minfo.closed_jaxpr.consts)
    return minfo._replace(closed_jaxpr=new_closed), bundles
