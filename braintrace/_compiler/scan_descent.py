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

.. note:: Single-step (perturbation) — the per-step hidden perturbation is
   added to the descended scan's *carry* outvar. A loss that reads the
   hidden state through the scan's stacked ys instead (e.g.
   ``for_loop(...)[-1]``) bypasses the perturbation; its same-step credit
   is recovered by the output-bypass rewrite (``output_bypass.py``,
   issue #185), so both readouts give the same gradient.

.. note:: Body analysis always runs in the default "without recurrence"
   grouping mode (``include_recurrent_mixing=False``), independent of the
   outer ``compile_etrace_graph(include_recurrent_mixing=...)`` flag: the
   per-substep trace fold consumes diagonal Jacobians, so tracing recurrent
   ETP mixing into a body transition has no consumer.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, TYPE_CHECKING, cast

from braintrace._compatible_imports import (
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Var,
    is_cond_primitive,
    is_scan_primitive,
    is_while_primitive,
    jaxpr_all_invars,
    jaxpr_constvars,
    new_var,
    scan_num_consts_carry,
    scan_params_add_ys,
    shaped_aval,
)
import brainstate
import jax.numpy as jnp

from braintrace._op import is_etp_enable_gradient_primitive, is_etp_primitive
from braintrace._typing import Path
from .canonicalize import ControlFlowPolicy
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit
from .jaxpr_graph import _body_has_etp, _is_etp_relevant, _sub_closed_jaxprs  # noqa: F401  (re-exported)

if TYPE_CHECKING:
    from .hid_param_op import HiddenParamOpRelation
    from .hidden_group import HiddenGroup
    from .module_info import ModuleInfo

__all__ = [
    'GroupDescent',
    'LiftTemplate',
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
    ``scan.stacked_var_map`` (stacked over the substep axis). A ``lifted``
    relation keeps an outer ``x_var`` and ``y_var`` (see :attr:`lifted`).
    """

    scan: ScanDescentInfo

    lifted: bool = False
    """``True`` for a relation lifted from outside the scan (issue #211): its
    ETP primitive runs once per outer step, before the scan, and the scan
    reads a function of its output ``y`` as a loop constant. ``x_var`` and
    ``y_var`` are then **outer** vars, constant over the substeps (read
    unstacked); each ``y_to_hidden_group_jaxprs`` entry takes the outer ``y``
    as its invar, recomputes the loop constant from it, and continues in the
    body, so its constvars are body vars read through
    ``scan.stacked_var_map``."""


class LiftTemplate(NamedTuple):
    """The body side of an outer relation whose output the scan reads.

    Built by :func:`analyze_and_rewrite_scan` for each scan const computed
    from the output ``y`` of one top-level ETP primitive with a trainable
    input, by outer equations that read nothing else (issue #211); the
    compiler turns the outer relation into a lifted one with it.
    """

    const_var: Var
    """The outer scan operand computed from ``y`` (``y`` itself when the
    scan reads it unchanged)."""

    hidden_groups: List['HiddenGroup']
    """The body groups the constant reaches without crossing a hidden value
    or another trainable ETP primitive (re-scoped like the body's groups)."""

    y_to_hidden_group_jaxprs: List[Jaxpr]
    """Transition jaxprs from the outer ``y`` to each group: the outer
    equations from ``y`` to the constant, then the body's."""

    connected_hidden_paths: List[Path]
    """The carried hidden paths the constant reaches, in reach order."""

    path_classification: Dict[Path, str]
    """``{path: PathClassification.*}`` of the constant's paths in the body;
    the compiler reports the ``mixed`` ones when it lifts the relation."""

    blocking_eqns: Tuple[JaxprEqn, ...]
    """Body ETP primitives that cut the constant's path to the carry;
    reported when no group is reached."""

    body_producers: Dict[Var, JaxprEqn]
    """The body's producer map (to name the blocking primitives' weights)."""

    body_invar_to_weight_path: Dict[Var, Path]
    """Body invar -> weight path (to name the blocking primitives' weights)."""


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
    num_consts, num_carry = scan_num_consts_carry(eqn)
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
        stacked vars; the scan's input structure (consts/carry/xs) and
        ``length`` are unchanged — only the ys/output arity grows — and the
        body's eqns/invars keep their identity. On JAX 0.11 the output flattree
        (``ft_out``) is grown to match via :func:`scan_params_add_ys`.
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
        v: new_var('', shaped_aval(v).update(shape=(length, *shaped_aval(v).shape)))
        for v in extra
    }
    new_params = scan_params_add_ys(
        {**eqn.params, 'jaxpr': new_closed}, len(extra)
    )
    new_eqn = eqn.replace(
        params=new_params,
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

    lift_templates: Dict[Var, LiftTemplate]
    """Outer ETP output ``y`` the scan reads, through a loop constant
    computed from it alone -> the body side of its relation (issue #211).
    Empty when the scan reads no such value."""


def _const_from_etp_output(
    var: Var,
    minfo: 'ModuleInfo',
    producers: Dict[Var, JaxprEqn],
) -> Optional[Tuple[Var, List[JaxprEqn]]]:
    """``(y, eqns)`` when *var* is a function of the output ``y`` of one
    top-level ETP primitive with a trainable input and of nothing else.

    ``eqns`` are the outer equations between ``y`` and *var*, in jaxpr
    order; they read only ``y``, each other's outputs and literals. ``None``
    when *var* also depends on an outer input, a constant or another ETP
    primitive's output.
    """
    from braintrace._op import get_trainable_invars
    from .hid_param_op import _trace_var_to_param

    leaves: Set[Var] = set()
    selected: Set[int] = set()
    seen: Set[Var] = set()
    stack = [var]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        producer = producers.get(v)
        if producer is None:
            return None  # an outer input or constant
        if is_etp_primitive(producer.primitive):
            leaves.add(v)
            continue
        selected.add(id(producer))
        stack.extend(iv for iv in producer.invars if isinstance(iv, Var))
    if len(leaves) != 1:
        return None
    y = next(iter(leaves))
    producer = producers[y]
    trainable = get_trainable_invars(producer.primitive, producer.params)
    if not any(
        isinstance(producer.invars[i], Var)
        and _trace_var_to_param(producer.invars[i], producers, minfo.invar_to_weight_path)[0] is not None
        for i in trainable.values()
    ):
        return None
    return y, [e for e in minfo.jaxpr.eqns if id(e) in selected]


def _outer_relation_consts(
    eqn: JaxprEqn,
    minfo: 'ModuleInfo',
    num_consts: int,
    producers: Dict[Var, JaxprEqn],
) -> Dict[int, Tuple[Var, Var, List[JaxprEqn]]]:
    """Scan const positions computed from the output ``y`` of an outer
    relation alone: position -> ``(const, y, eqns from y to the const)``.

    A value passed in more than one operand position is left out: the
    compiler then reports the relation (a lifted relation has one constant).
    """
    operands = [v for v in eqn.invars if isinstance(v, Var)]
    found: Dict[int, Tuple[Var, Var, List[JaxprEqn]]] = {}
    for c in range(num_consts):
        v = eqn.invars[c]
        if not isinstance(v, Var) or operands.count(v) != 1:
            continue
        hit = _const_from_etp_output(v, minfo, producers)
        if hit is not None:
            found[c] = (v, *hit)
    return found


def _prepend_outer_eqns(
    y: Var,
    const_var: Var,
    body_const: Var,
    prefix: List[JaxprEqn],
    jaxpr: Jaxpr,
) -> Jaxpr:
    """Extend the body transition *jaxpr* (invar *body_const*) back to *y*.

    The outer *prefix* computes *const_var* from *y*; its producing equation
    is re-pointed to write *body_const*, which the body equations read.
    """
    if not prefix:
        return jaxpr  # the scan reads ``y`` itself: bound positionally
    eqns = [
        e.replace(outvars=[body_const if o is const_var else o for o in e.outvars])
        if any(o is const_var for o in e.outvars) else e
        for e in prefix
    ]
    return Jaxpr(
        constvars=jaxpr_constvars(jaxpr, 1),
        invars=[y],
        outvars=list(jaxpr.outvars),
        eqns=eqns + list(jaxpr.eqns),
        debug_info=jaxpr.debug_info,
    )


def _lift_template(
    y: Var,
    const_var: Var,
    prefix: List[JaxprEqn],
    body_const: Var,
    body: Jaxpr,
    body_outvar_to_hidden_path: Dict[Var, Path],
    body_hid_path_to_group: Dict[Path, 'HiddenGroup'],
    path_to_state: Dict[Path, brainstate.State],
    body_invar_to_weight_path: Dict[Var, Path],
) -> LiftTemplate:
    """The body side of the outer relation whose output *y* the scan reads,
    through *prefix*, as the body invar *body_const*.

    The same reachability, shape filter and classification as an ETP
    primitive in the body gets (:func:`~.hid_param_op._reach_hidden`),
    started at *body_const* with the shape of *y*; the transition jaxprs are
    then extended back to *y*. Nothing is reported here: the outer relation
    reports its shape mismatches, and the compiler its partial paths when it
    lifts the relation.
    """
    from .hid_param_op import _build_transition_jaxpr, _reach_hidden
    from .jaxpr_graph import build_consumer_map, build_producer_map

    producers = build_producer_map(body)
    reach = _reach_hidden(
        body_const, tuple(shaped_aval(y).shape), build_consumer_map(body), producers,
        body_outvar_to_hidden_path,
        {ov: body_hid_path_to_group[p].index for ov, p in body_outvar_to_hidden_path.items()},
        path_to_state, self_eqn=None,
    )
    groups: List['HiddenGroup'] = []
    for path in reach.paths:
        g = body_hid_path_to_group[path]
        if all(g is not h for h in groups):
            groups.append(g)
    return LiftTemplate(
        const_var=const_var,
        hidden_groups=groups,
        y_to_hidden_group_jaxprs=[
            _prepend_outer_eqns(y, const_var, body_const, prefix,
                                _build_transition_jaxpr(body_const, g, body))
            for g in groups],
        connected_hidden_paths=reach.paths,
        path_classification=reach.classification,
        blocking_eqns=reach.blocking,
        body_producers=producers,
        body_invar_to_weight_path=body_invar_to_weight_path,
    )


def _passes_credit(eqn: JaxprEqn) -> bool:
    """Whether a dependence crosses *eqn* for the eligibility trace.

    It does not cross a ``stop_gradient``, nor an ETP primitive whose value
    the compiler supplies separately: the hidden-group pass treats both as
    boundaries of the hidden-to-hidden transition.
    """
    if eqn.primitive.name == 'stop_gradient':
        return False
    return not (is_etp_primitive(eqn.primitive)
                and not is_etp_enable_gradient_primitive(eqn.primitive))


def _carries_derivative(v: Var) -> bool:
    """Whether a derivative can flow through *v*: only an inexact value carries
    one (a comparison's boolean or an integer has none)."""
    return bool(jnp.issubdtype(shaped_aval(v).dtype, jnp.inexact))


def _credit_closure(
    starts: Sequence[Var],
    consumers: Dict[Var, List[JaxprEqn]],
    feeds: Optional[Dict[Var, List[Var]]] = None,
) -> Set[Var]:
    """The variables that depend on *starts* (included) through equations
    that pass credit (:func:`_passes_credit`), into inexact outputs only.

    *feeds* maps a variable to the variables it becomes next (a carried output
    to its carried input in the next substep); they join the closure, which so
    grows in one pass over the body.
    """
    seen: Set[Var] = set()
    stack = list(starts)
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        if feeds is not None:
            stack.extend(feeds.get(v, ()))
        for e in consumers.get(v, ()):
            if _passes_credit(e):
                stack.extend(o for o in e.outvars
                             if isinstance(o, Var) and _carries_derivative(o))
    return seen


def _credit_coclosure(target: Var, producers: Dict[Var, JaxprEqn]) -> Set[Var]:
    """The variables on which *target* depends in the sense of
    :func:`_credit_closure`: *v* is in the result iff ``target`` is in
    ``_credit_closure([v], consumers)``."""
    seen: Set[Var] = set()
    stack = [target]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        e = producers.get(v)
        if e is not None and _passes_credit(e) and _carries_derivative(v):
            stack.extend(i for i in e.invars if isinstance(i, Var))
    return seen


def _hidden_derived_vars(minfo: 'ModuleInfo') -> Set[Var]:
    """Top-level values that depend on the step-entry hidden state through
    equations that pass credit (the hidden invars included)."""
    from .jaxpr_graph import build_consumer_map
    return _credit_closure(list(minfo.invar_to_hidden_path), build_consumer_map(minfo.jaxpr))


def _untracked_credit(
    eqn: JaxprEqn,
    body: Jaxpr,
    num_consts: int,
    num_carry: int,
    carry_hidden: Dict[int, Path],
    lift_positions: Sequence[int],
    hidden_derived: Set[Var],
) -> List[Tuple[int, List[Path]]]:
    """Scan inputs other than the hidden carries through which credit reaches
    a hidden carry without the substep fold seeing it.

    The fold of descent follows the hidden carries only: per substep it
    applies the body Jacobian of the hidden carries and the injections of
    the relations. Credit also reaches a hidden carry through

    * a constant or scanned input computed from the step's hidden state
      before the scan (the Jacobian treats it as a constant), and
    * a carried value that is not a hidden state (a local accumulator, a
      state transformed after the scan) when it depends, in some substep,
      on a hidden carry, an ETP output, a lifted constant or such an input:
      the fold does not carry it to the next substep.

    Dependence is differentiable dependence: it does not pass into a boolean
    or an integer (:func:`_credit_closure`). One forward closure, grown
    across substeps through the carries, and one backward closure per hidden
    carry keep the check linear in the size of the body.

    Returns ``(body invar position, hidden paths it reaches)`` for each such
    input of an inexact dtype.
    """
    from .jaxpr_graph import build_consumer_map, build_producer_map

    consumers = build_consumer_map(body)
    producers = build_producer_map(body)
    hidden_in = {num_consts + c for c in carry_hidden}
    hidden_out = {cast(Var, body.outvars[c]): p for c, p in carry_hidden.items()
                  if isinstance(body.outvars[c], Var)}
    derived = {j for j, v in enumerate(eqn.invars)
               if j not in hidden_in and isinstance(v, Var) and v in hidden_derived}
    sources = [body.invars[j] for j in sorted(hidden_in | derived | set(lift_positions))]
    sources += [o for e in body.eqns if is_etp_primitive(e.primitive)
                for o in e.outvars if isinstance(o, Var)]
    # a carried value reached in one substep carries it to the next
    feeds: Dict[Var, List[Var]] = {}
    for c in range(num_carry):
        out = body.outvars[c]
        if c not in carry_hidden and isinstance(out, Var):
            feeds.setdefault(out, []).append(body.invars[num_consts + c])
    reach = _credit_closure(sources, consumers, feeds)
    carried = {c for c in range(num_carry)
               if c not in carry_hidden and body.invars[num_consts + c] in reach}
    reaches = {o: _credit_coclosure(o, producers) for o in hidden_out}
    found = []
    for j in sorted(derived | {num_consts + c for c in carried}):
        v = body.invars[j]
        if not _carries_derivative(v):
            continue
        hit = [p for o, p in hidden_out.items() if v in reaches[o]]
        if hit:
            found.append((j, hit))
    return found


def _untracked_credit_message(
    found: List[Tuple[int, List[Path]]],
    body: Jaxpr,
    num_consts: int,
    num_carry: int,
    length: int,
) -> str:
    """The refusal for :func:`_untracked_credit` findings."""
    parts = []
    for j, _ in found:
        aval = shaped_aval(body.invars[j])
        if j < num_consts:
            what = f'the scan constant at position {j}, computed from the hidden state before the scan'
        elif j < num_consts + num_carry:
            what = f'the carried value at carry position {j - num_consts}'
        else:
            what = (f'the scanned input at position {j - num_consts - num_carry}, computed '
                    f'from the hidden state before the scan')
        parts.append(f'{what} (shape {tuple(aval.shape)}, {aval.dtype})')
    paths = list(dict.fromkeys(p for _, hit in found for p in hit))
    return (
        f'Structured scan descent cannot follow this length-{length} scan: '
        f'{"; ".join(parts)} {"is" if len(parts) == 1 else "are"} not a hidden state carried '
        f'by the scan, yet {"it carries" if len(parts) == 1 else "they carry"} a dependence on '
        f'the hidden state or on an ETP operation\'s output into the hidden state(s) {paths} '
        f'that the scan carries. The eligibility trace is folded over the hidden carries only, '
        f'so the online gradient would leave out the credit that flows through this value. '
        f'Carry the value as a '
        f'hidden state of this scan (merge chained scans into one, keep a value that persists '
        f'across steps in a HiddenState, apply any post-processing of a carried state inside '
        f'the body), or unroll the scan with ControlFlowPolicy(scan_unroll_limit={length}).'
    )


def analyze_and_rewrite_scan(
    eqn: JaxprEqn,
    minfo: 'ModuleInfo',
    producers: Optional[Dict[Var, JaxprEqn]] = None,
    hidden_derived: Optional[Set[Var]] = None,
) -> Optional[ScanDescentBundle]:
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
    producers : dict, optional
        The producer map of ``minfo.jaxpr`` (built when not given); it finds
        the outer ETP outputs the scan reads as loop constants.
    hidden_derived : set of Var, optional
        The top-level values that depend on the step-entry hidden state
        (:func:`_hidden_derived_vars`, computed when not given).

    Returns
    -------
    ScanDescentBundle or None
        ``None`` when the scan carries no hidden state (nothing to descend).

    Raises
    ------
    NotImplementedError
        When credit reaches a hidden carry through another scan input or
        carried value (:func:`_untracked_credit`): the fold would miss it.
    """
    from .hid_param_op import find_hidden_param_op_relations_from_jaxpr
    from .hidden_group import find_hidden_groups_from_jaxpr
    from .jaxpr_graph import inline_jit_calls

    policy = minfo.control_flow
    body_closed = inline_jit_calls(eqn.params['jaxpr'])
    body = body_closed.jaxpr
    num_consts, num_carry = scan_num_consts_carry(eqn)

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
                    f'unroll the scan with '
                    f"ControlFlowPolicy(scan_unroll_limit={eqn.params['length']}) "
                    f'(compile time grows with the length).'
                ),
                hidden_paths=(path,),
            )
            return None

    # ---- outer relations the scan reads as a loop constant (#211) ----------
    if producers is None:
        from .jaxpr_graph import build_producer_map
        producers = build_producer_map(minfo.jaxpr)
    lift_consts = _outer_relation_consts(eqn, minfo, num_consts, producers)

    # ---- credit the fold cannot follow --------------------------------------
    if hidden_derived is None:
        hidden_derived = _hidden_derived_vars(minfo)
    untracked = _untracked_credit(eqn, body, num_consts, num_carry, carry_hidden,
                                  list(lift_consts), hidden_derived)
    if untracked:
        raise NotImplementedError(_untracked_credit_message(
            untracked, body, num_consts, num_carry, eqn.params['length']))

    # ---- body-scope maps ---------------------------------------------------
    body_invar_to_hidden_path = {
        body.invars[num_consts + c]: p for c, p in carry_hidden.items()
    }
    # (typed as variables: a hidden carry output is used as a variable key)
    body_outvars = cast(List[Var], list(body.outvars))
    body_outvar_to_hidden_path = {
        body_outvars[c]: p for c, p in carry_hidden.items()
    }
    body_hidden_outvar_to_invar = {
        body_outvars[c]: body.invars[num_consts + c]
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
                f'Weight state(s) {missing} are read inside a descended scan '
                f'only through plain ops, not ETP primitives, so that use '
                f'gives them no online gradient (a use through an ETP '
                f'primitive elsewhere, such as an outer projection the scan '
                f'reads, still learns). Route the use through an ETP op '
                f'(e.g. braintrace.matmul) inside the scan body if it should '
                f'learn, or ignore this warning if the exclusion is '
                f'deliberate.'
            ),
            context={'weight_paths': tuple(missing)},
        )

    # ---- the lift templates (#211) ------------------------------------------
    outputs = [outer_y for _, outer_y, _ in lift_consts.values()]
    templates: Dict[Var, LiftTemplate] = {
        # an output read through two constants gets none: the compiler then
        # reports its relation
        outer_y: _lift_template(
            outer_y, const, prefix, body.invars[c], body,
            body_outvar_to_hidden_path, body_hid_path_to_group, path_to_state,
            body_invar_to_weight_path)
        for c, (const, outer_y, prefix) in lift_consts.items()
        if outputs.count(outer_y) == 1
    }

    # ---- assemble the hoist list (ordered dedup, body vars) ----------------
    hoist_seen: Dict[Var, None] = {}
    for t in templates.values():
        # the invar is the outer ``y``; only the body constvars are stacked
        for j in t.y_to_hidden_group_jaxprs:
            for v in jaxpr_constvars(j, 1):
                hoist_seen[v] = None
    for r in body_relations:
        if r.x_var is not None:
            hoist_seen[r.x_var] = None
        for j in r.y_to_hidden_group_jaxprs:
            for v in jaxpr_all_invars(j):
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
        # the v1 guard above checked every hidden carry init is a ``Var``
        outer_invars = [cast(Var, eqn.invars[num_consts + path_to_carry[p]])
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

    final_templates = {
        outer_y: t._replace(hidden_groups=[by_paths[tuple(g.hidden_paths)]
                                           for g in t.hidden_groups])
        for outer_y, t in templates.items()
    }

    return ScanDescentBundle(
        info=info,
        new_eqn=new_eqn,
        groups=final_groups,
        relations=final_relations,
        stacked_outer_vars=[stacked[v] for v in hoist],
        lift_templates=final_templates,
    )


def apply_scan_descent(minfo: 'ModuleInfo') -> Tuple['ModuleInfo', List[ScanDescentBundle]]:
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

    from .jaxpr_graph import build_producer_map

    weight_invars = set(minfo.weight_invars)
    # built at the first descendable scan: most modules have none
    producers: Optional[Dict[Var, JaxprEqn]] = None
    hidden_derived: Optional[Set[Var]] = None
    bundles: List[ScanDescentBundle] = []
    new_eqns: List[JaxprEqn] = []
    for eqn in minfo.jaxpr.eqns:
        if not (
            is_scan_primitive(eqn)
            and _is_etp_relevant(eqn)
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
        if producers is None:
            producers = build_producer_map(minfo.jaxpr)
            hidden_derived = _hidden_derived_vars(minfo)
        bundle = analyze_and_rewrite_scan(eqn, minfo, producers, hidden_derived)
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
