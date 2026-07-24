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

"""Jaxpr canonicalization passes for the ETrace compiler.

Control-flow constructs hide their bodies behind sub-jaxprs, but every ETrace
analysis identifies weights, hidden states, and ETP primitives by ``Var``
identity in one flat top-level jaxpr. The passes in this module rewrite
ETP-relevant control flow into flat, semantically identical equation
sequences before any analysis runs — the same role
:func:`~braintrace._compiler.jaxpr_graph.inline_jit_calls` plays for ``jit``
boundaries.

Phase 1 implements ``cond`` if-conversion (:func:`if_convert_conds`):
every ETP-relevant ``cond`` equation is replaced by the inlined bodies of
*all* its branches followed by one ``select_n`` equation per output. The
integer index semantics of ``cond`` and ``select_n`` match exactly, and the
JVP of ``select_n`` selects tangents, so — for branches with finite values
and Jacobians — the canonicalized jaxpr is exact in both value and
derivative (see :class:`ControlFlowPolicy` for the dead-branch NaN/Inf
caveat under reverse mode).

Phase 2 implements inner-``scan`` unrolling (:func:`unroll_inner_scans`):
every ETP-relevant ``scan`` equation with static length at most
``policy.scan_unroll_limit`` is replaced by ``length`` clones of its body
(fresh variables per iteration, carry threaded between them), with per-
iteration ``xs`` slices materialized as ``slice``/``squeeze`` equations and
consumed stacked ``ys`` rebuilt with ``broadcast_in_dim``/``concatenate``.
Unrolling is semantically identical — values and all derivatives are exact.

:func:`canonicalize_control_flow` runs both passes to a joint fixpoint, so
a ``cond`` inside a ``scan`` body (and vice versa) canonicalizes fully.
"""

from dataclasses import dataclass
from typing import Any, Callable, Container, Dict, Iterable, List, Optional, Set, Tuple

import jax

from braintrace._compatible_imports import (
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Var,
    is_cond_primitive,
    is_jit_primitive,
    is_scan_primitive,
    is_while_primitive,
    new_jaxpr_eqn,
    new_var,
    scan_num_consts_carry,
)
from braintrace._op import is_etp_primitive
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit
from .jaxpr_graph import build_producer_map, inline_jit_calls

__all__ = [
    'ControlFlowPolicy',
    'DEFAULT_CONTROL_FLOW_POLICY',
    'canonicalize_control_flow',
    'if_convert_conds',
    'unroll_inner_scans',
]


@dataclass(frozen=True)
class ControlFlowPolicy:
    """Policy knobs governing control-flow canonicalization.

    Parameters
    ----------
    cond : str, optional
        How ETP-relevant ``cond`` equations are handled. ``'convert'``
        (default) if-converts them into inlined branches + ``select_n``;
        ``'opaque'`` leaves every ``cond`` untouched, so the existing
        control-flow restrictions apply (weights used inside raise
        ``NotImplementedError``; ETP primitives inside are excluded with a
        warning).
    scan_unroll_limit : int, optional
        Maximum static length of an ETP-relevant inner ``scan`` that
        :func:`unroll_inner_scans` unrolls into flat equations. Longer (or
        effectful, or ``while``-containing) scans stay opaque with a
        :attr:`~braintrace.DiagnosticKind.SCAN_UNROLL_SKIPPED` warning, and
        the existing control-flow restrictions apply. ``0`` (or negative)
        disables unrolling entirely. Default ``16``.
    while_hidden : str, optional
        How an *opaque* control-flow equation (``while``, or a ``scan``/
        ``cond`` the canonicalizer left opaque) that produces a hidden-state
        output **without consuming any weight invar** is handled.
        ``'opaque-fwd'`` (default) keeps it as an opaque forward node: the
        whole equation is embedded in the hidden-to-hidden transition and
        its Jacobian is extracted in forward mode (``while`` has no
        reverse-mode rule), recorded as a
        :attr:`~braintrace.DiagnosticKind.CONTROL_FLOW_OPAQUE_FWD` INFO
        diagnostic. ``'error'`` restores the pre-opaque-forward behaviour of
        raising ``NotImplementedError``.
    etp_in_control_flow : str, optional
        How an ETP primitive found *inside* a remaining opaque control-flow
        body is handled during relation discovery. ``'error'`` (default)
        raises ``NotImplementedError`` — that weight would otherwise
        silently drop out of online learning. ``'exclude'`` restores the
        previous behaviour of a loud warning
        (:attr:`~braintrace.DiagnosticKind.PRIMITIVE_INSIDE_CONTROL_FLOW`)
        plus exclusion of the weight from ETP relations.
    scan_descent : str, optional
        How ETP-relevant scans too long to unroll are handled. ``'auto'``
        (default) rewrites the scan for structured descent: relations and
        hidden groups are discovered inside the body and the eligibility
        trace is folded over the substep axis (see
        ``braintrace._compiler.scan_descent``). ``'off'`` preserves the
        pre-Phase-4 behavior: the scan stays opaque and compilation fails
        on the existing control-flow restrictions.

    Notes
    -----
    If-conversion changes execution semantics: **both** branches of a
    converted ``cond`` execute every step, and ``select_n`` discards the
    dead branch's *value*. Guarded partial operations (a ``cond`` used to
    avoid ``sqrt`` of a negative number, for example) therefore need care:

    - **Values and forward-mode (JVP) derivatives are safe** — ``select_n``
      selects whole outputs and tangents, so a dead-branch NaN/Inf never
      reaches the selected result.
    - **Reverse-mode (VJP) gradients are NOT safe** when the dead branch's
      local Jacobian is NaN/Inf: ``select_n``'s transpose hands the dead
      branch an exact-zero cotangent, but the dead branch's VJP multiplies
      that zero by its NaN/Inf Jacobian (``0 * nan = nan``), contaminating
      gradients of inputs shared with the live branch — the classic
      single-``where`` pitfall. If a ``cond`` guards a partial operation's
      domain, keep it opaque (``cond='opaque'``) or guard the operand
      itself (e.g. ``sqrt(where(ok, x, 1.))``) inside the branch.

    For branches whose values and Jacobians are finite, the canonicalized
    jaxpr is exact in both value and derivative. Branches with effects,
    containing ``while``, or containing a ``scan`` that
    :func:`unroll_inner_scans` could not unroll, are never converted (see
    :func:`if_convert_conds`). Scan unrolling itself has no semantics
    change: the unrolled equations compute exactly what the loop computed.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> policy = braintrace.ControlFlowPolicy(cond='opaque')
        >>> policy.cond
        'opaque'
    """

    cond: str = 'convert'
    scan_unroll_limit: int = 16
    while_hidden: str = 'opaque-fwd'
    etp_in_control_flow: str = 'error'
    scan_descent: str = 'auto'

    def __post_init__(self):
        if self.scan_descent not in ('auto', 'off'):
            raise ValueError(
                f"ControlFlowPolicy.scan_descent must be 'auto' or 'off', "
                f"got {self.scan_descent!r}."
            )


DEFAULT_CONTROL_FLOW_POLICY = ControlFlowPolicy()

ControlFlowPolicy.__module__ = 'braintrace'


def _subjaxprs(eqn: JaxprEqn) -> Iterable[Jaxpr]:
    """Yield every sub-jaxpr stored on an equation's params.

    ``call_jaxpr`` (custom_jvp/custom_vjp bodies) is descended for
    *detection* only — those equations are never rewritten, but the gates
    must still see ETP or while/scan primitives inside them.
    """
    for key in ('jaxpr', 'cond_jaxpr', 'body_jaxpr', 'call_jaxpr'):
        sub = eqn.params.get(key)
        if sub is not None:
            yield getattr(sub, 'jaxpr', sub)
    branches = eqn.params.get('branches')
    if branches is not None:
        for b in branches:
            yield getattr(b, 'jaxpr', b)


def _jaxpr_contains(jaxpr: Jaxpr, pred: Callable[[JaxprEqn], bool]) -> bool:
    """Whether any equation in *jaxpr* (descending sub-jaxprs) satisfies *pred*."""
    for eqn in jaxpr.eqns:
        if pred(eqn):
            return True
        for sub in _subjaxprs(eqn):
            if _jaxpr_contains(sub, pred):
                return True
    return False


def _iter_eqns_transitive(jaxpr: Jaxpr) -> Iterable[JaxprEqn]:
    """Yield every equation in *jaxpr*, descending sub-jaxprs."""
    for eqn in jaxpr.eqns:
        yield eqn
        for sub in _subjaxprs(eqn):
            yield from _iter_eqns_transitive(sub)


def _is_etp_eqn(eqn: JaxprEqn) -> bool:
    return is_etp_primitive(eqn.primitive)


def if_convert_conds(
    closed_jaxpr: ClosedJaxpr,
    *,
    weight_invars: Container[Var],
    hidden_invars: Container[Var],
    hidden_outvars: Container[Var],
    policy: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
) -> ClosedJaxpr:
    """If-convert every ETP-relevant ``cond`` equation in *closed_jaxpr*.

    Each converted ``cond`` is replaced by the inlined bodies of all its
    branches (every internal variable freshened) followed by one
    ``select_n(index, out_0[k], ..., out_{n-1}[k])`` equation per output
    position ``k``, written to the *original* ``cond`` output variables.
    Because the pass preserves the jaxpr's ``invars`` and ``outvars`` by
    object identity, every Var-identity lookup table built from them stays
    valid.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The closed jaxpr to canonicalize. ``jit`` equations should already
        be inlined (the pass re-runs
        :func:`~braintrace._compiler.jaxpr_graph.inline_jit_calls` afterwards
        to flatten ``jit`` bodies surfaced from converted branches).
    weight_invars : Container[Var]
        Jaxpr input variables of the trainable weights.
    hidden_invars : Container[Var]
        Jaxpr input variables of the hidden states.
    hidden_outvars : Container[Var]
        Jaxpr output variables of the hidden states.
    policy : ControlFlowPolicy, optional
        Canonicalization policy. With ``policy.cond == 'opaque'`` the input
        is returned unchanged.

    Returns
    -------
    ClosedJaxpr
        The canonicalized closed jaxpr. When nothing is converted, the input
        object itself is returned unchanged.

    Raises
    ------
    ValueError
        If ``policy.cond`` is neither ``'convert'`` nor ``'opaque'``.

    Notes
    -----
    A ``cond`` equation is converted only when it is *relevant* — any branch
    transitively contains an ETP primitive, or the equation consumes a
    weight/hidden input variable, or produces a hidden output variable — and
    *safe*: no effects, no branch transitively contains ``while``, and every
    ``scan`` transitively inside a branch is eligible for unrolling (static
    length within ``policy.scan_unroll_limit``, no effects, no inner
    ``while``) so that :func:`canonicalize_control_flow` can flatten it once
    it surfaces. A relevant-but-unsafe ``cond`` stays opaque and a
    :attr:`~braintrace.DiagnosticKind.COND_CONVERSION_SKIPPED` warning is
    emitted; the existing control-flow restrictions then apply unchanged.
    Irrelevant ``cond`` equations always stay opaque, at zero cost.

    On the canonicalized jaxpr **both branches execute every step**; see
    :class:`ControlFlowPolicy` for the semantics discussion.
    """
    if policy.cond == 'opaque':
        return closed_jaxpr
    if policy.cond != 'convert':
        raise ValueError(
            f"policy.cond must be 'convert' or 'opaque', got {policy.cond!r}."
        )

    # Fixpoint loop: converting a cond can surface user ``jit`` equations
    # from its branches, and their inlined bodies can expose further conds.
    # Each iteration converts what is visible, then flattens surfaced jits;
    # nesting depth is finite, so this terminates. ``skip_warned`` carries
    # the equations already reported as skipped, so a relevant-but-unsafe
    # cond warns once, not once per iteration.
    result = closed_jaxpr
    skip_warned: set = set()
    while True:
        converted, n_converted = _convert_conds_once(
            result,
            weight_invars=weight_invars,
            hidden_invars=hidden_invars,
            hidden_outvars=hidden_outvars,
            skip_warned=skip_warned,
            policy=policy,
        )
        if n_converted == 0:
            return result
        result = inline_jit_calls(converted)


def _convert_conds_once(
    closed_jaxpr: ClosedJaxpr,
    *,
    weight_invars: Container[Var],
    hidden_invars: Container[Var],
    hidden_outvars: Container[Var],
    skip_warned: set,
    policy: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
):
    """One conversion sweep over the top-level equations.

    Returns ``(closed_jaxpr, n_converted)``; the input object itself when
    nothing is converted.
    """
    jaxpr = closed_jaxpr.jaxpr
    if not any(is_cond_primitive(eqn) for eqn in jaxpr.eqns):
        return closed_jaxpr, 0

    extra_constvars: List[Var] = []
    extra_consts: List[Any] = []
    new_eqns: List[JaxprEqn] = []
    n_converted = 0

    def fresh_like(v: Var) -> Var:
        return new_var('', v.aval)

    def relevance_reason(resolved_invars, outvars, branches) -> Optional[str]:
        for v in resolved_invars:
            if isinstance(v, Var):
                if v in weight_invars:
                    return 'it consumes a weight invar'
                if v in hidden_invars:
                    return 'it consumes a hidden-state invar'
        for v in outvars:
            if v in hidden_outvars:
                return 'it produces a hidden-state outvar'
        for b in branches:
            if _jaxpr_contains(getattr(b, 'jaxpr', b), _is_etp_eqn):
                return 'a branch contains an ETP primitive'
        return None

    def unsafe_reason(eqn: JaxprEqn) -> Optional[str]:
        if eqn.effects:
            return 'the cond has effects'
        for b in eqn.params['branches']:
            sub = getattr(b, 'jaxpr', b)
            if sub.effects:
                return 'a branch has effects'
            if _jaxpr_contains(sub, is_while_primitive):
                return 'a branch contains a while loop'
            # An inner scan is fine as long as it can itself be unrolled
            # once conversion surfaces it (canonicalize_control_flow runs
            # both passes to a fixpoint). Only the identity-free criteria
            # are checkable here — branch-internal vars cannot be matched
            # against the outer weight table; the weights-as-xs gate
            # re-runs with real identities after the scan surfaces.
            for sub_eqn in _iter_eqns_transitive(sub):
                if is_scan_primitive(sub_eqn):
                    bad = _scan_static_ineligibility(sub_eqn, policy)
                    if bad is not None:
                        return (
                            f'a branch contains an inner scan that cannot '
                            f'be unrolled ({bad})'
                        )
        return None

    def inline_branch(branch_closed, operand_atoms) -> List[Any]:
        """Splice one branch body into ``new_eqns`` with every internal var
        freshened; return the branch's output atoms."""
        br = getattr(branch_closed, 'jaxpr', branch_closed)
        consts = getattr(branch_closed, 'consts', [])
        # Freshen constvars (instead of adopting the branch's Var objects):
        # the same branch ClosedJaxpr object may be inlined at several call
        # sites, and adopting would define its vars more than once.
        subst: Dict[Var, Any] = {}
        for cv, cval in zip(br.constvars, consts):
            fresh = fresh_like(cv)
            subst[cv] = fresh
            extra_constvars.append(fresh)
            extra_consts.append(cval)
        for iv, atom in zip(br.invars, operand_atoms):
            subst[iv] = atom

        def resolve(atom):
            if isinstance(atom, Var):
                return subst.get(atom, atom)
            return atom

        for sub_eqn in br.eqns:
            handle_eqn(sub_eqn, resolve, subst)
        return [resolve(v) for v in br.outvars]

    def handle_eqn(eqn: JaxprEqn, resolve, subst: Optional[Dict[Var, Any]]) -> None:
        """Process one equation. ``subst`` is ``None`` at the top level (the
        equation's vars are kept); inside a branch, invars are resolved and
        outvars freshened into ``subst``."""
        nonlocal n_converted

        if is_cond_primitive(eqn) and 'branches' in eqn.params:
            index_atom = resolve(eqn.invars[0])
            operand_atoms = [resolve(v) for v in eqn.invars[1:]]
            if subst is None:
                outvars = list(eqn.outvars)
            else:
                outvars = [fresh_like(v) for v in eqn.outvars]
                for ov, fresh in zip(eqn.outvars, outvars):
                    subst[ov] = fresh

            branches = eqn.params['branches']
            reason = relevance_reason(
                [index_atom, *operand_atoms], outvars, branches
            )
            if reason is not None:
                bad = unsafe_reason(eqn)
                if bad is None:
                    per_branch_outs = [
                        inline_branch(b, operand_atoms) for b in branches
                    ]
                    for k, ov in enumerate(outvars):
                        cases = [outs[k] for outs in per_branch_outs]
                        new_eqns.append(new_jaxpr_eqn(
                            [index_atom, *cases],
                            [ov],
                            jax.lax.select_n_p,
                            {},
                            set(),
                            eqn.source_info.replace(),
                        ))
                    n_converted += 1
                    emit(
                        kind=DiagnosticKind.COND_IF_CONVERTED,
                        level=DiagnosticLevel.INFO,
                        message=(
                            f'If-converted a cond with {len(branches)} '
                            f'branches and {len(outvars)} outputs into '
                            f'inlined branches + select_n because {reason}. '
                            f'Both branches now execute every step.'
                        ),
                        context={
                            'n_branches': len(branches),
                            'n_outputs': len(outvars),
                            'reason': reason,
                        },
                    )
                    return
                if id(eqn) not in skip_warned:
                    skip_warned.add(id(eqn))
                    emit(
                        kind=DiagnosticKind.COND_CONVERSION_SKIPPED,
                        level=DiagnosticLevel.WARNING,
                        message=(
                            f'An ETP-relevant cond ({reason}) was NOT '
                            f'if-converted because {bad}; it stays opaque and '
                            f'the existing control-flow restrictions apply.'
                        ),
                        context={'reason': bad, 'relevance': reason},
                    )
            # Not converted: keep the cond eqn (resolved when inside a branch).
            if subst is None:
                new_eqns.append(eqn)
            else:
                new_eqns.append(eqn.replace(
                    invars=[index_atom, *operand_atoms],
                    outvars=outvars,
                ))
            return

        if subst is None:
            new_eqns.append(eqn)
        else:
            fresh_outvars = [fresh_like(v) for v in eqn.outvars]
            for ov, fresh in zip(eqn.outvars, fresh_outvars):
                subst[ov] = fresh
            new_eqns.append(eqn.replace(
                invars=[resolve(v) for v in eqn.invars],
                outvars=fresh_outvars,
            ))

    for eqn in jaxpr.eqns:
        handle_eqn(eqn, lambda atom: atom, None)

    if n_converted == 0:
        return closed_jaxpr, 0

    new_jaxpr = Jaxpr(
        constvars=list(jaxpr.constvars) + extra_constvars,
        invars=list(jaxpr.invars),
        outvars=list(jaxpr.outvars),
        eqns=new_eqns,
        effects=jaxpr.effects,
        debug_info=jaxpr.debug_info,
    )
    result = ClosedJaxpr(new_jaxpr, list(closed_jaxpr.consts) + extra_consts)
    return result, n_converted


# ---------------------------------------------------------------------------
# Phase 2 — inner-scan unrolling
# ---------------------------------------------------------------------------

def _scan_static_ineligibility(
    eqn: JaxprEqn,
    policy: ControlFlowPolicy,
) -> Optional[str]:
    """Identity-free reasons a ``scan`` equation cannot be unrolled.

    Checks only criteria that need no Var-identity resolution (so the cond
    safety gate can consult it for scans buried inside branches): the policy
    knob, the static length, effects, and inner ``while`` loops. The
    weights-as-xs gate needs real outer identities and lives in
    :func:`_unroll_scans_once`. Returns ``None`` when unrollable.
    """
    if policy.scan_unroll_limit <= 0:
        return 'scan unrolling is disabled (scan_unroll_limit <= 0)'
    length = eqn.params['length']
    if length == 0:
        return 'its length is 0'
    if length > policy.scan_unroll_limit:
        return (
            f'its length {length} exceeds '
            f'scan_unroll_limit={policy.scan_unroll_limit}'
        )
    body = eqn.params['jaxpr'].jaxpr
    if eqn.effects or body.effects:
        return 'the scan (or its body) has effects'
    if _jaxpr_contains(body, is_while_primitive):
        return 'its body contains a while loop'
    return None


def unroll_inner_scans(
    closed_jaxpr: ClosedJaxpr,
    *,
    weight_invars: Container[Var],
    hidden_invars: Container[Var],
    hidden_outvars: Container[Var],
    policy: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
) -> ClosedJaxpr:
    """Unroll every ETP-relevant, eligible ``scan`` equation in *closed_jaxpr*.

    Each unrolled ``scan`` is replaced by ``length`` clones of its body with
    fresh internal variables, the carry threaded from one clone to the next
    (starting from the outer init operands), per-iteration ``xs`` slices
    materialized as ``slice`` + ``squeeze`` equations, and — for stacked
    ``ys`` outputs that are actually consumed — ``broadcast_in_dim`` +
    ``concatenate`` equations rebuilding the stacked value. The original
    ``scan`` output variables are re-emitted by these equations, and the
    jaxpr's ``invars``/``outvars`` are preserved by object identity, so every
    Var-identity lookup table built from them stays valid. Unrolling is
    semantically identical: values and all derivatives are exact.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The closed jaxpr to canonicalize. ``jit`` equations should already be
        inlined (the pass re-runs
        :func:`~braintrace._compiler.jaxpr_graph.inline_jit_calls` after each
        sweep to flatten ``jit`` bodies surfaced from unrolled scan bodies).
    weight_invars : Container[Var]
        Jaxpr input variables of the trainable weights.
    hidden_invars : Container[Var]
        Jaxpr input variables of the hidden states.
    hidden_outvars : Container[Var]
        Jaxpr output variables of the hidden states.
    policy : ControlFlowPolicy, optional
        Canonicalization policy. With ``policy.scan_unroll_limit <= 0`` the
        input is returned unchanged.

    Returns
    -------
    ClosedJaxpr
        The canonicalized closed jaxpr. When nothing is unrolled, the input
        object itself is returned unchanged.

    Notes
    -----
    A ``scan`` equation is unrolled only when it is *relevant* — its body
    transitively contains an ETP primitive, or the equation consumes a
    weight/hidden input variable, or produces a hidden output variable — and
    *eligible*: static ``length`` in ``[1, policy.scan_unroll_limit]``, no
    effects, no inner ``while``, and no scanned-over (``xs``) operand that is
    (or is computed from) a trainable weight. A relevant scan whose ``xs``
    carry a weight stays opaque with a
    :attr:`~braintrace.DiagnosticKind.RELATION_EXCLUDED_SLICED_WEIGHT`
    warning — after unrolling, each iteration would consume a *slice* of the
    stacked parameter, which the relation pass would mis-attribute to the
    full parameter (slice-aware relations are future work). Any other
    ineligible-but-relevant scan warns
    :attr:`~braintrace.DiagnosticKind.SCAN_UNROLL_SKIPPED`. In both cases
    the existing control-flow restrictions then apply unchanged. Irrelevant
    scans always stay opaque, at zero cost — in particular the outer
    time-step loop of a training harness is never touched.
    """
    if policy.scan_unroll_limit <= 0:
        return closed_jaxpr

    # Fixpoint: unrolling a scan can surface user ``jit`` equations and
    # nested scans from its body. Each sweep unrolls the visible top-level
    # scans, then flattens surfaced jits; nesting depth is finite, so this
    # terminates. ``skip_warned`` carries the equations already reported, so
    # a relevant-but-ineligible scan warns once, not once per sweep.
    result = closed_jaxpr
    skip_warned: set = set()
    while True:
        converted, n_unrolled = _unroll_scans_once(
            result,
            weight_invars=weight_invars,
            hidden_invars=hidden_invars,
            hidden_outvars=hidden_outvars,
            policy=policy,
            skip_warned=skip_warned,
        )
        if n_unrolled == 0:
            return result
        result = inline_jit_calls(converted)


def _unroll_scans_once(
    closed_jaxpr: ClosedJaxpr,
    *,
    weight_invars: Container[Var],
    hidden_invars: Container[Var],
    hidden_outvars: Container[Var],
    policy: ControlFlowPolicy,
    skip_warned: set,
):
    """One unrolling sweep over the top-level equations.

    Returns ``(closed_jaxpr, n_unrolled)``; the input object itself when
    nothing is unrolled.
    """
    jaxpr = closed_jaxpr.jaxpr
    if not any(is_scan_primitive(eqn) for eqn in jaxpr.eqns):
        return closed_jaxpr, 0

    # Consumption of the original scan outvars, for dead-output elision.
    # Downstream equations keep referencing the original outvars (they are
    # re-emitted under the same identity), so the original jaxpr's edges are
    # exactly the right consumption record.
    consumed: Set[Var] = set()
    for eqn in jaxpr.eqns:
        for iv in eqn.invars:
            if isinstance(iv, Var):
                consumed.add(iv)
    for ov in jaxpr.outvars:
        if isinstance(ov, Var):
            consumed.add(ov)

    producers = build_producer_map(jaxpr)

    def reaches_weight(atom: Any) -> bool:
        """Backward reachability from *atom* to any weight invar."""
        if not isinstance(atom, Var):
            return False
        frontier = [atom]
        visited: Set[Var] = set()
        while frontier:
            v = frontier.pop()
            if v in visited:
                continue
            visited.add(v)
            if v in weight_invars:
                return True
            producer = producers.get(v)
            if producer is not None:
                for iv in producer.invars:
                    if isinstance(iv, Var) and iv not in visited:
                        frontier.append(iv)
        return False

    def relevance_reason(eqn: JaxprEqn) -> Optional[str]:
        for v in eqn.invars:
            if isinstance(v, Var):
                if v in weight_invars:
                    return 'it consumes a weight invar'
                if v in hidden_invars:
                    return 'it consumes a hidden-state invar'
        for v in eqn.outvars:
            if v in hidden_outvars:
                return 'it produces a hidden-state outvar'
        if _jaxpr_contains(eqn.params['jaxpr'].jaxpr, _is_etp_eqn):
            return 'its body contains an ETP primitive'
        return None

    extra_constvars: List[Var] = []
    extra_consts: List[Any] = []
    new_eqns: List[JaxprEqn] = []
    n_unrolled = 0

    def fresh_like(v: Var) -> Var:
        return new_var('', v.aval)

    def unroll_scan(eqn: JaxprEqn) -> None:
        params = eqn.params
        body_closed = params['jaxpr']
        body = body_closed.jaxpr
        length: int = params['length']
        num_consts, num_carry = scan_num_consts_carry(eqn)
        reverse: bool = params['reverse']

        const_atoms = list(eqn.invars[:num_consts])
        init_atoms = list(eqn.invars[num_consts:num_consts + num_carry])
        xs_atoms = list(eqn.invars[num_consts + num_carry:])
        num_ys = len(eqn.outvars) - num_carry
        source_info = eqn.source_info

        # Body consts are read-only: hoist each ONCE as a fresh outer
        # constvar shared by every iteration (freshened, not adopted, for
        # the same reason as in cond inlining).
        const_subst: Dict[Var, Any] = {}
        for cv, cval in zip(body.constvars, body_closed.consts):
            fresh = fresh_like(cv)
            const_subst[cv] = fresh
            extra_constvars.append(fresh)
            extra_consts.append(cval)

        carry_atoms: List[Any] = list(init_atoms)
        ys_atoms: List[List[Any]] = [[None] * length for _ in range(num_ys)]

        # For the final processing iteration, produce each ORIGINAL carry
        # outvar directly from the cloned body equation instead of a fresh
        # var + identity copy. This matters beyond economy: a body that
        # returns its new carry as a y output (the for_loop shape) aliases
        # the two by VALUE, and downstream consumers of the stacked ys must
        # flow through the hidden outvar for perturbation-based learning
        # signals (dL/dh) to see them. Positions whose body outvar is a
        # passthrough invar, a Literal, or a duplicate of an already-mapped
        # carry fall back to the identity-copy path below.
        body_produced: Set[Var] = set()
        for body_eqn in body.eqns:
            body_produced.update(
                v for v in body_eqn.outvars if isinstance(v, Var)
            )
        final_carry_map: Dict[Var, Var] = {}
        for bv, ov in zip(body.outvars[:num_carry], eqn.outvars[:num_carry]):
            if (
                ov in consumed
                and isinstance(bv, Var)
                and bv in body_produced
                and bv not in final_carry_map
            ):
                final_carry_map[bv] = ov

        # reverse=True threads the carry from the LAST xs slice to the
        # first, but the y computed for slice t still lands at ys[t].
        order = range(length - 1, -1, -1) if reverse else range(length)
        final_t = order[-1] if length else None
        for t in order:
            subst: Dict[Var, Any] = dict(const_subst)
            for iv, atom in zip(body.invars[:num_consts], const_atoms):
                subst[iv] = atom
            for iv, atom in zip(
                body.invars[num_consts:num_consts + num_carry], carry_atoms
            ):
                subst[iv] = atom

            # Materialize this iteration's xs slices.
            for iv, xs_atom in zip(
                body.invars[num_consts + num_carry:], xs_atoms
            ):
                tail = tuple(iv.aval.shape)
                sliced = new_var('', iv.aval.update(shape=(1,) + tail))
                new_eqns.append(new_jaxpr_eqn(
                    [xs_atom],
                    [sliced],
                    jax.lax.slice_p,
                    dict(
                        start_indices=(t,) + (0,) * len(tail),
                        limit_indices=(t + 1,) + tail,
                        strides=None,
                    ),
                    set(),
                    source_info.replace(),
                ))
                squeezed = fresh_like(iv)
                new_eqns.append(new_jaxpr_eqn(
                    [sliced],
                    [squeezed],
                    jax.lax.squeeze_p,
                    dict(dimensions=(0,)),
                    set(),
                    source_info.replace(),
                ))
                subst[iv] = squeezed

            def resolve(atom: Any) -> Any:
                if isinstance(atom, Var):
                    return subst.get(atom, atom)
                return atom

            # Clone the body equations with fresh outvars; on the final
            # processing iteration, eligible carry outputs write the
            # original scan outvars directly.
            outvar_map = final_carry_map if t == final_t else {}
            for body_eqn in body.eqns:
                fresh_outvars = [
                    outvar_map.get(v) if isinstance(v, Var) and v in outvar_map
                    else fresh_like(v)
                    for v in body_eqn.outvars
                ]
                for ov, fresh in zip(body_eqn.outvars, fresh_outvars):
                    subst[ov] = fresh
                new_eqns.append(body_eqn.replace(
                    invars=[resolve(v) for v in body_eqn.invars],
                    outvars=fresh_outvars,
                ))

            out_atoms = [resolve(v) for v in body.outvars]
            carry_atoms = out_atoms[:num_carry]
            for j, y_atom in enumerate(out_atoms[num_carry:]):
                ys_atoms[j][t] = y_atom

        # Re-emit the original scan outvars. Dead outputs (never consumed —
        # including DropVars) get no producing equation, which is legal;
        # direct-mapped carries were already written by the final clone.
        for ov, final_atom in zip(eqn.outvars[:num_carry], carry_atoms):
            if ov not in consumed or final_atom is ov:
                continue
            shape = tuple(ov.aval.shape)
            new_eqns.append(new_jaxpr_eqn(
                [final_atom],
                [ov],
                jax.lax.broadcast_in_dim_p,
                dict(
                    shape=shape,
                    broadcast_dimensions=tuple(range(len(shape))),
                    sharding=None,
                ),
                set(),
                source_info.replace(),
            ))
        for j, ov in enumerate(eqn.outvars[num_carry:]):
            if ov not in consumed:
                continue
            stacked_shape = tuple(ov.aval.shape)
            slice_shape = stacked_shape[1:]
            slice_vars = []
            for y_atom in ys_atoms[j]:
                sv = new_var('', ov.aval.update(shape=(1,) + slice_shape))
                new_eqns.append(new_jaxpr_eqn(
                    [y_atom],
                    [sv],
                    jax.lax.broadcast_in_dim_p,
                    dict(
                        shape=(1,) + slice_shape,
                        broadcast_dimensions=tuple(
                            range(1, 1 + len(slice_shape))
                        ),
                        sharding=None,
                    ),
                    set(),
                    source_info.replace(),
                ))
                slice_vars.append(sv)
            new_eqns.append(new_jaxpr_eqn(
                slice_vars,
                [ov],
                jax.lax.concatenate_p,
                dict(dimension=0),
                set(),
                source_info.replace(),
            ))

    for eqn in jaxpr.eqns:
        if not (is_scan_primitive(eqn) and 'jaxpr' in eqn.params):
            new_eqns.append(eqn)
            continue
        reason = relevance_reason(eqn)
        if reason is None:
            new_eqns.append(eqn)
            continue
        bad = _scan_static_ineligibility(eqn, policy)
        if bad is not None:
            if id(eqn) not in skip_warned:
                skip_warned.add(id(eqn))
                # Matches the descent-eligibility rule in
                # ``scan_descent._descent_blockers``: any positive-length
                # scan beyond the limit descends, including when unrolling
                # is disabled entirely (limit <= 0).
                over_limit = eqn.params['length'] > policy.scan_unroll_limit
                if over_limit and policy.scan_descent == 'auto':
                    # An over-limit scan is no longer a dead end: the
                    # scan-descent pass (Phase 4) picks it up downstream.
                    emit(
                        kind=DiagnosticKind.SCAN_UNROLL_SKIPPED,
                        level=DiagnosticLevel.INFO,
                        message=(
                            f'An ETP-relevant scan ({reason}) was NOT '
                            f'unrolled because {bad}; structured scan '
                            f'descent will handle it (see '
                            f'ControlFlowPolicy.scan_descent).'
                        ),
                        context={'reason': bad, 'relevance': reason},
                    )
                else:
                    emit(
                        kind=DiagnosticKind.SCAN_UNROLL_SKIPPED,
                        level=DiagnosticLevel.WARNING,
                        message=(
                            f'An ETP-relevant scan ({reason}) was NOT '
                            f'unrolled because {bad}; it stays opaque and '
                            f'the existing control-flow restrictions apply. '
                            f'Raise ControlFlowPolicy.scan_unroll_limit or '
                            f'restructure the loop if its weights should '
                            f'learn online.'
                        ),
                        context={'reason': bad, 'relevance': reason},
                    )
            new_eqns.append(eqn)
            continue
        num_consts, num_carry = scan_num_consts_carry(eqn)
        num_prefix = num_consts + num_carry
        sliced_weight = [
            v for v in eqn.invars[num_prefix:] if reaches_weight(v)
        ]
        if sliced_weight:
            if id(eqn) not in skip_warned:
                skip_warned.add(id(eqn))
                emit(
                    kind=DiagnosticKind.RELATION_EXCLUDED_SLICED_WEIGHT,
                    level=DiagnosticLevel.WARNING,
                    message=(
                        f'An ETP-relevant scan ({reason}) was NOT unrolled '
                        f'because it scans over a trainable weight (a '
                        f'stacked parameter passed as xs). Each unrolled '
                        f'iteration would consume a *slice* of the '
                        f'parameter, which relation analysis cannot yet '
                        f'attribute correctly. Pass per-iteration weights '
                        f'as separate parameters, or keep the loop out of '
                        f'online learning.'
                    ),
                    context={'relevance': reason,
                             'n_sliced': len(sliced_weight)},
                )
            new_eqns.append(eqn)
            continue

        unroll_scan(eqn)
        n_unrolled += 1
        emit(
            kind=DiagnosticKind.SCAN_UNROLLED,
            level=DiagnosticLevel.INFO,
            message=(
                f'Unrolled a scan of length {eqn.params["length"]} '
                f'(num_consts={num_consts}, '
                f'num_carry={num_carry}, '
                f'reverse={eqn.params["reverse"]}) into flat equations '
                f'because {reason}.'
            ),
            context={
                'length': eqn.params['length'],
                'num_consts': num_consts,
                'num_carry': num_carry,
                'reverse': eqn.params['reverse'],
                'reason': reason,
            },
        )

    if n_unrolled == 0:
        return closed_jaxpr, 0

    new_jaxpr = Jaxpr(
        constvars=list(jaxpr.constvars) + extra_constvars,
        invars=list(jaxpr.invars),
        outvars=list(jaxpr.outvars),
        eqns=new_eqns,
        effects=jaxpr.effects,
        debug_info=jaxpr.debug_info,
    )
    result = ClosedJaxpr(new_jaxpr, list(closed_jaxpr.consts) + extra_consts)
    return result, n_unrolled


# ---------------------------------------------------------------------------
# Joint fixpoint driver
# ---------------------------------------------------------------------------

def canonicalize_control_flow(
    closed_jaxpr: ClosedJaxpr,
    *,
    weight_invars: Container[Var],
    hidden_invars: Container[Var],
    hidden_outvars: Container[Var],
    policy: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
) -> ClosedJaxpr:
    """Run every control-flow canonicalization pass to a joint fixpoint.

    Alternates :func:`if_convert_conds`'s conversion sweep and
    :func:`unroll_inner_scans`'s unrolling sweep (re-inlining surfaced
    ``jit`` bodies after each) until neither changes the jaxpr. The joint
    fixpoint handles mutual nesting: a ``cond`` inside a ``scan`` body
    surfaces once the scan unrolls and is then converted; an eligible
    ``scan`` inside a ``cond`` branch surfaces once the branch inlines and
    is then unrolled.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The closed jaxpr to canonicalize, with ``jit`` equations already
        inlined.
    weight_invars : Container[Var]
        Jaxpr input variables of the trainable weights.
    hidden_invars : Container[Var]
        Jaxpr input variables of the hidden states.
    hidden_outvars : Container[Var]
        Jaxpr output variables of the hidden states.
    policy : ControlFlowPolicy, optional
        Canonicalization policy; each pass honors its own knob
        (``policy.cond``, ``policy.scan_unroll_limit``).

    Returns
    -------
    ClosedJaxpr
        The canonicalized closed jaxpr. When nothing changes, the input
        object itself is returned unchanged.

    Raises
    ------
    ValueError
        If ``policy.cond`` is neither ``'convert'`` nor ``'opaque'``.
    """
    if policy.cond not in ('convert', 'opaque'):
        raise ValueError(
            f"policy.cond must be 'convert' or 'opaque', got {policy.cond!r}."
        )

    result = closed_jaxpr
    cond_skip_warned: set = set()
    scan_skip_warned: set = set()
    while True:
        n_total = 0
        if policy.cond == 'convert':
            converted, n = _convert_conds_once(
                result,
                weight_invars=weight_invars,
                hidden_invars=hidden_invars,
                hidden_outvars=hidden_outvars,
                skip_warned=cond_skip_warned,
                policy=policy,
            )
            if n:
                result = inline_jit_calls(converted)
                n_total += n
        if policy.scan_unroll_limit > 0:
            converted, n = _unroll_scans_once(
                result,
                weight_invars=weight_invars,
                hidden_invars=hidden_invars,
                hidden_outvars=hidden_outvars,
                policy=policy,
                skip_warned=scan_skip_warned,
            )
            if n:
                result = inline_jit_calls(converted)
                n_total += n
        if n_total == 0:
            return result
