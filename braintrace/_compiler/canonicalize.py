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

Inner-``scan`` unrolling is Phase 2; the ``scan_unroll_limit`` knob on
:class:`ControlFlowPolicy` is reserved for it.
"""

from dataclasses import dataclass
from typing import Any, Callable, Container, Dict, Iterable, List, Optional

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
)
from braintrace._op import is_etp_primitive
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit
from .jaxpr_graph import inline_jit_calls

__all__ = [
    'ControlFlowPolicy',
    'DEFAULT_CONTROL_FLOW_POLICY',
    'if_convert_conds',
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
        Reserved for the Phase 2 inner-``scan`` unrolling pass; currently
        unused. Default ``16``.

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
    jaxpr is exact in both value and derivative. Branches with effects, or
    containing ``while``/``scan``, are never converted (see
    :func:`if_convert_conds`).

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
    *safe*: no effects, and no branch transitively contains ``while`` or
    ``scan`` (inner-``scan`` unrolling lands in Phase 2). A relevant-but-
    unsafe ``cond`` stays opaque and a
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
            if _jaxpr_contains(sub, is_scan_primitive):
                return (
                    'a branch contains an inner scan '
                    '(scan unrolling is not implemented yet)'
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
