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
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

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
from .canonicalize import ControlFlowPolicy
from .jaxpr_graph import _sub_closed_jaxprs

__all__ = []


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
