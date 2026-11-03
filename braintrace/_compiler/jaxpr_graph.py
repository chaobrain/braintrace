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

"""Shared jaxpr-graph utilities for the ETrace compiler.

Every compiler pass needs the same handful of graph primitives over a
``Jaxpr``: a producer map (var -> defining equation), a consumer map
(var -> reading equations), ordered forward reachability, and — since the
analyses cannot see through ``jit`` call boundaries — a pass that splices
``jit``/``pjit`` bodies into the surrounding jaxpr. They live here so the
passes in :mod:`hid_param_op`, :mod:`hidden_group`, and
:mod:`hidden_pertubation` share one implementation.

All traversals use insertion-ordered ``dict`` objects as ordered sets, so
results are deterministic across processes (plain ``set`` iteration follows
memory-address hashes for jaxpr ``Var`` objects).
"""

from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Sequence, TypeAlias, Union

from braintrace._compatible_imports import (
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Literal,
    Var,
    is_jit_primitive,
    new_var,
)
from braintrace._op import is_etp_primitive

# A jaxpr equation input: either a variable or an inlined constant. JAX spells
# this ``Atom``, but that name is private and has moved between releases, so the
# union is written out here (mirroring ``_compatible_imports``' policy of not
# depending on private JAX paths).
Atom: TypeAlias = Union[Var, Literal]

# Recurrent-weight mixing primitives -- dense / convolutional weights -- whose
# consumption of a hidden state is a genuine *cross-position* coupling, i.e. that
# can make ``h_i^t`` depend on ``h_j^{t-1}`` for ``i != j`` through a learned or
# fixed recurrent weight. The default ("without recurrence") grouping mode
# excludes these (when they read the hidden state) from the hidden-to-hidden
# transition so it stays position-diagonal.
#
# This set is deliberately narrow: it does NOT list within-position
# reductions/gathers (e.g. the ``gather`` that splits a stacked
# ``HiddenGroupState`` such as an ALIF ``('neu', 'st')`` into its ``V``/``a``
# components over the ``num_state`` axis). Those operate *within* a position and
# must stay in the transition -- excluding them would drop a real, diagonal
# ``D^t`` term and corrupt the grouped-state transition. The ETP mixing
# primitives (``etp_mv``/``etp_mm``/``etp_conv``) are handled separately by the
# ETP-boundary skip in ``hidden_group.JaxprEvalForHiddenGroup._eval_eqn``.
# Membership makes an equation a *candidate*: the hidden-group pass keeps a
# ``dot_general`` that provably contracts no position axis (#213), while the
# other passes still treat every member as mixing (conservative).
# It lives in this leaf module so every pass can import it at module level.
_RECURRENT_WEIGHT_MIXING_PRIMITIVES = frozenset({
    'dot_general', 'conv_general_dilated',
})

__all__ = [
    'build_producer_map',
    'build_consumer_map',
    'forward_closure',
    'inline_jit_calls',
]


def build_producer_map(jaxpr: Jaxpr) -> Dict[Var, JaxprEqn]:
    """Map each variable to the equation that produces it.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to index.

    Returns
    -------
    dict
        Mapping from each output ``Var`` to its producing equation. Jaxpr
        invars and constvars have no producer and are absent.
    """
    producers: Dict[Var, JaxprEqn] = {}
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            producers[v] = eqn
    return producers


def build_consumer_map(jaxpr: Jaxpr) -> Dict[Var, List[JaxprEqn]]:
    """Map each variable to the equations that consume it.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to index.

    Returns
    -------
    dict
        Mapping from each ``Var`` to the list of equations reading it, in
        equation order.
    """
    consumers: Dict[Var, List[JaxprEqn]] = {}
    for eqn in jaxpr.eqns:
        for v in eqn.invars:
            if isinstance(v, Var):
                consumers.setdefault(v, []).append(eqn)
    return consumers


def forward_closure(
    start_var: Var,
    consumer_map: Dict[Var, List[JaxprEqn]],
) -> Dict[Var, None]:
    """Collect every variable reachable forward from *start_var*.

    Parameters
    ----------
    start_var : Var
        The variable to start from (included in the result).
    consumer_map : dict
        Consumer map built by :func:`build_consumer_map`.

    Returns
    -------
    dict
        Insertion-ordered dict used as an ordered set; iteration follows BFS
        encounter order, so the result is deterministic.
    """
    closure: Dict[Var, None] = {}
    frontier: deque = deque([start_var])
    while frontier:
        v = frontier.popleft()
        if v in closure:
            continue
        closure[v] = None
        for eqn in consumer_map.get(v, []):
            for ov in eqn.outvars:
                if ov not in closure:
                    frontier.append(ov)
    return closure


def _contains_jit_eqn(jaxpr: Jaxpr) -> bool:
    return any(is_jit_primitive(eqn) for eqn in jaxpr.eqns)


def _sub_closed_jaxprs(eqn: JaxprEqn) -> Iterator[ClosedJaxpr]:
    """Yield the ``ClosedJaxpr`` bodies of a control-flow equation.

    Covers ``scan`` (``jaxpr``), ``while`` (``cond_jaxpr``/``body_jaxpr``),
    and ``cond`` (``branches``). ``call_jaxpr`` (custom-derivative calls) is
    deliberately excluded: those bodies carry derivative semantics and are
    never rewritten by :func:`inline_jit_calls`.
    """
    for key in ('jaxpr', 'cond_jaxpr', 'body_jaxpr'):
        sub = eqn.params.get(key)
        if isinstance(sub, ClosedJaxpr):
            yield sub
    for b in eqn.params.get('branches', ()) or ():
        if isinstance(b, ClosedJaxpr):
            yield b


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


def _is_etp_relevant(eqn: JaxprEqn) -> bool:
    """Whether a ``scan`` can hold an ETP relation.

    A scan is ETP-relevant iff its body (recursively) holds an ETP primitive.
    A weight that any other scan consumes is read through plain ops only,
    exactly like a plain use at top level, so the scan holds no relation
    (issue #209). Scan descent (:mod:`.scan_descent`) only considers
    ETP-relevant scans. The unroller (:mod:`.canonicalize`) may keep any
    other scan rolled, and the hidden-group pass embeds it whole as an opaque
    transition node, its weights being constants of that node
    (``base.check_unsupported_op``).

    Parameters
    ----------
    eqn : JaxprEqn
        The scan equation.

    Returns
    -------
    bool
        ``True`` when descent (or unrolling) must consider this scan; also
        when the equation carries no body to inspect.
    """
    body = eqn.params.get('jaxpr')
    if body is None:
        return True
    return _body_has_etp(body.jaxpr)


def _contains_jit(jaxpr: Jaxpr) -> bool:
    """``True`` when a jit eqn exists at top level or inside any
    (transitively nested) control-flow body."""
    for eqn in jaxpr.eqns:
        if is_jit_primitive(eqn) and 'jaxpr' in eqn.params:
            return True
        for sub in _sub_closed_jaxprs(eqn):
            if _contains_jit(sub.jaxpr):
                return True
    return False


#: Equations whose ``jaxpr`` param is an *open* jaxpr, which may not hold
#: constants (JAX's ``remat_jvp`` asserts it has no constvars). Since JAX 0.11
#: ``ClosedJaxpr`` and ``Jaxpr`` are one class, so the type does not tell an
#: open body from a closed one. A jit inlined into one of these bodies has its
#: constants hoisted into leading operands of the equation (#251).
_OPEN_BODY_CALLS = frozenset({'remat2', 'remat', 'checkpoint'})

#: Equations whose body params are closed jaxprs that may gain constants.
_CLOSED_BODY_PRIMITIVES = frozenset({'scan', 'while', 'cond'})


def _open_body(closed: ClosedJaxpr, template: Jaxpr) -> Jaxpr:
    """*closed* as an open jaxpr: its constvars become leading invars."""
    body = closed.jaxpr
    debug_info = getattr(template, 'debug_info', None)
    if debug_info is not None:
        debug_info = debug_info._replace(arg_names=None)
    return Jaxpr(
        constvars=[],
        invars=[*body.constvars, *body.invars],
        outvars=list(body.outvars),
        eqns=list(body.eqns),
        effects=body.effects,
        debug_info=debug_info,
    )


def _inline_jits_in_control_flow_params(eqn: JaxprEqn) -> Optional[tuple]:
    """Rewrite a control-flow eqn's body params so jit calls inside them are
    inlined.

    Returns ``(params, hoisted)``, or ``None`` when every body is already
    jit-free (so the caller can keep the eqn by identity). ``hoisted`` lists
    ``(constvar, value)`` pairs the caller must bind and prepend to the
    equation's operands: the constants of jits inlined into the open body of a
    checkpoint (:data:`_OPEN_BODY_CALLS`, #251). A ``scan``/``while``/``cond``
    body is closed and keeps its new constants. Any other equation with a body
    keeps a jit whose constants it could not hold.
    """
    name = eqn.primitive.name
    new_params = dict(eqn.params)
    changed = False
    hoisted: List[tuple] = []
    for key in ('jaxpr', 'cond_jaxpr', 'body_jaxpr'):
        sub = new_params.get(key)
        if isinstance(sub, ClosedJaxpr) and _contains_jit(sub.jaxpr):
            inlined = inline_jit_calls(sub)
            if name in _OPEN_BODY_CALLS and key == 'jaxpr':
                hoisted.extend(zip(inlined.jaxpr.constvars, inlined.consts))
                inlined = _open_body(inlined, sub)
                prevent_cse = new_params.get('prevent_cse')
                if isinstance(prevent_cse, tuple):
                    new_params['prevent_cse'] = (True,) * len(hoisted) + prevent_cse
            elif name not in _CLOSED_BODY_PRIMITIVES and len(inlined.consts) > len(getattr(sub, 'consts', ())):
                continue  # an open body of another kind: leave its jits
            new_params[key] = inlined
            changed = True
    branches = new_params.get('branches')
    if branches is not None:
        new_branches = tuple(
            inline_jit_calls(b)
            if isinstance(b, ClosedJaxpr) and _contains_jit(b.jaxpr) else b
            for b in branches
        )
        if any(nb is not ob for nb, ob in zip(new_branches, branches)):
            new_params['branches'] = new_branches
            changed = True
    return (new_params, hoisted) if changed else None


def inline_jit_calls(closed_jaxpr: ClosedJaxpr) -> ClosedJaxpr:
    """Splice every top-level ``jit``/``pjit`` body into the enclosing jaxpr.

    The ETrace analyses identify weights, hidden states, and ETP primitives by
    ``Var`` identity in one flat jaxpr; a ``jit`` call boundary hides its body
    behind fresh inner variables, so weights or hidden states used inside a
    user's ``jax.jit``-wrapped function were previously either rejected or
    silently excluded. Inlining removes the boundary before any analysis runs.

    The pass is recursive (``jit`` inside ``jit`` is flattened), lifts inner
    closure constants into the outer ``constvars``/``consts``, and rewires
    pass-through outputs by substitution. Every spliced body has its internal
    variables freshened *per call site*: JAX's trace cache hands every call
    of the same jitted function the same inner jaxpr object, so splicing it
    verbatim more than once would define its variables twice and silently
    alias the calls' results. Top-level equations that are not jit calls keep
    their original ``Var`` objects. Control-flow bodies
    (``scan``/``while``/``cond``) are descended: a body containing a jit call
    is rewritten with that call inlined (the eqn is rebuilt with new body
    params), while eqns whose bodies are jit-free are kept unchanged **by
    identity** — later passes key exemption sets on eqn identity.
    Custom-derivative call primitives are left untouched:
    ``custom_jvp_call``/``custom_vjp_call`` carry derivative semantics that
    must not be flattened away.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The closed jaxpr to inline.

    Returns
    -------
    ClosedJaxpr
        A new closed jaxpr with no top-level ``jit`` equations, evaluating to
        the same outputs. When the input contains no ``jit`` equation, the
        input object itself is returned unchanged.
    """
    jaxpr = closed_jaxpr.jaxpr
    if not _contains_jit(jaxpr):
        return closed_jaxpr

    # Substitution at the top level only maps a jit call's outer outvars to
    # their replacement atoms; every other top-level equation keeps its
    # original Var objects, so downstream Var-identity tables built from the
    # result see the same vars for untouched equations.
    top_subst: Dict[Var, Any] = {}  # Var -> Var | Literal

    def resolve_top(atom: Atom) -> Atom:
        if isinstance(atom, Var):
            return top_subst.get(atom, atom)
        return atom

    extra_constvars: List[Var] = []
    extra_consts: List[Any] = []
    new_eqns: List[JaxprEqn] = []

    def hoist(pairs: Sequence[tuple]) -> List[Var]:
        """Bind hoisted ``(constvar, value)`` pairs as fresh outer constvars."""
        fresh_vars = []
        for cv, cval in pairs:
            fresh = new_var('', cv.aval)
            extra_constvars.append(fresh)
            extra_consts.append(cval)
            fresh_vars.append(fresh)
        return fresh_vars

    def splice(inner_closed: ClosedJaxpr, call_arg_atoms: Sequence[Atom]) -> List[Any]:
        """Inline one jit body with every internal var freshened; return the
        body's (resolved) output atoms.

        Freshening is per call site: JAX's trace cache hands every call of
        the same jitted function the SAME inner ``ClosedJaxpr`` object, so
        splicing it verbatim twice would define its vars twice (an SSA
        violation that silently aliases the two calls' results).
        """
        inner = getattr(inner_closed, 'jaxpr', inner_closed)
        inner_consts = getattr(inner_closed, 'consts', [])
        local: Dict[Var, Any] = {}
        for cv, cval in zip(inner.constvars, inner_consts):
            fresh = new_var('', cv.aval)
            local[cv] = fresh
            extra_constvars.append(fresh)
            extra_consts.append(cval)
        for iv, arg in zip(inner.invars, call_arg_atoms):
            local[iv] = arg

        def res(atom: Atom) -> Atom:
            if isinstance(atom, Var):
                return local.get(atom, atom)
            return atom

        for eqn in inner.eqns:
            if is_jit_primitive(eqn) and 'jaxpr' in eqn.params:
                outs = splice(eqn.params['jaxpr'], [res(v) for v in eqn.invars])
                for ov, out_atom in zip(eqn.outvars, outs):
                    local[ov] = out_atom
            else:
                fresh_outvars = [new_var('', v.aval) for v in eqn.outvars]
                for ov, fresh in zip(eqn.outvars, fresh_outvars):
                    local[ov] = fresh
                replace_kwargs: Dict[str, Any] = dict(
                    invars=[res(v) for v in eqn.invars],
                    outvars=fresh_outvars,
                )
                # a control-flow eqn inside a jit body may itself hide jits
                rewritten = _inline_jits_in_control_flow_params(eqn)
                if rewritten is not None:
                    replace_kwargs['params'] = rewritten[0]
                    replace_kwargs['invars'] = [*hoist(rewritten[1]), *replace_kwargs['invars']]
                new_eqns.append(eqn.replace(**replace_kwargs))
        return [res(v) for v in inner.outvars]

    for eqn in jaxpr.eqns:
        if is_jit_primitive(eqn) and 'jaxpr' in eqn.params:
            outs = splice(
                eqn.params['jaxpr'],
                [resolve_top(v) for v in eqn.invars],
            )
            for outer_ov, out_atom in zip(eqn.outvars, outs):
                top_subst[outer_ov] = out_atom
        else:
            rewritten = _inline_jits_in_control_flow_params(eqn)
            new_invars = [resolve_top(v) for v in eqn.invars]
            if rewritten is not None:
                new_eqns.append(eqn.replace(
                    invars=[*hoist(rewritten[1]), *new_invars], params=rewritten[0]))
            elif all(a is b for a, b in zip(new_invars, eqn.invars)):
                new_eqns.append(eqn)
            else:
                new_eqns.append(eqn.replace(invars=new_invars))

    new_outvars = [resolve_top(v) for v in jaxpr.outvars]
    new_jaxpr = Jaxpr(
        constvars=list(jaxpr.constvars) + extra_constvars,
        invars=list(jaxpr.invars),
        outvars=new_outvars,
        eqns=new_eqns,
        effects=jaxpr.effects,
        debug_info=jaxpr.debug_info,
    )
    return ClosedJaxpr(new_jaxpr, list(closed_jaxpr.consts) + extra_consts)
