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
from typing import Any, Dict, List

from braintrace._compatible_imports import (
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Var,
    is_jit_primitive,
)

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


def inline_jit_calls(closed_jaxpr: ClosedJaxpr) -> ClosedJaxpr:
    """Splice every top-level ``jit``/``pjit`` body into the enclosing jaxpr.

    The ETrace analyses identify weights, hidden states, and ETP primitives by
    ``Var`` identity in one flat jaxpr; a ``jit`` call boundary hides its body
    behind fresh inner variables, so weights or hidden states used inside a
    user's ``jax.jit``-wrapped function were previously either rejected or
    silently excluded. Inlining removes the boundary before any analysis runs.

    The pass is recursive (``jit`` inside ``jit`` is flattened), lifts inner
    closure constants into the outer ``constvars``/``consts``, and rewires
    pass-through outputs by substitution. Control-flow bodies
    (``scan``/``while``/``cond``) and custom-derivative call primitives are
    left untouched: control flow is diagnosed separately, and
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
    if not _contains_jit_eqn(jaxpr):
        return closed_jaxpr

    subst: Dict[Var, Any] = {}  # Var -> Var | Literal

    def resolve(atom):
        while isinstance(atom, Var) and atom in subst:
            atom = subst[atom]
        return atom

    extra_constvars: List[Var] = []
    extra_consts: List[Any] = []
    new_eqns: List[JaxprEqn] = []

    def process(eqns) -> None:
        for eqn in eqns:
            if is_jit_primitive(eqn) and 'jaxpr' in eqn.params:
                inner_closed = eqn.params['jaxpr']
                inner = getattr(inner_closed, 'jaxpr', inner_closed)
                inner_consts = getattr(inner_closed, 'consts', [])
                # Inner constvars are fresh Var objects: adopt them directly
                # as outer constvars, carrying their values.
                extra_constvars.extend(inner.constvars)
                extra_consts.extend(inner_consts)
                # Wire the inner parameters to the (resolved) call arguments.
                for iv, arg in zip(inner.invars, eqn.invars):
                    subst[iv] = resolve(arg)
                # Splice the body (recursively inlining nested jit).
                process(inner.eqns)
                # Wire the call results to the inner outputs.
                for outer_ov, inner_ov in zip(eqn.outvars, inner.outvars):
                    subst[outer_ov] = resolve(inner_ov)
            else:
                new_invars = [
                    resolve(v) if isinstance(v, Var) else v
                    for v in eqn.invars
                ]
                if all(a is b for a, b in zip(new_invars, eqn.invars)):
                    new_eqns.append(eqn)
                else:
                    new_eqns.append(eqn.replace(invars=new_invars))

    process(jaxpr.eqns)

    new_outvars = [
        resolve(v) if isinstance(v, Var) else v
        for v in jaxpr.outvars
    ]
    new_jaxpr = Jaxpr(
        constvars=list(jaxpr.constvars) + extra_constvars,
        invars=list(jaxpr.invars),
        outvars=new_outvars,
        eqns=new_eqns,
        effects=jaxpr.effects,
        debug_info=jaxpr.debug_info,
    )
    return ClosedJaxpr(new_jaxpr, list(closed_jaxpr.consts) + extra_consts)
