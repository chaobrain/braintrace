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

r"""Same-step gradients along paths to the output that bypass every new
hidden value (issue #185).

Under ``vjp_method='single-step'`` a parameter's gradient is assembled from
the eligibility trace, ``dL/dh^t \cdot e^t``, where ``dL/dh^t`` is read off the
hidden perturbation. The parameters themselves are constants of the
transposed step. A path from a parameter to a step output that crosses *no*
new hidden value (a pre-reset spike read as the output, a readout weight
applied inside the step, ...) is therefore in neither term, and its
contribution was dropped silently.

The same holds for the *previous* hidden values: an output computed from
``h^{t-1}`` without going through ``h^t`` (a spike read from the pre-reset
membrane potential ``0.9 V^{t-1} + x @ w``) carries the learning signal
``dL^t/dh^{t-1}`` that RTRL pairs with the previous trace ``e^{t-1}``, and the
hidden perturbation at ``h^t`` does not see it either.

This module rewrites the perturbed step so that these *bypass* terms are
computed exactly, and computed alone. The *sources* are the parameter leaves
and the hidden invars, each group under a caller-chosen key:

1. A forward pass finds, for every source, the *cone* of variables it reaches
   without crossing a hidden-state slot or a ``stop_gradient``. A source whose
   cone reaches a merge slot (a model output or a non-hidden state output)
   has a bypass.
2. Each bypass source gets one new invar per leaf (a differentiable copy),
   and every equation of the copies' cone is duplicated. A duplicate consumes
   the copies (or earlier duplicates) for cone inputs and a ``stop_gradient``
   of the original variable for every other input. The original equations are
   untouched.
3. At each merge slot the duplicate ``d`` is attached to the original value
   ``o`` as a pure tangent,
   ``select_n(is_finite(d), o, o + (d - stop_gradient(d)))``. The forward
   value is exactly ``o`` (the guard keeps an overflowing output ``inf``
   rather than ``inf - inf = nan``), and the reverse pass sends the output
   cotangent into the copies along the cone only.

A cone equation that cannot be duplicated is cut, and the sources whose
bypass it carries are reported. That is a ``while`` anywhere inside it (no
reverse-mode rule) or an equation with effects (duplicating it would repeat
the effect).
"""

from typing import Dict, FrozenSet, Hashable, List, NamedTuple, Sequence, Set, Tuple

import jax
import jax.numpy as jnp

from braintrace._compatible_imports import (
    Jaxpr,
    JaxprEqn,
    Literal,
    Var,
    is_while_primitive,
    new_jaxpr_eqn,
    new_var,
    stop_gradient_p,
)
from .hidden_group import _transition_contains_while

__all__ = [
    'OutputBypass',
    'add_output_bypass',
]


class OutputBypass(NamedTuple):
    """The result of :func:`add_output_bypass`.

    Attributes
    ----------
    eqns : list of JaxprEqn
        The equations of the rewritten step: the originals, the cone
        duplicates and the merge equations.
    outvars : list
        The rewritten jaxpr outvars (merge slots point at the merged values).
    bypass_vars : list of Var
        The new invars, one per leaf of each bypass source, ordered by
        ``bypass_keys`` and then by leaf.
    bypass_keys : list
        The sources with a bypass, in ``sources`` order.
    dropped_keys : list
        The sources whose bypass was (partly) cut at a ``while`` or an
        effectful equation.
    """

    eqns: List[JaxprEqn]
    outvars: List
    bypass_vars: List[Var]
    bypass_keys: List[Hashable]
    dropped_keys: List[Hashable]


def _is_blocking(eqn: JaxprEqn) -> bool:
    # A ``while`` (anywhere inside the equation) has no transpose rule, and
    # duplicating an effectful equation would repeat its effect.
    if eqn.effects:
        return True
    return is_while_primitive(eqn) or _transition_contains_while(
        Jaxpr(constvars=[], invars=[], outvars=[], eqns=[eqn]))


def _is_inexact(v: Var) -> bool:
    return jnp.issubdtype(v.aval.dtype, jnp.inexact)


def _find_cones(
    eqns: Sequence[JaxprEqn],
    sources: Dict[Hashable, Sequence[Var]],
    cut_vars: Set[Var],
) -> Tuple[Dict[Var, FrozenSet[Hashable]], Dict[Var, FrozenSet[Hashable]]]:
    """Forward reachability from the sources.

    Returns ``(reach, blocked)``. ``reach[v]`` holds the sources reaching
    ``v`` along duplicable equations; ``blocked[v]`` holds the sources whose
    path to ``v`` crosses a blocking equation.
    """
    reach: Dict[Var, FrozenSet[Hashable]] = {}
    blocked: Dict[Var, FrozenSet[Hashable]] = {}
    for key, invars in sources.items():
        for v in invars:
            if isinstance(v, Var) and v not in cut_vars:
                reach[v] = reach.get(v, frozenset()) | {key}
    for eqn in eqns:
        if eqn.primitive is stop_gradient_p:
            continue
        ins = [v for v in eqn.invars if isinstance(v, Var)]
        r = frozenset().union(*(reach.get(v, frozenset()) for v in ins))
        b = frozenset().union(*(blocked.get(v, frozenset()) for v in ins))
        if _is_blocking(eqn):
            r, b = frozenset(), b | r
        if not (r or b):
            continue
        for ov in eqn.outvars:
            if ov in cut_vars:
                continue
            if r:
                reach[ov] = r
            if b:
                blocked[ov] = b
    return reach, blocked


def add_output_bypass(
    eqns: Sequence[JaxprEqn],
    outvars: Sequence,
    *,
    sources: Dict[Hashable, Sequence[Var]],
    cut_vars: Set[Var],
    merge_positions: Sequence[int],
) -> OutputBypass:
    """Add the exact same-step bypass gradients of the sources to a step.

    Parameters
    ----------
    eqns : sequence of JaxprEqn
        The equations of the (perturbed) step, in order.
    outvars : sequence
        The step's jaxpr outvars.
    sources : dict
        Key to the source's leaf invars (parameter leaves, hidden invars).
    cut_vars : set of Var
        Variables the cones never cross: the hidden-state slots of the step,
        whose same-step credit the eligibility trace already carries.
    merge_positions : sequence of int
        Positions of ``outvars`` where a bypass is attached (model outputs and
        non-hidden state outputs).

    Returns
    -------
    OutputBypass
        The rewritten step. When no source has a bypass, ``eqns`` and
        ``outvars`` are returned unchanged and ``bypass_vars`` is empty.
    """
    eqns = list(eqns)
    outvars = list(outvars)
    reach, blocked = _find_cones(eqns, sources, cut_vars)

    slots = [
        outvars[k] for k in merge_positions
        if isinstance(outvars[k], Var) and _is_inexact(outvars[k])
    ]
    reaching: Set[Hashable] = set()
    dropped: Set[Hashable] = set()
    for v in slots:
        reaching |= reach.get(v, frozenset())
        dropped |= blocked.get(v, frozenset())
    order = list(sources)
    bypass_keys = [k for k in order if k in reaching]
    dropped_keys = [k for k in order if k in dropped]
    if not bypass_keys:
        return OutputBypass(eqns, outvars, [], [], dropped_keys)

    # -- duplicate the cones of the bypass sources' copies -- #
    dup: Dict[Var, Var] = {}
    bypass_vars: List[Var] = []
    for key in bypass_keys:
        for v in sources[key]:
            copy = new_var('', jax.core.ShapedArray(v.aval.shape, v.aval.dtype))
            bypass_vars.append(copy)
            if v not in cut_vars:
                dup[v] = copy

    detached: Dict[Var, Var] = {}
    new_eqns: List[JaxprEqn] = []

    def _detached(v: Var) -> Var:
        if v not in detached:
            fresh = new_var('', jax.core.ShapedArray(v.aval.shape, v.aval.dtype))
            new_eqns.append(new_jaxpr_eqn([v], [fresh], stop_gradient_p, {}, set()))
            detached[v] = fresh
        return detached[v]

    for eqn in eqns:
        new_eqns.append(eqn)
        if eqn.primitive is stop_gradient_p or _is_blocking(eqn):
            continue
        if not any(isinstance(v, Var) and v in dup for v in eqn.invars):
            continue
        if all(ov in cut_vars for ov in eqn.outvars):
            continue
        dup_invars = [
            v if isinstance(v, Literal) else (dup[v] if v in dup else _detached(v))
            for v in eqn.invars
        ]
        dup_outvars = [
            new_var('', jax.core.ShapedArray(ov.aval.shape, ov.aval.dtype))
            for ov in eqn.outvars
        ]
        new_eqns.append(eqn.replace(invars=dup_invars, outvars=dup_outvars))
        for ov, dv in zip(eqn.outvars, dup_outvars):
            if ov not in cut_vars:
                dup[ov] = dv

    # -- attach each duplicate at its merge slots as a pure tangent -- #
    merged: Dict[Var, Var] = {}

    def _like(v: Var, dtype=None) -> Var:
        return new_var('', jax.core.ShapedArray(v.aval.shape, dtype or v.aval.dtype))

    for k in merge_positions:
        v = outvars[k]
        if not (isinstance(v, Var) and v in dup and _is_inexact(v)):
            continue
        if v not in merged:
            d = dup[v]
            d_sg, diff, total, finite, out = _like(d), _like(d), _like(v), _like(d, jnp.bool_), _like(v)
            new_eqns.append(new_jaxpr_eqn([d], [d_sg], stop_gradient_p, {}, set()))
            new_eqns.append(new_jaxpr_eqn([d, d_sg], [diff], jax.lax.sub_p, {}, set()))
            new_eqns.append(new_jaxpr_eqn([v, diff], [total], jax.lax.add_p, {}, set()))
            new_eqns.append(new_jaxpr_eqn([d], [finite], jax.lax.is_finite_p, {}, set()))
            new_eqns.append(new_jaxpr_eqn([finite, v, total], [out], jax.lax.select_n_p, {}, set()))
            merged[v] = out
        outvars[k] = merged[v]

    return OutputBypass(new_eqns, outvars, bypass_vars, bypass_keys, dropped_keys)
