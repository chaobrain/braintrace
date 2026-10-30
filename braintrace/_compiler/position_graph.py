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

r"""Position-adjacency analysis for SnAp-n (``recurrence_scope='sparse_n'``).

SnAp-n (Menick et al., 2021) fixes the influence matrix
:math:`J^t = \partial h^t / \partial \theta` to the sparsity pattern of its
instantaneous term propagated :math:`n - 1` times through the hidden-to-hidden
Jacobian, and masks every update back onto that pattern. This module derives
the pattern -- which hidden *position* can influence which, within one hidden
group -- from the group's own transition jaxpr, so the user supplies only the
integer ``n``.

Representation
--------------

The one-step dependency ``A[p, q]`` ("``h_p^t`` may depend on ``h_q^{t-1}``")
is stored **per axis** of the group's ``varshape``, one square boolean pattern
per axis, with the flat adjacency their Kronecker product. That is not an
optimisation: a hidden state of shape ``(batch, n_rec)`` has a leading axis
across which coupling is structurally impossible, and a flat all-to-all pattern
would inflate the neighbourhood width -- and hence the trace -- by a factor of
``batch``.

The factorised :math:`n`-step closure :math:`\bigvee_{k<n} A^k` is **exact when
at most one axis is non-identity** and a conservative superset otherwise, since
:math:`\bigvee_k \bigotimes_i A_i^k \subseteq \bigotimes_i \bigvee_k A_i^k`
(the right-hand side also admits combinations that take a different number of
steps on different axes). Both registered adjacency rules produce at most one
non-identity axis, so on those the factorisation is exact.

Soundness
---------

Every uncertain case widens to all-to-all. A **superset** neighbourhood costs
memory and moves the learning rule *towards* exact; a subset would silently
compute a different rule than the one the caller asked for. The analysis
returns a precise pattern only when all of the following hold, and reports
which condition failed otherwise:

1. Exactly one position-mixing equation is reachable from the hidden invars.
2. Every *other* equation reachable from the hidden invars is elementwise and
   position-preserving -- no reshape, transpose, reduction, gather, ``cond``
   or call-like equation. A reshape between the mixing equation and the
   hidden state relabels the axes an axis-indexed pattern is attached to; a
   reduction couples every position without being a mixing primitive at all.
   A ``scan`` or ``while`` on the hidden path is followed into its body, which
   is held to the same two rules wherever the hidden path runs through it
   (issue #210): a loop whose body passes this check composes a
   position-diagonal step any number of times, so it is position-diagonal as a
   whole, as the same loop unrolled. The rolled pattern is the unrolled one,
   or a superset of it where what is reached or live differs between
   iterations: the body is held to the rules for every iteration at once, so
   an equation reached only in the last iteration (or live only in the first)
   counts as on the hidden path in all of them. A mixing equation inside a
   loop that runs more than once applies once per iteration and transfers a
   power of its pattern (a single ``dot`` inside a length-``L`` scan transfers
   :math:`A^L`, not :math:`A`), so it is refused, as two mixing equations are
   in the unrolled loop. So are hidden-derived values scanned over as ``xs``
   or stacked as ``ys`` on the hidden path (their leading axis counts
   iterations, not positions), and a ``while`` whose trip count reads the
   hidden state (the number of iterations the pattern is taken over is then
   not fixed).
3. That equation's primitive registered a ``snap_adjacency`` rule, and the rule
   accepted the equation (see :data:`~braintrace._op.ETP_RULES_SNAP_ADJACENCY`).
4. Its ``x`` and ``y`` both have exactly the group's ``varshape``, so the
   rule's last-axis pattern refers to the group's own last position axis.
"""

from __future__ import annotations

import itertools
from typing import List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import numpy as np

from braintrace._compatible_imports import (
    Jaxpr,
    JaxprEqn,
    Literal,
    Var,
    is_scan_primitive,
    is_while_primitive,
    scan_num_consts_carry,
)
from braintrace._misc import NotSupportedError
from braintrace._op import (
    get_snap_adjacency_rule,
    get_x_invar_index,
    get_y_outvar_index,
    is_etp_enable_gradient_primitive,
    is_etp_primitive,
)
from .jaxpr_graph import _RECURRENT_WEIGHT_MIXING_PRIMITIVES

__all__ = [
    'AxisAdjacency',
    'SnapPattern',
    'analyze_position_adjacency',
    'build_snap_pattern',
    'close_adjacency',
    'flat_adjacency',
]

#: Default ceiling on the widened block Jacobian ``Dg``, in elements:
#: ``num_position * (num_neighbour * num_state) ** 2``.
#:
#: ``Dg`` -- not the trace -- is what the ceiling has to be read against,
#: because it is the only quantity that grows *quadratically* in ``K``. The
#: trace widens by a factor ``K·S`` per leaf; ``Dg`` is ``P·(K·S)²``, so at
#: ``P = K = 512, S = 1`` a linear ``P·K`` budget of ``1 << 18`` passes exactly
#: while the allocation is 134M float32 values (512 MiB) per operator. The
#: default here admits ``P = K = 256, S = 1`` (16.7M elements, 64 MiB) and
#: rejects the 512 case, which is a configuration error worth naming rather
#: than an out-of-memory crash deep inside the Jacobian extraction.
DEFAULT_MAX_JACOBIAN_ELEMENTS = 1 << 24

# Equations that map position ``p`` of their input to position ``p`` of their
# output and touch no other position. Deliberately an allow-list: a deny-list
# would be unsound, because shape preservation alone does not imply position
# preservation (``rev`` reverses an axis, ``transpose`` of a square array
# permutes positions, ``sort`` reorders them -- all shape-preserving).
_POSITION_PRESERVING_PRIMITIVES = frozenset({
    # arithmetic
    'add', 'add_any', 'sub', 'mul', 'div', 'rem', 'pow', 'integer_pow',
    'neg', 'abs', 'sign', 'max', 'min', 'nextafter',
    # exponential / logarithmic / power
    'exp', 'exp2', 'expm1', 'log', 'log1p', 'logistic', 'sqrt', 'rsqrt', 'cbrt',
    'square',
    # trigonometric / hyperbolic
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    # special
    'erf', 'erfc', 'erf_inv', 'lgamma', 'digamma',
    # rounding / clamping / selection
    'floor', 'ceil', 'round', 'clamp', 'select_n',
    # comparison / logical
    'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'and', 'or', 'xor', 'not',
    'is_finite',
    # structural no-ops
    'convert_element_type', 'stop_gradient', 'copy', 'real', 'imag', 'conj',
})


class AxisAdjacency(NamedTuple):
    """One-step position dependency of a hidden group, factorised per axis.

    Attributes
    ----------
    axes : tuple of numpy.ndarray
        One square boolean pattern per axis of the group's ``varshape``.
        ``axes[i][p, q]`` is ``True`` iff position index ``p`` on axis ``i`` may
        depend on index ``q`` on that axis after one transition step. Empty for
        a group whose ``varshape`` is ``()``.
    conservative : bool
        ``True`` when the analysis could not derive the pattern and widened to
        all-to-all.
    reason : str
        Why the analysis was conservative; the empty string when it was not.
    """

    axes: Tuple[np.ndarray, ...]
    conservative: bool
    reason: str


class SnapPattern(NamedTuple):
    """The :math:`n`-step neighbourhood a SnAp-n trace is widened onto.

    Attributes
    ----------
    n : int
        The SnAp order this pattern was built for.
    varshape : tuple of int
        The hidden group's position layout.
    axes : tuple of numpy.ndarray
        The **closed** per-axis adjacency :math:`\\bigvee_{k<n} A_i^k`.
    neighbours : numpy.ndarray
        ``(num_position, num_neighbour)`` integer index. ``neighbours[p, k]`` is
        the flat position index of ``p``'s ``k``-th neighbour;
        ``neighbours[p, 0] == p`` always. Padded slots repeat ``p`` itself, so a
        gather on this array can never read out of bounds.
    valid : numpy.ndarray
        ``(num_position, num_neighbour)`` boolean mask; ``False`` marks a padded
        slot, whose row and column in the widened transition operator -- and
        whose learning signal -- are zeroed.
    conservative : bool
        Whether the underlying one-step analysis widened to all-to-all.
    reason : str
        Why, if it did.
    """

    n: int
    varshape: Tuple[int, ...]
    axes: Tuple[np.ndarray, ...]
    neighbours: np.ndarray
    valid: np.ndarray
    conservative: bool
    reason: str

    @property
    def num_position(self) -> int:
        """int : Number of hidden positions in the group, ``prod(varshape)``."""
        return int(self.neighbours.shape[0])

    @property
    def num_neighbour(self) -> int:
        """int : Neighbourhood width ``K``; the trace's state axis widens by this."""
        return int(self.neighbours.shape[1])

    @property
    def is_degenerate(self) -> bool:
        """bool : Whether the neighbourhood collapsed to the position itself.

        A degenerate pattern makes ``sparse_n`` compute exactly what ``coupled``
        computes. That is legitimate on a model with no cross-position coupling
        (the truncation has nothing to retain), but it is worth reporting, since
        it is also what a silently failing analysis would produce.
        """
        return self.num_neighbour == 1

    @property
    def is_saturated(self) -> bool:
        """bool : Whether every position sees every other -- full within-group RTRL."""
        return self.num_neighbour == self.num_position and bool(np.all(self.valid))


# -----------------------------------------------------------------------------
# jaxpr walk
# -----------------------------------------------------------------------------

def _is_mixing(eqn: JaxprEqn) -> bool:
    """Whether *eqn* is a candidate cross-position weight-mixing equation.

    Mirrors the classification the grouping compiler already applies: the
    structural-marker ETP primitives (``etp_mm``, ``etp_conv``, ``etp_sp_mv``,
    ...) plus the raw mixing primitives. ``etp_elemwise`` is gradient-enabled
    and elementwise, so it is *not* mixing.
    """
    if eqn.primitive.name in _RECURRENT_WEIGHT_MIXING_PRIMITIVES:
        return True
    return (
        is_etp_primitive(eqn.primitive)
        and not is_etp_enable_gradient_primitive(eqn.primitive)
    )


def _is_position_preserving(eqn: JaxprEqn, varshape: Tuple[int, ...]) -> bool:
    """Whether *eqn* maps each position to itself and touches no other.

    Requires an allow-listed elementwise primitive whose reachable operands and
    outputs all carry ``varshape`` as their leading axes (scalars, which JAX
    admits as broadcast operands, are exempt).
    """
    if is_etp_primitive(eqn.primitive):
        if not is_etp_enable_gradient_primitive(eqn.primitive):
            return False
    elif eqn.primitive.name not in _POSITION_PRESERVING_PRIMITIVES:
        return False
    rank = len(varshape)
    for v in list(eqn.invars) + list(eqn.outvars):
        aval = getattr(v, 'aval', None)
        shape = tuple(getattr(aval, 'shape', ()))
        if shape == ():
            continue  # a scalar operand broadcasts to every position alike
        if len(shape) < rank or shape[:rank] != varshape:
            return False
    return True


def _aligned_shape(var: Union[Var, Literal], varshape: Tuple[int, ...]) -> bool:
    """Whether *var* has exactly the group's position layout."""
    aval = getattr(var, 'aval', None)
    return tuple(getattr(aval, 'shape', ())) == varshape


def _axis_patterns(varshape: Tuple[int, ...], last: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Identity on every axis but the last, *last* on the last."""
    return tuple(
        [np.eye(d, dtype=bool) for d in varshape[:-1]] + [np.asarray(last, dtype=bool)]
    )


def _all_full(varshape: Tuple[int, ...]) -> Tuple[np.ndarray, ...]:
    return tuple(np.ones((d, d), dtype=bool) for d in varshape)


def _all_identity(varshape: Tuple[int, ...]) -> Tuple[np.ndarray, ...]:
    return tuple(np.eye(d, dtype=bool) for d in varshape)


def analyze_position_adjacency(
    transition_jaxpr: Jaxpr,
    varshape: Tuple[int, ...],
    *,
    hidden_invars: Optional[Sequence[Var]] = None,
) -> AxisAdjacency:
    """Derive a hidden group's one-step position adjacency from its transition.

    Parameters
    ----------
    transition_jaxpr : Jaxpr
        The group's transition jaxpr, whose ``invars`` are the hidden states at
        the previous step and whose ``constvars`` carry the inputs (the layout
        :meth:`~braintrace._compiler.hidden_group.HiddenGroup.transition` calls
        it with).
    varshape : tuple of int
        The group's position layout.
    hidden_invars : sequence of Var, optional
        The variables to seed forward reachability from. Default ``None``, i.e.
        ``transition_jaxpr.invars``.

    Returns
    -------
    AxisAdjacency
        The per-axis one-step patterns, together with whether the analysis had
        to widen to all-to-all and why.

    Notes
    -----
    The result may over-approximate the true dependency but never
    under-approximates it -- see the module docstring.

    Examples
    --------
    .. code-block:: python

        >>> import jax, jax.numpy as jnp, braintrace
        >>> w = jnp.ones((4, 4))
        >>> jaxpr = jax.make_jaxpr(
        ...     lambda h: jnp.tanh(braintrace.matmul(h, w)))(jnp.zeros((1, 4))).jaxpr
        >>> adj = analyze_position_adjacency(jaxpr, (1, 4))
        >>> adj.conservative
        False
        >>> [a.shape for a in adj.axes]
        [(1, 1), (4, 4)]
    """
    varshape = tuple(int(d) for d in varshape)
    if not varshape:
        # a single position: nothing can couple to anything
        return AxisAdjacency(axes=(), conservative=False, reason='')

    seeds = list(transition_jaxpr.invars if hidden_invars is None else hidden_invars)
    reachable: Set[Var] = {v for v in seeds if isinstance(v, Var)}
    used = _read_vars(transition_jaxpr.eqns, transition_jaxpr.outvars)
    walk = _Walk(varshape)
    reason = walk.run(transition_jaxpr.eqns, reachable, used)
    if reason is not None:
        return AxisAdjacency(axes=_all_full(varshape), conservative=True, reason=reason)
    if walk.last is None:
        # No cross-position coupling exists at all: every position evolves on its
        # own. Legitimate, and exactly what two_state_rnn-style models produce.
        return AxisAdjacency(axes=_all_identity(varshape), conservative=False, reason='')
    return AxisAdjacency(
        axes=_axis_patterns(varshape, walk.last), conservative=False, reason=''
    )


def _read_vars(eqns: Sequence[JaxprEqn], outvars: Sequence[Union[Var, Literal]]) -> Set[Var]:
    """Every variable *eqns* read, plus the variables in *outvars*."""
    read = {v for e in eqns for v in e.invars if isinstance(v, Var)}
    read.update(v for v in outvars if isinstance(v, Var))
    return read


def _live_eqns(body: Jaxpr, roots: Set[Var]) -> Tuple[List[JaxprEqn], Set[Var]]:
    """The body equations that contribute to *roots*, in order, and every
    variable they read (with *roots*)."""
    needed = set(roots)
    live: List[JaxprEqn] = []
    for eqn in reversed(body.eqns):
        if any(isinstance(o, Var) and o in needed for o in eqn.outvars):
            live.append(eqn)
            needed.update(v for v in eqn.invars if isinstance(v, Var))
    live.reverse()
    return live, needed


class _Walk:
    """Forward walk of the hidden path through a transition, with one rule
    for every equation it reaches, at top level or inside a loop body.

    ``mixing`` lists the position-mixing equations reached, of which at most
    one is allowed; ``last`` is its last-axis pattern.
    """

    def __init__(self, varshape: Tuple[int, ...], loop: Optional[str] = None):
        self.varshape = varshape
        #: the loop whose body this walk covers, when that body runs more than
        #: once: a mixing equation there is refused, naming this loop even
        #: when it sits in a body nested inside it that runs once
        self.loop = loop
        self.mixing: List[JaxprEqn] = []
        self.last: Optional[np.ndarray] = None

    def run(
        self,
        eqns: Sequence[JaxprEqn],
        reachable: Set[Var],
        used: Set[Var],
        where: str = '',
    ) -> Optional[str]:
        """Walk *eqns* in order, growing *reachable*; a reason string when the
        pattern cannot be derived, else ``None``.

        *used* holds the variables read downstream of *eqns* (a loop's outputs
        outside it are only on the hidden path when they are used); *where*
        names the enclosing loop body in the reasons.
        """
        for eqn in eqns:
            hits = [
                j for j, v in enumerate(eqn.invars)
                if isinstance(v, Var) and v in reachable
            ]
            if not hits:
                continue
            if is_scan_primitive(eqn):
                outcome = self._scan(eqn, hits, used)
            elif is_while_primitive(eqn):
                outcome = self._while(eqn, hits, used)
            elif _is_mixing(eqn) and self.loop is not None:
                return (
                    f"the position-mixing equation '{eqn.primitive.name}' sits inside a "
                    f"'{self.loop}' body"
                    f"{'' if self.loop == 'while' else ' that runs more than once'}: it "
                    f"applies once per iteration, so the loop transfers a power of its "
                    f"pattern, not the pattern."
                )
            elif _is_mixing(eqn):
                self.mixing.append(eqn)
                if len(self.mixing) > 1:
                    return (
                        f"the transition applies more than one position-mixing equation "
                        f"reachable from the hidden state "
                        f"({', '.join(e.primitive.name for e in self.mixing)}); their "
                        f"composed transfer is not the pattern of either one."
                    )
                # Validate the mixing equation where it is found, so the reason
                # names it rather than whatever incidental downstream equation
                # trips next.
                pattern = _mixing_axis_pattern(eqn, self.varshape, set(hits))
                if isinstance(pattern, str):
                    return pattern
                self.last = pattern
                outcome = [v for v in eqn.outvars if isinstance(v, Var)]
            elif not _is_position_preserving(eqn, self.varshape):
                return (
                    f"equation '{eqn.primitive.name}' on the hidden path{where} is not "
                    f"an elementwise position-preserving operation, so it may relabel or "
                    f"couple positions in a way an axis-indexed pattern cannot express."
                )
            else:
                outcome = [v for v in eqn.outvars if isinstance(v, Var)]
            if isinstance(outcome, str):
                return outcome
            reachable.update(outcome)
        return None

    def _body(
        self,
        live: List[JaxprEqn],
        needed: Set[Var],
        seeds: Set[Var],
        feedback: List[Tuple[Var, Var]],
        once: bool,
        loop: str,
    ) -> Union[str, Set[Var]]:
        """Walk the *live* equations of a loop body from *seeds*; the body
        variables reached, or a reason string.

        *feedback* pairs each carried output with the carried input it becomes
        in the next iteration. A body that runs once (*once*) is walked as
        straight-line code, sharing this walk's mixing count, as the unrolled
        loop would be. Otherwise reachability is closed over the iterations (a
        carried output reached in one iteration is a reached carried input in
        the next), and a mixing equation reached is refused: it applies once
        per iteration.
        """
        where = f" inside a '{loop}' body"
        if once:
            reached = set(seeds)
            reason = self.run(live, reached, needed, where)
            return reason if reason is not None else reached
        while True:
            reached = set(seeds)
            reason = _Walk(self.varshape, loop).run(live, reached, needed, where)
            if reason is not None:
                return reason
            fed = {inp for out, inp in feedback if isinstance(out, Var) and out in reached}
            if fed <= seeds:
                return reached
            seeds = seeds | fed

    def _scan(self, eqn: JaxprEqn, hits: Sequence[int], used: Set[Var]) -> Union[str, List[Var]]:
        """The scan outputs on the hidden path, or a reason string."""
        num_consts, num_carry = scan_num_consts_carry(eqn)
        length = int(eqn.params['length'])
        body = eqn.params['jaxpr'].jaxpr
        n_in = num_consts + num_carry
        carried = range(num_carry)
        if length == 0:
            # the body never runs: the carried outputs are the inits
            return [eqn.outvars[c] for c in carried if num_consts + c in hits]
        ys_used = [i for i in range(len(eqn.outvars) - num_carry)
                   if eqn.outvars[num_carry + i] in used]
        roots = {v for v in [body.outvars[c] for c in carried if eqn.outvars[c] in used]
                 + [body.outvars[num_carry + i] for i in ys_used] if isinstance(v, Var)}
        feedback = [(body.outvars[c], body.invars[num_consts + c]) for c in carried]
        once = length == 1
        live, needed = _live_eqns(body, roots if once else _closed_roots(body, roots, feedback))
        if any(body.invars[j] in needed for j in hits if j >= n_in):
            return (
                "a 'scan' iterates over hidden-derived values as xs on the hidden "
                "path: their leading axis counts iterations, not positions."
            )
        seeds = {body.invars[j] for j in hits if j < n_in}
        reached = self._body(live, needed, seeds, feedback, once, 'scan')
        if isinstance(reached, str):
            return reached
        if any(_in(body.outvars[num_carry + i], reached) for i in ys_used):
            return (
                "a 'scan' stacks hidden-derived values over its iterations as ys on "
                "the hidden path: their leading axis counts iterations, not positions."
            )
        return [eqn.outvars[c] for c in carried if _in(body.outvars[c], reached)]

    def _while(self, eqn: JaxprEqn, hits: Sequence[int], used: Set[Var]) -> Union[str, List[Var]]:
        """The while outputs on the hidden path, or a reason string."""
        cond_nconsts = eqn.params['cond_nconsts']
        body_nconsts = eqn.params['body_nconsts']
        body = eqn.params['body_jaxpr'].jaxpr
        cond = eqn.params['cond_jaxpr'].jaxpr
        num_carry = len(eqn.outvars)
        trip = (
            "the trip count of a 'while' reads the hidden state, so the number of "
            "iterations the pattern is taken over is not fixed: the analysis keeps "
            "every position."
        )
        if any(j < cond_nconsts for j in hits):
            return trip
        cond_reads = _read_vars(cond.eqns, cond.outvars)
        cond_carry = [c for c in range(num_carry) if cond.invars[cond_nconsts + c] in cond_reads]
        # the trip count is computed every iteration: what it reads is live
        roots = {body.outvars[c] for c in range(num_carry)
                 if (eqn.outvars[c] in used or c in cond_carry)
                 and isinstance(body.outvars[c], Var)}
        feedback = [(body.outvars[c], body.invars[body_nconsts + c]) for c in range(num_carry)]
        live, needed = _live_eqns(body, _closed_roots(body, roots, feedback))
        seeds = {body.invars[j - cond_nconsts] for j in hits if j >= cond_nconsts}
        reached = self._body(live, needed, seeds, feedback, False, 'while')
        if isinstance(reached, str):
            return reached
        # a carry reached in any iteration, or at the start, reaches the cond
        # (a hit carry init is a seed, so its body invar is reached)
        carry_reached = {c for c in range(num_carry)
                         if body.invars[body_nconsts + c] in reached
                         or _in(body.outvars[c], reached)}
        if carry_reached & set(cond_carry):
            return trip
        # the loop may not run at all: a reached init is a reached output
        return [eqn.outvars[c] for c in sorted(carry_reached)]


def _in(atom: Union[Var, Literal], variables: Set[Var]) -> bool:
    """Whether *atom* is one of *variables* (a literal never is)."""
    return isinstance(atom, Var) and atom in variables


def _closed_roots(body: Jaxpr, roots: Set[Var], feedback: List[Tuple[Var, Var]]) -> Set[Var]:
    """The body outputs used in any iteration of a loop, given the *roots*
    used after it: a carried input the body reads makes the carried output
    feeding it used in the previous iteration."""
    while True:
        _, needed = _live_eqns(body, roots)
        grown = {out for out, inp in feedback if inp in needed and isinstance(out, Var)}
        if grown <= roots:
            return roots
        roots = roots | grown


def _mixing_axis_pattern(
    eqn: JaxprEqn,
    varshape: Tuple[int, ...],
    hit_positions: Set[int],
) -> Union[str, np.ndarray]:
    """The last-axis pattern of a mixing equation, or a reason string.

    Returns the boolean ``(d, d)`` pattern on success and a human-readable
    reason for widening to all-to-all on any failure. Checks run
    most-specific-first so the reason names the actual obstacle: no registered
    rule, then the hidden state entering somewhere other than ``x``, then a
    position layout that is not the group's, then a declining or malformed rule.
    """
    rule = get_snap_adjacency_rule(eqn.primitive)
    if rule is None:
        return (
            f"the mixing equation '{eqn.primitive.name}' registered no "
            f"snap_adjacency rule, so its cross-position coupling is unknown."
        )
    x_index = get_x_invar_index(eqn.primitive)
    if x_index is None or hit_positions != {x_index}:
        return (
            f"the hidden state reaches the mixing equation "
            f"'{eqn.primitive.name}' at invar position(s) "
            f"{sorted(hit_positions)}, not only at its declared 'x' input."
        )
    x_var = eqn.invars[x_index]
    y_var = eqn.outvars[get_y_outvar_index(eqn.primitive)]
    if not (_aligned_shape(x_var, varshape) and _aligned_shape(y_var, varshape)):
        return (
            f"the mixing equation '{eqn.primitive.name}' has x shape "
            f"{tuple(getattr(x_var.aval, 'shape', ()))} and y shape "
            f"{tuple(getattr(y_var.aval, 'shape', ()))}; both must equal the "
            f"group's varshape {varshape} for the rule's axes to be the group's "
            f"axes."
        )
    pattern = rule(dict(eqn.params), varshape[-1])
    if pattern is None:
        return (
            f"the snap_adjacency rule of '{eqn.primitive.name}' declined this "
            f"equation."
        )
    pattern = np.asarray(pattern, dtype=bool)
    if pattern.shape != (varshape[-1], varshape[-1]):
        return (
            f"the snap_adjacency rule of '{eqn.primitive.name}' returned a "
            f"{pattern.shape} pattern, expected {(varshape[-1], varshape[-1])}."
        )
    return pattern


# -----------------------------------------------------------------------------
# closure and neighbourhood
# -----------------------------------------------------------------------------

def close_adjacency(
    axes: Sequence[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, ...]:
    r"""Close each per-axis pattern over :math:`n` SnAp steps.

    Computes :math:`\bigvee_{k=0}^{n-1} A^k` per axis, so ``n = 1`` is the
    identity (SnAp-1 retains the instantaneous pattern only, propagating it zero
    times) and larger ``n`` admits paths of up to ``n - 1`` steps.

    Parameters
    ----------
    axes : sequence of numpy.ndarray
        The one-step per-axis boolean patterns.
    n : int
        The SnAp order; must be at least 1.

    Returns
    -------
    tuple of numpy.ndarray
        The closed per-axis patterns.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f'SnAp order n must be at least 1, got {n}.')
    closed: List[np.ndarray] = []
    for a in axes:
        a = np.asarray(a, dtype=bool)
        out = np.eye(a.shape[0], dtype=bool)
        power = out.copy()
        for _ in range(n - 1):
            # Boolean matmul, *not* an integer one. NumPy's bool `@` accumulates
            # with logical-or, so it saturates; casting to uint8 first would
            # count paths and wrap modulo 256, turning a position pair joined by
            # exactly 256 parallel two-hop paths into "unreachable" -- a silent
            # under-approximation of the neighbourhood, which is the one error
            # mode this analysis must never have.
            power = (power @ a)
            new = out | power
            if np.array_equal(new, out):
                break  # saturated: further steps add nothing
            out = new
        closed.append(out)
    return tuple(closed)


def flat_adjacency(axes: Sequence[np.ndarray]) -> np.ndarray:
    """Assemble the flat ``(P, P)`` adjacency from the per-axis patterns.

    The flat index is C-order over ``varshape``, which is exactly the ordering
    the Kronecker product produces.

    Parameters
    ----------
    axes : sequence of numpy.ndarray
        The per-axis boolean patterns.

    Returns
    -------
    numpy.ndarray
        The ``(P, P)`` boolean adjacency, or the ``1 x 1`` identity when there
        are no axes.
    """
    out = np.ones((1, 1), dtype=bool)
    for a in axes:
        out = np.kron(out, np.asarray(a, dtype=bool))
    return out


def build_snap_pattern(
    transition_jaxpr: Jaxpr,
    varshape: Tuple[int, ...],
    n: int,
    *,
    num_state: int = 1,
    hidden_invars: Optional[Sequence[Var]] = None,
    max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
) -> SnapPattern:
    """Build the SnAp-n neighbourhood index for one hidden group.

    Parameters
    ----------
    transition_jaxpr : Jaxpr
        The group's transition jaxpr.
    varshape : tuple of int
        The group's position layout.
    n : int
        The SnAp order; ``n = 1`` yields the degenerate single-neighbour pattern
        that reproduces ``recurrence_scope='coupled'``.
    num_state : int, optional
        States per position, ``S``. Only the memory guard uses it, and only
        because the widened block Jacobian is ``P·(K·S)²`` -- a group with
        several states per position pays ``S²`` on top of the widening.
        Default ``1``.
    hidden_invars : sequence of Var, optional
        Seeds for the reachability sweep. Default ``None`` (the jaxpr's invars).
    max_jacobian_elements : int, optional
        Ceiling on ``num_position * (num_neighbour * num_state) ** 2``. Default
        :data:`DEFAULT_MAX_JACOBIAN_ELEMENTS`.

    Returns
    -------
    SnapPattern
        The closed adjacency plus the padded neighbour index and validity mask.

    Raises
    ------
    ValueError
        If ``n < 1``.
    NotSupportedError
        If the widened block Jacobian would exceed *max_jacobian_elements*.
    """
    varshape = tuple(int(d) for d in varshape)
    adj = analyze_position_adjacency(
        transition_jaxpr, varshape, hidden_invars=hidden_invars
    )
    closed = close_adjacency(adj.axes, n)
    num_position = int(np.prod(varshape)) if varshape else 1

    if not closed:
        return SnapPattern(
            n=n,
            varshape=varshape,
            axes=(),
            neighbours=np.zeros((1, 1), dtype=np.int32),
            valid=np.ones((1, 1), dtype=bool),
            conservative=adj.conservative,
            reason=adj.reason,
        )

    # Per-axis neighbour lists, taken **column-wise**.
    #
    # ``closed[i][r, p]`` reads "r depends on p" (see AxisAdjacency), but the
    # trace slot for a parameter anchored at ``p`` has to cover the positions
    # that ``p`` *influences*: SnAp-n is the instantaneous pattern -- nonzero at
    # ``p`` alone -- propagated n-1 times through D, which spreads it to every
    # ``r`` with ``(A^k)[r, p]``. That is column ``p``, not row ``p``. Taking the
    # row instead is not a mere relabelling: on a directed graph it retains the
    # positions ``p`` reads *from*, whose influence on ``p``'s parameter is
    # exactly zero, so the trace pays the full widening and the gradient does
    # not improve at all until the neighbourhood saturates.
    #
    # K is the max over positions of the product of per-axis counts, and because
    # the position indices range over the full Cartesian product that max is
    # exactly the product of the per-axis maxima.
    axis_neighbours = [
        [np.flatnonzero(pattern[:, i]) for i in range(pattern.shape[1])]
        for pattern in closed
    ]
    num_neighbour = 1
    for lists in axis_neighbours:
        num_neighbour *= max(len(entry) for entry in lists)

    num_state = int(num_state)
    jacobian_elements = num_position * (num_neighbour * num_state) ** 2
    if jacobian_elements > max_jacobian_elements:
        raise NotSupportedError(
            f"recurrence_scope='sparse_n' with sparse_n={n} widens this hidden "
            f"group's trace state axis by K={num_neighbour} over "
            f"P={num_position} positions (varshape={varshape}, "
            f'num_state={num_state}), so its block Jacobian is '
            f'P*(K*S)**2 = {jacobian_elements} elements, above the '
            f'{max_jacobian_elements} ceiling. '
            + (
                'The position analysis was conservative here '
                f'({adj.reason}) so K is the whole group; '
                if adj.conservative else ''
            )
            + "Use a smaller sparse_n, recurrence_scope='coupled', or pass a "
              'larger snap_max_jacobian_elements if the memory is affordable.'
        )

    strides = [int(s) for s in np.array(varshape[1:] + (1,))[::-1].cumprod()[::-1]]
    neighbours = np.zeros((num_position, num_neighbour), dtype=np.int32)
    valid = np.zeros((num_position, num_neighbour), dtype=bool)
    for flat_p in range(num_position):
        idx = np.unravel_index(flat_p, varshape)
        per_axis = [axis_neighbours[i][idx[i]] for i in range(len(varshape))]
        flat_q = sorted(
            int(sum(int(q) * s for q, s in zip(combo, strides)))
            for combo in itertools.product(*per_axis)
        )
        # self first, so slot 0 always carries the instantaneous term
        flat_q.remove(flat_p)
        flat_q.insert(0, flat_p)
        neighbours[flat_p, :len(flat_q)] = flat_q
        valid[flat_p, :len(flat_q)] = True
        neighbours[flat_p, len(flat_q):] = flat_p  # in-range dummy for the gather

    return SnapPattern(
        n=n,
        varshape=varshape,
        axes=closed,
        neighbours=neighbours,
        valid=valid,
        conservative=adj.conservative,
        reason=adj.reason,
    )
