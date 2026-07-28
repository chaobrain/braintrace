# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
#
# ==============================================================================
#
# Author: Chaoming Wang <chao.brain@qq.com>
# Copyright: 2024, Chaoming Wang
# Date: 2024-04-03
#
# Refinement History:
#   [2024-04-03] Created
#   [2024-04-06] Added the traceback information for the error messages.
#   [2024-04-16] Changed the "op" in the "HiddenWeightOpTracer" to "JaxprEqn".
#                Added the support for the "pjit" operator.
#   [2024-05] Add the support for vjp_time == 't_minus_1'
#   [2024-06] Conditionally support control flows, including `scan`, `while`, and `cond`
#   [2024-09] version 0.0.2
#   [2024-11-22] compatible with `brainstate>=0.1.0` (#17)
#   [2024-11-23] Add the support for vjp_time_ahead > 1, it can combine the
#                advantage of etrace learning and backpropagation through time.
#   [2024-11-26] version 0.0.3, a complete new revision for better model debugging.
#   [2024-12-05] change the ETraceWeight to NonETraceWeight if the hidden states are not found;
#                remove the connected hidden states when y=x@w is not shape broadcastable with the hidden states.
#   [2024-12-09] small updates, related to the key items in "CompiledVjpGraph"
#   [2025-02-06]
#       - [x] unify model retrieved states (brainstate.graph.states)
#             and compiled states (brainstate.transform.StatefulFunction)
#       - [x] add the support for the "HiddenGroupState" and "ETraceTreeState"
#       - [x] add the support for the "ElemWiseParam"
#       - [x] split into "_compiler.py", "_etrace_vjp_compiler_graph.py", and "hidden_group.py",
#
# ==============================================================================

# -*- coding: utf-8 -*-

from itertools import combinations
from typing import TYPE_CHECKING
from typing import List, Dict, FrozenSet, Sequence, Tuple, Set, Optional, Callable, NamedTuple, Any, cast

import brainstate
import brainunit as u
import jax.core
import jax.numpy as jnp
import numpy as np
from brainstate import HiddenGroupState

from braintrace._compatible_imports import (
    Var,
    Literal,
    JaxprEqn,
    Jaxpr,
    is_scan_primitive,
    is_while_primitive,
    is_cond_primitive,
    scan_num_consts_carry,
)
from braintrace._op import is_etp_primitive, is_etp_enable_gradient_primitive
from braintrace._misc import NotSupportedError
from braintrace._typing import (
    PyTree,
    HiddenInVar,
    HiddenOutVar,
    Path,
)
from .base import JaxprEvaluation, find_matched_vars
from .canonicalize import ControlFlowPolicy, DEFAULT_CONTROL_FLOW_POLICY
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit
from .module_info import extract_module_info, ModuleInfo
from .position_graph import DEFAULT_MAX_JACOBIAN_ELEMENTS, build_snap_pattern

__all__ = [
    'HiddenGroup',
    'full_position_jacobian',
    'widened_block_jacobian',
    'widen_instant_term',
    'gather_learning_signal',
    'find_hidden_groups_from_minfo',
    'find_hidden_groups_from_module',
]

if TYPE_CHECKING:
    from .scan_descent import GroupDescent  # noqa: F401
    from .position_graph import SnapPattern  # noqa: F401

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
# ETP-boundary skip in ``_eval_eqn``.
_RECURRENT_WEIGHT_MIXING_PRIMITIVES = frozenset({
    'dot_general', 'conv_general_dilated',
})


class HiddenGroup(NamedTuple):
    r"""The data structure recording a hidden-group relation.

    A hidden group bundles the hidden states that are mutually connected through
    a recurrence transition, together with the jaxpr that computes that
    transition

    .. math::

        h_1^t, h_2^t, \ldots = f(h_1^{t-1}, h_2^{t-1}, \ldots, x^t).

    Attributes
    ----------
    index : int
        Position of this group in the compiled group sequence.
    hidden_paths : list of Path
        The module path to each hidden state in the group.
    hidden_states : list of brainstate.HiddenState
        The hidden states in the group.
    hidden_invars : list of HiddenInVar
        The input jaxpr ``Var`` of each hidden state (at the previous step).
    hidden_outvars : list of HiddenOutVar
        The output jaxpr ``Var`` of each hidden state (at the current step).
    transition_jaxpr : Jaxpr
        The jaxpr computing the hidden-state transition for the group.
    transition_jaxpr_constvars : list of Var
        The other input variables required to evaluate ``transition_jaxpr``.
    is_diagonal_recurrence : bool
        Whether the recurrence is diagonal across the leading ``varshape``
        positions (see the field comment for the full contract).
    snap : SnapPattern or None
        The SnAp-n neighbourhood the trace is widened onto
        (``recurrence_scope='sparse_n'``); ``None`` for every other scope.
    descent : GroupDescent or None
        Descent context when this group's transition is one substep of a
        descended scan (Phase 4 structured scan descent); ``None`` for
        ordinary groups.

    See Also
    --------
    find_hidden_groups_from_module : Build hidden groups directly from a model.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> hidden_groups, _ = braintrace.find_hidden_groups_from_module(gru, inputs)
        >>> len(hidden_groups)
        1
    """

    index: int  # type: ignore[assignment]  # intentional NamedTuple field; shadows tuple.index

    # hidden states and their paths
    hidden_paths: List[Path]  # the hidden state paths
    hidden_states: List[brainstate.HiddenState]  # the hidden states

    # the jax Var at the last time step
    hidden_invars: List[HiddenInVar]  # the input hidden states

    # the jax Var at the current time step
    hidden_outvars: List[HiddenOutVar]  # the output hidden states

    # the jaxpr for computing hidden state transitions
    #
    # h_1^t, h_2^t, ... = f(h_1^{t-1}, h_2^{t-1}, ..., x)
    #
    transition_jaxpr: Jaxpr

    # the other input variables for transition_jaxpr evaluation
    transition_jaxpr_constvars: List[Var]

    # whether the recurrence is diagonal across the leading ``varshape``
    # positions, i.e. ``h_i^t`` depends only on ``h_i^{t-1}`` (and the input),
    # never on ``h_j^{t-1}`` for ``i != j``. When ``True`` the cheap column-sum
    # Jacobian computed by :func:`jacrev_last_dim` already equals the true
    # per-position block diagonal; when ``False`` (a recurrent weight couples the
    # positions) the column sum over-counts the off-diagonal cross-position terms,
    # so the true block diagonal is extracted explicitly by
    # :func:`block_diagonal_last_dim`.
    #
    # This flag is determined entirely by the grouping mode:
    # ``is_diagonal_recurrence = not include_recurrent_mixing``. In the default
    # ("without recurrence") mode the cross-position weight-mixing primitives are
    # excluded from the transition (see ``_eval_eqn``), so it is position-diagonal
    # by construction even when the transition still contains within-position ops
    # (a stacked-state ``gather``, an element-wise leak). ``include_recurrent_mixing``
    # opts into the coupled transition that needs the block-diagonal path. Defaults
    # to ``True`` to preserve the cheap behavior for any positional construction.
    is_diagonal_recurrence: bool = True

    snap: Optional['SnapPattern'] = None
    """The SnAp-n neighbourhood this group's trace is widened onto, or ``None``.

    Set only when ``recurrence_scope='sparse_n'``; ``'diagonal'`` and
    ``'coupled'`` groups carry ``None`` and every pre-P3 code path is reached
    unchanged. When set, :meth:`diagonal_jacobian` returns the widened operator
    (:func:`widened_block_jacobian`) instead of the per-position block diagonal,
    and :attr:`trace_state_width` -- not :attr:`num_state` -- sizes the trailing
    axis of every trace leaf.

    Orthogonal to :attr:`is_diagonal_recurrence`, which answers a different
    question: *is this transition position-diagonal, so the cheap column-sum
    Jacobian is exact?* SnAp-n needs the coupled transition, so a widened group
    always has ``is_diagonal_recurrence=False``.
    """

    descent: Optional['GroupDescent'] = None
    """Set when this group's transition is one substep of a descended scan
    (Phase 4 structured scan descent): ``transition_jaxpr``/
    ``transition_jaxpr_constvars`` are body-scoped while ``hidden_invars``/
    ``hidden_outvars`` are the outer scan carry vars. ``None`` for ordinary
    groups."""

    @property
    def varshape(self) -> Tuple[int, ...]:
        """The shape of each state variable.

        Returns
        -------
        tuple of int
            The variable shape shared by the hidden states in the group.
        """
        return self.hidden_states[0].varshape

    @property
    def num_state(self) -> int:
        """The number of hidden states.

        Returns
        -------
        int
            The total number of hidden states across the group.
        """
        return sum([st.num_state for st in self.hidden_states])

    @property
    def trace_state_width(self) -> int:
        """The width of a trace leaf's trailing axis.

        ``num_state`` normally; ``K * num_state`` under SnAp-n, where the
        trailing axis carries a ``(neighbour, state)`` pair rather than a state
        alone. Every trace allocation, recursion and solve reads this rather
        than :attr:`num_state`, which keeps the per-primitive kernels -- all
        generic in that axis's size -- untouched.

        Returns
        -------
        int
            The trailing-axis width of this group's trace leaves.
        """
        if self.snap is None:
            return self.num_state
        return int(self.snap.num_neighbour) * self.num_state

    def check_consistent_varshape(self):
        """Check whether the shapes of the hidden states are consistent.

        Raises
        ------
        NotSupportedError
            If the shapes of the hidden states are not consistent.
        """

        varshapes = set([tuple(st.varshape) for st in self.hidden_states])
        if len(varshapes) > 1:
            raise NotSupportedError(
                f'Error: the shapes of the hidden states are not consistent. \n'
                f'{varshapes}'
            )

    def transition(
        self,
        hidden_vals: Sequence[jax.Array],
        input_vals: PyTree,
    ) -> List[jax.Array]:
        r"""Compute the hidden-state transitions.

        Evaluates the group transition jaxpr

        .. math::

            h_1^t, h_2^t, \cdots = f(h_1^{t-1}, h_2^{t-1}, \cdots, x^t).

        Parameters
        ----------
        hidden_vals : sequence of jax.Array
            The old hidden-state values.
        input_vals : PyTree
            The input values.

        Returns
        -------
        list of jax.Array
            The new hidden-state values.
        """
        return jax.core.eval_jaxpr(self.transition_jaxpr, input_vals, *hidden_vals)

    def diagonal_jacobian(
        self,
        hidden_vals: Sequence[jax.Array],
        input_vals: PyTree,
    ):
        """Compute the diagonal Jacobian matrix along the last dimension.

        Parameters
        ----------
        hidden_vals : sequence of jax.Array
            The hidden-state values.
        input_vals : PyTree
            The input values.

        Returns
        -------
        jax.Array
            The per-position block-diagonal of the recurrent Jacobian
            ``d h^t / d h^{t-1}``, with shape
            ``(*varshape, num_states, num_states)`` -- or, when :attr:`snap` is
            set, the SnAp-n widened operator of shape
            ``(*varshape, trace_state_width, trace_state_width)`` whose entry
            ``[p, (k, a), (k', b)]`` is
            ``d h^t[nbr[p,k], a] / d h^{t-1}[nbr[p,k'], b]``. Entry ``[p, a, b]`` is
            ``d h^t[p, a] / d h^{t-1}[p, b]`` -- the cross-position terms
            ``d h^t[p] / d h^{t-1}[q]`` (``p != q``) are intentionally dropped
            (the D-RTRL / e-prop diagonal approximation).

        Notes
        -----
        For diagonal recurrence (:attr:`is_diagonal_recurrence` is ``True``) the
        positions are independent, so the cheap column-sum produced by
        ``jacrev_last_dim`` already equals this block diagonal. For coupled
        recurrence the column sum would instead add in the off-diagonal
        cross-position terms -- inflating every entry and driving the eligibility
        trace to overflow -- so the true block diagonal is extracted directly via
        ``block_diagonal_last_dim``.

        When the transition contains a ``while`` equation (an opaque forward
        node), reverse-mode differentiation is unavailable (JAX has no
        transpose rule for ``while``), so the Jacobian is extracted in
        forward mode instead (``jacfwd_last_dim``, or
        ``block_diagonal_last_dim`` with ``use_forward_mode=True``) --
        same values, different derivative mode.
        """
        fn = lambda hid: self.concat_hidden(self.transition(self.split_hidden(hid), input_vals))
        concat_hid = self.concat_hidden(hidden_vals)
        needs_fwd = _transition_contains_while(self.transition_jaxpr)
        if self.snap is not None:
            return widened_block_jacobian(
                fn, concat_hid, self.snap.neighbours, self.snap.valid,
                use_forward_mode=needs_fwd,
            )
        if self.is_diagonal_recurrence:
            extract = jacfwd_last_dim if needs_fwd else jacrev_last_dim
            return extract(fn, concat_hid)
        return block_diagonal_last_dim(fn, concat_hid, use_forward_mode=needs_fwd)

    def full_jacobian(
        self,
        hidden_vals: Sequence[jax.Array],
        input_vals: PyTree,
    ):
        """Compute the complete within-group hidden-to-hidden Jacobian.

        The sibling of :meth:`diagonal_jacobian` that keeps the cross-position
        terms that method drops. Selected by the graph executor when the
        algorithm's ``trace_factorization`` is ``'random_projection'``: UORO's
        rank-1 estimator is unbiased for the recursion it rolls, so rolling the
        block diagonal would make it an unbiased estimate of an already-biased
        trace (matrix rule 11 rejects that coordinate).

        Parameters
        ----------
        hidden_vals : sequence of jax.Array
            The hidden-state values.
        input_vals : PyTree
            The input values.

        Returns
        -------
        jax.Array
            Shape ``(*varshape, num_state, *varshape, num_state)``, with entry
            ``[p, a, q, b] = d h^t[p, a] / d h^{t-1}[q, b]``.

        Notes
        -----
        :attr:`snap` is ignored: the SnAp neighbourhood is a *sparsity pattern
        for a stored trace*, and this Jacobian is consumed immediately by a
        matrix-vector product rather than stored, so there is nothing to
        sparsify. Rule 11 rejects ``recurrence_scope='sparse_n'`` under
        ``'random_projection'`` for that reason, so the combination cannot
        reach here.

        The Jacobian is only *full* if the recurrent ETP mixing was traced into
        the transition, i.e. under ``include_recurrent_mixing``. Rule 11
        guarantees that by requiring ``recurrence_scope='coupled'``.
        """
        fn = lambda hid: self.concat_hidden(self.transition(self.split_hidden(hid), input_vals))
        concat_hid = self.concat_hidden(hidden_vals)
        needs_fwd = _transition_contains_while(self.transition_jaxpr)
        return full_position_jacobian(fn, concat_hid, use_forward_mode=needs_fwd)

    def concat_hidden(self, splitted_hid_vals: Sequence[jax.Array]):
        """Concatenate split hidden-state values into a single array.

        Concatenates a sequence of split hidden-state values along the last
        axis. For non-``HiddenGroupState`` values, an extra trailing dimension
        is added before concatenation.

        Parameters
        ----------
        splitted_hid_vals : sequence of jax.Array
            A sequence of split hidden-state values, each corresponding to a
            hidden state in the group.

        Returns
        -------
        jax.Array
            A single array containing all hidden-state values concatenated
            along the last axis.
        """
        splitted_hid_vals = [
            val
            if isinstance(st, HiddenGroupState) else
            u.math.expand_dims(val, axis=-1)
            for val, st in zip(splitted_hid_vals, self.hidden_states)
        ]
        return u.math.concatenate(splitted_hid_vals, axis=-1)

    def split_hidden(self, concat_hid_vals: jax.Array):
        """Split a concatenated hidden-state array into individual arrays.

        Splits a concatenated array of hidden-state values into separate arrays,
        one per hidden state in the group. ``HiddenGroupState`` and
        non-``HiddenGroupState`` values are handled differently.

        Parameters
        ----------
        concat_hid_vals : jax.Array
            A concatenated array of hidden-state values. The last dimension is
            assumed to contain the concatenated states.

        Returns
        -------
        list of jax.Array
            A list of split hidden-state arrays. For non-``HiddenGroupState``
            values, the last dimension is squeezed.
        """
        num_states = [st.num_state for st in self.hidden_states]
        indices = np.cumsum(num_states)
        splitted_hid_vals = u.math.split(concat_hid_vals, indices, axis=-1)
        splitted_hid_vals = [
            val
            if isinstance(st, HiddenGroupState) else
            u.math.squeeze(val, axis=-1)
            for val, st in zip(splitted_hid_vals, self.hidden_states)
        ]
        return splitted_hid_vals

    def dict(self) -> Dict[str, Any]:
        """Return this group's named fields as a plain dictionary.

        Returns
        -------
        dict
            An ordered mapping from field name to value, as produced by the
            underlying :class:`typing.NamedTuple`.
        """
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


HiddenGroup.__module__ = 'braintrace'


def jacrev_last_dim(
    fn: Callable[..., jax.Array],
    hid_vals: jax.Array,
) -> jax.Array:
    """
    Compute the Jacobian of a function with respect to its last dimension.

    This function calculates the Jacobian matrix of the given function 'fn'
    with respect to the last dimension of the input 'hid_vals'. It uses
    JAX's vector-Jacobian product (vjp) and vmap for efficient computation.

    Parameters
    ----------
    fn : Callable[..., jax.Array]
        The function for which to compute the Jacobian. It should take a JAX
        array as input and return a JAX array.
    hid_vals : jax.Array
        The input values for which to compute the Jacobian. The last dimension
        is considered as the dimension of interest.

    Returns
    -------
    jax.Array
        The Jacobian matrix. Its shape is ``(*varshape, num_state, num_state)``,
        where ``varshape`` is the shape of the input excluding the last
        dimension, and ``num_state`` is the size of the last dimension.

    Raises
    ------
    AssertionError
        If the number of input and output states are not the same.
    """
    new_hid_vals, f_vjp = jax.vjp(fn, hid_vals)
    num_state = new_hid_vals.shape[-1]
    varshape = new_hid_vals.shape[:-1]
    assert num_state == hid_vals.shape[-1], 'Error: the number of input/output states should be the same.'
    g_primals = u.math.broadcast_to(u.math.eye(num_state), (*varshape, num_state, num_state))
    jac = jax.vmap(f_vjp, in_axes=-2, out_axes=-2)(g_primals)
    return jac[0]


def jacfwd_last_dim(
    fn: Callable[..., jax.Array],
    hid_vals: jax.Array,
) -> jax.Array:
    """Forward-mode counterpart of :func:`jacrev_last_dim`.

    Computes the same ``(*varshape, num_state, num_state)`` last-dimension
    Jacobian, but by pushing ``num_state`` one-hot tangents through
    :func:`jax.jvp` instead of pulling cotangents through ``vjp``. Valid on
    the same position-diagonal maps as :func:`jacrev_last_dim`, and — unlike
    it — usable when ``fn`` contains a ``while`` loop (JAX supports
    forward-mode but not reverse-mode differentiation of ``while``).

    Parameters
    ----------
    fn : Callable[[jax.Array], jax.Array]
        A shape-preserving map on ``(*varshape, num_state)`` arrays.
    hid_vals : jax.Array
        The point at which to linearize, shape ``(*varshape, num_state)``.

    Returns
    -------
    jax.Array
        The Jacobian with shape ``(*varshape, num_state, num_state)``; entry
        ``[p, a, b]`` is ``d fn(hid)[p, a] / d hid[p, b]``.

    Raises
    ------
    AssertionError
        If the number of input and output states are not the same.
    """
    num_state = hid_vals.shape[-1]
    varshape = hid_vals.shape[:-1]
    basis = u.math.broadcast_to(u.math.eye(num_state), (*varshape, num_state, num_state))

    def _push(tangent):
        out, tang = jax.jvp(fn, (hid_vals,), (tangent,))
        assert out.shape[-1] == num_state, 'Error: the number of input/output states should be the same.'
        return tang

    return jax.vmap(_push, in_axes=-2, out_axes=-1)(basis)


def full_position_jacobian(
    fn: Callable[..., jax.Array],
    hid_vals: jax.Array,
    use_forward_mode: bool = False,
) -> jax.Array:
    """Materialize ``fn``'s complete Jacobian over hidden units.

    The quantity :func:`block_diagonal_last_dim` computes and then throws most of
    away. UORO (``trace_factorization='random_projection'``) rolls its rank-1
    hidden factor through the *whole* within-group transition rather than its
    per-position block diagonal, so it needs the undiminished array.

    Parameters
    ----------
    fn : Callable[[jax.Array], jax.Array]
        A shape-preserving map on ``(*varshape, num_state)`` arrays.
    hid_vals : jax.Array
        The point at which to linearize, shape ``(*varshape, num_state)``.
    use_forward_mode : bool, optional
        Use :func:`jax.jacfwd` instead of :func:`jax.jacrev`. Required when
        ``fn`` contains a ``while`` loop (no reverse-mode rule); the values are
        identical. Default ``False``.

    Returns
    -------
    jax.Array
        Shape ``(*varshape, num_state, *varshape, num_state)``, with entry
        ``[p, a, q, b] = d fn(hid)[p, a] / d hid[q, b]``.

    Notes
    -----
    ``O((prod(varshape) * num_state) ** 2)`` in memory -- the same array
    :func:`block_diagonal_last_dim` already builds for every ``'coupled'``
    step, so this is not a new memory regime. It *is* a transient one: the
    random-projection engine must roll it inside the fused stepper
    (``_make_etrace_stepper``) so the executor never stacks it over ``T``.

    A matrix-free ``jax.jvp`` would avoid materializing it at all, since UORO
    only ever needs ``D @ s_tilde``; that is deferred as F-32.
    """
    jac_fn = jax.jacfwd if use_forward_mode else jax.jacrev
    return jac_fn(fn)(hid_vals)


def block_diagonal_last_dim(
    fn: Callable[..., jax.Array],
    hid_vals: jax.Array,
    use_forward_mode: bool = False,
) -> jax.Array:
    """Compute the per-position block diagonal of ``fn``'s Jacobian.

    Like :func:`jacrev_last_dim`, but valid when ``fn`` *couples* the leading
    ``varshape`` positions (e.g. a recurrent weight matrix). It materializes the
    full Jacobian ``(*varshape, num_state, *varshape, num_state)`` and extracts,
    for every position ``p``, the ``num_state x num_state`` block
    ``d fn(hid)[p] / d hid[p]`` -- dropping the cross-position terms. This is the
    quantity :func:`jacrev_last_dim` only happens to return when the recurrence is
    already diagonal.

    Parameters
    ----------
    fn : Callable[[jax.Array], jax.Array]
        A shape-preserving map on ``(*varshape, num_state)`` arrays.
    hid_vals : jax.Array
        The point at which to linearize, shape ``(*varshape, num_state)``.
    use_forward_mode : bool, optional
        Materialize the full Jacobian with :func:`jax.jacfwd` instead of
        :func:`jax.jacrev`. Required when ``fn`` contains a ``while`` loop
        (no reverse-mode rule); the extracted blocks are identical.
        Default ``False``.

    Returns
    -------
    jax.Array
        The block-diagonal Jacobian with shape ``(*varshape, num_state, num_state)``.

    Notes
    -----
    The full Jacobian is ``O((prod(varshape) * num_state) ** 2)`` in memory --
    the same order as the recurrent-weight eligibility trace it feeds, so it is
    affordable for the dense recurrent cells that need it. If a far larger
    coupled group ever appears, a per-position ``vmap(jacrev)`` (recomputing the
    transition once per position) trades this memory for compute.
    """
    num_state = hid_vals.shape[-1]
    varshape = hid_vals.shape[:-1]
    num_pos = int(np.prod(varshape)) if varshape else 1
    full_jac = full_position_jacobian(fn, hid_vals, use_forward_mode)
    full_jac = u.math.reshape(full_jac, (num_pos, num_state, num_pos, num_state))
    # take the per-position block: block[p] = full_jac[p, :, p, :]
    block = u.math.diagonal(full_jac, axis1=0, axis2=2)  # (num_state, num_state, num_pos)
    block = u.math.moveaxis(block, -1, 0)  # (num_pos, num_state, num_state)
    return u.math.reshape(block, (*varshape, num_state, num_state))


def widened_block_jacobian(
    fn: Callable[..., jax.Array],
    hid_vals: jax.Array,
    neighbours: np.ndarray,
    valid: np.ndarray,
    use_forward_mode: bool = False,
) -> jax.Array:
    r"""Gather the SnAp-n widened transition operator out of the full Jacobian.

    Where :func:`block_diagonal_last_dim` keeps only ``J[p, :, p, :]``, this
    keeps the whole ``K x K`` block over position ``p``'s neighbourhood:

    .. math::

        Dg[p, (k, a), (k', b)] = J[\mathrm{nbr}[p,k],\, a,\,
                                   \mathrm{nbr}[p,k'],\, b]
                                 \cdot v[p,k] \cdot v[p,k'],

    which is exactly the recursion SnAp-n truncates the influence matrix onto.
    The result's trailing pair of axes has width ``M = K * num_state``, so every
    downstream kernel -- all of which are generic in the size of that axis --
    runs unchanged.

    Parameters
    ----------
    fn : Callable[[jax.Array], jax.Array]
        A shape-preserving map on ``(*varshape, num_state)`` arrays.
    hid_vals : jax.Array
        The point at which to linearize, shape ``(*varshape, num_state)``.
    neighbours : numpy.ndarray
        ``(num_position, K)`` flat neighbour indices, ``neighbours[p, 0] == p``.
    valid : numpy.ndarray
        ``(num_position, K)`` boolean mask; a ``False`` slot gets a zero row and
        a zero column, so its trace stays zero for all time and contributes
        nothing to the gradient.
    use_forward_mode : bool, optional
        Materialize the full Jacobian with :func:`jax.jacfwd` instead of
        :func:`jax.jacrev`, as :func:`block_diagonal_last_dim` does for a
        transition containing a ``while``. Default ``False``.

    Returns
    -------
    jax.Array
        The widened operator, shape ``(*varshape, K * num_state, K * num_state)``.

    Notes
    -----
    Peak memory is the full ``(P*S) x (P*S)`` Jacobian -- the same array
    ``block_diagonal_last_dim`` already materializes for the coupled path -- but
    the *stored* result is ``O(P * K^2 * S^2)``, which at saturation is
    ``P^3 S^2``. Affordable for the small dense groups this targets; a
    matrix-free per-position ``vmap(jacrev)`` would trade it for compute.

    At ``K == 1`` with an all-valid mask this returns exactly
    :func:`block_diagonal_last_dim`'s output: the scale's identity element is
    exact, not approximately exact.
    """
    num_state = hid_vals.shape[-1]
    varshape = hid_vals.shape[:-1]
    num_pos = int(np.prod(varshape)) if varshape else 1
    num_nbr = int(neighbours.shape[1])
    jac_fn = jax.jacfwd if use_forward_mode else jax.jacrev
    full_jac = jac_fn(fn)(hid_vals)  # (*varshape, num_state, *varshape, num_state)
    full_jac = u.math.reshape(full_jac, (num_pos, num_state, num_pos, num_state))

    neighbours = np.asarray(neighbours)
    rows = np.repeat(neighbours[:, :, None], num_nbr, axis=2)  # (P, K, K)
    cols = np.repeat(neighbours[:, None, :], num_nbr, axis=1)  # (P, K, K)
    # Advanced indices separated by a slice: the broadcast (P, K, K) leads, the
    # two sliced state axes follow -> sub[p, k, k', a, b].
    sub = full_jac[rows, :, cols, :]
    mask = (valid[:, :, None] & valid[:, None, :]).astype(hid_vals.dtype)
    sub = sub * mask[:, :, :, None, None]
    sub = u.math.transpose(sub, (0, 1, 3, 2, 4))  # (P, K, S, K', S)
    width = num_nbr * num_state
    return u.math.reshape(sub, (*varshape, width, width))


def widen_instant_term(df: Any, num_neighbour: int) -> Any:
    r"""Zero-pad an instantaneous term onto the SnAp-n widened state axis.

    The trace slot ``(p, (k, a))`` holds
    :math:`\partial h[\mathrm{nbr}[p,k], a] / \partial \theta_p`, and the
    instantaneous term of a relation anchored at position ``p`` lands on ``p``
    itself. Since ``nbr[p, 0] == p`` by construction, that is slot ``k = 0``:

    .. math::

        \widetilde{D_f}[\dots, p, (k, a)] =
            \begin{cases} D_f[\dots, p, a] & k = 0 \\ 0 & \text{otherwise.}\end{cases}

    Padded (invalid) slots therefore start at zero and, because
    :func:`widened_block_jacobian` gives them a zero row, stay zero for all
    time — which is what makes the repeated ``p`` index in ``neighbours``
    harmless rather than a double count.

    Parameters
    ----------
    df : ArrayLike
        The instantaneous term, shape ``(..., num_state)``. Only the trailing
        axis is touched, so single-step ``(*varshape, S)`` and stacked
        ``(T, *varshape, S)`` layouts are both accepted.
    num_neighbour : int
        The neighbourhood width ``K``.

    Returns
    -------
    ArrayLike
        Shape ``(..., K * num_state)``; the input unchanged when ``K == 1``.
    """
    if num_neighbour == 1:
        return df
    tail = u.math.repeat(u.math.zeros_like(df), num_neighbour - 1, axis=-1)
    return u.math.concatenate([df, tail], axis=-1)


def gather_learning_signal(signal: Any, snap: 'SnapPattern') -> Any:
    r"""Gather :math:`\partial L / \partial h` onto the SnAp-n neighbour index.

    The solve contracts the trace's widened axis with the learning signal, so
    the signal has to be expressed in the *same* coordinates the trace is:

    .. math::

        \widetilde{s}[\dots, p, (k, a)] =
            \frac{\partial L}{\partial h}[\dots, \mathrm{nbr}[p,k], a]
            \cdot v[p, k] .

    The mask is applied for the same reason it is applied to the transition
    operator: a padded slot's neighbour index is ``p`` itself (an in-range dummy
    that keeps the gather safe), and zeroing it makes the slot's contribution
    identically zero rather than merely zero-by-consequence.

    Parameters
    ----------
    signal : ArrayLike
        Shape ``(..., *varshape, num_state)``; leading axes are carried through.
    snap : SnapPattern
        The group's neighbourhood.

    Returns
    -------
    ArrayLike
        Shape ``(..., *varshape, K * num_state)``.
    """
    varshape = tuple(snap.varshape)
    num_position = snap.num_position
    num_neighbour = snap.num_neighbour
    rank = len(varshape)
    lead = tuple(u.math.shape(signal))[:u.math.ndim(signal) - rank - 1]
    num_state = tuple(u.math.shape(signal))[-1]

    flat = u.math.reshape(signal, (*lead, num_position, num_state))
    # One advanced index on axis -2, so the gathered (P, K) block stays in place
    # and the result is (*lead, P, K, num_state).
    gathered = flat[..., np.asarray(snap.neighbours), :]
    mask = jnp.asarray(snap.valid, dtype=jnp.result_type(u.get_mantissa(flat)))
    gathered = gathered * mask[:, :, None]
    return u.math.reshape(
        gathered, (*lead, *varshape, num_neighbour * num_state))


def _param_subjaxprs(eqn: JaxprEqn) -> List[Jaxpr]:
    """Every jaxpr-valued param of an equation (ClosedJaxpr unwrapped;
    tuple/list params such as cond ``branches`` flattened)."""
    found: List[Jaxpr] = []
    for val in eqn.params.values():
        for item in (val if isinstance(val, (tuple, list)) else (val,)):
            sub = getattr(item, 'jaxpr', item)
            if isinstance(sub, Jaxpr):
                found.append(sub)
    return found


def _transition_contains_while(jaxpr: Jaxpr) -> bool:
    """Whether *jaxpr* (descending sub-jaxprs) contains a ``while`` equation.

    Used by :meth:`HiddenGroup.diagonal_jacobian` to switch Jacobian
    extraction to forward mode: ``while`` has no reverse-mode rule.
    """
    for eqn in jaxpr.eqns:
        if is_while_primitive(eqn):
            return True
        for sub in _param_subjaxprs(eqn):
            if _transition_contains_while(sub):
                return True
    return False


def _map_positions_to_subjaxpr_seeds(
    eqn: JaxprEqn,
    positions: Sequence[int],
) -> List[Tuple[Jaxpr, List[Var], Dict[Var, Var]]]:
    """Map hidden-derived invar *positions* of a control-flow equation to the
    corresponding body-jaxpr variables.

    Returns ``(sub_jaxpr, seed_vars, carry_feedback)`` triples, one per body
    to inspect. ``carry_feedback`` maps a body outvar at carry position ``c``
    to the body invar receiving it on the next iteration, so reachability can
    be closed over loop-carried values.

    Layouts (JAX): ``while`` invars are ``[*cond_consts, *body_consts,
    *carry]`` with ``body.invars = [*body_consts, *carry]`` (cond consts only
    steer the trip count and are skipped); ``scan`` invars are ``[*consts,
    *carry, *xs]`` positionally identical to ``body.invars``; ``cond`` invars
    are ``[pred, *operands]`` with each branch taking the operands.
    """
    results: List[Tuple[Jaxpr, List[Var], Dict[Var, Var]]] = []
    if is_while_primitive(eqn):
        cn = eqn.params['cond_nconsts']
        bn = eqn.params['body_nconsts']
        body = eqn.params['body_jaxpr'].jaxpr
        n_carry = len(eqn.invars) - cn - bn
        seeds = []
        for j in positions:
            if j < cn:
                continue  # cond consts influence the trip count, not carried values
            seeds.append(body.invars[j - cn])
        feedback: Dict[Var, Var] = {}
        for c in range(n_carry):
            ov = body.outvars[c]
            if isinstance(ov, Var):
                feedback[ov] = body.invars[bn + c]
        if seeds:
            results.append((body, seeds, feedback))
    elif is_scan_primitive(eqn):
        body = eqn.params['jaxpr'].jaxpr
        num_consts, num_carry = scan_num_consts_carry(eqn)
        seeds = [body.invars[j] for j in positions]
        feedback = {}
        for c in range(num_carry):
            ov = body.outvars[c]
            if isinstance(ov, Var):
                feedback[ov] = body.invars[num_consts + c]
        if seeds:
            results.append((body, seeds, feedback))
    elif is_cond_primitive(eqn):
        for branch in eqn.params['branches']:
            b = getattr(branch, 'jaxpr', branch)
            seeds = [b.invars[j - 1] for j in positions if j >= 1]
            if seeds:
                results.append((b, seeds, {}))
    return results


def _jaxpr_mixes_from(
    jaxpr: Jaxpr,
    seeds: Sequence[Var],
    carry_feedback: Dict[Var, Var],
) -> bool:
    """Whether a recurrent weight-mixing primitive in *jaxpr* consumes a
    value forward-reachable from *seeds*.

    Runs a forward reachability sweep from the seed variables, returning
    ``True`` as soon as a ``_RECURRENT_WEIGHT_MIXING_PRIMITIVES`` equation
    consumes a reachable value; nested control flow is descended with the
    same position mapping. Call-like equations (``jit``/``pjit``, ``remat``,
    ``custom_jvp_call``/``custom_vjp_call``) are descended positionally, so a
    ``jax.jit``-wrapped helper inside a loop body cannot hide mixing; any
    other equation carrying a body jaxpr whose arity does not match is
    conservatively reported as mixing (degrading to the zero-recurrence
    fallback plus a warning, never to a silently wrong Jacobian). The sweep
    is repeated until the carry feedback (loop outvar fed back as
    next-iteration invar) adds no new seeds, so mixing that only appears
    after the first loop iteration is still found.
    """
    seed_set = set(seeds)
    while True:
        reachable = set(seed_set)
        for eqn in jaxpr.eqns:
            hit = [
                j for j, v in enumerate(eqn.invars)
                if isinstance(v, Var) and v in reachable
            ]
            if not hit:
                continue
            if eqn.primitive.name in _RECURRENT_WEIGHT_MIXING_PRIMITIVES:
                return True
            if is_scan_primitive(eqn) or is_while_primitive(eqn) or is_cond_primitive(eqn):
                for sub, sub_seeds, sub_fb in _map_positions_to_subjaxpr_seeds(eqn, hit):
                    if _jaxpr_mixes_from(sub, sub_seeds, sub_fb):
                        return True
            else:
                for sub in _param_subjaxprs(eqn):
                    if len(sub.invars) != len(eqn.invars):
                        return True  # unknown invar mapping: assume mixing
                    sub_seeds = [sub.invars[j] for j in hit]
                    if _jaxpr_mixes_from(sub, sub_seeds, {}):
                        return True
            reachable.update(v for v in eqn.outvars if isinstance(v, Var))
        new_seeds = set(seed_set)
        for outvar, invar in carry_feedback.items():
            if outvar in reachable:
                new_seeds.add(invar)
        if new_seeds == seed_set:
            return False
        seed_set = new_seeds


class HiddenToHiddenGroupTracer(NamedTuple):
    """
    The data structure for the tracing of the hidden-to-hidden states.

    The variable collections are insertion-ordered ``dict`` objects used as
    ordered sets (values are always ``None``), so every downstream ordering
    (group members, transition constvars) is deterministic across processes;
    plain ``set`` iteration follows memory-address hashes for jaxpr ``Var``
    objects.

    Attributes
    ----------
    hidden_invar : Var
        The input variable representing the hidden state.
    connected_hidden_outvars : Dict[Var, None]
        Ordered set of output variables representing the connected hidden states.
    other_invars : Dict[Var, None]
        Ordered set of other input variables involved in the tracing.
    invar_needed_in_oth_eqns : Dict[Var, None]
        Ordered set of variables needed in other equations for trace analysis.
    trace : List[JaxprEqn]
        A list of JAX equations representing the trace of operations.
    """
    hidden_invar: Var
    connected_hidden_outvars: Dict[Var, None]
    other_invars: Dict[Var, None]
    invar_needed_in_oth_eqns: Dict[Var, None]
    trace: List[JaxprEqn]

    def dict(self) -> Dict[str, Any]:
        """Return this tracer's named fields as a plain dictionary.

        Returns
        -------
        dict
            An ordered mapping from field name to value, as produced by the
            underlying :class:`typing.NamedTuple`.
        """
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


class Hidden2GroupTransition(NamedTuple):
    """
    Represents a hidden state transition in a computational graph.

    This class captures the transition of hidden states from one time step to the next
    within a neural network model. It includes information about the input hidden state,
    the connected output hidden states, and the JAX program representation (jaxpr) that
    defines the transition.

    Attributes
    ----------
    hidden_invar : Var
        The input variable representing the hidden state at the previous time step.
    hidden_path : Path
        The path to the hidden state in the model hierarchy.
    connected_hidden_outvars : List[Var]
        A list of output variables representing the connected hidden states at
        the current time step.
    connected_hidden_paths : List[Path]
        A list of paths to the connected hidden states in the model hierarchy.
    transition_jaxpr : Jaxpr
        The JAX program representation for computing the hidden state transitions.
    other_invars : List[Var]
        A list of other input variables required for evaluating the
        ``transition_jaxpr``.
    """

    # the hidden state h_i^{t-1}
    hidden_invar: Var
    hidden_path: Path

    # the connected hidden states h_1^t, h_2^t, ...
    connected_hidden_outvars: List[Var]
    connected_hidden_paths: List[Path]

    # the jaxpr for computing hidden state transitions
    #
    # h_1^t, h_2^t, ... = f(h_i^{t-1}, x)
    #
    transition_jaxpr: Jaxpr

    # the other input variables for jaxpr evaluation
    other_invars: List[Var]

    def state_transition(
        self,
        old_hidden_val: jax.Array,
        other_input_vals: PyTree,
        return_index: Optional[int] = None
    ) -> List[jax.Array] | jax.Array:
        """
        Computing the hidden state transitions :math:`h^t = f(h_i^t, x)`.

        Parameters
        ----------
        old_hidden_val : jax.Array
            The old hidden state value.
        other_input_vals : PyTree
            The input values.
        return_index : int, optional
            Index of the hidden state to return.

        Returns
        -------
        list of jax.Array or jax.Array
            The new hidden state values.
        """
        new_hidden_vals = jax.core.eval_jaxpr(self.transition_jaxpr, other_input_vals, old_hidden_val)
        if return_index is not None:
            return new_hidden_vals[return_index]
        return new_hidden_vals

    def dict(self) -> Dict[str, Any]:
        """Return this transition's named fields as a plain dictionary.

        Returns
        -------
        dict
            An ordered mapping from field name to value, as produced by the
            underlying :class:`typing.NamedTuple`.
        """
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


def _same_recurrence_layer(path1: Path, path2: Path) -> bool:
    """
    Check if two hidden state paths belong to the same recurrence layer.

    Paths that diverge at a numeric index (e.g., ('layers', 0, ...) vs
    ('layers', 1, ...)) indicate different sequential layers and should
    be in separate groups. Paths that diverge at string keys (e.g.,
    ('neu', 'V') vs ('neu', 'a')) are within the same layer.

    This is a *path heuristic*, not a graph-structural test: sibling layers
    keyed by strings (``self.cell0`` / ``self.cell1``, dicts of modules) are
    NOT separated by it. Their separation instead relies on the recurrent-
    mixing boundary skips in ``_eval_eqn`` — any matmul/conv between layers
    cuts the trace — which holds for every realistic layered model. Two
    purely-elementwise-coupled sibling string-keyed layers would merge into
    one group; a graph-structural replacement is future work.
    """
    min_len = min(len(path1), len(path2))
    for i in range(min_len):
        if path1[i] != path2[i]:
            return not (isinstance(path1[i], int) or isinstance(path2[i], int))
    return True


def _simplify_hid2hid_tracer(
    tracer: HiddenToHiddenGroupTracer,
    hidden_invar_to_path: Dict[HiddenInVar, Path],
    hidden_outvar_to_path: Dict[HiddenOutVar, Path],
    path_to_state: Dict[Path, brainstate.HiddenState],
    debug_info=None,
) -> Optional[Hidden2GroupTransition]:
    """
    Simplifying the hidden-to-hidden state tracer.

    Parameters
    ----------
    tracer : HiddenToHiddenGroupTracer
        The hidden-to-hidden state tracer.
    hidden_invar_to_path : dict
        The mapping from the hidden input variable to the hidden state path.
    hidden_outvar_to_path : dict
        The mapping from the hidden output variable to the hidden state path.
    path_to_state : dict
        The mapping from the hidden state path to the state.
    debug_info : optional
        The debug info threaded from the source model jaxpr onto the simplified
        transition jaxpr (avoids the missing-DebugInfo deprecation).

    Returns
    -------
    Hidden2GroupTransition or None
        The hidden-to-hidden state transition.
    """
    #
    # [pre-step]
    #
    # Filter out hidden outvars from different recurrence layers.
    # In multi-layer networks, a hidden state from one layer may be
    # connected to hidden outvars of other layers through the computation
    # graph. These cross-layer connections should not be in the same group.
    # Two filters are applied:
    #   1. Shape compatibility: outvars must match the invar's shape.
    #   2. Layer membership: outvars whose paths diverge from the invar's
    #      path at a numeric index (e.g., layers.0 vs layers.1) are in
    #      different sequential layers and are excluded.
    invar_path = hidden_invar_to_path[tracer.hidden_invar]
    invar_state = path_to_state[invar_path]
    # Ordered dict-as-set: preserves the tracer's encounter order so the
    # resulting transition outvars/constvars are deterministic.
    compatible_outvars = dict.fromkeys(
        hv for hv in tracer.connected_hidden_outvars
        if (path_to_state[hidden_outvar_to_path[hv]].varshape == invar_state.varshape
            and _same_recurrence_layer(invar_path, hidden_outvar_to_path[hv]))
    )
    if not compatible_outvars:
        return None

    #
    # [first step]
    #
    # Remove the unnecessary equations in the trace.
    # The unnecessary equations are the equations
    # that do not contain the hidden states.
    tracer.invar_needed_in_oth_eqns.clear()
    new_trace = []
    whole_trace_needed_vars = dict.fromkeys(compatible_outvars)
    visited_needed_vars: Dict[Var, None] = {}  # needed_vars has been satisfied
    for eqn in reversed(tracer.trace):
        need_outvars = []
        for outvar in eqn.outvars:
            if outvar in whole_trace_needed_vars:
                need_outvars.append(outvar)
        if len(need_outvars):
            visited_needed_vars.update(dict.fromkeys(need_outvars))
            new_trace.append(eqn)
            whole_trace_needed_vars.update(
                dict.fromkeys(invar for invar in eqn.invars if isinstance(invar, Var))
            )

    # [second step]
    #
    # Shape filtering was already done in the pre-step.
    hidden_outvars = tuple(compatible_outvars)

    # [third step]
    #
    # Simplify the trace
    visited_needed_vars[tracer.hidden_invar] = None
    constvars = [v for v in whole_trace_needed_vars if v not in visited_needed_vars]
    jaxpr_opt = Jaxpr(
        # the const vars are not the hidden states, they are
        # intermediate data that are not used in the hidden states
        constvars=constvars,
        # the invars are always the weight output
        invars=[tracer.hidden_invar],
        # the outvars are always the connected hidden states of this weight
        outvars=list(hidden_outvars),
        # the new equations which are simplified
        eqns=list(reversed(new_trace)),
        debug_info=debug_info,
    )

    # [final step]
    #
    # Change the "HiddenWeightOpTracer" to "Hidden2GroupTransition"
    return Hidden2GroupTransition(
        hidden_invar=tracer.hidden_invar,
        hidden_path=hidden_invar_to_path[tracer.hidden_invar],
        connected_hidden_outvars=list(hidden_outvars),
        connected_hidden_paths=[hidden_outvar_to_path[var] for var in hidden_outvars],
        transition_jaxpr=jaxpr_opt,
        other_invars=constvars,
    )


class JaxprEvalForHiddenGroup(JaxprEvaluation):
    """
    Evaluating the jaxpr for extracting the hidden state ``hidden-to-hidden`` relationships.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr for the model.
    hidden_outvar_to_invar : dict
        The mapping from the hidden output variable to the hidden input variable.
    weight_invars : set
        The weight input variables.
    invar_to_hidden_path : dict
        The mapping from the weight input variable to the hidden state path.
    outvar_to_hidden_path : dict
        The mapping from the hidden output variable to the hidden state path.
    path_to_state : dict
        The mapping from the hidden state path to the state.
    include_recurrent_mixing : bool
        Whether recurrent ETP mixing primitives are traced into the
        hidden-to-hidden transition jaxpr.
    sparse_n : int or None, optional
        SnAp order for ``recurrence_scope='sparse_n'``; ``None`` (default)
        leaves every group's ``snap`` pattern unset.
    control_flow : ControlFlowPolicy
        The :class:`~braintrace.ControlFlowPolicy` governing opaque
        control-flow handling (see ``base.check_unsupported_op``).
    descended_scan_eqn_ids : frozenset of int
        ``id()`` values of scan equations rewritten by structured scan descent;
        those equations are skipped by this walker.
    descended_hidden_paths : set of Path
        Hidden-state paths covered by descended scan bodies; excluded from the
        zero-recurrence fallback grouping.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        jaxpr: Jaxpr,
        hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
        weight_invars: Set[Var],
        invar_to_hidden_path: Dict[HiddenInVar, Path],
        outvar_to_hidden_path: Dict[HiddenOutVar, Path],
        path_to_state: Dict[Path, brainstate.HiddenState],
        include_recurrent_mixing: bool = False,
        sparse_n: Optional[int] = None,
        snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
        control_flow: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
        descended_scan_eqn_ids: FrozenSet[int] = frozenset(),
        descended_hidden_paths: FrozenSet[Path] = frozenset(),
    ):
        # the jaxpr of the original model, assuming that the model is well-defined,
        # see the doc for the model which can be online learning compiled.
        self.jaxpr = jaxpr

        # Structured scan descent (Phase 4): scan equations already analyzed
        # by ``scan_descent.apply_scan_descent`` (keyed by ``id(eqn)``) are
        # skipped entirely, and the hidden paths their body groups cover are
        # excluded from the zero-recurrence fallback (their groups are merged
        # into the outer graph by ``compile_etrace_graph``).
        self.descended_scan_eqn_ids = descended_scan_eqn_ids
        self.descended_hidden_paths = descended_hidden_paths

        # whether the recurrent ETP mixing primitives (``etp_mv``/``etp_mm``/
        # ``etp_conv``) are *traced into* the hidden-to-hidden transition jaxpr
        # (``True``) or treated as boundaries and excluded (``False``, default).
        # See :func:`find_hidden_groups_from_jaxpr` for the rationale.
        self.include_recurrent_mixing = include_recurrent_mixing

        # SnAp order (``recurrence_scope='sparse_n'``): when set, every group
        # built below carries the derived n-step neighbourhood its trace is
        # widened onto. ``None`` (every other scope) leaves ``HiddenGroup.snap``
        # unset and no pre-P3 code path changes.
        self.sparse_n = sparse_n

        # Ceiling on the widened block Jacobian, in elements. Only consulted
        # when ``sparse_n`` is set; see ``DEFAULT_MAX_JACOBIAN_ELEMENTS``.
        self.snap_max_jacobian_elements = snap_max_jacobian_elements

        # the hidden state groups
        self.hidden_outvar_to_invar = hidden_outvar_to_invar
        self.hidden_invar_to_outvar = {invar: outvar for outvar, invar in hidden_outvar_to_invar.items()}
        hidden_invars = set(hidden_outvar_to_invar.values())
        hidden_outvars = set(hidden_outvar_to_invar.keys())
        self.path_to_state = path_to_state

        # the data structures for the tracing hidden-hidden relationships
        self.active_tracers: Dict[Var, HiddenToHiddenGroupTracer] = dict()

        super().__init__(
            weight_invars=weight_invars,
            hidden_invars=hidden_invars,
            hidden_outvars=hidden_outvars,
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path,
            control_flow=control_flow,
        )

    def compile(self) -> Tuple[
        Sequence[HiddenGroup],
        Dict[Path, HiddenGroup],
    ]:
        """
        Compiling the jaxpr for the etrace relationships.
        """

        # the data structures for the tracing hidden-hidden relationships
        self.active_tracers = dict()

        # evaluating the jaxpr
        self._eval_jaxpr(self.jaxpr)

        # post checking
        hid_groups, hid_path_to_group = self._post_check()

        # reset the temporal data structures
        self.active_tracers = dict()
        return hid_groups, hid_path_to_group

    def _eval_scan(self, eqn: JaxprEqn) -> None:
        # A descended scan is fully handled by scan-descent analysis: its
        # hidden groups are registered from the body (re-scoped to the outer
        # carry vars) and merged by ``compile_etrace_graph``, so the equation
        # must be neither rejected nor traced as an opaque transition here.
        if id(eqn) in self.descended_scan_eqn_ids:
            return
        super()._eval_scan(eqn)

    def _eval_eqn(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the normal jaxpr equation.
        """
        if eqn.primitive.name == 'stop_gradient':
            return

        # Treat a recurrent ETP *mixing* primitive (e.g. ``etp_mv``/``etp_mm``/
        # ``etp_conv`` -- ``is_etp_primitive`` but not ``is_etp_enable_gradient_primitive``)
        # as a boundary: its output is supplied separately (carried by the weight
        # eligibility trace), so it must not be traced into the hidden-to-hidden
        # transition. Skipping it here keeps the transition element-wise, which
        # restores the bounded D-RTRL recurrence (the 0.1.2 behaviour). Identity-
        # like, gradient-enabled ETP ops (e.g. ``etp_elemwise``) are *not* skipped.
        # ``include_recurrent_mixing=True`` opts back into tracing through them.
        if (
            not self.include_recurrent_mixing
            and is_etp_primitive(eqn.primitive)
            and not is_etp_enable_gradient_primitive(eqn.primitive)
        ):
            return

        # A *non-ETP* recurrent-weight mixing primitive that reads the hidden
        # state (a plain ``dot_general``/``conv_general_dilated`` recurrent weight,
        # as in a reservoir) couples the leading ``varshape`` positions just like
        # the ETP matmul does. In the default ("without recurrence") mode it is
        # likewise treated as a boundary so the hidden-to-hidden transition stays
        # position-diagonal -- which is what makes
        # ``is_diagonal_recurrence = not include_recurrent_mixing`` correct rather
        # than a footgun (a cross-position-coupled transition driven by the cheap
        # column-sum Jacobian is exactly what overflows the eligibility trace).
        # Only ops that *read the hidden state* are skipped: a feed-forward input
        # projection (``x @ W_in``) does not couple the recurrence and is kept.
        # The set is deliberately narrow (matmul/conv only): within-position
        # reductions/gathers over the ``num_state`` axis -- e.g. the ``gather`` that
        # splits a stacked ``HiddenGroupState`` (ALIF ``V``/``a``) -- are NOT
        # cross-position coupling and must remain in the transition (``jacrev_last_dim``
        # already handles the resulting per-position block correctly).
        if (
            not self.include_recurrent_mixing
            and eqn.primitive.name in _RECURRENT_WEIGHT_MIXING_PRIMITIVES
            and self._eqn_consumes_hidden(eqn)
        ):
            return

        # An opaque control-flow equation (while, or a scan/cond the
        # canonicalizer left opaque) whose *body* applies a recurrent
        # weight-mixing primitive to the carried hidden state couples the
        # positions exactly like a top-level recurrent matmul -- so in the
        # default mode it is likewise a boundary. Bodies that only mix
        # loop constants (an input projection ``x @ W_in``) are kept.
        if (
            not self.include_recurrent_mixing
            and (is_scan_primitive(eqn) or is_while_primitive(eqn) or is_cond_primitive(eqn))
            and self._control_flow_mixes_hidden(eqn)
        ):
            emit(
                kind=DiagnosticKind.CONTROL_FLOW_RECURRENT_MIXING,
                level=DiagnosticLevel.WARNING,
                message=(
                    f'An opaque {eqn.primitive.name} whose body applies a '
                    f'recurrent weight-mixing primitive to the carried hidden '
                    f'state was excluded from the hidden-to-hidden transition '
                    f'(default "without recurrence" mode). Its temporal credit '
                    f'falls back to the zero-recurrence (e-prop) approximation; '
                    f'pass include_recurrent_mixing=True to trace through it.'
                ),
                context={'op_name': eqn.primitive.name},
            )
            return

        # check whether the invars have one of the hidden states.
        # If it is true, add a new tracer.
        other_invars = []
        hidden_invars = []
        for invar in eqn.invars:
            if isinstance(invar, Literal):
                continue
            elif invar in self.hidden_invars:
                hidden_invars.append(invar)
            else:
                other_invars.append(invar)
        if len(hidden_invars) > 0:
            # A hidden invar may be used in multiple places.
            # All places share a common tracer.
            if len(hidden_invars) != 1:
                paths = [str(self.invar_to_hidden_path[var]) for var in hidden_invars]
                hidden_paths = "\n".join(paths)
                raise ValueError(
                    f'Currently, we only support one hidden state in a single equation. \n'
                    f'{eqn}\n'
                    f'The hidden states consumed by this equation are: \n'
                    f'{hidden_paths}\n'
                    f'If these states form one multi-component neuron, stack '
                    f'them into a single array held by a '
                    f'brainstate.HiddenGroupState (or HiddenTreeState) so the '
                    f'equation consumes one hidden variable.'
                )
            hidden_var = hidden_invars[0]
            hidden_outvars = dict.fromkeys(outvar for outvar in eqn.outvars if outvar in self.hidden_outvars)
            needed_invars = dict.fromkeys(outvar for outvar in eqn.outvars if outvar not in self.hidden_outvars)
            if hidden_var in self.active_tracers:
                self.active_tracers[hidden_var].trace.append(eqn.replace())
                self.active_tracers[hidden_var].other_invars.update(dict.fromkeys(other_invars))
                self.active_tracers[hidden_var].invar_needed_in_oth_eqns.update(needed_invars)
                self.active_tracers[hidden_var].connected_hidden_outvars.update(hidden_outvars)
            else:
                tracer = HiddenToHiddenGroupTracer(
                    hidden_invar=hidden_var,
                    connected_hidden_outvars=hidden_outvars,
                    other_invars=dict.fromkeys(other_invars),
                    invar_needed_in_oth_eqns=needed_invars,
                    trace=[eqn.replace()]
                )
                self.active_tracers[hidden_var] = tracer

        # check whether this equation is used in other tracers
        for tracer in tuple(self.active_tracers.values()):
            matched = find_matched_vars(eqn.invars, tracer.invar_needed_in_oth_eqns)

            # if matched, add the eqn to the trace
            # if not matched, skip
            if len(matched):
                self._add_eqn_in_a_tracer(eqn, tracer)

    def _add_eqn_in_a_tracer(
        self,
        eqn: JaxprEqn,
        tracer: HiddenToHiddenGroupTracer
    ) -> None:

        tracer.trace.append(eqn.replace())
        tracer.invar_needed_in_oth_eqns.update(dict.fromkeys(eqn.outvars))

        # check whether the hidden states are needed in the other equations
        for outvar in eqn.outvars:
            if outvar in self.hidden_outvars:
                tracer.connected_hidden_outvars[outvar] = None

    def _eqn_consumes_hidden(self, eqn: JaxprEqn) -> bool:
        """Whether ``eqn`` reads a hidden-derived value.

        Returns ``True`` when any input variable is a previous hidden state
        (:attr:`hidden_invars`) or a value transitively derived from one (tracked
        by an active tracer's ``invar_needed_in_oth_eqns``). Used to decide
        whether a recurrent-mixing primitive couples the hidden state and must be
        excluded from the transition in the default grouping mode.
        """
        for invar in eqn.invars:
            if isinstance(invar, Var) and invar in self.hidden_invars:
                return True
        for tracer in self.active_tracers.values():
            if find_matched_vars(eqn.invars, tracer.invar_needed_in_oth_eqns):
                return True
        return False

    def _hidden_derived_invar_positions(self, eqn: JaxprEqn) -> List[int]:
        """Positions of ``eqn.invars`` carrying a hidden-derived value.

        A position qualifies when the invar is a previous hidden state
        (:attr:`hidden_invars`) or transitively derived from one (tracked by
        an active tracer's ``invar_needed_in_oth_eqns``).
        """
        positions = []
        for j, invar in enumerate(eqn.invars):
            if not isinstance(invar, Var):
                continue
            if invar in self.hidden_invars:
                positions.append(j)
                continue
            for tracer in self.active_tracers.values():
                if invar in tracer.invar_needed_in_oth_eqns:
                    positions.append(j)
                    break
        return positions

    def _control_flow_mixes_hidden(self, eqn: JaxprEqn) -> bool:
        """Whether a control-flow equation's body applies a recurrent
        weight-mixing primitive to a hidden-derived input."""
        positions = self._hidden_derived_invar_positions(eqn)
        if not positions:
            return False
        return any(
            _jaxpr_mixes_from(sub, seeds, feedback)
            for sub, seeds, feedback in _map_positions_to_subjaxpr_seeds(eqn, positions)
        )

    def _attach_snap_pattern(self, group: HiddenGroup) -> HiddenGroup:
        """Derive and attach this group's SnAp-n neighbourhood, if one is asked for.

        A degenerate result (``K == 1``) is legitimate -- a group with no
        cross-position coupling has nothing to retain, and ``sparse_n`` there is
        exactly ``coupled`` -- but it is also what a silently failing analysis
        would produce, so it is reported. A conservative result is reported too,
        since it can inflate the trace by orders of magnitude.
        """
        if self.sparse_n is None or group.descent is not None:
            return group
        pattern = build_snap_pattern(
            group.transition_jaxpr, group.varshape, self.sparse_n,
            num_state=group.num_state,
            max_jacobian_elements=self.snap_max_jacobian_elements,
        )
        if pattern.conservative:
            emit(
                kind=DiagnosticKind.SNAP_PATTERN_CONSERVATIVE,
                level=DiagnosticLevel.WARNING,
                message=(
                    f'The SnAp-n position analysis could not derive a pattern for '
                    f'hidden group {group.index} ({group.hidden_paths}) and widened '
                    f'to all-to-all (K={pattern.num_neighbour} of '
                    f'{pattern.num_position} positions): {pattern.reason} The '
                    f'approximation is not degraded -- a superset neighbourhood '
                    f'only moves the rule towards within-group RTRL -- but the '
                    f'trace is larger than sparse_n={self.sparse_n} implies. '
                    f'Note that within-group RTRL equals BPTT only when the '
                    f'group\'s tail is position-preserving; see F-31 in '
                    f'docs/specs/2026-07-25-known-limitations.md.'
                ),
                hidden_paths=tuple(group.hidden_paths),
                context={'num_neighbour': pattern.num_neighbour,
                         'num_position': pattern.num_position,
                         'sparse_n': self.sparse_n},
            )
        elif pattern.is_degenerate:
            emit(
                kind=DiagnosticKind.SNAP_PATTERN_DEGENERATE,
                level=DiagnosticLevel.INFO,
                message=(
                    f'Hidden group {group.index} ({group.hidden_paths}) has no '
                    f'cross-position coupling in its transition, so SnAp-n at '
                    f'sparse_n={self.sparse_n} retains a single position and '
                    f"computes exactly what recurrence_scope='coupled' computes."
                ),
                hidden_paths=tuple(group.hidden_paths),
                context={'sparse_n': self.sparse_n},
            )
        return group._replace(snap=pattern)

    def _post_check(self) -> Tuple[
        Sequence[HiddenGroup],
        Dict[Path, HiddenGroup],
    ]:
        # [ First step ]
        #
        # check the following items:
        #
        # 1. the shape of connected hidden states should be the same
        # 2. simplify the trace
        # 3. remove the unnecessary hidden states

        hidden_to_group_transition = [
            t for t in (
                _simplify_hid2hid_tracer(
                    tracer,
                    self.invar_to_hidden_path,
                    self.outvar_to_hidden_path,
                    self.path_to_state,
                    debug_info=self.jaxpr.debug_info,
                )
                for tracer in self.active_tracers.values()
            )
            if t is not None
        ]

        # [ second step ]
        #
        # Find out the hidden group,
        # i.e., the hidden states that are connected to each other, the union of all hidden-to-group.
        #
        # The merge is deterministic and the result is canonicalized against
        # the compiled state order (``hidden_outvar_to_invar`` insertion
        # order): members within a group and the groups themselves follow the
        # order the hidden states appear in the compiled model, never the
        # address-hash order of an intermediate set.
        outvar_groups: list = [
            [self.hidden_invar_to_outvar[transition.hidden_invar]]
            + list(transition.connected_hidden_outvars)
            for transition in hidden_to_group_transition
        ]
        outvar_groups = _merge_groups_ordered(outvar_groups)
        outvar_order = {ov: i for i, ov in enumerate(self.hidden_outvar_to_invar)}
        outvar_groups = [
            sorted(group, key=outvar_order.__getitem__)
            for group in outvar_groups
        ]
        outvar_groups.sort(key=lambda group: outvar_order[group[0]])
        invar_groups = [
            [self.hidden_outvar_to_invar[outvar] for outvar in group]
            for group in outvar_groups
        ]

        # [ third step ]
        #
        # compile the state transitions in a hidden group
        #
        #   h_1^t, h_2^t, ... h_n^t = f(h_1^t-1, h_2^t-1, ...., h_n^t-1)
        #
        hidden_invar_to_transition = {
            transition.hidden_invar: transition
            for transition in hidden_to_group_transition
        }
        jaxpr_groups = []
        for hidden_invars, hidden_outvars in zip(invar_groups, outvar_groups):
            jaxpr_groups.append(
                write_jaxpr_of_hidden_group_transition(
                    hidden_invar_to_transition,
                    hidden_invars,
                    hidden_outvars,
                    debug_info=self.jaxpr.debug_info,
                )
            )

        # [ fourth step ]
        #
        # compile HiddenGroup
        #
        hidden_groups: list = []
        for hidden_invars, hidden_outvars, jaxpr in zip(invar_groups, outvar_groups, jaxpr_groups):
            # ``is_diagonal_recurrence`` is fully determined by the grouping mode:
            # in the default mode the recurrent-weight boundary skip (see
            # ``_eval_eqn``) removes every *cross-position* coupling, leaving a
            # transition that is position-diagonal across the leading ``varshape``
            # axis -- so the cheap column-sum Jacobian (:func:`jacrev_last_dim`) is
            # exact, even when the transition still contains within-position
            # operations (a stacked-state ``gather``, an element-wise leak).
            # ``include_recurrent_mixing=True`` opts into the cross-position-coupled
            # transition that needs the block-diagonal path. (A structural re-check
            # of the transition would be unreliable here: it cannot cheaply tell a
            # within-position gather/reduction -- legitimately diagonal across
            # positions -- from genuine cross-position coupling.)
            group = HiddenGroup(
                index=len(hidden_groups),
                hidden_invars=list(hidden_invars),
                hidden_outvars=list(hidden_outvars),
                hidden_paths=[
                    self.outvar_to_hidden_path[outvar]
                    for outvar in hidden_outvars
                ],
                hidden_states=[
                    self.path_to_state[self.outvar_to_hidden_path[outvar]]
                    for outvar in hidden_outvars
                ],
                transition_jaxpr=jaxpr,
                transition_jaxpr_constvars=list(jaxpr.constvars),
                is_diagonal_recurrence=not self.include_recurrent_mixing,
            )
            group = self._attach_snap_pattern(group)
            # Belt-and-braces: the per-transition shape filter in
            # ``_simplify_hid2hid_tracer`` should already guarantee this, but
            # a merged group violating it would corrupt concat/split downstream.
            group.check_consistent_varshape()
            if len(group.hidden_paths) > 1:
                emit(
                    kind=DiagnosticKind.HIDDEN_GROUP_MERGED,
                    level=DiagnosticLevel.INFO,
                    message=(
                        f'Hidden states {group.hidden_paths} are mutually '
                        f'recurrent and were merged into one hidden group.'
                    ),
                    hidden_paths=tuple(group.hidden_paths),
                )
            hidden_groups.append(group)

        # [ fourth-b step ]
        #
        # Zero-recurrence groups for hidden states whose entire recurrence was
        # excluded.
        #
        # When recurrent ETP mixing primitives are treated as boundaries
        # (``include_recurrent_mixing=False``), a hidden state whose *only*
        # dependence on its previous value flows through such a primitive (e.g. a
        # vanilla RNN ``h^t = tanh(W @ [x, h^{t-1}])``) has no surviving
        # hidden-to-hidden path, so the steps above produce no group for it.
        # Every hidden outvar must nonetheless carry a group index (the
        # hidden->weight relation compiler asserts this). Give each uncovered
        # hidden state a singleton group whose transition is independent of
        # ``h^{t-1}`` -- i.e. ``D^t = 0`` -- by routing its current value through
        # a constvar. The recurrent weight's temporal credit is then carried
        # entirely by its eligibility trace's immediate term (the e-prop / RFLO
        # approximation), and the trace stays bounded.
        covered_outvars: Set[Var] = set()
        for group in hidden_groups:
            covered_outvars.update(group.hidden_outvars)
        # Iterate the insertion-ordered outvar->invar mapping (compiled state
        # order), NOT the base-class ``hidden_outvars`` set, so the fallback
        # groups are appended in deterministic order.
        for outvar in self.hidden_outvar_to_invar:
            if outvar in covered_outvars:
                continue
            # Hidden states carried by a descended scan get their group from
            # the scan-body analysis (merged downstream); a zero-recurrence
            # fallback here would shadow it.
            if self.outvar_to_hidden_path[outvar] in self.descended_hidden_paths:
                continue
            invar = self.hidden_outvar_to_invar[outvar]
            # ``h^t = outvar`` (a constvar): no eqns, output does not depend on the
            # ``h^{t-1}`` invar, so the recurrent Jacobian is exactly zero.
            zero_jaxpr = Jaxpr(
                constvars=[outvar],
                invars=[invar],
                outvars=[outvar],
                eqns=[],
                debug_info=self.jaxpr.debug_info,
            )
            group = HiddenGroup(
                index=len(hidden_groups),
                hidden_invars=[invar],
                hidden_outvars=[outvar],
                hidden_paths=[self.outvar_to_hidden_path[outvar]],
                hidden_states=[self.path_to_state[self.outvar_to_hidden_path[outvar]]],
                transition_jaxpr=zero_jaxpr,
                transition_jaxpr_constvars=list(zero_jaxpr.constvars),
                # A zero-recurrence transition (``D^t = 0``) is trivially diagonal;
                # keep the flag mode-derived for uniformity (this fallback only
                # fires in the default mode in practice).
                is_diagonal_recurrence=not self.include_recurrent_mixing,
            )
            hidden_groups.append(group)

        # [ fifth step ]
        #
        # transform the hidden group set to the HiddenGroup
        #
        # hidden outvar to group
        #
        hidden_path_to_group: Dict[Path, HiddenGroup] = dict()
        for group in hidden_groups:
            for path in group.hidden_paths:
                if path in hidden_path_to_group:
                    raise ValueError(
                        f'Error: the hidden state {path} '
                        f'is found in multiple groups. \n'
                        f'{hidden_path_to_group[path].hidden_paths} '
                        f'\n\n'
                        f'{group.hidden_paths}'
                    )
                hidden_path_to_group[path] = group

        return hidden_groups, hidden_path_to_group


def write_jaxpr_of_hidden_group_transition(
    hidden_invar_to_transition: Dict[HiddenInVar, Hidden2GroupTransition],
    hidden_invars: List[HiddenInVar],
    hidden_outvars: List[HiddenOutVar],
    debug_info=None,
) -> Jaxpr:
    assert len(hidden_invars) >= 1

    #
    # step 1:
    #
    # filter out
    #
    # 1. all invars + constvars
    # 2. equations
    # 3. all outvars
    #
    eqns = []
    # Ordered dict-as-set bookkeeping keeps the derived ``constvars`` order
    # deterministic across processes (Var hashing is address-based).
    all_invars: Dict[Var, None] = {}
    all_outvars: Dict[Var, None] = {}
    for invar in hidden_invars:
        if invar in hidden_invar_to_transition:
            transition = hidden_invar_to_transition[invar]
            for eq in transition.transition_jaxpr.eqns:
                this_eq_exist = [outvar in all_outvars for outvar in eq.outvars]
                if not all(this_eq_exist):
                    eqns.append(eq.replace())
                    all_invars.update(
                        dict.fromkeys(invar for invar in eq.invars if not isinstance(invar, Literal))
                    )
                    all_outvars.update(dict.fromkeys(eq.outvars))
    other_invars = [
        v for v in all_invars
        if v not in all_outvars and v not in hidden_invars
    ]

    #
    # step 2:
    #
    # order the equations so that data dependencies are satisfied
    #
    new_eqns = []
    env = set(list(hidden_invars) + other_invars)
    max_iterations = len(eqns) * len(eqns) + 1  # upper bound for topological sort passes
    iteration_count = 0
    while len(eqns) > 0:
        iteration_count += 1
        if iteration_count > max_iterations:
            unresolved_invars = []
            for eqn in eqns:
                missing = [v for v in eqn.invars if not isinstance(v, Literal) and v not in env]
                unresolved_invars.append((eqn, missing))
            raise RuntimeError(
                f'Topological sort failed: could not resolve all equation dependencies. '
                f'{len(eqns)} equations remain unresolved. '
                f'This may indicate a cyclic dependency or missing input variables. '
                f'Unresolved equations: {unresolved_invars}'
            )
        eqn = eqns.pop(0)
        if all((invar in env) for invar in eqn.invars if not isinstance(invar, Literal)):
            # Execute the equation
            new_eqns.append(eqn)
            # Add outvars to env
            env.update(eqn.outvars)
        else:
            # If invars are not in env, put the equation back to the queue
            eqns.append(eqn)

    #
    # step 3:
    #
    # produce the new jaxpr
    #
    return Jaxpr(
        constvars=list(other_invars),
        invars=hidden_invars,
        outvars=hidden_outvars,
        eqns=new_eqns,
        debug_info=debug_info,
    )


def _merge_groups_ordered(groups: Sequence[Sequence[HiddenOutVar]]) -> List[List[HiddenOutVar]]:
    """Union intersecting groups, deterministically.

    Semantically identical to :func:`group_merging` (transitive union of any
    groups sharing a member) but order-preserving: the result lists groups by
    first appearance in ``groups``, and each group's members by first
    appearance, so the output never depends on ``Var`` hash (memory-address)
    order. Used by the compiler; :func:`group_merging` is kept for its direct
    importers.

    Parameters
    ----------
    groups : sequence of sequence of Var
        The groups to merge.

    Returns
    -------
    list of list of Var
        Disjoint merged groups, in deterministic order.
    """
    merged: List[Dict[HiddenOutVar, None]] = []
    for g in groups:
        new = dict.fromkeys(g)
        hits = [m for m in merged if any(v in m for v in new)]
        if hits:
            base = hits[0]
            for other in hits[1:]:
                base.update(other)
                merged.remove(other)
            base.update(new)
        else:
            merged.append(new)
    return [list(m) for m in merged]


def group_merging(groups, version: int = 1) -> List[frozenset[HiddenOutVar]]:
    """
    Merging the hidden groups using the intersection of the hidden states.

    For example, if we have the following hidden states:

        [(h_1, h_2),
         (h_2, h_3),
         (h_4, h_5)]

    The merged hidden states are:

        [(h_1, h_2, h_3),
         (h_4, h_5)]


    This function takes a list of hidden groups and merges them if they share
    any common hidden states. The merging process is controlled by the specified
    version of the algorithm.

    Parameters
    ----------
    groups : list
        A list of hidden groups, where each group is a collection of hidden
        states represented as frozensets.
    version : int, optional
        An integer specifying the version of the merging algorithm to use.
        Default is 1. Version 0 and 1 are supported, with version 1 being
        more efficient and readable.

    Returns
    -------
    list of frozenset
        A list of merged hidden groups, where each group is a frozenset of
        HiddenOutVar objects. The groups are merged based on shared hidden states.
    """

    if version == 0:
        previous = frozenset([frozenset(g) for g in groups])
        while True:
            new_groups = []
            old_groups = list(previous)
            not_merged = list(range(len(old_groups)))
            while len(not_merged) > 0:
                i = not_merged.pop()
                merged = False
                for j in tuple(not_merged):
                    if len(old_groups[i].intersection(old_groups[j])) > 0:
                        new_groups.append(old_groups[i].union(old_groups[j]))
                        not_merged.remove(j)
                        merged = True
                if not merged:
                    new_groups.append(old_groups[i])
            new = frozenset([frozenset(g) for g in new_groups])
            if new == previous:
                break
            previous = new
        return list(new)

    elif version == 1:
        # This code has been upgraded for better readability and efficiency.
        prev = [frozenset(g) for g in set(map(frozenset, groups))]
        while True:
            new_groups = []
            merged_indices = set()
            for i, j in combinations(range(len(prev)), 2):
                if i in merged_indices or j in merged_indices:
                    continue
                if prev[i].intersection(prev[j]):
                    new_groups.append(prev[i].union(prev[j]))
                    merged_indices.update([i, j])
            new_groups.extend(
                prev[k]
                for k in range(len(prev))
                if k not in merged_indices
            )
            cur = frozenset(new_groups)
            if cur == frozenset(prev):
                break
            prev = list(cur)
        return list(cur)

    else:
        raise ValueError(f'Error: the version {version} is not supported.')


def find_hidden_groups_from_jaxpr(
    jaxpr: Jaxpr,
    hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
    weight_invars: Set[Var],
    invar_to_hidden_path: Dict[HiddenInVar, Path],
    outvar_to_hidden_path: Dict[HiddenOutVar, Path],
    path_to_state: Dict[Path, brainstate.State],
    include_recurrent_mixing: bool = False,
    sparse_n: Optional[int] = None,
    snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    control_flow: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
    descended_scan_eqn_ids: FrozenSet[int] = frozenset(),
    descended_hidden_paths: FrozenSet[Path] = frozenset(),
) -> Tuple[Sequence[HiddenGroup], brainstate.util.PrettyDict]:
    """
    Find hidden groups from the jaxpr.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr for the model.
    hidden_outvar_to_invar : dict
        Mapping from hidden output variable to hidden input variable.
    weight_invars : set
        Set of weight input variables.
    invar_to_hidden_path : dict
        Mapping from weight input variable to hidden state path.
    outvar_to_hidden_path : dict
        Mapping from hidden output variable to hidden state path.
    path_to_state : dict
        Mapping from hidden state path to state.
    include_recurrent_mixing : bool, optional
        Whether to trace recurrent ETP *mixing* primitives
        (``etp_mv``/``etp_mm``/``etp_conv``) into the hidden-to-hidden
        transition jaxpr.

        - ``False`` (default, "without recurrence"): these primitives are
          treated as boundaries and excluded, so the transition keeps only
          element-wise (and non-ETP) state-to-state paths. The recurrent
          weight's temporal credit is carried by its eligibility trace. This
          keeps the recurrent Jacobian ``D^t`` contractive and the trace
          bounded (the standard D-RTRL / e-prop diagonal approximation).
        - ``True`` ("with recurrence"): the mixing primitives are traced into
          the transition, so ``D^t`` carries the full per-step recurrent
          coupling. The resulting (coupled) Jacobian is extracted per
          position via :func:`block_diagonal_last_dim` (selected automatically
          by :attr:`HiddenGroup.is_diagonal_recurrence`).
    sparse_n : int or None, optional
        SnAp order for ``recurrence_scope='sparse_n'``; ``None`` (default)
        leaves every group's ``snap`` pattern unset.
    snap_max_jacobian_elements : int, optional
        Ceiling on each hidden group's widened block Jacobian,
        ``P * (K * S) ** 2`` elements. Configurations exceeding it are
        refused with :class:`~braintrace.NotSupportedError`.
    control_flow : ControlFlowPolicy, optional
        The :class:`~braintrace.ControlFlowPolicy` governing how opaque
        control-flow equations touching hidden states are handled
        (``'opaque-fwd'`` keeps a weight-free while/scan/cond as an opaque
        forward node; ``'error'`` raises as before Phase 3).
    descended_scan_eqn_ids : frozenset of int, optional
        ``id()`` values of scan equations rewritten by structured scan descent
        (Phase 4); skipped by this walker.
    descended_hidden_paths : frozenset of Path, optional
        Hidden-state paths covered by descended scan bodies; excluded from the
        zero-recurrence fallback grouping.

    Returns
    -------
    tuple
        A tuple containing:

        - Sequence of HiddenGroup objects.
        - PrettyDict mapping hidden state paths to hidden groups.
    """
    evaluator = JaxprEvalForHiddenGroup(
        jaxpr=jaxpr,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_invars=weight_invars,
        invar_to_hidden_path=invar_to_hidden_path,
        outvar_to_hidden_path=outvar_to_hidden_path,
        # the evaluator only indexes hidden-state paths, whose entries are HiddenStates,
        # even though the passed mapping carries every model state. The cast is a real
        # State -> HiddenState narrowing; mypy flags it as redundant only because
        # brainstate is currently untyped (both collapse to Any).
        path_to_state=cast(Dict[Path, brainstate.HiddenState], path_to_state),  # type: ignore[redundant-cast]
        include_recurrent_mixing=include_recurrent_mixing,
        sparse_n=sparse_n,
        snap_max_jacobian_elements=snap_max_jacobian_elements,
        control_flow=control_flow,
        descended_scan_eqn_ids=descended_scan_eqn_ids,
        descended_hidden_paths=descended_hidden_paths,
    )
    hidden_groups, hid_path_to_group = evaluator.compile()
    return hidden_groups, brainstate.util.PrettyDict(hid_path_to_group)


def find_hidden_groups_from_minfo(
    minfo: ModuleInfo,
    include_recurrent_mixing: bool = False,
    sparse_n: Optional[int] = None,
    snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    descended_scan_eqn_ids: FrozenSet[int] = frozenset(),
    descended_hidden_paths: FrozenSet[Path] = frozenset(),
):
    """Find the hidden groups from the model information.

    Parameters
    ----------
    minfo : ModuleInfo
        The model information.
    include_recurrent_mixing : bool, default False
        Whether to trace recurrent ETP mixing primitives into the transition
        jaxpr. See the internal ``find_hidden_groups_from_jaxpr`` helper for
        the full semantics.
    sparse_n : int, optional
        SnAp order for ``recurrence_scope='sparse_n'``. When given, each group
        carries the derived n-step neighbourhood in its ``snap`` field. Default
        ``None``.
    descended_scan_eqn_ids : frozenset of int, default ``frozenset()``
        ``id()`` values of scan equations rewritten by structured scan descent
        (Phase 4); those equations are skipped by the hidden-group walker.
    descended_hidden_paths : frozenset, default ``frozenset()``
        Hidden-state paths covered by descended scan bodies; excluded from the
        zero-recurrence fallback grouping.

    Returns
    -------
    hidden_groups : sequence of HiddenGroup
        The hidden groups.
    hid_path_to_group : dict
        Mapping from each hidden-state path to its :class:`HiddenGroup`.

    See Also
    --------
    find_hidden_groups_from_module : Equivalent helper starting from a model.
    """
    (
        hidden_groups,
        hid_path_to_group,
    ) = find_hidden_groups_from_jaxpr(
        jaxpr=minfo.jaxpr,
        hidden_outvar_to_invar=minfo.hidden_outvar_to_invar,
        weight_invars=set(minfo.weight_invars),
        invar_to_hidden_path=minfo.invar_to_hidden_path,
        outvar_to_hidden_path=minfo.outvar_to_hidden_path,
        path_to_state=minfo.retrieved_model_states,
        include_recurrent_mixing=include_recurrent_mixing,
        sparse_n=sparse_n,
        snap_max_jacobian_elements=snap_max_jacobian_elements,
        control_flow=minfo.control_flow,
        descended_scan_eqn_ids=descended_scan_eqn_ids,
        descended_hidden_paths=descended_hidden_paths,
    )
    return hidden_groups, hid_path_to_group


def find_hidden_groups_from_module(
    model: brainstate.nn.Module,
    *model_args,
    include_recurrent_mixing: bool = False,
    sparse_n: Optional[int] = None,
    snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    **model_kwargs,
) -> Tuple[Sequence[HiddenGroup], brainstate.util.PrettyDict]:
    """Find hidden groups from a model.

    Parameters
    ----------
    model : brainstate.nn.Module
        The model.
    *model_args
        The positional arguments of the model.
    include_recurrent_mixing : bool, default False
        Whether to trace recurrent ETP mixing primitives into the transition
        jaxpr. Keyword-only. See the internal
        ``find_hidden_groups_from_jaxpr`` helper for the full semantics.
    sparse_n : int, optional
        SnAp order for ``recurrence_scope='sparse_n'``. Keyword-only. Default
        ``None``.
    **model_kwargs
        The keyword arguments of the model.

    Returns
    -------
    hidden_groups : sequence of HiddenGroup
        The hidden groups.
    hid_path_to_group : brainstate.util.PrettyDict
        Mapping from each hidden-state path to its :class:`HiddenGroup`.

    See Also
    --------
    find_hidden_groups_from_minfo : Equivalent helper starting from ``ModuleInfo``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> hidden_groups, hid_path_to_group = braintrace.find_hidden_groups_from_module(gru, inputs)
        >>> len(hidden_groups)
        1
    """
    minfo = extract_module_info(model, *model_args, **model_kwargs)
    return find_hidden_groups_from_minfo(
        minfo,
        include_recurrent_mixing=include_recurrent_mixing,
        sparse_n=sparse_n,
        snap_max_jacobian_elements=snap_max_jacobian_elements,
    )
