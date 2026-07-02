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
from typing import List, Dict, Sequence, Tuple, Set, Optional, Callable, NamedTuple, Any, cast

import brainstate
import brainunit as u
import jax.core
import numpy as np
from brainstate import HiddenGroupState

from braintrace._compatible_imports import Var, Literal, JaxprEqn, Jaxpr
from braintrace._op import is_etp_primitive, is_etp_enable_gradient_primitive
from braintrace._misc import NotSupportedError
from braintrace._typing import (
    PyTree,
    HiddenInVar,
    HiddenOutVar,
    Path,
)
from .base import JaxprEvaluation, find_matched_vars
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit
from .module_info import extract_module_info, ModuleInfo

__all__ = [
    'HiddenGroup',
    'find_hidden_groups_from_minfo',
    'find_hidden_groups_from_module',
]

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
            ``(*varshape, num_states, num_states)``. Entry ``[p, a, b]`` is
            ``d h^t[p, a] / d h^{t-1}[p, b]`` -- the cross-position terms
            ``d h^t[p] / d h^{t-1}[q]`` (``p != q``) are intentionally dropped
            (the D-RTRL / e-prop diagonal approximation).

        Notes
        -----
        For diagonal recurrence (:attr:`is_diagonal_recurrence` is ``True``) the
        positions are independent, so the cheap column-sum produced by
        :func:`jacrev_last_dim` already equals this block diagonal. For coupled
        recurrence the column sum would instead add in the off-diagonal
        cross-position terms -- inflating every entry and driving the eligibility
        trace to overflow -- so the true block diagonal is extracted directly via
        :func:`block_diagonal_last_dim`.
        """
        fn = lambda hid: self.concat_hidden(self.transition(self.split_hidden(hid), input_vals))
        concat_hid = self.concat_hidden(hidden_vals)
        if self.is_diagonal_recurrence:
            return jacrev_last_dim(fn, concat_hid)
        return block_diagonal_last_dim(fn, concat_hid)

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

    Args:
        fn (Callable[[...], jax.Array]): The function for which to compute
            the Jacobian. It should take a JAX array as input and return
            a JAX array.
        hid_vals (jax.Array): The input values for which to compute the
            Jacobian. The last dimension is considered as the dimension
            of interest.

    Returns:
        jax.Array: The Jacobian matrix. Its shape is (*varshape, num_state, num_state),
        where varshape is the shape of the input excluding the last dimension,
        and num_state is the size of the last dimension.

    Raises:
        AssertionError: If the number of input and output states are not the same.
    """
    new_hid_vals, f_vjp = jax.vjp(fn, hid_vals)
    num_state = new_hid_vals.shape[-1]
    varshape = new_hid_vals.shape[:-1]
    assert num_state == hid_vals.shape[-1], 'Error: the number of input/output states should be the same.'
    g_primals = u.math.broadcast_to(u.math.eye(num_state), (*varshape, num_state, num_state))
    jac = jax.vmap(f_vjp, in_axes=-2, out_axes=-2)(g_primals)
    return jac[0]


def block_diagonal_last_dim(
    fn: Callable[..., jax.Array],
    hid_vals: jax.Array,
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
    full_jac = jax.jacrev(fn)(hid_vals)  # (*varshape, num_state, *varshape, num_state)
    full_jac = u.math.reshape(full_jac, (num_pos, num_state, num_pos, num_state))
    # take the per-position block: block[p] = full_jac[p, :, p, :]
    block = u.math.diagonal(full_jac, axis1=0, axis2=2)  # (num_state, num_state, num_pos)
    block = u.math.moveaxis(block, -1, 0)  # (num_pos, num_state, num_state)
    return u.math.reshape(block, (*varshape, num_state, num_state))


class HiddenToHiddenGroupTracer(NamedTuple):
    """
    The data structure for the tracing of the hidden-to-hidden states.

    The variable collections are insertion-ordered ``dict`` objects used as
    ordered sets (values are always ``None``), so every downstream ordering
    (group members, transition constvars) is deterministic across processes;
    plain ``set`` iteration follows memory-address hashes for jaxpr ``Var``
    objects.

    Attributes:
        hidden_invar (Var): The input variable representing the hidden state.
        connected_hidden_outvars (Dict[Var, None]): Ordered set of output variables representing the connected hidden states.
        other_invars (Dict[Var, None]): Ordered set of other input variables involved in the tracing.
        invar_needed_in_oth_eqns (Dict[Var, None]): Ordered set of variables needed in other equations for trace analysis.
        trace (List[JaxprEqn]): A list of JAX equations representing the trace of operations.
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

    Attributes:
        hidden_invar (Var): The input variable representing the hidden state at the previous time step.
        hidden_path (Path): The path to the hidden state in the model hierarchy.
        connected_hidden_outvars (List[Var]): A list of output variables representing the connected hidden states at the current time step.
        connected_hidden_paths (List[Path]): A list of paths to the connected hidden states in the model hierarchy.
        transition_jaxpr (Jaxpr): The JAX program representation for computing the hidden state transitions.
        other_invars (List[Var]): A list of other input variables required for evaluating the transition_jaxpr.
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

        Args:
          old_hidden_val: The old hidden state value.
          other_input_vals: The input values.
          return_index: index of the hidden state to return.

        Returns:
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

    Args:
        tracer: The hidden-to-hidden state tracer.
        hidden_invar_to_path: The mapping from the hidden input variable to the hidden state path.
        hidden_outvar_to_path: The mapping from the hidden output variable to the hidden state path.
        path_to_state: The mapping from the hidden state path to the state.
        debug_info: The debug info threaded from the source model jaxpr onto the
            simplified transition jaxpr (avoids the missing-DebugInfo deprecation).

    Returns:
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

    Args:
        jaxpr: The jaxpr for the model.
        hidden_outvar_to_invar: The mapping from the hidden output variable to the hidden input variable.
        weight_invars: The weight input variables.
        invar_to_hidden_path: The mapping from the weight input variable to the hidden state path.
        outvar_to_hidden_path: The mapping from the hidden output variable to the hidden state path.
        path_to_state: The mapping from the hidden state path to the state.
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
    ):
        # the jaxpr of the original model, assuming that the model is well-defined,
        # see the doc for the model which can be online learning compiled.
        self.jaxpr = jaxpr

        # whether the recurrent ETP mixing primitives (``etp_mv``/``etp_mm``/
        # ``etp_conv``) are *traced into* the hidden-to-hidden transition jaxpr
        # (``True``) or treated as boundaries and excluded (``False``, default).
        # See :func:`find_hidden_groups_from_jaxpr` for the rationale.
        self.include_recurrent_mixing = include_recurrent_mixing

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
            outvar_to_hidden_path=outvar_to_hidden_path
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
                    f'{hidden_paths}'
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

    Args:
        groups: A list of hidden groups, where each group is a collection of
            hidden states represented as frozensets.
        version: An integer specifying the version of the merging algorithm to use.
            Default is 1. Version 0 and 1 are supported, with version 1 being
            more efficient and readable.

    Returns:
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
) -> Tuple[Sequence[HiddenGroup], brainstate.util.PrettyDict]:
    """
    Find hidden groups from the jaxpr.

    Args:
        jaxpr: The jaxpr for the model.
        hidden_outvar_to_invar: Mapping from hidden output variable to hidden input variable.
        weight_invars: Set of weight input variables.
        invar_to_hidden_path: Mapping from weight input variable to hidden state path.
        outvar_to_hidden_path: Mapping from hidden output variable to hidden state path.
        path_to_state: Mapping from hidden state path to state.
        include_recurrent_mixing: Whether to trace recurrent ETP *mixing* primitives
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

    Returns:
        A tuple containing:
        - Sequence of HiddenGroup objects
        - PrettyDict mapping hidden state paths to hidden groups
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
    )
    hidden_groups, hid_path_to_group = evaluator.compile()
    return hidden_groups, brainstate.util.PrettyDict(hid_path_to_group)


def find_hidden_groups_from_minfo(
    minfo: ModuleInfo,
    include_recurrent_mixing: bool = False,
):
    """Find the hidden groups from the model information.

    Parameters
    ----------
    minfo : ModuleInfo
        The model information.
    include_recurrent_mixing : bool, default False
        Whether to trace recurrent ETP mixing primitives into the transition
        jaxpr. See :func:`find_hidden_groups_from_jaxpr` for the full semantics.

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
    )
    return hidden_groups, hid_path_to_group


def find_hidden_groups_from_module(
    model: brainstate.nn.Module,
    *model_args,
    include_recurrent_mixing: bool = False,
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
        jaxpr. Keyword-only. See :func:`find_hidden_groups_from_jaxpr` for the
        full semantics.
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
    return find_hidden_groups_from_minfo(minfo, include_recurrent_mixing=include_recurrent_mixing)
