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
#       - [x] split into "_etrace_compiler.py", "_etrace_vjp_compiler_graph.py", and "_etrace_compiler_hidden_group.py",
#
# ==============================================================================

# -*- coding: utf-8 -*-

from itertools import combinations
from typing import List, Dict, Sequence, Tuple, Set, Optional, Callable, NamedTuple, Any

import brainstate
import brainunit as u
import jax.core
import numpy as np
from brainstate import HiddenGroupState

from ._compatible_imports import Var, Literal, JaxprEqn, Jaxpr
from ._etrace_compiler_base import JaxprEvaluation, find_matched_vars
from ._etrace_compiler_module_info import extract_module_info, ModuleInfo
from ._misc import NotSupportedError
from ._typing import (
    PyTree,
    HiddenInVar,
    HiddenOutVar,
    Path,
)

__all__ = [
    'HiddenGroup',
    'find_hidden_groups_from_minfo',
    'find_hidden_groups_from_module',
]


class HiddenGroup(NamedTuple):
    r"""
    The data structure for recording the hidden group relation.

    The following fields are included:

    - ``hidden_paths``: the path to each hidden state
    - ``hidden_states``: the hidden states
    - ``hidden_invars``: the input jax Var of hidden states
    - ``hidden_outvars``: the output jax Var of hidden states
    - ``transition_jaxpr``: the jaxpr for computing hidden state transitions, i.e.,
      $h_1^t, h_2^t, ... = f(h_1^{t-1}, h_2^{t-1}, ..., x_t)$
    - ``transition_jaxpr_constvars``: the other input variables for jaxpr evaluation of ``transition_jaxpr``


    Example::

        >>> import braintrace
        >>> import brainstate
        >>> gru = braintrace.nn.GRUCell(10, 20)
        >>> gru.init_state()
        >>> inputs = brainstate.random.randn(10)
        >>> hidden_groups, _ = braintrace.find_hidden_groups_from_module(gru, inputs)
        >>> for group in hidden_groups:
        ...     print(group.hidden_paths)
    """

    index: int  # the index of the hidden group

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

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        The shape of each state variable.
        """
        return self.hidden_states[0].varshape

    @property
    def num_state(self) -> int:
        """
        The number of hidden states.
        """
        return sum([st.num_state for st in self.hidden_states])

    def check_consistent_varshape(self):
        """
        Checking whether the shapes of the hidden states are consistent.

        Raises:
            NotSupportedError: If the shapes of the hidden states are not consistent.
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
        r"""
        Computing the hidden state transitions $h_1^t, h_2^t, \cdots = f(h_1^{t-1}, h_2^{t-1}, \cdots, x^t)$.

        Args:
            hidden_vals: The old hidden state value.
            input_vals: The input values.

        Returns:
            The new hidden state values.
        """
        return jax.core.eval_jaxpr(self.transition_jaxpr, input_vals, *hidden_vals)

    def diagonal_jacobian(
        self,
        hidden_vals: Sequence[jax.Array],
        input_vals: PyTree,
    ):
        """
        Computing the diagonal Jacobian matrix along the last dimension.

        Args:
            hidden_vals: The hidden state values.
            input_vals: The input values.

        Returns:
            The diagonal Jacobian matrix, which has the shape of
            ``(*varshape, num_states, num_states)``.
        """
        return jacrev_last_dim(
            lambda hid: self.concat_hidden(self.transition(self.split_hidden(hid), input_vals)),
            self.concat_hidden(hidden_vals)
        )

    def concat_hidden(self, splitted_hid_vals: Sequence[jax.Array]):
        """
        Concatenate split hidden state values into a single array.

        This function takes a sequence of split hidden state values and concatenates them
        along the last axis. For non-HiddenGroupState values, it adds an extra dimension
        before concatenation.

        Args:
            splitted_hid_vals (Sequence[jax.Array]): A sequence of split hidden state
                values, each corresponding to a hidden state in the group.

        Returns:
            jax.Array: A single concatenated array containing all hidden state values.
                The concatenation is performed along the last axis.
        """
        splitted_hid_vals = [
            val
            if isinstance(st, HiddenGroupState) else
            u.math.expand_dims(val, axis=-1)
            for val, st in zip(splitted_hid_vals, self.hidden_states)
        ]
        return u.math.concatenate(splitted_hid_vals, axis=-1)

    def split_hidden(self, concat_hid_vals: jax.Array):
        """
        Split concatenated hidden state values into individual arrays.

        This function takes a concatenated array of hidden state values and splits it
        into separate arrays for each hidden state in the group. It handles both
        HiddenGroupState and non-HiddenGroupState values differently.

        Args:
            concat_hid_vals (jax.Array): A concatenated array of hidden state values.
                The last dimension is assumed to contain the concatenated states.

        Returns:
            List[jax.Array]: A list of split hidden state arrays. For non-HiddenGroupState
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
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


HiddenGroup.__module__ = 'braintrace'


def jacrev_last_dim(
    fn: Callable[[...], jax.Array],
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


class HiddenToHiddenGroupTracer(NamedTuple):
    """
    The data structure for the tracing of the hidden-to-hidden states.

    Attributes:
        hidden_invar (Var): The input variable representing the hidden state.
        connected_hidden_outvars (set[Var]): A set of output variables representing the connected hidden states.
        other_invars (set[Var]): A set of other input variables involved in the tracing.
        invar_needed_in_oth_eqns (set[Var]): A set of variables needed in other equations for trace analysis.
        trace (List[JaxprEqn]): A list of JAX equations representing the trace of operations.
    """
    hidden_invar: Var
    connected_hidden_outvars: set[Var]
    other_invars: set[Var]
    invar_needed_in_oth_eqns: set[Var]
    trace: List[JaxprEqn]

    def dict(self) -> Dict[str, Any]:
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
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


def _simplify_hid2hid_tracer(
    tracer: HiddenToHiddenGroupTracer,
    hidden_invar_to_path: Dict[HiddenInVar, Path],
    hidden_outvar_to_path: Dict[HiddenOutVar, Path],
    path_to_state: Dict[Path, brainstate.HiddenState],
) -> Hidden2GroupTransition:
    """
    Simplifying the hidden-to-hidden state tracer.

    Args:
        tracer: The hidden-to-hidden state tracer.
        hidden_invar_to_path: The mapping from the hidden input variable to the hidden state path.
        hidden_outvar_to_path: The mapping from the hidden output variable to the hidden state path.
        path_to_state: The mapping from the hidden state path to the state.

    Returns:
        The hidden-to-hidden state transition.
    """
    #
    # [first step]
    #
    # Remove the unnecessary equations in the trace.
    # The unnecessary equations are the equations
    # that do not contain the hidden states.
    tracer.invar_needed_in_oth_eqns.clear()
    new_trace = []
    whole_trace_needed_vars = set(tracer.connected_hidden_outvars)
    visited_needed_vars = set()  # needed_vars has been satisfied
    for eqn in reversed(tracer.trace):
        need_outvars = []
        for outvar in eqn.outvars:
            if outvar in whole_trace_needed_vars:
                need_outvars.append(outvar)
        if len(need_outvars):
            visited_needed_vars.update(need_outvars)
            new_trace.append(eqn)
            whole_trace_needed_vars.update([invar for invar in eqn.invars if isinstance(invar, Var)])

    # [second step]
    #
    # Checking whether the shape of each hidden state is consistent.
    # Currently, we only support the element-wise state transition.
    hidden_outvars = tuple(tracer.connected_hidden_outvars)
    invar_state = path_to_state[hidden_invar_to_path[tracer.hidden_invar]]
    for hidden_var in hidden_outvars:
        # The most direct way when the shapes of "y" and "hidden var" are the same is using "identity()" function.
        # However, there may be bugs, for examples, the output is reshaped to the same shape as the hidden state,
        # or, the split and concatenate operators are used while the shapes are the same between the outputs and
        # hidden states.
        # The most safe way is using automatic shape inverse transformation.
        #
        # However, the automatic inverse transformation may also cause errors, for example, if the following
        # operators are used:
        #     def f(a):
        #         s = jnp.sum(a, axis=[1,2], keepdims=True)
        #         return a / s
        #
        # this will result in the following jaxpr:
        #     { lambda ; a:f32[10,20,5]. let
        #         b:f32[10] = reduce_sum[axes=(1, 2)] a
        #         c:f32[10,1,1] = broadcast_in_dim[broadcast_dimensions=(0,) shape=(10, 1, 1)] b
        #         d:f32[10,20,5] = div a c
        #       in (d,) }
        #
        # It seems that the automatic shape inverse transformation is complex for handling such cases.\
        # Therefore, currently, we only consider the simple cases, and raise an error for the complex cases.

        outvar_state = path_to_state[hidden_outvar_to_path[hidden_var]]
        if invar_state.varshape != outvar_state.varshape:
            raise NotSupportedError(
                f'Currently, we only support the state group that hase the same shape. \n'
                f'However, we got {invar_state.varshape} != {outvar_state.varshape}. \n'
                f'Please check the hidden state transition function. \n\n'
                f'{invar_state}'
                f'\n\n'
                f'{outvar_state}\n'
            )

    # [third step]
    #
    # Simplify the trace
    visited_needed_vars.add(tracer.hidden_invar)
    constvars = list(whole_trace_needed_vars.difference(visited_needed_vars))
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
    ):
        # the jaxpr of the original model, assuming that the model is well-defined,
        # see the doc for the model which can be online learning compiled.
        self.jaxpr = jaxpr

        # the hidden state groups
        self.hidden_outvar_to_invar = hidden_outvar_to_invar
        self.hidden_invar_to_outvar = {invar: outvar for outvar, invar in hidden_outvar_to_invar.items()}
        hidden_invars = set(hidden_outvar_to_invar.values())
        hidden_outvars = set(hidden_outvar_to_invar.keys())
        self.path_to_state = path_to_state

        # the data structures for the tracing hidden-hidden relationships
        self.active_tracers = dict()

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
        self.active_tracers: Dict[Var, HiddenToHiddenGroupTracer] = dict()

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
            hidden_outvars = set([outvar for outvar in eqn.outvars if outvar in self.hidden_outvars])
            needed_invars = set([outvar for outvar in eqn.outvars if outvar not in self.hidden_outvars])
            if hidden_var in self.active_tracers:
                self.active_tracers[hidden_var].trace.append(eqn.replace())
                self.active_tracers[hidden_var].other_invars.update(other_invars)
                self.active_tracers[hidden_var].invar_needed_in_oth_eqns.update(needed_invars)
                self.active_tracers[hidden_var].connected_hidden_outvars.update(hidden_outvars)
            else:
                tracer = HiddenToHiddenGroupTracer(
                    hidden_invar=hidden_var,
                    connected_hidden_outvars=hidden_outvars,
                    other_invars=set(other_invars),
                    invar_needed_in_oth_eqns=needed_invars,
                    trace=[eqn.replace()]
                )
                self.active_tracers[hidden_var] = tracer

        # check whether this equation is used in other tracers
        for tracer in tuple(self.active_tracers.values()):
            tracer: HiddenToHiddenGroupTracer
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
        tracer.invar_needed_in_oth_eqns.update(eqn.outvars)

        # check whether the hidden states are needed in the other equations
        for outvar in eqn.outvars:
            if outvar in self.hidden_outvars:
                tracer.connected_hidden_outvars.add(outvar)

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
            _simplify_hid2hid_tracer(
                tracer,
                self.invar_to_hidden_path,
                self.outvar_to_hidden_path,
                self.path_to_state,
            )
            for tracer in self.active_tracers.values()
        ]

        # [ second step ]
        #
        # Find out the hidden group,
        # i.e., the hidden states that are connected to each other, the union of all hidden-to-group
        outvar_groups = [
            set(
                [self.hidden_invar_to_outvar[transition.hidden_invar]] +
                list(transition.connected_hidden_outvars)
            )
            for transition in hidden_to_group_transition
        ]
        outvar_groups = group_merging(outvar_groups, version=0)
        outvar_groups = [list(group) for group in outvar_groups]
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
                )
            )

        # [ fourth step ]
        #
        # compile HiddenGroup
        #
        hidden_groups = []
        for hidden_invars, hidden_outvars, jaxpr in zip(invar_groups, outvar_groups, jaxpr_groups):
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
    all_invars = set()
    all_outvars = set()
    for invar in hidden_invars:
        if invar in hidden_invar_to_transition:
            transition = hidden_invar_to_transition[invar]
            for eq in transition.transition_jaxpr.eqns:
                this_eq_exist = [outvar in all_outvars for outvar in eq.outvars]
                # this_eq_exist = False
                # for outvar in eq.outvars:
                #     if outvar in all_outvars:
                #         this_eq_exist = True
                #         break
                if not all(this_eq_exist):
                    eqns.append(eq.replace())
                    all_invars.update([invar for invar in eq.invars if not isinstance(invar, Literal)])
                    all_outvars.update(eq.outvars)
    other_invars = all_invars.difference(all_outvars).difference(hidden_invars)
    other_invars = list(other_invars)

    #
    # step 2:
    #
    # order the equations so that data dependencies are satisfied
    #
    new_eqns = []
    env = set(list(hidden_invars) + other_invars)
    while len(eqns) > 0:
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
        eqns=new_eqns
    )


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
        previous = [frozenset(g) for g in set(map(frozenset, groups))]
        while True:
            new_groups = []
            merged_indices = set()
            for i, j in combinations(range(len(previous)), 2):
                if i in merged_indices or j in merged_indices:
                    continue
                if previous[i].intersection(previous[j]):
                    new_groups.append(previous[i].union(previous[j]))
                    merged_indices.update([i, j])
            new_groups.extend(
                previous[k]
                for k in range(len(previous))
                if k not in merged_indices
            )
            new = frozenset(new_groups)
            if new == frozenset(previous):
                break
            previous = list(new)
        return list(new)

    else:
        raise ValueError(f'Error: the version {version} is not supported.')


def find_hidden_groups_from_jaxpr(
    jaxpr: Jaxpr,
    hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
    weight_invars: Set[Var],
    invar_to_hidden_path: Dict[HiddenInVar, Path],
    outvar_to_hidden_path: Dict[HiddenOutVar, Path],
    path_to_state: Dict[Path, brainstate.State],
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
        path_to_state=path_to_state,
    )
    hidden_groups, hid_path_to_group = evaluator.compile()
    return hidden_groups, brainstate.util.PrettyDict(hid_path_to_group)


def find_hidden_groups_from_minfo(
    minfo: ModuleInfo
):
    """
    Finding the hidden groups from the model.

    Args:
        minfo: The model information.

    Returns:
        The hidden groups,
        and the mapping from the hidden state path to the hidden group.
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
    )
    return hidden_groups, hid_path_to_group


def find_hidden_groups_from_module(
    model: brainstate.nn.Module,
    *model_args,
    **model_kwargs,
) -> Tuple[Sequence[HiddenGroup], brainstate.util.PrettyDict]:
    """
    Find hidden groups from the model.

    Args:
        model: The model.
        model_args: The model arguments.
        model_kwargs: The model keyword arguments.

    Returns:
        A tuple containing:
        - Sequence of HiddenGroup objects
        - PrettyDict mapping hidden state paths to hidden groups
    """
    minfo = extract_module_info(model, *model_args, **model_kwargs)
    return find_hidden_groups_from_minfo(minfo)
