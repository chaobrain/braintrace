# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


import warnings
from typing import List, Dict, Tuple, Sequence, NamedTuple, Any

import brainstate
import jax.core
from jax.extend import source_info_util

from ._compatible_imports import (
    Var,
    Literal,
    JaxprEqn,
    Jaxpr,
)
from ._etrace_compiler_base import (
    JaxprEvaluation,
    check_unsupported_op,
    find_matched_vars,
)
from ._etrace_compiler_hidden_group import (
    HiddenGroup,
    find_hidden_groups_from_minfo,
)
from ._etrace_compiler_module_info import (
    extract_module_info,
    ModuleInfo,
)
from ._etrace_concepts import ETraceParam
from ._etrace_debug_jaxpr2code import jaxpr_to_python_code
from ._etrace_operators import (
    is_etrace_op,
    is_etrace_op_enable_gradient,
    is_etrace_op_elemwise,
)
from ._misc import (
    git_issue_addr,
    NotSupportedError,
    CompilationError,
)
from ._typing import (
    WeightXVar,
    WeightYVar,
    HiddenInVar,
    HiddenOutVar,
    Path
)

__all__ = [
    'HiddenParamOpRelation',
    'find_hidden_param_op_relations_from_minfo',
    'find_hidden_param_op_relations_from_module',
]


# TODO
#
# - [x] The visualization of the etrace graph.
# - [ ] Evaluate whether the `df` is the same for different weights.
#       For example,
#
#          h = f(x1 @ w1 + x2 @ w2)
#
#       The `df` for w1 and w2 are the same, although them have the different weight y.

class HiddenParamOpRelation(NamedTuple):
    r"""
    The data structure for recording the hidden group, parameter, and operator relationship.

    This is one of the most important data structures for the eligibility trace compiler.
    It summarizes the parameter, operator, and hidden group relationship, which is used for computing
    the weight spatial gradients and the hidden state Jacobian.

    Usually, the hidden state $h^t$, the weight $\theta$, and the operator $f$ are connected in the following way:

    $$
    h^t = f(y), \quad y = x @ \theta,
    $$

    where $x$ is the input data, $\theta$ is the weight, $y$ is the weight output,
    and $h^t$ is the hidden state at time $t$. The operator $@$ is the operator that transforms
    the input data $x$ into the weight output $y$. The operator $f$ is the operator that transforms
    the weight output $y$ into the hidden state $h^t$.


    An instance of :py:class:`HiddenParamOpRelation` records the following information:

    - ``weight``: the instance of :class:`ETraceParam`, i.e., $\theta$
    - ``path``: the path to the weight.
    - ``x``: the jax Var for the weight input, i.e., $x$. It can be None if the weight is a :class:`ElemWiseParam` instance.
    - ``y``: the jax Var for the weight output, i.e., $y$.
    - ``hidden_groups``: the hidden groups that the weight is associated with, i.e., $h^t$.
    - ``y_to_hidden_group_jaxprs``: the jaxpr for computing y --> hidden groups, i.e., $f$.
    - ``connected_hidden_paths``: the connected hidden paths.

    .. note::

        :py:class:`HiddenParamOpRelation` is uniquely identified by the ``y`` variable.

        Each parameter weight may be accompanied by multiple :class:`HiddenParamOpRelation` instances.
        This is because the weight may be used in multiple times.

    Example::

        >>> import braintrace
        >>> import brainstate
        >>> gru = braintrace.nn.GRUCell(10, 20)
        >>> gru.init_state()
        >>> inputs = brainstate.random.randn(10)
        >>> hpo_relations = braintrace.find_hidden_param_op_relations_from_module(gru, inputs)
        >>> for relation in hpo_relations:
        ...     print(relation)
    """

    weight: ETraceParam  # the weight itself
    path: Path  # the path to the weight
    x: WeightXVar | None  # the input jax var, None if the weight is ElemWiseParam
    y: WeightYVar  # the output jax var
    hidden_groups: List[HiddenGroup]  # the hidden groups that the weight is associated with
    y_to_hidden_group_jaxprs: List[Jaxpr]  # the jaxpr for computing y --> hidden groups
    connected_hidden_paths: List[Path]  # the connected hidden paths

    def y_to_hidden_groups(
        self,
        y_val: jax.Array,
        const_vals: Dict[Var, jax.Array],
        concat_hidden_vals: bool = True
    ):
        """
        Computing the hidden groups from the weight output.

        Args:
            y_val: The value of the weight output.
            const_vals: The constant values for the jax variables.
            concat_hidden_vals: Whether to concatenate the hidden values.

        Returns:
            The hidden states.
        """
        vals_of_hidden_groups = []
        for jaxpr, group in zip(self.y_to_hidden_group_jaxprs, self.hidden_groups):
            assert len(jaxpr.invars) == 1, 'The weight y should be unique.'
            consts = [const_vals[var] for var in jaxpr.constvars]
            hidden_vals = jax.core.eval_jaxpr(
                jaxpr,
                consts,
                y_val,
            )
            if concat_hidden_vals:
                hidden_vals = group.concat_hidden(hidden_vals)
            vals_of_hidden_groups.append(hidden_vals)
        return vals_of_hidden_groups

    def dict(self) -> Dict[str, Any]:
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


HiddenParamOpRelation.__module__ = 'braintrace'


def _trace_simplify(
    tracer: 'HiddenWeightOpTracer',
    hid_path_to_group: Dict[Path, HiddenGroup],
    state_id_to_path: Dict[int, Path],
    outvar_to_hidden_path: Dict[Var, Path],
) -> HiddenParamOpRelation | None:
    """
    Simplifying the trace from the weight output to the hidden state.

    Args:
        tracer: The traced weight operation.
        hid_path_to_group: The mapping from the hidden state path to the hidden group.
        state_id_to_path: The mapping from the state id to the state path.
        outvar_to_hidden_path: The mapping from the output variable to the hidden state path.

    Returns:
        The simplified traced weight operation.
    """

    # [First step]
    # 
    # Finding out how the shape of each hidden state is converted to the size of df.
    y = tracer.y
    connected_hidden_vars = []
    connected_hidden_paths = []
    for hidden_var in list(tracer.hidden_vars):
        # The direct way to check whether the shapes of "y" and "hidden var" are the same is
        # using "identity()" function. However, there may be bugs, for examples, the output is
        # reshaped to the same shape as the hidden state, or, the split and concatenate operators
        # are used while the shapes are the same between the outputs and hidden states.
        #
        # The most safe way is using automatic shape inverse transformation.
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
        # It seems that the automatic shape inverse transformation is complex for handling such cases.
        #
        # Therefore, currently, we only consider the simple cases, and raise an error for the complex cases.

        try:
            jax.numpy.broadcast_shapes(y.aval.shape, hidden_var.aval.shape)
            connected_hidden_paths.append(outvar_to_hidden_path[hidden_var])
            connected_hidden_vars.append(hidden_var)
        except ValueError:
            msg = (
                f'\n'
                f'In your computational graph, we found ETraceParam {tracer.weight_path} is connected '
                f'with hidden state {outvar_to_hidden_path[hidden_var]}.\n'
                f'Our online learning is only applied to the case of: h^t = f(y), y = x @ w, '
                f'where y has the broadcastable shape with h^t. \n'
                f'while we found the shape of y is {y.aval.shape} and the shape of h^t is {hidden_var.aval.shape}.\n'
                f'Therefore, we remove the connection between the weight {tracer.weight_path} and the hidden state '
                f'{outvar_to_hidden_path[hidden_var]}.\n'
                f'If you found this is a bug, please report an issue to the developers at {git_issue_addr}. \n'
            )
            warnings.warn(msg, UserWarning)

    # [Second step]
    # 
    # check hidden states once again
    tracer = tracer.replace(hidden_vars=connected_hidden_vars)
    _post_check(tracer)
    # if there are no connected-hidden paths, we return None
    if len(connected_hidden_paths) == 0:
        return None

    # [Third step]
    # 
    # find out all hidden groups that this weight operation is associated with
    hidden_group_ids = set()
    connected_hidden_groups = []
    for path in connected_hidden_paths:
        group = hid_path_to_group[path]
        if group.index not in hidden_group_ids:
            hidden_group_ids.add(group.index)
            connected_hidden_groups.append(group)

    # [fourth step]
    # 
    # create the weight "y" to hidden groups jaxpr
    # each hidden group has its own jaxpr
    # 
    # The key is to remove the unnecessary equations in the trace.
    # 
    # The unnecessary equations are the equations
    # that do not contain the hidden states in one hidden group

    tracer.invar_needed_in_oth_eqns.clear()
    y_to_hid_group_jaxprs = []
    for group in connected_hidden_groups:
        group: HiddenGroup

        # find the trace for this hidden group
        new_trace = []
        whole_trace_needed_vars = set(group.hidden_outvars)
        visited_needed_vars = set()
        for eqn in reversed(tracer.trace):
            need_outvars = []
            for outvar in eqn.outvars:
                if outvar in whole_trace_needed_vars:
                    need_outvars.append(outvar)
            if len(need_outvars):
                for outvar in need_outvars:
                    visited_needed_vars.add(outvar)
                new_trace.append(eqn)
                whole_trace_needed_vars.update([invar for invar in eqn.invars if isinstance(invar, Var)])

        # reverse the equations
        equations = list(reversed(new_trace))

        # Simplify the trace
        visited_needed_vars.add(tracer.y)
        jaxpr_opt = Jaxpr(
            # the const vars are not the hidden states, they are
            # intermediate data that are not used in the hidden states
            constvars=list(whole_trace_needed_vars.difference(visited_needed_vars)),
            # the invars are always the weight output
            invars=[tracer.y],
            # the outvars are always the connected hidden states of this weight
            outvars=list(group.hidden_outvars),
            # the new equations which are simplified
            eqns=equations,
        )

        # append the jaxpr to the list
        y_to_hid_group_jaxprs.append(jaxpr_opt)

    # [final step]
    # 
    # Change the "HiddenWeightOpTracer" to "HiddenParamOpRelation"
    # 
    return HiddenParamOpRelation(
        weight=tracer.weight,
        path=state_id_to_path[id(tracer.weight)],
        x=tracer.x,
        y=tracer.y,
        hidden_groups=brainstate.util.PrettyList(connected_hidden_groups),
        y_to_hidden_group_jaxprs=y_to_hid_group_jaxprs,
        connected_hidden_paths=brainstate.util.PrettyList(connected_hidden_paths),
    )


def _jax_eqn_to_jaxpr(eqn: JaxprEqn) -> Jaxpr:
    """
    Convert the jax equation to the jaxpr.

    Args:
        eqn: The jax equation.

    Returns:
        The jaxpr.
    """
    return Jaxpr(
        constvars=[],
        invars=eqn.invars,
        outvars=eqn.outvars,
        eqns=[eqn]
    )


def _post_check(trace: 'HiddenWeightOpTracer') -> 'HiddenWeightOpTracer':
    # Check the hidden states of the given weight. If the hidden states are not
    # used in the model, we raise an error. This is to avoid the situation that
    # the weight is defined but not used in the model.
    if len(trace.hidden_vars) == 0:
        source_info = trace.weight.source_info
        name_stack = source_info_util.current_name_stack() + source_info.name_stack
        with source_info_util.user_context(source_info.traceback, name_stack=name_stack):
            trace.weight.is_etrace = False
            msg = (
                f'\n'
                f'Warning: The ETraceParam {trace.weight_path} does not found the associated hidden states. \n'
                f'We have changed is as a weight that is not trained with eligibility trace. However, if you \n'
                f'found this is a compilation error, please report an issue to the developers at '
                f'{git_issue_addr}. \n\n'
            )
            warnings.warn(msg, UserWarning)
    return trace


class HiddenWeightOpTracer(NamedTuple):
    """
    The data structure for tracing ETraceParam operations through the computational graph.

    This class keeps track of connections between weights, operations, and hidden states
    during the compilation process of eligibility trace. It maintains information about
    how weights are transformed by operations and how they connect to hidden states.

    Attributes:
        op (JaxprEqn): The JAX equation representing the operation that transforms x and weight into y.
        weight (ETraceParam): The weight parameter being traced.
        weight_path (Path): The path to the weight in the module hierarchy.
        x (Var): The input variable to the operation (None for elementwise parameters).
        y (Var): The output variable from the operation.
        trace (List[JaxprEqn]): The sequence of JAX equations connecting the weight output to hidden states.
        hidden_vars (set[Var]): The set of hidden state variables connected to this weight.
        invar_needed_in_oth_eqns (set[Var]): Temporary set of variables needed in subsequent equations
                                             for trace analysis.

    This class is used during the compilation process to track how weights are connected to
    hidden states through a series of operations, which is essential for computing
    eligibility traces and implementing online learning.
    """
    op: JaxprEqn  # f: how x is transformed into y, i.e., y = f(x, w)
    weight: ETraceParam  # w
    weight_path: Path  # w
    x: Var  # y
    y: Var  # x
    trace: List[JaxprEqn]
    hidden_vars: set[Var]
    invar_needed_in_oth_eqns: set[Var]

    def replace(
        self,
        weight=None,
        weight_path=None,
        op=None,
        x=None,
        y=None,
        trace=None,
        hidden_vars=None,
        invar_needed_in_oth_eqns=None
    ):
        return HiddenWeightOpTracer(
            op=(op if op is not None else self.op),
            weight=(weight if weight is not None else self.weight),
            weight_path=(weight_path if weight_path is not None else self.weight_path),
            x=(x if x is not None else self.x),
            y=(y if y is not None else self.y),
            trace=(trace if trace is not None else self.trace),
            hidden_vars=(hidden_vars if hidden_vars is not None else self.hidden_vars),
            invar_needed_in_oth_eqns=(invar_needed_in_oth_eqns
                                      if invar_needed_in_oth_eqns is not None
                                      else self.invar_needed_in_oth_eqns)
        )

    def dict(self) -> Dict[str, Any]:
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


class JaxprEvalForWeightOpHiddenRelation(JaxprEvaluation):
    """
    Evaluating the jaxpr for extracting the etrace (hidden, operator, weight) relationships.

    Args:
        jaxpr: The jaxpr for the model.
        weight_path_to_invars: The mapping from the weight id to the jax vars.
        invar_to_weight_path: The mapping from the jax var to the weight id.
        path_to_state: The mapping from the state id to the state.

    Returns:
        The list of the traced weight operations.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        jaxpr: Jaxpr,
        hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
        weight_path_to_invars: Dict[Path, List[Var]],
        invar_to_weight_path: Dict[Var, Path],
        path_to_state: Dict[Path, brainstate.State],
        state_id_to_path: Dict[int, Path],
        weight_invars: set[Var],
        hid_path_to_group: Dict[Path, HiddenGroup],
        invar_to_hidden_path: Dict[HiddenInVar, Path],
        outvar_to_hidden_path: Dict[HiddenOutVar, Path],
    ):
        # the jaxpr of the original model, assuming that the model is well-defined,
        # see the doc for the model which can be online learning compiled.
        self.jaxpr = jaxpr

        #  the mapping from the weight id to the jax vars, one weight id may contain multiple jax vars
        self.weight_path_to_invars = weight_path_to_invars

        # the mapping from the jax var to the weight id, one jax var for one weight id
        self.invar_to_weight_path = invar_to_weight_path

        # the mapping from the state id to the state
        self.path_to_state = path_to_state
        self.state_id_to_path = state_id_to_path

        # jax vars of weights
        self.hid_path_to_group = hid_path_to_group

        super().__init__(
            weight_invars=weight_invars,
            hidden_invars=set(hidden_outvar_to_invar.values()),
            hidden_outvars=set(hidden_outvar_to_invar.keys()),
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path
        )

    def compile(self) -> Sequence[HiddenParamOpRelation]:
        """
        Compiling the jaxpr for the etrace relationships.
        """

        # TODO:
        # - [x] Add the traceback information for the error messages. [done at 2024-04-06]
        # - [ ] Add the support for the scan, while, cond, pjit, and other operators.
        # - [ ] Add the support for the pytree inputs and outputs within one etrace operator.
        #       Currently, there is no need to consider this.

        # the data structures for the tracing weights, variables and operations
        self.active_tracings: List[HiddenWeightOpTracer] = []

        # evaluating the jaxpr
        self._eval_jaxpr(self.jaxpr)

        # finalizing the traces
        final_traces = [
            _trace_simplify(
                _post_check(trace),
                hid_path_to_group=self.hid_path_to_group,
                state_id_to_path=self.state_id_to_path,
                outvar_to_hidden_path=self.outvar_to_hidden_path,
            )
            for trace in self.active_tracings
        ]

        # reset the temporal data structures
        self.active_tracings = []
        return tuple([trace for trace in final_traces if trace is not None])

    def _eval_pjit(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.
        """

        # --- part 1:  is etrace operator  --- #

        if is_etrace_op(eqn.params['name']):
            # checking outvars
            if len(eqn.outvars) != 1:
                name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
                with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                    raise NotSupportedError(
                        f'Currently, the etrace operator only supports single input and single output. \n'
                        f'But we got {len(eqn.outvars)} outputs in the following operator: \n\n'
                        f'The Jaxpr for the operator: \n\n'
                        f'{eqn} \n\n'
                        f'The corresponding Python code for the operator: \n\n'
                        f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                        f'You may need to define the operator as multiple operators, or raise an issue '
                        f'to the developers at {git_issue_addr} if you found this is a bug. \n'
                        f'Moreover, see the above traceback information for where the operation is defined in your code.'
                    )

            # Check old traces
            # If the old traces are valid, we add the new trace to the old
            # traces. If not, we remove the old traces.
            self._eval_old_traces_are_valid_or_not(eqn)

            # input, output, weight checking
            weight_path, x = self._get_state_and_inp_and_checking(eqn)

            # add new trace
            self.active_tracings.append(
                HiddenWeightOpTracer(
                    weight=self.path_to_state[weight_path],
                    weight_path=weight_path,
                    x=x,
                    y=eqn.outvars[0],
                    # --- The jaxpr for the operator [TODO] checking whether there are bugs
                    # Although the jaxpr var are not the same are the eqn var, we can still
                    # use this closed jaxpr expression, since the ordering of the vars are the same.
                    # Therefore, once the arguments and parameters are given correctly, the jaxpr
                    # can be used to evaluate the same operator.
                    # op=eqn.params['jaxpr'],
                    # ---- changed it to the JaxprEqn (@chaoming0625, 16/04/2024)
                    op=eqn,
                    trace=[],  # the following eqns to hidden states
                    hidden_vars=set(),  # the jax var of hidden states
                    invar_needed_in_oth_eqns=set(eqn.outvars)  # temporary data for tracing eqn to hidden states
                )
            )

        #   --- part 2:  not etrace operator  --- #
        else:
            # check whether the operator is supported
            check_unsupported_op(self, eqn, 'jit')

            # treat the pjit as a normal jaxpr equation
            self._eval_eqn(eqn)

    def _eval_eqn(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the normal jaxpr equation.
        """
        if eqn.primitive.name == 'stop_gradient':
            return
        for trace in tuple(self.active_tracings):
            matched = find_matched_vars(
                eqn.invars,
                trace.invar_needed_in_oth_eqns
            )
            # if matched, add the eqn to the trace
            # if not matched, skip
            if len(matched):
                self._add_eqn_in_a_trace(eqn, trace)

    def _eval_old_traces_are_valid_or_not(self, eqn: JaxprEqn) -> None:
        for trace in tuple(self.active_tracings):
            # Avoid "Weight -> Hidden -> Weight" pathway.
            # But the "Weight -> Weight -> Hidden" pathway is allowed.
            # However, it is hard to correctly handle the following pathways:
            #              Hidden -> Weight
            #            /
            #     Weight -> Weight -> Hidden
            #
            # This kind of connection pathways may also not be possible in real neural circuits.
            # But we need to consider the possibility of the existence of such pathways in the future (TODO).

            matched = find_matched_vars(eqn.invars, trace.invar_needed_in_oth_eqns)
            if len(matched) > 0:  # weight -> ? -> weight
                # TODO: how to judge this kind of pathway?
                # The current solution is only applied to the deep neural network models,
                # since the weights, hidden states, and operators are well-defined along the
                # depth. However, for a very complex recurrent graph models, the weights, hidden
                # states, and operators may be connected in a very complex way. Therefore, we
                # need to consider the handling of such complex models in the future.
                if len(trace.hidden_vars) > 0:  # weight -> hidden -> weight:
                    pass
                else:  # weight -> weight -> ?
                    if is_etrace_op_enable_gradient(eqn.params['name']):
                        # weight -> diagonal weight -> ?
                        self._add_eqn_in_a_trace(eqn, trace)
                    else:
                        # avoid off " weight -> weight -> ? "  pathway
                        pass

    def _add_eqn_in_a_trace(
        self,
        eqn: JaxprEqn,
        trace: HiddenWeightOpTracer
    ) -> None:
        trace.trace.append(eqn.replace())
        trace.invar_needed_in_oth_eqns.update(eqn.outvars)
        # check whether the hidden states are needed in the other equations
        for outvar in eqn.outvars:
            if outvar in self.hidden_outvars:
                trace.hidden_vars.add(outvar)

    def _get_state_and_inp_and_checking(
        self,
        eqn: JaxprEqn
    ) -> Tuple[Path, Var]:

        # Currently, only single input/output are supported, i.e.,
        #       y = f(x, w1, w2, ...)
        # This may be changed in the future, to support multiple inputs and outputs, i.e.,
        #       y1, y2, ... = f(x1, x2, ..., w1, w2, ...)
        #
        # However, I do not see any possibility or necessity for this kind of design in the
        # current stage. In most situations, single input/output is enough for the brain dynamics model.

        found_invars_in_this_op = set()
        weight_paths = set()
        xs = []
        for invar in eqn.invars:
            if isinstance(invar, Literal):
                xs.append(invar)
                continue
            weight_path = self.invar_to_weight_path.get(invar, None)
            if weight_path is None:
                xs.append(invar)
            else:
                weight_paths.add(weight_path)
                found_invars_in_this_op.add(invar)

        # --- checking whether the weight variables are all used in the same etrace operation --- #
        if len(weight_paths) == 0:
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                raise CompilationError(
                    f'Error: no ETraceParam are found in this operation: \n\n'
                    f'The Jaxpr for the operator: \n\n'
                    f'{eqn} \n\n'
                    f'The corresponding Python code for the operator: \n\n'
                    f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                    f'See the above traceback information for where the operation is defined in your code.'
                )

        if len(weight_paths) > 1:
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                raise CompilationError(
                    f'Error: multiple ETraceParam ({weight_paths}) are found in this operation. '
                    f'This is not allowed for automatic online learning: \n\n'
                    f'The Jaxpr for the operator: \n\n'
                    f'{eqn} \n\n'
                    f'The corresponding Python code for the operator: \n\n'
                    f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                    f'See the above traceback information for where the operation is defined in your code.'
                )

        weight_path = tuple(weight_paths)[0]  # the only ETraceParam found in the operation
        if len(found_invars_in_this_op.difference(set(self.weight_path_to_invars[weight_path]))) > 0:
            name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                raise CompilationError(
                    f'Error: The found jax vars are {found_invars_in_this_op}, '
                    f'but the ETraceParam contains vars {self.weight_path_to_invars[weight_path]}. \n'
                    f'This means that the operator has used multiple ETraceParam. '
                    f'Please define the trainable weights in a single ETraceParam. \n\n'
                    f'The Jaxpr for the operator: \n\n'
                    f'{eqn} \n\n'
                    f'The corresponding Python code for the operator: \n\n'
                    f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                    f'See the above traceback information for where the operation is defined in your code.'
                )

        if is_etrace_op_elemwise(eqn.params['name']):
            # element-wise operator do not support input data

            if len(xs) != 0:
                name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
                with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                    raise CompilationError(
                        f'Currently, the element-wise etrace operator do not support input. But we got {xs} \n'
                        f'You may need to check your model, or raise an issue to the developers at {git_issue_addr} '
                        f'if you found this is a bug.\n\n'
                        f'The Jaxpr for the operator: \n\n'
                        f'{eqn} \n\n'
                        f'The corresponding Python code for the operator: \n\n'
                        f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                        f'See the above traceback information for where the operation is defined in your code.'
                    )
            xs = [None]
        else:
            # other operators only support single input data

            if len(xs) != 1:
                name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
                with source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack):
                    raise CompilationError(
                        'Currently, the etrace operator only supports single input. \n'
                        'You may need to define the model as multiple operators, or raise an issue '
                        f'to the developers at {git_issue_addr}.\n\n'
                        f'The Jaxpr for the operator: \n\n'
                        f'{eqn} \n\n'
                        f'The corresponding Python code for the operator: \n\n'
                        f'{jaxpr_to_python_code(_jax_eqn_to_jaxpr(eqn))} \n\n'
                        f'See the above traceback information for where the operation is defined in your code.'
                    )

        # --- get the weight id and the input variable --- #
        return weight_path, xs[0]


def find_hidden_param_op_relations_from_jaxpr(
    jaxpr: Jaxpr,
    hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
    weight_path_to_invars: Dict[Path, List[Var]],
    invar_to_weight_path: Dict[Var, Path],
    path_to_state: Dict[Path, brainstate.State],
    state_id_to_path: Dict[int, Path],
    weight_invars: set[Var],
    hid_path_to_group: Dict[Path, HiddenGroup],
    invar_to_hidden_path: Dict[HiddenInVar, Path],
    outvar_to_hidden_path: Dict[HiddenOutVar, Path],
) -> Sequence[HiddenParamOpRelation]:
    """
    Finding the hidden-param-op relations from the jaxpr.
    
    Args:
        jaxpr: The jaxpr.
        hidden_outvar_to_invar: The mapping from the hidden outvar to the hidden invar.
        weight_path_to_invars: The mapping from the weight path to the jax vars.
        invar_to_weight_path: The mapping from the jax var to the weight path.
        path_to_state: The mapping from the state path to the state.
        state_id_to_path: The mapping from the state id to the state path.
        weight_invars: The jax vars of the weights.
        hid_path_to_group: The mapping from the hidden path to the hidden group.
        invar_to_hidden_path: The mapping from the hidden invar to the hidden path.
        outvar_to_hidden_path: The mapping from the hidden outvar to the hidden path.
        
    Returns:
        The hidden-param-op relations.
    """
    return JaxprEvalForWeightOpHiddenRelation(
        jaxpr=jaxpr,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_path_to_invars=weight_path_to_invars,
        invar_to_weight_path=invar_to_weight_path,
        path_to_state=path_to_state,
        state_id_to_path=state_id_to_path,
        weight_invars=weight_invars,
        hid_path_to_group=hid_path_to_group,
        invar_to_hidden_path=invar_to_hidden_path,
        outvar_to_hidden_path=outvar_to_hidden_path,
    ).compile()


def find_hidden_param_op_relations_from_minfo(
    minfo: ModuleInfo,
    hid_path_to_group: Dict[Path, HiddenGroup],
) -> Sequence[HiddenParamOpRelation]:
    """
    Finding the hidden-param-op relations from the model information.

    Args:
        minfo: The model information.
        hid_path_to_group: The mapping from the hidden path to the hidden group.

    Returns:
        The hidden-param-op relations.
    """
    return find_hidden_param_op_relations_from_jaxpr(
        jaxpr=minfo.jaxpr,
        hidden_outvar_to_invar=minfo.hidden_outvar_to_invar,
        weight_path_to_invars=minfo.weight_path_to_invars,
        invar_to_weight_path=minfo.invar_to_weight_path,
        weight_invars=set(minfo.weight_invars),
        invar_to_hidden_path=minfo.invar_to_hidden_path,
        outvar_to_hidden_path=minfo.outvar_to_hidden_path,
        path_to_state=minfo.retrieved_model_states,
        state_id_to_path=minfo.state_id_to_path,
        hid_path_to_group=hid_path_to_group,
    )


def find_hidden_param_op_relations_from_module(
    model: brainstate.nn.Module,
    *model_args,
    **model_kwargs,
) -> Sequence[HiddenParamOpRelation]:
    """
    Finding the hidden-param-op relations from the model.

    Args:
        model: The model.
        model_args: The model arguments.
        model_kwargs: The model keyword arguments.

    Returns:
        The hidden-param-op relations.
    """
    minfo = extract_module_info(model, *model_args, **model_kwargs)
    hidden_groups, hid_path_to_group = find_hidden_groups_from_minfo(minfo)
    return find_hidden_param_op_relations_from_minfo(minfo, hid_path_to_group)
