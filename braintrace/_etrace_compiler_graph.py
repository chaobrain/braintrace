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


import threading
from contextlib import contextmanager
from typing import Dict, Sequence, Tuple, Optional, NamedTuple

import brainstate
import jax

from ._etrace_compiler_hid_param_op import (
    find_hidden_param_op_relations_from_minfo,
    HiddenParamOpRelation,
)
from ._etrace_compiler_hidden_group import (
    find_hidden_groups_from_minfo,
    HiddenGroup,
)
from ._etrace_compiler_hidden_pertubation import (
    add_hidden_perturbation_from_minfo,
    HiddenPerturbation,
)
from ._etrace_compiler_module_info import (
    extract_module_info,
    ModuleInfo,
)
from ._typing import (
    Inputs,
    Path,
)

__all__ = [
    'ETraceGraph',
    'compile_etrace_graph',
]


def order_hidden_group_index(
    hidden_groups: Sequence[HiddenGroup],
):
    """
    Verifies that hidden group indices match their positions in the sequence.

    This function ensures that the index attribute of each HiddenGroup in the sequence
    matches its position in the sequence. This validation is important for maintaining
    the correct ordering of hidden groups in the eligibility trace compilation process.

    Args:
        hidden_groups (Sequence[HiddenGroup]): A sequence of HiddenGroup objects to validate.

    Raises:
        AssertionError: If any hidden group's index doesn't match its position in the sequence.
    """
    for i, group in enumerate(hidden_groups):
        assert group.index == i, f"Hidden group index {group.index} should be equal to its position {i}."


class ETraceGraph(NamedTuple):
    """
    The overall compiled graph for the eligibility trace.

    The eligibility trace graph, tracking the relationship between the etrace weights
    :py:class:`ETraceParam`, the etrace variables :py:class:`ETraceState`, and the etrace
    operations :py:class:`ETraceOp`.

    The following fields are included:

    - ``module_info``: The model information, instance of :class:`ModuleInfo`.
    - ``hidden_groups``: The hidden groups, sequence of :class:`HiddenGroup`.
    - ``hid_path_to_group``: The mapping from the hidden path to the hidden group :class:`HiddenGroup`.
    - ``hidden_param_op_relations``: The hidden parameter operation relations, sequence of :class:`HiddenParamOpRelation`.
    - ``hidden_perturb``: The hidden perturbation, instance of :class:`HiddenPerturbation`, or None.

    Example::

        >>> import braintrace
        >>> import brainstate
        >>> gru = braintrace.nn.GRUCell(10, 20)
        >>> gru.init_state()
        >>> inputs = brainstate.random.randn(10)
        >>> compiled_graph = braintrace.compile_etrace_graph(gru, inputs)
        >>> compiled_graph.dict().keys()

    """

    module_info: ModuleInfo
    hidden_groups: Sequence[HiddenGroup]
    hid_path_to_group: Dict[Path, HiddenGroup]
    hidden_param_op_relations: Sequence[HiddenParamOpRelation]
    hidden_perturb: HiddenPerturbation | None

    def call_hidden_perturb(
        self,
        args: Inputs,
        perturb_data: Sequence[jax.Array],
        old_state_vals: Optional[Sequence[jax.Array]] = None,
    ):
        # state checking
        if old_state_vals is None:
            old_state_vals = [st.value for st in self.module_info.compiled_model_states]

        # calling the function
        jaxpr_outs = self.hidden_perturb.eval_jaxpr(
            jax.tree.leaves((args, old_state_vals)),
            perturb_data,
        )

        return self.module_info._process(*args, jaxpr_outs=jaxpr_outs)

    def dict(self) -> Dict:
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


ETraceGraph.__module__ = 'braintrace'


class CONTEXT(threading.local):
    """
    The context for the eligibility trace compiler.

    The context is a thread-local object, which is used to store the compiled graph
    for the eligibility trace.
    """

    def __init__(self):
        self.compilers = []

    def add_compiler(self, name: str):
        self.compilers.append(name)


context = CONTEXT()


@contextmanager
def compiler_context(name: str):
    """
    Provides a context manager for managing the eligibility trace compiler context.

    This function manages the context for compiling eligibility trace graphs, ensuring
    that recursive graph compilations are detected and handled appropriately.

    Args:
        name (str): The name of the compiler to be added to the context.

    Yields:
        None: This context manager does not yield any value.

    Raises:
        NotImplementedError: If a recursive call to "compile_graph" is detected.
    """
    try:
        # add the compiler to the context
        context.add_compiler(name)

        # check if there is a recursive graph compilation
        if len(context.compilers) > 1:
            raise NotImplementedError(
                'Detected recursive call to "compile_graph". '
                'This is not supported currently.'
            )

        yield
    finally:
        context.compilers.pop()


def compile_etrace_graph(
    model: brainstate.nn.Module,
    *model_args: Tuple,
    include_hidden_perturb: bool = True,
) -> ETraceGraph:
    """
    Constructs the eligibility trace graph for a given model based on the provided inputs.

    This is the most important method for the eligibility trace graph. It builds the
    graph for the model, tracking the relationship between the etrace weights
    :py:class:`ETraceParam`, the etrace sattes :py:class:`ETraceState`, and the etrace
    operations :py:class:`ETraceOp`, which will be used for computing the weight
    spatial gradients, the hidden state Jacobian, and the hidden state-weight Jacobian.

    This function is crucial for building the eligibility trace graph, which tracks the
    relationships between eligibility trace weights, states, and operations. These relationships
    are used to compute weight spatial gradients, hidden state Jacobians, and hidden state-weight
    Jacobians.

    Args:
        model (brainstate.nn.Module): The model for which the eligibility trace graph is to be built.
        model_args (Tuple): The arguments required by the model.
        include_hidden_perturb (bool): Indicates whether to include hidden perturbations in the graph.
            Defaults to True.

    Returns:
        ETraceGraph: The compiled eligibility trace graph containing module information, hidden groups,
        hidden parameter operation relations, and optional hidden perturbations.
    """

    with compiler_context('compile_graph'):

        assert isinstance(model_args, tuple)
        minfo = extract_module_info(model, *model_args)

        # ---       evaluating the relationship for hidden-to-hidden        --- #
        hidden_groups, hid_path_to_group = find_hidden_groups_from_minfo(minfo)
        order_hidden_group_index(hidden_groups)

        # ---       evaluating the jaxpr for (hidden, param, op) relationships      --- #

        hidden_param_op_relations = find_hidden_param_op_relations_from_minfo(
            minfo=minfo,
            hid_path_to_group=hid_path_to_group,
        )

        # ---      Rewrite the jaxpr for computing the needed variables      --- #

        # Rewrite jaxpr to return all necessary variables, including
        #
        #   1. the original function outputs
        #   2. the hidden states
        #   3. the weight x   ===>  for computing the weight spatial gradients
        #   4. the y-to-hidden variables   ===>  for computing the weight spatial gradients
        #   5. the hidden-hidden transition variables   ===>  for computing the hidden-hidden jacobian
        #

        # all weight x
        out_wx_jaxvars = list(set([
            relation.x for relation in hidden_param_op_relations
            if relation.x is not None
        ]))

        # all y-to-hidden vars
        out_wy2hid_jaxvars = set()
        for relation in hidden_param_op_relations:
            for hpo_jaxpr in relation.y_to_hidden_group_jaxprs:
                out_wy2hid_jaxvars.update(hpo_jaxpr.invars + hpo_jaxpr.constvars)
        out_wy2hid_jaxvars = list(out_wy2hid_jaxvars)

        # hidden-hidden transition vars
        hid2hid_jaxvars = set()
        for group in hidden_groups:
            hid2hid_jaxvars.update(group.hidden_invars)
            hid2hid_jaxvars.update(group.transition_jaxpr_constvars)
        hid2hid_jaxvars = list(hid2hid_jaxvars)

        # all temporary outvars
        temp_outvars = set(
            minfo.jaxpr.outvars[minfo.num_var_out:] +  # all state variables
            out_wx_jaxvars +  # all weight x
            out_wy2hid_jaxvars +  # all y-to-hidden invars
            hid2hid_jaxvars  # all hidden-hidden transition vars
        ).difference(
            minfo.jaxpr.outvars  # exclude the original function outputs
        )

        # rewrite module_info
        minfo = minfo.add_jaxpr_outs(list(temp_outvars))

        # ---               add perturbations to the hidden states                  --- #
        # --- new jaxpr with hidden state perturbations for computing the residuals --- #

        hidden_perturb = add_hidden_perturbation_from_minfo(minfo) if include_hidden_perturb else None

        # ---              return the compiled graph               --- #

        return ETraceGraph(
            module_info=minfo,
            hidden_groups=hidden_groups,
            hid_path_to_group=hid_path_to_group,
            hidden_param_op_relations=hidden_param_op_relations,
            hidden_perturb=hidden_perturb,
        )
