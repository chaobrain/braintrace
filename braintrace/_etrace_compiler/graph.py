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

from braintrace._typing import (
    Inputs,
    Path,
)
from .diagnostics import (
    CompilationRecord,
    DiagnosticKind,
    diagnostic_context,
)
from .hid_param_op import (
    find_hidden_param_op_relations_from_minfo,
    HiddenParamOpRelation,
)
from .hidden_group import (
    find_hidden_groups_from_minfo,
    HiddenGroup,
)
from .hidden_pertubation import (
    add_hidden_perturbation_from_minfo,
    HiddenPerturbation,
)
from .module_info import (
    extract_module_info,
    ModuleInfo,
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
    """The overall compiled graph for the eligibility trace.

    Tracks the relationship between the eligibility-trace weights
    (``ParamState``), the eligibility-trace variables (``HiddenState``), and
    the eligibility-trace operations (ETP primitives). It is the object
    returned by :func:`compile_etrace_graph` and consumed by the online-learning
    algorithms.

    Attributes
    ----------
    module_info : ModuleInfo
        The model information.
    hidden_groups : sequence of HiddenGroup
        The hidden groups.
    hid_path_to_group : dict
        Mapping from each hidden-state path to its :class:`HiddenGroup`.
    hidden_param_op_relations : sequence of HiddenParamOpRelation
        The hidden parameter-operation relations.
    hidden_perturb : HiddenPerturbation or None
        The hidden perturbation, or ``None`` when perturbations are excluded.
    diagnostics : tuple of CompilationRecord
        The structured compilation records emitted while building the graph.

    See Also
    --------
    compile_etrace_graph : Build an ``ETraceGraph`` from a model.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> graph = braintrace.compile_etrace_graph(gru, inputs)
        >>> isinstance(graph, braintrace.ETraceGraph)
        True
    """

    module_info: ModuleInfo
    hidden_groups: Sequence[HiddenGroup]
    hid_path_to_group: Dict[Path, HiddenGroup]
    hidden_param_op_relations: Sequence[HiddenParamOpRelation]
    hidden_perturb: HiddenPerturbation | None
    diagnostics: Tuple[CompilationRecord, ...] = ()

    def explain(
        self,
        *,
        weight_path: Optional[Path] = None,
        hidden_path: Optional[Path] = None,
        kind: Optional[DiagnosticKind] = None,
    ) -> Tuple[CompilationRecord, ...]:
        """Return compilation records filtered by weight path, hidden path, or kind.

        ``weight_path`` and ``hidden_path`` match the record's ``weight_path``
        exactly and ``hidden_paths`` membership respectively. ``kind`` matches
        ``CompilationRecord.kind``. All filters are optional; with no filters
        the full diagnostic log is returned.

        Parameters
        ----------
        weight_path : Path or None, optional
            If given, keep only records whose ``weight_path`` equals this
            value. Default ``None``.
        hidden_path : Path or None, optional
            If given, keep only records whose ``hidden_paths`` contain this
            value. Default ``None``.
        kind : DiagnosticKind or None, optional
            If given, keep only records whose ``kind`` is this value. Default
            ``None``.

        Returns
        -------
        tuple of CompilationRecord
            The matching records, in emission order.
        """
        result = []
        for record in self.diagnostics:
            if weight_path is not None and record.weight_path != weight_path:
                continue
            if hidden_path is not None and hidden_path not in record.hidden_paths:
                continue
            if kind is not None and record.kind is not kind:
                continue
            result.append(record)
        return tuple(result)

    def call_hidden_perturb(
        self,
        args: Inputs,
        perturb_data: Sequence[jax.Array],
        old_state_vals: Optional[Sequence[jax.Array]] = None,
    ):
        r"""Run the forward pass with additive perturbations injected at the hidden states.

        Evaluates the perturbed-forward jaxpr built during compilation, which is
        the forward computation augmented so that each tracked hidden state has a
        perturbation term added to it. This is the primitive used to probe
        hidden->hidden and hidden->output sensitivities.

        Parameters
        ----------
        args : Inputs
            The model inputs for this step, matching the signature captured at
            compile time.
        perturb_data : Sequence[jax.Array]
            One perturbation array per tracked hidden state, added at the
            corresponding perturbation site.
        old_state_vals : Sequence[jax.Array] or None, optional
            The state values to run from. When ``None`` (default) the current
            values of the compiled model states are used.

        Returns
        -------
        object
            The processed model outputs, in the same structure produced by a
            normal forward call.
        """
        # state checking
        if old_state_vals is None:
            old_state_vals = [st.value for st in self.module_info.compiled_model_states]

        # calling the function
        assert self.hidden_perturb is not None
        jaxpr_outs = self.hidden_perturb.eval_jaxpr(
            jax.tree.leaves((args, old_state_vals)),
            perturb_data,
        )

        return self.module_info._process(*args, jaxpr_outs=jaxpr_outs)

    def dict(self) -> Dict:
        """Return the graph's fields as a plain dictionary.

        Returns
        -------
        dict
            A mapping from field name to value for every attribute of this
            :class:`ETraceGraph`.
        """
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
    include_recurrent_mixing: bool = False,
) -> ETraceGraph:
    """Construct the eligibility-trace graph for a given model and inputs.

    This is the primary entry point of the ETrace compiler. It builds the graph
    for the model, tracking the relationship between the eligibility-trace
    weights (``ParamState``), the eligibility-trace states (``HiddenState``),
    and the eligibility-trace operations (ETP primitives). These relationships
    are later used to compute the weight spatial gradients, the hidden-state
    Jacobian, and the hidden-state-to-weight Jacobian.

    Parameters
    ----------
    model : brainstate.nn.Module
        The model for which the eligibility-trace graph is built.
    *model_args : tuple
        The positional arguments required by the model.
    include_hidden_perturb : bool, optional
        Whether to include hidden perturbations in the graph. Default ``True``.
    include_recurrent_mixing : bool, optional
        Hidden-group grouping mode for the hidden-to-hidden transition. When
        ``False`` (default, "without recurrence"), recurrent ETP mixing
        primitives (e.g. the recurrent ``etp_mv``/``etp_mm``) are treated as
        boundaries and excluded from the transition jaxpr, so the transition is
        element-wise and the per-position recurrent Jacobian is diagonal (the
        bounded D-RTRL / e-prop approximation). When ``True`` ("with
        recurrence"), those primitives are traced into the transition, the
        recurrence becomes coupled, and the true per-position block-diagonal
        Jacobian is extracted (RTRL-exact temporal credit, e.g. for
        :class:`~braintrace.OSTLRecurrent` / :class:`~braintrace.OSTTP`).

    Returns
    -------
    ETraceGraph
        The compiled eligibility-trace graph containing module information,
        hidden groups, hidden parameter-operation relations, and optional
        hidden perturbations.

    Raises
    ------
    NotImplementedError
        If a recursive call to the compiler is detected.

    See Also
    --------
    ETraceGraph : The returned compiled-graph data structure.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> graph = braintrace.compile_etrace_graph(gru, inputs)
        >>> len(graph.hidden_groups)
        1
    """

    with compiler_context('compile_graph'), diagnostic_context() as reporter:

        assert isinstance(model_args, tuple)
        minfo = extract_module_info(model, *model_args)

        # ---       evaluating the relationship for hidden-to-hidden        --- #
        hidden_groups, hid_path_to_group = find_hidden_groups_from_minfo(
            minfo, include_recurrent_mixing=include_recurrent_mixing
        )
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

        # all weight x (deduplicate while preserving insertion order)
        out_wx_jaxvars = list(dict.fromkeys(
            relation.x_var for relation in hidden_param_op_relations
            if relation.x_var is not None
        ))

        # all y-to-hidden vars (deduplicate while preserving insertion order)
        out_wy2hid_jaxvars_dict: dict = dict()
        for relation in hidden_param_op_relations:
            for hpo_jaxpr in relation.y_to_hidden_group_jaxprs:
                for v in hpo_jaxpr.invars + hpo_jaxpr.constvars:
                    out_wy2hid_jaxvars_dict[v] = None
        out_wy2hid_jaxvars = list(out_wy2hid_jaxvars_dict)

        # hidden-hidden transition vars (deduplicate while preserving insertion order)
        hid2hid_jaxvars_dict: dict = dict()
        for group in hidden_groups:
            for v in group.hidden_invars:
                hid2hid_jaxvars_dict[v] = None
            for v in group.transition_jaxpr_constvars:
                hid2hid_jaxvars_dict[v] = None
        hid2hid_jaxvars = list(hid2hid_jaxvars_dict)

        # all temporary outvars (deduplicate while preserving insertion order, exclude original outputs)
        original_outvars = set(minfo.jaxpr.outvars)
        all_vars = (
            minfo.jaxpr.outvars[minfo.num_var_out:] +  # all state variables
            out_wx_jaxvars +  # all weight x
            out_wy2hid_jaxvars +  # all y-to-hidden invars
            hid2hid_jaxvars  # all hidden-hidden transition vars
        )
        temp_outvars = list(dict.fromkeys(
            v for v in all_vars if v not in original_outvars
        ))

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
            diagnostics=reporter.records(),
        )
