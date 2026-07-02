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

import functools
from typing import Dict, Sequence, List, Tuple, Optional, NamedTuple, Any

import brainstate
import jax
import brainunit as u

from braintrace._compatible_imports import (
    Var,
    Jaxpr,
    ClosedJaxpr,
)
from braintrace._misc import (
    NotSupportedError,
    unknown_state_path,
    _remove_quantity,
)
from braintrace._state_managment import sequence_split_state_values
from braintrace._typing import (
    Path,
    PyTree,
    StateID,
    Inputs,
    Outputs,
    ETraceVals,
    StateVals,
    TempData,
)
from .canonicalize import ControlFlowPolicy, DEFAULT_CONTROL_FLOW_POLICY, if_convert_conds
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit
from .jaxpr_graph import inline_jit_calls

__all__ = [
    'ModuleInfo',
    'extract_module_info',
]


def _model_that_not_allow_param_assign(model, *args_, **kwargs_):
    with brainstate.StateTraceStack() as trace:
        out = model(*args_, **kwargs_)

    for st, write in zip(trace.states, trace.been_writen):
        if isinstance(st, brainstate.ParamState) and write:
            raise NotSupportedError(
                f'The parameter state "{st}" is rewritten in the model. Currently, the '
                f'online learning method we provided does not support the dynamical '
                f'weight parameters. '
            )
    return out


def _check_consistent_states_between_model_and_compiler(
    compiled_model_states: Sequence[brainstate.State],
    retrieved_model_states: Dict[Path, brainstate.State],
    verbose: bool = True,  # whether to print the information
):
    id_to_compiled_state = {
        id(st): st
        for st in compiled_model_states
    }
    id_to_path = {
        id(st): path
        for path, st in retrieved_model_states.items()
    }

    paths_to_remove = []
    for id_ in id_to_path:
        if id_ not in id_to_compiled_state:
            path = id_to_path[id_]
            paths_to_remove.append(path)
            if verbose:
                emit(
                    kind=DiagnosticKind.STATE_MISMATCH,
                    level=DiagnosticLevel.WARNING,
                    message=f"The state {path} is not found in the compiled model.",
                    weight_path=path if isinstance(path, tuple) else None,
                    context={
                        'direction': 'retrieved_not_in_compiled',
                        'state_path': path,
                    },
                )
    for path in paths_to_remove:
        retrieved_model_states.pop(path)
    i_unknown = 0
    for id_ in id_to_compiled_state:
        if id_ not in id_to_path:
            st = id_to_compiled_state[id_]
            if verbose:
                emit(
                    kind=DiagnosticKind.STATE_MISMATCH,
                    level=DiagnosticLevel.WARNING,
                    message=(
                        f"The state {st} is not found in the retrieved model. "
                        f"We have added this state."
                    ),
                    context={
                        'direction': 'compiled_not_in_retrieved',
                        'state': st,
                    },
                )
            retrieved_model_states[unknown_state_path(i=i_unknown)] = st
            i_unknown += 1


def _check_in_out_consistent_units(
    state_tree_invars,
    state_tree_outvars,
    state_tree_path,
):
    assert len(state_tree_invars) == len(state_tree_outvars), 'The number of invars and outvars should be the same.'
    assert len(state_tree_invars) == len(state_tree_path), 'The number of invars and paths should be the same.'
    for invar, outvar, path in zip(state_tree_invars, state_tree_outvars, state_tree_path):
        in_leaves = jax.tree.leaves(invar, is_leaf=u.math.is_quantity)
        out_leaves = jax.tree.leaves(outvar, is_leaf=u.math.is_quantity)
        assert len(in_leaves) == len(out_leaves), 'The number of leaves should be the same.'
        for in_leaf, out_leaf in zip(in_leaves, out_leaves):
            if u.get_unit(in_leaf) != u.get_unit(out_leaf):
                raise ValueError(
                    f'The input/output unit of the state {path} does not match. \n'
                    f'Input unit: {u.get_unit(in_leaf)}\n'
                    f'Output unit: {u.get_unit(out_leaf)}\n'
                    f'We now only support the consistent unit between the input and output, '
                    f'since all our eligibility trace compilation is based on the unit consistency so that '
                    f'units can be omitted and data can be dimensionless processing. '
                )


def abstractify_model(
    model: brainstate.nn.Module,
    *model_args,
    **model_kwargs
):
    """
    Abstracts a model into a stateful representation suitable for compilation and state extraction.

    This function ensures that the model is an instance of `brainstate.nn.Module` and compiles it into a
    stateful function. It retrieves the model's states and checks for consistency between the 
    compiled states and the retrieved states.

    Args:
        model (brainstate.nn.Module): The model to be abstracted. It must be an instance of `brainstate.nn.Module`.
        *model_args: Positional arguments to be passed to the model during compilation.
        **model_kwargs: Keyword arguments to be passed to the model during compilation.

    Returns:
        Tuple[brainstate.transform.StatefulFunction, Dict[Path, brainstate.State]]:
            - A stateful function representing the compiled model.
            - A dictionary of the model's retrieved states with their paths.
    """
    assert isinstance(model, brainstate.nn.Module), (
        "The model should be an instance of brainstate.nn.Module. "
        "Since it allows the explicit definition of the model structure."
    )
    model_retrieved_states = brainstate.graph.states(model)

    # --- stateful model, for extracting states, weights, and variables --- #
    #
    # [ NOTE ]
    # The model does not support "static_argnums" for now.
    # Please always use ``functools.partial`` to fix the static arguments.
    #
    # wrap the model so that we can track the iteration number
    stateful_model = brainstate.transform.StatefulFunction(
        functools.partial(_model_that_not_allow_param_assign, model),
        return_only_write=False
    )

    # -- compile the model -- #
    #
    # NOTE:
    # The model does not support "static_argnums" for now.
    # Please always use functools.partial to fix the static arguments.
    #
    stateful_model.make_jaxpr(*model_args, **model_kwargs)

    # -- states -- #
    compiled_states = stateful_model.get_states(*model_args, **model_kwargs, compile_if_miss=True)

    # check the consistency between the model and the compiler
    _check_consistent_states_between_model_and_compiler(
        compiled_states,
        model_retrieved_states
    )

    return stateful_model, model_retrieved_states


class ModuleInfo(NamedTuple):
    """The model information for the ETrace compiler.

    Bundles the abstract representation of a model and all the lookup tables the
    compiler needs. It groups information into five categories: the stateful
    model, the jaxpr, the states, the hidden states, and the parameter weights.

    Attributes
    ----------
    stateful_model : brainstate.transform.StatefulFunction
        The stateful function that compiles the model into an abstract jaxpr
        representation.
    closed_jaxpr : ClosedJaxpr
        The closed-jaxpr representation of the model.
    retrieved_model_states : brainstate.util.FlattedDict
        The model states retrieved from ``model.states()``, with well-defined
        paths and structures.
    compiled_model_states : sequence of brainstate.State
        The model states compiled from the stateful model; accurate and
        consistent with the model jaxpr but lacking path information.
    state_id_to_path : dict
        Mapping from each state id to its state path.
    state_tree_invars : PyTree of Var
        The input jaxpr variables of the states, as a pytree.
    state_tree_outvars : PyTree of Var
        The output jaxpr variables of the states, as a pytree.
    hidden_path_to_invar : dict
        Mapping from each hidden path to its input variable.
    hidden_path_to_outvar : dict
        Mapping from each hidden path to its output variable.
    invar_to_hidden_path : dict
        Mapping from each input variable to its hidden path.
    outvar_to_hidden_path : dict
        Mapping from each output variable to its hidden path.
    hidden_outvar_to_invar : dict
        Mapping from each output variable to its input variable.
    weight_invars : list of Var
        The weight input variables.
    weight_path_to_invars : dict
        Mapping from each weight path to its input variables.
    invar_to_weight_path : dict
        Mapping from each input variable to its weight path.
    num_var_out : int
        Number of original output variables.
    num_var_state : int
        Number of state-variable outputs.

    See Also
    --------
    extract_module_info : Build a ``ModuleInfo`` from a model.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> module_info = braintrace.extract_module_info(gru, inputs)
        >>> isinstance(module_info, braintrace.ModuleInfo)
        True
    """
    # stateful model
    stateful_model: brainstate.transform.StatefulFunction

    # jaxpr
    closed_jaxpr: ClosedJaxpr

    # states
    retrieved_model_states: brainstate.util.FlattedDict
    compiled_model_states: Sequence[brainstate.State]
    state_id_to_path: Dict[StateID, Path]
    state_tree_invars: PyTree
    state_tree_outvars: PyTree

    # hidden states
    hidden_path_to_invar: Dict[Path, Var]
    hidden_path_to_outvar: Dict[Path, Var]
    invar_to_hidden_path: Dict[Var, Path]
    outvar_to_hidden_path: Dict[Var, Path]
    hidden_outvar_to_invar: Dict[Var, Var]

    # parameter weights
    weight_invars: List[Var]
    weight_path_to_invars: Dict[Path, List[Var]]
    invar_to_weight_path: Dict[Var, Path]

    # output
    num_var_out: int  # number of original output variables
    num_var_state: int  # number of state variable outputs

    @property
    def jaxpr(self) -> Jaxpr:
        """The jaxpr of the model.

        Returns
        -------
        Jaxpr
            The jaxpr extracted from ``closed_jaxpr``.
        """
        return self.closed_jaxpr.jaxpr

    def add_jaxpr_outs(
        self,
        jax_vars: Sequence[Var],
    ) -> 'ModuleInfo':
        """Add extra jaxpr outputs to the model jaxpr.

        Returns a new ``ModuleInfo`` whose jaxpr additionally outputs the given
        variables, so the compiler can recover the intermediate values it needs.

        Parameters
        ----------
        jax_vars : sequence of Var
            The extra jaxpr variables to append to the jaxpr outputs.

        Returns
        -------
        ModuleInfo
            A new ``ModuleInfo`` with the extended jaxpr.
        """
        assert all(isinstance(v, Var) for v in jax_vars), 'The jax_vars should be the instance of Var.'

        # jaxpr
        jaxpr = Jaxpr(
            constvars=list(self.jaxpr.constvars),
            invars=list(self.jaxpr.invars),
            outvars=list(self.jaxpr.outvars) + list(jax_vars),
            eqns=list(self.jaxpr.eqns),
            effects=self.jaxpr.effects,
            debug_info=self.jaxpr.debug_info,
        )

        # closed jaxpr
        closed_jaxpr = ClosedJaxpr(
            jaxpr=jaxpr,
            consts=self.closed_jaxpr.consts,
        )

        # new instance of `ModuleInfo`
        items = self.dict()
        items['closed_jaxpr'] = closed_jaxpr
        return ModuleInfo(**items)

    def split_state_outvars(self):
        """Split the state outvars into weight, hidden, and other states.

        Returns
        -------
        weight_jaxvar_tree : PyTree of Var
            The weight tree of jaxpr variables.
        hidden_jaxvar : PyTree of Var
            The hidden tree of jaxpr variables.
        other_state_jaxvar_tree : PyTree of Var
            The other-state tree of jaxpr variables.
        """
        (
            weight_jaxvar_tree,
            hidden_jaxvar,
            other_state_jaxvar_tree
        ) = sequence_split_state_values(self.compiled_model_states, self.state_tree_outvars)
        return weight_jaxvar_tree, hidden_jaxvar, other_state_jaxvar_tree

    def jaxpr_call(
        self,
        *args: Inputs,
        old_state_vals: Optional[Sequence[jax.Array]] = None,
    ) -> Tuple[
        Outputs,
        ETraceVals,
        StateVals,
        TempData,
    ]:
        """Evaluate the model on the given inputs using the compiled jaxpr.

        Parameters
        ----------
        *args : Inputs
            The inputs of the model.
        old_state_vals : sequence of jax.Array or None, optional
            The old state values. When ``None``, the current values of the
            compiled model states are used. Default ``None``.

        Returns
        -------
        out : Outputs
            The output of the model.
        etrace_vals : ETraceVals
            The values for the eligibility-trace (hidden) states.
        oth_state_vals : StateVals
            The other state values.
        temps : TempData
            The temporary intermediate values.
        """

        # state checking
        if old_state_vals is None:
            old_state_vals = [st.value for st in self.compiled_model_states]

        # calling the function
        jaxpr_outs = jax.core.eval_jaxpr(
            self.closed_jaxpr.jaxpr,
            self.closed_jaxpr.consts,
            *jax.tree.leaves((args, old_state_vals))
        )

        return self._process(*args, jaxpr_outs=jaxpr_outs)

    def _process(self, *args, jaxpr_outs: Sequence[jax.Array]):

        # intermediate values contain three parts:
        #
        # 1. "jaxpr_outs[:self.num_out]" corresponds to model original outputs
        #     - Outputs
        # 2. "jaxpr_outs[self.num_out:]" corresponds to extra output in  "augmented_jaxpr"
        #     - others
        temps = {
            v: r for v, r in
            zip(
                self.jaxpr.outvars[self.num_var_out:],
                jaxpr_outs[self.num_var_out:]
            )
        }
        # 3. "etrace state" old values
        for st, val in zip(self.compiled_model_states, self.state_tree_invars):
            if isinstance(st, brainstate.HiddenState):
                temps[val] = u.get_mantissa(st.value)

        #
        # recovery outputs of ``stateful_model``
        #
        cache_key = self.stateful_model.get_arg_cache_key(*args, compile_if_miss=True)
        i_start = self.num_var_out
        i_end = i_start + self.num_var_state
        # brainstate types the cached treedef as the broad ``PyTree``; at runtime it is a
        # ``jax`` ``PyTreeDef`` that exposes ``unflatten``.
        out, new_state_vals = self.stateful_model.get_out_treedef_by_cache(cache_key).unflatten(  # type: ignore[attr-defined]
            jaxpr_outs[:i_end])

        #
        # check state value
        assert len(self.compiled_model_states) == len(new_state_vals), 'State length mismatch.'

        #
        # split the state values
        #
        etrace_vals = dict()
        oth_state_vals = dict()
        for st, st_val in zip(self.compiled_model_states, new_state_vals):
            if isinstance(st, brainstate.HiddenState):
                etrace_vals[self.state_id_to_path[id(st)]] = st_val
            elif isinstance(st, brainstate.ParamState):
                # assume they are not changed
                pass
            else:
                oth_state_vals[self.state_id_to_path[id(st)]] = st_val

        return out, etrace_vals, oth_state_vals, temps

    def dict(self) -> Dict[str, Any]:
        """Return this module info's named fields as a plain dictionary.

        Returns
        -------
        dict
            An ordered mapping from field name to value, as produced by the
            underlying :class:`typing.NamedTuple`.
        """
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


ModuleInfo.__module__ = 'braintrace'


def extract_module_info(
    model: brainstate.nn.Module,
    *model_args,
    control_flow: Optional[ControlFlowPolicy] = None,
    **model_kwargs
) -> ModuleInfo:
    """Extract the model information for the ETrace compiler.

    Parameters
    ----------
    model : brainstate.nn.Module
        The model from which to extract the information.
    *model_args
        The positional arguments of the model.
    control_flow : ControlFlowPolicy or None, optional
        Policy governing control-flow canonicalization (currently ``cond``
        if-conversion; see
        :func:`~braintrace._compiler.canonicalize.if_convert_conds`).
        ``None`` (default) uses the default policy, which converts every
        ETP-relevant ``cond``.
    **model_kwargs
        The keyword arguments of the model.

    Returns
    -------
    ModuleInfo
        The model information.

    See Also
    --------
    ModuleInfo : The returned data structure.

    Notes
    -----
    Prefer positional arguments. ``**model_kwargs`` is accepted here for
    tracing, but ``ModuleInfo.jaxpr_call`` and the downstream
    ``compile_etrace_graph`` pipeline rebuild inputs from positional
    arguments only — bind static keyword arguments with
    ``functools.partial`` before compiling.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> module_info = braintrace.extract_module_info(gru, inputs)
        >>> module_info.num_var_out
        1
    """

    # abstract the model
    (
        stateful_model,
        model_retrieved_states
    ) = abstractify_model(
        model,
        *model_args,
        **model_kwargs
    )

    # state information
    cache_key = stateful_model.get_arg_cache_key(*model_args, **model_kwargs)
    compiled_states = stateful_model.get_states_by_cache(cache_key)
    compiled_states = brainstate.util.PrettyList(compiled_states)

    state_id_to_path: Dict[StateID, Path] = {
        id(state): path
        for path, state in model_retrieved_states.items()
    }
    state_id_to_path = brainstate.util.PrettyDict(state_id_to_path)

    closed_jaxpr = stateful_model.get_jaxpr_by_cache(cache_key)
    # Splice user ``jax.jit`` bodies into the top-level jaxpr before any
    # lookup table is built: every downstream analysis (weight/hidden var
    # tables, hidden-group discovery, relation finding) identifies states
    # and ETP primitives by ``Var`` identity in this one flat jaxpr, and a
    # jit call boundary hides its body behind fresh inner variables.
    closed_jaxpr = inline_jit_calls(closed_jaxpr)
    jaxpr = closed_jaxpr.jaxpr

    # out information
    out_shapes = stateful_model.get_out_shapes_by_cache(cache_key)[0]
    state_vals = [state.value for state in compiled_states]
    in_avals, _ = jax.tree.flatten((model_args, model_kwargs))
    out_avals, _ = jax.tree.flatten(out_shapes)
    num_in = len(in_avals)
    num_out = len(out_avals)
    state_avals, state_tree = jax.tree.flatten(state_vals)
    state_tree_invars = jax.tree.unflatten(state_tree, jaxpr.invars[num_in:])
    state_tree_outvars = jax.tree.unflatten(state_tree, jaxpr.outvars[num_out:])

    # check the consistency between the invars and outvars
    state_tree_path = [state_id_to_path[id(st)] for st in compiled_states]
    _check_in_out_consistent_units(
        state_tree_invars,
        state_tree_outvars,
        state_tree_path,
    )

    # remove the quantity from the invars and outvars
    state_tree_invars = _remove_quantity(state_tree_invars)
    state_tree_outvars = _remove_quantity(state_tree_outvars)
    state_tree_invars = brainstate.util.PrettyList(state_tree_invars)
    state_tree_outvars = brainstate.util.PrettyList(state_tree_outvars)

    # -- checking weights as invar -- #
    # Map ALL ParamState (not just ParamState) so primitive-based
    # ETP scanning can find weights used with etp_mm_p / etp_mv_p / etc.
    weight_path_to_invars = {
        state_id_to_path[id(st)]: jax.tree.leaves(invar)
        for invar, st in zip(state_tree_invars, compiled_states)
        if isinstance(st, brainstate.ParamState)
    }
    weight_path_to_invars = brainstate.util.PrettyDict(weight_path_to_invars)

    hidden_path_to_invar = {  # one-to-many mapping
        state_id_to_path[id(st)]: invar  # ETraceState only contains one Array, "invar" is the jaxpr var
        for invar, st in zip(state_tree_invars, compiled_states)
        if isinstance(st, brainstate.HiddenState)
    }
    hidden_path_to_invar = brainstate.util.PrettyDict(hidden_path_to_invar)

    invar_to_hidden_path = {
        invar: path
        for path, invar in hidden_path_to_invar.items()
    }
    invar_to_hidden_path = brainstate.util.PrettyDict(invar_to_hidden_path)

    invar_to_weight_path = {  # many-to-one mapping
        v: k
        for k, vs in weight_path_to_invars.items()
        for v in vs
    }
    invar_to_weight_path = brainstate.util.PrettyDict(invar_to_weight_path)

    # -- checking states as outvar -- #
    hidden_path_to_outvar = {  # one-to-one mapping
        state_id_to_path[id(st)]: outvar  # ETraceState only contains one Array, "outvar" is the jaxpr var
        for outvar, st in zip(state_tree_outvars, compiled_states)
        if isinstance(st, brainstate.HiddenState)
    }
    hidden_path_to_outvar = brainstate.util.PrettyDict(hidden_path_to_outvar)

    outvar_to_hidden_path = {  # one-to-one mapping
        v: state_id
        for state_id, v in hidden_path_to_outvar.items()
    }
    outvar_to_hidden_path = brainstate.util.PrettyDict(outvar_to_hidden_path)

    hidden_outvar_to_invar = {
        outvar: hidden_path_to_invar[hid]
        for hid, outvar in hidden_path_to_outvar.items()
    }
    hidden_outvar_to_invar = brainstate.util.PrettyDict(hidden_outvar_to_invar)

    weight_invars = brainstate.util.PrettyList(dict.fromkeys(v for vs in weight_path_to_invars.values() for v in vs))

    # Canonicalize ETP-relevant control flow (Phase 1: cond -> select_n).
    # The pass reuses each converted equation's original outvars and never
    # touches the jaxpr's invars/outvars, so every Var-identity table built
    # above stays valid; only the equation list (and consts) change.
    closed_jaxpr = if_convert_conds(
        closed_jaxpr,
        weight_invars=set(weight_invars),
        hidden_invars=set(hidden_path_to_invar.values()),
        hidden_outvars=set(hidden_path_to_outvar.values()),
        policy=control_flow if control_flow is not None else DEFAULT_CONTROL_FLOW_POLICY,
    )

    return ModuleInfo(
        # stateful model
        stateful_model=stateful_model,

        # jaxpr
        closed_jaxpr=closed_jaxpr,

        # states
        retrieved_model_states=model_retrieved_states,
        compiled_model_states=compiled_states,
        state_id_to_path=state_id_to_path,
        state_tree_invars=state_tree_invars,
        state_tree_outvars=state_tree_outvars,

        # hidden states
        hidden_path_to_invar=hidden_path_to_invar,
        invar_to_hidden_path=invar_to_hidden_path,
        hidden_path_to_outvar=hidden_path_to_outvar,
        outvar_to_hidden_path=outvar_to_hidden_path,
        hidden_outvar_to_invar=hidden_outvar_to_invar,

        # parameter weights
        weight_invars=weight_invars,
        weight_path_to_invars=weight_path_to_invars,
        invar_to_weight_path=invar_to_weight_path,

        # output parameters
        num_var_out=num_out,
        num_var_state=len(jaxpr.outvars[num_out:]),
    )
