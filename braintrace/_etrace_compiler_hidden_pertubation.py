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


from typing import Dict, Set, Sequence, NamedTuple, Any

import brainstate
import brainunit as u
import jax.core

from ._compatible_imports import (
    Var,
    JaxprEqn,
    Jaxpr,
    ClosedJaxpr,
    new_var
)
from ._etrace_compiler_base import (
    JaxprEvaluation,
)
from ._etrace_compiler_hidden_group import (
    HiddenGroup,
)
from ._etrace_compiler_module_info import (
    extract_module_info,
    ModuleInfo,
)
from ._misc import (
    git_issue_addr,
)
from ._typing import (
    HiddenInVar,
    HiddenOutVar,
    Path,
)

__all__ = [
    'HiddenPerturbation',
    'add_hidden_perturbation_from_minfo',
    'add_hidden_perturbation_in_module',
]


class HiddenPerturbation(NamedTuple):
    r"""
    The hidden perturbation information.

    Hidden perturbation means that we add perturbations to the hidden states in the jaxpr,
    and replace the hidden states with the perturbed states.
    
    Mathematically, we have the following equation:
    
    $$
    h^t = f(x)  \Rightarrow  h^t = f(x) + \text{perturb_var}
    $$
    
    where $h$ is the hidden state, $f$ is the function, $x$ is the input, and $\text{perturb_var}$
    is the perturbation variable.

    Technically, we first define a new variable $\hat{h}^t = f(x)$, and then add a new equation:

    $$
    h^t = \hat{h}^t + \text{perturb_var}
    $$

    Actually, we add the perturbation to the hidden states in the jaxpr for computing the hidden state gradients:

    $$
    \frac{\partial L^t}{\partial h^t} = \frac{\partial L^t}{\partial \text{perturb_var}}
    $$

    Example::

        >>> import braintrace
        >>> import brainstate
        >>> gru = braintrace.nn.GRUCell(10, 20)
        >>> gru.init_state()
        >>> inputs = brainstate.random.randn(10)
        >>> hidden_perturb = braintrace.add_hidden_perturbation_in_module(gru, inputs)


    """
    perturb_vars: Sequence[Var]  # the perturbation variables
    perturb_hidden_paths: Sequence[Path]  # the hidden state paths that are perturbed
    perturb_hidden_states: Sequence[brainstate.HiddenState]  # the hidden states that are perturbed
    perturb_jaxpr: ClosedJaxpr  # the perturbed jaxpr

    def eval_jaxpr(
        self,
        inputs: Sequence[jax.Array],
        perturb_data: Sequence[jax.Array]
    ) -> Sequence[jax.Array]:
        """
        Evaluate the perturbed jaxpr.
        """
        return jax.core.eval_jaxpr(
            self.perturb_jaxpr.jaxpr,
            self.perturb_jaxpr.consts,
            *(tuple(inputs) + tuple(perturb_data))
        )

    def init_perturb_data(self) -> Sequence[jax.Array]:
        """
        Initialize the perturbation data.
        """
        return [jax.numpy.zeros_like(v.aval) for v in self.perturb_vars]

    def perturb_data_to_hidden_group_data(
        self,
        perturb_data: Sequence[jax.Array],
        hidden_groups: Sequence[HiddenGroup],
    ) -> Sequence[jax.Array]:
        """
        Convert the perturbation data to the hidden group data.
        """
        assert len(perturb_data) == len(self.perturb_vars), (
            f'The length of the perturb data is not correct. '
            f'Expected: {len(self.perturb_vars)}, '
            f'Got: {len(perturb_data)}'
        )
        path_to_perturb_data = {
            path: data
            for path, data in zip(self.perturb_hidden_paths, perturb_data)
        }
        return [
            group.concat_hidden(
                [
                    # dimensionless processing
                    u.get_mantissa(path_to_perturb_data[path])
                    for path in group.hidden_paths
                ]
            )
            for group in hidden_groups
        ]

    def dict(self) -> Dict[str, Any]:
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


HiddenPerturbation.__module__ = 'braintrace'


class JaxprEvalForHiddenPerturbation(JaxprEvaluation):
    """
    Adding perturbations to the hidden states in the jaxpr, and replacing the hidden states with the perturbed states.

    Args:
        closed_jaxpr: The closed jaxpr for the model.
        outvar_to_hidden_path: The mapping from the outvar to the state id.
        hidden_outvar_to_invar: The mapping from the hidden output variable to the hidden input variable.
        weight_invars: The weight input variables.
        invar_to_hidden_path: The mapping from the weight input variable to the hidden state path.

    Returns:
        The revised closed jaxpr with the perturbations.

    """
    __module__ = 'braintrace'

    def __init__(
        self,
        closed_jaxpr: ClosedJaxpr,
        hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
        weight_invars: Set[Var],
        invar_to_hidden_path: Dict[HiddenInVar, Path],
        outvar_to_hidden_path: Dict[Var, Path],
        path_to_state: Dict[Path, brainstate.HiddenState],
    ):
        # necessary data structures
        self.closed_jaxpr = closed_jaxpr

        # initialize the super class
        super().__init__(
            weight_invars=weight_invars,
            hidden_invars=set(hidden_outvar_to_invar.values()),
            hidden_outvars=set(hidden_outvar_to_invar.keys()),
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path
        )

        self.path_to_state = path_to_state

    def compile(self) -> HiddenPerturbation:
        # new invars, the var order is the same as the hidden_outvars
        self.perturb_invars = {
            v: self._new_var_like(v)
            for v in self.hidden_outvars
        }

        # the hidden states that are not found in the code
        self.hidden_jaxvars_to_remove = set(self.hidden_outvars)

        # final revised equations
        self.revised_eqns = []

        # revising equations
        self._eval_jaxpr(self.closed_jaxpr.jaxpr)

        # [final checking]
        # If there are hidden states that are not found in the code, we raise an error.
        if len(self.hidden_jaxvars_to_remove) > 0:
            hid_paths = [self.outvar_to_hidden_path[v] for v in self.hidden_jaxvars_to_remove]
            hid_info = '\n'.join([f'{v} -> {path}' for v, path in zip(self.hidden_jaxvars_to_remove, hid_paths)])
            raise ValueError(
                f'Error: we did not found your defined hidden state '
                f'(see the following information) in the code. \n'
                f'Please report an issue to the developers at {git_issue_addr}. \n'
                f'The missed hidden states are: \n'
                f'{hid_info}'
            )

        # new jaxpr
        jaxpr = Jaxpr(
            constvars=list(self.closed_jaxpr.jaxpr.constvars),
            invars=list(self.closed_jaxpr.jaxpr.invars) + list(self.perturb_invars.values()),
            outvars=list(self.closed_jaxpr.jaxpr.outvars),
            eqns=self.revised_eqns
        )
        revised_closed_jaxpr = ClosedJaxpr(jaxpr, self.closed_jaxpr.literals)

        # finalizing
        perturb_hidden_paths = [self.outvar_to_hidden_path[v] for v in self.hidden_outvars]
        perturb_hidden_states = [self.path_to_state[self.outvar_to_hidden_path[v]] for v in self.hidden_outvars]
        info = HiddenPerturbation(
            perturb_vars=brainstate.util.PrettyList(self.perturb_invars.values()),
            perturb_hidden_paths=brainstate.util.PrettyList(perturb_hidden_paths),
            perturb_hidden_states=brainstate.util.PrettyList(perturb_hidden_states),
            perturb_jaxpr=revised_closed_jaxpr
        )

        # remove the temporal data
        self.perturb_invars = dict()
        self.revised_eqns = []
        self.hidden_jaxvars_to_remove = set()
        return info

    def _eval_pjit(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.
        """
        self._eval_eqn(eqn)

    def _add_perturb_eqn(
        self,
        eqn: JaxprEqn,
        perturb_var: Var
    ):
        # ------------------------------------------------
        #
        # For the hidden var eqn, we want to add a perturbation:
        #    y = f(x)  =>  y = f(x) + perturb_var
        #
        # Particularly, we first define a new variable
        #    new_outvar = f(x)
        #
        # Then, we add a new equation for the perturbation
        #    y = new_outvar + perturb_var
        #
        # ------------------------------------------------

        hidden_var = eqn.outvars[0]

        # Frist step, define the hidden var as a new variable
        new_outvar = self._new_var_like(perturb_var)
        old_eqn = eqn.replace(outvars=[new_outvar])
        self.revised_eqns.append(old_eqn)

        # Second step, add the perturbation equation
        new_eqn = jax.core.new_jaxpr_eqn(
            [new_outvar, perturb_var],
            [hidden_var],
            jax.lax.add_p,
            {},
            set(),
            eqn.source_info.replace()
        )
        self.revised_eqns.append(new_eqn)

    def _eval_eqn(self, eqn: JaxprEqn):
        if len(eqn.outvars) == 1:
            if eqn.outvars[0] in self.hidden_jaxvars_to_remove:
                hidden_var = eqn.outvars[0]
                self.hidden_jaxvars_to_remove.remove(hidden_var)
                self._add_perturb_eqn(eqn, self.perturb_invars[hidden_var])
                return
        self.revised_eqns.append(eqn.replace())

    def _new_var_like(self, v):
        return new_var('', jax.core.ShapedArray(v.aval.shape, v.aval.dtype))


def add_hidden_perturbation_in_jaxpr(
    closed_jaxpr: ClosedJaxpr,
    hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
    weight_invars: Set[Var],
    invar_to_hidden_path: Dict[HiddenInVar, Path],
    outvar_to_hidden_path: Dict[Var, Path],
    path_to_state: Dict[Path, brainstate.HiddenState],
) -> HiddenPerturbation:
    """
    Adding perturbations to the hidden states in the jaxpr, and replacing the hidden states with the perturbed states.

    Args:
        closed_jaxpr: The closed jaxpr for the model.
        outvar_to_hidden_path: The mapping from the outvar to the state id.
        hidden_outvar_to_invar: The mapping from the hidden output variable to the hidden input variable.
        weight_invars: The weight input variables.
        invar_to_hidden_path: The mapping from the weight input variable to the hidden state path.
        path_to_state: The mapping from the hidden state path to the state.

    Returns:
        The revised closed jaxpr with the perturbations.
    """
    return JaxprEvalForHiddenPerturbation(
        closed_jaxpr=closed_jaxpr,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_invars=weight_invars,
        invar_to_hidden_path=invar_to_hidden_path,
        outvar_to_hidden_path=outvar_to_hidden_path,
        path_to_state=path_to_state,
    ).compile()


def add_hidden_perturbation_from_minfo(
    minfo: ModuleInfo,
) -> HiddenPerturbation:
    """
    Adding perturbations to the hidden states in the module,
    and replacing the hidden states with the perturbed states.

    Args:
        minfo: The model information.

    Returns:
        The hidden perturbation information.
    """
    return add_hidden_perturbation_in_jaxpr(
        closed_jaxpr=minfo.closed_jaxpr,
        hidden_outvar_to_invar=minfo.hidden_outvar_to_invar,
        weight_invars=set(minfo.weight_invars),
        invar_to_hidden_path=minfo.invar_to_hidden_path,
        outvar_to_hidden_path=minfo.outvar_to_hidden_path,
        path_to_state=minfo.retrieved_model_states,
    )


def add_hidden_perturbation_in_module(
    model: brainstate.nn.Module,
    *model_args,
    **model_kwargs,
) -> HiddenPerturbation:
    """
    Adds perturbations to the hidden states in the given module and replaces the hidden states with the perturbed states.

    Parameters
    ----------
    model (brainstate.nn.Module): The neural network module to which hidden state perturbations will be added.
    *model_args: Additional positional arguments to be passed to the model.
    **model_kwargs: Additional keyword arguments to be passed to the model.

    Returns
    -------
    HiddenPerturbation: An object containing information about the perturbations added to the hidden states, including the perturbed variables, paths, states, and the revised jaxpr.
    """
    minfo = extract_module_info(model, *model_args, **model_kwargs)
    return add_hidden_perturbation_from_minfo(minfo)
