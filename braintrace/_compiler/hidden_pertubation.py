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
import jax.core
import brainunit as u

from braintrace._compatible_imports import (
    Var,
    JaxprEqn,
    Jaxpr,
    ClosedJaxpr,
    new_var
)
from braintrace._misc import (
    git_issue_addr,
)
from braintrace._typing import (
    HiddenInVar,
    HiddenOutVar,
    Path,
)
from .base import (
    JaxprEvaluation,
)
from .hidden_group import (
    HiddenGroup,
)
from .module_info import (
    extract_module_info,
    ModuleInfo,
)

__all__ = [
    'HiddenPerturbation',
    'add_hidden_perturbation_from_minfo',
    'add_hidden_perturbation_in_module',
]


class HiddenPerturbation(NamedTuple):
    r"""The hidden-perturbation information.

    Hidden perturbation adds a perturbation variable to each hidden state in the
    jaxpr and replaces the hidden states with the perturbed states:

    .. math::

        h^t = f(x) \;\Rightarrow\; h^t = f(x) + \mathrm{perturb\_var},

    where :math:`h` is the hidden state, :math:`f` is the function, :math:`x` is
    the input, and :math:`\mathrm{perturb\_var}` is the perturbation variable.

    Attributes
    ----------
    perturb_vars : sequence of Var
        The perturbation variables.
    perturb_hidden_paths : sequence of Path
        The hidden-state paths that are perturbed.
    perturb_hidden_states : sequence of brainstate.HiddenState
        The hidden states that are perturbed.
    perturb_jaxpr : ClosedJaxpr
        The perturbed jaxpr.

    See Also
    --------
    add_hidden_perturbation_in_module : Build perturbations directly from a model.

    Notes
    -----
    Internally a new variable :math:`\hat{h}^t = f(x)` is defined and an extra
    equation :math:`h^t = \hat{h}^t + \mathrm{perturb\_var}` is added. The
    perturbation lets the hidden-state gradient be read off the perturbation
    variable

    .. math::

        \frac{\partial L^t}{\partial h^t}
        = \frac{\partial L^t}{\partial \mathrm{perturb\_var}}.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> hidden_perturb = braintrace.add_hidden_perturbation_in_module(gru, inputs)
        >>> isinstance(hidden_perturb, braintrace.HiddenPerturbation)
        True
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
        """Evaluate the perturbed jaxpr.

        Parameters
        ----------
        inputs : sequence of jax.Array
            The flat input values of the original jaxpr.
        perturb_data : sequence of jax.Array
            The perturbation values, one per entry of ``perturb_vars``.

        Returns
        -------
        sequence of jax.Array
            The outputs of the perturbed jaxpr.
        """
        return jax.core.eval_jaxpr(
            self.perturb_jaxpr.jaxpr,
            self.perturb_jaxpr.consts,
            *(tuple(inputs) + tuple(perturb_data))
        )

    def init_perturb_data(self) -> Sequence[jax.Array]:
        """Initialize the perturbation data to zeros.

        Returns
        -------
        sequence of jax.Array
            One zero array per perturbation variable, matching its shape and
            dtype.
        """
        return [jax.numpy.zeros(v.aval.shape, dtype=v.aval.dtype) for v in self.perturb_vars]

    def perturb_data_to_hidden_group_data(
        self,
        perturb_data: Sequence[jax.Array],
        hidden_groups: Sequence[HiddenGroup],
    ) -> Sequence[jax.Array]:
        """Convert the perturbation data to per-hidden-group data.

        Parameters
        ----------
        perturb_data : sequence of jax.Array
            The perturbation values, one per entry of ``perturb_vars``.
        hidden_groups : sequence of HiddenGroup
            The hidden groups to map the perturbation data onto.

        Returns
        -------
        sequence of jax.Array
            One concatenated perturbation array per hidden group.

        Raises
        ------
        AssertionError
            If ``perturb_data`` does not have the same length as
            ``perturb_vars``.
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
        """Return this perturbation's named fields as a plain dictionary.

        Returns
        -------
        dict
            An ordered mapping from field name to value, as produced by the
            underlying :class:`typing.NamedTuple`.
        """
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
        # Use dict.fromkeys to deduplicate while preserving insertion order
        # (avoids non-deterministic set iteration for jaxpr variable ordering)
        hidden_outvars_ordered = list(dict.fromkeys(hidden_outvar_to_invar.keys()))
        hidden_invars_ordered = list(dict.fromkeys(hidden_outvar_to_invar.values()))
        super().__init__(
            weight_invars=weight_invars,
            hidden_invars=set(hidden_invars_ordered),
            hidden_outvars=set(hidden_outvars_ordered),
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path
        )
        # Keep an ordered version for deterministic iteration in compile()
        self._hidden_outvars_ordered = hidden_outvars_ordered

        self.path_to_state = path_to_state

    def compile(self) -> HiddenPerturbation:
        # new invars, the var order is the same as the hidden_outvars (use ordered list for determinism)
        self.perturb_invars = {
            v: self._new_var_like(v)
            for v in self._hidden_outvars_ordered
        }

        # the hidden states that are not found in the code
        self.hidden_jaxvars_to_remove = set(self.hidden_outvars)

        # final revised equations
        self.revised_eqns: list = []

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
        perturb_hidden_paths = [self.outvar_to_hidden_path[v] for v in self._hidden_outvars_ordered]
        perturb_hidden_states = [self.path_to_state[self.outvar_to_hidden_path[v]] for v in
                                 self._hidden_outvars_ordered]
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

        Note: Unlike the base class, the perturbation compiler must process ALL
        equations (including etrace ops without gradient) to maintain a valid jaxpr.
        Skipping equations would leave variable references unresolved.
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
    """Add hidden-state perturbations from a ``ModuleInfo``.

    Adds perturbations to the hidden states in the module jaxpr and replaces
    the hidden states with the perturbed states.

    Parameters
    ----------
    minfo : ModuleInfo
        The model information.

    Returns
    -------
    HiddenPerturbation
        The hidden-perturbation information.

    See Also
    --------
    add_hidden_perturbation_in_module : Equivalent helper starting from a model.
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
    """Add hidden-state perturbations from a model.

    Adds perturbations to the hidden states of the given module and replaces the
    hidden states with the perturbed states.

    Parameters
    ----------
    model : brainstate.nn.Module
        The neural-network module to which hidden-state perturbations are added.
    *model_args
        Additional positional arguments passed to the model.
    **model_kwargs
        Additional keyword arguments passed to the model.

    Returns
    -------
    HiddenPerturbation
        Information about the perturbations added to the hidden states,
        including the perturbed variables, paths, states, and the revised
        jaxpr.

    See Also
    --------
    add_hidden_perturbation_from_minfo : Equivalent helper starting from ``ModuleInfo``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> hidden_perturb = braintrace.add_hidden_perturbation_in_module(gru, inputs)
        >>> len(hidden_perturb.perturb_vars)
        1
    """
    minfo = extract_module_info(model, *model_args, **model_kwargs)
    return add_hidden_perturbation_from_minfo(minfo)
