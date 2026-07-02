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

from typing import Container, Dict, Sequence, Set, List

from braintrace._compatible_imports import (
    Var,
    JaxprEqn,
    is_jit_primitive,
    is_scan_primitive,
    is_while_primitive,
    is_cond_primitive,
)
from braintrace._op import (
    is_etp_primitive,
    is_etp_enable_gradient_primitive,
)
from braintrace._typing import Path
from .canonicalize import ControlFlowPolicy, DEFAULT_CONTROL_FLOW_POLICY
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit

__all__ = [
    'JaxprEvaluation',
]


def find_matched_vars(
    invars: Sequence[Var],
    invar_needed_in_oth_eqns: Container[Var]
) -> List[Var]:
    """
    Checking whether the invars are matched with the invar_needed_in_oth_eqns.

    Parameters
    ----------
    invars : Sequence[Var]
        The input variables of the equation.
    invar_needed_in_oth_eqns : Container[Var]
        The variables needed in the other equations (a set, or an
        insertion-ordered dict used as an ordered set).

    Returns
    -------
    List[Var]
        The list of matched variables.
    """
    matched = []
    for invar in invars:
        if isinstance(invar, Var) and invar in invar_needed_in_oth_eqns:
            matched.append(invar)
    return matched


def find_element_exist_in_the_set(
    elements: Sequence[Var],
    the_set: Set[Var]
) -> Var | None:
    """
    Checking whether the jaxpr vars contain the weight variables.

    Parameters
    ----------
    elements : Sequence[Var]
        The input variables of the equation.
    the_set : Set[Var]
        The set of the weight variables.

    Returns
    -------
    Var | None
        The first element found in the set, or None if no element is found.
    """
    for invar in elements:
        if isinstance(invar, Var) and invar in the_set:
            return invar
    return None


def check_unsupported_op(
    self,
    eqn: JaxprEqn,
    op_name: str
):
    """
    Checks for unsupported operations involving weight or hidden state variables in the given equation.

    This function verifies whether the specified JAX equation (`eqn`) uses weight or hidden state variables
    in a manner that is currently unsupported. If such usage is detected, a `NotImplementedError` is raised
    with a detailed message.

    Parameters
    ----------
    self : JaxprEvaluation
        The instance of the class containing this method. Its optional
        ``control_flow`` attribute (a :class:`ControlFlowPolicy`) governs
        the hidden-state branch; instances without the attribute behave as
        the default policy.
    eqn : JaxprEqn
        The JAX equation to be checked.
    op_name : str
        The name of the operation being checked (e.g., 'pjit', 'scan', 'while', 'cond').

    Raises
    ------
    NotImplementedError
        If the equation uses weight variables, or computes hidden state
        variables in an unsupported manner (``jit`` regions always; opaque
        control flow when ``control_flow.while_hidden == 'error'``).
    ValueError
        If ``control_flow.while_hidden`` is not ``'opaque-fwd'`` or
        ``'error'``.
    """
    policy = getattr(self, 'control_flow', DEFAULT_CONTROL_FLOW_POLICY)

    # checking whether the weight variables are used in the equation
    # Note: user ``jax.jit`` boundaries are inlined at extraction time
    # (see ``jaxpr_graph.inline_jit_calls``), so reaching this check means
    # a weight is used inside a genuinely opaque region (scan/while/cond).
    invar = find_element_exist_in_the_set(eqn.invars, self.weight_invars)
    if invar is not None:
        if op_name == 'while':
            # A while body has no fixed trip count, so there is no
            # per-iteration hoisting that could connect the weight to the
            # hidden states — always a hard error with its own kind.
            guidance = (
                'Weight state used inside a while loop; while bodies cannot '
                'participate in online learning (a data-dependent trip count '
                'admits no fixed per-iteration decomposition). Move the '
                'weight application outside the loop so the loop consumes '
                'only its result, or use a fixed-length scan/for_loop (which '
                'the compiler unrolls). Note that a hidden-producing while '
                'drops same-step reverse-mode credit through its inputs: a '
                'parameter whose only same-step path to the loss crosses the '
                'loop receives a zero learning signal (see the while-hidden '
                'limitation in the changelog).'
            )
            emit(
                kind=DiagnosticKind.WEIGHT_IN_WHILE,
                level=DiagnosticLevel.ERROR,
                message=guidance,
                context={'op_name': op_name, 'invar': invar},
            )
            raise NotImplementedError(
                f'{guidance} \n\n'
                f'The weight state variable is: {invar}. \n'
                f'The Jaxpr of the while function is: \n\n'
                f'{eqn} \n\n'
            )
        emit(
            kind=DiagnosticKind.WEIGHT_IN_CONTROL_FLOW,
            level=DiagnosticLevel.ERROR,
            message=(
                f'Weight state used inside a {op_name} region; the ETrace '
                f'compiler cannot trace through it and compilation fails.'
            ),
            context={'op_name': op_name, 'invar': invar},
        )
        raise NotImplementedError(
            f'Currently, we do not support the weight states are used within a {op_name} function. \n'
            f'Please remove your {op_name} on the intermediate steps. \n\n'
            f'The weight state variable is: {invar}. \n'
            f'The Jaxpr of the {op_name} function is: \n\n'
            f'{eqn} \n\n'
        )

    # checking whether the hidden variables are computed in the equation
    outvar = find_element_exist_in_the_set(eqn.outvars, self.hidden_outvars)
    if outvar is not None:
        if op_name in ('scan', 'while', 'cond'):
            if policy.while_hidden == 'opaque-fwd':
                # Weight-free control flow producing a hidden state is kept
                # as an opaque forward node: the transition embeds the whole
                # equation and its Jacobian is extracted in forward mode.
                emit(
                    kind=DiagnosticKind.CONTROL_FLOW_OPAQUE_FWD,
                    level=DiagnosticLevel.INFO,
                    message=(
                        f'Hidden state {self.outvar_to_hidden_path[outvar]} is '
                        f'produced by an opaque {op_name} with no weight inside; '
                        f'treating it as an opaque forward node (forward-mode '
                        f'Jacobians; reverse-mode signals through it are detached '
                        f'in the perturbed jaxpr for while).'
                    ),
                    context={
                        'op_name': op_name,
                        'evaluator': type(self).__name__,
                        'hidden_path': self.outvar_to_hidden_path[outvar],
                    },
                )
                return
            if policy.while_hidden != 'error':
                raise ValueError(
                    f"policy.while_hidden must be 'opaque-fwd' or 'error', "
                    f'got {policy.while_hidden!r}.'
                )
        raise NotImplementedError(
            f'Currently, we do not support the hidden states are computed within a {op_name} function. \n'
            f'Please remove your {op_name} on the intermediate steps. \n\n'
            f'The hidden state is: {self.outvar_to_hidden_path[outvar]}. \n'
            f'The Jaxpr of the {op_name} function is: \n\n'
            f'{eqn} \n\n'
        )


class JaxprEvaluation(object):
    """
    A base class for evaluating JAX program representations (jaxpr) to extract eligibility trace relationships.

    This class analyzes the computational graph represented as JAX primitives to identify and track
    relationships between weight parameters and hidden states for eligibility trace computation.
    Subclasses must implement the `_eval_eqn` method to define specific evaluation behavior.

    The class handles special JAX primitives such as pjit, scan, while, and cond operations,
    providing appropriate handling or restrictions for eligibility trace compilation.

    Parameters
    ----------
    weight_invars : Set[Var]
        Input variables representing weight parameters in the computational graph.
    hidden_invars : Set[Var]
        Input variables representing hidden states in the computational graph.
    hidden_outvars : Set[Var]
        Output variables representing hidden states in the computational graph.
    invar_to_hidden_path : Dict[Var, Path]
        Mapping from input variables to their paths in the hidden state hierarchy.
    outvar_to_hidden_path : Dict[Var, Path]
        Mapping from output variables to their paths in the hidden state hierarchy.
    control_flow : ControlFlowPolicy, optional
        Policy governing opaque control-flow handling (see
        :class:`~braintrace.ControlFlowPolicy`). Keyword-only. Defaults to
        the package default policy.

    Attributes
    ----------
    weight_invars : Set[Var]
        Stored input weight variables.
    hidden_invars : Set[Var]
        Stored input hidden state variables.
    hidden_outvars : Set[Var]
        Stored output hidden state variables.
    invar_to_hidden_path : Dict[Var, Path]
        Stored mapping from input variables to hidden paths.
    outvar_to_hidden_path : Dict[Var, Path]
        Stored mapping from output variables to hidden paths.
    control_flow : ControlFlowPolicy
        Stored control-flow policy.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        weight_invars: Set[Var],
        hidden_invars: Set[Var],
        hidden_outvars: Set[Var],
        invar_to_hidden_path: Dict[Var, Path],
        outvar_to_hidden_path: Dict[Var, Path],
        *,
        control_flow: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
    ):
        self.weight_invars = weight_invars
        self.hidden_invars = hidden_invars
        self.hidden_outvars = hidden_outvars
        self.invar_to_hidden_path = invar_to_hidden_path
        self.outvar_to_hidden_path = outvar_to_hidden_path
        self.control_flow = control_flow

    def _eval_jaxpr(self, jaxpr) -> None:
        """
        Evaluating the jaxpr for extracting the etrace relationships.

        Parameters
        ----------
        jaxpr : Jaxpr
            The jaxpr for the model.
        """

        for eqn in jaxpr.eqns:
            # TODO: add the support for the scan, while, cond, pjit, and other operators
            # Currently, scan, while, and cond are usually not the common operators used in
            # the definition of a brain dynamics model. So we may not need to consider them
            # during the current phase.
            # However, for the long-term maintenance and development, we need to consider them,
            # since users usually create crazy models.

            if is_jit_primitive(eqn):
                self._eval_pjit(eqn)
            elif is_scan_primitive(eqn):
                self._eval_scan(eqn)
            elif is_while_primitive(eqn):
                self._eval_while(eqn)
            elif is_cond_primitive(eqn):
                self._eval_cond(eqn)
            else:
                self._eval_eqn(eqn)

    def _eval_pjit(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.

        Parameters
        ----------
        eqn : JaxprEqn
            The JAX equation to evaluate.
        """
        # Defensive branch: in the dispatch flow a jit equation's primitive is
        # never an ETP primitive (those are custom Primitive instances), so
        # this is reachable only when ``_eval_pjit`` is called directly — as
        # unit tests do. Kept for that reason.
        if is_etp_primitive(eqn.primitive):
            if is_etp_enable_gradient_primitive(eqn.primitive):
                self._eval_eqn(eqn)
            return
        check_unsupported_op(self, eqn, 'jit')
        # treat the pjit as a normal jaxpr equation
        self._eval_eqn(eqn)

    def _eval_scan(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the scan primitive.

        Parameters
        ----------
        eqn : JaxprEqn
            The JAX equation to evaluate.
        """
        check_unsupported_op(self, eqn, 'scan')
        self._eval_eqn(eqn)

    def _eval_while(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the while primitive.

        Parameters
        ----------
        eqn : JaxprEqn
            The JAX equation to evaluate.
        """
        check_unsupported_op(self, eqn, 'while')
        self._eval_eqn(eqn)

    def _eval_cond(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the cond primitive.

        Parameters
        ----------
        eqn : JaxprEqn
            The JAX equation to evaluate.
        """
        check_unsupported_op(self, eqn, 'cond')
        self._eval_eqn(eqn)

    def _eval_eqn(self, eqn):
        """
        Evaluate a single JAX equation.

        This method must be implemented by subclasses to define specific
        evaluation behavior for equations.

        Parameters
        ----------
        eqn : JaxprEqn
            The JAX equation to evaluate.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        raise NotImplementedError(
            'The method "_eval_eqn" should be implemented in the subclass.'
        )
