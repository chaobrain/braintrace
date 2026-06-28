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
# Author: Chaoming Wang <chao.brain@qq.com>
# Copyright: 2024, Chaoming Wang
# Date: 2024-04-03
#
# ==============================================================================
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
#   [2024-11] version 0.0.3, a complete new revision for better model debugging.
#
# ==============================================================================

# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional

import brainstate

from braintrace._etrace_compiler import ETraceGraph, compile_etrace_graph
from .._input_data import get_single_step_data
from .._typing import Path

__all__ = [
    'ETraceGraphExecutor',
]


class ETraceGraphExecutor:
    r"""
    The eligibility trace graph executor.

    This class is used for computing the weight spatial gradients and the hidden state residuals.
    It is the most foundational class for the ETrace algorithms.

    It is important to note that the graph is built no matter whether the model is
    batched or not. This means that this graph can be applied to any kind of models.
    However, the compilation is sensitive to the shape of hidden states.

    Parameters
    ----------
    model: brainstate.nn.Module
        The model to build the eligibility trace graph. The models should only define the one-step behavior.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        include_recurrent_mixing: bool = False,
    ):
        # The original model
        if not isinstance(model, brainstate.nn.Module):
            raise TypeError(
                'The model should be an instance of "brainstate.nn.Module" since '
                'we can extract the program structure from the model for '
                'better debugging.'
            )
        self.model = model

        # hidden-group grouping mode for the hidden-to-hidden transition; see
        # ``compile_etrace_graph(..., include_recurrent_mixing=...)``.
        self.include_recurrent_mixing = include_recurrent_mixing

        # the compiled graph
        self._compiled_graph: Optional[ETraceGraph] = None
        self._state_id_to_path: Optional[Dict[int, Path]] = None

    @property
    def graph(self) -> ETraceGraph:
        """
        Retrieve the compiled eligibility trace graph for the model.

        This property provides access to the compiled graph, which is a crucial data structure
        for the eligibility trace algorithm. It contains various attributes that describe the
        relationships between the model's variables, states, and operations.

        Returns
        -------
        ETraceGraph
            The compiled graph for the model. This graph includes detailed information about
            the model's structure, such as output variables, state variables,
            hidden-to-hidden variable relationships, and more.

        Raises
        ------
        ValueError
            If the graph has not been compiled yet. Ensure to call the
            :meth:`compile_graph` method before accessing this property.
        """
        if self._compiled_graph is None:
            raise ValueError('The graph is not compiled yet. Please call ".compile_graph()" first.')
        return self._compiled_graph

    @property
    def states(self) -> brainstate.util.FlattedDict:
        """
        The states for the model.

        Returns
        -------
        brainstate.util.FlattedDict[Path, brainstate.State]
            The states for the model.
        """
        return self.graph.module_info.retrieved_model_states

    @property
    def path_to_states(self) -> brainstate.util.FlattedDict:
        """
        The path to the states.

        Returns
        -------
        brainstate.util.FlattedDict[Path, brainstate.State]
            The path to the states.
        """
        return self.states

    @property
    def state_id_to_path(self) -> Dict[int, Path]:
        """
        The state id to the path.

        Returns
        -------
        Dict[int, Path]
            The mapping from state id to the path.
        """
        if self._state_id_to_path is None:
            self._state_id_to_path = {id(state): path for path, state in self.states.items()}
        return self._state_id_to_path

    def compile_graph(self, *args) -> None:
        r"""
        Build the eligibility trace graph for the model based on the provided inputs.

        This method is crucial for constructing the graph used in the eligibility trace
        algorithm, which is essential for calculating weight spatial gradients and the
        hidden state Jacobian.

        Parameters
        ----------
        *args
            Positional arguments for the model, which may include inputs, parameters, or
            other necessary data required for graph compilation.

        Returns
        -------
        None
            This method does not return any value. It initializes the compiled graph
            attribute of the instance.
        """

        # invalidate cached mappings on recompilation
        self._state_id_to_path = None

        # process the inputs
        args = get_single_step_data(*args)

        # compile the graph
        self._compiled_graph = compile_etrace_graph(
            self.model, *args,
            include_recurrent_mixing=self.include_recurrent_mixing,
        )

    def show_graph(
        self,
        verbose: bool = True,
        return_msg: bool = False,
    ) -> None | str:
        """Display the graph illustrating weights, operators, and hidden states.

        Renders via :class:`braintrace.CompilationReport`, the single source of
        truth for the structural summary.

        Parameters
        ----------
        verbose : bool, optional
            If True (default), print the summary to stdout.
        return_msg : bool, optional
            If True, also return the summary string. Default False.

        Returns
        -------
        None or str
            The summary string if ``return_msg`` is True, else None.
        """
        from braintrace._etrace_compiler import CompilationReport
        msg = CompilationReport(self.graph).to_str(1)
        if verbose:
            print(msg)
        if return_msg:
            return msg
        return None

    def solve_h2w_h2h_jacobian(
        self,
        *args,
    ) -> Any:
        r"""
        Compute the hidden-to-weight and hidden-to-hidden Jacobian matrices.

        This function is designed to calculate the forward propagation of the hidden-to-weight Jacobian
        and the hidden-to-hidden Jacobian based on the provided inputs and parameters. It is a crucial
        part of the eligibility trace algorithm, which helps in understanding the influence of weights
        and previous hidden states on the current hidden state.

        Parameters
        ----------
        *args
            Positional arguments for the model, which may include inputs, parameters, or other necessary
            data required for the computation of the Jacobians.

        Returns
        -------
        Any
            A tuple containing the following elements:

            - The function output (e.g., model predictions).
            - The updated hidden states after the current computation step.
            - Other states that may be relevant to the model's operation.
            - The spatial gradients of the weights, represented by the hidden-to-weight Jacobian.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Notes
        -----
        For the state transition function :math:`y, h^t = f(h^{t-1}, \theta, x)`, this function aims
        to solve:

        1. The function output :math:`y`.
        2. The updated hidden states :math:`h^t`.
        3. The Jacobian matrix of hidden-to-weight, i.e., :math:`\partial h^t / \partial \theta^t`.
        4. The Jacobian matrix of hidden-to-hidden, i.e., :math:`\partial h^t / \partial h^{t-1}`.
        """
        raise NotImplementedError('The method "solve_h2w_h2h_jacobian" should be '
                                  'implemented in the subclass.')

    def solve_h2w_h2h_l2h_jacobian(
        self, *args,
    ) -> Any:
        r"""
        Compute the hidden-to-weight and hidden-to-hidden Jacobian matrices, along with the VJP transformed
        loss-to-hidden gradients based on the provided inputs.

        This function is designed to calculate both the forward propagation of the hidden-to-weight Jacobian
        and the loss-to-hidden gradients at the current time-step. It is essential for understanding the
        influence of weights and previous hidden states on the current hidden state, as well as the impact
        of the loss on the hidden states.

        Parameters
        ----------
        *args
            Positional arguments for the model, which may include inputs, parameters, or other necessary
            data required for the computation of the Jacobians and gradients.

        Returns
        -------
        Any
            A tuple containing the following elements:

            - The function output (e.g., model predictions).
            - The updated hidden states after the current computation step.
            - Other states that may be relevant to the model's operation.
            - The spatial gradients of the weights, represented by the hidden-to-weight Jacobian.
            - The residuals, which are the partial gradients of the loss with respect to the hidden states.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Notes
        -----
        Particularly, this function aims to solve:

        1. The Jacobian matrix of hidden-to-weight. That is,
           :math:`\partial h / \partial w`, where :math:`h` is the hidden state and :math:`w` is the weight.
        2. The Jacobian matrix of hidden-to-hidden. That is,
           :math:`\partial h / \partial h`, where :math:`h` is the hidden state.
        3. The partial gradients of the loss with respect to the hidden states.
           That is, :math:`\partial L / \partial h`, where :math:`L` is the loss and :math:`h` is the hidden state.
        """
        raise NotImplementedError('The method "solve_h2w_h2h_jacobian_and_l2h_vjp" '
                                  'should be implemented in the subclass.')
