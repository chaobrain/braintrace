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
# Date: 2024-04-03
# Copyright: 2024, Chaoming Wang
# ==============================================================================

# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional

import brainstate

from ._etrace_compiler_graph import ETraceGraph
from ._etrace_graph_executor import ETraceGraphExecutor
from ._typing import Path

__all__ = [
    'ETraceAlgorithm',
    'EligibilityTrace',
]


class EligibilityTrace(brainstate.ShortTermState):
    """
    The state for storing the eligibility trace during the computation of
    online learning algorithms.

    Examples
    --------
    When you are using :class:`braintrace.IODimVjpAlgorithm`, you can get
    the eligibility trace of the weight by calling:

    .. code-block:: python

        >>> etrace = etrace_algorithm.etrace_of(weight)

    """
    __module__ = 'braintrace'


class ETraceAlgorithm(brainstate.nn.Module):
    r"""
    The base class for the eligibility trace algorithm.

    Parameters
    ----------
    model : brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    name : str, optional
        The name of the etrace algorithm.

    Attributes
    ----------
    graph : ETraceGraphExecutor
        The etrace graph.
    param_states : Dict[Hashable, brainstate.ParamState]
        The weight states.
    hidden_states : Dict[Hashable, brainstate.HiddenState]
        The hidden states.
    other_states : Dict[Hashable, brainstate.State]
        The other states.
    is_compiled : bool
        Whether the etrace algorithm has been compiled.
    running_index : brainstate.ParamState[int]
        The running index.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        graph_executor: ETraceGraphExecutor,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # the model
        if not isinstance(model, brainstate.nn.Module):
            raise ValueError(
                f'The model should be a brainstate.nn.Module, this can help us to '
                f'better obtain the program structure. But we got {type(model)}.'
            )
        self.model4compile = model

        # the graph
        if not isinstance(graph_executor, ETraceGraphExecutor):
            raise ValueError(
                f'The graph should be a ETraceGraphExecutor, this can help us to '
                f'better obtain the program structure. But we got {type(graph_executor)}.'
            )
        self.graph_executor = graph_executor

        # The flag to indicate whether the etrace algorithm has been compiled
        self.is_compiled = False

        # the running index
        self.running_index = brainstate.LongTermState(0)

        # other states
        self._param_states = None
        self._hidden_states = None
        self._other_states = None

    @property
    def graph(self) -> ETraceGraph:
        """
        Get the etrace graph.

        Returns
        -------
        ETraceGraph
            The etrace graph.
        """
        return self.graph_executor.graph

    @property
    def executor(self) -> ETraceGraphExecutor:
        """
        Get the etrace graph executor.

        Returns
        -------
        ETraceGraphExecutor
            The etrace graph executor.
        """
        return self.graph_executor

    @property
    def param_states(self) -> brainstate.util.FlattedDict[Path, brainstate.ParamState]:
        """
        Get the parameter weight states.

        Returns
        -------
        brainstate.util.FlattedDict[Path, brainstate.ParamState]
            The parameter weight states.
        """
        if self._param_states is None:
            self._split_state()
        return self._param_states

    @property
    def hidden_states(self) -> brainstate.util.FlattedDict[Path, brainstate.HiddenState]:
        """
        Get the hidden states.

        Returns
        -------
        brainstate.util.FlattedDict[Path, brainstate.HiddenState]
            The hidden states.
        """
        if self._hidden_states is None:
            self._split_state()
        return self._hidden_states

    @property
    def other_states(self) -> brainstate.util.FlattedDict[Path, brainstate.State]:
        """
        Get the other states.

        Returns
        -------
        brainstate.util.FlattedDict[Path, brainstate.State]
            The other states.
        """
        if self._other_states is None:
            self._split_state()
        return self._other_states

    def _split_state(self):
        # --- the state separation --- #
        #
        # [NOTE]
        #
        # The `ETraceGraphExecutor` and the following states suggests that
        # `ETraceAlgorithm` depends on the states we created in the
        # `ETraceGraphExecutor`, including:
        #
        #   - the weight states, which is invariant during the training process
        #   - the hidden states, the recurrent states, which may be changed between different training epochs
        #   - the other states, which may be changed between different training epochs
        (
            self._param_states,
            self._hidden_states,
            self._other_states
        ) = self.graph.module_info.retrieved_model_states.split(brainstate.ParamState, brainstate.HiddenState, ...)

    def compile_graph(self, *args) -> None:
        r"""
        Compile the eligibility trace graph of the relationship between etrace weights, states and operators.

        The compilation process includes:

        - building the etrace graph
        - separating the states
        - initializing the etrace states

        Parameters
        ----------
        *args
            The input arguments.
        """

        if not self.is_compiled:
            # --- the model etrace graph -- #
            self.graph_executor.compile_graph(*args)

            # --- the initialization of the states --- #
            self.init_etrace_state(*args)

            # mark the graph is compiled
            self.is_compiled = True

    @property
    def path_to_states(self) -> brainstate.util.FlattedDict[Path, brainstate.State]:
        """
        Get the path to the states.

        Returns
        -------
        brainstate.util.FlattedDict[Path, brainstate.State]
            The mapping from path to states.
        """
        return self.graph_executor.path_to_states

    @property
    def state_id_to_path(self) -> Dict[int, Path]:
        """
        Get the state ID to the path.

        Returns
        -------
        Dict[int, Path]
            The mapping from state ID to path.
        """
        return self.graph_executor.state_id_to_path

    def show_graph(self) -> None:
        """
        Show the etrace graph.
        """
        return self.graph_executor.show_graph()

    def __call__(self, *args) -> Any:
        """
        Update the model and the eligibility trace states.

        Parameters
        ----------
        *args
            The input arguments.

        Returns
        -------
        Any
            The output of the update method.
        """
        return self.update(*args)

    def update(self, *args) -> Any:
        """
        Update the model and the eligibility trace states.

        Parameters
        ----------
        *args
            The input arguments.

        Returns
        -------
        Any
            The model output.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def init_etrace_state(self, *args, **kwargs) -> None:
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See `.compile_graph()` for the details.

        Parameters
        ----------
        *args
            The positional arguments.
        **kwargs
            The keyword arguments.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Any:
        """
        Get the eligibility trace of the given weight.

        Parameters
        ----------
        weight : brainstate.ParamState | Path
            The parameter weight or path to the weight.

        Returns
        -------
        Any
            The eligibility trace.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError
