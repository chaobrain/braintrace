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

from typing import Sequence, Tuple, List, Hashable, Dict

import brainstate

from ._etrace_concepts import ETraceParam
from ._typing import Path, WeightVals, HiddenVals, StateVals


def assign_state_values(
    states: Sequence[brainstate.State],
    state_values: Sequence[brainstate.typing.PyTree],
    write: bool = True
):
    """
    Assign or restore values to a sequence of states.

    This function assigns new values to the given states or restores their previous values
    based on the `write` flag.

    Parameters
    -----------
    states : Sequence[brainstate.State]
        A sequence of state objects to which values will be assigned or restored.
    state_values : Sequence[brainstate.typing.PyTree]
        A sequence of values corresponding to each state in `states`.
    write : bool, optional
        A flag indicating whether to assign (`True`) or restore (`False`) the values.
        Defaults to `True`.

    Returns
    --------
    None
    """
    if write:
        for st, val in zip(states, state_values):
            st.value = val
    else:
        for st, val in zip(states, state_values):
            st.restore_value(val)


def assign_dict_state_values(
    states: Dict[Path, brainstate.State],
    state_values: Dict[Path, brainstate.typing.PyTree],
    write: bool = True
):
    """
    Assign or restore values to a dictionary of states.

    This function assigns new values to the given states or restores their previous values
    based on the `write` flag. It ensures that the keys of the `states` and `state_values`
    dictionaries match before proceeding with the assignment or restoration.

    Parameters
    -----------
    states : Dict[Path, brainstate.State]
        A dictionary where keys are paths and values are state objects
        to which values will be assigned or restored.
    state_values : Dict[Path, brainstate.typing.PyTree]
        A dictionary where keys are paths and values are the values
        corresponding to each state in `states`.
    write : bool, optional
        A flag indicating whether to assign (`True`) or restore (`False`) the values.
        Defaults to `True`.

    Returns
    --------
    None
    """
    if set(states.keys()) != set(state_values.keys()):
        raise ValueError('The keys of states and state_values must be the same.')

    if write:
        for key in states.keys():
            states[key].value = state_values[key]
    else:
        for key in states.keys():
            states[key].restore_value(state_values[key])


def assign_state_values_v2(
    states: Dict[Hashable, brainstate.State],
    state_values: Dict[Hashable, brainstate.typing.PyTree],
    write: bool = True
):
    """
    Assign or restore values to a dictionary of states.

    This function assigns new values to the given states or restores their previous values
    based on the `write` flag. It ensures that the keys of the `states` and `state_values`
    dictionaries match before proceeding with the assignment or restoration.

    Parameters
    -----------
    states : Dict[Hashable, brainstate.State]
        A dictionary where keys are hashable identifiers and values are state objects
        to which values will be assigned or restored.
    state_values : Dict[Hashable, brainstate.typing.PyTree]
        A dictionary where keys are hashable identifiers and values are the values
        corresponding to each state in `states`.
    write : bool, optional
        A flag indicating whether to assign (`True`) or restore (`False`) the values.
        Defaults to `True`.

    Returns
    --------
    None
    """
    assert set(states.keys()) == set(state_values.keys()), (
        f'The keys of states and state_values must be '
        f'the same. Got: \n '
        f'{states.keys()} \n '
        f'{state_values.keys()}'
    )

    if write:
        for key in states.keys():
            states[key].value = state_values[key]
    else:
        for key in states.keys():
            states[key].restore_value(state_values[key])


def split_states(
    states: Sequence[brainstate.State]
) -> Tuple[List[brainstate.ParamState], List[brainstate.HiddenState], List[brainstate.State]]:
    """
    Split the states into weight states, hidden states, and other states.

    This function categorizes a sequence of states into three distinct groups:
    weight parameter states, hidden states, and other states. It helps in organizing
    the states based on their types for further processing or analysis.

    Parameters
    -----------
    states : Sequence[brainstate.State]
        A sequence of state objects to be split into different categories.

    Returns
    --------
    Tuple[List[brainstate.ParamState], List[brainstate.HiddenState], List[brainstate.State]]
        A tuple containing three lists:
        - param_states: The list of weight parameter states.
        - hidden_states: The list of hidden states.
        - other_states: The list of other states that do not fall into the first two categories.
    """
    param_states, hidden_states, other_states = [], [], []
    for st in states:
        if isinstance(st, brainstate.HiddenState):  # etrace hidden variables
            hidden_states.append(st)
        elif isinstance(st, brainstate.ParamState):  # including all weight states, ParamState, ETraceParam
            param_states.append(st)
        else:
            other_states.append(st)
    return param_states, hidden_states, other_states


def split_states_v2(
    states: Sequence[brainstate.State]
) -> Tuple[
    List[ETraceParam],
    List[brainstate.HiddenState],
    List[brainstate.ParamState],
    List[brainstate.State]
]:
    """
    Categorize a sequence of states into etrace parameter states, hidden states, 
    parameter states, and other states.

    This function is crucial for determining which ParamState should be trained 
    with the eligibility trace and which should not.

    Parameters
    -----------
    states : Sequence[brainstate.State]
        A sequence of state objects to be categorized.

    Returns
    --------
    Tuple[List[ETraceParam], List[brainstate.HiddenState], List[brainstate.ParamState], List[brainstate.State]]
        A tuple containing four lists:
        - etrace_param_states: The list of etrace parameter states.
        - hidden_states: The list of hidden states.
        - param_states: The list of other kinds of parameter states.
        - other_states: The list of other states that do not fall into the first three categories.
    """
    etrace_param_states, hidden_states, param_states, other_states = [], [], [], []
    for st in states:
        if isinstance(st, brainstate.HiddenState):
            hidden_states.append(st)
        elif isinstance(st, ETraceParam):
            if st.is_etrace:
                etrace_param_states.append(st)
            else:
                # The ETraceParam is not set to "is_etrace" since
                # no hidden state is associated with it,
                # so it should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states.append(st)

        else:
            if isinstance(st, brainstate.ParamState):
                # The ParamState which is not an ETraceParam,
                # should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states.append(st)
            else:
                other_states.append(st)
    return etrace_param_states, hidden_states, param_states, other_states


def sequence_split_state_values(
    states: Sequence[brainstate.State],
    state_values: List[brainstate.typing.PyTree],
    include_weight: bool = True
) -> (
    Tuple[
        Sequence[brainstate.typing.PyTree],
        Sequence[brainstate.typing.PyTree],
        Sequence[brainstate.typing.PyTree]
    ]
    |
    Tuple[
        Sequence[brainstate.typing.PyTree],
        Sequence[brainstate.typing.PyTree]
    ]
):
    """
    Split the state values into the weight values, the hidden values, and the other state values.

    The weight values are the values of the ``braincore.ParamState`` states (including ``ETraceParam``).
    The hidden values are the values of the ``ETraceState`` states.
    The other state values are the values of the other states.

    Parameters
    -----------
    states: Sequence[brainstate.State]
      The states of the model.
    state_values: List[PyTree]
      The values of the states.
    include_weight: bool
      Whether to include the weight values.

    Returns
    --------
    The weight values, the hidden values, and the other state values.

    Examples
    ---------
    >>> sequence_split_state_values(states, state_values)
    (weight_path_to_vals, hidden_vals, other_vals)

    >>> sequence_split_state_values(states, state_values, include_weight=False)
    (hidden_vals, other_vals)
    """
    if include_weight:
        weight_vals, hidden_vals, other_vals = [], [], []
        for st, val in zip(states, state_values):
            if isinstance(st, brainstate.ParamState):
                weight_vals.append(val)
            elif isinstance(st, brainstate.HiddenState):
                hidden_vals.append(val)
            else:
                other_vals.append(val)
        return weight_vals, hidden_vals, other_vals
    else:
        hidden_vals, other_vals = [], []
        for st, val in zip(states, state_values):
            if isinstance(st, brainstate.ParamState):
                pass
            elif isinstance(st, brainstate.HiddenState):
                hidden_vals.append(val)
            else:
                other_vals.append(val)
        return hidden_vals, other_vals


def dict_split_state_values(
    states: Dict[Path, brainstate.State],
    state_values: Dict[Path, brainstate.typing.PyTree],
) -> Tuple[WeightVals, HiddenVals, StateVals]:
    """
    Split the state values into weight values, hidden values, and other state values.

    This function categorizes the given state values into three distinct groups based on the type
    of the corresponding state: weight values, hidden values, and other state values.

    Parameters
    -----------
    states : Dict[Path, brainstate.State]
        A dictionary where keys are paths and values are state objects to be categorized.
    state_values : Dict[Path, brainstate.typing.PyTree]
        A dictionary where keys are paths and values are the values associated with each state.

    Returns
    --------
    Tuple[WeightVals, HiddenVals, StateVals]
        A tuple containing three dictionaries:
        - weight_vals: The values of the weight parameter states.
        - hidden_vals: The values of the hidden states.
        - other_vals: The values of the other states.
    """
    weight_vals = dict()
    hidden_vals = dict()
    other_vals = dict()
    for path, state in states.items():
        val = state_values[path]
        if isinstance(state, brainstate.ParamState):
            weight_vals[path] = val
        elif isinstance(state, brainstate.HiddenState):
            hidden_vals[path] = val
        else:
            other_vals[path] = val
    return weight_vals, hidden_vals, other_vals


def split_dict_states_v1(
    states: Dict[Path, brainstate.State]
) -> Tuple[
    Dict[Path, brainstate.HiddenState],
    Dict[Path, brainstate.ParamState],
    Dict[Path, brainstate.State]
]:
    """
    Categorize the given states into hidden states, parameter states, and other states.

    This function is crucial for determining which ParamState should be trained with the eligibility trace.

    .. note::

        This function is important since it determines what ParamState should be
        trained with the eligibility trace and what should not.

    Parameters
    -----------
    states : Dict[Path, brainstate.State]
        A dictionary where keys are paths and values are state objects to be split.

    Returns
    --------
    Tuple[Dict[Path, brainstate.HiddenState], Dict[Path, brainstate.ParamState], Dict[Path, brainstate.State]]
        A tuple containing three dictionaries:
        - hidden_states: The hidden states.
        - param_states: The other kinds of parameter states.
        - other_states: The other states.
    """
    hidden_states = dict()
    param_states = dict()
    other_states = dict()
    for key, st in states.items():
        if isinstance(st, brainstate.HiddenState):
            hidden_states[key] = st
        elif isinstance(st, brainstate.ParamState):
            # The ParamState which is not an ETraceParam,
            # should be treated as a normal parameter state
            # and be trained with spatial gradients only
            param_states[key] = st
        else:
            other_states[key] = st
    return hidden_states, param_states, other_states


def split_dict_states_v2(
    states: Dict[Path, brainstate.State]
) -> Tuple[
    Dict[Path, ETraceParam],
    Dict[Path, brainstate.HiddenState],
    Dict[Path, brainstate.ParamState],
    Dict[Path, brainstate.State]
]:
    """
    Split the states into etrace parameter states, hidden states, parameter states, and other states.

    .. note::

        This function is important since it determines what ParamState should be
        trained with the eligibility trace and what should not.

    This function categorizes the given states into four distinct groups based on their types:
    etrace parameter states, hidden states, parameter states, and other states. It is crucial
    for determining which ParamState should be trained with the eligibility trace.

    Parameters
    -----------
    states : Dict[Path, brainstate.State]
        A dictionary where keys are paths and values are state objects to be split.

    Returns
    --------
    Tuple[Dict[Path, ETraceParam], Dict[Path, brainstate.HiddenState], Dict[Path, brainstate.ParamState], Dict[Path, brainstate.State]]
        A tuple containing four dictionaries:
        - etrace_param_states: The etrace parameter states.
        - hidden_states: The hidden states.
        - param_states: The other kinds of parameter states.
        - other_states: The other states.
    """
    etrace_param_states = dict()
    hidden_states = dict()
    param_states = dict()
    other_states = dict()
    for key, st in states.items():
        if isinstance(st, brainstate.HiddenState):
            hidden_states[key] = st
        elif isinstance(st, ETraceParam):
            if st.is_etrace:
                etrace_param_states[key] = st
            else:
                # The ETraceParam is set to "is_etrace = False" since
                # no hidden state is associated with it,
                # so it should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states[key] = st
        else:
            if isinstance(st, brainstate.ParamState):
                # The ParamState which is not an ETraceParam,
                # should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states[key] = st
            else:
                other_states[key] = st
    return etrace_param_states, hidden_states, param_states, other_states
