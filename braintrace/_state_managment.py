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

from typing import Sequence, Tuple, List, Hashable, Dict, Mapping, Any

import brainstate

pass  # ParamState removed (primitive-based ETP)
from ._typing import Path, PyTree


def assign_dict_state_values(
    states: Mapping[Path, brainstate.State],
    state_values: Mapping[Path, PyTree],
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
    state_values : Dict[Path, PyTree]
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
    states: Mapping[Any, brainstate.State],
    state_values: Mapping[Any, PyTree],
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
    state_values : Dict[Hashable, PyTree]
        A dictionary where keys are hashable identifiers and values are the values
        corresponding to each state in `states`.
    write : bool, optional
        A flag indicating whether to assign (`True`) or restore (`False`) the values.
        Defaults to `True`.

    Returns
    --------
    None
    """
    if set(states.keys()) != set(state_values.keys()):
        raise ValueError(
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


def sequence_split_state_values(
    states: Sequence[brainstate.State],
    state_values: List[PyTree],
    include_weight: bool = True
) -> (
    Tuple[
        Sequence[PyTree],
        Sequence[PyTree],
        Sequence[PyTree]
    ]
    |
    Tuple[
        Sequence[PyTree],
        Sequence[PyTree]
    ]
):
    """
    Split the state values into the weight values, the hidden values, and the other state values.

    The weight values are the values of the ``braincore.ParamState`` states (including ``ParamState``).
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


def split_dict_states_v2(
    states: Dict[Path, brainstate.State]
) -> Tuple[
    Dict[Path, brainstate.ParamState],
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
    Tuple[Dict[Path, brainstate.ParamState], Dict[Path, brainstate.HiddenState], Dict[Path, brainstate.ParamState], Dict[Path, brainstate.State]]
        A tuple containing four dictionaries:
        - etrace_param_states: ParamState instances used with ETP primitives (all ParamState for now).
        - hidden_states: The hidden states.
        - param_states: Other ParamState not used with ETP primitives (empty — split determined by compiler).
        - other_states: The other states.
    """
    etrace_param_states = dict()
    hidden_states = dict()
    param_states: dict = dict()  # stays empty; value type cannot be inferred
    other_states = dict()
    for key, st in states.items():
        if isinstance(st, brainstate.HiddenState):
            hidden_states[key] = st
        elif isinstance(st, brainstate.ParamState):
            # All ParamState go to etrace_param_states.
            # The compiler determines which ones actually participate
            # in ETP via primitive scanning.
            etrace_param_states[key] = st
        else:
            other_states[key] = st
    return etrace_param_states, hidden_states, param_states, other_states
