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
# ==============================================================================

# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Union, FrozenSet, List, Tuple

import brainstate
import jax

from ._compatible_imports import Var

ArrayLike = brainstate.typing.ArrayLike
DType = brainstate.typing.DType
DTypeLike = brainstate.typing.DTypeLike

# --- types --- #
PyTree = brainstate.typing.PyTree
StateID = int
WeightID = int
Size = brainstate.typing.Size
Axis = int
Axes = Union[int, Sequence[int]]
Path = Tuple[str, ...]

# --- inputs and outputs --- #
Inputs = PyTree
Outputs = PyTree

# --- state values --- #
HiddenVals = Dict[Path, PyTree]
StateVals = Dict[Path, PyTree]
WeightVals = Dict[Path, PyTree]
ETraceVals = Dict[Path, PyTree]

HiddenOutVar = Var
HiddenInVar = Var

# --- gradients --- #
dG_Inputs = PyTree  # gradients of inputs
dG_Weight = Sequence[PyTree]  # gradients of weights
dG_Hidden = Sequence[PyTree]  # gradients of hidden states
dG_State = Sequence[PyTree]  # gradients of other states

VarID = int

HiddenGroupName = str
ETraceX_Key = VarID
ETraceY_Key = VarID
ETraceDF_Key = Tuple[VarID, HiddenGroupName]

_WeightPath = Path
_HiddenPath = Path
ETraceWG_Key = Tuple[_WeightPath, ETraceY_Key, HiddenGroupName]
HidHidJac_Key = Tuple[Path, Path]

# --- data --- #
WeightXVar = Var
WeightYVar = Var
WeightXs = Dict[Var, jax.Array]
WeightDfs = Dict[Var, jax.Array]
TempData = Dict[Var, jax.Array]
Current = ArrayLike  # the synaptic current
Conductance = ArrayLike  # the synaptic conductance
Spike = ArrayLike  # the spike signal
# the diagonal Jacobian of the hidden-to-hidden function
Hid2HidDiagJacobian = Dict[
    FrozenSet[HiddenOutVar],
    Dict[HiddenOutVar, List[jax.Array]]
]
Hid2WeightJacobian = Tuple[
    Dict[ETraceX_Key, jax.Array],
    Dict[ETraceDF_Key, jax.Array]
]
Hid2HidJacobian = Dict[
    HidHidJac_Key,
    jax.Array
]
HiddenGroupJacobian = Sequence[jax.Array]
