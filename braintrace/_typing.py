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

from typing import Dict, Sequence, Union, FrozenSet, List, Tuple, Any, TypeAlias

import brainstate
import jax
import numpy as np

from ._compatible_imports import Var

ArrayLike: TypeAlias = brainstate.typing.ArrayLike
DType: TypeAlias = brainstate.typing.DType
DTypeLike: TypeAlias = brainstate.typing.DTypeLike

# --- types --- #
PyTree: TypeAlias = Any
StateID: TypeAlias = int
WeightID: TypeAlias = int
Size: TypeAlias = brainstate.typing.Size
Axis: TypeAlias = int


def as_size_tuple(size: Size) -> Tuple[int, ...]:
    """Normalize an ``in_size``/``out_size`` spec to a tuple of ints.

    ``brainstate``'s size setters accept a scalar ``int`` or a sequence, while
    the matching getters are typed as the broad :data:`Size` union, which static
    type checkers do not treat as indexable. Routing values through this helper
    yields a concrete ``tuple[int, ...]`` so both property assignment and
    trailing-dimension lookups (``size[-1]``) type-check cleanly, reproducing
    ``brainstate``'s own normalization at runtime.

    Parameters
    ----------
    size : Size
        A scalar ``int`` / numpy integer, or a sequence of them.

    Returns
    -------
    tuple of int
        The size expressed as a tuple of Python ints.
    """
    if isinstance(size, (int, np.integer)):
        return (int(size),)
    return tuple(int(s) for s in size)
Axes: TypeAlias = Union[int, Sequence[int]]
Path: TypeAlias = Tuple[str, ...]

# --- inputs and outputs --- #
Inputs: TypeAlias = PyTree
Outputs: TypeAlias = PyTree

# --- state values --- #
HiddenVals: TypeAlias = Dict[Path, PyTree]
StateVals: TypeAlias = Dict[Path, PyTree]
WeightVals: TypeAlias = Dict[Path, PyTree]
ETraceVals: TypeAlias = Dict[Path, PyTree]

HiddenOutVar: TypeAlias = Var
HiddenInVar: TypeAlias = Var

# --- gradients --- #
dG_Inputs: TypeAlias = PyTree  # gradients of inputs
dG_Weight: TypeAlias = Sequence[PyTree]  # gradients of weights
dG_Hidden: TypeAlias = Sequence[PyTree]  # gradients of hidden states
dG_State: TypeAlias = Sequence[PyTree]  # gradients of other states

VarID: TypeAlias = int

HiddenGroupName: TypeAlias = str
ETraceX_Key: TypeAlias = VarID
ETraceY_Key: TypeAlias = VarID
ETraceDF_Key: TypeAlias = Tuple[VarID, HiddenGroupName]

_WeightPath: TypeAlias = Path
_HiddenPath: TypeAlias = Path
# D-RTRL keys weight-gradient traces by (weight y-var id, hidden-group index).
ETraceWG_Key: TypeAlias = Tuple[ETraceY_Key, int]
HidHidJac_Key: TypeAlias = Tuple[Path, Path]

# --- data --- #
WeightXVar: TypeAlias = Var
WeightYVar: TypeAlias = Var
WeightXs: TypeAlias = Dict[Var, jax.Array]
WeightDfs: TypeAlias = Dict[Var, jax.Array]
TempData: TypeAlias = Dict[Var, jax.Array]
Current: TypeAlias = ArrayLike  # the synaptic current
Conductance: TypeAlias = ArrayLike  # the synaptic conductance
Spike: TypeAlias = ArrayLike  # the spike signal
# the diagonal Jacobian of the hidden-to-hidden function
Hid2HidDiagJacobian: TypeAlias = Dict[
    FrozenSet[HiddenOutVar],
    Dict[HiddenOutVar, List[jax.Array]]
]
Hid2WeightJacobian: TypeAlias = Tuple[
    Dict[ETraceX_Key, jax.Array],
    Dict[ETraceDF_Key, jax.Array]
]
Hid2HidJacobian: TypeAlias = Dict[
    HidHidJac_Key,
    jax.Array
]
HiddenGroupJacobian: TypeAlias = Sequence[jax.Array]
