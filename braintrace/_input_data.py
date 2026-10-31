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


from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple

import brainunit as u
import jax
from jax.tree_util import register_pytree_node_class

__all__ = [
    'SingleStepData',
    'MultiStepData',
]


class ETraceInputData:
    __module__ = 'braintrace'

    def __init__(self, data: Any) -> None:
        """
        Initializes an instance of ETraceInputData.

        Parameters
        ----------
        data : Any
            The data to be stored in the instance.
        """
        self.data = data

    def tree_flatten(self) -> tuple[tuple[Any], tuple[()]]:
        """
        Flattens the data for processing with JAX's tree utilities.

        Returns
        -------
        tuple
            A tuple containing the flattened data and an empty auxiliary data structure.
        """
        return (self.data,), ()

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Any) -> ETraceInputData:
        """
        Reconstructs an instance of ETraceInputData from flattened data.

        Parameters
        ----------
        aux : Any
            Auxiliary data structure, not used in this implementation.
        data : Any
            The flattened data to be reconstructed into an instance.

        Returns
        -------
        ETraceInputData
            An instance of ETraceInputData with the provided data.
        """
        return cls(*data)


@register_pytree_node_class
class SingleStepData(ETraceInputData):
    """A container marking input data as belonging to a single time step.

    Wraps an arbitrary pytree of arrays so the online-learning machinery
    can distinguish per-step inputs (which are reused at every step) from
    time-major inputs. Registered as a JAX pytree node, so instances can
    cross ``jit``/``grad`` boundaries transparently.

    Parameters
    ----------
    data : Any
        The pytree of arrays to store for a single time step.

    See Also
    --------
    MultiStepData : Container for inputs spanning multiple time steps.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> data = braintrace.SingleStepData(brainstate.random.randn(2, 3))
        >>> print(data.data.shape)
        (2, 3)
    """
    __module__ = 'braintrace'


@register_pytree_node_class
class MultiStepData(ETraceInputData):
    """A container marking input data as spanning multiple time steps.

    Wraps an arbitrary pytree of arrays whose leading axis is the time
    dimension, so the online-learning machinery can iterate over time
    steps. Registered as a JAX pytree node, so instances can cross
    ``jit``/``grad`` boundaries transparently.

    Parameters
    ----------
    data : Any
        The pytree of arrays to store. The first dimension of each array
        represents the time steps.

    See Also
    --------
    SingleStepData : Container for an input at a single time step.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> # data at 10 time steps, 2 samples each, 3 features per sample
        >>> data = braintrace.MultiStepData(brainstate.random.randn(10, 2, 3))
        >>> print(data.data.shape)
        (10, 2, 3)
    """
    __module__ = 'braintrace'


def is_input(x: Any) -> bool:
    return isinstance(x, (SingleStepData, MultiStepData))


def split_input_data_types(*args: Any) -> tuple[dict[int, SingleStepData], dict[int, MultiStepData], dict]:
    """
    Splits input data into dictionaries based on their type, distinguishing between
    SingleStepData and MultiStepData instances.

    Parameters
    ----------
    *args : Any
        Variable length argument list, expected to contain instances of SingleStepData
        or MultiStepData, or other data types.

    Returns
    -------
    tuple
        A tuple containing three elements:

        - A dictionary mapping indices to SingleStepData instances.
        - A dictionary mapping indices to MultiStepData instances.
        - A JAX tree structure definition of the input data.
    """
    leaves, tree_def = jax.tree.flatten(args, is_leaf=is_input)
    data_at_single_step = dict()
    data_at_multi_step = dict()
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, SingleStepData):
            data_at_single_step[i] = leaf.data
        elif isinstance(leaf, MultiStepData):
            data_at_multi_step[i] = leaf.data
        else:
            data_at_single_step[i] = leaf

    return data_at_single_step, data_at_multi_step, tree_def


def merge_data(tree_def: Any, *args: dict[int, Any]) -> Any:
    """
    Merges multiple dictionaries of data into a single structure based on a JAX tree definition.

    Parameters
    ----------
    tree_def : Any
        The JAX tree structure definition used to unflatten the data.
    *args : dict
        Variable length argument list, expected to contain dictionaries of data to be merged.

    Returns
    -------
    Any
        The merged data structure, reconstructed using the provided JAX tree definition.

    Raises
    ------
    ValueError
        If any expected data index is missing in the merged data.
    """
    data = dict()
    for arg in args:
        data.update(arg)
    for i in range(len(data)):
        if i not in data:
            raise ValueError(f"Data at index {i} is missing.")
    return jax.tree.unflatten(tree_def, tuple(data[i] for i in range(len(data))))


def get_single_step_data(*args: Any) -> Any:
    """
    Extracts and returns data corresponding to a single time step from the provided input data.

    This function processes input data, which may include instances of SingleStepData and
    MultiStepData, and returns a structure where MultiStepData is reduced to a single time step.

    Parameters
    ----------
    *args : Any
        Variable length argument list, expected to contain instances of SingleStepData,
        MultiStepData, or other data types.

    Returns
    -------
    Any
        The processed data structure, where MultiStepData instances are reduced to a single
        time step.
    """
    leaves, tree_def = jax.tree.flatten(args, is_leaf=is_input)
    leaves_processed = []
    for leaf in leaves:
        if isinstance(leaf, SingleStepData):
            leaves_processed.append(leaf.data)
        elif isinstance(leaf, MultiStepData):
            # we need the data at only single time step
            leaves_processed.append(jax.tree.map(lambda x: x[0], leaf.data))
        else:
            leaves_processed.append(leaf)
    args = jax.tree.unflatten(tree_def, leaves_processed)
    return args


def has_multistep_data(*args: Any) -> bool:
    """
    Determines if any of the provided input data contains MultiStepData instances.

    This function processes the input data, which may include instances of SingleStepData,
    MultiStepData, or other data types, and checks if any of the data is of type MultiStepData.

    Parameters
    ----------
    *args : Any
        Variable length argument list, expected to contain instances of SingleStepData,
        MultiStepData, or other data types.

    Returns
    -------
    bool
        True if any of the input data is an instance of MultiStepData, False otherwise.
    """
    leaves, _ = jax.tree.flatten(args, is_leaf=is_input)
    return any(isinstance(leaf, MultiStepData) for leaf in leaves)


# ---------------------------------------------------------------------------
# The signature of one step of the inputs (#227, #241)
# ---------------------------------------------------------------------------


def _is_quantity(x: Any) -> bool:
    return isinstance(x, u.Quantity)


@dataclasses.dataclass(frozen=True)
class LeafSignature:
    """The abstract value of one input leaf at one step.

    ``shape`` and ``dtype`` come from :func:`jax.typeof`, so a Python scalar has
    its canonical dtype and the weak type is dropped. ``unit`` is the unit of a
    :class:`brainunit.Quantity` and ``None`` for a plain array. A value that is
    not a JAX type is compared by its Python type only.
    """
    kind: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: Any = None
    unit: Any = None
    per_step: bool = dataclasses.field(default=False, compare=False)

    def describe(self) -> str:
        if self.shape is None:
            return f'a {self.kind}'
        text = f"{self.dtype}[{','.join(str(n) for n in self.shape)}]"
        if self.unit is not None:
            text += f' in {self.unit}'
        if self.per_step:
            text += ' per step of a MultiStepData'
        return text


@dataclasses.dataclass(frozen=True)
class InputSignature:
    """The structure and per-leaf abstract values of one step of the inputs.

    A :class:`brainunit.Quantity` is one leaf, so ``treedef`` does not depend on
    units; :class:`SingleStepData` and :class:`MultiStepData` are unwrapped, and
    a :class:`MultiStepData` leaf loses its leading time axis.
    """
    treedef: Any
    leaves: Tuple[Tuple[str, LeafSignature], ...]


def _leaf_signature(leaf: Any, per_step: bool) -> LeafSignature:
    if _is_quantity(leaf):
        unit, value = leaf.unit, leaf.mantissa
    else:
        unit, value = None, leaf
    try:
        aval = jax.typeof(value)
    except TypeError:
        return LeafSignature(kind=type(leaf).__name__)
    shape = tuple(aval.shape)
    if per_step:
        shape = shape[1:]
    return LeafSignature(
        kind='Quantity' if unit is not None else 'array',
        shape=shape,
        dtype=aval.dtype,
        unit=unit,
        per_step=per_step,
    )


def input_signature(*args: Any) -> InputSignature:
    """The signature of one step of *args*, as the model sees it.

    Parameters
    ----------
    *args : Any
        The inputs, which may hold :class:`SingleStepData` and
        :class:`MultiStepData`.

    Returns
    -------
    InputSignature
        The structure of the unwrapped inputs, with a :class:`brainunit.Quantity`
        as one leaf, and each leaf's key path and :class:`LeafSignature`.
    """
    leaves, outer = jax.tree.flatten(args, is_leaf=lambda x: is_input(x) or _is_quantity(x))
    one_step = []
    for leaf in leaves:
        if is_input(leaf):
            per_step = isinstance(leaf, MultiStepData)
            one_step.append(
                jax.tree.map(lambda x: _leaf_signature(x, per_step), leaf.data, is_leaf=_is_quantity)
            )
        else:
            one_step.append(_leaf_signature(leaf, False))
    with_path, treedef = jax.tree_util.tree_flatten_with_path(jax.tree.unflatten(outer, one_step))
    return InputSignature(
        treedef=treedef,
        leaves=tuple(('args' + jax.tree_util.keystr(path), sig) for path, sig in with_path),
    )


def _hint(compiled: LeafSignature, given: LeafSignature) -> str:
    if compiled.shape is None or given.shape is None or compiled.kind != given.kind:
        return f'pass {compiled.describe()}'
    if compiled.shape != given.shape:
        return 'compile a new learner with an example input of this shape'
    if compiled.unit is not None and given.unit is not None and compiled.unit != given.unit:
        if compiled.unit.dim == given.unit.dim:
            return f'convert it with `.in_unit({compiled.unit})`'
        return f'it has another physical dimension than {compiled.unit}'
    return f'cast it with `.astype({compiled.dtype})`'


def check_input_signature(compiled: InputSignature, given: InputSignature, caller: str) -> None:
    """Raise a ``TypeError`` if *given* differs from the *compiled* signature.

    Parameters
    ----------
    compiled : InputSignature
        The signature of the example inputs the program was traced with.
    given : InputSignature
        The signature of the inputs of this call.
    caller : str
        The name of the call, for the message.

    Raises
    ------
    TypeError
        If the structures differ, or any leaf's shape, dtype, unit or kind does.
        The message names every differing leaf by its key path.
    """
    why = (
        'The compiled program replays the jaxpr traced from the example inputs '
        'of `compile_graph()`; it neither retraces nor casts.'
    )
    if compiled.treedef != given.treedef:
        raise TypeError(
            f'{caller} was given inputs of structure {given.treedef}, but it was '
            f'compiled with inputs of structure {compiled.treedef}. {why} Pass '
            f'inputs of the compiled structure, or compile a new learner with an '
            f'example input like this one.'
        )
    lines = [
        f'  {path}: compiled with {c.describe()}, given {g.describe()}; {_hint(c, g)}.'
        for (path, c), (_, g) in zip(compiled.leaves, given.leaves)
        if c != g
    ]
    if lines:
        raise TypeError(
            f'{caller} was given inputs that differ from the example inputs it '
            f'was compiled with:\n' + '\n'.join(lines) + f'\n{why}'
        )
