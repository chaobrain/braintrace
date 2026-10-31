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
import jax.numpy as jnp
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
# One step of the inputs against the example inputs (#227, #241)
# ---------------------------------------------------------------------------


def _is_quantity(x: Any) -> bool:
    return isinstance(x, u.Quantity)


_PYTHON_SCALARS = (bool, int, float, complex)


@dataclasses.dataclass(frozen=True)
class LeafSignature:
    """The abstract value of one input leaf at one step.

    ``kind`` is ``'array'`` for a JAX value (an array, a NumPy array, a Python
    scalar or a :class:`brainunit.Quantity`) and the name of the Python type for
    any other value, which is compared by that type only. ``shape``, ``dtype``
    and ``weak`` come from :func:`jax.typeof`, so a Python scalar has its
    canonical dtype. ``unit`` is a ``Quantity``'s unit and ``None`` for a plain
    array. ``weak``, ``per_step`` (a step of a :class:`MultiStepData`) and
    ``scalar`` (the type of a Python scalar, for messages) are not compared.
    """
    kind: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: Any = None
    unit: Any = None
    weak: bool = dataclasses.field(default=False, compare=False)
    per_step: bool = dataclasses.field(default=False, compare=False)
    scalar: Optional[str] = dataclasses.field(default=None, compare=False)

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
    units, and a plain ``list`` counts as a ``tuple`` in it, at any depth.
    :class:`SingleStepData` and :class:`MultiStepData` are unwrapped, and a
    :class:`MultiStepData` leaf loses its leading time axis. ``structure`` is the
    structure as given, for messages.
    """
    treedef: Any
    leaves: Tuple[Tuple[str, LeafSignature], ...]
    structure: str = dataclasses.field(default='', compare=False)


def _leaf_signature(leaf: Any, per_step: bool) -> LeafSignature:
    unit, value = (leaf.unit, leaf.mantissa) if _is_quantity(leaf) else (None, leaf)
    try:
        aval = jax.typeof(value)
    except TypeError:
        return LeafSignature(kind=type(leaf).__name__)
    shape = tuple(aval.shape)
    return LeafSignature(
        kind='array',
        shape=shape[1:] if per_step else shape,
        dtype=aval.dtype,
        unit=unit,
        weak=bool(getattr(aval, 'weak_type', False)),
        per_step=per_step,
        scalar=type(value).__name__ if type(value) in _PYTHON_SCALARS else None,
    )


def _lists_as_tuples(tree: Any) -> Any:
    if type(tree) in (list, tuple):
        return tuple(_lists_as_tuples(x) for x in tree)
    if type(tree) is dict:
        return {k: _lists_as_tuples(v) for k, v in tree.items()}
    return tree


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
        as one leaf and lists as tuples, and each leaf's key path and
        :class:`LeafSignature`.
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
    tree = jax.tree.unflatten(outer, one_step)
    with_path, structure = jax.tree_util.tree_flatten_with_path(tree)
    return InputSignature(
        treedef=jax.tree.structure(_lists_as_tuples(tree)),
        leaves=tuple(('args' + jax.tree_util.keystr(path), sig) for path, sig in with_path),
        structure=str(structure),
    )


# the dtype categories; a weak-typed value takes a compiled dtype of its rank or above
_CATEGORIES = (
    ('bool', jnp.bool_),
    ('int', jnp.unsignedinteger),
    ('int', jnp.signedinteger),
    ('float', jnp.floating),
    ('complex', jnp.complexfloating),
)
_RANK = {'bool': 0, 'int': 1, 'float': 2, 'complex': 3}


def _category(dtype: Any) -> Optional[str]:
    """The category of *dtype*, ``None`` for an extended dtype such as a PRNG key."""
    for name, category in _CATEGORIES:
        if jnp.issubdtype(dtype, category):
            return name
    return None


def _takes_the_dtype(compiled: LeafSignature, given: LeafSignature) -> bool:
    """Whether *given* takes *compiled*'s dtype by a cast that changes no value,
    or by JAX's weak-type rule.

    A strong value is cast only within its category (bool, unsigned int, signed
    int, float, complex) and only to a type its own type promotes to, so
    ``float16`` and ``bfloat16`` are not cast to each other. A weak-typed value
    (a Python scalar) is cast to a dtype of its category or a higher one, as
    JAX would.
    """
    c, g = _category(compiled.dtype), _category(given.dtype)
    if c is None or g is None:
        return False
    if given.weak:
        return _RANK[g] <= _RANK[c]
    if jnp.issubdtype(given.dtype, jnp.unsignedinteger) != jnp.issubdtype(compiled.dtype, jnp.unsignedinteger):
        return False
    return c == g and jnp.promote_types(given.dtype, compiled.dtype) == compiled.dtype


def _python_type(dtype: Any) -> str:
    return {'bool': 'bool', 'int': 'int', 'float': 'float', 'complex': 'complex'}.get(_category(dtype) or '', 'value')


def _article(word: str) -> str:
    return 'an' if word[:1] in 'aeio' else 'a'


def _conform_leaf(leaf: Any, compiled: LeafSignature, given: LeafSignature) -> Tuple[Any, list]:
    """*leaf* in the compiled form, or the remedies for each difference."""
    if compiled.shape is None or given.shape is None:
        return leaf, ([] if compiled.kind == given.kind else [f'pass {compiled.describe()}'])
    remedies = []
    if compiled.shape != given.shape:
        remedies.append('compile a new learner with an example input of this shape')
    c_unit = u.UNITLESS if compiled.unit is None else compiled.unit
    g_unit = u.UNITLESS if given.unit is None else given.unit
    if c_unit != g_unit:
        if c_unit == u.UNITLESS:
            remedies.append('pass a plain array')
        elif g_unit == u.UNITLESS:
            remedies.append(f'pass a Quantity in {c_unit}')
        elif c_unit.dim == g_unit.dim:
            remedies.append(f'convert it with `.in_unit({c_unit})`')
        else:
            remedies.append(f'it has another physical dimension than {c_unit}')
    cast = compiled.dtype != given.dtype
    if cast and not _takes_the_dtype(compiled, given):
        if _category(compiled.dtype) is None or _category(given.dtype) is None:
            remedies.append(f'pass {_article(str(compiled.dtype))} {compiled.dtype} value')
        elif given.scalar is not None:
            kind = _python_type(compiled.dtype)
            dtype = str(compiled.dtype)
            remedies.append(f'pass a Python {kind} or {_article(dtype)} {dtype} scalar')
        else:
            remedies.append(f'cast it with `.astype({compiled.dtype})`')
    if remedies:
        return leaf, remedies
    if cast:
        leaf = leaf.astype(compiled.dtype) if _is_quantity(leaf) else jax.lax.convert_element_type(leaf, compiled.dtype)
    if compiled.unit is None and given.unit is not None:
        leaf = leaf.mantissa
    elif compiled.unit is not None and given.unit is None:
        leaf = u.Quantity(leaf, unit=compiled.unit)
    return leaf, []


def conform_inputs(compiled: InputSignature, args: Tuple[Any, ...], caller: str) -> Tuple[Any, ...]:
    """*args* conformed to the *compiled* signature, or a ``TypeError``.

    The structures must match, with a plain list standing for a tuple. Per leaf,
    the shape must match; the unit must match, a plain array standing for unit 1;
    the dtype must match, or *given*'s must take the compiled one by a lossless
    cast within its category (``float16`` to ``float32``, ``int8`` to ``int32``)
    or, for a weak-typed value such as a Python scalar, by JAX's weak-type rule.
    Such a leaf is cast to the compiled dtype, and a unit-1 leaf takes the
    compiled form; the structure is left as given.

    Parameters
    ----------
    compiled : InputSignature
        The signature of the example inputs the program was traced with.
    args : tuple
        The inputs of this call.
    caller : str
        The name of the call, for the message.

    Returns
    -------
    tuple
        *args*, itself when nothing is converted.

    Raises
    ------
    TypeError
        If the structures differ, or any leaf differs in another way. The message
        names every differing leaf by its key path, with every remedy that applies.
    """
    given = input_signature(*args)
    if given == compiled:  # the common case: nothing to conform
        return args
    rule = (
        'The compiled program replays the jaxpr traced from the example inputs of '
        '`compile_graph()`: it takes a lossless upcast and a Python scalar in the '
        'compiled dtype, a plain array for unit 1 and a list for a tuple, and '
        'nothing else.'
    )
    if compiled.treedef != given.treedef:
        note = ''
        if ('OrderedDict' in given.structure) != ('OrderedDict' in compiled.structure):
            note = (
                ' A dict flattens in sorted key order and an OrderedDict in insertion '
                'order, so their leaves need not line up.'
            )
        raise TypeError(
            f'{caller} was given inputs of structure {given.structure}, but it was '
            f'compiled with inputs of structure {compiled.structure}.{note} {rule} Pass '
            f'inputs of the compiled structure, or compile a new learner with an '
            f'example input like this one.'
        )
    leaves, treedef = jax.tree.flatten(args, is_leaf=_is_quantity)
    conformed, lines, changed = [], [], False
    for leaf, (path, c), (_, g) in zip(leaves, compiled.leaves, given.leaves):
        new, remedies = _conform_leaf(leaf, c, g)
        if remedies:
            lines.append(f'  {path}: compiled with {c.describe()}, given {g.describe()}; {"; ".join(remedies)}.')
        changed = changed or new is not leaf
        conformed.append(new)
    if lines:
        raise TypeError(
            f'{caller} was given inputs that differ from the example inputs it '
            f'was compiled with:\n' + '\n'.join(lines) + f'\n{rule}'
        )
    return tuple(jax.tree.unflatten(treedef, conformed)) if changed else args
