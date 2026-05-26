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


from typing import Any

import jax
from jax.tree_util import register_pytree_node_class

__all__ = [
    'SingleStepData',
    'MultiStepData',
]


class ETraceInputData:
    __module__ = 'braintrace'

    def __init__(self, data: Any):
        """
        Initializes an instance of ETraceInputData.

        Args:
            data (Any): The data to be stored in the instance.
        """
        self.data = data

    def tree_flatten(self):
        """
        Flattens the data for processing with JAX's tree utilities.

        Returns:
            tuple: A tuple containing the flattened data and an empty auxiliary data structure.
        """
        return (self.data,), ()

    @classmethod
    def tree_unflatten(cls, aux, data):
        """
        Reconstructs an instance of ETraceInputData from flattened data.

        Args:
            aux: Auxiliary data structure, not used in this implementation.
            data: The flattened data to be reconstructed into an instance.

        Returns:
            ETraceInputData: An instance of ETraceInputData with the provided data.
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


def is_input(x):
    return isinstance(x, (SingleStepData, MultiStepData))


def split_input_data_types(*args) -> tuple[dict[int, SingleStepData], dict[int, MultiStepData], dict]:
    """
    Splits input data into dictionaries based on their type, distinguishing between
    SingleStepData and MultiStepData instances.

    Args:
        *args: Variable length argument list, expected to contain instances of SingleStepData
               or MultiStepData, or other data types.

    Returns:
        tuple: A tuple containing three elements:
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


def merge_data(tree_def, *args):
    """
    Merges multiple dictionaries of data into a single structure based on a JAX tree definition.

    Args:
        tree_def: The JAX tree structure definition used to unflatten the data.
        *args: Variable length argument list, expected to contain dictionaries of data to be merged.

    Returns:
        The merged data structure, reconstructed using the provided JAX tree definition.

    Raises:
        ValueError: If any expected data index is missing in the merged data.
    """
    data = dict()
    for arg in args:
        data.update(arg)
    for i in range(len(data)):
        if i not in data:
            raise ValueError(f"Data at index {i} is missing.")
    return jax.tree.unflatten(tree_def, tuple(data[i] for i in range(len(data))))


def get_single_step_data(*args):
    """
    Extracts and returns data corresponding to a single time step from the provided input data.

    This function processes input data, which may include instances of SingleStepData and 
    MultiStepData, and returns a structure where MultiStepData is reduced to a single time step.

    Args:
        *args: Variable length argument list, expected to contain instances of SingleStepData,
               MultiStepData, or other data types.

    Returns:
        The processed data structure, where MultiStepData instances are reduced to a single time step.
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


def has_multistep_data(*args):
    """
    Determines if any of the provided input data contains MultiStepData instances.

    This function processes the input data, which may include instances of SingleStepData,
    MultiStepData, or other data types, and checks if any of the data is of type MultiStepData.

    Args:
        *args: Variable length argument list, expected to contain instances of SingleStepData,
               MultiStepData, or other data types.

    Returns:
        bool: True if any of the input data is an instance of MultiStepData, False otherwise.
    """
    leaves, _ = jax.tree.flatten(args, is_leaf=is_input)
    return any(isinstance(leaf, MultiStepData) for leaf in leaves)
