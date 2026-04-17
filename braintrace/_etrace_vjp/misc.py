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

from functools import partial
from typing import Any, Dict, Optional

import brainstate
import saiunit as u
import jax


def _reset_state_in_a_dict(
    state_dict: Dict[Any, brainstate.State],
    batch_size: Optional[int],
):
    """
    Reset the values in a dictionary of states to zero.

    This function iterates over a dictionary of states and resets each state's
    value to a zero array. The shape of the zero array is determined by the
    original shape of the state's value and the specified batch size.

    Args:
        state_dict (Dict[Any, brainstate.State]): A dictionary where keys are any
            type and values are brainstate.State objects. Each state's value will be
            reset to a zero array.
        batch_size (Optional[int]): The size of the batch. If provided, the
            zero array will include a batch dimension; otherwise, it will not.

    Returns:
        None: The function modifies the state_dict in place, resetting each
        state's value to a zero array.
    """
    for k, v in state_dict.items():
        state_dict[k].value = jax.tree.map(partial(_zeros_like_batch_or_not, batch_size), v.value)


def _zeros_like_batch_or_not(
    batch_size: Optional[int],
    x: jax.Array
):
    """
    Create a zeros array with the same shape and type as the input array,
    optionally including a batch dimension.

    This function generates a zeros array that matches the shape and data type
    of the input array `x`. If a batch size is provided, the zeros array will
    include an additional batch dimension at the beginning.

    Args:
        batch_size (Optional[int]): The size of the batch. If provided, the
            zeros array will include a batch dimension. If None, the zeros
            array will have the same shape as `x`.
        x (jax.Array): The input array whose shape and data type are used as
            a reference for creating the zeros array.

    Returns:
        jax.Array: A zeros array with the same shape and data type as the
        input array, optionally including a batch dimension if `batch_size`
        is provided.
    """
    if batch_size is not None:
        assert isinstance(batch_size, int), 'The batch size should be an integer. '
        return u.math.zeros((batch_size,) + x.shape[1:], x.dtype)
    else:
        return u.math.zeros_like(x)


def _batched_zeros_like(
    batch_size: Optional[int],
    num_state: int,  # the number of hidden states
    x: jax.Array  # the input array
):
    """
    Create a batched zeros array with the same shape as the input array,
    extended by the number of hidden states.

    This function generates a zeros array that matches the shape of the
    input array `x`, with an additional dimension for the number of hidden
    states. If a batch size is provided, the zeros array will also include
    a batch dimension.

    Args:
        batch_size (Optional[int]): The size of the batch. If None, the
            batch dimension is not included.
        num_state (int): The number of hidden states, which determines the
            size of the additional dimension in the zeros array.
        x (jax.Array): The input array whose shape is used as a reference
            for creating the zeros array.

    Returns:
        jax.Array: A zeros array with the same shape as the input array,
        extended by the number of hidden states, and optionally including
        a batch dimension.
    """
    if batch_size is None:
        return u.math.zeros((*x.shape, num_state), x.dtype)
    else:
        return u.math.zeros((batch_size, *x.shape, num_state), x.dtype)


def _sum_dim(xs: jax.Array, axis: int = -1):
    """
    Sums the elements along the last dimension of each array in a PyTree.

    This function applies a sum operation along the last dimension of each array
    within a PyTree structure. It is useful for reducing the dimensionality of
    arrays by aggregating values along the specified axis.

    Args:
        xs (jax.Array): A PyTree of arrays where each array will have its last
                        dimension summed.

    Returns:
        jax.Array: A PyTree with the same structure as the input, where each array
                   has been reduced by summing over its last dimension.
    """
    return jax.tree.map(lambda x: u.math.sum(x, axis=axis), xs)


def _unit_safe_add(a, b):
    """Add two leaves, stripping units only when one side has units and the other does not.

    Gradient contributions for the same weight may come from paths that
    preserve physical units (e.g. VJP through the original jaxpr) and paths
    that already strip them (e.g. ETP ``xy_to_dw`` rules). When the two sides
    disagree on unit representation, both are reduced to plain arrays before
    adding; otherwise units are preserved.
    """
    a_is_q = isinstance(a, u.Quantity)
    b_is_q = isinstance(b, u.Quantity)
    if a_is_q != b_is_q:
        a = u.get_mantissa(a) if a_is_q else a
        b = u.get_mantissa(b) if b_is_q else b
    return u.math.add(a, b)


def _extract_weight_leaf(weight_val: brainstate.typing.PyTree, leaf_idx: int):
    """Extract the weight array from a (possibly pytree) ``ParamState`` value.

    When ``Linear`` (and related layers) merge weight and bias into a single
    ``ParamState({'weight': W, 'bias': b})``, runtime code that needs only the
    weight array must pick it out of the pytree value. ``leaf_idx`` is the
    ``weight_leaf_idx`` recorded on the ``HiddenParamOpRelation`` at compile
    time — it points into ``jax.tree.leaves(weight_val)`` so the right leaf is
    returned regardless of dict-key ordering or pytree shape.
    """
    leaves = jax.tree.leaves(weight_val)
    if not leaves:
        return weight_val
    if leaf_idx >= len(leaves):
        # Defensive: fall back to the first leaf if the index is out of range.
        return leaves[0]
    return leaves[leaf_idx]


def _wrap_leaf_as_pytree(
    leaf_val: jax.Array,
    reference_pytree: brainstate.typing.PyTree,
    leaf_idx: int,
):
    """Insert ``leaf_val`` at position ``leaf_idx`` in a pytree with the same
    structure as ``reference_pytree``, filling other leaves with zeros.

    Used to turn an etrace-computed weight gradient (just one array) into a
    gradient pytree that matches the ``ParamState``'s pytree value, e.g.
    ``{'weight': dW, 'bias': zeros_like(b)}``. The bias slot is zero because
    the current etrace rules only track the weight leaf; bias gradients from
    this pathway are simply absent.

    If ``reference_pytree`` is a bare array (no container nesting), the
    wrapping is a no-op — the unchanged ``leaf_val`` is returned.
    """
    ref_treedef = jax.tree.structure(reference_pytree)
    # Bare arrays have the trivial treedef of a single leaf. In that case the
    # ParamState is already shaped like the raw weight, so no wrapping needed.
    if ref_treedef.num_leaves <= 1 and ref_treedef == jax.tree.structure(0):
        return leaf_val
    leaves = jax.tree.leaves(reference_pytree)
    new_leaves = [
        leaf_val if i == leaf_idx else u.math.zeros_like(leaf)
        for i, leaf in enumerate(leaves)
    ]
    return jax.tree.unflatten(ref_treedef, new_leaves)


def _update_dict(
    the_dict: Dict,
    key: Any,
    value: brainstate.typing.PyTree,
    error_when_no_key: Optional[bool] = False
):
    """Update the dictionary.

    If the key exists, then add the value to the existing value.
    Otherwise, create a new key-value pair.

    Args:
      the_dict: The dictionary.
      key: The key.
      value: The value.
      error_when_no_key: bool, whether to raise an error when the key does not exist.

    """
    if key not in the_dict:
        if error_when_no_key:
            raise ValueError(f'The key {key} does not exist in the dictionary. ')
        the_dict[key] = value
    else:
        old_value = the_dict[key]
        if old_value is None:
            the_dict[key] = value
        else:
            the_dict[key] = jax.tree.map(
                _unit_safe_add,
                old_value,
                value,
                is_leaf=lambda x: isinstance(x, u.Quantity)
            )
