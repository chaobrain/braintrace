# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Shared helpers for the SNN online-learning algorithms."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional

import brainstate
import jax
import jax.numpy as jnp
import brainunit as u

from braintrace._typing import ArrayLike, PyTree

__all__ = [
    'PresynapticTrace',
    'KappaFilter',
    'FixedRandomFeedback',
    'extract_y_target',
]


class _ZeroResetState(brainstate.ShortTermState):
    """``ShortTermState`` that records its init shape/dtype and resets to zeros.

    Both :class:`PresynapticTrace` and :class:`KappaFilter` are leaky scalar-rate
    accumulators that share the same reset semantics: on ``reset_state`` they are
    re-zeroed at the original shape, optionally with the leading dimension swapped
    for ``batch_size`` (or prepended for scalar-shaped states).
    """

    def __init__(self, init_value: Any) -> None:
        super().__init__(init_value)
        self._init_shape = jnp.shape(init_value)
        self._init_dtype = init_value.dtype

    def reset_state(self, batch_size: Optional[int] = None, **kwargs: Any) -> None:
        """Re-zero the state at its original shape.

        Parameters
        ----------
        batch_size : int or None, optional
            If given, the state is re-zeroed with ``batch_size`` as the leading
            dimension (prepended when the original state was scalar-shaped).
            When ``None`` (default) the original unbatched shape is restored.
        **kwargs
            Ignored; accepted for compatibility with the brainstate state-reset
            protocol.
        """
        if batch_size is None:
            shape = self._init_shape
        elif len(self._init_shape) == 0:
            shape = (batch_size,)
        else:
            shape = (batch_size, *self._init_shape[1:])
        self.value = jnp.zeros(shape, dtype=self._init_dtype)


class PresynapticTrace(_ZeroResetState):
    r"""Leaky presynaptic accumulator used by OTTT and OTPE-Approx.

    The trace accumulates the presynaptic input with a multiplicative decay,
    following :math:`\hat{a} \leftarrow \lambda \cdot \hat{a} + x_t`.

    Parameters
    ----------
    init_value : jax.Array
        Initial value; also dictates the shape and dtype of the trace.
    leak : float
        Decay factor :math:`\lambda` in ``(0, 1)``. Pulled from the neuron's
        membrane leak in SNN usage.

    Raises
    ------
    ValueError
        If ``leak`` is not strictly inside the open interval ``(0, 1)``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> trace = braintrace.PresynapticTrace(jnp.zeros(3), leak=0.5)
        >>> out = trace.update(jnp.ones(3))
        >>> print(out)
        [1. 1. 1.]
        >>> out = trace.update(jnp.ones(3))
        >>> print(out)
        [1.5 1.5 1.5]
    """

    __module__ = 'braintrace'

    def __init__(self, init_value: Any, leak: float) -> None:
        super().__init__(init_value)
        if not (0.0 < leak < 1.0):
            raise ValueError(f'leak must be in (0, 1); got {leak}')
        self.leak = float(leak)

    def update(self, x: ArrayLike) -> Any:
        r"""Apply one accumulation step :math:`\hat{a} \leftarrow \lambda \cdot \hat{a} + x`.

        Parameters
        ----------
        x : jax.Array
            The new presynaptic input added to the decayed trace.

        Returns
        -------
        jax.Array
            The updated trace value.
        """
        self.value = self.leak * self.value + x
        return self.value


class KappaFilter(_ZeroResetState):
    r"""Low-pass filter helper state.

    :class:`~braintrace.EProp` no longer uses this class directly — it filters
    the eligibility trace internally instead. ``KappaFilter`` remains public
    and available for user-side filtering of an output-side (or any other)
    signal outside the algorithm's own hooks.

    The filter smooths the signal following
    :math:`x_{\mathrm{filt}} \leftarrow (1-\kappa) \cdot x + \kappa \cdot x_{\mathrm{filt}}`.

    Parameters
    ----------
    init_value : jax.Array
        Initial value; also dictates the shape and dtype of the filtered state.
    kappa : float
        Decay factor :math:`\kappa` in ``[0, 1)``. A value of ``0`` disables filtering.

    Raises
    ------
    ValueError
        If ``kappa`` is not inside the half-open interval ``[0, 1)``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> filt = braintrace.KappaFilter(jnp.zeros(3), kappa=0.5)
        >>> out = filt.update(jnp.ones(3))
        >>> print(out)
        [0.5 0.5 0.5]
        >>> out = filt.update(jnp.ones(3))
        >>> print(out)
        [0.75 0.75 0.75]
    """

    __module__ = 'braintrace'

    def __init__(self, init_value: Any, kappa: float) -> None:
        super().__init__(init_value)
        if not (0.0 <= kappa < 1.0):
            raise ValueError(f'kappa must be in [0, 1); got {kappa}')
        self.kappa = float(kappa)

    def update(self, x: ArrayLike) -> Any:
        r"""Apply one low-pass step :math:`x_{\mathrm{filt}} \leftarrow (1-\kappa) x + \kappa\, x_{\mathrm{filt}}`.

        Parameters
        ----------
        x : jax.Array
            The new input mixed into the filtered state.

        Returns
        -------
        jax.Array
            The updated, filtered value.
        """
        new = (1.0 - self.kappa) * x + self.kappa * self.value
        self.value = new
        return new


class FixedRandomFeedback:
    r"""Frozen random feedback matrix with a stop-gradient guard.

    The feedback matrix :math:`B \in \mathbb{R}^{n_{\mathrm{target}} \times n_{\mathrm{layer}}}`
    is sampled once at construction and frozen via :func:`jax.lax.stop_gradient`. It is used by
    OSTTP (per-HiddenGroup target projection) and EProp-random-feedback.

    Parameters
    ----------
    n_target : int
        Number of target dimensions (the row count of ``B``).
    n_layer : int
        Number of layer dimensions (the column count of ``B``).
    key : jax.Array
        A JAX PRNG key used to sample the feedback matrix.
    init_scale : float, optional
        Standard-deviation scaling applied to the sampled normal entries. Default is ``0.1``.

    Attributes
    ----------
    B : jax.Array
        The frozen feedback matrix of shape ``(n_target, n_layer)``.
    n_target : int
        Number of target dimensions.
    n_layer : int
        Number of layer dimensions.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import braintrace
        >>> fb = braintrace.FixedRandomFeedback(2, 3, jax.random.PRNGKey(0))
        >>> print(fb.B.shape)
        (2, 3)
        >>> y = jax.numpy.ones(2)
        >>> print(fb.project(y).shape)
        (3,)
    """

    __module__ = 'braintrace'

    def __init__(self, n_target: int, n_layer: int, key: Any, init_scale: float = 0.1) -> None:
        self.B = jax.lax.stop_gradient(
            init_scale * brainstate.random.normal(size=(n_target, n_layer), key=key)
        )
        self.n_target = int(n_target)
        self.n_layer = int(n_layer)

    def project(self, y_target: Any) -> Any:
        """Project the target onto the frozen feedback matrix.

        Parameters
        ----------
        y_target : jax.Array
            The target tensor to project. Both batched and unbatched layouts are handled.

        Returns
        -------
        jax.Array
            The projection ``y_target @ B`` with ``B`` frozen.
        """
        return y_target @ self.B


def extract_y_target(args: tuple, *, index: int = -1) -> Optional[jax.Array]:
    """Fetch the target tensor from a positional-args tuple.

    Returns ``None`` if ``args`` is empty. ``index`` defaults to the last position
    (OSTTP's convention: ``algo.update(x, y_target)``).
    """
    if not args:
        return None
    return args[index]


def _reset_state_in_a_dict(
    state_dict: Dict[Any, brainstate.State],
    batch_size: Optional[int],
) -> None:
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
) -> Any:
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
) -> Any:
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


def _sum_dim(xs: jax.Array, axis: int = -1) -> Any:
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


def _unit_safe_add(a: Any, b: Any) -> Any:
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


def _extract_leaf(pytree_val: PyTree, leaf_idx: int) -> Any:
    """Return the leaf at ``leaf_idx`` in ``jax.tree.leaves(pytree_val)``.

    Bare arrays (treedef with a single leaf) return the array unchanged.
    Raises ``IndexError`` if ``leaf_idx`` is outside ``len(leaves)``.
    """
    leaves = jax.tree.leaves(pytree_val)
    if not leaves:
        return pytree_val
    if leaf_idx < 0 or leaf_idx >= len(leaves):
        raise IndexError(
            f'leaf_idx {leaf_idx} out of range for pytree with {len(leaves)} leaves'
        )
    return leaves[leaf_idx]


def _wrap_leaves_as_pytree(
    reference_pytree: PyTree,
    leaf_grads: Dict[int, jax.Array],
) -> Any:
    """Build a pytree matching ``reference_pytree`` with ``leaf_grads``
    inserted at the given leaf indices; any other leaf is zero-filled.

    When the reference is a bare array, ``leaf_grads`` must contain at most one
    entry at index 0 and that value is returned directly (no wrapping).

    Raises ``IndexError`` if any supplied index is outside
    ``len(jax.tree.leaves(reference_pytree))``.
    """
    ref_treedef = jax.tree.structure(reference_pytree)
    # Bare-array fast path.
    # jax's PyTreeDef stubs omit num_leaves and __eq__; both are valid at runtime.
    if ref_treedef.num_leaves <= 1 and ref_treedef == jax.tree.structure(0):  # type: ignore[attr-defined, operator]
        if 0 in leaf_grads:
            return leaf_grads[0]
        return u.math.zeros_like(reference_pytree)
    leaves = jax.tree.leaves(reference_pytree)
    n = len(leaves)
    for idx in leaf_grads:
        if idx < 0 or idx >= n:
            raise IndexError(
                f'leaf_idx {idx} out of range for pytree with {n} leaves'
            )
    new_leaves = [
        leaf_grads[i] if i in leaf_grads else u.math.zeros_like(leaf)
        for i, leaf in enumerate(leaves)
    ]
    return jax.tree.unflatten(ref_treedef, new_leaves)


def _route_grads_by_path(
    relation: Any,
    per_key_grads: Dict[str, jax.Array],
    weight_vals: Dict[Any, PyTree],
    target_dict: Dict[Any, PyTree],
) -> None:
    """Route per-key gradients from a dict-API rule into per-path pytrees.

    Both D-RTRL and ES-D-RTRL share this bookkeeping: for each key in
    ``per_key_grads`` (returned by ``xy_to_dw`` or ``dt_to_t``), look up the
    owning ``ParamState`` path and the leaf index from ``relation``, accumulate
    into ``per_path``, then wrap with ``_wrap_leaves_as_pytree`` and merge into
    ``target_dict`` via ``_update_dict``.

    Args:
        relation: HiddenParamOpRelation — provides ``trainable_paths`` and
            ``trainable_leaf_indices``.
        per_key_grads: Dict[str, Array] — gradient contributions keyed by
            trainable invar name (e.g. ``'weight'``, ``'lora_b'``).
        weight_vals: Dict[Path, PyTree] — current ParamState pytree values;
            used as the structure template for ``_wrap_leaves_as_pytree``.
        target_dict: Dict[Path, PyTree] — accumulation target, modified in place.
    """
    per_path: Dict[Any, Dict[int, jax.Array]] = {}
    for key, grad in per_key_grads.items():
        path = relation.trainable_paths[key]
        leaf_idx = relation.trainable_leaf_indices[key]
        per_path.setdefault(path, {})[leaf_idx] = grad
    for path, leaf_to_grad in per_path.items():
        wrapped = _wrap_leaves_as_pytree(weight_vals[path], leaf_to_grad)
        _update_dict(target_dict, path, wrapped)


def _update_dict(
    the_dict: Dict,
    key: Any,
    value: PyTree,
    error_when_no_key: Optional[bool] = False
) -> None:
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
