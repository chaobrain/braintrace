# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Blocked randomized Hadamard rotation.

The full Walsh-Hadamard transform over a length-``d`` axis costs ``log2(d)``
sweeps of the array. On a bandwidth-bound host that price is paid in full for
every rotation. Restricting the butterfly to contiguous blocks of ``block``
elements folds those stages into a single small ``dot_general`` that XLA emits
as one fused pass, trading a bounded amount of mixing for a constant number of
memory sweeps.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'block_hadamard_matrix',
    'sign_diagonal',
    'rotate_blocks',
    'unrotate_blocks',
]


@functools.lru_cache(maxsize=16)
def block_hadamard_matrix(block: int) -> jax.Array:
    """Return the orthonormal Hadamard matrix of order ``block``.

    Parameters
    ----------
    block : int
        Matrix order. Must be a positive power of two.

    Returns
    -------
    jax.Array
        ``(block, block)`` float32 array ``H`` with ``H @ H.T == I``.

    Raises
    ------
    ValueError
        If ``block`` is not a positive power of two.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintrace._quant import block_hadamard_matrix
        >>> h = block_hadamard_matrix(2)
        >>> bool(jnp.allclose(h @ h.T, jnp.eye(2), atol=1e-6))
        True
    """
    if block <= 0 or block & (block - 1):
        raise ValueError(f'block must be a positive power of two, got {block}')
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < block:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return jnp.asarray(matrix / np.sqrt(block), dtype=jnp.float32)


def sign_diagonal(key: jax.Array, size: int) -> jax.Array:
    """Draw the random +/-1 diagonal that randomizes the Hadamard rotation.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    size : int
        Length of the diagonal.

    Returns
    -------
    jax.Array
        ``(size,)`` float32 array of +1.0 and -1.0 entries.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from braintrace._quant import sign_diagonal
        >>> signs = sign_diagonal(jax.random.PRNGKey(0), 8)
        >>> signs.shape
        (8,)
    """
    if size <= 0:
        raise ValueError(f'size must be positive, got {size}')
    return jnp.where(jax.random.bernoulli(key, 0.5, (size,)), 1.0, -1.0).astype(
        jnp.float32
    )


def _reshaped(values: jax.Array, block: int) -> jax.Array:
    trailing = values.shape[-1]
    if trailing % block:
        raise ValueError(
            f'trailing axis {trailing} is not a multiple of block {block}'
        )
    return values.reshape(*values.shape[:-1], trailing // block, block)


def rotate_blocks(values: jax.Array, signs: jax.Array, block: int) -> jax.Array:
    """Apply ``H_block . D_signs`` along the trailing axis of ``values``.

    Parameters
    ----------
    values : jax.Array
        Array whose trailing axis is a multiple of ``block``.
    signs : jax.Array
        Sign diagonal broadcastable against the trailing axis.
    block : int
        Hadamard block order.

    Returns
    -------
    jax.Array
        Rotated array with the shape of ``values``.

    Examples
    --------
    .. code-block:: python

        >>> import jax, jax.numpy as jnp
        >>> from braintrace._quant import rotate_blocks, unrotate_blocks, sign_diagonal
        >>> x = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
        >>> s = sign_diagonal(jax.random.PRNGKey(2), 16)
        >>> bool(jnp.allclose(unrotate_blocks(rotate_blocks(x, s, 8), s, 8), x, atol=1e-5))
        True
    """
    matrix = block_hadamard_matrix(block)
    blocked = _reshaped(values * signs, block)
    return (blocked @ matrix).reshape(values.shape)


def unrotate_blocks(values: jax.Array, signs: jax.Array, block: int) -> jax.Array:
    """Invert :func:`rotate_blocks`.

    Parameters
    ----------
    values : jax.Array
        Rotated array whose trailing axis is a multiple of ``block``.
    signs : jax.Array
        Sign diagonal used by the forward rotation.
    block : int
        Hadamard block order.

    Returns
    -------
    jax.Array
        Array in the original basis.
    """
    matrix = block_hadamard_matrix(block)
    blocked = _reshaped(values, block)
    return (blocked @ matrix.T).reshape(values.shape) * signs
