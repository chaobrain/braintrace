# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Lloyd-Max scalar codebooks for unit-variance Gaussian coordinates.

After a randomized Hadamard rotation the coordinates of a normalized vector in
dimension ``d`` follow ``Beta((d - 3) / 2)``, which converges to ``N(0, 1 / d)``.
The centroids below are the minimum-distortion levels for ``N(0, 1)``; callers
scale them by the measured coordinate deviation.

References
----------
.. [1] Max, J. (1960). "Quantizing for Minimum Distortion." *IRE Transactions
   on Information Theory*, 6(1), 7-12.
.. [2] Lloyd, S. (1982). "Least Squares Quantization in PCM." *IEEE
   Transactions on Information Theory*, 28(2), 129-137.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'LLOYDMAX_CENTROIDS',
    'lloydmax_codebook',
    'encode_nearest',
    'decode_centroids',
    'gaussian_codebook_scale',
]

LLOYDMAX_CENTROIDS: dict[int, tuple[float, ...]] = {
    1: (-0.7978845608, 0.7978845608),
    2: (-1.5104, -0.4528, 0.4528, 1.5104),
    3: (
        -2.1519, -1.3440, -0.7560, -0.2451,
        0.2451, 0.7560, 1.3440, 2.1519,
    ),
    4: (
        -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
        0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
    ),
}


def lloydmax_codebook(bits: int, scale: float | jax.Array = 1.0) -> jax.Array:
    """Return the ascending Lloyd-Max centroids for ``bits`` levels.

    Parameters
    ----------
    bits : int
        Code width in bits. Supported values are 1 through 4.
    scale : float or jax.Array, optional
        Multiplier applied to the unit-variance centroids.

    Returns
    -------
    jax.Array
        ``(2 ** bits,)`` float32 array of centroids in ascending order.

    Raises
    ------
    ValueError
        If ``bits`` has no tabulated codebook.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace._quant import lloydmax_codebook
        >>> lloydmax_codebook(1).shape
        (2,)
    """
    if bits not in LLOYDMAX_CENTROIDS:
        raise ValueError(
            f'bits must be one of {sorted(LLOYDMAX_CENTROIDS)}, got {bits}'
        )
    table = jnp.asarray(LLOYDMAX_CENTROIDS[bits], dtype=jnp.float32)
    return table * jnp.asarray(scale, dtype=jnp.float32)


def encode_nearest(values: jax.Array, codebook: jax.Array) -> jax.Array:
    """Map every element of ``values`` to its nearest codebook index.

    Parameters
    ----------
    values : jax.Array
        Array to quantize.
    codebook : jax.Array
        Ascending ``(levels,)`` centroid array.

    Returns
    -------
    jax.Array
        Unsigned 8-bit indices with the shape of ``values``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintrace._quant import lloydmax_codebook, encode_nearest
        >>> book = lloydmax_codebook(1)
        >>> encode_nearest(jnp.asarray([-2.0, 2.0]), book).tolist()
        [0, 1]
    """
    edges = (codebook[:-1] + codebook[1:]) * 0.5
    return jnp.searchsorted(edges, values.reshape(-1)).reshape(
        values.shape
    ).astype(jnp.uint8)


def decode_centroids(indices: jax.Array, codebook: jax.Array) -> jax.Array:
    """Look up centroids for ``indices``.

    Parameters
    ----------
    indices : jax.Array
        Integer code indices.
    codebook : jax.Array
        Ascending ``(levels,)`` centroid array.

    Returns
    -------
    jax.Array
        Float32 reconstruction with the shape of ``indices``.
    """
    return codebook[indices.astype(jnp.int32)]


def gaussian_codebook_scale(bits: int) -> float:
    """Return the residual standard deviation left by a ``bits``-wide codebook.

    Parameters
    ----------
    bits : int
        Code width in bits.

    Returns
    -------
    float
        Root mean squared quantization error for unit-variance input.
    """
    book = np.asarray(LLOYDMAX_CENTROIDS[bits], dtype=np.float64)
    samples = np.linspace(-6.0, 6.0, 200001)
    density = np.exp(-0.5 * samples**2) / np.sqrt(2.0 * np.pi)
    nearest = book[np.abs(samples[:, None] - book[None, :]).argmin(axis=1)]
    error = np.trapezoid((samples - nearest) ** 2 * density, samples)
    return float(np.sqrt(error))
