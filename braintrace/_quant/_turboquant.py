# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Two-stage TurboQuant codec over the trailing axis of a batch of vectors.

Stage one normalizes each vector, rotates it with a randomized blocked Hadamard
transform, and assigns every coordinate a Lloyd-Max centroid using ``bits - 1``
bits. Stage two spends the remaining bit on the sign pattern of a second,
independently rotated projection of the stage-one residual, which removes the
first stage's bias in inner-product estimates.

References
----------
.. [1] Zandieh, A., Han, I., Daliri, M., & Karbasi, A. (2026). "TurboQuant:
   Online Vector Quantization with Near-optimal Distortion Rate." *ICLR*.
   arXiv:2504.19874.
.. [2] Zandieh, A., Han, I., Mirrokni, V., & Karbasi, A. (2024). "QJL: 1-Bit
   Quantized JL Transform for KV Cache Quantization." arXiv:2406.03482.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._hadamard import rotate_blocks, sign_diagonal, unrotate_blocks
from ._lloydmax import decode_centroids, encode_nearest, lloydmax_codebook

__all__ = [
    'TurboQuantSpec',
    'TurboQuantCode',
    'build_spec',
    'encode',
    'decode',
    'relative_distortion',
]

_QJL_GAIN = math.sqrt(math.pi / 2.0)


class TurboQuantSpec(NamedTuple):
    """Immutable codec configuration shared by :func:`encode` and :func:`decode`.

    Attributes
    ----------
    bits : int
        Total bit budget per coordinate. One bit funds the QJL stage.
    block : int
        Hadamard block order used by both rotation stages.
    dim : int
        Length of the quantized trailing axis.
    polar_signs : jax.Array
        Sign diagonal of the stage-one rotation.
    qjl_signs : jax.Array
        Sign diagonal of the stage-two rotation.
    use_qjl : bool
        Whether the residual correction stage is active.
    """

    bits: int
    block: int
    dim: int
    polar_signs: jax.Array
    qjl_signs: jax.Array
    use_qjl: bool


class TurboQuantCode(NamedTuple):
    """Compressed representation produced by :func:`encode`.

    Attributes
    ----------
    indices : jax.Array
        Stage-one Lloyd-Max indices, one uint8 per coordinate.
    norms : jax.Array
        Euclidean norm of every input vector.
    signs : jax.Array
        Packed stage-two sign bits, or an empty array when QJL is disabled.
    residual_norms : jax.Array
        Euclidean norm of every stage-one residual.
    """

    indices: jax.Array
    norms: jax.Array
    signs: jax.Array
    residual_norms: jax.Array


def build_spec(
    key: jax.Array,
    dim: int,
    bits: int = 4,
    block: int = 64,
    use_qjl: bool = True,
) -> TurboQuantSpec:
    """Draw the random rotations for a codec over vectors of length ``dim``.

    Parameters
    ----------
    key : jax.Array
        PRNG key seeding both sign diagonals.
    dim : int
        Length of the trailing axis to quantize.
    bits : int, optional
        Total bit budget per coordinate, between 2 and 5.
    block : int, optional
        Hadamard block order. Must divide ``dim`` and be a power of two.
    use_qjl : bool, optional
        Whether to spend one bit on the residual correction stage.

    Returns
    -------
    TurboQuantSpec
        Configuration usable by :func:`encode` and :func:`decode`.

    Raises
    ------
    ValueError
        If ``dim`` is not a multiple of ``block`` or the stage-one width is
        outside the tabulated Lloyd-Max codebooks.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from braintrace._quant import build_spec
        >>> spec = build_spec(jax.random.PRNGKey(0), 256, bits=4)
        >>> spec.bits, spec.block
        (4, 64)
    """
    if dim % block:
        raise ValueError(f'dim {dim} is not a multiple of block {block}')
    stage_one = bits - 1 if use_qjl else bits
    if not 1 <= stage_one <= 4:
        raise ValueError(f'stage-one width {stage_one} is outside 1..4 bits')
    polar_key, qjl_key = jax.random.split(key)
    return TurboQuantSpec(
        bits=bits,
        block=block,
        dim=dim,
        polar_signs=sign_diagonal(polar_key, dim),
        qjl_signs=sign_diagonal(qjl_key, dim),
        use_qjl=use_qjl,
    )


def _stage_one_bits(spec: TurboQuantSpec) -> int:
    return spec.bits - 1 if spec.use_qjl else spec.bits


def encode(values: jax.Array, spec: TurboQuantSpec) -> TurboQuantCode:
    """Compress ``values`` along its trailing axis.

    Parameters
    ----------
    values : jax.Array
        Array whose trailing axis has length ``spec.dim``.
    spec : TurboQuantSpec
        Codec configuration.

    Returns
    -------
    TurboQuantCode
        Compressed representation.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from braintrace._quant import build_spec, encode, decode
        >>> spec = build_spec(jax.random.PRNGKey(0), 128, bits=4)
        >>> x = jax.random.normal(jax.random.PRNGKey(1), (4, 128))
        >>> code = encode(x, spec)
        >>> code.indices.shape
        (4, 128)
    """
    if values.shape[-1] != spec.dim:
        raise ValueError(
            f'trailing axis {values.shape[-1]} does not match spec dim {spec.dim}'
        )
    norms = jnp.linalg.norm(values, axis=-1, keepdims=True)
    safe = jnp.where(norms > 0.0, norms, 1.0)
    unit = values / safe
    rotated = rotate_blocks(unit, spec.polar_signs, spec.block)
    codebook = lloydmax_codebook(_stage_one_bits(spec), 1.0 / math.sqrt(spec.dim))
    indices = encode_nearest(rotated, codebook)
    if not spec.use_qjl:
        empty = jnp.zeros(values.shape[:-1] + (0,), dtype=jnp.uint8)
        zeros = jnp.zeros_like(norms)
        return TurboQuantCode(indices, norms, empty, zeros)
    stage_one = unrotate_blocks(
        decode_centroids(indices, codebook), spec.polar_signs, spec.block
    )
    residual = unit - stage_one
    residual_norms = jnp.linalg.norm(residual, axis=-1, keepdims=True)
    projected = rotate_blocks(residual, spec.qjl_signs, spec.block)
    return TurboQuantCode(
        indices, norms, (projected >= 0.0).astype(jnp.uint8), residual_norms
    )


def decode(code: TurboQuantCode, spec: TurboQuantSpec) -> jax.Array:
    """Reconstruct the array compressed into ``code``.

    Parameters
    ----------
    code : TurboQuantCode
        Compressed representation from :func:`encode`.
    spec : TurboQuantSpec
        Codec configuration used to produce ``code``.

    Returns
    -------
    jax.Array
        Float32 reconstruction.
    """
    codebook = lloydmax_codebook(_stage_one_bits(spec), 1.0 / math.sqrt(spec.dim))
    stage_one = unrotate_blocks(
        decode_centroids(code.indices, codebook), spec.polar_signs, spec.block
    )
    if not spec.use_qjl:
        return stage_one * code.norms
    bipolar = code.signs.astype(jnp.float32) * 2.0 - 1.0
    correction = unrotate_blocks(bipolar, spec.qjl_signs, spec.block)
    gain = _QJL_GAIN / math.sqrt(spec.dim)
    return (stage_one + correction * gain * code.residual_norms) * code.norms


def relative_distortion(values: jax.Array, spec: TurboQuantSpec) -> jax.Array:
    """Return the per-vector relative reconstruction error of a round trip.

    Parameters
    ----------
    values : jax.Array
        Array whose trailing axis has length ``spec.dim``.
    spec : TurboQuantSpec
        Codec configuration.

    Returns
    -------
    jax.Array
        ``||x - decode(encode(x))|| / ||x||`` for every vector.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from braintrace._quant import build_spec, relative_distortion
        >>> spec = build_spec(jax.random.PRNGKey(0), 4096, bits=4)
        >>> x = jax.random.normal(jax.random.PRNGKey(1), (2, 4096))
        >>> bool((relative_distortion(x, spec) < 0.25).all())
        True
    """
    reconstructed = decode(encode(values, spec), spec)
    error = jnp.linalg.norm(values - reconstructed, axis=-1)
    norms = jnp.linalg.norm(values, axis=-1)
    return error / jnp.where(norms > 0.0, norms, 1.0)
