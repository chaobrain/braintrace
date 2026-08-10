# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.

import jax
import jax.numpy as jnp
import pytest

from braintrace._quant import (
    block_hadamard_matrix,
    rotate_blocks,
    sign_diagonal,
    unrotate_blocks,
)


@pytest.mark.parametrize('block', [1, 2, 8, 64])
def test_hadamard_matrix_is_orthonormal(block):
    matrix = block_hadamard_matrix(block)
    assert matrix.shape == (block, block)
    assert jnp.allclose(matrix @ matrix.T, jnp.eye(block), atol=1e-5)


@pytest.mark.parametrize('block', [0, 3, 6, -4])
def test_hadamard_matrix_rejects_non_power_of_two(block):
    with pytest.raises(ValueError, match='power of two'):
        block_hadamard_matrix(block)


def test_sign_diagonal_is_bipolar():
    signs = sign_diagonal(jax.random.PRNGKey(0), 512)
    assert set(jnp.unique(signs).tolist()) <= {-1.0, 1.0}


def test_sign_diagonal_rejects_empty():
    with pytest.raises(ValueError, match='positive'):
        sign_diagonal(jax.random.PRNGKey(0), 0)


@pytest.mark.parametrize('block', [2, 16, 64])
def test_rotation_round_trips(block):
    values = jax.random.normal(jax.random.PRNGKey(1), (5, 256))
    signs = sign_diagonal(jax.random.PRNGKey(2), 256)
    restored = unrotate_blocks(rotate_blocks(values, signs, block), signs, block)
    assert jnp.allclose(restored, values, atol=1e-4)


def test_rotation_preserves_norm():
    values = jax.random.normal(jax.random.PRNGKey(3), (3, 128))
    signs = sign_diagonal(jax.random.PRNGKey(4), 128)
    rotated = rotate_blocks(values, signs, 64)
    assert jnp.allclose(
        jnp.linalg.norm(rotated, axis=-1), jnp.linalg.norm(values, axis=-1), atol=1e-4
    )


def test_rotation_rejects_indivisible_axis():
    values = jnp.zeros((2, 100))
    signs = sign_diagonal(jax.random.PRNGKey(5), 100)
    with pytest.raises(ValueError, match='multiple of block'):
        rotate_blocks(values, signs, 64)


def test_rotation_spreads_a_spike():
    values = jnp.zeros((1, 128)).at[0, 0].set(1.0)
    signs = sign_diagonal(jax.random.PRNGKey(6), 128)
    rotated = rotate_blocks(values, signs, 64)
    assert jnp.count_nonzero(rotated) == 64
    assert float(jnp.max(jnp.abs(rotated))) == pytest.approx(1.0 / 8.0, rel=1e-4)
