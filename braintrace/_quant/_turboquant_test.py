# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.

import jax
import jax.numpy as jnp
import pytest

from braintrace._quant import (
    build_spec,
    decode,
    encode,
    relative_distortion,
)
from braintrace._quant._lloydmax import gaussian_codebook_scale

DIM = 4096


def _gaussian(rows=8, dim=DIM, seed=1):
    return jax.random.normal(jax.random.PRNGKey(seed), (rows, dim))


def test_build_spec_rejects_indivisible_dim():
    with pytest.raises(ValueError, match='multiple of block'):
        build_spec(jax.random.PRNGKey(0), 100, block=64)


def test_build_spec_rejects_untabulated_width():
    with pytest.raises(ValueError, match='stage-one width'):
        build_spec(jax.random.PRNGKey(0), 256, bits=9)


def test_encode_rejects_shape_mismatch():
    spec = build_spec(jax.random.PRNGKey(0), 256)
    with pytest.raises(ValueError, match='does not match spec dim'):
        encode(jnp.zeros((2, 128)), spec)


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_scalar_stage_matches_lloydmax_theory(bits):
    spec = build_spec(jax.random.PRNGKey(0), DIM, bits=bits, use_qjl=False)
    measured = float(relative_distortion(_gaussian(), spec).mean())
    assert measured == pytest.approx(gaussian_codebook_scale(bits), rel=0.05)


def test_distortion_shrinks_with_bit_budget():
    errors = [
        float(
            relative_distortion(
                _gaussian(),
                build_spec(jax.random.PRNGKey(0), DIM, bits=bits, use_qjl=False),
            ).mean()
        )
        for bits in (2, 3, 4)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_zero_vectors_round_trip_to_zero():
    spec = build_spec(jax.random.PRNGKey(0), DIM)
    reconstructed = decode(encode(jnp.zeros((3, DIM)), spec), spec)
    assert jnp.all(reconstructed == 0.0)


def test_norm_is_preserved_within_distortion():
    spec = build_spec(jax.random.PRNGKey(0), DIM, bits=4, use_qjl=False)
    values = _gaussian() * jnp.asarray([[1.0], [50.0], [0.02]] * 2 + [[7.0], [7.0]])
    reconstructed = decode(encode(values, spec), spec)
    ratio = jnp.linalg.norm(reconstructed, axis=-1) / jnp.linalg.norm(values, axis=-1)
    assert jnp.all(jnp.abs(ratio - 1.0) < 0.05)


def _inner_product_error(spec, values, queries):
    estimate = jnp.einsum('ij,ij->i', decode(encode(values, spec), spec), queries)
    return estimate - jnp.einsum('ij,ij->i', values, queries)


def test_qjl_stage_reduces_inner_product_bias_at_matched_stage_one():
    values = _gaussian(rows=256)
    queries = _gaussian(rows=256, seed=2)
    scalar = build_spec(jax.random.PRNGKey(0), DIM, bits=3, use_qjl=False)
    corrected = build_spec(jax.random.PRNGKey(0), DIM, bits=4, use_qjl=True)
    scalar_error = _inner_product_error(scalar, values, queries)
    corrected_error = _inner_product_error(corrected, values, queries)
    assert abs(float(jnp.mean(corrected_error))) < abs(float(jnp.mean(scalar_error)))
    assert float(jnp.sqrt(jnp.mean(corrected_error**2))) < float(
        jnp.sqrt(jnp.mean(scalar_error**2))
    )


def test_scalar_stage_wins_at_matched_total_budget():
    values = _gaussian(rows=256)
    queries = _gaussian(rows=256, seed=2)
    scalar = build_spec(jax.random.PRNGKey(0), DIM, bits=4, use_qjl=False)
    split = build_spec(jax.random.PRNGKey(0), DIM, bits=4, use_qjl=True)
    assert float(jnp.sqrt(jnp.mean(_inner_product_error(scalar, values, queries) ** 2))) < float(
        jnp.sqrt(jnp.mean(_inner_product_error(split, values, queries) ** 2))
    )


def test_rotation_rescues_a_coordinate_spike():
    spiky = jnp.zeros((4, DIM)).at[:, :8].set(10.0)
    spiky = spiky + jax.random.normal(jax.random.PRNGKey(7), (4, DIM)) * 0.01
    unrotated = build_spec(jax.random.PRNGKey(0), DIM, bits=4, block=1, use_qjl=False)
    rotated = build_spec(jax.random.PRNGKey(0), DIM, bits=4, block=256, use_qjl=False)
    assert float(relative_distortion(spiky, rotated).mean()) < float(
        relative_distortion(spiky, unrotated).mean()
    )


def test_encode_is_jit_compatible():
    spec = build_spec(jax.random.PRNGKey(0), DIM)
    values = _gaussian()
    round_trip = jax.jit(lambda x: decode(encode(x, spec), spec))
    assert jnp.allclose(
        round_trip(values), decode(encode(values, spec), spec), atol=1e-5
    )


def test_code_is_byte_sized():
    spec = build_spec(jax.random.PRNGKey(0), DIM)
    code = encode(_gaussian(), spec)
    assert code.indices.dtype == jnp.uint8
    assert code.signs.dtype == jnp.uint8
    assert code.norms.shape == (8, 1)
