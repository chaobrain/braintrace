# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.

import json

import jax
import jax.numpy as jnp
import pytest

import turboquant_state_study as study


@pytest.fixture(scope='module')
def tensors():
    return study.collect_state(neurons=512, degree=8, batch_size=4, steps=4)


def test_collect_state_returns_named_live_tensors(tensors):
    names = {tensor.name for tensor in tensors}
    assert any(name.startswith('etrace_x') for name in names)
    assert any(name.startswith('etrace_df') for name in names)
    assert 'recurrent_csr_values' in names
    assert 'feedforward_dense' in names


def test_collected_tensors_are_finite_and_sized(tensors):
    for tensor in tensors:
        assert jnp.all(jnp.isfinite(tensor.values))
        assert tensor.mebibytes == pytest.approx(tensor.values.size * 4 / 2**20)


def test_flatten_picks_the_widest_axis():
    values = jnp.zeros((4, 512, 3))
    flat = study._flatten_to_vectors(values, 64)
    assert flat.shape == (12, 512)


def test_flatten_rejects_a_block_wider_than_the_axis():
    assert study._flatten_to_vectors(jnp.zeros((4, 32)), 64) is None


def test_distortion_improves_with_bit_width(tensors):
    target = next(t for t in tensors if t.name == 'feedforward_dense')
    records = study.measure_distortion(target)
    by_bits = {}
    for record in records:
        if record['block'] == 64:
            by_bits[record['bits']] = record['relative_error']
    assert by_bits[2] > by_bits[3] > by_bits[4]


def test_compressed_size_tracks_the_bit_budget(tensors):
    target = next(t for t in tensors if t.name == 'feedforward_dense')
    for record in study.measure_distortion(target):
        expected = target.mebibytes * record['bits'] / 32.0
        assert record['compressed_mib'] == pytest.approx(expected)


def _error_at(tensor, block, bits=4):
    return next(
        r['relative_error']
        for r in study.measure_distortion(tensor)
        if r['bits'] == bits and r['block'] == block
    )


def test_rotation_is_neutral_on_gaussian_initialised_weights(tensors):
    weight = next(t for t in tensors if t.name == 'feedforward_dense')
    assert _error_at(weight, 1) / _error_at(weight, 256) == pytest.approx(1.0, abs=0.1)


def test_rotation_helps_a_heavy_tailed_vector():
    key = jax.random.PRNGKey(11)
    magnitude, sign = jax.random.split(key)
    heavy = jnp.exp(jax.random.normal(magnitude, (8, 2048)) * 3.0) * jnp.sign(
        jax.random.normal(sign, (8, 2048))
    )
    tensor = study.StateTensor('heavy_tailed', heavy)
    assert _error_at(tensor, 1) / _error_at(tensor, 256) > 1.5


def test_rotation_does_not_help_a_concentrated_vector():
    key = jax.random.PRNGKey(12)
    concentrated = jnp.ones((8, 2048)) + jax.random.normal(key, (8, 2048)) * 0.01
    tensor = study.StateTensor('concentrated', concentrated)
    assert _error_at(tensor, 1) < _error_at(tensor, 256)


def test_conversion_throughput_reports_three_kernels():
    rates = study.measure_conversion_throughput(elements=2**20)
    assert set(rates) == {
        'float32_elementwise',
        'int8_elementwise',
        'int8_widen_to_float32',
    }
    assert all(rate > 0.0 for rate in rates.values())


def test_main_emits_schema_versioned_json(tmp_path, capsys):
    destination = tmp_path / 'study.json'
    exit_code = study.main(
        [
            '--neurons', '256', '--degree', '8', '--batch-size', '4',
            '--steps', '3', '--json-output', str(destination),
        ]
    )
    assert exit_code == 0
    payload = json.loads(destination.read_text(encoding='utf-8'))
    assert payload['schema_version'] == 1
    assert payload['total_float32_mib'] > 0.0
    assert json.loads(capsys.readouterr().out)['config']['neurons'] == 256
