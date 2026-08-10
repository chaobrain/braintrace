# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.

import json

import jax
import jax.numpy as jnp
import pytest

import turboquant_gpu_probe as probe

SMALL = probe.ProbeShapes(batch=2, neurons=256, nnz=1024)


def _bench(candidates):
    return probe.paired_bench(candidates, reps=2, inner=2)


def test_probe_shapes_derive_tensor_extents():
    assert SMALL.jacobian == (2, 256, 3, 3)
    assert SMALL.trace == (2, 256, 3)


def test_paired_bench_times_every_candidate():
    timings = _bench({
        'a': (jax.jit(lambda x: x + 1.0), (jnp.ones((64,)),)),
        'b': (jax.jit(lambda x: x * 2.0), (jnp.ones((64,)),)),
    })
    assert set(timings) == {'a', 'b'}
    assert all(value > 0.0 for value in timings.values())


def test_conversion_throughput_reports_three_widths():
    rates = probe.measure_conversion_throughput(elements=2**14)
    assert set(rates) == {
        'float32_elementwise', 'int8_elementwise', 'int8_widen_to_float32'
    }
    assert all(rate > 0.0 for rate in rates.values())


def test_unrolled_contraction_matches_einsum():
    jacobian_key, trace_key = jax.random.split(jax.random.PRNGKey(7))
    jacobian = jax.random.normal(jacobian_key, (2, 32, 3, 3))
    trace = jax.random.normal(trace_key, (2, 32, 3))
    expected = jnp.einsum('...ij,...j->...i', jacobian, trace)
    assert jnp.allclose(probe.contract_unrolled(jacobian, trace), expected, atol=1e-5)


@pytest.mark.parametrize('num_state', [1, 2, 4])
def test_unrolled_contraction_matches_einsum_for_other_state_counts(num_state):
    key = jax.random.PRNGKey(num_state)
    jacobian_key, trace_key = jax.random.split(key)
    jacobian = jax.random.normal(jacobian_key, (3, 8, num_state, num_state))
    trace = jax.random.normal(trace_key, (3, 8, num_state))
    expected = jnp.einsum('...ij,...j->...i', jacobian, trace)
    assert jnp.allclose(probe.contract_unrolled(jacobian, trace), expected, atol=1e-5)


def test_nibble_round_trip_recovers_the_codebook_approximation():
    values = jax.random.normal(jax.random.PRNGKey(3), (4, 64))
    codebook = probe.lloydmax_codebook(4, jnp.std(values.reshape(-1)))
    packed = probe._pack_nibbles(values, codebook)
    restored = probe._unpack_nibbles(packed, codebook, values.shape)
    assert packed.dtype == jnp.uint8
    assert packed.size == values.size // 2
    assert restored.shape == values.shape
    error = jnp.linalg.norm(restored - values) / jnp.linalg.norm(values)
    assert float(error) < 0.2


def test_contraction_reports_every_stored_width():
    timings = probe.measure_contraction(SMALL)
    assert set(timings) == {
        'einsum_float32', 'unrolled_float32',
        'unrolled_int8_stored', 'unrolled_int4_stored',
    }
    assert all(value > 0.0 for value in timings.values())


def test_edge_buffers_report_both_kernels_at_both_widths():
    timings = probe.measure_edge_buffers(SMALL)
    assert set(timings) == {
        'batch_reduce_float32', 'batch_reduce_int8_stored',
        'gather_float32', 'gather_int8_stored',
    }
    assert all(value > 0.0 for value in timings.values())


def test_main_refuses_to_report_when_gpu_is_required_and_absent():
    if jax.devices()[0].platform == 'gpu':
        pytest.skip('probe binds a gpu, so the refusal path cannot be exercised')
    assert probe.main(['--require-gpu']) == 1


def test_main_emits_schema_versioned_json(tmp_path, capsys):
    destination = tmp_path / 'probe.json'
    exit_code = probe.main([
        '--neurons', '256', '--batch-size', '2', '--nnz', '1024',
        '--elements', '4096', '--json-output', str(destination),
    ])
    assert exit_code == 0
    payload = json.loads(destination.read_text(encoding='utf-8'))
    assert payload['schema_version'] == 1
    assert payload['shapes']['neurons'] == 256
    assert json.loads(capsys.readouterr().out)['backend'] == jax.default_backend()
