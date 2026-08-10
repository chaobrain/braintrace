# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Backend-portability probe for the TurboQuant sparse pp-prop conclusions.

Two conclusions drawn on the XLA CPU backend are backend-specific and are
re-derived here on whatever device the process binds. First, whether a narrow
stored representation can be faster, which is decided by the throughput of the
integer-to-float conversion every arithmetic consumer pays. Second, whether the
unrolled hidden-Jacobian contraction adopted for CPU codegen still beats the
``dot_general`` it replaced.

Every kernel is launched ``inner`` times before a single synchronization so the
dispatch latency, which dominates a single small launch on some backends, is
amortized rather than measured. Candidates within a comparison are interleaved
so thermal drift is charged to both.

.. code-block:: python

    >>> from turboquant_gpu_probe import measure_conversion_throughput
    >>> rates = measure_conversion_throughput(elements=2 ** 16)
    >>> rates['int8_widen_to_float32'] > 0.0
    True
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrace._quant import decode_centroids, encode_nearest, lloydmax_codebook

_INNER = 20
_REPS = 15


@dataclass(frozen=True)
class ProbeShapes:
    """Tensor extents the probe benchmarks.

    Attributes
    ----------
    batch : int
        Parallel training examples.
    neurons : int
        Recurrent neuron count.
    state : int
        Hidden states per neuron group.
    nnz : int
        Stored recurrent edges.
    """

    batch: int = 32
    neurons: int = 131072
    state: int = 3
    nnz: int = 1048576

    @property
    def jacobian(self) -> tuple[int, ...]:
        """Return the stacked hidden-Jacobian shape."""
        return (self.batch, self.neurons, self.state, self.state)

    @property
    def trace(self) -> tuple[int, ...]:
        """Return the eligibility-trace shape."""
        return (self.batch, self.neurons, self.state)


def _timed(fn: Callable, args: Sequence, inner: int) -> float:
    output = None
    started = time.perf_counter()
    for _ in range(inner):
        output = fn(*args)
    jax.block_until_ready(output)
    return (time.perf_counter() - started) / inner


def paired_bench(
    candidates: dict, reps: int = _REPS, inner: int = _INNER
) -> dict[str, float]:
    """Interleave candidates and return their median milliseconds per call.

    Parameters
    ----------
    candidates : dict
        Maps a label to a ``(callable, args)`` pair.
    reps : int
        Interleaved rounds contributing to the median.
    inner : int
        Launches queued per round before synchronizing.

    Returns
    -------
    dict
        Median wall-clock milliseconds per call, keyed by label.
    """
    for fn, args in candidates.values():
        jax.block_until_ready(fn(*args))
    samples: dict[str, list[float]] = {name: [] for name in candidates}
    for _ in range(reps):
        for name, (fn, args) in candidates.items():
            samples[name].append(_timed(fn, args, inner))
    return {name: statistics.median(v) * 1e3 for name, v in samples.items()}


def measure_conversion_throughput(elements: int = 2**26) -> dict[str, float]:
    """Return elementwise kernel throughput in Gelem/s by stored width.

    Parameters
    ----------
    elements : int
        Vector length driving each kernel.

    Returns
    -------
    dict
        Throughput for float32, int8 without widening, and int8 widened.
    """
    floats = jax.random.normal(jax.random.PRNGKey(0), (elements,), jnp.float32)
    narrow = (floats * 20.0).astype(jnp.int8)
    milliseconds = paired_bench({
        'float32_elementwise': (jax.jit(lambda x: x * 1.0001), (floats,)),
        'int8_elementwise': (jax.jit(lambda x: x + jnp.int8(1)), (narrow,)),
        'int8_widen_to_float32': (
            jax.jit(lambda x: x.astype(jnp.float32) * 1.0001), (narrow,)
        ),
    })
    return {
        name: elements / (value * 1e-3) / 1e9 for name, value in milliseconds.items()
    }


def contract_unrolled(jacobian: jax.Array, trace: jax.Array) -> jax.Array:
    """Contract the trailing Jacobian axis by explicit accumulation.

    Parameters
    ----------
    jacobian : jax.Array
        Array whose last two axes are ``(state, state)``.
    trace : jax.Array
        Array whose last axis is ``state``.

    Returns
    -------
    jax.Array
        Contraction matching ``jnp.einsum('...ij,...j->...i', ...)``.
    """
    contracted = jacobian[..., 0] * trace[..., 0:1]
    for index in range(1, trace.shape[-1]):
        contracted = contracted + jacobian[..., index] * trace[..., index:index + 1]
    return contracted


def _pack_nibbles(values: jax.Array, codebook: jax.Array) -> jax.Array:
    codes = encode_nearest(values.reshape(-1), codebook).astype(jnp.uint8)
    low, high = jnp.split(codes, 2)
    return jnp.bitwise_or(low, jnp.left_shift(high, 4))


def _unpack_nibbles(
    packed: jax.Array, codebook: jax.Array, shape: tuple[int, ...]
) -> jax.Array:
    low = jnp.bitwise_and(packed, jnp.uint8(0x0F))
    high = jnp.right_shift(packed, jnp.uint8(4))
    halves = [decode_centroids(low, codebook), decode_centroids(high, codebook)]
    return jnp.concatenate(halves).reshape(shape)


def measure_contraction(shapes: ProbeShapes = ProbeShapes()) -> dict[str, float]:
    """Return median milliseconds for each hidden-Jacobian contraction variant.

    Parameters
    ----------
    shapes : ProbeShapes
        Tensor extents to benchmark.

    Returns
    -------
    dict
        Timings for the einsum and unrolled forms at float32, int8 and packed
        4-bit stored width.
    """
    jacobian_key, trace_key = jax.random.split(jax.random.PRNGKey(1))
    jacobian = jax.random.normal(jacobian_key, shapes.jacobian, jnp.float32)
    trace = jax.random.normal(trace_key, shapes.trace, jnp.float32)
    scale = jnp.float32(jnp.max(jnp.abs(jacobian)) / 127.0)
    narrow = (jacobian / scale).astype(jnp.int8)
    codebook = lloydmax_codebook(4, jnp.std(jacobian.reshape(-1)))
    packed = _pack_nibbles(jacobian, codebook)
    return paired_bench({
        'einsum_float32': (
            jax.jit(lambda a, b: jnp.einsum('...ij,...j->...i', a, b)),
            (jacobian, trace),
        ),
        'unrolled_float32': (jax.jit(contract_unrolled), (jacobian, trace)),
        'unrolled_int8_stored': (
            jax.jit(lambda a, b, s: contract_unrolled(a.astype(jnp.float32) * s, b)),
            (narrow, trace, scale),
        ),
        'unrolled_int4_stored': (
            jax.jit(
                lambda p, b, c: contract_unrolled(
                    _unpack_nibbles(p, c, shapes.jacobian), b
                )
            ),
            (packed, trace, codebook),
        ),
    })


def measure_edge_buffers(shapes: ProbeShapes = ProbeShapes()) -> dict[str, float]:
    """Return median milliseconds for the two dominant edge-buffer kernels.

    Parameters
    ----------
    shapes : ProbeShapes
        Tensor extents to benchmark.

    Returns
    -------
    dict
        Timings for the batch reduction and the gather at both stored widths.
    """
    edge_key, index_key = jax.random.split(jax.random.PRNGKey(2))
    edges = jax.random.normal(edge_key, (shapes.nnz, shapes.batch), jnp.float32)
    scale = jnp.float32(jnp.max(jnp.abs(edges)) / 127.0)
    narrow_edges = (edges / scale).astype(jnp.int8)
    indices = jax.random.randint(index_key, (shapes.nnz,), 0, shapes.neurons)
    source = jax.random.normal(edge_key, (shapes.neurons, shapes.batch), jnp.float32)
    narrow_source = (source / scale).astype(jnp.int8)
    reduction = paired_bench({
        'batch_reduce_float32': (jax.jit(lambda x: jnp.sum(x, axis=1)), (edges,)),
        'batch_reduce_int8_stored': (
            jax.jit(lambda x, s: jnp.sum(x.astype(jnp.float32) * s, axis=1)),
            (narrow_edges, scale),
        ),
    })
    gather = paired_bench({
        'gather_float32': (jax.jit(lambda x, i: x[i]), (source, indices)),
        'gather_int8_stored': (
            jax.jit(lambda x, i, s: x[i].astype(jnp.float32) * s),
            (narrow_source, indices, scale),
        ),
    })
    return {**reduction, **gather}


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neurons', type=int, default=131072)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nnz', type=int, default=1048576)
    parser.add_argument('--elements', type=int, default=2**26)
    parser.add_argument('--json-output', type=pathlib.Path, default=None)
    parser.add_argument('--require-gpu', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every probe and emit one schema-versioned JSON document."""
    options = _parse(argv)
    device = jax.devices()[0]
    if options.require_gpu and device.platform != 'gpu':
        print(f'backend is {device.platform}, not gpu', file=sys.stderr)
        return 1
    shapes = ProbeShapes(
        batch=options.batch_size, neurons=options.neurons, nnz=options.nnz
    )
    payload = {
        'schema_version': 1,
        'backend': jax.default_backend(),
        'device': getattr(device, 'device_kind', str(device)),
        'shapes': {'batch': shapes.batch, 'neurons': shapes.neurons, 'nnz': shapes.nnz},
        'conversion_gelem_per_second': measure_conversion_throughput(options.elements),
        'contraction_milliseconds': measure_contraction(shapes),
        'edge_buffer_milliseconds': measure_edge_buffers(shapes),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if options.json_output is not None:
        options.json_output.parent.mkdir(parents=True, exist_ok=True)
        options.json_output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
