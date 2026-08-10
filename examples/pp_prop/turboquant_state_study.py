# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""TurboQuant compression study for the sparse pp-prop benchmark state.

Answers two questions with measurements rather than assertion. First, how much
of the benchmark's live state TurboQuant removes, and whether the randomized
Hadamard rotation earns its cost on these tensors as opposed to the synthetic
Gaussians the codec is derived for. Second, whether a narrow stored
representation can ever be faster on the host backend, which is decided by the
throughput of the integer-to-float conversion the consumer has to pay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterator

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrace._quant import build_spec, decode, encode, relative_distortion

_BLOCKS = (1, 16, 64, 256)
_BITS = (2, 3, 4)


@dataclass(frozen=True)
class StateTensor:
    """One named live tensor pulled out of a running pp-prop learner.

    Attributes
    ----------
    name : str
        Human-readable origin of the tensor.
    values : jax.Array
        Float32 contents.
    """

    name: str
    values: jax.Array

    @property
    def mebibytes(self) -> float:
        """Return the float32 footprint in MiB."""
        return self.values.size * 4 / 2**20


def _load_example() -> Any:
    path = pathlib.Path(__file__).with_name('15-sparse-temporal-learning.py')
    spec = importlib.util.spec_from_file_location('_tq_study_example', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the sparse learning example from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_state(neurons: int, degree: int, batch_size: int, steps: int) -> list[StateTensor]:
    """Run a few pp-prop steps and return the resulting live state tensors.

    Parameters
    ----------
    neurons : int
        Recurrent neuron count.
    degree : int
        Stored recurrent edges per neuron.
    batch_size : int
        Parallel training examples.
    steps : int
        Neural timesteps driven before the state is read back.

    Returns
    -------
    list of StateTensor
        Eligibility traces, hidden states, and recurrent edge values.

    Examples
    --------
    .. code-block:: python

        >>> from turboquant_state_study import collect_state
        >>> tensors = collect_state(256, 8, 4, 3)
        >>> all(t.values.ndim >= 1 for t in tensors)
        True
    """
    example = _load_example()
    config = example._RunConfig(
        seed=0, n_epochs=1, batch_size=batch_size, n_rec=neurons, degree=degree,
        n_step=steps, final_window=min(steps, 5), learning_rate=3e-3,
        decay_or_rank=0.95, clip_norm=1.0, sparse_backend='jax_raw',
        recurrent_scale_basis='degree',
    )
    with brainstate.environ.context(dt=1.0 * u.ms):
        experiment = example._build_experiment(config)
        train_batch = example._make_train_batch(experiment, config)
        rng = np.random.default_rng(0)
        spikes = jnp.asarray(
            rng.random((steps, batch_size, 64)) < 0.15, dtype=jnp.float32
        )
        jax.block_until_ready(
            train_batch(spikes, jnp.zeros((batch_size,), dtype=jnp.int32))
        )
        return list(_state_tensors(experiment))


def _state_tensors(experiment: Any) -> Iterator[StateTensor]:
    learner = experiment.learner
    for key, trace in getattr(learner, 'etrace_xs', {}).items():
        yield StateTensor(f'etrace_x{_key_tag(key)}', jnp.asarray(trace.value))
    for key, trace in getattr(learner, 'etrace_dfs', {}).items():
        yield StateTensor(f'etrace_df{_key_tag(key)}', jnp.asarray(trace.value))
    weight = experiment.model.cell.rec_syn.comm.weight.value['weight']
    yield StateTensor('recurrent_csr_values', jnp.asarray(u.get_mantissa(weight)))
    dense = experiment.model.cell.ff_syn.comm.weight.value['weight']
    yield StateTensor('feedforward_dense', jnp.asarray(u.get_mantissa(dense)))


def _key_tag(key: Any) -> str:
    text = '_'.join(str(part) for part in key) if isinstance(key, tuple) else str(key)
    return '[' + text.replace(' ', '')[:48] + ']'


def _flatten_to_vectors(values: jax.Array, block: int) -> jax.Array | None:
    """Reshape ``values`` so that its widest axis is the quantized trailing axis.

    Parameters
    ----------
    values : jax.Array
        Tensor of any rank.
    block : int
        Rotation block order the trailing axis must accommodate.

    Returns
    -------
    jax.Array or None
        Two-dimensional view, or ``None`` when the widest axis cannot host a
        rotation of order ``block``.
    """
    if values.ndim == 1:
        moved = values.reshape(1, -1)
    else:
        widest = int(np.argmax(values.shape))
        moved = jnp.moveaxis(values, widest, -1).reshape(-1, values.shape[widest])
    dim = moved.shape[-1]
    if dim % block or dim < block:
        return None
    return moved


def measure_distortion(tensor: StateTensor) -> list[dict[str, float]]:
    """Report round-trip distortion for every bit width and rotation block.

    Parameters
    ----------
    tensor : StateTensor
        Tensor to compress.

    Returns
    -------
    list of dict
        One record per ``(bits, block)`` pair carrying the mean relative error
        and the compressed footprint in MiB.
    """
    records: list[dict[str, float]] = []
    for bits in _BITS:
        for block in _BLOCKS:
            flat = _flatten_to_vectors(tensor.values, block)
            if flat is None:
                continue
            spec = build_spec(
                jax.random.PRNGKey(0), flat.shape[-1], bits=bits, block=block,
                use_qjl=False,
            )
            error = float(jnp.mean(relative_distortion(flat, spec)))
            records.append(
                {
                    'bits': bits,
                    'block': block,
                    'relative_error': error,
                    'compressed_mib': tensor.mebibytes * bits / 32.0,
                }
            )
    return records


def measure_conversion_throughput(elements: int = 33_554_432) -> dict[str, float]:
    """Measure the host cost of reading float32 versus decoding int8.

    Parameters
    ----------
    elements : int, optional
        Number of array elements exercised by each kernel.

    Returns
    -------
    dict
        Giga-elements per second for a float32 sweep, an int8 sweep, and an
        int8 sweep whose result is widened to float32.
    """
    wide = jax.random.normal(jax.random.PRNGKey(0), (elements // 32, 32), jnp.float32)
    narrow = jnp.round(wide * 10).astype(jnp.int8)
    kernels = {
        'float32_elementwise': (jax.jit(lambda a: a * 1.0001), wide),
        'int8_elementwise': (jax.jit(lambda a: a + jnp.int8(1)), narrow),
        'int8_widen_to_float32': (jax.jit(lambda a: a.astype(jnp.float32) * 0.1), narrow),
    }
    rates = {}
    for name, (kernel, operand) in kernels.items():
        jax.block_until_ready(kernel(operand))
        started = time.perf_counter()
        for _ in range(10):
            result = kernel(operand)
        jax.block_until_ready(result)
        rates[name] = elements * 10 / (time.perf_counter() - started) / 1e9
    return rates


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neurons', type=int, default=131072)
    parser.add_argument('--degree', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--json-output', type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the study and print a schema-versioned JSON report.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments without the executable name.

    Returns
    -------
    int
        Process exit code.
    """
    options = _parse(argv)
    tensors = collect_state(
        options.neurons, options.degree, options.batch_size, options.steps
    )
    payload = {
        'schema_version': 1,
        'config': {
            'neurons': options.neurons,
            'degree': options.degree,
            'batch_size': options.batch_size,
            'steps': options.steps,
        },
        'conversion_throughput_gelem_per_second': measure_conversion_throughput(),
        'tensors': [
            {
                'name': tensor.name,
                'shape': list(tensor.values.shape),
                'float32_mib': tensor.mebibytes,
                'distortion': measure_distortion(tensor),
            }
            for tensor in tensors
        ],
    }
    payload['total_float32_mib'] = sum(t.mebibytes for t in tensors)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if options.json_output is not None:
        options.json_output.parent.mkdir(parents=True, exist_ok=True)
        options.json_output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
