# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Shared data generators and thin training helpers for examples/drtrl/*.py."""

from __future__ import annotations

from typing import Callable, Tuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np


def make_integrator_batch(
    num_step: int = 25,
    num_batch: int = 64,
    mean: float = 0.025,
    scale: float = 0.01,
    dt: float = 0.04,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Cumulative-sum regression task (continuous-valued)."""
    rng = np.random.default_rng(seed)
    bias_sample = rng.standard_normal((1, num_batch, 1)).astype(np.float32)
    bias = mean * 2.0 * (bias_sample - 0.5)
    noise = (scale / np.sqrt(dt)) * rng.standard_normal((num_step, num_batch, 1)).astype(np.float32)
    inputs = bias + noise
    targets = np.cumsum(inputs, axis=0)
    return jnp.asarray(inputs), jnp.asarray(targets)


def make_copy_batch(
    time_lag: int,
    batch_size: int,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Copying-memory task. 10 input symbols (0..8 content, 9 = cue)."""
    rng = np.random.default_rng(seed)
    seq_length = time_lag + 20
    ids = np.zeros((batch_size, seq_length), dtype=np.int32)
    ids[:, :10] = rng.integers(1, 9, (batch_size, 10))
    ids[:, -10:] = 9
    one_hot = np.zeros((batch_size, seq_length, 10), dtype=np.float32)
    for b in range(batch_size):
        one_hot[b, np.arange(seq_length), ids[b]] = 1.0
    x = jnp.asarray(np.transpose(one_hot, (1, 0, 2)))  # (T, B, 10)
    y = jnp.asarray(ids[:, :10].T)  # (10, B)
    return x, y


def make_xor_batch(
    seq_len: int,
    delay: int,
    batch_size: int,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Delayed XOR. Two bits shown at t=0 and t=delay, answer needed at t=seq_len-1."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=(batch_size,)).astype(np.float32)
    b = rng.integers(0, 2, size=(batch_size,)).astype(np.float32)
    x = np.zeros((seq_len, batch_size, 2), dtype=np.float32)
    x[0, :, 0] = a
    x[delay, :, 1] = b
    y = (a.astype(np.int32) ^ b.astype(np.int32)).astype(np.int32)
    return jnp.asarray(x), jnp.asarray(y)


def make_sine_batch(
    num_step: int,
    batch_size: int,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Random-frequency sine continuation regression."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.5, 2.5, size=(batch_size,)).astype(np.float32)
    t = np.arange(num_step, dtype=np.float32)[:, None] / 10.0
    signal = np.sin(2.0 * np.pi * freqs[None, :] * t)
    x = signal[:, :, None].astype(np.float32)
    y = np.roll(signal, -1, axis=0)[:, :, None].astype(np.float32)
    return jnp.asarray(x), jnp.asarray(y)


def make_char_batches(
    text: str,
    seq_len: int,
    batch_size: int,
    seed: int | None = None,
) -> Tuple[str, jnp.ndarray, jnp.ndarray]:
    """Character-level batch from a single corpus string."""
    vocab = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    data = np.asarray([char2idx[c] for c in text], dtype=np.int32)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - seq_len - 1, size=(batch_size,))
    x_ids = np.stack([data[s : s + seq_len] for s in starts], axis=1)  # (T, B)
    y_ids = np.stack([data[s + 1 : s + 1 + seq_len] for s in starts], axis=1)
    one_hot = np.eye(len(vocab), dtype=np.float32)[x_ids]
    return ''.join(vocab), jnp.asarray(one_hot), jnp.asarray(y_ids)


def accumulate_grads(
    weights,
    step_grad_fn: Callable,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> Tuple[dict, jnp.ndarray]:
    """Scan ``step_grad_fn`` over (inputs, targets) accumulating gradients."""
    init = jax.tree.map(lambda a: jnp.zeros_like(a), {k: v.value for k, v in weights.items()})

    def body(prev_grads, pair):
        inp, tar = pair
        cur_grads, loss = step_grad_fn(inp, tar)
        next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
        return next_grads, loss

    return brainstate.transform.scan(body, init, (inputs, targets))
