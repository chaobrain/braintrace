# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Shared SNN data generators, cell wrappers, and train-step helpers for examples/pp_prop."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import brainpy.state
import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u


DEFAULT_DT = 1.0 * u.ms
DEFAULT_SEED = 42


# --- Data generators -----------------------------------------------------


def make_integrator_spikes(
    num_step: int = 50,
    num_batch: int = 32,
    rate_hz: float = 50.0,
    dt: float = 1e-3,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Poisson input spikes. Target = cumulative mean rate (continuous)."""
    rng = np.random.default_rng(seed)
    p = rate_hz * dt
    spikes = (rng.random((num_step, num_batch, 1)) < p).astype(np.float32)
    targets = np.cumsum(spikes, axis=0) / num_step
    return jnp.asarray(spikes), jnp.asarray(targets)


def make_dms_spikes(
    num_step: int = 40,
    num_batch: int = 32,
    n_in: int = 16,
    fr_hz: float = 80.0,
    dt: float = 1e-3,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Delayed match-to-sample: sample window + delay + test window, binary label."""
    rng = np.random.default_rng(seed)
    t_sample = num_step // 4
    t_delay = num_step // 2
    labels = rng.integers(0, 2, size=(num_batch,)).astype(np.int32)
    sample_dir = rng.integers(0, n_in, size=(num_batch,))
    test_dir = np.where(labels == 1, sample_dir, (sample_dir + n_in // 2) % n_in)
    fr = fr_hz * dt
    xs = np.zeros((num_step, num_batch, n_in), dtype=np.float32)
    for b in range(num_batch):
        xs[:t_sample, b, sample_dir[b]] = fr
        xs[t_sample + t_delay:, b, test_dir[b]] = fr
    xs = (rng.random(xs.shape) < xs).astype(np.float32)
    return jnp.asarray(xs), jnp.asarray(labels)


def make_memory_pattern(
    num_step: int = 30,
    num_batch: int = 32,
    n_in: int = 8,
    cue_frac: float = 0.2,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Working memory: binary pattern during cue window, silent delay, recall target at end."""
    rng = np.random.default_rng(seed)
    cue_steps = max(1, int(num_step * cue_frac))
    pattern = rng.integers(0, 2, size=(num_batch, n_in)).astype(np.float32)
    xs = np.zeros((num_step, num_batch, n_in), dtype=np.float32)
    xs[:cue_steps] = pattern[None, :, :]
    return jnp.asarray(xs), jnp.asarray(pattern)


def make_poisson_mnist(
    num_step: int = 25,
    num_batch: int = 32,
    rate_hz: float = 80.0,
    dt: float = 1e-3,
    digits: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Poisson-encoded sklearn 8x8 digits. Falls back to synthetic patterns if sklearn missing."""
    try:
        from sklearn.datasets import load_digits
        data = load_digits()
        imgs = data.images / data.images.max()
        labels = data.target
        mask = np.isin(labels, list(digits))
        imgs = imgs[mask]
        labels = labels[mask]
    except ImportError:
        rng_fallback = np.random.default_rng(seed)
        n_per = 50
        imgs = rng_fallback.random((n_per * len(digits), 8, 8)).astype(np.float32)
        labels = np.repeat(np.arange(len(digits)), n_per)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(num_batch,))
    flat = imgs[idx].reshape(num_batch, 64)
    label_idx = np.array([list(digits).index(int(l)) for l in labels[idx]], dtype=np.int32)
    p = flat[None, :, :] * rate_hz * dt
    spikes = (rng.random((num_step, num_batch, 64)) < p).astype(np.float32)
    return jnp.asarray(spikes), jnp.asarray(label_idx)
