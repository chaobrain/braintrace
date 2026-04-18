"""Unit tests for examples/pp_prop/_shared.py data generators."""

import importlib.util
import pathlib

import jax.numpy as jnp
import numpy as np


def _load_shared():
    spec = importlib.util.spec_from_file_location(
        "_pp_prop_shared",
        pathlib.Path(__file__).resolve().parents[1] / "_shared.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_integrator_spikes_shape_and_values():
    m = _load_shared()
    xs, ys = m.make_integrator_spikes(num_step=25, num_batch=4, rate_hz=50.0, dt=1e-3, seed=0)
    assert xs.shape == (25, 4, 1)
    assert ys.shape == (25, 4, 1)
    assert jnp.all((xs == 0.0) | (xs == 1.0))
    diffs = jnp.diff(ys, axis=0)
    assert jnp.all(diffs >= 0)


def test_make_dms_spikes_shape_and_labels():
    m = _load_shared()
    xs, ys = m.make_dms_spikes(num_step=40, num_batch=8, n_in=16, fr_hz=80.0, dt=1e-3, seed=0)
    assert xs.shape == (40, 8, 16)
    assert ys.shape == (8,)
    assert set(np.unique(np.asarray(ys)).tolist()).issubset({0, 1})


def test_make_memory_pattern_shape():
    m = _load_shared()
    xs, ys = m.make_memory_pattern(num_step=30, num_batch=4, n_in=8, cue_frac=0.2, seed=0)
    assert xs.shape == (30, 4, 8)
    assert ys.shape[-1] == 8


def test_make_poisson_mnist_shape_and_labels():
    m = _load_shared()
    xs, ys = m.make_poisson_mnist(num_step=20, num_batch=8, rate_hz=80.0, dt=1e-3, digits=(0, 1, 2), seed=0)
    assert xs.shape == (20, 8, 64)
    assert ys.shape == (8,)
    assert set(np.unique(np.asarray(ys)).tolist()).issubset({0, 1, 2})
