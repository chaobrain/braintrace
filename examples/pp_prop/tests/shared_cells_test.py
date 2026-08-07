"""Unit tests for examples/pp_prop/_shared.py SNN cells."""

import importlib.util
import pathlib

import brainstate
import jax.numpy as jnp
import brainunit as u


def _load_shared():
    spec = importlib.util.spec_from_file_location(
        "_pp_prop_shared",
        pathlib.Path(__file__).resolve().parents[1] / "_shared.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lif_cell_returns_spikes():
    m = _load_shared()
    with brainstate.environ.context(dt=1.0 * u.ms):
        cell = m.LIFCell(n_in=4, n_rec=8)
        brainstate.nn.init_all_states(cell)
        spk = cell(jnp.ones((4,), dtype=jnp.float32))
    assert spk.shape == (8,)


def test_alif_cell_runs():
    m = _load_shared()
    with brainstate.environ.context(dt=1.0 * u.ms):
        cell = m.ALIFCell(n_in=4, n_rec=8)
        brainstate.nn.init_all_states(cell)
        spk = cell(jnp.ones((4,), dtype=jnp.float32))
    assert spk.shape == (8,)


def test_leaky_readout_returns_rate():
    m = _load_shared()
    with brainstate.environ.context(dt=1.0 * u.ms):
        readout = m.LeakyReadout(n_rec=8, n_out=3)
        brainstate.nn.init_all_states(readout)
        out = readout(jnp.ones((8,), dtype=jnp.float32))
    assert out.shape == (3,)
