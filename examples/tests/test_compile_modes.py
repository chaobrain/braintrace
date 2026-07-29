# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Verify the example SNN cells compile and run under BOTH
``braintrace.compile(vmap=False)`` (batched, internal batch primitive) and
``braintrace.compile(vmap=True)`` (per-sample vmap lanes).

The custom ``GIF`` neuron in ``snn_models.py`` originally defined
``init_state(self)`` without ``batch_size``, so the non-vmap path
(``init_all_states(model, batch_size=B)``) raised ``TypeError``. Every sibling
neuron (LIF/ALIF and brainpy's own Gif) accepts ``batch_size=None, **kwargs``;
this test pins that both modes work for all four SNN cells.
"""
import pathlib
import sys

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import pytest

import braintrace

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1]
for p in (EXAMPLES_DIR, EXAMPLES_DIR / "pp_prop"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import _shared as pp  # noqa: E402  (examples/pp_prop/_shared.py)

B, T, N_IN, N_REC = 4, 6, 3, 8


def _build_net(cell_name):
    if cell_name == "lif":
        cell = pp.LIFCell(n_in=N_IN, n_rec=N_REC)
    elif cell_name == "alif":
        cell = pp.ALIFCell(n_in=N_IN, n_rec=N_REC)
    elif cell_name == "gif":
        cell = pp.GIFCell(n_in=N_IN, n_rec=N_REC)
    elif cell_name == "cobaei":
        cell = pp.COBAEICell(n_in=N_IN, n_exc=N_REC // 2, n_inh=N_REC // 2)
    else:  # pragma: no cover
        raise ValueError(cell_name)

    class SNNNet(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = cell
            self.readout = pp.LeakyReadout(n_rec=N_REC, n_out=2)

        def update(self, x):
            return self.readout(self.cell(x))

    return SNNNet()


@pytest.mark.parametrize("cell_name", ["lif", "alif", "gif", "cobaei"])
@pytest.mark.parametrize("vmap", [False, True])
def test_snn_cell_compiles_and_runs_in_both_modes(cell_name, vmap):
    """compile(vmap=False/True) must build, forward, and back-prop a finite grad."""
    with brainstate.environ.context(dt=1.0 * u.ms):
        xs = jnp.asarray(brainstate.random.bernoulli(0.5, (T, B, N_IN)), dtype=float)
        net = _build_net(cell_name)
        learner = braintrace.compile(
            net, braintrace.pp_prop, xs[0], batch_size=B, vmap=vmap, decay_or_rank=0.95
        )
        weights = net.states(brainstate.ParamState)

        def total_loss(xs):
            def step(carry, x):
                return carry, jnp.mean(jnp.asarray(learner(x)) ** 2)

            _, ls = brainstate.transform.scan(step, None, xs)
            return jnp.sum(ls)

        grads = brainstate.transform.grad(total_loss, weights)(xs)
        flat = jnp.concatenate([jnp.ravel(jnp.asarray(g)) for g in jax.tree.leaves(grads)])
        assert bool(jnp.all(jnp.isfinite(flat))), f"{cell_name} vmap={vmap}: non-finite grad"


def test_gif_neuron_init_state_accepts_batch_size():
    """Regression: the custom GIF neuron's init_state must accept batch_size
    (the non-vmap compile path calls init_all_states(model, batch_size=B))."""
    from snn_models import GIF  # type: ignore

    with brainstate.environ.context(dt=1.0 * u.ms):
        neu = GIF(N_REC)
        brainstate.nn.init_all_states(neu, batch_size=B)
        assert neu.V.value.shape == (B, N_REC)
        assert neu.I2.value.shape == (B, N_REC)
