# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""07 · vjp_method='single-step' on LIF integrator.

Demonstrates pp_prop's default VJP mode. The VJP of the loss w.r.t. hidden
state is computed at each current time step only (no lookback), which keeps
the online cost small and is the mode used when each step provides its own
target signal.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 50, plot: bool = True) -> Dict:
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=1, n_rec=48, n_out=1)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, targets):
            return _shared.online_train_epoch(
                model, opt, inputs, targets,
                loss_fn=lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
                decay_or_rank=0.95,
                vjp_method="single-step",
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_integrator_spikes(
                num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[07-vjp-single] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="07 · vjp_method='single-step'")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
