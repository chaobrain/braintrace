# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""13 · Sweep decay_or_rank on LIF integrator.

IODimVjpAlgorithm accepts decay_or_rank as either a float (exponential-smoothing
decay, 0 < alpha < 1) or an int (approximation rank). The two parameterisations
are duals: num_rank = 2/(1-decay) - 1. This file trains multiple models, one
per value, and plots their final-epoch losses side-by-side.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict, List

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


def _train_once(decay_or_rank, n_epochs, batch_size, num_step):
    model = Net(n_in=1, n_rec=48, n_out=1)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=1e-3)
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def train_step(inputs, targets):
        return _shared.online_train_epoch(
            model, opt, inputs, targets,
            loss_fn=lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
            decay_or_rank=decay_or_rank,
        )

    losses = []
    for epoch in range(n_epochs):
        xs, ys = _shared.make_integrator_spikes(
            num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
        )
        losses.append(float(train_step(xs, ys)))
    return losses


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 50, plot: bool = True) -> Dict:
    sweep_values = [0.9, 0.95, 0.99, 3, 10, 40]
    with brainstate.environ.context(dt=1.0 * u.ms):
        all_losses: Dict[str, List[float]] = {}
        for val in sweep_values:
            label = f"decay={val}" if isinstance(val, float) else f"rank={val}"
            all_losses[label] = _train_once(val, n_epochs, batch_size, num_step)
            print(f"[13-sweep] {label}  final_loss={all_losses[label][-1]:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, curve in all_losses.items():
            ax.plot(curve, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("13 · decay_or_rank sweep (pp_prop)")
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    for curve in all_losses.values():
        assert jnp.isfinite(jnp.asarray(curve[-1]))
    first_key = next(iter(all_losses))
    return {"losses": all_losses[first_key], "sweep": all_losses}


if __name__ == "__main__":
    main()
