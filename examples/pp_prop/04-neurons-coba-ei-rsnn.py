# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""04 · Dale-law E/I recurrent SNN on Poisson-encoded digits.

Excitatory neurons emit positive recurrent weights, inhibitory neurons emit
negative ones (soft Dale, enforced by initialisation). pp_prop trains the
full recurrent matrix without violating the sign constraint because the
signs are fixed by the initialiser, not by the gradient step.
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
    def __init__(self, n_in: int, n_exc: int, n_inh: int, n_out: int):
        super().__init__()
        self.cell = _shared.COBAEICell(n_in=n_in, n_exc=n_exc, n_inh=n_inh)
        self.readout = _shared.LeakyReadout(n_rec=n_exc + n_inh, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 5, batch_size: int = 32, num_step: int = 25, plot: bool = True) -> Dict:
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=64, n_exc=48, n_inh=16, n_out=len(digits))
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels, decay_or_rank=0.95,
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_poisson_mnist(
                num_step=num_step, num_batch=batch_size, rate_hz=80.0, dt=1e-3,
                digits=digits, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[04-coba-ei] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="04 · COBA E/I RSNN (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
