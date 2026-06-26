# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""03 · GIF neurons with heterogeneous tau_I2 on a working-memory recall task.

GIF's slow adaptation current gives per-neuron memory timescales. pp_prop
tracks the trace through the slow state via the same diagonal approximation
it uses for membrane voltage.
"""

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import brainunit as u

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.GIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 5, batch_size: int = 32, num_step: int = 30, plot: bool = True) -> Dict:
    n_in = 8
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=n_in, n_rec=64, n_out=n_in)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, targets_seq):
            return _shared.online_train_epoch(
                model, opt, inputs, targets_seq,
                loss_fn=lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
                decay_or_rank=0.98,
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_memory_pattern(
                num_step=num_step, num_batch=batch_size, n_in=n_in, cue_frac=0.2, seed=epoch,
            )
            ys_seq = jnp.broadcast_to(ys[None], (num_step, batch_size, n_in))
            losses.append(float(train_step(xs, ys_seq)))
            print(f"[03-gif-memory] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="03 · GIF working-memory (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
