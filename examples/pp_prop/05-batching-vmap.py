# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""05 · Batching via brainstate.nn.Vmap(vmap_states='new').

The network and the pp_prop algorithm are defined unbatched, then replicated
across the batch dimension via vmap_new_states. pp_prop's per-rule init is
aware of batching and allocates batched eligibility traces automatically.
This is the default batching path used by examples 01-04.
"""

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

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
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_integrator_spikes(
                num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[05-vmap] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="05 · vmap batching (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
