# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""08 · vjp_method='multi-step' on delayed-match-to-sample.

Multi-step VJP lets pp_prop combine gradients computed at multiple time steps
(partial L^{t'}/partial h^{t-k}). Useful when the task has a sparse target signal
(one label per sequence) and we still want temporal credit assignment on top
of the eligibility-trace diagonal approximation.
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


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 40, plot: bool = True) -> Dict:
    n_in = 16
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=n_in, n_rec=64, n_out=2)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels,
                decay_or_rank=0.97,
                vjp_method="multi-step",
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_dms_spikes(
                num_step=num_step, num_batch=batch_size, n_in=n_in, fr_hz=80.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[08-vjp-multi] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="08 · vjp_method='multi-step'")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
