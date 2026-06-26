# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""12 · Flagship classification --- pp_prop vs BPTT on Poisson-MNIST (10 classes).

Trains two identical LIF RSNNs on the same Poisson-encoded sklearn digits,
one with pp_prop, one with BPTT. Reports per-epoch loss and final accuracy.
Demonstrates that pp_prop tracks BPTT's performance with the O(BI+BO)
memory footprint advertised by the algorithm.
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
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec, ff_scale=4.0, rec_scale=1.0)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _accuracy(outputs_seq, labels):
    mean_out = outputs_seq.mean(axis=0)
    return float(jnp.mean(jnp.argmax(mean_out, axis=-1) == labels))


def _eval(model, inputs, labels):
    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)

    init()
    vmap_model = brainstate.nn.Vmap(model, vmap_states="new")
    outs = brainstate.transform.for_loop(lambda x: vmap_model(x), inputs)
    return _accuracy(outs, labels)


def main(n_epochs: int = 4, batch_size: int = 32, num_step: int = 25, plot: bool = True) -> Dict:
    digits = tuple(range(10))
    n_out = len(digits)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model_pp = Net(n_in=64, n_rec=128, n_out=n_out)
        w_pp = model_pp.states(brainstate.ParamState)
        opt_pp = braintools.optim.Adam(lr=1e-3)
        opt_pp.register_trainable_weights(w_pp)

        @brainstate.transform.jit
        def train_pp(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model_pp, opt_pp, inputs, labels,
                decay_or_rank=0.97, vjp_method="single-step",
            )

        model_bp = Net(n_in=64, n_rec=128, n_out=n_out)
        w_bp = model_bp.states(brainstate.ParamState)
        opt_bp = braintools.optim.Adam(lr=1e-3)
        opt_bp.register_trainable_weights(w_bp)

        @brainstate.transform.jit
        def train_bp(inputs, labels):
            return _shared.bptt_train_epoch_fixed_target(model_bp, opt_bp, inputs, labels)

        pp_losses, bp_losses = [], []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_poisson_mnist(
                num_step=num_step, num_batch=batch_size, digits=digits, seed=epoch,
            )
            pp_losses.append(float(train_pp(xs, ys)))
            bp_losses.append(float(train_bp(xs, ys)))
            print(
                f"[12-flagship] epoch {epoch}  pp_prop={pp_losses[-1]:.4f}  "
                f"bptt={bp_losses[-1]:.4f}"
            )

        xs_e, ys_e = _shared.make_poisson_mnist(
            num_step=num_step, num_batch=batch_size, digits=digits, seed=9999,
        )
        acc_pp = _eval(model_pp, xs_e, ys_e)
        acc_bp = _eval(model_bp, xs_e, ys_e)
        print(f"[12-flagship] final acc  pp_prop={acc_pp:.3f}  bptt={acc_bp:.3f}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(pp_losses, label="pp_prop")
        ax.plot(bp_losses, label="BPTT")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("12 · pp_prop vs BPTT on Poisson-MNIST")
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    assert jnp.isfinite(jnp.asarray(pp_losses[-1])), f"pp_prop loss not finite: {pp_losses[-1]}"
    assert jnp.isfinite(jnp.asarray(bp_losses[-1])), f"BPTT loss not finite: {bp_losses[-1]}"
    # Chance-level accuracy assertion applies only when training is long enough
    # to beat noise; the smoke harness calls with n_epochs=1 and should not
    # gate on accuracy.
    if n_epochs >= 3:
        chance = 1.0 / n_out
        assert acc_pp > chance, f"pp_prop accuracy {acc_pp} <= chance {chance}"
        assert acc_bp > chance, f"BPTT accuracy {acc_bp} <= chance {chance}"
    return {"losses": pp_losses, "bptt_losses": bp_losses, "acc_pp": acc_pp, "acc_bp": acc_bp}


if __name__ == "__main__":
    main()
