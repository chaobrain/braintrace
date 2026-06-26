# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""10 · Low-rank recurrent weights via braintrace.lora_matmul.

Parameterise the recurrent matrix as W = alpha * B @ A with rank r << n_rec.
pp_prop dispatches to the LoRA ETP primitive etp_lora_mm_p, which registers
its own xy_to_dw / yw_to_w rules so the eligibility trace propagates through
the low-rank factors rather than a dense W.
"""

import pathlib
import sys
from typing import Dict

import brainpy.state
import brainstate
import braintools
import jax.numpy as jnp
import brainunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class LoRARecLIFCell(brainstate.nn.Module):
    """LIF cell: dense ff Linear + low-rank recurrence via braintrace.lora_matmul."""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        rank: int = 8,
        alpha: float = 1.0,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
    ):
        super().__init__()
        self.alpha = alpha
        self.neu = brainpy.state.LIF(
            n_rec, R=1. * u.ohm, tau=tau_mem, V_th=V_th,
            V_reset=0. * u.mV, V_rest=0. * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        self.syn = brainpy.state.Expon(
            n_rec, tau=tau_syn,
            g_initializer=braintools.init.ZeroInit(unit=u.mA),
        )
        ff_init = braintools.init.KaimingNormal(2.0, unit=u.mA)
        self.ff = braintrace.nn.Linear(
            n_in, n_rec, w_init=ff_init,
            b_init=braintools.init.ZeroInit(unit=u.mA),
        )
        # LoRA factors B (n_rec, rank), A (rank, n_rec); carry mA on one side
        b_init = braintools.init.KaimingNormal(1.0, unit=u.mA)
        a_init = braintools.init.KaimingNormal(1.0)  # dimensionless
        self.B = brainstate.ParamState(b_init((n_rec, rank)))
        self.A = brainstate.ParamState(a_init((rank, n_rec)))

    def update(self, x):
        last_spk = self.neu.get_spike()
        ff_current = self.ff(x)
        rec_current = braintrace.lora_matmul(
            last_spk, self.B.value, self.A.value, alpha=self.alpha
        )
        g = self.syn(ff_current + rec_current)
        self.neu(g)
        return self.neu.get_spike()


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int, rank: int):
        super().__init__()
        self.cell = LoRARecLIFCell(n_in=n_in, n_rec=n_rec, rank=rank)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 25, plot: bool = True) -> Dict:
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=64, n_rec=96, n_out=len(digits), rank=8)
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
                num_step=num_step, num_batch=batch_size, digits=digits, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[10-lora] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="10 · LoRA recurrence (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
