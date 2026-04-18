# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""11 · Conv-SNN on Poisson-encoded 8x8 digits via braintrace.nn.Conv2d.

Single Conv2d -> Expon -> LIF -> global-avg-pool -> readout. pp_prop
dispatches to the convolutional ETP primitive etp_conv_p for gradient
computation on the conv kernel.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainpy.state
import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class ConvLIFCell(brainstate.nn.Module):
    """Conv2d feature extractor -> flatten -> Expon -> LIF. Flattening lets
    the hidden group for pp_prop match the conv primitive's post-squeeze
    output shape (the primitive records a 4D shape while Expon/LIF hold 3D,
    so we work in 1D after conv to avoid the mismatch)."""

    def __init__(self, in_shape=(8, 8, 1), out_ch: int = 4, kernel: int = 3):
        super().__init__()
        self.conv = braintrace.nn.Conv2d(
            in_size=in_shape,
            out_channels=out_ch,
            kernel_size=kernel,
            padding="SAME",
            w_init=braintools.init.KaimingNormal(1.0, unit=u.mA),
            b_init=braintools.init.ZeroInit(unit=u.mA),
        )
        h, w = in_shape[0], in_shape[1]
        self.flat_size = h * w * out_ch
        self.out_ch = out_ch
        self.h = h
        self.w = w
        self.syn = brainpy.state.Expon(
            self.flat_size, tau=10.0 * u.ms,
            g_initializer=braintools.init.ZeroInit(unit=u.mA),
        )
        self.neu = brainpy.state.LIF(
            self.flat_size, R=1. * u.ohm, tau=20.0 * u.ms, V_th=1.0 * u.mV,
            V_reset=0. * u.mV, V_rest=0. * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )

    def update(self, x):
        current = self.conv(x)  # (..., h, w, out_ch)
        flat = current.reshape(*current.shape[:-3], self.flat_size)
        g = self.syn(flat)
        self.neu(g)
        spk = self.neu.get_spike()
        return spk.reshape(*spk.shape[:-1], self.h, self.w, self.out_ch)


class Net(brainstate.nn.Module):
    def __init__(self, n_out: int, out_ch: int = 4):
        super().__init__()
        self.cell = ConvLIFCell(in_shape=(8, 8, 1), out_ch=out_ch)
        self.readout = _shared.LeakyReadout(n_rec=out_ch, n_out=n_out)

    def update(self, x):
        spikes_2d = self.cell(x)  # (..., 8, 8, out_ch)
        pooled = spikes_2d.mean(axis=(-3, -2))  # -> (..., out_ch)
        return self.readout(pooled)


def main(n_epochs: int = 3, batch_size: int = 16, num_step: int = 20, plot: bool = True) -> Dict:
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_out=len(digits))
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs_flat, labels):
            inputs = inputs_flat.reshape(*inputs_flat.shape[:2], 8, 8, 1)
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels, decay_or_rank=0.95,
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_poisson_mnist(
                num_step=num_step, num_batch=batch_size, digits=digits, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[11-conv] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="11 · Conv-SNN (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
