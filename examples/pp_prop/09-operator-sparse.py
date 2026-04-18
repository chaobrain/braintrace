# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""09 · Sparse recurrent weights via a fixed connectivity mask.

The recurrent block of a LIF RSNN uses `braintrace.nn.Linear` with a
Bernoulli sparsity mask on the recurrent rows. Zero entries stay frozen at
zero because Linear multiplies the weight by `w_mask` on every forward
pass; pp_prop sees the combined op as a dense matmul and gradients for
absent connections are zeroed by the same mask.

Note: the sparse ETP primitive `etp_sp_mm_p` exists but `saiunit.sparse`
COO/CSR formats lack JAX batching rules today (same constraint noted for
`examples/drtrl/06-operator-sparse.py`). This masked-dense fallback still
exercises `pp_prop` over a sparse connectivity pattern end-to-end.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainpy.state
import brainstate
import braintools
import jax.numpy as jnp
import numpy as np
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class SparseLIFCell(brainstate.nn.Module):
    """LIF cell whose recurrent rows obey a fixed Bernoulli sparsity mask."""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        density: float = 0.1,
        seed: int = 0,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(
            n_rec, R=1. * u.ohm, tau=tau_mem, V_th=V_th,
            V_reset=0. * u.mV, V_rest=0. * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        ff_w = braintools.init.KaimingNormal(ff_scale, unit=u.mA)((n_in, n_rec))
        rec_w = braintools.init.KaimingNormal(rec_scale, unit=u.mA)((n_rec, n_rec))
        w = u.math.concatenate([ff_w, rec_w], axis=0)
        rng = np.random.default_rng(seed)
        ff_mask = np.ones((n_in, n_rec), dtype=np.float32)
        rec_mask = (rng.random((n_rec, n_rec)) < density).astype(np.float32)
        w_mask = jnp.asarray(np.concatenate([ff_mask, rec_mask], axis=0))
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec, w_init=w,
                b_init=braintools.init.ZeroInit(unit=u.mA),
                w_mask=w_mask,
            ),
            syn=brainpy.state.Expon(
                n_rec, tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.),
            post=self.neu,
        )

    def update(self, x):
        self.syn(u.math.concatenate([x, self.neu.get_spike()], axis=-1))
        self.neu(0. * u.mA)
        return self.neu.get_spike()


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int, density: float):
        super().__init__()
        self.cell = SparseLIFCell(n_in=n_in, n_rec=n_rec, density=density)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 25, plot: bool = True) -> Dict:
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=64, n_rec=96, n_out=len(digits), density=0.1)
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
            print(f"[09-sparse] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="09 · Sparse recurrence (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
