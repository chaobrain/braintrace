# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""09 · Native CSR recurrent weights with pp-prop.

The feed-forward projection remains dense while the recurrent projection uses
``braintrace.nn.SparseLinear`` over a direct ``brainevent.CSR`` structure. The
fixed-degree connectivity and trainable values both scale with represented
edges, and the batched path lowers through the sparse ETP primitive.
"""

import importlib.util
import pathlib
from typing import Dict, Optional

import brainevent
import brainpy.state
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

import braintrace

_shared_path = pathlib.Path(__file__).resolve().with_name("_shared.py")
_shared_spec = importlib.util.spec_from_file_location("_pp_prop_shared", _shared_path)
if _shared_spec is None or _shared_spec.loader is None:
    raise ImportError(f"Cannot load pp-prop shared helpers from {_shared_path}")
_shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared)


class SparseLIFCell(brainstate.nn.Module):
    """LIF cell with dense feed-forward and native sparse recurrent projections."""

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
        sparse_backend: Optional[str] = None,
    ):
        super().__init__()
        self.neu = brainpy.state.LIF(
            n_rec, R=1. * u.ohm, tau=tau_mem, V_th=V_th,
            V_reset=0. * u.mV, V_rest=0. * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        ff_w = braintools.init.KaimingNormal(ff_scale, unit=u.mA)((n_in, n_rec))
        rec_csr = _fixed_degree_csr(
            n_rec=n_rec,
            density=density,
            scale=rec_scale,
            seed=seed,
            backend=sparse_backend,
        )
        rec_linear = braintrace.nn.SparseLinear(rec_csr, b_init=None)
        rec_params = dict(rec_linear.weight.value)
        rec_params["weight"] = rec_params["weight"] * u.mA
        rec_linear.weight.value = rec_params
        self.ff_syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in, n_rec, w_init=ff_w,
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            syn=brainpy.state.Expon(
                n_rec, tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.),
            post=self.neu,
        )
        self.rec_syn = brainpy.state.AlignPostProj(
            comm=rec_linear,
            syn=brainpy.state.Expon(
                n_rec, tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.),
            post=self.neu,
        )

    def update(self, x):
        self.ff_syn(x)
        self.rec_syn(self.neu.get_spike())
        self.neu(0. * u.mA)
        return self.neu.get_spike()


def _fixed_degree_csr(
    n_rec: int,
    density: float,
    scale: float,
    seed: int,
    backend: Optional[str],
) -> brainevent.CSR:
    """Build an O(nnz) fixed-out-degree recurrent CSR matrix."""
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density!r}")
    degree = max(1, min(n_rec, round(n_rec * density)))
    rng = brainstate.random.RandomState(seed)
    offsets = jnp.sort(
        rng.choice(jnp.arange(n_rec, dtype=jnp.int32), size=degree, replace=False)
    )
    rows = jnp.arange(n_rec, dtype=jnp.int32)[:, None]
    indices = jnp.sort((rows + offsets[None, :]) % n_rec, axis=1).reshape(-1)
    indptr = jnp.arange(n_rec + 1, dtype=jnp.int32) * degree
    values = rng.randn(n_rec * degree) * (scale / n_rec) ** 0.5
    return brainevent.CSR(
        values,
        indices,
        indptr,
        shape=(n_rec, n_rec),
        backend=backend,
    )


class Net(brainstate.nn.Module):
    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        density: float,
        sparse_backend: Optional[str] = None,
    ):
        super().__init__()
        self.cell = SparseLIFCell(
            n_in=n_in,
            n_rec=n_rec,
            density=density,
            sparse_backend=sparse_backend,
        )
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(
    n_epochs: int = 3,
    batch_size: int = 32,
    num_step: int = 25,
    plot: bool = True,
    sparse_backend: Optional[str] = None,
) -> Dict:
    """Train the native sparse recurrent example and return its losses."""
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(
            n_in=64,
            n_rec=96,
            n_out=len(digits),
            density=0.1,
            sparse_backend=sparse_backend,
        )
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
