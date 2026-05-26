# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""06 · Batching via the batched ETP primitive path (no vmap wrapper).

Inputs carry a batch dimension directly. braintrace.matmul dispatches to the
batched primitive etp_mm_p when x.ndim >= 2, so the network runs natively
batched and pp_prop does not need to wrap the model in Vmap.
"""

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax
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


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 50, plot: bool = True) -> Dict:
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=1, n_rec=48, n_out=1)
        brainstate.nn.init_all_states(model, batch_size=batch_size)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        online_model = braintrace.pp_prop(
            model, decay_or_rank=0.95, vjp_method="single-step"
        )
        online_model.compile_graph(jnp.zeros((batch_size, 1)))

        def step_loss(inp, tar):
            out = online_model(inp)
            return braintools.metric.squared_error(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        @brainstate.transform.jit
        def train_step(inputs, targets):
            init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
            grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
            grads = brainstate.nn.clip_grad_norm(grads, 1.0)
            opt.update(grads)
            return step_losses.mean()

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_integrator_spikes(
                num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[06-batched] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="06 · batched primitive (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
