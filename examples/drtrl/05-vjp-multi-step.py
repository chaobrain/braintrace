# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""05 · vjp_method='multi-step' on the copying-memory task.

Multi-step computes the VJP over a window, reducing single-step bias at the
cost of more compute and memory. Compare the loss curve here with Task 04.
"""
from __future__ import annotations

import pathlib
import sys

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class GRUNet(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = braintrace.nn.GRUCell(n_in, n_rec)
        self.readout = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(*, n_epochs: int = 50, batch_size: int = 32, plot: bool = True) -> dict:
    n_in, n_rec, n_out, time_lag = 10, 64, 10, 10
    model = GRUNet(n_in, n_rec, n_out)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(1e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model, vjp_method='multi-step')

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online, vmap_states='new')
        warmup_len = inputs.shape[0] - 10
        brainstate.transform.for_loop(lambda inp: vmap_model(inp), inputs[:warmup_len])

        def step_loss(inp, tar):
            out = vmap_model(inp)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, step_losses = brainstate.transform.scan(
            grad_step, init_grads, (inputs[warmup_len:], targets)
        )
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_copy_batch(time_lag=time_lag, batch_size=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('cross-entropy')
        plt.title('05 · multi-step VJP — copying task'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
