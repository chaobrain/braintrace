# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""02 · Batching via ``vmap_new_states``.

Shows the per-sample-init pattern explicitly:
    1. wrap model in D_RTRL
    2. inside a vmapped new-states scope: init_all_states + compile_graph
    3. outside, wrap the online model in brainstate.nn.Vmap

Pick this pattern when every sample needs its own eligibility trace state
(the usual case).
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


class RNN(brainstate.nn.Module):
    def __init__(self, num_in: int, num_hidden: int):
        super().__init__()
        self.rnn = braintrace.nn.ValinaRNNCell(in_size=num_in, out_size=num_hidden, activation='tanh')
        self.out = braintrace.nn.Linear(num_hidden, 1)

    def update(self, x):
        return x >> self.rnn >> self.out


def main(*, n_epochs: int = 30, batch_size: int = 64, plot: bool = True) -> dict:
    num_step, num_hidden = 25, 32
    model = RNN(1, num_hidden)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=5e-3, eps=1e-1)
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online_model = braintrace.D_RTRL(model)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online_model.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online_model, vmap_states='new')

        def step_loss(inp, tar):
            out = vmap_model(inp)
            return braintools.metric.squared_error(out, tar).mean(), out

        def grad_step(prev_grads, x):
            inp, tar = x
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('MSE')
        plt.title('02 · Batching via vmap_new_states'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
