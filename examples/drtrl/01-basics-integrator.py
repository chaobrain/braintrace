# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""01 · Minimal D_RTRL on the integrator task.

The smallest working example. Trains a vanilla RNN on noisy cumulative-sum
regression and compares D_RTRL against BPTT on the same initialisation.
Read the inline comments top to bottom.
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


def _train_online(n_epochs, num_step, num_batch, num_hidden, lr):
    model = RNN(1, num_hidden)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=lr, eps=1e-1)
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
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=num_batch)
        losses.append(float(f_train(x, y)))
    return model, losses


def _train_bptt(n_epochs, num_step, num_batch, num_hidden, lr):
    model = RNN(1, num_hidden)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=lr, eps=1e-1)
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_predict(inputs):
        brainstate.nn.init_all_states(model, batch_size=inputs.shape[1])
        return brainstate.transform.for_loop(model.update, inputs)

    def f_loss(inputs, targets):
        preds = f_predict(inputs)
        return braintools.metric.squared_error(preds, targets).mean()

    @brainstate.transform.jit
    def f_train(inputs, targets):
        grads, l = brainstate.transform.grad(f_loss, weights, return_value=True)(inputs, targets)
        opt.update(grads)
        return l

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=num_batch)
        losses.append(float(f_train(x, y)))
    return model, losses


def main(*, n_epochs: int = 50, batch_size: int = 64, plot: bool = True) -> dict:
    num_step = 25
    num_hidden = 32
    _, online_losses = _train_online(n_epochs, num_step, batch_size, num_hidden, lr=5e-3)
    _, bptt_losses = _train_bptt(n_epochs, num_step, batch_size, num_hidden, lr=2.5e-2)
    if plot:
        plt.plot(online_losses, label='D_RTRL')
        plt.plot(bptt_losses, label='BPTT')
        plt.xlabel('epoch'); plt.ylabel('MSE')
        plt.legend(); plt.title('01 · Basics — integrator')
        plt.show()
    return {"losses": online_losses, "bptt_losses": bptt_losses}


if __name__ == "__main__":
    main()
