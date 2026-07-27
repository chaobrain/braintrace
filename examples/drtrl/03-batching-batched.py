# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""03 · Batching via ``brainstate.mixin.Batching``.

Alternative batching path: the algorithm sees the batch axis directly
instead of relying on vmap. Init once with ``batch_size=...``, compile
on a batched sample.
"""

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
    online_model = braintrace.compile(
        model, 'D_RTRL', jnp.zeros((batch_size, 1)), batch_size=batch_size,
    )
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=5e-3, eps=1e-1)
    opt.register_trainable_weights(weights)

    def step_loss(inp, tar):
        out = online_model(inp)
        return braintools.metric.squared_error(out, tar).mean()

    @brainstate.transform.jit
    def f_train(inputs, targets):
        # reduction='sum' preserves the accumulated-gradient scale this example
        # was tuned at; the reported loss stays the per-step mean.
        grads, step_losses = online_model.etrace_grad(
            inputs, targets, step_fn=step_loss, reduction='sum', return_value=True)
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses);
        plt.xlabel('epoch');
        plt.ylabel('MSE')
        plt.title('03 · Batching via brainstate.mixin.Batching');
        plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
