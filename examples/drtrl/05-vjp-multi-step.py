# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""05 · vjp_method='multi-step' on the copying-memory task.

Multi-step computes the VJP over a window, reducing single-step bias at the
cost of more compute and memory. Compare the loss curve here with Task 04.
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
    online = braintrace.compile(
        model, 'D_RTRL', jnp.zeros((batch_size, n_in)), batch_size=batch_size,
        vjp_method='multi-step',
    )
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(1e-3);
    opt.register_trainable_weights(weights)

    def step_loss(inp, tar):
        out = online(inp)
        return braintools.metric.softmax_cross_entropy_with_integer_labels(out, tar).mean()

    warmup_len = time_lag + 20 - 10  # seq_length - output_window

    @brainstate.transform.jit
    def f_train(inputs, targets):
        # Free-run the prefix: hidden states and the eligibility trace advance,
        # no loss is computed. etrace_grad then continues from that state --
        # the two calls describe one trajectory.
        online.etrace_evolve(inputs[:warmup_len])
        grads, step_losses = online.etrace_grad(
            inputs[warmup_len:], targets, step_fn=step_loss,
            reduction='sum', return_value=True)
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_copy_batch(time_lag=time_lag, batch_size=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses);
        plt.xlabel('epoch');
        plt.ylabel('cross-entropy')
        plt.title('05 · multi-step VJP — copying task');
        plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
