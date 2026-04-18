# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""08 · ``braintrace.nn.Conv1d`` as an ETP operator.

Each timestep delivers a 28-pixel row of synthetic Poisson rates. A Conv1d
extracts local features; MiniGRU integrates over time; a Linear head
classifies. The Conv kernel is a standard ParamState but routes through
``etp_conv_p`` via ``braintrace.nn.Conv1d``, so it appears in the eligibility
trace.
"""

import pathlib
import sys

import brainstate
import braintools
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _make_poisson_rows(batch_size: int, n_classes: int = 4, seed: int = 0):
    """Generate synthetic row-streams: each class has a distinct rate profile."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=(batch_size,))
    t, w = 28, 28
    profiles = rng.uniform(0.1, 0.9, size=(n_classes, w)).astype(np.float32)
    rates = profiles[labels]
    stream = rng.poisson(rates[:, None, :].repeat(t, axis=1)).astype(np.float32)
    # (T, B, 28, 1) — ``braintrace.nn.Conv1d`` uses channel-last layout:
    # ``in_size=(spatial, channels)``.
    x = jnp.asarray(np.transpose(stream, (1, 0, 2))[:, :, :, None])
    y = jnp.asarray(labels)
    return x, y


class ConvRNN(brainstate.nn.Module):
    def __init__(self, n_hidden: int, n_out: int):
        super().__init__()
        self.conv = braintrace.nn.Conv1d(
            in_size=(28, 1), out_channels=8, kernel_size=3, padding='SAME'
        )
        self.rnn = braintrace.nn.MiniGRU(in_size=28 * 8, out_size=n_hidden)
        self.readout = braintrace.nn.Linear(n_hidden, n_out)

    def update(self, x):
        y = self.conv(x)  # (B, 28, 8) or (28, 8)
        y = y.reshape(y.shape[0], -1) if y.ndim > 2 else y.reshape(-1)
        y = self.rnn(y)
        return self.readout(y)


def main(*, n_epochs: int = 30, batch_size: int = 16, plot: bool = True) -> dict:
    n_classes, n_hidden = 4, 32
    model = ConvRNN(n_hidden, n_classes)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(3e-3);
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online, vmap_states='new')
        brainstate.transform.for_loop(lambda inp: vmap_model(inp), inputs[:-1])

        def final_loss():
            out = vmap_model(inputs[-1])
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean(), out

        grads, loss, _ = brainstate.transform.grad(final_loss, weights, has_aux=True, return_value=True)()
        opt.update(grads)
        return loss

    losses = []
    for epoch in range(n_epochs):
        x, y = _make_poisson_rows(batch_size, n_classes, seed=epoch)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses);
        plt.xlabel('epoch');
        plt.ylabel('cross-entropy')
        plt.title('08 · Conv1d + MiniGRU');
        plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
