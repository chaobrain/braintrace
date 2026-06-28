# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""09 · Row-scan MNIST classification with LSTM.

Flagship example: treats each MNIST image as 28 timesteps × 28 input features
and classifies the digit. Compares D_RTRL and BPTT on matched hyperparams.

Requires ``torchvision``. First run downloads MNIST to ``examples/data/MNIST``.
"""

import pathlib

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import braintrace


def _load_mnist(batch_size: int):
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    root = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'MNIST'
    root.mkdir(parents=True, exist_ok=True)
    tfm = T.Compose([T.ToTensor(), lambda img: img.squeeze(0).numpy()])
    train = torchvision.datasets.MNIST(str(root), train=True, download=True, transform=tfm)
    test = torchvision.datasets.MNIST(str(root), train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)


class LSTMNet(brainstate.nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.cell = braintrace.nn.LSTMCell(n_in, n_hidden)
        self.readout = braintrace.nn.Linear(n_hidden, n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _to_batch(batch):
    import jax.numpy as jnp
    x, y = batch
    x_np = np.stack([np.asarray(img) for img in x], axis=0)  # (B, 28, 28)
    y_np = np.asarray(y)
    return jnp.asarray(np.transpose(x_np, (1, 0, 2))), jnp.asarray(y_np)


def main(*, n_epochs: int = 1, batch_size: int = 64, max_batches: int | None = 100, plot: bool = True) -> dict:
    n_hidden = 64
    model_online = LSTMNet(28, n_hidden, 10)
    model_bptt = LSTMNet(28, n_hidden, 10)
    for (_, a), (_, b) in zip(model_online.states(brainstate.ParamState).items(),
                              model_bptt.states(brainstate.ParamState).items()):
        b.value = jax.tree.map(lambda x: x, a.value)

    w_online = model_online.states(brainstate.ParamState)
    w_bptt = model_bptt.states(brainstate.ParamState)
    opt_online = braintools.optim.Adam(1e-3);
    opt_online.register_trainable_weights(w_online)
    opt_bptt = braintools.optim.Adam(1e-3);
    opt_bptt.register_trainable_weights(w_bptt)

    train_loader, _ = _load_mnist(batch_size)

    # compile outside jit: init_all_states + compile_graph run once eagerly
    om = braintrace.compile(model_online, 'D_RTRL', jnp.zeros((batch_size, 28)), batch_size=batch_size)

    @brainstate.transform.jit
    def online_step(inputs, targets):
        brainstate.transform.for_loop(lambda inp: om(inp), inputs[:-1])

        def final_loss():
            out = om(inputs[-1])
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean(), out

        grads, loss, _ = brainstate.transform.grad(final_loss, w_online, has_aux=True, return_value=True)()
        opt_online.update(grads)
        return loss

    @brainstate.transform.jit
    def bptt_step(inputs, targets):
        brainstate.nn.init_all_states(model_bptt, batch_size=inputs.shape[1])

        def f_loss():
            out = brainstate.transform.for_loop(model_bptt.update, inputs)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out[-1], targets).mean()

        grads, loss = brainstate.transform.grad(f_loss, w_bptt, return_value=True)()
        opt_bptt.update(grads)
        return loss

    online_losses, bptt_losses = [], []
    for _ in range(n_epochs):
        for i, batch in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = _to_batch(batch)
            online_losses.append(float(online_step(x, y)))
            bptt_losses.append(float(bptt_step(x, y)))

    if plot:
        plt.plot(online_losses, label='D_RTRL')
        plt.plot(bptt_losses, label='BPTT')
        plt.xlabel('batch');
        plt.ylabel('cross-entropy')
        plt.legend();
        plt.title('09 · row-scan MNIST');
        plt.show()

    return {"losses": online_losses, "bptt_losses": bptt_losses}


if __name__ == "__main__":
    main()
