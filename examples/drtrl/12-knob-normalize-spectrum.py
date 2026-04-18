# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""12 · ``normalize_matrix_spectrum``: trace stability knob.

Intentionally starts with a spectrally unstable recurrent weight (scale 1.5)
and shows three training curves:

  * D_RTRL, normalize_matrix_spectrum=False   (baseline — may diverge)
  * D_RTRL, normalize_matrix_spectrum=True    (branch-free trace clip)
  * BPTT                                       (true-gradient reference)

Task: delayed XOR (short sequence, still long enough to surface instability).
"""
from __future__ import annotations

import pathlib
import sys

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class UnstableRNN(brainstate.nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, scale: float):
        super().__init__()
        self.cell = braintrace.nn.ValinaRNNCell(
            in_size=n_in, out_size=n_hidden,
            w_init=lambda shape: scale * jax.random.normal(jax.random.PRNGKey(0), shape) / np.sqrt(shape[-2]),
            activation='tanh',
        )
        self.readout = braintrace.nn.Linear(n_hidden, n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _clone_weights(src, dst):
    for (_, a), (_, b) in zip(src.states(brainstate.ParamState).items(),
                              dst.states(brainstate.ParamState).items()):
        b.value = jax.tree.map(lambda x: x, a.value)


def _online_train(model, weights, iter_batches, *, normalize: bool, n_epochs: int):
    opt = braintools.optim.Adam(3e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model, normalize_matrix_spectrum=normalize)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vm = brainstate.nn.Vmap(online, vmap_states='new')
        brainstate.transform.for_loop(lambda inp: vm(inp), inputs[:-1])

        def final_loss():
            out = vm(inputs[-1])
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean(), out

        grads, loss, _ = brainstate.transform.grad(final_loss, weights, has_aux=True, return_value=True)()
        opt.update(grads)
        return loss

    losses = []
    for x, y in iter_batches(n_epochs):
        losses.append(float(f_train(x, y)))
    return losses


def _bptt_train(model, weights, iter_batches, *, n_epochs: int):
    opt = braintools.optim.Adam(3e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        brainstate.nn.init_all_states(model, batch_size=inputs.shape[1])

        def f_loss():
            out = brainstate.transform.for_loop(model.update, inputs)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out[-1], targets).mean()

        grads, loss = brainstate.transform.grad(f_loss, weights, return_value=True)()
        opt.update(grads)
        return loss

    losses = []
    for x, y in iter_batches(n_epochs):
        losses.append(float(f_train(x, y)))
    return losses


def main(*, n_epochs: int = 30, batch_size: int = 32, plot: bool = True) -> dict:
    seq_len, delay, n_hidden = 16, 8, 32

    def iter_batches(n):
        for i in range(n):
            yield _shared.make_xor_batch(seq_len=seq_len, delay=delay, batch_size=batch_size, seed=i)

    model_base = UnstableRNN(2, n_hidden, 2, scale=1.5)
    model_fix = UnstableRNN(2, n_hidden, 2, scale=1.5)
    model_bptt = UnstableRNN(2, n_hidden, 2, scale=1.5)
    _clone_weights(model_base, model_fix)
    _clone_weights(model_base, model_bptt)

    loss_base = _online_train(model_base, model_base.states(brainstate.ParamState), iter_batches, normalize=False, n_epochs=n_epochs)
    loss_fix = _online_train(model_fix, model_fix.states(brainstate.ParamState), iter_batches, normalize=True, n_epochs=n_epochs)
    loss_bptt = _bptt_train(model_bptt, model_bptt.states(brainstate.ParamState), iter_batches, n_epochs=n_epochs)

    if plot:
        plt.plot(loss_base, label='D_RTRL (normalize=False)')
        plt.plot(loss_fix, label='D_RTRL (normalize=True)')
        plt.plot(loss_bptt, label='BPTT')
        plt.xlabel('epoch'); plt.ylabel('cross-entropy')
        plt.legend(); plt.title('12 · normalize_matrix_spectrum — delayed XOR'); plt.show()

    return {
        "losses": loss_fix,
        "baseline_losses": loss_base,
        "bptt_losses": loss_bptt,
    }


if __name__ == "__main__":
    main()
