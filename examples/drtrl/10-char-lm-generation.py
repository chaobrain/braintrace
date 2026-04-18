# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""10 · Toy character-level language model.

Trains a MiniGRU on a short embedded corpus string with D_RTRL and BPTT,
then autoregressively samples a short continuation from each trained model.
"""

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

CORPUS = (
             "the quick brown fox jumps over the lazy dog. "
             "pack my box with five dozen liquor jugs. "
             "how vexingly quick daft zebras jump! "
         ) * 6


class CharRNN(brainstate.nn.Module):
    def __init__(self, vocab_size: int, n_hidden: int):
        super().__init__()
        self.cell = braintrace.nn.MiniGRU(in_size=vocab_size, out_size=n_hidden)
        self.readout = braintrace.nn.Linear(n_hidden, vocab_size)

    def update(self, x):
        return self.readout(self.cell(x))


def _sample(model, vocab: str, prompt: str, length: int) -> str:
    char2idx = {c: i for i, c in enumerate(vocab)}
    brainstate.nn.init_all_states(model)
    out_chars = []
    x = jax.nn.one_hot(jnp.asarray(char2idx.get(prompt[0], 0)), len(vocab))
    for _ in range(length):
        logits = model.update(x)
        probs = jax.nn.softmax(logits)
        next_idx = int(np.argmax(np.asarray(probs)))
        out_chars.append(vocab[next_idx])
        x = jax.nn.one_hot(jnp.asarray(next_idx), len(vocab))
    return ''.join(out_chars)


def main(*, n_epochs: int = 20, batch_size: int = 16, plot: bool = True) -> dict:
    seq_len = 32
    vocab, _, _ = _shared.make_char_batches(CORPUS, seq_len=seq_len, batch_size=batch_size, seed=0)
    vocab_size = len(vocab)

    model_online = CharRNN(vocab_size, n_hidden=64)
    model_bptt = CharRNN(vocab_size, n_hidden=64)
    w_online = model_online.states(brainstate.ParamState)
    w_bptt = model_bptt.states(brainstate.ParamState)
    for (_, a), (_, b) in zip(w_online.items(), w_bptt.items()):
        b.value = jax.tree.map(lambda x: x, a.value)

    opt_online = braintools.optim.Adam(3e-3);
    opt_online.register_trainable_weights(w_online)
    opt_bptt = braintools.optim.Adam(3e-3);
    opt_bptt.register_trainable_weights(w_bptt)

    @brainstate.transform.jit
    def online_step(inputs, targets):
        om = braintrace.D_RTRL(model_online)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model_online)
            om.compile_graph(inputs[0, 0])

        init()
        vm = brainstate.nn.Vmap(om, vmap_states='new')

        def step_loss(inp, tar):
            out = vm(inp)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f_grad = brainstate.transform.grad(step_loss, w_online, has_aux=True, return_value=True)
            cur_grads, loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in w_online.items()})
        grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
        opt_online.update(grads)
        return step_losses.mean()

    @brainstate.transform.jit
    def bptt_step(inputs, targets):
        brainstate.nn.init_all_states(model_bptt, batch_size=inputs.shape[1])

        def f_loss():
            outs = brainstate.transform.for_loop(model_bptt.update, inputs)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(outs, targets).mean()

        grads, loss = brainstate.transform.grad(f_loss, w_bptt, return_value=True)()
        opt_bptt.update(grads)
        return loss

    online_losses, bptt_losses = [], []
    for epoch in range(n_epochs):
        _, x, y = _shared.make_char_batches(CORPUS, seq_len=seq_len, batch_size=batch_size, seed=epoch)
        online_losses.append(float(online_step(x, y)))
        bptt_losses.append(float(bptt_step(x, y)))

    online_sample = _sample(model_online, vocab, prompt=vocab[0], length=100)
    bptt_sample = _sample(model_bptt, vocab, prompt=vocab[0], length=100)

    if plot:
        plt.plot(online_losses, label='D_RTRL')
        plt.plot(bptt_losses, label='BPTT')
        plt.xlabel('epoch');
        plt.ylabel('cross-entropy')
        plt.legend();
        plt.title('10 · toy char-LM');
        plt.show()

    return {
        "losses": online_losses,
        "bptt_losses": bptt_losses,
        "samples": {"online": online_sample, "bptt": bptt_sample},
    }


if __name__ == "__main__":
    out = main()
    print("[online] ", out["samples"]["online"])
    print("[bptt]   ", out["samples"]["bptt"])
