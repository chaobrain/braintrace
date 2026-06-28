# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""11 · ``fast_solve=True`` vs ``fast_solve=False``.

Demonstrates:
  * numerical equivalence (allclose on summed gradients)
  * wall-clock speedup from the einsum fast path

The fast path applies when every ETP primitive in the graph has an
elementwise ``yw_to_w`` rule (matmul, mv, element_wise). For this example
(ValinaRNN + Linear) all primitives qualify.
"""

import pathlib
import sys
import time

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


def _grad_run(model, weights, inputs, targets, *, fast_solve: bool):
    # compile outside jit so init_all_states + compile_graph run eagerly
    online = braintrace.compile(
        model, 'D_RTRL', inputs[0], batch_size=inputs.shape[1],
        fast_solve=fast_solve,
    )

    def step_loss(inp, tar):
        out = online(inp)
        return braintools.metric.squared_error(out, tar).mean(), out

    def grad_step(prev_grads, pair):
        inp, tar = pair
        f = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
        cur, l, _ = f(inp, tar)
        return jax.tree.map(lambda a, b: a + b, prev_grads, cur), l

    @brainstate.transform.jit
    def f_grad(inputs, targets):
        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, _ = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
        return grads

    return f_grad(inputs, targets)


def main(*, n_epochs: int = 5, batch_size: int = 16, plot: bool = True) -> dict:
    num_step, num_hidden = 25, 32
    model_a = RNN(1, num_hidden)
    model_b = RNN(1, num_hidden)
    for (_, a), (_, b) in zip(model_a.states(brainstate.ParamState).items(),
                              model_b.states(brainstate.ParamState).items()):
        b.value = jax.tree.map(lambda x: x, a.value)
    wa = model_a.states(brainstate.ParamState)
    wb = model_b.states(brainstate.ParamState)

    x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=batch_size, seed=0)

    # warmup (compile)
    _ = _grad_run(model_a, wa, x, y, fast_solve=True)
    _ = _grad_run(model_b, wb, x, y, fast_solve=False)

    def timed(model, w, flag):
        t0 = time.perf_counter()
        for _ in range(n_epochs):
            g = _grad_run(model, w, x, y, fast_solve=flag)
            jax.tree.map(lambda a: a.block_until_ready(), g)
        return (time.perf_counter() - t0) / n_epochs, g

    fast_time, fast_grads = timed(model_a, wa, True)
    slow_time, slow_grads = timed(model_b, wb, False)

    max_diff = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(fast_grads), jax.tree.leaves(slow_grads))
    )
    allclose = bool(
        all(
            jnp.allclose(a, b, atol=1e-5, rtol=1e-4)
            for a, b in zip(jax.tree.leaves(fast_grads), jax.tree.leaves(slow_grads))
        )
    )

    print(f"fast_solve=True  mean time/epoch: {fast_time * 1000:.2f} ms")
    print(f"fast_solve=False mean time/epoch: {slow_time * 1000:.2f} ms")
    print(f"max |grad_fast - grad_slow|     : {max_diff:.3e}")
    print(f"allclose (atol=1e-5, rtol=1e-4) : {allclose}")

    if plot:
        plt.bar(['fast_solve=True', 'fast_solve=False'], [fast_time * 1000, slow_time * 1000])
        plt.ylabel('ms / epoch')
        plt.title(f'11 · fast_solve runtime (max-grad-diff {max_diff:.1e})');
        plt.show()

    return {
        "losses": [],
        "fast_time_ms": fast_time * 1000,
        "slow_time_ms": slow_time * 1000,
        "max_grad_diff": max_diff,
        "allclose": allclose,
    }


if __name__ == "__main__":
    main()
