# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""07 · ``braintrace.lora_matmul`` adapter on a frozen base.

The base recurrent weight is a regular ``brainstate.ParamState`` accessed via
plain ``x @ w`` — therefore NOT part of any ETP primitive, therefore frozen
from D_RTRL's perspective. The LoRA layer uses ``braintrace.lora_matmul``
internally, so only ``lora_a``/``lora_b`` appear in the eligibility trace.

Task: random-frequency sine wave one-step-ahead prediction.
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


class LoRACell(brainstate.nn.RNNCell):
    """Frozen base recurrent + trainable LoRA residual on the hidden update."""

    def __init__(self, n_in: int, n_hidden: int, rank: int = 4):
        super().__init__()
        self.in_size = n_in
        self.out_size = n_hidden
        self.frozen_base = brainstate.ParamState(
            braintools.init.XavierNormal()((n_in + n_hidden, n_hidden))
        )
        self.lora = braintrace.nn.LoRA(
            in_features=n_in + n_hidden,
            lora_rank=rank,
            out_features=n_hidden,
            kernel_init=braintools.init.ZeroInit(),
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(
            braintools.init.param(braintools.init.ZeroInit(), self.out_size, batch_size)
        )

    def reset_state(self, batch_size=None, **kwargs):
        self.h.value = braintools.init.param(braintools.init.ZeroInit(), self.out_size, batch_size)

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        base = xh @ self.frozen_base.value  # plain matmul — excluded from ETP
        residual = self.lora(xh)  # ETP-aware via lora_matmul
        self.h.value = jax.nn.tanh(base + residual)
        return self.h.value


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_hidden: int):
        super().__init__()
        self.cell = LoRACell(n_in, n_hidden)
        self.readout = braintrace.nn.Linear(n_hidden, 1)

    def update(self, x):
        return self.readout(self.cell(x))


def main(*, n_epochs: int = 30, batch_size: int = 16, plot: bool = True) -> dict:
    num_step, n_hidden = 40, 32
    model = Net(1, n_hidden)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(5e-3);
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        vmap_model = braintrace.compile(
            model, braintrace.D_RTRL, inputs[0],
            batch_size=inputs.shape[1], vmap=True,
        )

        def step_loss(inp, tar):
            out = vmap_model(inp)
            return braintools.metric.squared_error(out, tar).mean()

        # reduction='sum' preserves the accumulated-gradient scale this
        # example was tuned at; the reported loss stays the per-step mean.
        grads, step_losses = vmap_model.etrace_grad(
            inputs, targets, step_fn=step_loss, reduction='sum', return_value=True)
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_sine_batch(num_step=num_step, batch_size=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses);
        plt.xlabel('epoch');
        plt.ylabel('MSE')
        plt.title('07 · LoRA adapter on frozen base — sine');
        plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
