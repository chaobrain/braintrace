# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
# kept manual: this file *is* the worked expansion of compile(..., vmap=True)
"""02 · Batching via ``vmap_new_states``.

Shows the per-sample-init pattern explicitly:
    1. wrap model in D_RTRL
    2. inside a vmapped new-states scope: init_all_states + compile_graph
    3. outside, wrap the online model in braintrace.ETraceVmap

``braintrace.compile(model, braintrace.D_RTRL, inputs[0], batch_size=B,
vmap=True)`` does exactly these three steps in one call, and that is what an
application should write. The steps are spelled out here so the wiring is
visible — which state gets the per-sample axis, and on which *unbatched*
sample the eligibility-trace graph is built.

Step 3 uses ``braintrace.ETraceVmap`` rather than ``brainstate.nn.Vmap``: it
*is* a ``brainstate.nn.Vmap`` (same call, same isinstance checks) and adds the
sequence drivers, which a bare ``Vmap`` does not carry.

Pick this pattern when every sample needs its own eligibility trace state
(the usual case).
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
        # ETraceVmap, not brainstate.nn.Vmap: same wrapper, plus the drivers.
        vmap_model = braintrace.ETraceVmap(online_model, vmap_states='new')

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
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses);
        plt.xlabel('epoch');
        plt.ylabel('MSE')
        plt.title('02 · Batching via vmap_new_states');
        plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
