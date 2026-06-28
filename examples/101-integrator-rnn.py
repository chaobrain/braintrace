# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

"""
Integrator RNN: Online Learning vs BPTT
========================================

This example trains an RNN to integrate (cumulative sum) a noisy input signal.
It compares two training approaches:

1. **BPTT** (Backpropagation Through Time) — standard offline gradient computation
2. **Online Learning** (D-RTRL via braintrace) — eligibility-trace-based online gradient computation

The model uses ``braintrace.nn.ValinaRNNCell`` and ``braintrace.nn.Linear``,
which internally use ETP primitives (``braintrace.matmul``) so the compiler
can automatically build eligibility trace graphs for online learning.
"""

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # headless backend: render to file, no display needed
import matplotlib.pyplot as plt
import numpy as np

import braintrace

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

dt = 0.04
num_step = int(1.0 / dt)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@brainstate.transform.jit(static_argnums=2)
def build_inputs_and_targets(mean=0.025, scale=0.01, batch_size=10):
    # Create the white noise input
    sample = brainstate.random.normal(size=(1, batch_size, 1))
    bias = mean * 2.0 * (sample - 0.5)
    samples = brainstate.random.normal(size=(num_step, batch_size, 1))
    noise_t = scale / dt ** 0.5 * samples
    inputs = bias + noise_t
    targets = jnp.cumsum(inputs, axis=0)
    return inputs, targets


def train_data(num_batch=512, n_batches_per_epoch=500):
    for _ in range(n_batches_per_epoch):
        yield build_inputs_and_targets(0.025, 0.01, num_batch)


# ---------------------------------------------------------------------------
# Model — uses braintrace.nn layers (ETP-aware)
# ---------------------------------------------------------------------------

class RNN(brainstate.nn.Module):
    def __init__(self, num_in, num_hidden):
        super().__init__()
        self.rnn = braintrace.nn.MiniGRU(in_size=num_in, out_size=num_hidden)
        self.out = braintrace.nn.Linear(num_hidden, 1)

    def update(self, x):
        return x >> self.rnn >> self.out


# ---------------------------------------------------------------------------
# BPTT Training
# ---------------------------------------------------------------------------

def train_bptt(n_epochs=5, num_batch=512, num_hidden=100, n_batches_per_epoch=500):
    model = RNN(1, num_hidden)
    weights = model.states(brainstate.ParamState)

    lr = braintools.optim.ExponentialDecayLR(0.025, decay_steps=1, decay_rate=0.99975)
    opt = braintools.optim.Adam(lr=lr, eps=1e-1)
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_predict(inputs):
        brainstate.nn.init_all_states(model, batch_size=inputs.shape[1])
        return brainstate.transform.for_loop(model.update, inputs)

    def f_loss(inputs, targets, l2_reg=2e-4):
        predictions = f_predict(inputs)
        mse = braintools.metric.squared_error(predictions, targets).mean()
        l2 = 0.0
        for weight in weights.values():
            for leaf in jax.tree.leaves(weight.value):
                l2 += jnp.sum(leaf ** 2)
        return mse + l2_reg * l2

    @brainstate.transform.jit
    def f_train(inputs, targets):
        grads, l = brainstate.transform.grad(f_loss, weights, return_value=True)(inputs, targets)
        opt.update(grads)
        return l

    losses = []
    for i_epoch in range(n_epochs):
        epoch_losses = []
        for i_batch, (inps, tars) in enumerate(train_data(num_batch=num_batch, n_batches_per_epoch=n_batches_per_epoch)):
            loss = f_train(inps, tars)
            epoch_losses.append(float(loss))
            if (i_batch + 1) % 100 == 0:
                print(f'[BPTT]   Epoch {i_epoch}, Batch {i_batch + 1:3d}, Loss {loss:.5f}')
        # Always record at least the mean epoch loss
        if epoch_losses:
            losses.append(float(sum(epoch_losses) / len(epoch_losses)))

    return model, f_predict, losses


# ---------------------------------------------------------------------------
# Online Learning Training (D-RTRL)
# ---------------------------------------------------------------------------

def train_online(n_epochs=5, num_batch=512, num_hidden=100, n_batches_per_epoch=500):
    model = RNN(1, num_hidden)
    weights = model.states(brainstate.ParamState)

    lr = braintools.optim.ExponentialDecayLR(5e-3, decay_steps=1, decay_rate=0.99975)
    opt = braintools.optim.Adam(lr=lr, eps=1e-1)
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        # Wrap model with the D-RTRL online learning algorithm and vmap over the batch dimension
        vmap_model = braintrace.compile(model, braintrace.D_RTRL, inputs[0],
                                        batch_size=inputs.shape[1], vmap=True)

        # Loss at a single timestep (with L2 regularization matching BPTT)
        def step_loss(inp, tar, l2_reg=2e-4):
            out = vmap_model(inp)
            mse = braintools.metric.squared_error(out, tar).mean()
            l2 = sum(jnp.sum(leaf ** 2) for leaf in jax.tree.leaves(
                {k: v.value for k, v in weights.items()}
            ))
            return mse + l2_reg * l2, out

        # Accumulate gradients over timesteps
        def grad_step(prev_grads, x):
            inp, tar = x
            f_grad = brainstate.transform.grad(
                step_loss, weights, has_aux=True, return_value=True
            )
            cur_grads, local_loss, out = f_grad(inp, tar)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, local_loss

        # Initialize gradient accumulators to zeros
        grads = jax.tree.map(
            lambda a: jnp.zeros_like(a),
            {k: v.value for k, v in weights.items()}
        )

        # Scan over all timesteps, accumulating online gradients
        grads, step_losses = brainstate.transform.scan(
            grad_step, grads, (inputs, targets)
        )

        # Update weights
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for i_epoch in range(n_epochs):
        epoch_losses = []
        for i_batch, (inps, tars) in enumerate(train_data(num_batch=num_batch, n_batches_per_epoch=n_batches_per_epoch)):
            loss = f_train(inps, tars)
            epoch_losses.append(float(loss))
            if (i_batch + 1) % 100 == 0:
                print(f'[Online] Epoch {i_epoch}, Batch {i_batch + 1:3d}, Loss {loss:.5f}')
        # Always record at least the mean epoch loss
        if epoch_losses:
            losses.append(float(sum(epoch_losses) / len(epoch_losses)))

    # Build a predict function for evaluation
    @brainstate.transform.jit
    def f_predict(inputs):
        brainstate.nn.init_all_states(model, batch_size=inputs.shape[1])
        return brainstate.transform.for_loop(model.update, inputs)

    return model, f_predict, losses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    *,
    n_epochs: int = 5,
    num_batch: int = 512,
    num_hidden: int = 100,
    n_batches_per_epoch: int = 500,
    run_bptt: bool = True,
    plot: bool = True,
) -> dict:
    result = {}

    if run_bptt:
        print("=" * 60)
        print("Training with BPTT")
        print("=" * 60)
        bptt_model, bptt_predict, bptt_losses = train_bptt(
            n_epochs=n_epochs,
            num_batch=num_batch,
            num_hidden=num_hidden,
            n_batches_per_epoch=n_batches_per_epoch,
        )
        result["bptt_losses"] = bptt_losses

    print()
    print("=" * 60)
    print("Training with Online Learning (D-RTRL)")
    print("=" * 60)
    online_model, online_predict, online_losses = train_online(
        n_epochs=n_epochs,
        num_batch=num_batch,
        num_hidden=num_hidden,
        n_batches_per_epoch=n_batches_per_epoch,
    )
    result["losses"] = online_losses

    if plot:
        x, y = build_inputs_and_targets(0.025, 0.01, 1)
        online_preds = online_predict(x)

        if run_bptt:
            bptt_preds = bptt_predict(x)
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(np.asarray(y[:, 0]).flatten(), label='Ground Truth', color='black', linewidth=2)
            axes[0].plot(np.asarray(bptt_preds[:, 0]).flatten(), label='BPTT', linestyle='--')
            axes[0].plot(np.asarray(online_preds[:, 0]).flatten(), label='Online (D-RTRL)', linestyle='--')
            axes[0].set_xlabel('Time step')
            axes[0].set_ylabel('Value')
            axes[0].set_title('Integrator Predictions')
            axes[0].legend()
            axes[1].plot(bptt_losses, label='BPTT')
            axes[1].plot(online_losses, label='Online (D-RTRL)')
            axes[1].set_xlabel('Training checkpoint')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Loss')
            axes[1].legend()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(np.asarray(y[:, 0]).flatten(), label='Ground Truth', color='black', linewidth=2)
            axes[0].plot(np.asarray(online_preds[:, 0]).flatten(), label='Online (D-RTRL)', linestyle='--')
            axes[0].set_xlabel('Time step')
            axes[0].set_ylabel('Value')
            axes[0].set_title('Integrator Predictions')
            axes[0].legend()
            axes[1].plot(online_losses, label='Online (D-RTRL)')
            axes[1].set_xlabel('Training checkpoint')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Loss')
            axes[1].legend()

        plt.tight_layout()
        plt.show()

    return result


if __name__ == '__main__':
    main()
