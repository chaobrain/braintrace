# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Ground-truth gradient oracle for online-learning algorithms (test support).

The sequence loss is fixed to sum-of-squares over every step and element::

    L = sum_t (model(x_t) ** 2).sum()

BPTT differentiates this through an unrolled ``for_loop``; this is the exact
total gradient any *exact* online algorithm must reproduce.
"""

from typing import Callable

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

import braintrace


def _sse(y):
    return (y ** 2).sum()


def bptt_param_gradients(model_factory: Callable[[], brainstate.nn.Module], inputs):
    """Exact BPTT gradient of the sequence sum-of-squares loss w.r.t. all ParamStates."""
    model = model_factory()
    brainstate.nn.init_all_states(model, batch_size=1)

    def total_loss():
        losses = brainstate.transform.for_loop(lambda x: _sse(model(x)), inputs)
        return losses.sum()

    return brainstate.transform.grad(total_loss, model.states(brainstate.ParamState))()


def finite_difference_param_gradients(
    model_factory: Callable[[], brainstate.nn.Module], inputs, *, eps: float = 1e-3
):
    """Central finite-difference gradient of the sequence SSE loss for every ParamState.

    Independent arbiter for the BPTT implementation. O(num_params) loss evals;
    intended for small toy models only.
    """
    template = model_factory()
    brainstate.nn.init_all_states(template, batch_size=1)
    base_values = {
        k: np.asarray(v.value) for k, v in template.states(brainstate.ParamState).items()
    }

    def loss_with(values):
        model = model_factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        params = model.states(brainstate.ParamState)
        for k, arr in values.items():
            params[k].value = jnp.asarray(arr)
        losses = brainstate.transform.for_loop(lambda x: _sse(model(x)), inputs)
        return float(losses.sum())

    grads = {}
    for key, base in base_values.items():
        g = np.zeros_like(base)
        flat = base.reshape(-1)
        gflat = g.reshape(-1)
        for idx in range(flat.size):
            plus = {k: v.copy() for k, v in base_values.items()}
            minus = {k: v.copy() for k, v in base_values.items()}
            plus[key].reshape(-1)[idx] = flat[idx] + eps
            minus[key].reshape(-1)[idx] = flat[idx] - eps
            gflat[idx] = (loss_with(plus) - loss_with(minus)) / (2 * eps)
        grads[key] = jnp.asarray(g)
    return grads


def online_param_gradients(
    model_factory: Callable[[], brainstate.nn.Module],
    inputs,
    *,
    algo_factory: Callable[[brainstate.nn.Module], object],
):
    """Total sequence gradient from an online algorithm via the multi-step VJP path.

    ``algo_factory(model)`` must return an algorithm whose ``__call__`` accepts a
    ``braintrace.MultiStepData`` and returns the stacked per-step outputs. The loss
    ``(out ** 2).sum()`` over the whole stacked output equals the BPTT sequence loss.
    """
    model = model_factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()

    return brainstate.transform.grad(
        lambda seq: (algo(braintrace.MultiStepData(seq)) ** 2).sum(),
        model.states(brainstate.ParamState),
    )(inputs)


def online_param_gradients_singlestep_naive(
    model_factory: Callable[[], brainstate.nn.Module],
    inputs,
    *,
    algo_factory: Callable[[brainstate.nn.Module], object],
):
    """Naive 'single-step' total gradient: sum of per-step grad((algo(x_t)**2).sum()).

    Kept to document finding F-SINGLESTEP — this recipe does NOT equal BPTT even
    for the exact D_RTRL algorithm, while the multi-step path does. See
    dev/superpowers/specs/2026-05-26-comprehensive-test-strategy-design.md.
    """
    model = model_factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()
    params = model.states(brainstate.ParamState)

    total = None
    for t in range(inputs.shape[0]):
        g = brainstate.transform.grad(lambda x: (algo(x) ** 2).sum(), params)(inputs[t])
        total = g if total is None else jax.tree.map(lambda a, b: a + b, total, g)
    return total


def assert_param_gradients_close(actual, expected, *, atol=1e-4, rtol=0.0, keys=None):
    """Assert two param-gradient dicts match, with a per-key diagnostic on failure.

    ``keys`` restricts the comparison to a subset (e.g. only ETP params). When
    None, every key present in ``expected`` is compared.
    """
    compare_keys = list(expected.keys()) if keys is None else list(keys)
    failures = []
    for key in compare_keys:
        a = jnp.asarray(actual[key])
        e = jnp.asarray(expected[key])
        if not bool(jnp.allclose(a, e, atol=atol, rtol=rtol)):
            failures.append(f"  {key}: maxabsdiff={float(jnp.max(jnp.abs(a - e))):.3e}")
    if failures:
        raise AssertionError(
            "param gradients differ beyond tolerance "
            f"(atol={atol}, rtol={rtol}):\n" + "\n".join(failures)
        )
