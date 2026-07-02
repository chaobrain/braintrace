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
    template_params = template.states(brainstate.ParamState)
    assert isinstance(template_params, brainstate.util.FlattedDict)
    base_values = {
        k: np.asarray(v.value) for k, v in template_params.items()
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
    algo_factory: Callable[[brainstate.nn.Module], braintrace.ETraceAlgorithm],
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


def chunked_online_param_gradients(
    model_factory: Callable[[], brainstate.nn.Module],
    inputs,
    *,
    algo_factory: Callable[[brainstate.nn.Module], braintrace.ETraceAlgorithm],
    chunk_size: int,
):
    """Total sequence gradient accumulated over multi-step chunks.

    Splits ``inputs`` into consecutive chunks of ``chunk_size`` steps, calls
    the algorithm once per chunk (hidden and eligibility-trace state persist
    across calls), and sums the per-chunk parameter gradients. Unlike
    :func:`online_param_gradients` (one whole-sequence call, where the
    within-call gradient is exact reverse-mode and the trace only enters at
    the sequence boundary), chunking makes the total depend on the
    eligibility trace at every chunk boundary — this is the oracle that
    actually validates trace correctness.

    Parameters
    ----------
    model_factory : Callable[[], brainstate.nn.Module]
        Zero-arg factory returning an uninitialized model.
    inputs : jax.Array
        ``(T, ...)`` input sequence.
    algo_factory : Callable[[brainstate.nn.Module], braintrace.ETraceAlgorithm]
        Builds the online algorithm; must accept ``MultiStepData``.
    chunk_size : int
        Steps per chunk; the last chunk may be shorter.

    Returns
    -------
    dict
        Path-keyed total gradients for every ``ParamState``.
    """
    model = model_factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = algo_factory(model)
    algo.compile_graph(inputs[0])
    algo.init_etrace_state()
    params = model.states(brainstate.ParamState)

    total = None
    # test-support chunk loop (few iterations), not a model step driver
    for start in range(0, inputs.shape[0], chunk_size):
        chunk = inputs[start:start + chunk_size]
        g = brainstate.transform.grad(
            lambda seq: (algo(braintrace.MultiStepData(seq)) ** 2).sum(),
            params,
        )(chunk)
        total = g if total is None else jax.tree.map(
            lambda a, b: a + b, total, g)
    return total


def online_param_gradients_singlestep_naive(
    model_factory: Callable[[], brainstate.nn.Module],
    inputs,
    *,
    algo_factory: Callable[[brainstate.nn.Module], braintrace.ETraceAlgorithm],
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


def cosine_similarity(a, b) -> float:
    """Cosine of the angle between two gradient arrays (flattened). Returns NaN if
    either is all-zero. The robust direction signal for approximate algorithms:
    it ignores magnitude, which carries the F-SINGLESTEP / approximation bias."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float('nan')
    return float(a @ b / denom)


def sign_agreement(a, b) -> float:
    """Fraction of elements where ``a`` and ``b`` share the same sign, over the
    elements where both are non-negligible (|.| > 1e-8)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mask = (np.abs(a) > 1e-8) & (np.abs(b) > 1e-8)
    if mask.sum() == 0:
        return float('nan')
    return float((np.sign(a[mask]) == np.sign(b[mask])).mean())


def relative_magnitude(a, b) -> float:
    """``||a|| / ||b||`` (flattened). >1 means ``a`` is larger than the reference
    ``b``; used to quantify magnitude bias of approximate gradients."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    nb = np.linalg.norm(b)
    if nb == 0:
        return float('nan')
    return float(np.linalg.norm(a) / nb)


def assert_direction_aligned(
    approx, reference, *, min_cosine, min_sign_agreement=0.0, keys=None, mag_bounds=None
):
    """Assert an approximate gradient tree is *directionally* aligned with a
    reference (typically BPTT).

    For each compared key: cosine similarity must be >= ``min_cosine`` and sign
    agreement >= ``min_sign_agreement``; if ``mag_bounds=(lo, hi)`` is given, the
    relative magnitude must lie in ``[lo, hi]``. This is the C-level criterion for
    approximate algorithms, which are not expected to match BPTT element-wise.
    """
    compare = list(reference.keys()) if keys is None else list(keys)
    failures = []
    for key in compare:
        c = cosine_similarity(approx[key], reference[key])
        s = sign_agreement(approx[key], reference[key])
        if not (c >= min_cosine):
            failures.append(f"  {key}: cosine {c:.4f} < {min_cosine}")
        if not (s >= min_sign_agreement):
            failures.append(f"  {key}: sign_agreement {s:.4f} < {min_sign_agreement}")
        if mag_bounds is not None:
            r = relative_magnitude(approx[key], reference[key])
            lo, hi = mag_bounds
            if not (lo <= r <= hi):
                failures.append(f"  {key}: relmag {r:.4f} not in [{lo}, {hi}]")
    if failures:
        raise AssertionError("gradient direction not aligned:\n" + "\n".join(failures))
