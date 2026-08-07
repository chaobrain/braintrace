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

from typing import Any, Callable, cast

import brainstate
import brainunit as u
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

    Warnings
    --------
    **With a full-sequence input this path is blind to every learning-rule axis**
    (finding F-23). One whole-sequence call makes the within-call gradient exact
    reverse-mode, so the eligibility trace only enters at a sequence boundary --
    of which there is none. Every algorithm therefore returns gradients bitwise
    equal to BPTT, at every hyperparameter setting: ``D_RTRL``,
    ``OSTLRecurrent``, ``EProp`` at any ``kappa_filter_decay`` and ``pp_prop`` at
    any ``decay_or_rank`` are indistinguishable here.

    That makes this function a good test of the *compiler and the ETP
    per-primitive rules* -- it is the right instrument for asserting that an
    exact algorithm reproduces BPTT on a realistic model -- and the wrong
    instrument for any assertion whose subject is a trace factorization, a
    temporal recursion, a recurrence scope, a filter or a learning signal. For
    those use :func:`chunked_online_param_gradients` with ``chunk_size`` below
    the sequence length, and guard the test with :func:`assert_gradients_differ`.

    See Also
    --------
    chunked_online_param_gradients : finite-window path; sees the trace.
    assert_gradients_differ : negative control for a knob that must matter.
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
    for the exact D_RTRL algorithm, while the multi-step path does. This is the
    most aggressive finite window (one step), so it is maximally sensitive to
    learning-rule axes and maximally divergent from BPTT.

    See Also
    --------
    chunked_online_param_gradients : intermediate window size.
    docs/specs/2026-07-25-known-limitations.md : F-SINGLESTEP and F-23.
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


def flat_gradient_leaves(tree) -> dict:
    """Flatten a path-keyed gradient tree into ``{label: plain array}``.

    Gradient trees returned by the oracle are keyed by ``ParamState`` path and
    each value may itself be a pytree (a ``Linear`` contributes ``weight`` and
    ``bias``) whose leaves may carry ``brainunit`` units. Comparisons need plain
    arrays, so this strips both layers.

    Parameters
    ----------
    tree : dict
        Mapping from state path tuple to gradient pytree.

    Returns
    -------
    dict
        Mapping from ``'a/b|.leaf'`` label to a unit-stripped ``jax.Array``.
    """
    out = {}
    for key, value in tree.items():
        path_label = '/'.join(map(str, key)) if isinstance(key, tuple) else str(key)
        for leaf_path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
            label = f'{path_label}|{jax.tree_util.keystr(leaf_path)}'
            out[label] = jnp.asarray(u.get_mantissa(leaf))
    return out


def gradient_norm(tree) -> float:
    """Euclidean norm of every leaf of a gradient tree, taken together.

    Parameters
    ----------
    tree : dict
        Path-keyed gradient tree.

    Returns
    -------
    float
        The joint Euclidean norm.
    """
    leaves = flat_gradient_leaves(tree)
    # Accumulate in NumPy float64: JAX truncates an explicit float64 astype
    # unless x64 is enabled, and these sums are over whole gradient trees.
    total = sum(float((np.asarray(arr, dtype=np.float64) ** 2).sum())
                for arr in leaves.values())
    return float(np.sqrt(total))


def relative_deviation(actual, expected) -> float:
    """``||actual - expected|| / ||expected||`` over all leaves jointly.

    Parameters
    ----------
    actual, expected : dict
        Path-keyed gradient trees with the same structure.

    Returns
    -------
    float
        The relative deviation; ``inf`` when ``expected`` is all-zero and
        ``actual`` is not, ``0.0`` when both are.

    Raises
    ------
    AssertionError
        If the two trees do not have the same set of leaf labels.
    """
    a = flat_gradient_leaves(actual)
    e = flat_gradient_leaves(expected)
    if set(a) != set(e):
        raise AssertionError(
            f'gradient trees have different leaves: {sorted(set(a) ^ set(e))}')
    num = sum(float(((np.asarray(a[k], dtype=np.float64)
                      - np.asarray(e[k], dtype=np.float64)) ** 2).sum()) for k in e)
    den = sum(float((np.asarray(e[k], dtype=np.float64) ** 2).sum()) for k in e)
    if den == 0.0:
        return float('inf') if num > 0.0 else 0.0
    return float(np.sqrt(num) / np.sqrt(den))


def assert_model_is_live(model_factory, inputs, *, min_norm: float = 1e-8) -> float:
    """Assert the BPTT gradient of ``model_factory`` on ``inputs`` is non-trivial.

    A comparison against an all-zero reference gradient passes for every
    algorithm and therefore asserts nothing. SNN models are the common way to
    hit this: at a low input scale the neurons never reach threshold, the loss is
    zero, and so is the gradient. Spiking alone is not sufficient — a model can
    spike and still have a zero gradient — so the criterion is the gradient norm
    itself.

    Parameters
    ----------
    model_factory : Callable[[], brainstate.nn.Module]
        Zero-arg factory returning an uninitialized model.
    inputs : jax.Array
        ``(T, ...)`` input sequence.
    min_norm : float, optional
        Minimum acceptable BPTT gradient norm.

    Returns
    -------
    float
        The observed BPTT gradient norm.

    Raises
    ------
    AssertionError
        If the norm is at or below ``min_norm``.
    """
    norm = gradient_norm(bptt_param_gradients(model_factory, inputs))
    if not (norm > min_norm):
        raise AssertionError(
            f'model is not live: BPTT gradient norm {norm:.3e} <= {min_norm:.3e}. '
            'Any gradient comparison on this model/input pair is vacuous.'
        )
    return norm


def assert_gradients_differ(a, b, *, min_rel: float = 1e-6) -> float:
    """Assert two gradient trees are *distinguishable* — a negative control.

    Use this whenever a test intends to exercise a learning-rule knob. If the
    oracle path chosen cannot see the knob, this fails loudly instead of letting
    the test pass vacuously. See finding F-23.

    Parameters
    ----------
    a, b : dict
        Path-keyed gradient trees.
    min_rel : float, optional
        Minimum relative deviation between them.

    Returns
    -------
    float
        The observed relative deviation.

    Raises
    ------
    AssertionError
        If the deviation is below ``min_rel``.

    See Also
    --------
    online_param_gradients : the path on which knobs are invisible.
    chunked_online_param_gradients : the path on which they are visible.
    """
    rel = relative_deviation(a, b)
    if not (rel >= min_rel):
        raise AssertionError(
            f'gradients are indistinguishable: relative deviation {rel:.3e} < '
            f'{min_rel:.3e}. The knob under test does not move the gradient on '
            'this oracle path.'
        )
    return rel


def assert_param_gradients_close(actual, expected, *, atol=1e-4, rtol=0.0, keys=None):
    """Assert two param-gradient trees match, with a per-leaf diagnostic on failure.

    ``keys`` restricts the comparison to a subset of top-level state paths (e.g.
    only ETP params). When None, every key present in ``expected`` is compared.
    Nested pytrees and unit-carrying leaves are supported.
    """
    compare_keys = list(expected.keys()) if keys is None else list(keys)
    failures = []
    for key in compare_keys:
        # Flatten per key so the diagnostic can name the offending state key
        # exactly as the caller wrote it, and append the leaf path within it.
        a = flat_gradient_leaves({key: actual[key]})
        e = flat_gradient_leaves({key: expected[key]})
        if set(a) != set(e):
            raise AssertionError(
                f'gradient trees differ in structure at {key!r}: '
                f'{sorted(set(a) ^ set(e))}')
        for label in sorted(e):
            if not bool(jnp.allclose(a[label], e[label], atol=atol, rtol=rtol)):
                leaf = label.split('|', 1)[1]
                failures.append(
                    f"  {key!r}{leaf}: maxabsdiff="
                    f"{float(jnp.max(jnp.abs(a[label] - e[label]))):.3e}")
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


# ---------------------------------------------------------------------------
# Statistical acceptance: the third paradigm
# ---------------------------------------------------------------------------
#
# Element-wise equality and direction metrics cannot judge an *unbiased but
# noisy* estimator such as UORO. The obvious substitute -- "the deviation of the
# seed-mean from the reference shrinks like C/sqrt(N)" -- is not a bias test at
# all: a fixed multiplicative bias satisfies any such bound at every N a test can
# afford, because the bound is loosest exactly where the bias is measurable. What
# distinguishes unbiased-and-noisy from biased-and-noisy is the *sample variance*,
# so these helpers build a genuine confidence interval out of it.
#
# Gradient trees are compared through fixed scalar projections rather than
# leaf-by-leaf: a per-element interval over thousands of elements would need a
# multiplicity correction so severe that nothing could fail, while a handful of
# fixed random directions keeps the correction mild and still exposes any bias
# with a component along them (a bias orthogonal to all K directions is a measure
# -zero coincidence, and the directions are seeded, so a failure is reproducible).


def seed_gradient_samples(
    model_factory: Callable[[], brainstate.nn.Module],
    inputs,
    *,
    algo_factory: Callable[[brainstate.nn.Module, int], 'braintrace.ETraceAlgorithm'],
    seeds,
    chunk_size: int,
):
    """Gradient trees for one model and sequence under many estimator seeds.

    Parameters
    ----------
    model_factory : Callable[[], brainstate.nn.Module]
        Zero-arg factory returning an uninitialized model. Called once per seed,
        so every sample starts from identical parameters and hidden state -- the
        only thing that varies between samples is the estimator's own randomness.
    inputs : jax.Array
        ``(T, ...)`` input sequence, shared by every sample.
    algo_factory : Callable[[brainstate.nn.Module, int], ETraceAlgorithm]
        Builds the algorithm for a given seed.
    seeds : sequence of int
        The estimator seeds; ``len(seeds)`` is the sample size ``N``.
    chunk_size : int
        Window length, forwarded to :func:`chunked_online_param_gradients`. Must
        be smaller than ``T``, or the trace never enters the gradient (F-23).

    Returns
    -------
    list of dict
        One path-keyed gradient tree per seed, in ``seeds`` order.
    """
    def _bind(s: Any) -> Callable:
        """Bind ``s`` now, not at call time -- the classic late-binding trap."""
        return lambda m: algo_factory(m, s)

    return [
        chunked_online_param_gradients(
            model_factory, inputs,
            algo_factory=_bind(seed),
            chunk_size=chunk_size,
        )
        for seed in seeds
    ]


def fixed_gradient_directions(tree, num: int, *, seed: int = 0, keys=None):
    """``num`` fixed unit directions in the flattened gradient space of ``tree``.

    Parameters
    ----------
    tree : dict
        A representative path-keyed gradient tree; only its leaf labels and
        shapes are used.
    num : int
        How many directions to draw.
    seed : int, default 0
        NumPy seed. Fixed so a statistical failure is reproducible.
    keys : sequence of str, optional
        Restrict the directions' support to these leaf labels (as produced by
        :func:`flat_gradient_leaves`). Use it to confine a comparison to the
        parameter class the axis under test actually touches -- a whole-tree
        projection can be dominated by leaves the axis never reaches, which is
        how an acceptance criterion becomes vacuous.

    Returns
    -------
    list of dict
        Each a ``{leaf label: array}`` mapping with joint Euclidean norm 1.
    """
    leaves = flat_gradient_leaves(tree)
    labels = sorted(leaves) if keys is None else list(keys)
    missing = [k for k in labels if k not in leaves]
    if missing:
        raise AssertionError(
            f'unknown gradient leaf labels {missing}; available: {sorted(leaves)}')
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(num):
        raw = {k: rng.randn(*np.shape(leaves[k])) for k in labels}
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in raw.values()))
        out.append({k: jnp.asarray(v / norm) for k, v in raw.items()})
    return out


def project_gradient(tree, direction) -> float:
    """Inner product of a gradient tree with one direction from
    :func:`fixed_gradient_directions`.

    Parameters
    ----------
    tree : dict
        Path-keyed gradient tree.
    direction : dict
        ``{leaf label: array}``, as returned by
        :func:`fixed_gradient_directions`.

    Returns
    -------
    float
        The scalar projection, accumulated in float64.
    """
    leaves = flat_gradient_leaves(tree)
    return float(sum(
        float((np.asarray(leaves[k], dtype=np.float64)
               * np.asarray(v, dtype=np.float64)).sum())
        for k, v in direction.items()
    ))


def assert_unbiased_estimator(
    samples,
    reference,
    *,
    num_directions: int = 8,
    seed: int = 0,
    z: float = 3.5,
    tightness: float = 0.25,
    keys=None,
    directions=None,
):
    """Assert a set of gradient samples is an unbiased estimate of ``reference``.

    For each fixed direction ``d``, with ``v_s = <sample_s, d>``, sample mean
    ``m``, sample standard deviation ``sd`` and ``N = len(samples)``:

    - **unbiasedness** -- ``|m - <reference, d>| <= z * sd / sqrt(N)``;
    - **non-vacuity** -- ``sd / sqrt(N) <= tightness * ||reference||``.

    Both halves are required. The first alone passes for any estimator whose
    variance is large enough to swallow its bias; the second is what makes the
    interval mean something. A biased estimator fails the first as ``N`` grows; a
    hopelessly noisy one fails the second.

    The non-vacuity scale is the reference's **joint norm over the compared
    leaves**, not the individual projection ``|<reference, d>|``: a direction that
    happens to fall nearly orthogonal to the reference gradient has an almost-zero
    projection, and holding its standard error to a fraction of *that* would be
    unmeetable however good the estimator is.

    Parameters
    ----------
    samples : sequence of dict
        Per-seed gradient trees, e.g. from :func:`seed_gradient_samples`.
    reference : dict
        The quantity the estimator is supposed to be unbiased *for*. For UORO
        this is the exact within-group influence gradient (saturated SnAp-n
        through the same finite-window path), not necessarily BPTT.
    num_directions : int, default 8
        Number of fixed projections. Ignored when ``directions`` is given.
    seed : int, default 0
        Seed for the directions.
    z : float, default 3.5
        Interval half-width in standard errors. 3.5 is roughly a two-sided
        ``5e-4`` normal quantile, i.e. Bonferroni-corrected for eight
        comparisons at ``4e-3``.
    tightness : float, default 0.25
        Largest standard error, as a fraction of the reference gradient's norm,
        that still counts as an informative interval.
    keys : sequence of str, optional
        Restrict to these leaf labels (see :func:`fixed_gradient_directions`).
    directions : sequence of dict, optional
        Supply the directions explicitly instead of drawing them.

    Raises
    ------
    AssertionError
        If fewer than two samples are given, or any direction fails either half.
        The message lists **every** direction with its numbers -- a statistical
        failure that prints one number is unactionable.
    """
    samples = list(samples)
    if len(samples) < 2:
        raise AssertionError(
            f'a confidence interval needs at least 2 samples, got {len(samples)}.')
    if directions is None:
        directions = fixed_gradient_directions(
            reference, num_directions, seed=seed, keys=keys)
    n = len(samples)
    ref_leaves = flat_gradient_leaves(reference)
    support = sorted(directions[0]) if directions else sorted(ref_leaves)
    scale = float(np.sqrt(sum(
        float((np.asarray(ref_leaves[k], dtype=np.float64) ** 2).sum())
        for k in support)))
    bound = tightness * scale
    rows = []
    failures = False
    for i, d in enumerate(directions):
        ref = project_gradient(reference, d)
        vals = np.asarray([project_gradient(s, d) for s in samples], dtype=np.float64)
        mean = float(vals.mean())
        # ddof=1: the sample standard deviation, since the mean is estimated too.
        sd = float(vals.std(ddof=1))
        stderr = sd / np.sqrt(n)
        gap = abs(mean - ref)
        biased = gap > z * stderr
        wide = stderr > bound
        note = []
        if biased:
            note.append('mean outside its confidence interval')
        if wide:
            note.append('interval too wide to be informative')
        failures = failures or biased or wide
        rows.append(
            f'  direction {i}: reference={ref:+.6e} mean={mean:+.6e} '
            f'gap={gap:.3e} stderr={stderr:.3e} (z*stderr={z * stderr:.3e}, '
            f'tightness bound={bound:.3e})'
            + ('  <-- ' + '; '.join(note) if note else '')
        )
    if failures:
        raise AssertionError(
            f'gradient samples are not an unbiased estimate of the reference '
            f'(N={n}, z={z}, tightness={tightness}, '
            f'||reference||={scale:.6e}):\n' + '\n'.join(rows)
        )


def future_hidden_gradients(
    model_factory: Callable[[], brainstate.nn.Module],
    inputs,
    boundaries,
):
    """``d(sum_{t >= b} L_t) / d h^b`` at each window boundary ``b``.

    The regression target a DNI synthesiser is trained to predict, and the
    "pinned to the true value" oracle of the ``learning_signal='bootstrapped'``
    acceptance criteria. The sum is **strictly future** and the interval
    half-open: with windows ``[a, b)``, the step that writes ``h^b`` belongs to
    the window *before* the boundary, so its loss is excluded here and counted
    inside that window instead. Getting this off by one double-counts one loss
    per boundary.

    The differentiation route is the *public* one the DNI training recipe
    documents -- put the hidden states in ``brainstate.transform.grad``'s state
    set and read the entry they produce -- so the oracle and the recipe cannot
    drift apart. A raw ``jax.grad`` would not work here at all: the rollout writes
    ``HiddenState``, which brainstate refuses to let a bare JAX transformation
    track.

    A *fresh* model is built per boundary, so there is no live state to perturb
    and nothing to restore.

    Parameters
    ----------
    model_factory : Callable[[], brainstate.nn.Module]
        Zero-arg factory returning an uninitialized model.
    inputs : jax.Array
        ``(T, ...)`` input sequence.
    boundaries : sequence of int
        Step indices ``b`` at which to evaluate, ``0 <= b <= T``. ``b == T`` has
        no future loss and yields zeros.

    Returns
    -------
    list of dict
        One ``{hidden state path: cotangent}`` mapping per boundary, in
        ``boundaries`` order.
    """
    total_steps = int(inputs.shape[0])
    out = []
    for b in boundaries:
        model = model_factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        hidden = model.states(brainstate.HiddenState)

        # Roll to the boundary outside the gradient: h^b is a constant here, and
        # the target is the derivative of the *suffix* loss with respect to it.
        if b > 0:
            brainstate.transform.for_loop(lambda x: model(x), inputs[:b])

        if b >= total_steps:
            # A single-filter `states()` call returns one dict, not the tuple its
            # union return type also admits.
            out.append({k: jax.tree.map(u.math.zeros_like, st.value)
                        for k, st in cast(Any, hidden).items()})
            continue

        def suffix_loss(_b=b):
            losses = brainstate.transform.for_loop(
                lambda x: _sse(model(x)), inputs[_b:])
            return losses.sum()

        out.append(brainstate.transform.grad(suffix_loss, hidden)())
    return out
