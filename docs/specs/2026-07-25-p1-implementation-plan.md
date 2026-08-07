# P1 Verification Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gradient oracle able to distinguish one learning rule from another, then rebuild the known-limitation list against it.

**Architecture:** Additive changes to `braintrace/_algorithm/oracle.py` (negative-control helpers, pytree/unit-aware comparison, documented window semantics) and `oracle_models.py` (deterministic, spiking SNN `ModelSpec`s). Two new test modules pin the findings. One contained fix in `io_dim_vjp.py` reduces broadcast-parameter cotangents to the parameter's own shape. Documentation lands in `docs/specs/` and `AGENTS.md`.

**Tech Stack:** Python, JAX, `brainstate`, `brainunit`, `brainpy`, `braintools`, pytest.

Spec: [`2026-07-25-p1-verification-harness.md`](2026-07-25-p1-verification-harness.md)

## Global Constraints

- Tests are co-located as siblings with the `_test.py` suffix — never a `tests/` directory with `test_*.py` prefixes. The exception is the pre-existing `braintrace/_algorithm/tests/` directory, which holds cross-cutting suites; new cross-cutting suites go there to match.
- Never drive a model with a bare Python `for`/`while` loop when it runs repeatedly — use `brainstate.transform.for_loop` / `scan` / `jit`. Test-support chunk loops over a handful of chunks are exempt and already exist in `oracle.py`.
- Use `brainstate.random`, never `jax.random` directly, for random number generation in models.
- All public functions use NumPy-style docstrings with the canonical section order.
- SNN models require an integration step: wrap calls in `brainstate.environ.context(dt=0.1 * u.ms)`.
- Gradient trees may be nested pytrees and may carry `brainunit` units. Never call `jnp.asarray` on a gradient tree without flattening leaves and stripping units first.
- Commit messages must not contain a `Co-Authored-By` trailer.
- Verification bar for the phase: `pytest braintrace/` fully green and `mypy braintrace` clean.

---

### Task 1: Oracle negative controls and unit-aware comparison

**Files:**
- Modify: `braintrace/_algorithm/oracle.py`
- Test: `braintrace/_algorithm/oracle_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, all in `braintrace._algorithm.oracle`:
  - `flat_gradient_leaves(tree: dict) -> dict[str, jax.Array]` — flattens a `{state_path: pytree}` gradient dict to `{label: unit-stripped array}`.
  - `gradient_norm(tree: dict) -> float`
  - `relative_deviation(actual: dict, expected: dict) -> float`
  - `assert_model_is_live(model_factory, inputs, *, min_norm: float = 1e-8) -> float` — returns the BPTT gradient norm.
  - `assert_gradients_differ(a: dict, b: dict, *, min_rel: float = 1e-6) -> float` — returns the observed relative deviation.
  - `assert_param_gradients_close` gains pytree and unit support; its signature is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `braintrace/_algorithm/oracle_test.py`:

```python
# --- P1: negative-control helpers -------------------------------------------

def test_flat_gradient_leaves_handles_nested_and_units():
    """Gradient trees are nested dicts and may carry units; the flattener must
    yield plain arrays keyed by a stable label."""
    import brainunit as u
    from braintrace._algorithm.oracle import flat_gradient_leaves
    tree = {
        ('syn', 'comm', 'weight'): {'weight': jnp.ones((2, 3)) * u.mS,
                                    'bias': jnp.zeros((3,)) * u.mS},
        ('w',): jnp.arange(4.0),
    }
    flat = flat_gradient_leaves(tree)
    assert len(flat) == 3
    for arr in flat.values():
        assert not isinstance(arr, u.Quantity)
    assert sorted(k.split('|')[0] for k in flat) == ['syn/comm/weight',
                                                     'syn/comm/weight', 'w']


def test_gradient_norm_and_relative_deviation():
    from braintrace._algorithm.oracle import gradient_norm, relative_deviation
    a = {('w',): jnp.array([3.0, 4.0])}
    b = {('w',): jnp.array([3.0, 4.0])}
    assert gradient_norm(a) == pytest.approx(5.0, abs=1e-6)
    assert relative_deviation(a, b) == pytest.approx(0.0, abs=1e-12)
    c = {('w',): jnp.array([0.0, 4.0])}
    assert relative_deviation(a, c) == pytest.approx(3.0 / 5.0, abs=1e-6)


def test_assert_model_is_live_passes_on_live_model():
    from braintrace._algorithm.oracle import assert_model_is_live
    from braintrace._algorithm.oracle_models import tanh_rnn
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    xs = jnp.asarray(np.random.RandomState(0).randn(4, 3).astype('float32'))
    norm = assert_model_is_live(spec.factory, xs)
    assert norm > 0.0


def test_assert_model_is_live_rejects_a_dead_model():
    """A model whose output is constant has a zero gradient, so any comparison
    against it asserts nothing. The guard must reject it."""
    from braintrace._algorithm.oracle import assert_model_is_live

    def factory():
        class Dead(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.zeros((3, 3)))
                self.h = brainstate.HiddenState(jnp.zeros((1, 3)))

            def update(self, x):
                # output does not depend on w at all
                self.h.value = jnp.zeros((1, 3))
                return self.h.value

        return Dead()

    xs = jnp.zeros((3, 3))
    with pytest.raises(AssertionError, match='gradient norm'):
        assert_model_is_live(factory, xs)


def test_assert_gradients_differ_flags_a_dead_knob():
    from braintrace._algorithm.oracle import assert_gradients_differ
    a = {('w',): jnp.array([1.0, 2.0])}
    assert_gradients_differ(a, {('w',): jnp.array([1.0, 5.0])})
    with pytest.raises(AssertionError, match='indistinguishable'):
        assert_gradients_differ(a, {('w',): jnp.array([1.0, 2.0])})


def test_assert_param_gradients_close_supports_nested_unit_trees():
    """The pre-existing helper only handled flat, unitless dicts. SNN models
    have nested weight dicts carrying units."""
    import brainunit as u
    from braintrace._algorithm.oracle import assert_param_gradients_close
    a = {('syn',): {'weight': jnp.ones((2, 2)) * u.mS, 'bias': jnp.zeros(2) * u.mS}}
    b = {('syn',): {'weight': jnp.ones((2, 2)) * u.mS, 'bias': jnp.zeros(2) * u.mS}}
    assert_param_gradients_close(a, b, atol=1e-6)
    c = {('syn',): {'weight': jnp.full((2, 2), 2.0) * u.mS, 'bias': jnp.zeros(2) * u.mS}}
    with pytest.raises(AssertionError, match='maxabsdiff'):
        assert_param_gradients_close(a, c, atol=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest braintrace/_algorithm/oracle_test.py -k "flat_gradient or gradient_norm or model_is_live or gradients_differ or nested_unit" -v`

Expected: FAIL — `ImportError: cannot import name 'flat_gradient_leaves'` for the new helpers, and `test_assert_param_gradients_close_supports_nested_unit_trees` fails inside `jnp.asarray` on a dict.

- [ ] **Step 3: Implement the helpers**

In `braintrace/_algorithm/oracle.py`, add `import brainunit as u` to the imports, then insert these functions immediately before `assert_param_gradients_close`:

```python
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
    """Euclidean norm of every leaf of a gradient tree, taken together."""
    leaves = flat_gradient_leaves(tree)
    total = sum(float((arr.astype('float64') ** 2).sum()) for arr in leaves.values())
    return float(np.sqrt(total))


def relative_deviation(actual, expected) -> float:
    """``||actual - expected|| / ||expected||`` over all leaves jointly.

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
    num = sum(float(((a[k].astype('float64') - e[k].astype('float64')) ** 2).sum())
              for k in e)
    den = sum(float((e[k].astype('float64') ** 2).sum()) for k in e)
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
    """
    rel = relative_deviation(a, b)
    if not (rel >= min_rel):
        raise AssertionError(
            f'gradients are indistinguishable: relative deviation {rel:.3e} < '
            f'{min_rel:.3e}. The knob under test does not move the gradient on '
            'this oracle path.'
        )
    return rel
```

Then replace the body of `assert_param_gradients_close` (currently `oracle.py:196-213`) so it compares flattened leaves:

```python
def assert_param_gradients_close(actual, expected, *, atol=1e-4, rtol=0.0, keys=None):
    """Assert two param-gradient trees match, with a per-leaf diagnostic on failure.

    ``keys`` restricts the comparison to a subset of top-level state paths (e.g.
    only ETP params). When None, every key present in ``expected`` is compared.
    Nested pytrees and unit-carrying leaves are supported.
    """
    compare_keys = list(expected.keys()) if keys is None else list(keys)
    sub_actual = {k: actual[k] for k in compare_keys}
    sub_expected = {k: expected[k] for k in compare_keys}
    a = flat_gradient_leaves(sub_actual)
    e = flat_gradient_leaves(sub_expected)
    if set(a) != set(e):
        raise AssertionError(
            f'gradient trees have different leaves: {sorted(set(a) ^ set(e))}')
    failures = []
    for label in sorted(e):
        if not bool(jnp.allclose(a[label], e[label], atol=atol, rtol=rtol)):
            failures.append(
                f"  {label}: maxabsdiff={float(jnp.max(jnp.abs(a[label] - e[label]))):.3e}")
    if failures:
        raise AssertionError(
            "param gradients differ beyond tolerance "
            f"(atol={atol}, rtol={rtol}):\n" + "\n".join(failures)
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest braintrace/_algorithm/oracle_test.py -v`

Expected: PASS, including all pre-existing tests in the module — the `assert_param_gradients_close` change is a generalization, so flat unitless dicts behave identically.

- [ ] **Step 5: Document the window semantics of each oracle entry point**

This is the core of finding F-23 and it must be visible where people write tests. Note that `chunked_online_param_gradients` **already** documents it ("this is the oracle that actually validates trace correctness"); the gap is that `online_param_gradients` does not warn, and its stale `dev/` cross-reference is unresolvable.

In `online_param_gradients`, replace the docstring with:

```python
    """Total sequence gradient from an online algorithm via the multi-step VJP path.

    ``algo_factory(model)`` must return an algorithm whose ``__call__`` accepts a
    ``braintrace.MultiStepData`` and returns the stacked per-step outputs. The loss
    ``(out ** 2).sum()`` over the whole stacked output equals the BPTT sequence loss.

    Warnings
    --------
    **This path is blind to every learning-rule axis** (finding F-23). One
    whole-sequence call makes the within-call gradient exact reverse-mode, so the
    eligibility trace only enters at the sequence boundary — of which there is
    none. Every algorithm therefore returns gradients bitwise equal to BPTT, at
    every hyperparameter setting: ``D_RTRL``, ``OSTLRecurrent``, ``EProp`` at any
    ``kappa_filter_decay`` and ``pp_prop`` at any ``decay_or_rank`` are
    indistinguishable here.

    That makes this function a good test of the *compiler and ETP per-primitive
    rules* — it is exactly the right instrument for asserting an exact algorithm
    reproduces BPTT on a realistic model — and the wrong instrument for any
    assertion whose subject is a trace factorization, a temporal recursion, a
    recurrence scope, a filter, or a learning signal. For those use
    :func:`chunked_online_param_gradients` with ``chunk_size`` < ``T``, and guard
    the test with :func:`assert_gradients_differ`.

    See Also
    --------
    chunked_online_param_gradients : finite-window path; sees the trace.
    assert_gradients_differ : negative control for a knob that must matter.
    """
```

In `online_param_gradients_singlestep_naive`, replace the stale `dev/` reference (`oracle.py:180`) so the docstring resolves in-tree:

```python
    """Naive 'single-step' total gradient: sum of per-step grad((algo(x_t)**2).sum()).

    Kept to document finding F-SINGLESTEP — this recipe does NOT equal BPTT even
    for the exact D_RTRL algorithm, while the multi-step path does. This is the
    most aggressive finite window (one step), so it is maximally sensitive to
    learning-rule axes and maximally divergent from BPTT.

    See Also
    --------
    docs/specs/2026-07-25-known-limitations.md : F-SINGLESTEP and F-23.
    """
```

- [ ] **Step 6: Verify nothing regressed**

Run: `pytest braintrace/_algorithm/ -x -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add braintrace/_algorithm/oracle.py braintrace/_algorithm/oracle_test.py
git commit -m "Add oracle negative controls and document the axis-blind window

assert_model_is_live rejects a zero reference gradient; assert_gradients_differ
rejects a knob that does not move the gradient. assert_param_gradients_close now
handles nested pytrees and unit-carrying leaves, which SNN models need.

online_param_gradients now warns that a full-sequence window equals BPTT for any
algorithm (F-23), so it tests the compiler and ETP rules rather than the rule,
and points at the finite-window path for axis assertions. Also replaces an
unresolvable dev/ cross-reference."
```

---

### Task 2: Pin F-23 with an axis-discrimination meta-test

**Files:**
- Create: `braintrace/_algorithm/tests/axis_discrimination_test.py`

**Interfaces:**
- Consumes: `oracle.assert_gradients_differ`, `oracle.relative_deviation`, `oracle.online_param_gradients`, `oracle.chunked_online_param_gradients`, `oracle_models.tanh_rnn`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the test module**

Create `braintrace/_algorithm/tests/axis_discrimination_test.py`:

```python
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

"""F-23: which oracle paths can see a learning-rule axis, and which cannot.

A full-sequence ``MultiStepData`` call makes the within-call gradient exact
reverse-mode, so the eligibility trace enters only at a sequence boundary that
does not exist. Every algorithm then returns BPTT. This module pins that in both
directions, so a future test cannot silently assert an approximation's behaviour
on a path that cannot observe it -- which is how F-21 came to attribute the
effect to the model instead of the harness.
"""

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_gradients_differ,
    chunked_online_param_gradients,
    online_param_gradients,
    relative_deviation,
)
from braintrace._algorithm.oracle_models import tanh_rnn

T = 8
CHUNK = 2


def _spec():
    return tanh_rnn(n_in=3, n_rec=4, seed=0)


def _inputs():
    return jnp.asarray(np.random.RandomState(0).randn(T, 3).astype('float32'))


# Pairs of configurations that differ ONLY in a learning-rule axis value.
AXIS_PAIRS = {
    # Axis 1/2: IO-dim trace factorization strength (decay).
    'pp_prop decay': (
        lambda m: braintrace.pp_prop(m, decay_or_rank=0.99, vjp_method='multi-step'),
        lambda m: braintrace.pp_prop(m, decay_or_rank=0.01, vjp_method='multi-step'),
    ),
    # Axis 5: trace filter (kappa).
    'EProp kappa': (
        lambda m: braintrace.EProp(m, kappa_filter_decay=0.0, vjp_method='multi-step'),
        lambda m: braintrace.EProp(m, kappa_filter_decay=0.95, vjp_method='multi-step'),
    ),
    # Axis 3: recurrence scope (diagonal vs coupled).
    'recurrence scope': (
        lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        lambda m: braintrace.OSTLRecurrent(m, vjp_method='multi-step'),
    ),
}


@pytest.mark.parametrize('axis', sorted(AXIS_PAIRS))
def test_full_window_multistep_cannot_see_any_axis(axis):
    """The full-window path collapses axis-distinct configs to identical
    gradients. Pinned so the semantics are understood rather than accidental --
    if this ever starts failing, the window semantics changed and every
    assertion written against this path needs review."""
    spec, xs = _spec(), _inputs()
    lo, hi = AXIS_PAIRS[axis]
    g_lo = online_param_gradients(spec.factory, xs, algo_factory=lo)
    g_hi = online_param_gradients(spec.factory, xs, algo_factory=hi)
    rel = relative_deviation(g_lo, g_hi)
    assert rel == 0.0, (
        f'{axis}: full-window multi-step distinguished two configurations '
        f'(rel={rel:.3e}); F-23 assumed it cannot. Re-check the oracle docs.'
    )


@pytest.mark.parametrize('axis', sorted(AXIS_PAIRS))
def test_finite_window_does_see_the_axis(axis):
    """A chunked window makes the trace enter at every chunk boundary, so the
    axis becomes observable. This is the path axis assertions must use."""
    spec, xs = _spec(), _inputs()
    lo, hi = AXIS_PAIRS[axis]
    g_lo = chunked_online_param_gradients(
        spec.factory, xs, algo_factory=lo, chunk_size=CHUNK)
    g_hi = chunked_online_param_gradients(
        spec.factory, xs, algo_factory=hi, chunk_size=CHUNK)
    assert_gradients_differ(g_lo, g_hi, min_rel=1e-6)


def test_full_window_still_reproduces_bptt_for_an_exact_algorithm():
    """The corollary that makes the full-window path useful: it is the right
    instrument for asserting an exact algorithm matches BPTT."""
    from braintrace._algorithm.oracle import (
        assert_param_gradients_close,
        bptt_param_gradients,
    )
    spec, xs = _spec(), _inputs()
    g_bptt = bptt_param_gradients(spec.factory, xs)
    g_online = online_param_gradients(
        spec.factory, xs,
        algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    assert_param_gradients_close(g_online, g_bptt, atol=1e-4)
```

- [ ] **Step 2: Run the module**

Run: `pytest braintrace/_algorithm/tests/axis_discrimination_test.py -v`

Expected: PASS — all 7 tests. If `test_finite_window_does_see_the_axis[recurrence scope]` fails, that is a real signal worth recording: it would mean `_include_recurrent_mixing` is unobservable even at `chunk_size=2` on `tanh_rnn`. In that case raise `T` to 16 and lower `CHUNK` to 1; if it still fails, mark that one parametrization `xfail` with a reason naming the axis, and add it to the findings list in Task 7 as a new active finding rather than deleting the test.

- [ ] **Step 3: Commit**

```bash
git add braintrace/_algorithm/tests/axis_discrimination_test.py
git commit -m "Pin F-23: which oracle paths can see a learning-rule axis

Asserts both directions -- the full-window multi-step path collapses
axis-distinct configurations to bitwise-identical gradients, and a chunked
window separates them. This is the meta-test whose absence let F-21 attribute
the effect to the model rather than the harness."
```

---

### Task 3: Deterministic, live SNN model specs

**Files:**
- Modify: `braintrace/_algorithm/oracle_models.py`
- Test: `braintrace/_algorithm/oracle_models_test.py` (create — no sibling test exists yet)

**Interfaces:**
- Consumes: `oracle.assert_model_is_live` from Task 1.
- Produces, in `braintrace._algorithm.oracle_models`:
  - `ModelSpec` gains two optional fields: `input_scale: float = 1.0` and `needs_dt: bool = False`.
  - `ModelSpec.make_inputs(self, T: int, n_in: int, *, seed: int = 0) -> jax.Array` — batched, scaled, non-negative inputs.
  - Spec factories, each `(n_in=4, n_rec=5, seed=7) -> ModelSpec`:
    `snn_if_delta`, `snn_lif_expcu`, `snn_alif_expcu`, `snn_alif_delta`,
    `snn_lif_std_expcu`, `snn_lif_stp_expcu`, `snn_alif_expco_ei`,
    `snn_lif_expcu_heterogeneous`, `snn_alif_expcu_heterogeneous`.
  - `SNN_SPECS: dict[str, Callable[..., ModelSpec]]` — name → factory, for parametrization.

**Why a wrapper and not a fix to the layer classes:** `_etrace_model_test.py`'s constructors call unseeded `braintools.init.*`, which draws from the global `brainstate.random` stream, so `factory()` returns a different model per call (F-24). Seeding inside the wrapper keeps the layer classes untouched — they are used by many existing tests whose behaviour must not shift.

- [ ] **Step 1: Write the failing tests**

Create `braintrace/_algorithm/oracle_models_test.py`:

```python
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

"""The SNN specs must be deterministic (F-24) and live (F-25) before any
gradient assertion built on them means anything."""

import brainstate
import brainunit as u
import jax.numpy as jnp
import pytest

from braintrace._algorithm.oracle import (
    assert_model_is_live,
    flat_gradient_leaves,
)
from braintrace._algorithm.oracle_models import SNN_SPECS


@pytest.mark.parametrize('name', sorted(SNN_SPECS))
def test_snn_spec_construction_is_deterministic(name):
    """F-24: the underlying layer classes seed from the global RNG, so two
    factory() calls must be pinned to produce identical weights."""
    spec = SNN_SPECS[name]()
    with brainstate.environ.context(dt=0.1 * u.ms):
        w1 = flat_gradient_leaves(
            {k: v.value for k, v in spec.factory().states(brainstate.ParamState).items()})
        w2 = flat_gradient_leaves(
            {k: v.value for k, v in spec.factory().states(brainstate.ParamState).items()})
    assert set(w1) == set(w2)
    for key in w1:
        assert bool(jnp.allclose(w1[key], w2[key])), f'{name}: {key} differs across calls'


@pytest.mark.parametrize('name', sorted(SNN_SPECS))
def test_snn_spec_is_live(name):
    """F-25: at the default input scale these networks never spike, so their
    gradients are identically zero and every comparison is vacuous. Each spec
    records a scale that produces a non-trivial gradient."""
    spec = SNN_SPECS[name]()
    with brainstate.environ.context(dt=0.1 * u.ms):
        xs = spec.make_inputs(6, 4)
        norm = assert_model_is_live(spec.factory, xs, min_norm=1e-6)
    assert norm > 1e-6


def test_default_input_scale_is_documented_as_dead():
    """The counterpart of the above: pins *why* the scale field exists. At
    scale 1.0 the same model has a zero gradient."""
    from braintrace._algorithm.oracle import gradient_norm, bptt_param_gradients
    spec = SNN_SPECS['lif_expcu']()
    with brainstate.environ.context(dt=0.1 * u.ms):
        dead_xs = spec.make_inputs(6, 4) / spec.input_scale  # undo the scaling
        assert gradient_norm(bptt_param_gradients(spec.factory, dead_xs)) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest braintrace/_algorithm/oracle_models_test.py -v`

Expected: FAIL — `ImportError: cannot import name 'SNN_SPECS'`.

- [ ] **Step 3: Extend `ModelSpec`**

In `braintrace/_algorithm/oracle_models.py`, replace the `ModelSpec` dataclass (currently at `oracle_models.py:29-40`) with:

```python
@dataclass(frozen=True)
class ModelSpec:
    """A zero-arg model factory plus metadata about its parameters.

    ``factory()`` returns a freshly constructed, *uninitialized* model with
    deterministic weights. Callers must call
    ``brainstate.nn.init_all_states(model, batch_size=...)`` themselves.

    Attributes
    ----------
    factory : Callable[[], brainstate.nn.Module]
        Deterministic zero-arg model constructor.
    etp_param_keys : tuple of tuple
        Parameter paths routed through an ETP primitive.
    plain_param_keys : tuple of tuple
        Parameter paths used via plain JAX ops, hence excluded from ETP.
    input_scale : float, optional
        Multiplier applied by :meth:`make_inputs`. Spiking models need a scale
        well above 1.0 to reach threshold at all; below it the loss and the
        gradient are identically zero and any comparison is vacuous (F-25).
    batched_input : bool, optional
        Whether :meth:`make_inputs` emits a leading batch axis of 1. SNN layers
        concatenate the input with the recurrent spike vector, so their ranks
        must match; the rate models broadcast instead and do not need it.
    """

    factory: Callable[[], brainstate.nn.Module]
    etp_param_keys: Tuple[tuple, ...]    # routed through an ETP primitive
    plain_param_keys: Tuple[tuple, ...]  # used via plain JAX ops (excluded from ETP)
    input_scale: float = 1.0
    batched_input: bool = False

    def make_inputs(self, T: int, n_in: int, *, seed: int = 0):
        """Build a ``(T, [1,] n_in)`` input sequence at this spec's scale.

        Values are non-negative so that spiking models receive net excitatory
        drive; a zero-mean drive largely cancels and leaves the network silent.
        """
        rng = np.random.RandomState(seed)
        shape = (T, 1, n_in) if self.batched_input else (T, n_in)
        return self.input_scale * jnp.asarray(np.abs(rng.randn(*shape)).astype('float32'))
```

Add `import numpy as np` to the module imports if absent.

- [ ] **Step 4: Add the SNN spec factories**

Append to `braintrace/_algorithm/oracle_models.py`:

```python
# ---------------------------------------------------------------------------
# SNN specs: the realistic-model end of the zoo.
#
# These wrap the layer classes in ``braintrace/_etrace_model_test.py`` for
# oracle use. Two things have to be fixed at this boundary:
#
# * F-24 -- those constructors call unseeded ``braintools.init.*``, which draws
#   from the global ``brainstate.random`` stream, so ``factory()`` returns a
#   different model on every call and a BPTT-vs-online comparison would compare
#   two different networks. Each factory re-seeds before constructing.
# * F-25 -- at unit input scale the neurons never reach threshold, so the loss
#   and the gradient are identically zero. Each spec records the scale that
#   makes it live; ``oracle_models_test.py`` asserts both properties.
#
# The layer classes themselves are left untouched: many existing tests depend
# on their current behaviour.
# ---------------------------------------------------------------------------

_SNN_SEED = 7
_SNN_SCALE = 20.0


def _snn_spec(cls, n_in, n_rec, seed, **kwargs) -> ModelSpec:
    """Wrap an SNN layer class as a deterministic, live ``ModelSpec``."""

    def factory():
        brainstate.random.seed(seed)
        return cls(n_in, n_rec, **kwargs)

    return ModelSpec(
        factory=factory,
        etp_param_keys=(),   # discovered by the compiler; not asserted per-spec
        plain_param_keys=(),
        input_scale=_SNN_SCALE,
        batched_input=True,
    )


def snn_if_delta(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """IF neuron, delta synapse. Single hidden state (``num_state == 1``)."""
    from braintrace._etrace_model_test import IF_Delta_Dense_Layer
    return _snn_spec(IF_Delta_Dense_Layer, n_in, n_rec, seed)


def snn_alif_delta(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """ALIF neuron, delta synapse. Membrane + adaptation (``num_state == 2``)."""
    from braintrace._etrace_model_test import ALIF_Delta_Dense_Layer
    return _snn_spec(ALIF_Delta_Dense_Layer, n_in, n_rec, seed)


def snn_lif_expcu(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """LIF neuron, exponential current synapse. Two timescales: tau_mem, tau_syn."""
    from braintrace._etrace_model_test import LIF_ExpCu_Dense_Layer
    return _snn_spec(LIF_ExpCu_Dense_Layer, n_in, n_rec, seed)


def snn_alif_expcu(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """ALIF + exponential current synapse. Three timescales, ``num_state == 3``."""
    from braintrace._etrace_model_test import ALIF_ExpCu_Dense_Layer
    return _snn_spec(ALIF_ExpCu_Dense_Layer, n_in, n_rec, seed)


def snn_lif_std_expcu(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """LIF + short-term depression. Adds tau_std as a fourth timescale."""
    from braintrace._etrace_model_test import LIF_STDExpCu_Dense_Layer
    return _snn_spec(LIF_STDExpCu_Dense_Layer, n_in, n_rec, seed)


def snn_lif_stp_expcu(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """LIF + short-term plasticity. Adds tau_f and tau_d."""
    from braintrace._etrace_model_test import LIF_STPExpCu_Dense_Layer
    return _snn_spec(LIF_STPExpCu_Dense_Layer, n_in, n_rec, seed)


def snn_alif_expco_ei(n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED) -> ModelSpec:
    """ALIF with an excitatory/inhibitory population split and conductance
    synapses. The heterogeneous-population case: separate E and I projections
    produce several ETP relations feeding one hidden group."""
    from braintrace._etrace_model_test import ALIF_ExpCo_Dense_Layer
    return _snn_spec(ALIF_ExpCo_Dense_Layer, n_in, n_rec, seed)


def snn_lif_expcu_heterogeneous(
    n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED
) -> ModelSpec:
    """LIF whose membrane time constant differs per neuron.

    The heterogeneous-leak case: ``tau_mem`` is a length-``n_rec`` vector, so no
    single global leak exists for the transition to factor out.
    """
    from braintrace._etrace_model_test import LIF_ExpCu_Dense_Layer
    tau_mem = jnp.linspace(3.0, 12.0, n_rec) * u.ms
    return _snn_spec(LIF_ExpCu_Dense_Layer, n_in, n_rec, seed, tau_mem=tau_mem)


def snn_alif_expcu_heterogeneous(
    n_in: int = 4, n_rec: int = 5, seed: int = _SNN_SEED
) -> ModelSpec:
    """ALIF with per-neuron membrane *and* adaptation time constants."""
    from braintrace._etrace_model_test import ALIF_ExpCu_Dense_Layer
    return _snn_spec(
        ALIF_ExpCu_Dense_Layer, n_in, n_rec, seed,
        tau_mem=jnp.linspace(3.0, 12.0, n_rec) * u.ms,
        tau_a=jnp.linspace(60.0, 150.0, n_rec) * u.ms,
    )


SNN_SPECS = {
    'if_delta': snn_if_delta,
    'alif_delta': snn_alif_delta,
    'lif_expcu': snn_lif_expcu,
    'alif_expcu': snn_alif_expcu,
    'lif_std_expcu': snn_lif_std_expcu,
    'lif_stp_expcu': snn_lif_stp_expcu,
    'alif_expco_ei': snn_alif_expco_ei,
    'lif_expcu_heterogeneous': snn_lif_expcu_heterogeneous,
    'alif_expcu_heterogeneous': snn_alif_expcu_heterogeneous,
}
```

Add `import brainunit as u` to the module imports if absent.

- [ ] **Step 5: Run the tests**

Run: `pytest braintrace/_algorithm/oracle_models_test.py -v`

Expected: PASS for determinism on all 9 specs and liveness on most.

**`alif_delta` is expected to fail liveness.** Scoping measurements found it spiking at rate 0.60 while its BPTT gradient norm was still 0 — spiking is not sufficient for a non-zero gradient. Handle it by tuning that one spec rather than deleting it: raise its scale, and if the gradient stays zero, lower `V_th` via a kwarg. If neither produces a live gradient, mark it `pytest.param('alif_delta', marks=pytest.mark.xfail(reason='F-27: ALIF+delta spikes but yields a zero BPTT gradient; surrogate-gradient window never opens', strict=True))` in both parametrizations, and record F-27 as a new active finding in Task 7. Do not silently drop the spec.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_algorithm/oracle_models.py braintrace/_algorithm/oracle_models_test.py
git commit -m "Add deterministic, live SNN model specs for oracle use

Fixes F-24 and F-25 at the spec boundary rather than in the layer classes,
which many existing tests depend on. Each spec re-seeds the global RNG before
construction, so repeated factory() calls are bitwise identical, and records the
input scale needed to produce a non-zero gradient at all. Covers LIF/ALIF x
delta/ExpCu/STD/STP, an E/I conductance split, and per-neuron heterogeneous
tau_mem and tau_a. Determinism and liveness are both asserted."
```

---

### Task 4: SNN correctness — multi-state, multi-timescale, heterogeneous, E/I

**Files:**
- Create: `braintrace/_algorithm/tests/snn_model_correctness_test.py`

**Interfaces:**
- Consumes: `SNN_SPECS` and `ModelSpec.make_inputs` from Task 3; `assert_model_is_live`, `assert_param_gradients_close`, `assert_gradients_differ` from Task 1.
- Produces: nothing later tasks depend on.

This task discharges the `AGENTS.md` prose limitations *by test*: heterogeneous-population leak resolution, multi-state HiddenGroups, and approximation validity beyond a single relation.

- [ ] **Step 1: Write the test module**

Create `braintrace/_algorithm/tests/snn_model_correctness_test.py`:

```python
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

"""Gradient correctness on realistic SNN models: multi-timescale synapses,
per-neuron heterogeneous leaks, multi-state HiddenGroups, and E/I populations.

These are the claims ``AGENTS.md`` carried only as prose ("heterogeneous-
population leak resolution", "multi-state HiddenGroups"). They are discharged
here as passing tests rather than as fixes -- the compiler already handles them.

Assertion paths, per F-23: an *exact* algorithm versus BPTT uses the full-window
multi-step path, whose subject is the compiler and the ETP per-primitive rules
and which is the right instrument for that. Anything comparing *across*
algorithms uses a finite window.
"""

import brainstate
import brainunit as u
import pytest

import braintrace
from braintrace._algorithm.oracle import (
    assert_gradients_differ,
    assert_model_is_live,
    assert_param_gradients_close,
    bptt_param_gradients,
    chunked_online_param_gradients,
    online_param_gradients,
)
from braintrace._algorithm.oracle_models import SNN_SPECS

T = 6
N_IN = 4
N_REC = 5
ATOL = 1e-4

# alif_delta is excluded pending F-27 (see oracle_models_test.py); if Task 3
# made it live, delete this tuple and the `set(...) - _NOT_LIVE` below.
_NOT_LIVE = {'alif_delta'}
_LIVE_SPECS = sorted(set(SNN_SPECS) - _NOT_LIVE)


def _setup(name):
    spec = SNN_SPECS[name]()
    xs = spec.make_inputs(T, N_IN)
    return spec, xs


@pytest.mark.parametrize('name', _LIVE_SPECS)
def test_d_rtrl_matches_bptt_on_snn_models(name):
    """D_RTRL is exact, so it must reproduce BPTT on every realistic model:
    multi-timescale synapses, heterogeneous leaks, E/I populations and
    HiddenGroups with num_state from 1 to 5."""
    spec, xs = _setup(name)
    with brainstate.environ.context(dt=0.1 * u.ms):
        assert_model_is_live(spec.factory, xs, min_norm=1e-6)
        g_bptt = bptt_param_gradients(spec.factory, xs)
        g_online = online_param_gradients(
            spec.factory, xs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        assert_param_gradients_close(g_online, g_bptt, atol=ATOL)


@pytest.mark.parametrize('name', ['lif_expcu_heterogeneous',
                                 'alif_expcu_heterogeneous'])
def test_heterogeneous_leaks_do_not_break_exactness(name):
    """A per-neuron time constant leaves no single global leak for the
    transition to factor out. The compiler takes a true Jacobian, so exactness
    survives -- this is what retires the AGENTS.md prose item."""
    spec, xs = _setup(name)
    with brainstate.environ.context(dt=0.1 * u.ms):
        assert_model_is_live(spec.factory, xs, min_norm=1e-6)
        g_bptt = bptt_param_gradients(spec.factory, xs)
        g_online = online_param_gradients(
            spec.factory, xs,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        assert_param_gradients_close(g_online, g_bptt, atol=ATOL)


@pytest.mark.parametrize('name,expected_min_state', [
    ('if_delta', 1),
    ('lif_expcu', 2),
    ('alif_expcu', 3),
    ('alif_expco_ei', 5),
])
def test_multi_state_hidden_groups_are_discovered(name, expected_min_state):
    """Pins the structural facts the exactness tests above rest on: these models
    really do form multi-state HiddenGroups, so the per-state axis is exercised
    rather than assumed."""
    spec, xs = _setup(name)
    with brainstate.environ.context(dt=0.1 * u.ms):
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='multi-step')
        algo.compile_graph(xs[0])
        assert len(algo.graph.hidden_groups) >= 1
        assert max(hg.num_state for hg in algo.graph.hidden_groups) >= expected_min_state
        assert len(algo.graph.hidden_param_op_relations) >= 1


def test_ei_population_split_yields_multiple_relations():
    """The E/I model routes separate excitatory and inhibitory projections into
    one hidden group, so the compiler must record more than one ETP relation."""
    spec, xs = _setup('alif_expco_ei')
    with brainstate.environ.context(dt=0.1 * u.ms):
        model = spec.factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = braintrace.D_RTRL(model, vjp_method='multi-step')
        algo.compile_graph(xs[0])
        assert len(algo.graph.hidden_param_op_relations) >= 2


@pytest.mark.parametrize('name', ['lif_expcu', 'alif_expco_ei'])
def test_approximation_is_measurable_on_snn_models(name):
    """The genuinely approximate configuration must be *distinguishable* from
    the exact one on a realistic model -- via a finite window, which is the only
    path that can see it (F-23). This is what F-22 was really asking for."""
    spec, xs = _setup(name)
    with brainstate.environ.context(dt=0.1 * u.ms):
        g_exact = chunked_online_param_gradients(
            spec.factory, xs, chunk_size=2,
            algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        g_approx = chunked_online_param_gradients(
            spec.factory, xs, chunk_size=2,
            algo_factory=lambda m: braintrace.pp_prop(
                m, decay_or_rank=0.5, vjp_method='multi-step'))
        assert_gradients_differ(g_exact, g_approx, min_rel=1e-6)
```

- [ ] **Step 2: Run the module**

Run: `pytest braintrace/_algorithm/tests/snn_model_correctness_test.py -v`

Expected: PASS. If `test_d_rtrl_matches_bptt_on_snn_models` fails for a spec, that is a genuine compiler finding — do not loosen `ATOL`. Record it as a new finding in Task 7 with the model, the failing leaf and the observed deviation, and mark that parametrization `xfail(strict=True)`.

If `test_approximation_is_measurable_on_snn_models` fails, `chunk_size=2` is too coarse to expose the approximation on that model: try `chunk_size=1`, then raise `T` to 12. If it still cannot be exposed, that is the substantive successor to F-22 and belongs in the findings list as active — say so explicitly rather than removing the test.

- [ ] **Step 3: Commit**

```bash
git add braintrace/_algorithm/tests/snn_model_correctness_test.py
git commit -m "Assert gradient correctness on realistic SNN models

Discharges the AGENTS.md prose limitations by test: heterogeneous per-neuron
leaks, multi-timescale synapses, multi-state HiddenGroups (num_state 1 to 5) and
E/I populations all reproduce BPTT under the exact algorithm. Structural facts
are pinned alongside, so the exactness claims rest on discovered relations
rather than assumption. Cross-algorithm comparisons use a finite window per
F-23; every test carries a liveness guard."
```

---

### Task 5: Retire F-22, correct F-21

**Files:**
- Modify: `braintrace/_algorithm/tests/approx_correctness_test.py`

**Interfaces:**
- Consumes: `assert_gradients_differ`, `chunked_online_param_gradients` from Task 1.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Replace the skipped F-22 test with a live one**

In `braintrace/_algorithm/tests/approx_correctness_test.py`, delete the whole skipped test (lines 154-161, the `@pytest.mark.skip(...)` decorator through `def test_approximations_diverge_on_snn_multipopulation_DEFERRED(): pass`) and put this in its place:

```python
def test_approximations_are_measurable_through_a_finite_window():
    """F-22, retired. The finding deferred this until an "SNN multi-population
    model zoo" existed, on the theory that the model was what made the
    approximations look exact. That premise was wrong: a multi-population SNN
    model (ALIF + E/I conductance split, 3 ETP relations, num_state 5) is
    *also* bitwise-exact through the full-window path, because the cause is the
    oracle path and not the model (F-23). No model can revive a knob the
    harness cannot see.

    Through a finite window the same nominally-approximate configurations that
    F-21 finds exact become measurably different from the exact algorithm, on
    the very same rate model. That is the assertion F-22 wanted.
    """
    spec = tanh_rnn(n_in=3, n_rec=4, seed=0)
    inputs = _inputs(8, 3)
    g_exact = chunked_online_param_gradients(
        spec.factory, inputs, chunk_size=2,
        algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
    for name, algo_factory in _EXACT_ON_RATE.items():
        g_approx = chunked_online_param_gradients(
            spec.factory, inputs, chunk_size=2, algo_factory=algo_factory)
        assert_gradients_differ(g_exact, g_approx, min_rel=1e-9)
```

Add `assert_gradients_differ` and `chunked_online_param_gradients` to the existing `from braintrace._algorithm.oracle import (...)` block at the top of the file.

- [ ] **Step 2: Correct F-21's docstring**

Replace the docstring of `test_rank_decay_random_approximations_are_exact_on_rate_model_F21` (lines 144-145) with:

```python
    """F-21: these nominally-approximate configs match BPTT element-wise here.

    The cause is the *oracle path*, not the model. All four configurations run
    through ``online_param_gradients`` with ``vjp_method='multi-step'`` over the
    whole sequence, where the within-call gradient is exact reverse-mode and the
    eligibility trace never enters -- so every algorithm returns BPTT at every
    hyperparameter setting (F-23). An earlier reading of this test attributed the
    exactness to the model being a single-HiddenGroup rate model and concluded
    that an SNN multi-population zoo was needed to expose the bias (F-22); that
    conclusion was wrong, and F-22 is retired by
    ``test_approximations_are_measurable_through_a_finite_window`` below.

    The test is kept because the equality is still a real property of this path
    and worth pinning.
    """
```

Also update the module docstring's `(F-21/F-22)` reference (line 19) to `(F-21/F-22/F-23)` and the comment block at lines 127-130, replacing "We assert exactness here and defer the genuine approximation stress to an SNN multi-population model zoo (F-22)." with "We assert exactness here; the cause is the full-window oracle path (F-23), and the genuine approximation stress is asserted through a finite window below."

- [ ] **Step 3: Run the module**

Run: `pytest braintrace/_algorithm/tests/approx_correctness_test.py -v`

Expected: PASS, with no skipped tests. Confirm with `pytest braintrace/_algorithm/tests/approx_correctness_test.py -v -rs` that the skip list is empty.

- [ ] **Step 4: Commit**

```bash
git add braintrace/_algorithm/tests/approx_correctness_test.py
git commit -m "Retire F-22 and correct F-21's attribution

F-22 deferred measuring the approximations' bias until an SNN multi-population
model zoo existed. The premise was wrong: such a model is bitwise-exact on the
same full-window path, because the path is what cannot see the approximation.
Replaces the skipped placeholder with a live assertion through a finite window,
on the same rate model, and corrects F-21's docstring to name the harness rather
than the model as the cause."
```

---

### Task 6: Fix the conv-bias cotangent shape in the IO-dim path

**Files:**
- Modify: `braintrace/_algorithm/io_dim_vjp.py`
- Test: `braintrace/_algorithm/oracle_test.py:488-517`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing later tasks depend on.

**Diagnosis (already done — do not re-derive).** `pp_prop` on a conv layer with a trainable bias raises:

```
ValueError: Custom VJP bwd rule must produce an output with the same type as the
args tuple of the primal function, but at output[1][('b',)] the bwd rule produced
an output of type float64[1,8,3] corresponding to an input of type float64[3]
```

`_conv_xy_to_dw` in `braintrace/_op/conv.py:315` deliberately returns the bias Jacobian **per output position**; its docstring states "We store this per-position (no spatial sum here); the sum is performed inside `_conv_dt_to_t` during trace propagation". The param-dim path calls `_conv_dt_to_t` and so reduces it — which is why `D_RTRL` handles conv+bias exactly. The IO-dim solver calls `xy_to_dw` directly at solve time (`_solve_IO_dim_weight_gradients`, `io_dim_vjp.py:418`) and never applies that reduction, so the un-reduced array reaches JAX's `custom_vjp` type check.

Fix the IO-dim solver, not the conv rule: reducing a produced gradient leaf to its parameter's own shape by summing the extra leading axes is the standard broadcast-gradient reduction and is primitive-agnostic, so it does not whitelist conv.

- [ ] **Step 1: Turn the pinned limitation into a failing exactness test**

In `braintrace/_algorithm/oracle_test.py`, replace `test_pp_prop_conv_bias_known_limitation` (lines 488-517) with:

```python
def test_pp_prop_conv_bias_matches_bptt():
    """Formerly ``test_pp_prop_conv_bias_known_limitation`` (finding F-26).

    ``_conv_xy_to_dw`` returns the bias Jacobian per output position by design;
    the param-dim path reduces it in ``_conv_dt_to_t``, but the IO-dim solver
    calls ``xy_to_dw`` directly at solve time and used to hand the un-reduced
    ``(batch, *spatial, out_ch)`` array back to ``custom_vjp``, which rejected it
    against the bias's own ``(out_ch,)``. The IO-dim solver now reduces every
    produced leaf to its parameter's shape. At T=1 there is no history to factor,
    so pp_prop must match BPTT exactly.
    """
    factory, seed = _FAMILIES['conv_nwc_bias']
    with brainstate.environ.context(precision=64):
        xs = _xs_for('conv_nwc_bias', 1, seed)
        g_bptt = bptt_param_gradients(factory, xs)
        g_online = online_param_gradients_singlestep_naive(
            factory, xs,
            algo_factory=lambda m: braintrace.pp_prop(
                m, decay_or_rank=0.9, vjp_method='single-step'),
        )
        for key in g_bptt:
            rel = _rel_err(g_bptt[key], g_online[key])
            assert rel < _TOL, f'conv_nwc_bias T=1 {key}: pp_prop vs BPTT rel={rel:.3e}'
```

Also re-include `conv_nwc_bias` in the two parametrizations that exclude it, at lines 443 and 465: change `sorted(set(_FAMILIES) - {'conv_nwc_bias'})` to `sorted(_FAMILIES)` in both, and delete the paragraph in `test_pp_prop_singlestep_exact_at_t1_across_families`'s docstring beginning "``conv_nwc_bias`` is excluded here".

- [ ] **Step 2: Run to verify it fails**

Run: `pytest braintrace/_algorithm/oracle_test.py -k conv -v`

Expected: FAIL with `ValueError: Custom VJP bwd rule must produce an output with the same type ...`.

- [ ] **Step 3: Reduce produced leaves to the parameter's shape**

In `braintrace/_algorithm/io_dim_vjp.py`, add this helper immediately above `_solve_IO_dim_weight_gradients` (before line 418):

```python
def _reduce_to_param_shape(grad: jax.Array, param: jax.Array) -> jax.Array:
    """Sum a produced gradient leaf down to its parameter's own shape.

    Some ``xy_to_dw`` rules deliberately return a *per-position* instantaneous
    Jacobian for parameters that broadcast over positions — a conv bias is the
    canonical case, and :func:`braintrace._op.conv._conv_xy_to_dw` documents that
    the spatial sum is deferred. The param-dim path performs that sum during
    trace propagation; the IO-dim path contracts at solve time and must perform
    it here, or ``custom_vjp`` rejects the shape mismatch (finding F-26).

    This is the standard broadcast-gradient reduction — sum the extra leading
    axes, then any axis the parameter holds as a singleton — so it is
    primitive-agnostic and a no-op whenever the shapes already agree.
    """
    g_shape = u.math.shape(grad)
    p_shape = u.math.shape(param)
    if g_shape == p_shape:
        return grad
    extra = len(g_shape) - len(p_shape)
    if extra < 0:
        return grad
    out = u.math.sum(grad, axis=tuple(range(extra))) if extra else grad
    squeeze = tuple(
        i for i, (gd, pd) in enumerate(zip(u.math.shape(out), p_shape))
        if pd == 1 and gd != 1
    )
    if squeeze:
        out = u.math.sum(out, axis=squeeze, keepdims=True)
    return out
```

Then apply it where the solver hands its result to the routing step. Replace the final line of the `for group in relation.hidden_groups:` body (`io_dim_vjp.py:538`, `_route_grads_by_path(relation, dg_dict, weight_vals, dG_weights)`) with:

```python
            # Reduce per-position leaves to their parameter's own shape before
            # routing; see _reduce_to_param_shape (finding F-26).
            dg_dict = {
                key: _reduce_to_param_shape(value, weights_dict[key])
                for key, value in dg_dict.items()
            }
            _route_grads_by_path(relation, dg_dict, weight_vals, dG_weights)
```

- [ ] **Step 4: Run the conv tests**

Run: `pytest braintrace/_algorithm/oracle_test.py -k conv -v`

Expected: PASS.

- [ ] **Step 5: Verify no other primitive regressed**

The reduction touches every IO-dim gradient, so the whole primitive matrix must be re-run.

Run: `pytest braintrace/_algorithm/oracle_test.py braintrace/_algorithm/io_dim_vjp_test.py braintrace/_algorithm/pp_prop_test.py braintrace/_op/ -q`

Expected: PASS. If a `sparse`, `lora`, `grouped` or `elemwise` test now fails, the reduction is too eager for that layout — narrow it to apply only when `len(g_shape) > len(p_shape)` (drop the singleton-squeeze branch) and re-run.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_algorithm/io_dim_vjp.py braintrace/_algorithm/oracle_test.py
git commit -m "Fix F-26: reduce per-position gradient leaves in the IO-dim solver

Conv's xy_to_dw rule returns the bias Jacobian per output position by design and
defers the spatial sum to _conv_dt_to_t, which only the param-dim path calls.
The IO-dim solver contracts at solve time, so it handed the un-reduced
(batch, *spatial, out_ch) array to custom_vjp, which rejected it against the
bias's own (out_ch,). Reduces every produced leaf to its parameter's shape with
the standard broadcast-gradient reduction -- primitive-agnostic, so conv is not
whitelisted -- and promotes the pinned limitation to an exactness assertion.
conv_nwc_bias rejoins the two pp_prop family parametrizations."
```

---

### Task 7: The in-tree known-limitations list

**Files:**
- Create: `docs/specs/2026-07-25-known-limitations.md`

**Interfaces:**
- Consumes: the outcomes of Tasks 2-6, including any new findings they surfaced (F-27 and any `xfail` recorded there).
- Produces: a document Task 8 links to from `AGENTS.md`.

Write this task **last among the code tasks**, so every status below is transcribed from an observed test result rather than predicted.

- [ ] **Step 1: Confirm every claim against the suite**

Run: `pytest braintrace/ -q -rs 2>&1 | tail -30`

Record: the pass/fail/skip counts, and the reason string of every skip and xfail. Each row you write in the next step must correspond to something in this output or to a test you can name.

- [ ] **Step 2: Write the document**

Create `docs/specs/2026-07-25-known-limitations.md`:

```markdown
# Known limitations — verified findings list

Status: living document
Baseline: commit `bc153da` plus the P1 phase
Supersedes: the untracked list formerly at
`dev/superpowers/specs/2026-05-26-comprehensive-test-strategy-design.md`, which
is gitignored and absent from the repository

This is the backlog of expected-failure and improvement items that `AGENTS.md`
§ Known limitations refers to. Every entry is verified against the current test
suite, not transcribed. An entry is **active** only if a test or a documented
scope boundary pins it today.

## Status legend

- **resolved** — the claim no longer holds; a passing test pins the resolution.
- **dead** — the claim described code that no longer exists.
- **active** — the claim still holds; a test or documented boundary pins it.
- **misattributed** — the observation was real, the stated cause was not.

## Findings

| ID | Claim | Status | Pinned by |
|---|---|---|---|
| F-01 / F-04 | multi-state (`num_state >= 2`) HiddenGroups mishandled | resolved | `snn_model_correctness_test.py::test_multi_state_hidden_groups_are_discovered` and `::test_d_rtrl_matches_bptt_on_snn_models` (num_state 1-5) |
| F-07 / F-08 / F-09 | OTTT / OTPE approximation bias | dead | removed with those algorithms in 0.2.5 |
| F-17 | implementation facts drifted from the instruction file | resolved | `braintrace/__init___test.py` Task 7 |
| F-19 / F-20 | OTTT / OSTTP exactness gaps | dead | removed with those algorithms in 0.2.5 |
| F-21 | rank / decay / random-feedback configs are exact on rate models | misattributed — the cause is the oracle path (F-23), not the model | `approx_correctness_test.py::test_rank_decay_random_approximations_are_exact_on_rate_model_F21`, docstring corrected |
| F-22 | exposing approximation bias needs an SNN multi-population zoo | **retired** — premise false; a multi-population SNN model is bitwise-exact on the same path | replaced by `approx_correctness_test.py::test_approximations_are_measurable_through_a_finite_window` |
| F-23 | the full-window multi-step oracle path is blind to every learning-rule axis | active, **by design** — documented, not a defect | `axis_discrimination_test.py`, both directions; warned in `online_param_gradients`' docstring |
| F-24 | `_etrace_model_test.py` factories are non-deterministic (unseeded global RNG) | active in the layer classes; neutralised for oracle use | `oracle_models_test.py::test_snn_spec_construction_is_deterministic` |
| F-25 | SNN models are silent at unit input scale, so comparisons are vacuous | active as a property; guarded | `oracle_models_test.py::test_snn_spec_is_live`, `test_default_input_scale_is_documented_as_dead`, and `assert_model_is_live` |
| F-26 | `pp_prop` / IODim raised on conv + trainable bias | resolved | `oracle_test.py::test_pp_prop_conv_bias_matches_bptt` |
| F-SCAN / F-SCAN-WEIGHT | weight inside control flow raised `KeyError` | resolved | `braintrace/_compiler/base_test.py::test_error_message_identifies_weight_variable` |
| F-SINGLESTEP | naive single-step summation does not equal BPTT even for an exact algorithm | resolved into a positive direction-alignment assertion | `braintrace/_algorithm/oracle_test.py` |

## Mapping from the `AGENTS.md` prose list

`AGENTS.md` described its limitations in prose. Each item maps onto a finding
above, so nothing survives only as prose:

| Prose item | Disposition |
|---|---|
| approximation-mode validity beyond shallow depth | successor to F-21 / F-23; measured through a finite window |
| heterogeneous-population leak resolution | resolved — `snn_model_correctness_test.py::test_heterogeneous_leaks_do_not_break_exactness` |
| target-signal threading under JIT | dead — died with `OSTTP`'s `y_target` path |
| single-readout / feedback-shape assumptions | see F-28 below |
| gaps in cross-algorithm equivalence coverage | successor to F-23; `axis_discrimination_test.py` covers the pairs |

## Notes on F-23

This is the load-bearing finding, and it is a property rather than a bug. A
whole-sequence `MultiStepData` call makes the within-call gradient exact
reverse-mode, so the eligibility trace enters only at a sequence boundary that
does not exist. `chunked_online_param_gradients` has documented this all along
— "this is the oracle that actually validates trace correctness" — but the
assertions written against `online_param_gradients` did not heed it, which is
how F-21 and F-22 reached the wrong cause.

The rule that follows: an assertion whose subject is a trace factorization, a
temporal recursion, a recurrence scope, a filter or a learning signal must use a
finite window and must be guarded by `assert_gradients_differ`. An assertion
whose subject is the compiler or an ETP per-primitive rule may use the
full-window path, and that is most of the ~95 `multi-step` call sites in the
suite.
```

- [ ] **Step 3: Reconcile the document with reality**

For each row, confirm the named test exists and has the stated outcome:

```bash
pytest braintrace/_algorithm/tests/axis_discrimination_test.py \
       braintrace/_algorithm/tests/snn_model_correctness_test.py \
       braintrace/_algorithm/oracle_models_test.py \
       braintrace/_algorithm/oracle_test.py \
       braintrace/_algorithm/tests/approx_correctness_test.py -q -rs
```

Then fix the document to match: add any finding Tasks 3-6 surfaced (F-27 for a
non-live spec, F-28 for the single-readout / feedback-shape probe, or a new
compiler finding), and correct any status you cannot back with a named test. If
the single-readout / feedback-shape assumption turned out to be a non-issue, say
so under F-28 with the test that shows it and mark it resolved; if it was never
probed, mark F-28 "unverified — carried forward" rather than claiming either.

- [ ] **Step 4: Commit**

```bash
git add docs/specs/2026-07-25-known-limitations.md
git commit -m "Add the in-tree verified known-limitations list

Replaces the untracked dev/ list AGENTS.md pointed at, which is gitignored and
absent. Every entry is verified against the current suite and names the test
that pins it. Records F-23 as the load-bearing finding -- the full-window
oracle path is axis-blind by design -- and maps each AGENTS.md prose item onto a
finding so none survives only as prose."
```

---

### Task 8: Roadmap and AGENTS.md updates

**Files:**
- Modify: `docs/specs/2026-07-25-algorithm-axes-roadmap.md`
- Modify: `AGENTS.md:148-156`

**Interfaces:**
- Consumes: `docs/specs/2026-07-25-known-limitations.md` from Task 7.
- Produces: nothing.

- [ ] **Step 1: Revise the P2 and P3 acceptance criteria**

The roadmap's P2 and P3 criteria are stated as element-wise equality against
existing algorithms. Measured through the full-window path, those comparisons
pass for any algorithm and would not guard the refactor at all.

In the P2 **Acceptance** block, replace the first bullet:

```markdown
- Element-wise equality against golden values frozen *before* the refactor, for
  all five surviving algorithms — frozen and compared through a **finite-window**
  oracle path (`chunked_online_param_gradients`, `chunk_size` < `T`). The
  full-window multi-step path returns BPTT for every algorithm regardless of its
  axis coordinates (F-23), so golden values captured there would be identical
  across all five and would guard nothing. Guard each comparison with
  `assert_gradients_differ` between any two presets that differ in an axis.
```

In the P3 **Acceptance (two-sided squeeze)** block, replace the first two
bullets:

```markdown
- `n = 1` equals the current `D_RTRL` element-wise, compared through a
  finite-window path (regression guard). On the full-window path this holds for
  every algorithm and so proves nothing (F-23).
- Whatever configuration expresses `recurrence_scope = coupled` after the
  refactor equals the current `OSTLRecurrent` element-wise — again through a
  finite-window path — regardless of whether that configuration turns out to be
  a point on the `n` scale or a sibling value beside it. `axis_discrimination_test.py`
  pins that `D_RTRL` and `OSTLRecurrent` *are* distinguishable on that path, so
  this comparison has content.
```

Leave the `n >= graph diameter` bullet unchanged: BPTT equality there is a
statement about the total gradient and is correct on either path.

- [ ] **Step 2: Rewrite the P1 section to match what was built**

Replace the body of the `### P1 — Compiler: multi-timescale and heterogeneous
populations` section (roadmap lines 233-260) with:

```markdown
### P1 — Verification harness and the limitation list — **done**

Retitled during implementation. P1 was scoped as compiler work; measurement
showed the compiler already passes every one of its targets, and that the
instrument was what failed. Full spec:
[`2026-07-25-p1-verification-harness.md`](2026-07-25-p1-verification-harness.md).

Delivered: negative-control oracle helpers (`assert_model_is_live`,
`assert_gradients_differ`) and pytree/unit-aware gradient comparison;
documented window semantics on every oracle entry point; deterministic and live
SNN model specs; an axis-discrimination meta-test; gradient-correctness tests on
realistic SNN models (multi-timescale, per-neuron heterogeneous leaks,
`num_state` 1-5, E/I populations); the conv-bias IO-dim fix; and the in-tree
findings list at [`2026-07-25-known-limitations.md`](2026-07-25-known-limitations.md).

`hidden_group.py` was **not** modified — no defect was found in its Jacobian
path — so P3 inherits today's representation and risk 2 below is retired rather
than mitigated.

**Acceptance:** met. The reconstructed limitation list is committed under
`docs/specs/`; every item has a passing test or a documented scope boundary; the
one skipped test (F-22) is retired rather than deferred.
```

- [ ] **Step 3: Add the lessons-learned section**

Append to the end of `docs/specs/2026-07-25-algorithm-axes-roadmap.md`:

```markdown
## Lessons learned during implementation

Recorded during P1, against commit `bc153da`. These are the things the roadmap
got wrong or could not have known, kept here because later phases rest on them.

1. **The instrument was the defect, not the compiler.** P1 was scoped as
   compiler work on multi-timescale and heterogeneous populations. Every one of
   those targets already passed: HiddenGroups with `num_state` up to 5,
   per-neuron heterogeneous `tau_mem` and `tau_a`, multi-timescale synapses
   (`tau_mem` / `tau_syn` / `tau_a` / `tau_std` / `tau_f` / `tau_d`), and E/I
   population splits all reproduce BPTT. Measure before scoping a phase around
   a suspected defect.

2. **A full-sequence multi-step VJP is BPTT, so it cannot see any axis**
   (F-23). `D_RTRL`, `OSTLRecurrent`, `EProp` at any `kappa_filter_decay` and
   `pp_prop` at any `decay_or_rank` return bitwise-identical gradients through
   `online_param_gradients`. This is correct semantics — no truncation, nothing
   to approximate — but it silently voids any assertion whose subject is a
   learning-rule axis. `chunked_online_param_gradients` documented this from the
   start; the tests written against the other entry point did not heed it. **Any
   acceptance criterion in this roadmap phrased as "equals X element-wise" must
   name a finite-window path.**

3. **A wrong cause propagates further than a wrong measurement.** F-21 observed
   real behaviour and attributed it to the model being a single-HiddenGroup rate
   model. F-22 then deferred an entire work item — an SNN multi-population model
   zoo — on that attribution, and P5 inherited it. Running exactly that model
   (`ALIF_ExpCo`, 3 relations, `num_state` 5, `T` 20) reproduced the same
   bitwise exactness, so no zoo could ever have helped. The fix was one
   parameter: `chunk_size`.

4. **Vacuous tests look like passing tests.** Three distinct ways a gradient
   assertion in this repository can assert nothing: the reference gradient is
   zero (a silent SNN — 8 of 9 probed configurations never reach threshold at
   unit input scale); the model differs between the two sides of the comparison
   (`_etrace_model_test.py` constructors draw from the unseeded global RNG, so
   `factory()` returns a different network per call); or the oracle path cannot
   see the knob under test (F-23). Spiking is *not* sufficient for the first —
   `ALIF_Delta` spiked at rate 0.60 with a zero BPTT gradient. Hence
   `assert_model_is_live` keys on gradient norm, and every new axis assertion
   carries `assert_gradients_differ`.

5. **`xy_to_dw` rules may return un-reduced, per-position leaves.** Conv's rule
   deliberately defers the spatial sum of the bias Jacobian to `_conv_dt_to_t`,
   which only the param-dim path calls. Any new solver that consumes `xy_to_dw`
   directly must reduce produced leaves to the parameter's own shape (F-26).
   This matters for P2: the axis strategies will introduce new contraction
   paths.

6. **The removed algorithms took findings with them, and left stale
   cross-references behind.** F-07/F-08/F-09/F-19/F-20 died with OTTT / OTPE /
   OSTTP, and the `AGENTS.md` prose item "target-signal threading under JIT"
   died with `OSTTP`'s `y_target` path. Meanwhile a docstring still pointed at
   `dev/superpowers/specs/...`, a path that is gitignored and absent. Findings
   lists must live in-tree.
```

- [ ] **Step 4: Repoint `AGENTS.md`**

Replace the `## Known limitations` section body (`AGENTS.md:150-156`) with:

```markdown
First-cut SNN algorithms pass smoke and cross-checks but carry approximation
edges and rough spots. These are enumerated, verified against the test suite,
and mapped to concrete improvement actions in the findings list at
`docs/specs/2026-07-25-known-limitations.md`. Treat that list as the backlog of
expected-failure and improvement items rather than duplicating it here.

One rule from that list is load-bearing enough to state here: a gradient
assertion whose subject is a **learning-rule property** — a trace
factorization, a temporal recursion, a recurrence scope, a filter, a learning
signal — must be measured through a *finite-window* oracle path. A
whole-sequence multi-step VJP has no truncation left to approximate and returns
BPTT for every algorithm at every hyperparameter, so such an assertion passes
vacuously there. Assertions about the compiler or an ETP per-primitive rule may
use the whole-sequence path.
```

- [ ] **Step 5: Verify the docs are consistent**

```bash
grep -rn "dev/" AGENTS.md docs/specs/ braintrace/ --include=*.md --include=*.py | grep -v "\.pyc"
```

Expected: only the historical mentions inside the roadmap and the known-limitations file that *describe* the absent `dev/` list. No live cross-reference telling a reader to go read a file under `dev/`.

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md docs/specs/2026-07-25-algorithm-axes-roadmap.md
git commit -m "Record P1 outcomes: revised P2/P3 criteria, lessons learned

P2 and P3 acceptance criteria now name a finite-window oracle path. As written
they were element-wise equalities measured through the full-window path, where
every algorithm returns BPTT regardless of its axis coordinates -- they would
have passed without guarding the refactor at all.

Rewrites the P1 section to match what was built, retires risk 2 (no
hidden_group.py defect was found, so P3 inherits today's representation), adds a
lessons-learned section, and repoints AGENTS.md off the absent dev/ list."
```

---

### Task 9: Full-suite verification

**Files:** none modified.

**Interfaces:**
- Consumes: every preceding task.
- Produces: the evidence for the phase's acceptance claim.

- [ ] **Step 1: Run the full test suite**

Run: `pytest braintrace/ -q -rs`

Expected: all tests pass. The P0 baseline was 2062 passed, 1 skipped; the new
count is higher and the skip count is **0**, because Task 5 retired the only
skip. If any test fails, fix it — do not adjust the claim.

- [ ] **Step 2: Run the type checker**

Run: `mypy braintrace`

Expected: clean, matching the P0 baseline. The likely new complaints are missing
annotations on the helpers added in Task 1 and `_reduce_to_param_shape` in Task
6; add the annotations rather than an ignore.

- [ ] **Step 3: Record the numbers and update the spec's status**

In `docs/specs/2026-07-25-p1-verification-harness.md`, change the header
`Status: approved, ready for implementation` to `Status: implemented` and append
the observed counts to the **Testing strategy** section:

```markdown
**Verified:** `pytest braintrace/` → <N> passed, 0 skipped; `mypy braintrace`
clean.
```

Fill in the real number from Step 1. Do not write a number you did not observe.

- [ ] **Step 4: Commit**

```bash
git add docs/specs/2026-07-25-p1-verification-harness.md
git commit -m "Mark the P1 spec implemented with verified suite counts"
```

---

## Self-review

**Spec coverage.** D1 → Task 7. D2 → Tasks 1 and 2. D3 → Task 3. D4 and D5 →
Task 4. D6 → Task 5. D7 → Task 6. D8 → Task 8. The spec's testing-strategy
invariants (liveness, discrimination) are enforced by the helpers from Task 1 and
used in Tasks 2, 4 and 5. The verification bar is Task 9. The user's instruction
to record lessons learned in the roadmap is Task 8 Step 3.

**Placeholders.** None. Every code step carries the code; every command carries
its expected output. The three conditional branches (Task 3 Step 5's non-live
`alif_delta`, Task 4 Step 2's possible compiler finding, Task 6 Step 5's
over-eager reduction) each name the concrete fallback and require recording a
finding rather than loosening an assertion.

**Type consistency.** `flat_gradient_leaves`, `gradient_norm`,
`relative_deviation`, `assert_model_is_live`, `assert_gradients_differ` are
defined in Task 1 and used with those exact names in Tasks 2, 3, 4 and 5.
`ModelSpec.make_inputs(T, n_in, *, seed=0)` and `SNN_SPECS` are defined in Task
3 and consumed with that signature in Tasks 3 and 4. `_reduce_to_param_shape` is
local to Task 6. `input_scale` is used by Task 3's
`test_default_input_scale_is_documented_as_dead` and defined in the same task.
