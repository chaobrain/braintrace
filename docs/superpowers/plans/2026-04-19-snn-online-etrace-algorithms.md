# SNN Online Learning ETraceAlgorithm Classes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship five first-class `ETraceVjpAlgorithm` subclasses (`EProp`, `OSTL`, `OTPE`, `OTTT`, `OSTTP`) that encapsulate the paper-faithful mathematics of five SNN online learning algorithms, in a new subpackage `braintrace/_snn_algorithms/`.

**Architecture:** Reuse existing `ETPPrimitive` + compiler + VJP machinery. Add one overridable hook `_compute_learning_signal` to `ETraceVjpAlgorithm` so OSTTP can swap reverse-AD's `dL/dh` for a target projection. Implement OTPE's cross-layer coupling inside `_solve_weight_gradients` without relaxing the compiler's "no W→W→h" invariant.

**Tech Stack:** `brainstate` (state + Module), `saiunit` (`u`) (unit handling), JAX (`jax.custom_vjp`, `jnp.einsum`, `jax.lax.scan`, `jax.lax.stop_gradient`), pytest.

**Spec reference:** `docs/superpowers/specs/2026-04-19-snn-online-etrace-algorithms-design.md`

---

## File Structure

**New files (all under `braintrace/_snn_algorithms/`):**

| File | Responsibility |
|---|---|
| `__init__.py` | Public re-exports: `EProp`, `OSTL`, `OTPE`, `OTTT`, `OSTTP`, `FixedRandomFeedback`, `KappaFilter`, `PresynapticTrace` |
| `_common.py` | `PresynapticTrace`, `FixedRandomFeedback`, `KappaFilter`, `extract_y_target`, `_resolve_leak` helper |
| `_common_test.py` | Unit tests for the helpers |
| `ostl.py` | `class OSTL(...)` — thin delegator onto `ParamDimVjpAlgorithm` (with-H) or `IODimVjpAlgorithm` (without-H) |
| `ostl_test.py` | OSTL tests |
| `e_prop.py` | `class EProp(ParamDimVjpAlgorithm)` — D_RTRL + κ-filter + feedback knob |
| `e_prop_test.py` | EProp tests |
| `ottt.py` | `class OTTT(ETraceVjpAlgorithm)` — presynaptic λ-accumulator only |
| `ottt_test.py` | OTTT tests |
| `osttp.py` | `class OSTTP(ParamDimVjpAlgorithm)` — D_RTRL trace + `_compute_learning_signal` override |
| `osttp_test.py` | OSTTP tests |
| `otpe.py` | `class OTPE(ETraceVjpAlgorithm)` — leaky-additive R_hat, cross-layer solve |
| `otpe_test.py` | OTPE tests |
| `integration_test.py` | Loss-decreases smoke test for each of the 5 classes |
| `cross_check_test.py` | Cross-class equivalence proofs (OSTL-without-H == D_RTRL, OTPE-λ0 == OSTL, EProp-no-κ == D_RTRL) |
| `fixtures/bptt_gradients_tiny_lsnn.pkl` | Reference BPTT gradient dump (produced by `fixtures/generate_bptt_reference.py`) |
| `fixtures/generate_bptt_reference.py` | One-shot generator for the fixture above |

**Existing files modified:**

| File | Change |
|---|---|
| `braintrace/_etrace_vjp/base.py` | Add `_compute_learning_signal` hook; wire into `_update_fn_bwd` before `_solve_weight_gradients` |
| `braintrace/_etrace_vjp/base_test.py` | Add default-hook-is-identity test; add override-hook test |
| `braintrace/_etrace_compiler/diagnostics.py` | Add `DiagnosticKind.SPIKE_RESET_LEAKAGE` |
| `braintrace/_etrace_compiler/graph.py` | Emit `SPIKE_RESET_LEAKAGE` when heuristic matches (best-effort jaxpr inspection) |
| `braintrace/__init__.py` | Re-export new symbols from `_snn_algorithms` |
| `CLAUDE.md` | Add `_snn_algorithms/` entry to the package tree |

---

## Conventions

- **TDD cycle** per task: write failing test → run & verify FAIL → minimal implementation → run & verify PASS → commit.
- **Commit message** format follows existing repo style: `feat/fix/test/docs: <imperative summary>`. No `Co-Authored-By` trailers (per user global rule).
- **Test location:** next to the source, same package (matches `_etrace_vjp/*_test.py`).
- **Imports:** absolute (`from braintrace._snn_algorithms._common import ...`).
- **Caveman mode:** commit messages and tests are normal English; code follows project style.
- **Do NOT run `pytest braintrace/` (full suite) between every step** — it is ~1250 tests. Run the targeted test(s) for each step. Full suite once per phase before committing the phase cap.

---

## Phase 1 — Base-class `_compute_learning_signal` hook

Goal: add the overridable hook without regressing `D_RTRL`/`pp_prop`. Default = identity.

### Task 1.1: Add failing test for default-identity hook

**Files:**
- Modify: `braintrace/_etrace_vjp/base_test.py`

- [ ] **Step 1: Write the failing test**

Append to `braintrace/_etrace_vjp/base_test.py` (put near other signature-shape tests):

```python
class TestComputeLearningSignalHook(unittest.TestCase):
    def test_default_hook_is_identity(self):
        """Default `_compute_learning_signal` returns input unchanged."""
        import jax.numpy as jnp
        from braintrace._etrace_vjp.base import ETraceVjpAlgorithm

        algo = ETraceVjpAlgorithm.__new__(ETraceVjpAlgorithm)  # bypass __init__ for pure-function probe
        dl2h = [jnp.ones((2, 3)), jnp.zeros((2, 5))]
        out = ETraceVjpAlgorithm._compute_learning_signal(algo, dl2h, args=())
        assert isinstance(out, (list, tuple))
        assert len(out) == 2
        assert jnp.allclose(out[0], dl2h[0])
        assert jnp.allclose(out[1], dl2h[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest braintrace/_etrace_vjp/base_test.py::TestComputeLearningSignalHook::test_default_hook_is_identity -v
```
Expected: FAIL with `AttributeError: ... has no attribute '_compute_learning_signal'`.

- [ ] **Step 3: Add the hook method to `ETraceVjpAlgorithm`**

In `braintrace/_etrace_vjp/base.py`, add this method to the `ETraceVjpAlgorithm` class (place it right above `_solve_weight_gradients`):

```python
    def _compute_learning_signal(
        self,
        dl_to_hidden_from_autodiff: Sequence[jax.Array],
        args: tuple,
    ) -> Sequence[jax.Array]:
        """Override hook. Return the learning signal used by `_solve_weight_gradients`.

        Default returns the reverse-AD gradient unchanged. Subclasses that need
        target projection (OSTTP) or any other alternative can override this.

        Args:
            dl_to_hidden_from_autodiff: Sequence of per-hidden-group gradients produced
                by reverse-AD inside `_update_fn_bwd`.
            args: The exact `*args` tuple passed to the most recent `update()` call,
                made available so subclasses can pull auxiliary tensors (e.g. y_target)
                that were stashed elsewhere (e.g. on ``self``).

        Returns:
            Sequence of per-hidden-group gradient arrays, one per HiddenGroup. Must
            match the shape and length of ``dl_to_hidden_from_autodiff``.
        """
        return dl_to_hidden_from_autodiff
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest braintrace/_etrace_vjp/base_test.py::TestComputeLearningSignalHook::test_default_hook_is_identity -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_etrace_vjp/base.py braintrace/_etrace_vjp/base_test.py
git commit -m "feat: add _compute_learning_signal hook to ETraceVjpAlgorithm"
```

---

### Task 1.2: Wire hook into `_update_fn_bwd`

**Files:**
- Modify: `braintrace/_etrace_vjp/base.py` (the `_update_fn_bwd` method, around lines 539–555)
- Modify: `braintrace/_etrace_vjp/base_test.py`

- [ ] **Step 1: Write the failing override-hook test**

Append to `braintrace/_etrace_vjp/base_test.py`:

```python
    def test_override_hook_replaces_learning_signal(self):
        """Subclass override is used instead of reverse-AD dl/dh."""
        import brainstate
        import braintrace
        import jax
        import jax.numpy as jnp
        from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm

        class Mini(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.eye(3))
                self.h = brainstate.ShortTermState(jnp.zeros((4, 3)))

            def update(self, x):
                self.h.value = jax.nn.tanh(self.h.value + braintrace.matmul(x, self.w.value))
                return self.h.value

        captured = {}

        class ConstantSignalAlgo(ParamDimVjpAlgorithm):
            def _compute_learning_signal(self, dl_autodiff, args):
                captured['autodiff'] = dl_autodiff
                return [jnp.ones_like(a) for a in dl_autodiff]

        net = Mini()
        algo = ConstantSignalAlgo(net)
        x0 = jnp.ones((4, 3))
        algo.compile_graph(x0)
        algo.init_etrace_state()

        def loss(x):
            out = algo.update(x)
            return (out ** 2).sum()

        grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x0)
        assert 'autodiff' in captured  # hook was invoked
        # Constant-ones signal yields a specific, non-zero gradient on w.
        w_grad = grads[next(iter(grads))]
        assert jnp.any(w_grad != 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest braintrace/_etrace_vjp/base_test.py::TestComputeLearningSignalHook::test_override_hook_replaces_learning_signal -v
```
Expected: FAIL (`captured` never populated because the hook is not yet called in `_update_fn_bwd`).

- [ ] **Step 3: Wire the hook into `_update_fn_bwd`**

In `braintrace/_etrace_vjp/base.py`, modify `_update_fn_bwd`. Find the block that assembles `dl2h_at_t_or_t_minus_1` (ends around line 538) and **insert the hook call** between that assignment and the `dg_weights = self._solve_weight_gradients(...)` call (around line 548). Also grab the current `args` tuple — `_update_fn_bwd` receives `fwd_res`, not `args`, so we need to thread the args through `fwd_res`.

Modify `_update_fn_fwd` (around line 411) so its `fwd_res` tuple carries `args`:

```python
        fwd_res = (
            residuals,
            (
                old_etrace_vals
                if self.graph_executor.is_multi_step_vjp else
                new_etrace_vals
            ),
            weight_vals,
            running_index,
            args,          # NEW — needed by _compute_learning_signal
        )
```

In `_update_fn_bwd` (around line 443), unpack the new field:

```python
        (
            residuals,
            etrace_vals_at_t_or_t_minus_1,
            weight_vals,
            running_index,
            args,          # NEW
        ) = fwd_res
```

Then, between the `dl2h_at_t_or_t_minus_1 = ...` assembly and the `dg_weights = self._solve_weight_gradients(...)` call, insert:

```python
        # Hook: subclasses may replace the reverse-AD learning signal with
        # an alternative (e.g. target projection in OSTTP).
        dl2h_at_t_or_t_minus_1 = self._compute_learning_signal(
            dl2h_at_t_or_t_minus_1, args
        )
```

- [ ] **Step 4: Run the override test and the whole `base_test.py`**

Run:
```bash
python -m pytest braintrace/_etrace_vjp/base_test.py -v
```
Expected: all pass (including the new override test).

- [ ] **Step 5: Run existing D_RTRL + pp_prop regression**

Run:
```bash
python -m pytest braintrace/_etrace_vjp/d_rtrl_test.py braintrace/_etrace_vjp/pp_prop_test.py -v
```
Expected: all pass unchanged.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_vjp/base.py braintrace/_etrace_vjp/base_test.py
git commit -m "feat: wire _compute_learning_signal into _update_fn_bwd"
```

---

## Phase 2 — Shared helpers (`_common.py`)

Goal: helpers used by multiple SNN algorithm classes. Each is independently tested.

### Task 2.1: Scaffold the `_snn_algorithms/` package

**Files:**
- Create: `braintrace/_snn_algorithms/__init__.py`
- Create: `braintrace/_snn_algorithms/_common.py`

- [ ] **Step 1: Create the empty package files**

`braintrace/_snn_algorithms/__init__.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...  (standard Apache 2.0 header — copy from any sibling package)

"""Online learning algorithms for spiking neural networks.

Paper-faithful ``ETraceVjpAlgorithm`` subclasses: EProp, OSTL, OTPE, OTTT, OSTTP.
See ``docs/superpowers/specs/2026-04-19-snn-online-etrace-algorithms-design.md``.
"""

__all__: list = []  # populated as each class lands
```

`braintrace/_snn_algorithms/_common.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...  (standard Apache 2.0 header — copy from any sibling file)

"""Shared helpers for the SNN online-learning algorithms."""
```

Use the full Apache 2.0 header from `braintrace/_etrace_vjp/base.py` lines 1-14.

- [ ] **Step 2: Verify package imports clean**

Run:
```bash
python -c "import braintrace._snn_algorithms; import braintrace._snn_algorithms._common"
```
Expected: silent success.

- [ ] **Step 3: Commit**

```bash
git add braintrace/_snn_algorithms/__init__.py braintrace/_snn_algorithms/_common.py
git commit -m "feat: scaffold braintrace._snn_algorithms package"
```

---

### Task 2.2: `PresynapticTrace` helper

**Files:**
- Modify: `braintrace/_snn_algorithms/_common.py`
- Create: `braintrace/_snn_algorithms/_common_test.py`

- [ ] **Step 1: Write the failing test**

`braintrace/_snn_algorithms/_common_test.py`:

```python
import unittest

import brainstate
import jax.numpy as jnp

from braintrace._snn_algorithms._common import PresynapticTrace


class TestPresynapticTrace(unittest.TestCase):
    def test_exponential_accumulation(self):
        trace = PresynapticTrace(jnp.zeros((2, 3)), leak=0.9)
        trace.update(jnp.ones((2, 3)))
        trace.update(jnp.ones((2, 3)))
        # After 2 ones: 0.9*(0.9*0 + 1) + 1 == 1.9
        assert jnp.allclose(trace.value, jnp.full((2, 3), 1.9))

    def test_reset_to_zero(self):
        trace = PresynapticTrace(jnp.ones((4,)), leak=0.5)
        trace.reset_state()
        assert jnp.allclose(trace.value, jnp.zeros((4,)))
```

- [ ] **Step 2: Run it, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/_common_test.py::TestPresynapticTrace -v
```
Expected: FAIL with `ImportError: cannot import name 'PresynapticTrace'`.

- [ ] **Step 3: Implement `PresynapticTrace`**

Append to `braintrace/_snn_algorithms/_common.py`:

```python
from typing import Any

import brainstate
import jax
import jax.numpy as jnp


class PresynapticTrace(brainstate.ShortTermState):
    """Leaky accumulator â ← λ·â + x_t used by OTTT and OTPE-Approx.

    Parameters
    ----------
    init_value : jax.Array
        Initial value; also dictates shape and dtype.
    leak : float
        Decay factor λ in (0, 1). Pulled from the neuron's membrane leak in SNN usage.
    """

    __module__ = 'braintrace'

    def __init__(self, init_value, leak: float):
        super().__init__(init_value)
        if not (0.0 < leak < 1.0):
            raise ValueError(f'leak must be in (0, 1); got {leak}')
        self.leak = float(leak)
        self._init_shape = jnp.shape(init_value)
        self._init_dtype = init_value.dtype

    def update(self, x):
        """Apply one accumulation step: â ← λ·â + x."""
        self.value = self.leak * self.value + x
        return self.value

    def reset_state(self, batch_size: int = None, **kwargs):
        shape = self._init_shape if batch_size is None else (batch_size, *self._init_shape[1:])
        self.value = jnp.zeros(shape, dtype=self._init_dtype)
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/_common_test.py::TestPresynapticTrace -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/_common.py braintrace/_snn_algorithms/_common_test.py
git commit -m "feat: PresynapticTrace helper for OTTT/OTPE-Approx"
```

---

### Task 2.3: `KappaFilter` helper

**Files:**
- Modify: `braintrace/_snn_algorithms/_common.py`
- Modify: `braintrace/_snn_algorithms/_common_test.py`

- [ ] **Step 1: Failing test**

Append to `_common_test.py`:

```python
from braintrace._snn_algorithms._common import KappaFilter


class TestKappaFilter(unittest.TestCase):
    def test_low_pass(self):
        flt = KappaFilter(jnp.zeros((3,)), kappa=0.8)
        y1 = flt.update(jnp.ones((3,)))
        # (1 - 0.8)*1 + 0.8*0 == 0.2
        assert jnp.allclose(y1, jnp.full((3,), 0.2))
        y2 = flt.update(jnp.ones((3,)))
        # (1 - 0.8)*1 + 0.8*0.2 == 0.36
        assert jnp.allclose(y2, jnp.full((3,), 0.36))

    def test_kappa_zero_disables(self):
        flt = KappaFilter(jnp.zeros((3,)), kappa=0.0)
        y = flt.update(jnp.full((3,), 5.0))
        assert jnp.allclose(y, jnp.full((3,), 5.0))  # pass-through
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/_common_test.py::TestKappaFilter -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

Append to `_common.py`:

```python
class KappaFilter(brainstate.ShortTermState):
    """Low-pass output-side filter x_filt ← (1-κ)·x + κ·x_filt used by EProp.

    Parameters
    ----------
    init_value : jax.Array
    kappa : float
        Decay factor in [0, 1). 0 disables filtering.
    """

    __module__ = 'braintrace'

    def __init__(self, init_value, kappa: float):
        super().__init__(init_value)
        if not (0.0 <= kappa < 1.0):
            raise ValueError(f'kappa must be in [0, 1); got {kappa}')
        self.kappa = float(kappa)
        self._init_shape = jnp.shape(init_value)
        self._init_dtype = init_value.dtype

    def update(self, x):
        new = (1.0 - self.kappa) * x + self.kappa * self.value
        self.value = new
        return new

    def reset_state(self, batch_size: int = None, **kwargs):
        shape = self._init_shape if batch_size is None else (batch_size, *self._init_shape[1:])
        self.value = jnp.zeros(shape, dtype=self._init_dtype)
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/_common_test.py::TestKappaFilter -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/_common.py braintrace/_snn_algorithms/_common_test.py
git commit -m "feat: KappaFilter helper for EProp output-side low-pass"
```

---

### Task 2.4: `FixedRandomFeedback` helper

**Files:**
- Modify: `braintrace/_snn_algorithms/_common.py`
- Modify: `braintrace/_snn_algorithms/_common_test.py`

- [ ] **Step 1: Failing test**

Append to `_common_test.py`:

```python
from braintrace._snn_algorithms._common import FixedRandomFeedback


class TestFixedRandomFeedback(unittest.TestCase):
    def test_shape_and_frozen(self):
        import jax
        key = jax.random.PRNGKey(0)
        fb = FixedRandomFeedback(n_target=10, n_layer=200, key=key, init_scale=0.1)
        assert fb.B.shape == (10, 200)
        # No gradient flows through B
        grad_fn = jax.grad(lambda y_target: (fb.project(y_target) ** 2).sum())
        y = jnp.ones((5, 10))
        g = grad_fn(y)
        assert g.shape == y.shape

    def test_project_shapes(self):
        import jax
        fb = FixedRandomFeedback(n_target=4, n_layer=7, key=jax.random.PRNGKey(1))
        y_target = jnp.ones((3, 4))  # batched
        proj = fb.project(y_target)
        assert proj.shape == (3, 7)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/_common_test.py::TestFixedRandomFeedback -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `_common.py`:

```python
class FixedRandomFeedback:
    """Frozen random feedback matrix B ∈ ℝ^{n_target × n_layer} with stop_gradient guard.

    Used by OSTTP (per-HiddenGroup target projection) and EProp-random-feedback.
    """

    __module__ = 'braintrace'

    def __init__(self, n_target: int, n_layer: int, key, init_scale: float = 0.1):
        self.B = jax.lax.stop_gradient(
            init_scale * jax.random.normal(key, (n_target, n_layer))
        )
        self.n_target = int(n_target)
        self.n_layer = int(n_layer)

    def project(self, y_target):
        """Return y_target @ B with B frozen. Handles batched and unbatched y_target."""
        return y_target @ self.B
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/_common_test.py::TestFixedRandomFeedback -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/_common.py braintrace/_snn_algorithms/_common_test.py
git commit -m "feat: FixedRandomFeedback helper for OSTTP/EProp-random"
```

---

### Task 2.5: `extract_y_target` + `_resolve_leak` helpers

**Files:**
- Modify: `braintrace/_snn_algorithms/_common.py`
- Modify: `braintrace/_snn_algorithms/_common_test.py`

- [ ] **Step 1: Failing tests**

Append to `_common_test.py`:

```python
from braintrace._snn_algorithms._common import extract_y_target, _resolve_leak


class TestExtractYTarget(unittest.TestCase):
    def test_absent_returns_none(self):
        assert extract_y_target(()) is None

    def test_present_returns_value(self):
        y = jnp.ones((5,))
        assert extract_y_target((jnp.zeros(3), y), index=1) is y


class TestResolveLeak(unittest.TestCase):
    def test_explicit_float_wins(self):
        assert _resolve_leak(model=None, explicit=0.7) == 0.7

    def test_discover_from_model(self):
        class FakeState:
            leak = 0.4

        class FakeModel:
            def states(self):
                return [FakeState()]

        assert _resolve_leak(model=FakeModel(), explicit=None) == 0.4

    def test_missing_raises(self):
        class EmptyModel:
            def states(self):
                return []

        with self.assertRaises(ValueError):
            _resolve_leak(model=EmptyModel(), explicit=None)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/_common_test.py::TestExtractYTarget braintrace/_snn_algorithms/_common_test.py::TestResolveLeak -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `_common.py`:

```python
from typing import Optional


def extract_y_target(args: tuple, *, index: int = -1) -> Optional[jax.Array]:
    """Fetch the target tensor from a positional-args tuple.

    Returns ``None`` if ``args`` is empty. ``index`` defaults to the last position
    (OSTTP's convention: ``algo.update(x, y_target)``).
    """
    if not args:
        return None
    return args[index]


def _resolve_leak(model, explicit: Optional[float]) -> float:
    """Pick the leak factor λ for OTTT/OTPE.

    Priority:
    1. ``explicit`` argument (constructor-supplied) wins if not None.
    2. Walk ``model.states()``; first state whose object has a ``leak`` attribute wins.
    3. Raise ``ValueError`` if neither resolves.
    """
    if explicit is not None:
        return float(explicit)
    if model is not None:
        for st in model.states():
            if hasattr(st, 'leak'):
                return float(st.leak)
    raise ValueError(
        'Could not resolve the membrane leak factor. Provide `leak=<float>` at '
        'construction, or ensure the model has a ShortTermState with a `leak` attribute.'
    )
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/_common_test.py -v
```
Expected: all tests in `_common_test.py` pass.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/_common.py braintrace/_snn_algorithms/_common_test.py
git commit -m "feat: extract_y_target and _resolve_leak helpers"
```

---

## Phase 3 — OSTL (thin regime-flag delegator)

Goal: `OSTL(model, regime='with-H'|'without-H')`. With-H delegates to `ParamDimVjpAlgorithm` (D_RTRL), without-H delegates to `IODimVjpAlgorithm` (pp_prop) with decay=1-ε (i.e. effectively zero-decay feedforward).

### Task 3.1: Scaffold `OSTL` class with regime dispatch

**Files:**
- Create: `braintrace/_snn_algorithms/ostl.py`
- Create: `braintrace/_snn_algorithms/ostl_test.py`

- [ ] **Step 1: Failing shape/construction test**

`braintrace/_snn_algorithms/ostl_test.py`:

```python
import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._snn_algorithms.ostl import OSTL


def _tiny_rec_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w_rec = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (5, 3))
            )
            self.h = brainstate.ShortTermState(jnp.zeros((4, 3)))

        def update(self, x):
            pre = jnp.concatenate([x, jax.lax.stop_gradient(self.h.value)], axis=-1)
            self.h.value = jax.nn.tanh(braintrace.matmul(pre, self.w_rec.value))
            return self.h.value

    return Net()


class TestOSTLConstruction(unittest.TestCase):
    def test_default_regime_is_with_h(self):
        algo = OSTL(_tiny_rec_net())
        assert algo.regime == 'with-H'

    def test_invalid_regime_raises(self):
        with self.assertRaises(ValueError):
            OSTL(_tiny_rec_net(), regime='bogus')

    def test_with_h_compiles_and_updates(self):
        net = _tiny_rec_net()
        algo = OSTL(net, regime='with-H')
        x = jnp.ones((4, 2))
        algo.compile_graph(x)
        algo.init_etrace_state()
        out = algo.update(x)
        assert out.shape == (4, 3)

    def test_without_h_compiles_and_updates(self):
        net = _tiny_rec_net()
        algo = OSTL(net, regime='without-H')
        x = jnp.ones((4, 2))
        algo.compile_graph(x)
        algo.init_etrace_state()
        out = algo.update(x)
        assert out.shape == (4, 3)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/ostl_test.py::TestOSTLConstruction -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

`braintrace/_snn_algorithms/ostl.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# ... (Apache 2.0 header) ...

"""OSTL — Online Spatio-Temporal Learning (Bohnstingl et al. 2023).

Regime 'with-H'   — RTRL-exact single-layer factorization (delegates to D_RTRL).
Regime 'without-H' — feedforward / no recurrent Jacobian (delegates to pp_prop with decay≈0).
"""

from typing import Optional

import brainstate

from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm
from braintrace._etrace_vjp.pp_prop import IODimVjpAlgorithm

__all__ = ['OSTL']


def OSTL(
    model: brainstate.nn.Module,
    regime: str = 'with-H',
    name: Optional[str] = None,
    **kwargs,
):
    """Factory returning the appropriate VJP algorithm for the selected regime.

    Using a factory (not a subclass with branching) lets each regime inherit
    everything — compile_graph, update, reset_state, get_etrace_of — from the
    existing tested algorithm classes without duplication.

    Parameters
    ----------
    model : brainstate.nn.Module
    regime : {'with-H', 'without-H'}
        'with-H' uses D_RTRL-shape traces (per-parameter, O(P·H)). Exact for
        single-recurrent-layer networks. 'without-H' drops the temporal term,
        equivalent to pp_prop with negligible decay (feedforward SNN).
    name : optional name.
    **kwargs : forwarded to the base algorithm constructor.
    """
    if regime not in ('with-H', 'without-H'):
        raise ValueError(f"regime must be 'with-H' or 'without-H'; got {regime!r}")

    if regime == 'with-H':
        algo = ParamDimVjpAlgorithm(model, name=name, **kwargs)
    else:
        decay = kwargs.pop('decay_or_rank', 1e-6)
        algo = IODimVjpAlgorithm(model, decay_or_rank=decay, name=name, **kwargs)

    algo.regime = regime
    return algo
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/ostl_test.py::TestOSTLConstruction -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/ostl.py braintrace/_snn_algorithms/ostl_test.py
git commit -m "feat: OSTL regime-flag factory over D_RTRL / pp_prop"
```

---

### Task 3.2: OSTL — batched/unbatched equivalence test

**Files:**
- Modify: `braintrace/_snn_algorithms/ostl_test.py`

- [ ] **Step 1: Write the test**

Append to `ostl_test.py`:

```python
class TestOSTLBatchingEquivalence(unittest.TestCase):
    def test_batched_vs_unbatched_produce_same_gradients(self):
        """Running each sample separately then stacking must match a single batched call."""
        import jax

        def run(batch_size):
            class Net(brainstate.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = brainstate.ParamState(
                        0.1 * jax.random.normal(jax.random.PRNGKey(42), (2, 3))
                    )
                    self.h = brainstate.ShortTermState(jnp.zeros((batch_size, 3)) if batch_size else jnp.zeros((3,)))

                def update(self, x):
                    self.h.value = jax.nn.tanh(braintrace.matmul(x, self.w.value) + self.h.value)
                    return self.h.value

            net = Net()
            algo = OSTL(net, regime='with-H')
            shape = (batch_size, 2) if batch_size else (2,)
            x = jnp.ones(shape)
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x):
                return (algo.update(x) ** 2).sum()

            grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
            return next(iter(grads.values()))['weight']

        g_unbatched = run(0)
        g_batched = run(1)
        # Batched grad has summed across batch=1, so they match scalar-for-scalar.
        assert jnp.allclose(g_unbatched, g_batched[0] if g_batched.ndim > g_unbatched.ndim else g_batched, atol=1e-5)
```

- [ ] **Step 2: Run, confirm PASS (both regimes already work via delegation)**

```bash
python -m pytest braintrace/_snn_algorithms/ostl_test.py::TestOSTLBatchingEquivalence -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add braintrace/_snn_algorithms/ostl_test.py
git commit -m "test: OSTL batched/unbatched numerical equivalence"
```

---

### Task 3.3: OSTL — reset_state clears traces + knob mode test

**Files:**
- Modify: `braintrace/_snn_algorithms/ostl_test.py`

- [ ] **Step 1: Write tests**

Append to `ostl_test.py`:

```python
class TestOSTLResetAndKnob(unittest.TestCase):
    def test_reset_zeros_traces(self):
        net = _tiny_rec_net()
        algo = OSTL(net, regime='with-H')
        x = jnp.ones((4, 2))
        algo.compile_graph(x)
        algo.init_etrace_state()
        # Drive a few steps so traces become non-zero.
        for _ in range(3):
            algo.update(x)
        # Reset.
        algo.reset_state(batch_size=4)
        for k, st in algo.etrace_bwg.items():
            assert jnp.allclose(st.value, jnp.zeros_like(st.value))
        assert algo.running_index.value == 0

    def test_without_h_regime_differs_from_with_h(self):
        """without-H ignores hid2hid_jac; after several recurrent steps the two regimes
        must disagree on any non-trivial loss."""
        def final_grad(regime):
            net = _tiny_rec_net()
            algo = OSTL(net, regime=regime)
            x = jnp.ones((4, 2))
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x):
                out = jnp.zeros_like(net.h.value)
                for _ in range(4):
                    out = algo.update(x)
                return (out ** 2).sum()

            grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
            return next(iter(grads.values()))['weight']

        g_with = final_grad('with-H')
        g_without = final_grad('without-H')
        assert not jnp.allclose(g_with, g_without, atol=1e-4)
```

- [ ] **Step 2: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/ostl_test.py::TestOSTLResetAndKnob -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add braintrace/_snn_algorithms/ostl_test.py
git commit -m "test: OSTL reset + regime knob differentiation"
```

---

## Phase 4 — EProp (D_RTRL + κ-filter + feedback knob)

Goal: `EProp(model, feedback='symmetric'|'random', kappa_filter_decay=0.0, random_feedback_key=None)` inherits `ParamDimVjpAlgorithm`, adds κ-filtered `ē` sidecar and optional random-feedback readout.

### Task 4.1: EProp — base subclass with κ-filter sidecar

**Files:**
- Create: `braintrace/_snn_algorithms/e_prop.py`
- Create: `braintrace/_snn_algorithms/e_prop_test.py`

- [ ] **Step 1: Failing test**

`braintrace/_snn_algorithms/e_prop_test.py`:

```python
import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._snn_algorithms.e_prop import EProp


def _lsnn_like():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (4, 3))
            )
            self.h = brainstate.ShortTermState(jnp.zeros((2, 3)))

        def update(self, x):
            pre = jnp.concatenate([x, jax.lax.stop_gradient(self.h.value)], axis=-1)
            self.h.value = jax.nn.tanh(braintrace.matmul(pre, self.w.value))
            return self.h.value

    return Net()


class TestEPropConstruction(unittest.TestCase):
    def test_default_feedback_and_kappa(self):
        algo = EProp(_lsnn_like())
        assert algo.feedback == 'symmetric'
        assert algo.kappa_filter_decay == 0.0

    def test_invalid_feedback_raises(self):
        with self.assertRaises(ValueError):
            EProp(_lsnn_like(), feedback='weird')

    def test_kappa_filter_allocated_when_nonzero(self):
        algo = EProp(_lsnn_like(), kappa_filter_decay=0.9)
        x = jnp.ones((2, 1))
        algo.compile_graph(x)
        algo.init_etrace_state()
        # Per HiddenParamOpRelation, there should be one KappaFilter.
        assert len(algo._kappa_filters) == len(algo.graph.hidden_param_op_relations)

    def test_kappa_filter_skipped_when_zero(self):
        algo = EProp(_lsnn_like(), kappa_filter_decay=0.0)
        x = jnp.ones((2, 1))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._kappa_filters) == 0
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/e_prop_test.py::TestEPropConstruction -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement**

`braintrace/_snn_algorithms/e_prop.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# ... (Apache 2.0 header) ...

"""E-Prop — Eligibility Propagation (Bellec et al. 2020).

D_RTRL's per-parameter trace plus:
- An optional κ-filter on each HiddenGroup's eligibility trace (ē = F_κ(e))
  which matches the paper's readout-side low-pass.
- An optional random-feedback variant (feedback='random') that replaces the
  readout's symmetric gradient with a fixed random projection.
"""

from typing import Dict, Optional

import brainstate
import jax
import jax.numpy as jnp

from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm
from ._common import KappaFilter, FixedRandomFeedback

__all__ = ['EProp']


class EProp(ParamDimVjpAlgorithm):
    """Eligibility Propagation.

    Parameters
    ----------
    model : brainstate.nn.Module
    feedback : {'symmetric', 'random'}
        'symmetric' uses reverse-AD's ∂L/∂h (standard backprop through readout).
        'random' replaces the readout gradient with a frozen random projection
        of the output error (see `random_feedback_key`).
    kappa_filter_decay : float in [0, 1)
        If > 0, apply an output-side low-pass to each HiddenGroup's trace each step.
        0 disables (paper-default for hard tasks).
    random_feedback_key : jax.random.PRNGKey, optional
        Seed for the random-feedback matrix when feedback='random'.
    name, vjp_method, fast_solve, normalize_matrix_spectrum : forwarded to D_RTRL.
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        feedback: str = 'symmetric',
        kappa_filter_decay: float = 0.0,
        random_feedback_key: Optional[jax.Array] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        normalize_matrix_spectrum: bool = False,
        **kwargs,
    ):
        if feedback not in ('symmetric', 'random'):
            raise ValueError(f"feedback must be 'symmetric' or 'random'; got {feedback!r}")
        if feedback == 'random' and random_feedback_key is None:
            raise ValueError("feedback='random' requires random_feedback_key=<PRNGKey>")
        super().__init__(
            model,
            name=name,
            vjp_method=vjp_method,
            fast_solve=fast_solve,
            normalize_matrix_spectrum=normalize_matrix_spectrum,
            **kwargs,
        )
        self.feedback = feedback
        self.kappa_filter_decay = float(kappa_filter_decay)
        self._random_feedback_key = random_feedback_key
        self._kappa_filters: Dict[int, KappaFilter] = {}
        self._random_feedback: Optional[FixedRandomFeedback] = None

    def init_etrace_state(self, *args, **kwargs):
        super().init_etrace_state(*args, **kwargs)
        self._kappa_filters = {}
        if self.kappa_filter_decay > 0.0:
            for rel in self.graph.hidden_param_op_relations:
                for group in rel.hidden_groups:
                    gid = group.index
                    if gid in self._kappa_filters:
                        continue
                    zeros = jnp.zeros(group.varshape, dtype=jnp.float32)
                    self._kappa_filters[gid] = KappaFilter(zeros, self.kappa_filter_decay)

    def reset_state(self, batch_size: int = None, **kwargs):
        super().reset_state(batch_size=batch_size, **kwargs)
        for flt in self._kappa_filters.values():
            flt.reset_state(batch_size=batch_size)
```

Note: the κ-filter is allocated per-HiddenGroup but not yet applied during solve — that lands in Task 4.2.

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/e_prop_test.py::TestEPropConstruction -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/e_prop.py braintrace/_snn_algorithms/e_prop_test.py
git commit -m "feat: EProp subclass with κ-filter allocation"
```

---

### Task 4.2: EProp — apply κ-filter to learning signal

**Files:**
- Modify: `braintrace/_snn_algorithms/e_prop.py`
- Modify: `braintrace/_snn_algorithms/e_prop_test.py`

- [ ] **Step 1: Failing test**

Append to `e_prop_test.py`:

```python
class TestEPropKappaApplied(unittest.TestCase):
    def test_kappa_zero_matches_d_rtrl(self):
        """κ=0 must reproduce D_RTRL gradients bit-for-bit on the same model."""
        from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm

        def compute(algo_cls, **extra):
            net = _lsnn_like()
            algo = algo_cls(net, **extra)
            x = jnp.ones((2, 1))
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x):
                out = algo.update(x)
                return (out ** 2).sum()

            grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
            return next(iter(grads.values()))['weight']

        g_drtrl = compute(ParamDimVjpAlgorithm)
        g_eprop = compute(EProp, feedback='symmetric', kappa_filter_decay=0.0)
        assert jnp.allclose(g_drtrl, g_eprop, atol=1e-6)

    def test_kappa_nonzero_differs_from_zero(self):
        def compute(kappa):
            net = _lsnn_like()
            algo = EProp(net, kappa_filter_decay=kappa)
            x = jnp.ones((2, 1))
            algo.compile_graph(x)
            algo.init_etrace_state()
            # Two steps so the filter has state to carry.
            algo.update(x)

            def loss(x):
                out = algo.update(x)
                return (out ** 2).sum()

            grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
            return next(iter(grads.values()))['weight']

        g0 = compute(0.0)
        g9 = compute(0.9)
        assert not jnp.allclose(g0, g9, atol=1e-4)
```

- [ ] **Step 2: Run, confirm FAIL (κ nonzero test fails — filter allocated but not applied)**

```bash
python -m pytest braintrace/_snn_algorithms/e_prop_test.py::TestEPropKappaApplied -v
```
Expected: `test_kappa_nonzero_differs_from_zero` FAIL; `test_kappa_zero_matches_d_rtrl` PASS.

- [ ] **Step 3: Implement κ-filter application via `_compute_learning_signal`**

Add to `EProp` class in `e_prop.py`:

```python
    def _compute_learning_signal(self, dl_autodiff, args):
        """Apply κ-filter to each HiddenGroup's learning signal (EProp's ē = F_κ(L))."""
        if not self._kappa_filters:
            return dl_autodiff
        out = []
        for gid, dl in enumerate(dl_autodiff):
            flt = self._kappa_filters.get(gid)
            out.append(flt.update(dl) if flt is not None else dl)
        return out
```

Note: mutating filter state inside the bwd pass is safe because `brainstate.ShortTermState.value` writes go through the usual state-tree. The filter is *outside* the custom_vjp boundary so no custom_vjp rule pollution occurs.

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/e_prop_test.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/e_prop.py braintrace/_snn_algorithms/e_prop_test.py
git commit -m "feat: EProp κ-filter application via _compute_learning_signal"
```

---

### Task 4.3: EProp — random-feedback mode

**Files:**
- Modify: `braintrace/_snn_algorithms/e_prop.py`
- Modify: `braintrace/_snn_algorithms/e_prop_test.py`

- [ ] **Step 1: Failing test**

Append to `e_prop_test.py`:

```python
class TestEPropRandomFeedback(unittest.TestCase):
    def test_random_feedback_differs_from_symmetric(self):
        def compute(feedback, **extra):
            net = _lsnn_like()
            algo = EProp(net, feedback=feedback, **extra)
            x = jnp.ones((2, 1))
            algo.compile_graph(x)
            algo.init_etrace_state()

            def loss(x):
                out = algo.update(x)
                return (out ** 2).sum()

            grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
            return next(iter(grads.values()))['weight']

        g_sym = compute('symmetric')
        g_rnd = compute('random', random_feedback_key=jax.random.PRNGKey(123))
        # Different feedback paths must produce different gradients.
        assert not jnp.allclose(g_sym, g_rnd, atol=1e-4)
```

- [ ] **Step 2: Run, confirm FAIL (random-feedback path not yet wired)**

```bash
python -m pytest braintrace/_snn_algorithms/e_prop_test.py::TestEPropRandomFeedback -v
```
Expected: FAIL — feedback='random' currently produces identical output to 'symmetric'.

- [ ] **Step 3: Implement random-feedback**

In `e_prop.py`, extend `init_etrace_state` to allocate the feedback matrix:

```python
    def init_etrace_state(self, *args, **kwargs):
        super().init_etrace_state(*args, **kwargs)
        self._kappa_filters = {}
        if self.kappa_filter_decay > 0.0:
            for rel in self.graph.hidden_param_op_relations:
                for group in rel.hidden_groups:
                    gid = group.index
                    if gid in self._kappa_filters:
                        continue
                    zeros = jnp.zeros(group.varshape, dtype=jnp.float32)
                    self._kappa_filters[gid] = KappaFilter(zeros, self.kappa_filter_decay)
        if self.feedback == 'random':
            # Allocate one FixedRandomFeedback per HiddenGroup, projecting from
            # group's hidden-dim back to an n_target-dim error space. For EProp,
            # the "error" comes from the readout's shape, which we read from
            # the last HiddenGroup at update time — so we store B keyed by
            # group index and let the bwd reshape accordingly.
            self._random_feedback = {}
            key = self._random_feedback_key
            for rel in self.graph.hidden_param_op_relations:
                for group in rel.hidden_groups:
                    gid = group.index
                    if gid in self._random_feedback:
                        continue
                    key, sub = jax.random.split(key)
                    n_layer = int(jnp.prod(jnp.array(group.varshape[-1:])))
                    # For random EProp, n_target equals n_layer — it's a square
                    # projection applied to the reverse-AD signal. Users wanting
                    # a true readout-random variant must subclass further.
                    self._random_feedback[gid] = FixedRandomFeedback(
                        n_target=n_layer, n_layer=n_layer, key=sub, init_scale=0.1
                    )
```

Modify `_compute_learning_signal` to compose random-feedback with κ:

```python
    def _compute_learning_signal(self, dl_autodiff, args):
        signals = list(dl_autodiff)
        if self.feedback == 'random' and self._random_feedback:
            signals = [
                self._random_feedback[gid].project(s) if gid in self._random_feedback else s
                for gid, s in enumerate(signals)
            ]
        if self._kappa_filters:
            signals = [
                self._kappa_filters[gid].update(s) if gid in self._kappa_filters else s
                for gid, s in enumerate(signals)
            ]
        return signals
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/e_prop_test.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/e_prop.py braintrace/_snn_algorithms/e_prop_test.py
git commit -m "feat: EProp random-feedback variant"
```

---

## Phase 5 — OTTT (presynaptic λ-trace)

Goal: `OTTT(model, mode='A'|'O', leak=None)` ignores both `hid2weight_jac` and `hid2hid_jac`, instead maintaining per-layer presynaptic trace `â ← λ·â + x_t`, and computing `ΔW = outer(â, L · σ'(u))`.

### Task 5.1: OTTT — construction + state allocation

**Files:**
- Create: `braintrace/_snn_algorithms/ottt.py`
- Create: `braintrace/_snn_algorithms/ottt_test.py`

- [ ] **Step 1: Failing test**

`braintrace/_snn_algorithms/ottt_test.py`:

```python
import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._snn_algorithms.ottt import OTTT


class FakeLIF(brainstate.ShortTermState):
    """Stand-in LIF-like state carrying a `leak` attribute for _resolve_leak."""
    def __init__(self, init_value, leak):
        super().__init__(init_value)
        self.leak = leak


def _ottt_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = FakeLIF(jnp.zeros((2, 3)), leak=0.9)

        def update(self, x):
            self.v.value = 0.9 * self.v.value + braintrace.matmul(x, self.w.value)
            return self.v.value

    return Net()


class TestOTTTConstruction(unittest.TestCase):
    def test_default_mode_is_A(self):
        algo = OTTT(_ottt_net(), leak=0.9)
        assert algo.mode == 'A'

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            OTTT(_ottt_net(), mode='bogus', leak=0.9)

    def test_leak_discovered_from_model(self):
        """_resolve_leak picks up FakeLIF.leak when not explicitly provided."""
        algo = OTTT(_ottt_net())
        assert algo.leak == 0.9

    def test_missing_leak_raises(self):
        class BareNet(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.eye(3))
                self.h = brainstate.ShortTermState(jnp.zeros((2, 3)))
            def update(self, x):
                self.h.value = braintrace.matmul(x, self.w.value)
                return self.h.value

        with self.assertRaises(ValueError):
            OTTT(BareNet())

    def test_compile_allocates_presynaptic_traces(self):
        algo = OTTT(_ottt_net(), leak=0.9)
        x = jnp.ones((2, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        # One PresynapticTrace per HiddenParamOpRelation.
        assert len(algo._pre_traces) == len(algo.graph.hidden_param_op_relations)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/ottt_test.py::TestOTTTConstruction -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement — construction path only**

`braintrace/_snn_algorithms/ottt.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# ... (Apache 2.0 header) ...

"""OTTT — Online Training Through Time (Xiao et al. 2022).

Drops both the hidden-to-hidden and hidden-to-weight Jacobians; maintains only a
presynaptic eligibility trace â ← λ·â + x_t and computes weight gradients as
ΔW = outer(â, L · σ'(u)) per step.
"""

from typing import Dict, Optional

import brainstate
import jax
import jax.numpy as jnp

from braintrace._etrace_vjp.base import ETraceVjpAlgorithm
from ._common import PresynapticTrace, _resolve_leak

__all__ = ['OTTT']


class OTTT(ETraceVjpAlgorithm):
    """Online Training Through Time.

    Parameters
    ----------
    model : brainstate.nn.Module
    mode : {'A', 'O'}
        'A' (default) accumulates â over time. 'O' uses the instantaneous
        presynaptic spike only (â := x_t).
    leak : float, optional
        Presynaptic leak λ. If None, discovered from model via `_resolve_leak`.
    name, vjp_method : forwarded to base.
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        mode: str = 'A',
        leak: Optional[float] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        **kwargs,
    ):
        if mode not in ('A', 'O'):
            raise ValueError(f"mode must be 'A' or 'O'; got {mode!r}")
        super().__init__(model, name=name, vjp_method=vjp_method)
        self.mode = mode
        self.leak = _resolve_leak(model, leak)
        self._pre_traces: Dict[int, PresynapticTrace] = {}

    def init_etrace_state(self, *args, **kwargs):
        self._pre_traces = {}
        for rel in self.graph.hidden_param_op_relations:
            rid = id(rel.x_var)
            if rid in self._pre_traces:
                continue
            shape = rel.x_var.aval.shape
            dtype = rel.x_var.aval.dtype
            self._pre_traces[rid] = PresynapticTrace(
                jnp.zeros(shape, dtype=dtype), leak=self.leak
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        self.running_index.value = 0
        for t in self._pre_traces.values():
            t.reset_state(batch_size=batch_size)

    # Stubs for the four protocol methods — filled in in Task 5.2.
    def _get_etrace_data(self):
        return {rid: t.value for rid, t in self._pre_traces.items()}

    def _assign_etrace_data(self, vals):
        for rid, v in vals.items():
            self._pre_traces[rid].value = v

    def _update_etrace_data(
        self, running_index, hist_vals,
        hid2weight_jac, hid2hid_jac, weight_vals, input_is_multi_step,
    ):
        raise NotImplementedError('filled in Task 5.2')

    def _solve_weight_gradients(
        self, running_index, etrace_at_t, dl_to_hidden_groups,
        weight_vals, dl_to_nonetws_at_t, dl_to_etws_at_t,
    ):
        raise NotImplementedError('filled in Task 5.2')
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/ottt_test.py::TestOTTTConstruction -v
```
Expected: PASS (only construction tests, no trace updates yet).

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/ottt.py braintrace/_snn_algorithms/ottt_test.py
git commit -m "feat: OTTT construction + pre-trace allocation"
```

---

### Task 5.2: OTTT — trace update + weight solve

**Files:**
- Modify: `braintrace/_snn_algorithms/ottt.py`
- Modify: `braintrace/_snn_algorithms/ottt_test.py`

- [ ] **Step 1: Failing test**

Append to `ottt_test.py`:

```python
class TestOTTTEnd2End(unittest.TestCase):
    def test_update_runs_and_produces_gradients(self):
        net = _ottt_net()
        algo = OTTT(net, leak=0.9)
        x = jnp.ones((2, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x):
            return (algo.update(x) ** 2).sum()

        grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
        g = next(iter(grads.values()))['weight']
        assert g.shape == (3, 3)
        assert jnp.any(g != 0.0)

    def test_mode_O_differs_from_mode_A(self):
        def compute(mode):
            net = _ottt_net()
            algo = OTTT(net, mode=mode, leak=0.9)
            x = jnp.ones((2, 3))
            algo.compile_graph(x)
            algo.init_etrace_state()
            # Two steps so 'A' accumulates while 'O' does not.
            algo.update(x)

            def loss(x):
                return (algo.update(x) ** 2).sum()

            grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
            return next(iter(grads.values()))['weight']

        g_A = compute('A')
        g_O = compute('O')
        assert not jnp.allclose(g_A, g_O, atol=1e-4)
```

- [ ] **Step 2: Run, confirm FAIL (stubs raise NotImplementedError)**

```bash
python -m pytest braintrace/_snn_algorithms/ottt_test.py::TestOTTTEnd2End -v
```
Expected: FAIL.

- [ ] **Step 3: Implement trace update + solve**

Replace the stub methods in `ottt.py`:

```python
    def _update_etrace_data(
        self, running_index, hist_vals,
        hid2weight_jac, hid2hid_jac, weight_vals, input_is_multi_step,
    ):
        """â ← λ·â + x_t  (mode='A')  or  â := x_t  (mode='O').

        ``hid2weight_jac[0]`` is a Dict[ETraceX_Key, jax.Array] of x values at time t.
        We ignore ``hid2hid_jac`` (OTTT's core approximation).
        """
        if input_is_multi_step:
            raise NotImplementedError('OTTT v1 supports single-step only')
        xs_at_t, _dfs_at_t = hid2weight_jac[0], hid2weight_jac[1]

        new_vals = {}
        for rid, old in hist_vals.items():
            # Find the matching relation by x_var id.
            x_t = xs_at_t[rid]
            if self.mode == 'A':
                new_vals[rid] = self.leak * old + x_t
            else:  # 'O'
                new_vals[rid] = x_t
        return new_vals

    def _solve_weight_gradients(
        self, running_index, etrace_at_t, dl_to_hidden_groups,
        weight_vals, dl_to_nonetws_at_t, dl_to_etws_at_t,
    ):
        """ΔW = outer(â, L · σ'(u)).

        ``dl_to_hidden_groups[g]`` already carries the L · σ'(u) term (reverse-AD
        injected σ' through the neuron's custom_vjp). We outer-product per
        relation and route to the ParamState path.
        """
        from braintrace._etrace_vjp.misc import _route_grads_by_path, _update_dict

        dG = {path: None for path in self.param_states}
        for rel in self.graph.hidden_param_op_relations:
            a_hat = etrace_at_t[id(rel.x_var)]
            for group in rel.hidden_groups:
                L = dl_to_hidden_groups[group.index]  # shape (*group.varshape, num_state)
                L_proj = L.sum(axis=-1)  # collapse num_state axis
                # outer(a_hat, L_proj) — handle batched vs unbatched x_var.
                if a_hat.ndim == 2:  # batched: (batch, in)
                    dw = jnp.einsum('bi,bo->bio', a_hat, L_proj)
                    dw = dw.sum(axis=0)
                else:
                    dw = jnp.einsum('i,o->io', a_hat, L_proj)
                _route_grads_by_path(rel, {'weight': dw}, weight_vals, dG)

        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG, path, dg)
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG, path, dg, error_when_no_key=True)
        return dG
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/ottt_test.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/ottt.py braintrace/_snn_algorithms/ottt_test.py
git commit -m "feat: OTTT trace update + weight solve"
```

---

## Phase 6 — OSTTP (target projection)

Goal: `OSTTP(model, B_list, target_timing='per-step'|'sequence-end')` inherits D_RTRL trace, overrides `_compute_learning_signal` to return `[B_l @ y_target for each HiddenGroup]`.

### Task 6.1: OSTTP — construction + shape validation

**Files:**
- Create: `braintrace/_snn_algorithms/osttp.py`
- Create: `braintrace/_snn_algorithms/osttp_test.py`

- [ ] **Step 1: Failing test**

`braintrace/_snn_algorithms/osttp_test.py`:

```python
import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._snn_algorithms.osttp import OSTTP


def _osttp_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (4, 3))
            )
            self.h = brainstate.ShortTermState(jnp.zeros((2, 3)))

        def update(self, x):
            pre = jnp.concatenate([x, jax.lax.stop_gradient(self.h.value)], axis=-1)
            self.h.value = jax.nn.tanh(braintrace.matmul(pre, self.w.value))
            return self.h.value

    return Net()


class TestOSTTPConstruction(unittest.TestCase):
    def test_default_target_timing(self):
        B_list = [0.1 * jax.random.normal(jax.random.PRNGKey(1), (5, 3))]
        algo = OSTTP(_osttp_net(), B_list=B_list)
        assert algo.target_timing == 'per-step'

    def test_invalid_timing_raises(self):
        B_list = [0.1 * jax.random.normal(jax.random.PRNGKey(1), (5, 3))]
        with self.assertRaises(ValueError):
            OSTTP(_osttp_net(), B_list=B_list, target_timing='never')

    def test_missing_B_list_raises(self):
        with self.assertRaises(TypeError):
            OSTTP(_osttp_net())

    def test_B_list_wrong_length_raises_on_compile(self):
        B_list = [
            jax.random.normal(jax.random.PRNGKey(1), (5, 3)),
            jax.random.normal(jax.random.PRNGKey(2), (5, 3)),  # one extra
        ]
        net = _osttp_net()
        algo = OSTTP(net, B_list=B_list)
        x = jnp.ones((2, 1))
        with self.assertRaises(ValueError):
            algo.compile_graph(x)

    def test_B_list_wrong_shape_raises_on_compile(self):
        B_list = [jax.random.normal(jax.random.PRNGKey(1), (5, 99))]  # mismatched n_l
        net = _osttp_net()
        algo = OSTTP(net, B_list=B_list)
        x = jnp.ones((2, 1))
        with self.assertRaises(ValueError):
            algo.compile_graph(x)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/osttp_test.py::TestOSTTPConstruction -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement construction + validation**

`braintrace/_snn_algorithms/osttp.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# ... (Apache 2.0 header) ...

"""OSTTP — Online Spatio-Temporal Target Projection (Ortner et al. 2023).

D_RTRL trace machinery + DRTP-style target projection replacing reverse-AD's
∂L/∂h. Each HiddenGroup receives a signal `B_l @ y_target` instead of the
autodiff gradient.
"""

from typing import List, Optional, Sequence

import brainstate
import jax
import jax.numpy as jnp

from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm
from ._common import extract_y_target

__all__ = ['OSTTP']


class OSTTP(ParamDimVjpAlgorithm):
    """Online Spatio-Temporal Target Projection.

    Parameters
    ----------
    model : brainstate.nn.Module
    B_list : Sequence[jax.Array]
        One feedback matrix per HiddenGroup, each of shape (n_target, n_l).
        Frozen (wrapped in stop_gradient at compile).
    target_timing : {'per-step', 'sequence-end'}
        'per-step' applies the projection every step (requires y_target each
        update() call). 'sequence-end' only applies at the final step (users
        must zero y_target on intermediate steps themselves in v1).
    name, vjp_method, fast_solve : forwarded to D_RTRL.
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        B_list: Sequence[jax.Array],
        target_timing: str = 'per-step',
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        **kwargs,
    ):
        if target_timing not in ('per-step', 'sequence-end'):
            raise ValueError(f"target_timing must be 'per-step' or 'sequence-end'; got {target_timing!r}")
        super().__init__(
            model, name=name, vjp_method=vjp_method, fast_solve=fast_solve, **kwargs
        )
        # Freeze each B matrix immediately.
        self._B_list = tuple(jax.lax.stop_gradient(B) for B in B_list)
        self.target_timing = target_timing
        self._current_y_target: Optional[jax.Array] = None

    def compile_graph(self, *args) -> None:
        super().compile_graph(*args)
        n_groups = len(self.graph.hidden_groups)
        if len(self._B_list) != n_groups:
            raise ValueError(
                f'B_list has {len(self._B_list)} entries but model has {n_groups} '
                f'HiddenGroup(s). One B matrix per HiddenGroup is required.'
            )
        for B, group in zip(self._B_list, self.graph.hidden_groups):
            # group's "n_l" is the trailing feature dim of its varshape.
            n_l = int(jnp.prod(jnp.array(group.varshape[-1:])))
            if B.shape[1] != n_l:
                raise ValueError(
                    f'B_list[{group.index}].shape[1] == {B.shape[1]} but HiddenGroup '
                    f'{group.index} has n_l={n_l}.'
                )

    def update(self, x, y_target=None):
        """Call super().update(x) after stashing y_target for the hook."""
        if self.target_timing == 'per-step' and y_target is None:
            raise ValueError(
                "OSTTP(target_timing='per-step') requires y_target at every update() call."
            )
        self._current_y_target = y_target
        try:
            return super().update(x)
        finally:
            self._current_y_target = None
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/osttp_test.py::TestOSTTPConstruction -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/osttp.py braintrace/_snn_algorithms/osttp_test.py
git commit -m "feat: OSTTP construction + B_list validation"
```

---

### Task 6.2: OSTTP — `_compute_learning_signal` override

**Files:**
- Modify: `braintrace/_snn_algorithms/osttp.py`
- Modify: `braintrace/_snn_algorithms/osttp_test.py`

- [ ] **Step 1: Failing test**

Append to `osttp_test.py`:

```python
class TestOSTTPTargetProjection(unittest.TestCase):
    def test_learning_signal_equals_B_at_y_target(self):
        key = jax.random.PRNGKey(7)
        B = 0.1 * jax.random.normal(key, (4, 3))
        net = _osttp_net()
        algo = OSTTP(net, B_list=[B])
        x = jnp.ones((2, 1))
        algo.compile_graph(x)
        algo.init_etrace_state()

        y_target = jnp.arange(8.0).reshape(2, 4)
        algo._current_y_target = y_target
        # Build a fake autodiff signal (random, shape must match HiddenGroup).
        fake = [jnp.zeros((2, 3))]  # one HiddenGroup, shape (batch=2, n_l=3)
        out = algo._compute_learning_signal(fake, args=(x, y_target))
        expected = y_target @ B
        assert jnp.allclose(out[0], expected, atol=1e-6)

    def test_target_differs_from_symmetric(self):
        key = jax.random.PRNGKey(7)
        B = 0.1 * jax.random.normal(key, (4, 3))
        net = _osttp_net()
        algo = OSTTP(net, B_list=[B])
        x = jnp.ones((2, 1))
        y = jnp.ones((2, 4))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x):
            out = algo.update(x, y_target=y)
            return (out ** 2).sum()

        grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
        g_osttp = next(iter(grads.values()))['weight']
        # Symmetric baseline (D_RTRL with same model).
        from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm
        net2 = _osttp_net()
        algo2 = ParamDimVjpAlgorithm(net2)
        algo2.compile_graph(x)
        algo2.init_etrace_state()
        grads2, _ = brainstate.augment.grad(loss, algo2.param_states, return_value=True)(x)
        g_drtrl = next(iter(grads2.values()))['weight']
        assert not jnp.allclose(g_osttp, g_drtrl, atol=1e-4)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/osttp_test.py::TestOSTTPTargetProjection -v
```
Expected: FAIL (hook not overridden yet; default identity returns autodiff).

- [ ] **Step 3: Add `_compute_learning_signal` override**

Append to the `OSTTP` class in `osttp.py`:

```python
    def _compute_learning_signal(self, dl_autodiff, args):
        """Replace reverse-AD dl/dh with B_l @ y_target for each HiddenGroup."""
        y_target = self._current_y_target
        if y_target is None:
            # target_timing='sequence-end' on a non-terminal step: pass zeros so
            # traces accumulate without weight updates.
            return [jnp.zeros_like(s) for s in dl_autodiff]
        out = []
        for gid, s in enumerate(dl_autodiff):
            B = self._B_list[gid]
            projected = y_target @ B
            # Broadcast/reshape projected into the shape reverse-AD returned.
            out.append(projected.reshape(s.shape))
        return out
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/osttp_test.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/osttp.py braintrace/_snn_algorithms/osttp_test.py
git commit -m "feat: OSTTP _compute_learning_signal target projection"
```

---

## Phase 7 — OTPE (leaky-additive R_hat, cross-layer solve)

Goal: `OTPE(model, mode='full'|'approx', leak=None)` stores per-HiddenGroup R_hat as a `ShortTermState`, updates via `R_hat ← λ·R_hat + hid2weight_jac_local`, and inside `_solve_weight_gradients` contracts `(L_l · W_next^T) ⊗ R_hat_{l-1}` for each layer pair.

### Task 7.1: OTPE — construction + R_hat allocation

**Files:**
- Create: `braintrace/_snn_algorithms/otpe.py`
- Create: `braintrace/_snn_algorithms/otpe_test.py`

- [ ] **Step 1: Failing test**

`braintrace/_snn_algorithms/otpe_test.py`:

```python
import unittest
import warnings

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._snn_algorithms.otpe import OTPE


class FakeLIF(brainstate.ShortTermState):
    def __init__(self, init_value, leak):
        super().__init__(init_value)
        self.leak = leak


def _otpe_net_single_layer():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.v = FakeLIF(jnp.zeros((2, 3)), leak=0.9)

        def update(self, x):
            self.v.value = 0.9 * self.v.value + braintrace.matmul(x, self.w.value)
            return self.v.value

    return Net()


class TestOTPEConstruction(unittest.TestCase):
    def test_default_mode_full(self):
        algo = OTPE(_otpe_net_single_layer(), leak=0.9)
        assert algo.mode == 'full'

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            OTPE(_otpe_net_single_layer(), mode='bogus', leak=0.9)

    def test_leak_resolved_from_model(self):
        algo = OTPE(_otpe_net_single_layer())
        assert algo.leak == 0.9

    def test_compile_allocates_R_hat(self):
        algo = OTPE(_otpe_net_single_layer(), leak=0.9)
        x = jnp.ones((2, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        assert len(algo._R_hat) == len(algo.graph.hidden_param_op_relations)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/otpe_test.py::TestOTPEConstruction -v
```
Expected: FAIL.

- [ ] **Step 3: Implement construction + R_hat allocation**

`braintrace/_snn_algorithms/otpe.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# ... (Apache 2.0 header) ...

"""OTPE — Online Training with Postsynaptic Estimates (Summe et al. 2024).

Replaces RTRL's full Jacobian with a leaky-additive per-parameter accumulator
R_hat ← λ·R_hat + ∂s/∂θ_local. Cross-layer coupling is handled inside
_solve_weight_gradients, NOT by relaxing the compiler.
"""

from typing import Dict, List, Optional, Sequence

import brainstate
import jax
import jax.numpy as jnp

from braintrace._etrace_vjp.base import ETraceVjpAlgorithm
from ._common import _resolve_leak

__all__ = ['OTPE']


class OTPE(ETraceVjpAlgorithm):
    """Online Training with Postsynaptic Estimates.

    Parameters
    ----------
    model : brainstate.nn.Module
    mode : {'full', 'approx'}
        'full' keeps the (batch, I, O) R_hat per layer. 'approx' factors R_hat
        as outer(ḡ_out, ẑ_in) for O(I+O) memory (F-OTPE variant); issues a
        UserWarning for depth > 1.
    leak : float, optional
        λ factor. If None, resolved via `_resolve_leak`.
    name, vjp_method, trace_clip_abs : forwarded / stored.
    trace_clip_abs : float or None
        Elementwise clip on R_hat each step. None disables.
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        mode: str = 'full',
        leak: Optional[float] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        trace_clip_abs: Optional[float] = None,
        **kwargs,
    ):
        if mode not in ('full', 'approx'):
            raise ValueError(f"mode must be 'full' or 'approx'; got {mode!r}")
        super().__init__(model, name=name, vjp_method=vjp_method)
        self.mode = mode
        self.leak = _resolve_leak(model, leak)
        self.trace_clip_abs = trace_clip_abs
        self._R_hat: Dict[int, brainstate.ShortTermState] = {}

    def compile_graph(self, *args) -> None:
        super().compile_graph(*args)
        if self.mode == 'approx':
            n_groups = len(self.graph.hidden_groups)
            if n_groups > 1:
                import warnings as _w
                _w.warn(
                    'OTPE(mode=\'approx\') bias compounds with network depth; '
                    'consider F-OTPE or mode=\'full\'.',
                    UserWarning,
                )
        # OTPE invariant: each HiddenParamOpRelation's hidden_groups must be length 1.
        for rel in self.graph.hidden_param_op_relations:
            if len(rel.hidden_groups) != 1:
                raise ValueError(
                    f'OTPE requires per-layer one-hop weight-to-hidden relations; '
                    f'found relation {id(rel)} reaching {len(rel.hidden_groups)} groups.'
                )

    def init_etrace_state(self, *args, **kwargs):
        self._R_hat = {}
        for rel in self.graph.hidden_param_op_relations:
            rid = id(rel.y_var)
            if rid in self._R_hat:
                continue
            # Shape matches the weight: for mm_p, (in, out).
            weight_shape = rel.trainable_vars[next(iter(rel.trainable_vars))].aval.shape
            # Additionally carry a leading batch axis if the primitive is batched.
            from braintrace._etrace_op import is_batched_primitive
            if is_batched_primitive(rel.primitive):
                batch = rel.x_var.aval.shape[0]
                shape = (batch, *weight_shape)
            else:
                shape = weight_shape
            self._R_hat[rid] = brainstate.ShortTermState(jnp.zeros(shape, dtype=jnp.float32))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.running_index.value = 0
        for r in self._R_hat.values():
            # Reshape if batch changed.
            old_shape = r.value.shape
            new_shape = (batch_size, *old_shape[1:]) if batch_size is not None else old_shape
            r.value = jnp.zeros(new_shape, dtype=r.value.dtype)

    def _get_etrace_data(self):
        return {rid: r.value for rid, r in self._R_hat.items()}

    def _assign_etrace_data(self, vals):
        for rid, v in vals.items():
            self._R_hat[rid].value = v

    # Protocol methods completed in Task 7.2.
    def _update_etrace_data(self, *a, **kw):
        raise NotImplementedError('Task 7.2')

    def _solve_weight_gradients(self, *a, **kw):
        raise NotImplementedError('Task 7.2')
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/otpe_test.py::TestOTPEConstruction -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/otpe.py braintrace/_snn_algorithms/otpe_test.py
git commit -m "feat: OTPE construction + R_hat allocation"
```

---

### Task 7.2: OTPE — trace update + weight solve (single-layer)

**Files:**
- Modify: `braintrace/_snn_algorithms/otpe.py`
- Modify: `braintrace/_snn_algorithms/otpe_test.py`

- [ ] **Step 1: Failing test**

Append to `otpe_test.py`:

```python
class TestOTPESingleLayer(unittest.TestCase):
    def test_update_runs_and_produces_gradients(self):
        net = _otpe_net_single_layer()
        algo = OTPE(net, leak=0.9, mode='full')
        x = jnp.ones((2, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()

        def loss(x):
            return (algo.update(x) ** 2).sum()

        grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
        g = next(iter(grads.values()))['weight']
        assert g.shape == (3, 3)
        assert jnp.any(g != 0.0)

    def test_leak_zero_reduces_to_ostl(self):
        """OTPE with λ=0 == OSTL-with-H on single-layer (R_hat degenerates to ∂s/∂θ)."""
        from braintrace._snn_algorithms.ostl import OSTL

        def compute(algo):
            net = _otpe_net_single_layer()
            # rebind
            if isinstance(algo, str):
                if algo == 'otpe':
                    a = OTPE(net, leak=1e-12, mode='full')  # ≈ 0
                else:
                    a = OSTL(net, regime='with-H')
            x = jnp.ones((2, 3))
            a.compile_graph(x)
            a.init_etrace_state()
            def loss(x): return (a.update(x) ** 2).sum()
            grads, _ = brainstate.augment.grad(loss, a.param_states, return_value=True)(x)
            return next(iter(grads.values()))['weight']

        g_otpe = compute('otpe')
        g_ostl = compute('ostl')
        # Equivalence holds to within O(leak) due to leak being tiny-but-not-zero.
        assert jnp.allclose(g_otpe, g_ostl, atol=1e-3)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/otpe_test.py::TestOTPESingleLayer -v
```
Expected: FAIL (NotImplementedError).

- [ ] **Step 3: Implement trace update + single-layer solve**

Replace stubs in `otpe.py`:

```python
    def _update_etrace_data(
        self, running_index, hist_vals,
        hid2weight_jac, hid2hid_jac, weight_vals, input_is_multi_step,
    ):
        """R_hat ← λ·R_hat + ∂s/∂θ_local. Ignores hid2hid_jac."""
        if input_is_multi_step:
            raise NotImplementedError('OTPE v1 supports single-step only')
        xs, dfs = hid2weight_jac[0], hid2weight_jac[1]

        new_R = {}
        for rel in self.graph.hidden_param_op_relations:
            rid = id(rel.y_var)
            group = rel.hidden_groups[0]  # OTPE invariant: exactly one group
            # Local ∂s/∂θ = x ⊗ df (for mm_p).
            from braintrace._misc import etrace_df_key
            x = xs[id(rel.x_var)]
            df = dfs[etrace_df_key(rel.y_var, group.index)]
            # Collapse num_state axis.
            df_proj = df.sum(axis=-1)
            from braintrace._etrace_op import is_batched_primitive
            if is_batched_primitive(rel.primitive):
                local = jnp.einsum('bi,bo->bio', x, df_proj)
            else:
                local = jnp.einsum('i,o->io', x, df_proj)
            updated = self.leak * hist_vals[rid] + local
            if self.trace_clip_abs is not None:
                updated = jnp.clip(updated, -self.trace_clip_abs, self.trace_clip_abs)
            new_R[rid] = updated
        return new_R

    def _solve_weight_gradients(
        self, running_index, etrace_at_t, dl_to_hidden_groups,
        weight_vals, dl_to_nonetws_at_t, dl_to_etws_at_t,
    ):
        """ΔW_l = (L_l) ⊗ R_hat_l  for the single-layer case.

        Cross-layer case (multi HiddenGroup) handled in Task 7.3.
        """
        from braintrace._etrace_vjp.misc import _route_grads_by_path, _update_dict

        dG = {path: None for path in self.param_states}
        for rel in self.graph.hidden_param_op_relations:
            rid = id(rel.y_var)
            R = etrace_at_t[rid]
            group = rel.hidden_groups[0]
            L = dl_to_hidden_groups[group.index].sum(axis=-1)  # collapse num_state
            # ΔW = L ⊙ R_hat (elementwise over the weight axes, summed over batch).
            from braintrace._etrace_op import is_batched_primitive
            if is_batched_primitive(rel.primitive):
                dw = jnp.einsum('bo,bio->io', L, R)
            else:
                dw = jnp.einsum('o,io->io', L, R)
            _route_grads_by_path(rel, {'weight': dw}, weight_vals, dG)

        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG, path, dg)
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG, path, dg, error_when_no_key=True)
        return dG
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/otpe_test.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/otpe.py braintrace/_snn_algorithms/otpe_test.py
git commit -m "feat: OTPE single-layer trace update + solve"
```

---

### Task 7.3: OTPE — approx mode (F-OTPE factorization)

**Files:**
- Modify: `braintrace/_snn_algorithms/otpe.py`
- Modify: `braintrace/_snn_algorithms/otpe_test.py`

- [ ] **Step 1: Failing test**

Append to `otpe_test.py`:

```python
class TestOTPEApproxMode(unittest.TestCase):
    def test_approx_uses_factored_traces(self):
        net = _otpe_net_single_layer()
        algo = OTPE(net, mode='approx', leak=0.9)
        x = jnp.ones((2, 3))
        algo.compile_graph(x)
        algo.init_etrace_state()
        # Factored storage: one (batch, in) + one (batch, out) per relation.
        assert len(algo._R_hat_x) == len(algo.graph.hidden_param_op_relations)
        assert len(algo._R_hat_g) == len(algo.graph.hidden_param_op_relations)

    def test_approx_warns_on_multi_layer(self):
        # Build a 2-HiddenGroup net.
        class TwoLayer(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 4))
                )
                self.v1 = FakeLIF(jnp.zeros((2, 4)), leak=0.9)
                self.w2 = brainstate.ParamState(
                    0.1 * jax.random.normal(jax.random.PRNGKey(1), (4, 3))
                )
                self.v2 = FakeLIF(jnp.zeros((2, 3)), leak=0.9)

            def update(self, x):
                self.v1.value = 0.9 * self.v1.value + braintrace.matmul(x, self.w1.value)
                s = jax.lax.stop_gradient((self.v1.value > 0).astype(jnp.float32))
                self.v2.value = 0.9 * self.v2.value + braintrace.matmul(s, self.w2.value)
                return self.v2.value

        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            algo = OTPE(TwoLayer(), mode='approx', leak=0.9)
            x = jnp.ones((2, 3))
            algo.compile_graph(x)
        assert any('approx' in str(w.message).lower() for w in wlist)
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/otpe_test.py::TestOTPEApproxMode -v
```
Expected: FAIL (`_R_hat_x` / `_R_hat_g` don't exist).

- [ ] **Step 3: Implement approx mode factored storage**

Modify `init_etrace_state` in `otpe.py` to branch on mode:

```python
    def init_etrace_state(self, *args, **kwargs):
        self._R_hat = {}
        self._R_hat_x = {}
        self._R_hat_g = {}
        from braintrace._etrace_op import is_batched_primitive
        for rel in self.graph.hidden_param_op_relations:
            rid = id(rel.y_var)
            in_shape = rel.x_var.aval.shape
            out_shape = rel.y_var.aval.shape
            if self.mode == 'full':
                weight_shape = rel.trainable_vars[next(iter(rel.trainable_vars))].aval.shape
                if is_batched_primitive(rel.primitive):
                    shape = (in_shape[0], *weight_shape)
                else:
                    shape = weight_shape
                self._R_hat[rid] = brainstate.ShortTermState(jnp.zeros(shape, dtype=jnp.float32))
            else:  # 'approx'
                self._R_hat_x[rid] = brainstate.ShortTermState(jnp.zeros(in_shape, dtype=jnp.float32))
                self._R_hat_g[rid] = brainstate.ShortTermState(jnp.zeros(out_shape, dtype=jnp.float32))
```

Update `_get_etrace_data`, `_assign_etrace_data`, `reset_state`, `_update_etrace_data`, `_solve_weight_gradients` to branch on `self.mode`. For `_update_etrace_data` in 'approx':

```python
        if self.mode == 'approx':
            new_Rx, new_Rg = {}, {}
            for rel in self.graph.hidden_param_op_relations:
                rid = id(rel.y_var)
                group = rel.hidden_groups[0]
                x = xs[id(rel.x_var)]
                from braintrace._misc import etrace_df_key
                df = dfs[etrace_df_key(rel.y_var, group.index)].sum(axis=-1)
                new_Rx[rid] = self.leak * hist_vals[0][rid] + x
                new_Rg[rid] = self.leak * hist_vals[1][rid] + df
            return (new_Rx, new_Rg)
```

And for 'approx' in `_solve_weight_gradients`:

```python
        if self.mode == 'approx':
            Rx, Rg = etrace_at_t
            for rel in self.graph.hidden_param_op_relations:
                rid = id(rel.y_var)
                group = rel.hidden_groups[0]
                L = dl_to_hidden_groups[group.index].sum(axis=-1)
                from braintrace._etrace_op import is_batched_primitive
                if is_batched_primitive(rel.primitive):
                    # ΔW = outer(R_hat_x_i, (L ⊙ R_hat_g)_o) summed over batch.
                    dw = jnp.einsum('bi,bo->io', Rx[rid], L * Rg[rid])
                else:
                    dw = jnp.einsum('i,o->io', Rx[rid], L * Rg[rid])
                _route_grads_by_path(rel, {'weight': dw}, weight_vals, dG)
```

Make `_get_etrace_data` / `_assign_etrace_data` return/accept `(dict, dict)` in approx mode and single `dict` in full mode. Parallel branch in `reset_state`.

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/otpe_test.py -v
```
Expected: all pass (including full-mode regression).

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/otpe.py braintrace/_snn_algorithms/otpe_test.py
git commit -m "feat: OTPE approx mode (F-OTPE factored R_hat)"
```

---

## Phase 8 — Spike-reset leakage diagnostic

Goal: best-effort jaxpr inspection emitting a `SPIKE_RESET_LEAKAGE` diagnostic when a hidden state's update consumes another HiddenState's value without a `stop_gradient_p` eqn in between AND that value's upstream has a `custom_vjp_call_jaxpr` with comparison ops.

### Task 8.1: Add `DiagnosticKind.SPIKE_RESET_LEAKAGE`

**Files:**
- Modify: `braintrace/_etrace_compiler/diagnostics.py`

- [ ] **Step 1: Write the failing test**

Append to `braintrace/_etrace_compiler/diagnostics_test.py` (or create if absent with an Apache header and a test class scaffold similar to siblings):

```python
def test_spike_reset_leakage_kind_exists():
    from braintrace._etrace_compiler.diagnostics import DiagnosticKind
    assert DiagnosticKind.SPIKE_RESET_LEAKAGE.value == 'spike_reset_leakage'
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_etrace_compiler/diagnostics_test.py::test_spike_reset_leakage_kind_exists -v
```
Expected: FAIL (AttributeError).

- [ ] **Step 3: Add the enum value**

In `braintrace/_etrace_compiler/diagnostics.py`, add to the `DiagnosticKind` enum (after the existing entries around line 80):

```python
    # Heuristic diagnostic for soft-reset spike-leakage (best-effort jaxpr inspection)
    SPIKE_RESET_LEAKAGE = 'spike_reset_leakage'
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_etrace_compiler/diagnostics_test.py::test_spike_reset_leakage_kind_exists -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_etrace_compiler/diagnostics.py braintrace/_etrace_compiler/diagnostics_test.py
git commit -m "feat: add SPIKE_RESET_LEAKAGE diagnostic kind"
```

---

### Task 8.2: Jaxpr walk emitting the diagnostic

**Files:**
- Modify: `braintrace/_etrace_compiler/graph.py`
- Create: `braintrace/_etrace_compiler/graph_leakage_test.py`

- [ ] **Step 1: Failing test**

`braintrace/_etrace_compiler/graph_leakage_test.py`:

```python
"""Spike-reset leakage heuristic — best-effort test."""
import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._etrace_compiler.diagnostics import DiagnosticKind, diagnostic_context


class TestSpikeResetLeakage(unittest.TestCase):
    def test_warn_when_spike_not_stop_gradient(self):
        @jax.custom_vjp
        def spike(v):
            return (v > 1.0).astype(v.dtype)

        def spike_fwd(v):
            return spike(v), v

        def spike_bwd(v, g):
            return (g * jax.nn.sigmoid(v),)

        spike.defvjp(spike_fwd, spike_bwd)

        class LeakyNet(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.eye(3))
                self.v = brainstate.ShortTermState(jnp.zeros((2, 3)))

            def update(self, x):
                inp = braintrace.matmul(x, self.w.value)
                z = spike(self.v.value)          # no stop_gradient!
                self.v.value = 0.9 * self.v.value - z + inp
                return self.v.value

        net = LeakyNet()
        x = jnp.ones((2, 3))
        with diagnostic_context() as reporter:
            braintrace.compile_etrace_graph(net, x)
        kinds = [r.kind for r in reporter.records]
        assert DiagnosticKind.SPIKE_RESET_LEAKAGE in kinds

    def test_no_warn_when_stop_gradient_present(self):
        @jax.custom_vjp
        def spike(v): return (v > 1.0).astype(v.dtype)
        def spike_fwd(v): return spike(v), v
        def spike_bwd(v, g): return (g * jax.nn.sigmoid(v),)
        spike.defvjp(spike_fwd, spike_bwd)

        class SafeNet(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.eye(3))
                self.v = brainstate.ShortTermState(jnp.zeros((2, 3)))

            def update(self, x):
                inp = braintrace.matmul(x, self.w.value)
                z = jax.lax.stop_gradient(spike(self.v.value))
                self.v.value = 0.9 * self.v.value - z + inp
                return self.v.value

        with diagnostic_context() as reporter:
            braintrace.compile_etrace_graph(SafeNet(), jnp.ones((2, 3)))
        kinds = [r.kind for r in reporter.records]
        assert DiagnosticKind.SPIKE_RESET_LEAKAGE not in kinds
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_etrace_compiler/graph_leakage_test.py -v
```
Expected: FAIL (no emit logic yet).

- [ ] **Step 3: Implement heuristic in `graph.py`**

In `braintrace/_etrace_compiler/graph.py`, locate the point where `HiddenParamOpRelation.transition_jaxpr` is fully built (search for "transition_jaxpr" construction). After each relation's transition jaxpr is available, walk its equations:

```python
def _detect_spike_reset_leakage(transition_jaxpr, reporter):
    """Heuristic: detect unshielded spike feedback in a transition jaxpr.

    Rule: an equation whose invar is another HiddenState value whose upstream
    contains a `custom_vjp_call_jaxpr` eqn with a comparison (`gt_p`, `ge_p`,
    `lt_p`, `le_p`), and whose invar itself is NOT produced by a
    `stop_gradient_p` eqn.
    """
    from jax._src.lax.lax import stop_gradient_p
    try:
        from jax._src.custom_derivatives import custom_vjp_call_jaxpr_p
    except ImportError:
        return  # JAX version without this primitive — skip
    from jax._src.lax.lax import gt_p, ge_p, lt_p, le_p

    stop_grad_outs = set()
    for eqn in transition_jaxpr.eqns:
        if eqn.primitive is stop_gradient_p:
            stop_grad_outs.update(id(v) for v in eqn.outvars)

    for eqn in transition_jaxpr.eqns:
        if eqn.primitive is custom_vjp_call_jaxpr_p:
            sub = eqn.params.get('fun_jaxpr', None)
            if sub is None:
                continue
            has_cmp = any(sub_eqn.primitive in (gt_p, ge_p, lt_p, le_p)
                          for sub_eqn in sub.jaxpr.eqns)
            if has_cmp:
                for out in eqn.outvars:
                    if id(out) not in stop_grad_outs:
                        from .diagnostics import emit, DiagnosticKind, DiagnosticLevel, CompilationRecord
                        emit(reporter, CompilationRecord(
                            kind=DiagnosticKind.SPIKE_RESET_LEAKAGE,
                            level=DiagnosticLevel.WARNING,
                            message=(
                                'Heuristic: spike output from a custom_vjp with threshold '
                                'comparison is not shielded by stop_gradient. '
                                'Surrogate gradients may leak through the reset term.'
                            ),
                            context={'var': repr(out)},
                        ))
```

Call `_detect_spike_reset_leakage(rel.transition_jaxpr, get_reporter())` inside the loop that finalizes each relation, guarded by `if self.check_spike_reset_leakage`.

For the guard flag, add a constructor param on `compile_etrace_graph` (or the compiler function, wherever graph construction begins). Default `True`.

The exact code location: search for where `HiddenParamOpRelation` objects are appended to the graph's relations list in `graph.py`. Insert the call immediately after each `relation.transition_jaxpr` is assigned.

**Quick audit first** — run `grep -n "transition_jaxpr" braintrace/_etrace_compiler/graph.py` to find the assignment points.

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_etrace_compiler/graph_leakage_test.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_etrace_compiler/graph.py braintrace/_etrace_compiler/graph_leakage_test.py
git commit -m "feat: spike-reset leakage jaxpr heuristic"
```

---

### Task 8.3: Wire `check_spike_reset_leakage` flag into SNN classes

**Files:**
- Modify: `braintrace/_snn_algorithms/{e_prop,ostl,ottt,osttp,otpe}.py`

Each of the 5 classes accepts `check_spike_reset_leakage: bool = True`. The flag is forwarded to `compile_graph()` if the compiler exposes it as a parameter; otherwise it toggles `diagnostic_context()` during compilation.

- [ ] **Step 1: Failing integration test**

Create `braintrace/_snn_algorithms/diagnostic_flag_test.py`:

```python
import unittest
import brainstate, jax, jax.numpy as jnp
import braintrace
from braintrace._snn_algorithms import EProp, OSTL, OTPE, OTTT, OSTTP
from braintrace._etrace_compiler.diagnostics import DiagnosticKind


class FakeLIF(brainstate.ShortTermState):
    def __init__(self, iv, leak):
        super().__init__(iv); self.leak = leak


@jax.custom_vjp
def leaky_spike(v): return (v > 1.0).astype(v.dtype)
leaky_spike.defvjp(lambda v: (leaky_spike(v), v), lambda v, g: (g * jax.nn.sigmoid(v),))


def _net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(jnp.eye(3))
            self.v = FakeLIF(jnp.zeros((2, 3)), leak=0.9)
        def update(self, x):
            self.v.value = 0.9 * self.v.value - leaky_spike(self.v.value) + braintrace.matmul(x, self.w.value)
            return self.v.value
    return Net()


class TestSpikeResetFlag(unittest.TestCase):
    def test_flag_true_emits_diagnostic(self):
        for cls, kw in [
            (EProp, {}), (OSTL, {}), (OTTT, {'leak': 0.9}),
            (OSTTP, {'B_list': [jnp.zeros((1, 3))]}), (OTPE, {'leak': 0.9}),
        ]:
            algo = cls(_net(), check_spike_reset_leakage=True, **kw)
            # Compile records diagnostics on the graph's reporter.
            algo.compile_graph(jnp.ones((2, 3)))
            kinds = [r.kind for r in algo.graph.diagnostics]
            assert DiagnosticKind.SPIKE_RESET_LEAKAGE in kinds, f'{cls.__name__}'

    def test_flag_false_suppresses(self):
        algo = EProp(_net(), check_spike_reset_leakage=False)
        algo.compile_graph(jnp.ones((2, 3)))
        kinds = [r.kind for r in algo.graph.diagnostics]
        assert DiagnosticKind.SPIKE_RESET_LEAKAGE not in kinds
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/diagnostic_flag_test.py -v
```
Expected: FAIL.

- [ ] **Step 3: Add `check_spike_reset_leakage` parameter to each of the 5 classes**

In each of `e_prop.py`, `ostl.py`, `ottt.py`, `osttp.py`, `otpe.py`, add a `check_spike_reset_leakage: bool = True` constructor parameter. Store on `self`. In `compile_graph` override (add if missing) wrap the super call:

```python
    def compile_graph(self, *args) -> None:
        if not self.check_spike_reset_leakage:
            # Temporarily flip the compiler's leakage-check flag via a context.
            from braintrace._etrace_compiler.diagnostics import diagnostic_context
            # Compile in a nested context that filters out SPIKE_RESET_LEAKAGE.
            super().compile_graph(*args)
            self.graph.diagnostics = [
                r for r in self.graph.diagnostics
                if r.kind is not DiagnosticKind.SPIKE_RESET_LEAKAGE
            ]
        else:
            super().compile_graph(*args)
```

Note: OSTL is a factory not a class; add the flag as a kwarg and stash on the returned instance — or switch OSTL to a subclass. Simpler: override `compile_graph` on the returned instance. Cleanest: change the OSTL factory so that after construction it patches `instance.check_spike_reset_leakage` and monkey-patches the compile_graph filter. Implement as:

```python
# in ostl.py, after `algo = ParamDimVjpAlgorithm(...)` / IODimVjpAlgorithm(...)
    algo.check_spike_reset_leakage = check_spike_reset_leakage
    _orig_compile = algo.compile_graph
    def _filtered_compile(*args, _orig=_orig_compile, _a=algo):
        _orig(*args)
        if not _a.check_spike_reset_leakage:
            _a.graph.diagnostics = [
                r for r in _a.graph.diagnostics
                if r.kind is not DiagnosticKind.SPIKE_RESET_LEAKAGE
            ]
    algo.compile_graph = _filtered_compile
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/diagnostic_flag_test.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/*.py
git commit -m "feat: check_spike_reset_leakage flag on SNN algorithm classes"
```

---

## Phase 9 — Public API re-exports

### Task 9.1: Re-export from `braintrace._snn_algorithms.__init__`

**Files:**
- Modify: `braintrace/_snn_algorithms/__init__.py`
- Modify: `braintrace/__init__.py`

- [ ] **Step 1: Failing test**

Create `braintrace/_snn_algorithms/public_api_test.py`:

```python
import unittest

class TestPublicAPI(unittest.TestCase):
    def test_subpackage_exports(self):
        import braintrace._snn_algorithms as pkg
        for name in ('EProp', 'OSTL', 'OTPE', 'OTTT', 'OSTTP',
                     'FixedRandomFeedback', 'KappaFilter', 'PresynapticTrace'):
            assert hasattr(pkg, name), f'missing export: {name}'

    def test_top_level_exports(self):
        import braintrace
        for name in ('EProp', 'OSTL', 'OTPE', 'OTTT', 'OSTTP'):
            assert hasattr(braintrace, name), f'missing top-level export: {name}'
            assert name in braintrace.__all__
```

- [ ] **Step 2: Run, confirm FAIL**

```bash
python -m pytest braintrace/_snn_algorithms/public_api_test.py -v
```
Expected: FAIL.

- [ ] **Step 3: Update imports**

In `braintrace/_snn_algorithms/__init__.py`:

```python
from ._common import PresynapticTrace, KappaFilter, FixedRandomFeedback
from .e_prop import EProp
from .ostl import OSTL
from .ottt import OTTT
from .osttp import OSTTP
from .otpe import OTPE

__all__ = [
    'EProp', 'OSTL', 'OTPE', 'OTTT', 'OSTTP',
    'FixedRandomFeedback', 'KappaFilter', 'PresynapticTrace',
]
```

In `braintrace/__init__.py`, add after the `_etrace_vjp` import block:

```python
from ._snn_algorithms import EProp, OSTL, OTPE, OTTT, OSTTP
```

And append to `__all__`:

```python
    # SNN online-learning algorithms
    'EProp', 'OSTL', 'OTPE', 'OTTT', 'OSTTP',
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/public_api_test.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/__init__.py braintrace/__init__.py braintrace/_snn_algorithms/public_api_test.py
git commit -m "feat: re-export EProp/OSTL/OTPE/OTTT/OSTTP at top level"
```

---

## Phase 10 — Cross-check + integration + fixture tests

### Task 10.1: Cross-check tests

**Files:**
- Create: `braintrace/_snn_algorithms/cross_check_test.py`

- [ ] **Step 1: Write the test file**

`braintrace/_snn_algorithms/cross_check_test.py`:

```python
"""Cross-class equivalence proofs.

These tests assert reduction identities from the spec:
- OSTL(without-H) with identity surrogate == D_RTRL on the same model
- EProp(κ=0, symmetric) == D_RTRL up to float tolerance
- OTPE(leak≈0, full) == OSTL(with-H) on single-layer net
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._etrace_vjp.d_rtrl import ParamDimVjpAlgorithm
from braintrace._snn_algorithms import OSTL, EProp, OTPE


def _net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            )
            self.h = brainstate.ShortTermState(jnp.zeros((2, 3)))

        def update(self, x):
            self.h.value = jax.nn.tanh(self.h.value + braintrace.matmul(x, self.w.value))
            return self.h.value

    return Net()


def _grad(algo):
    x = jnp.ones((2, 3))
    algo.compile_graph(x)
    algo.init_etrace_state()

    def loss(x):
        return (algo.update(x) ** 2).sum()

    grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x)
    return next(iter(grads.values()))['weight']


class TestCrossChecks(unittest.TestCase):
    def test_eprop_k0_matches_d_rtrl(self):
        g_eprop = _grad(EProp(_net(), feedback='symmetric', kappa_filter_decay=0.0))
        g_drtrl = _grad(ParamDimVjpAlgorithm(_net()))
        assert jnp.allclose(g_eprop, g_drtrl, atol=1e-6)

    def test_otpe_leak0_matches_ostl(self):
        g_otpe = _grad(OTPE(_net(), leak=1e-10, mode='full'))
        g_ostl = _grad(OSTL(_net(), regime='with-H'))
        assert jnp.allclose(g_otpe, g_ostl, atol=1e-3)
```

- [ ] **Step 2: Run**

```bash
python -m pytest braintrace/_snn_algorithms/cross_check_test.py -v
```
Expected: PASS (all reduction identities hold).

- [ ] **Step 3: Commit**

```bash
git add braintrace/_snn_algorithms/cross_check_test.py
git commit -m "test: cross-class equivalence proofs (EProp/OTPE reductions)"
```

---

### Task 10.2: Integration test — loss decreases

**Files:**
- Create: `braintrace/_snn_algorithms/integration_test.py`

- [ ] **Step 1: Write**

`braintrace/_snn_algorithms/integration_test.py`:

```python
"""End-to-end smoke: each of 5 algos drives a small toy task with decreasing loss."""
import unittest

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._snn_algorithms import EProp, OSTL, OTPE, OTTT, OSTTP


class FakeLIF(brainstate.ShortTermState):
    def __init__(self, iv, leak):
        super().__init__(iv); self.leak = leak


def _toy_net():
    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (4, 3))
            )
            self.v = FakeLIF(jnp.zeros((2, 3)), leak=0.9)

        def update(self, x):
            pre = jnp.concatenate([x, jax.lax.stop_gradient(self.v.value)], axis=-1)
            self.v.value = jax.nn.tanh(braintrace.matmul(pre, self.w.value) + 0.9 * self.v.value)
            return self.v.value

    return Net()


def _run(algo, n_steps=20, lr=0.05, y_target=None, pass_y=False):
    x = jnp.ones((2, 1))
    algo.compile_graph(x)
    algo.init_etrace_state()

    opt = brainstate.optim.SGD(lr=lr)
    opt.register_trainable_weights(algo.param_states)

    losses = []
    for _ in range(n_steps):
        def loss_fn(x):
            out = algo.update(x, y_target=y_target) if pass_y else algo.update(x)
            target = jnp.ones_like(out)
            return ((out - target) ** 2).mean()

        grads, loss_val = brainstate.augment.grad(loss_fn, algo.param_states, return_value=True)(x)
        opt.update(grads)
        losses.append(float(loss_val))
    return losses


class TestSmokeLossDecreases(unittest.TestCase):
    def test_eprop(self):
        losses = _run(EProp(_toy_net()))
        assert losses[-1] < losses[0]

    def test_ostl(self):
        losses = _run(OSTL(_toy_net()))
        assert losses[-1] < losses[0]

    def test_otpe(self):
        losses = _run(OTPE(_toy_net()))
        assert losses[-1] < losses[0]

    def test_ottt(self):
        losses = _run(OTTT(_toy_net()))
        assert losses[-1] < losses[0]

    def test_osttp(self):
        net = _toy_net()
        B = [0.1 * jax.random.normal(jax.random.PRNGKey(9), (3, 3))]
        algo = OSTTP(net, B_list=B)
        y = jnp.ones((2, 3))
        losses = _run(algo, y_target=y, pass_y=True)
        assert losses[-1] < losses[0]
```

- [ ] **Step 2: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/integration_test.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add braintrace/_snn_algorithms/integration_test.py
git commit -m "test: integration — loss-decreases smoke for all 5 algos"
```

---

### Task 10.3: BPTT-reference fixture generator + fixture tests

**Files:**
- Create: `braintrace/_snn_algorithms/fixtures/generate_bptt_reference.py`
- Create: `braintrace/_snn_algorithms/fixtures/bptt_gradients_tiny_lsnn.pkl` (output of the generator)
- Create: `braintrace/_snn_algorithms/fixture_test.py`

- [ ] **Step 1: Write the fixture generator**

`braintrace/_snn_algorithms/fixtures/generate_bptt_reference.py`:

```python
"""One-shot BPTT gradient generator for tiny LSNN on 10 steps.

Run once:
    python -m braintrace._snn_algorithms.fixtures.generate_bptt_reference

Writes `bptt_gradients_tiny_lsnn.pkl` next to this file.
"""
import pickle
from pathlib import Path

import brainstate
import jax
import jax.numpy as jnp


def main():
    key = jax.random.PRNGKey(0)
    in_dim, rec_dim, out_dim = 3, 4, 2
    T = 10

    w_rec = 0.1 * jax.random.normal(key, (in_dim + rec_dim, rec_dim))
    w_out = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (rec_dim, out_dim))

    x_seq = jax.random.normal(jax.random.PRNGKey(2), (T, rec_dim, in_dim))  # batch=rec_dim=4

    def forward(w_rec, w_out):
        h = jnp.zeros((rec_dim, rec_dim))
        total = 0.0
        for t in range(T):
            pre = jnp.concatenate([x_seq[t], h], axis=-1)
            h = jnp.tanh(pre @ w_rec)
            y = h @ w_out
            total = total + (y ** 2).sum()
        return total

    grads = jax.grad(forward, argnums=(0, 1))(w_rec, w_out)

    out = {
        'x_seq': x_seq,
        'w_rec': w_rec,
        'w_out': w_out,
        'grad_w_rec': grads[0],
        'grad_w_out': grads[1],
    }
    path = Path(__file__).parent / 'bptt_gradients_tiny_lsnn.pkl'
    path.write_bytes(pickle.dumps(out))
    print(f'wrote {path} ({path.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it to produce the fixture**

```bash
cd /mnt/d/codes/projects/braintrace && python -m braintrace._snn_algorithms.fixtures.generate_bptt_reference
```
Expected: prints `wrote .../bptt_gradients_tiny_lsnn.pkl`.

- [ ] **Step 3: Write the fixture test**

`braintrace/_snn_algorithms/fixture_test.py`:

```python
"""Cross-check SNN algos against BPTT reference gradients on tiny LSNN."""
import pickle
import unittest
from pathlib import Path

import brainstate
import jax
import jax.numpy as jnp

import braintrace
from braintrace._snn_algorithms import EProp, OSTL


def _load_fixture():
    path = Path(__file__).parent / 'fixtures' / 'bptt_gradients_tiny_lsnn.pkl'
    return pickle.loads(path.read_bytes())


def _run_online(algo_cls, w_rec, w_out, x_seq, **algo_kw):
    class LSNN(brainstate.nn.Module):
        def __init__(self, w_rec_, w_out_):
            super().__init__()
            self.w_rec = brainstate.ParamState(w_rec_)
            self.w_out = brainstate.ParamState(w_out_)
            self.h = brainstate.ShortTermState(jnp.zeros((w_rec_.shape[1],)))

        def update(self, x):
            pre = jnp.concatenate([x, jax.lax.stop_gradient(self.h.value)], axis=-1)
            self.h.value = jnp.tanh(braintrace.matmul(pre, self.w_rec.value))
            return self.h.value @ self.w_out.value

    net = LSNN(w_rec, w_out)
    algo = algo_cls(net, **algo_kw)
    x0 = x_seq[0, 0]
    algo.compile_graph(x0)
    algo.init_etrace_state()
    acc = None
    for t in range(x_seq.shape[0]):
        for b in range(x_seq.shape[1]):
            def loss(x):
                return (algo.update(x) ** 2).sum()
            grads, _ = brainstate.augment.grad(loss, algo.param_states, return_value=True)(x_seq[t, b])
            acc = grads if acc is None else jax.tree.map(lambda a, b: a + b, acc, grads)
    return acc


class TestFixtureAgreement(unittest.TestCase):
    def test_ostl_cosine_similarity(self):
        fx = _load_fixture()
        grads = _run_online(OSTL, fx['w_rec'], fx['w_out'], fx['x_seq'])
        g_rec = next(iter(grads.values()))['weight']
        bptt = fx['grad_w_rec']
        cos = (g_rec * bptt).sum() / (jnp.linalg.norm(g_rec) * jnp.linalg.norm(bptt) + 1e-9)
        assert float(cos) > 0.95

    def test_eprop_cosine_similarity(self):
        fx = _load_fixture()
        grads = _run_online(EProp, fx['w_rec'], fx['w_out'], fx['x_seq'], kappa_filter_decay=0.0)
        g_rec = next(iter(grads.values()))['weight']
        bptt = fx['grad_w_rec']
        cos = (g_rec * bptt).sum() / (jnp.linalg.norm(g_rec) * jnp.linalg.norm(bptt) + 1e-9)
        assert float(cos) > 0.95
```

- [ ] **Step 4: Run, confirm PASS**

```bash
python -m pytest braintrace/_snn_algorithms/fixture_test.py -v
```
Expected: PASS (both cosine similarities > 0.95 on the tanh-based tiny LSNN).

- [ ] **Step 5: Commit**

```bash
git add braintrace/_snn_algorithms/fixtures/ braintrace/_snn_algorithms/fixture_test.py
git commit -m "test: BPTT reference fixture + OSTL/EProp agreement"
```

---

## Phase 11 — Documentation

### Task 11.1: Add docstring examples (doctests)

**Files:**
- Modify: `braintrace/_snn_algorithms/e_prop.py`, `ostl.py`, `ottt.py`, `osttp.py`, `otpe.py`

Each class docstring gets a ≤5-line runnable example after the parameter block. Pattern:

```python
    >>> import brainstate, braintrace, jax.numpy as jnp, jax
    >>> class Net(brainstate.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.w = brainstate.ParamState(0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3)))
    ...         self.h = brainstate.ShortTermState(jnp.zeros((2, 3)))
    ...     def update(self, x):
    ...         self.h.value = jax.nn.tanh(braintrace.matmul(x, self.w.value)); return self.h.value
    >>> algo = braintrace.EProp(Net())
    >>> algo.compile_graph(jnp.ones((2, 3))); algo.init_etrace_state()
```

- [ ] **Step 1: Add doctest blocks to each class**

Edit each `.py` file, append an `Examples` section in the class docstring following numpydoc style. Use the template above, adjusting constructor args per class (OSTTP needs `B_list`, OTTT/OTPE need `leak=0.9`, etc.).

- [ ] **Step 2: Run doctests**

```bash
python -m pytest --doctest-modules braintrace/_snn_algorithms/e_prop.py braintrace/_snn_algorithms/ostl.py braintrace/_snn_algorithms/ottt.py braintrace/_snn_algorithms/osttp.py braintrace/_snn_algorithms/otpe.py
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add braintrace/_snn_algorithms/e_prop.py braintrace/_snn_algorithms/ostl.py braintrace/_snn_algorithms/ottt.py braintrace/_snn_algorithms/osttp.py braintrace/_snn_algorithms/otpe.py
git commit -m "docs: runnable doctest examples for SNN algos"
```

---

### Task 11.2: Update `CLAUDE.md` with new subpackage entry

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the subpackage to the package-structure block**

Open `CLAUDE.md`, find the block starting with:

```
├── _etrace_vjp/                   VJP-based online learning algorithms
```

Insert after it (before the `_legacy/` entry):

```
├── _snn_algorithms/               SNN online-learning algorithms (v0.2+)
│   ├── _common.py                 PresynapticTrace, FixedRandomFeedback, KappaFilter, extract_y_target
│   ├── e_prop.py                  class EProp(ParamDimVjpAlgorithm)
│   ├── ostl.py                    def OSTL(...)           # regime-flag factory
│   ├── ottt.py                    class OTTT(ETraceVjpAlgorithm)
│   ├── osttp.py                   class OSTTP(ParamDimVjpAlgorithm)
│   └── otpe.py                    class OTPE(ETraceVjpAlgorithm)
```

In the "Public API" block, add under the `# Algorithms` subsection:

```
braintrace.EProp, braintrace.OSTL, braintrace.OTPE, braintrace.OTTT, braintrace.OSTTP  # SNN-specific
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: list _snn_algorithms subpackage in CLAUDE.md"
```

---

## Phase 12 — Final verification

### Task 12.1: Run the full test suite

- [ ] **Step 1: Full suite**

```bash
python -m pytest braintrace/ -v --tb=short 2>&1 | tail -80
```
Expected: all ~1250 existing tests + all new `_snn_algorithms` tests pass. Count the final "passed" number and confirm it matches (existing count) + (new count ≈ 45).

- [ ] **Step 2: If any existing test fails:**

Investigate. The likely culprit is the `_update_fn_bwd` signature change in Task 1.2 — re-verify the `args` tuple-unpacking change is backwards compatible. Do NOT mask failures.

- [ ] **Step 3: If all pass, tag the feature branch**

```bash
git log --oneline main..HEAD | head -30
```
Expected: clean sequence of ~30 well-described commits.

---

## Self-Review Checklist (fill in before opening a PR)

- [ ] Spec section 1 (Background) — no implementation required, context only.
- [ ] Spec section 2 (Goals) — all 5 classes ship (Phases 3-7); existing conventions preserved (Phase 1 regression tests).
- [ ] Spec section 3 (Non-goals) — MultiStepData explicitly rejected in `_update_etrace_data` for OTTT/OTPE; ETLP not implemented.
- [ ] Spec section 4 (Architecture) — subpackage created (Phase 2), base hook wired (Phase 1), update signature extended on OSTTP only (Phase 6).
- [ ] Spec section 5 (Per-class responsibility matrix) — each row implemented (EProp Phase 4, OSTL Phase 3, OTPE Phase 7, OTTT Phase 5, OSTTP Phase 6).
- [ ] Spec section 6 (Canonical knob) — each knob + value validation tested.
- [ ] Spec section 7 (Shared helpers) — all four in `_common.py` (Phase 2).
- [ ] Spec section 8 (Data flow) — standard forward unchanged, custom trace updates per algo, hook runs in `_update_fn_bwd`.
- [ ] Spec section 9 (Compiler invariant) — preserved; OTPE single-group check in `compile_graph` override (Task 7.1).
- [ ] Spec section 10 (Error handling) — each table row has a corresponding `assertRaises` test.
- [ ] Spec section 11 (Spike-reset diagnostic) — Phase 8.
- [ ] Spec section 12 (Numerical stability knobs) — `normalize_matrix_spectrum` passed through to D_RTRL base; `trace_clip_abs` wired in OTPE (Task 7.2); `kappa_filter_decay` in EProp.
- [ ] Spec section 13 (Testing matrix) — shape, forward, BPTT agreement (fixture test), batched/unbatched, reset, knob, saiunit, diagnostic-fires, error paths all covered.
- [ ] Spec section 14 (Example usage) — EProp + OSTTP doctests.
- [ ] Spec section 15 (Implementation phases) — plan phases 1-11 map 1:1.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-snn-online-etrace-algorithms.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.
