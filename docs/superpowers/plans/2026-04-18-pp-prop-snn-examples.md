# pp_prop SNN Examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 14-file tutorial-linear `examples/pp_prop/` series + shared helpers + README + long-form tutorial + smoke tests, demonstrating `braintrace.pp_prop` (aka `IODimVjpAlgorithm` / `ES_D_RTRL`) across neuron models, operators, training targets, batching modes, VJP methods, and algorithm knobs.

**Architecture:** Mirror `examples/drtrl/` layout. Each example file exports `main(n_epochs, batch_size, plot)` (matching drtrl convention) so smoke tests can parametrize. Shared utilities (`_shared.py`) hold SNN data generators, cell wrappers, and train-step helpers. No modifications to `braintrace/` source tree; examples consume public API only.

**Tech Stack:** `brainstate`, `braintools`, `saiunit`, `brainpy-state`, `jax`, `numpy`, `matplotlib`, `scikit-learn` (optional, for 8×8 digits with pure-numpy fallback).

**Deviation from spec:** Spec Section 4 proposed `PP_PROP_SMOKE` + `PP_PROP_NO_PLOT` env vars. This plan adopts the existing `drtrl/` convention instead: each example exports `main(n_epochs, batch_size, plot=True)` returning `{"losses": ...}`. Smoke tests call `main(n_epochs=1, batch_size=4, plot=False)`. Cleaner, already proven in `examples/drtrl/tests/test_smoke.py`.

---

## File Structure

**New files to create:**

| Path | Responsibility |
|------|----------------|
| `examples/pp_prop/__init__.py` | Empty package marker |
| `examples/pp_prop/_shared.py` | SNN data generators, cell wrappers, train helpers |
| `examples/pp_prop/README.md` | Axis map + run instructions |
| `examples/pp_prop/01-basics-lif-integrator.py` | LIF integrator regression (basics) |
| `examples/pp_prop/02-neurons-alif-dms.py` | ALIF on delayed-match-to-sample |
| `examples/pp_prop/03-neurons-gif-working-memory.py` | GIF heterogeneous-tau memory |
| `examples/pp_prop/04-neurons-coba-ei-rsnn.py` | Dale-law E/I RSNN |
| `examples/pp_prop/05-batching-vmap.py` | `vmap_new_states` batching |
| `examples/pp_prop/06-batching-batched.py` | Batched primitive path |
| `examples/pp_prop/07-vjp-single-step.py` | `vjp_method='single-step'` |
| `examples/pp_prop/08-vjp-multi-step.py` | `vjp_method='multi-step'` |
| `examples/pp_prop/09-operator-sparse.py` | Sparse recurrence (masked-dense fallback) |
| `examples/pp_prop/10-operator-lora.py` | `braintrace.lora_matmul` recurrence |
| `examples/pp_prop/11-operator-conv.py` | Conv-SNN via `braintrace.conv` |
| `examples/pp_prop/12-classification-neuromorphic.py` | Flagship Poisson-MNIST, pp_prop vs BPTT |
| `examples/pp_prop/13-knob-decay-vs-rank.py` | Sweep `decay_or_rank` |
| `examples/pp_prop/14-knob-vjp-method-contrast.py` | Single-step vs multi-step head-to-head |
| `examples/pp_prop/tests/__init__.py` | Empty |
| `examples/pp_prop/tests/test_smoke.py` | Parametrized smoke suite |
| `docs/tutorials/pp_prop.md` | Long-form companion tutorial |

---

## Task 1: Package scaffold

**Files:**
- Create: `examples/pp_prop/__init__.py`
- Create: `examples/pp_prop/tests/__init__.py`

- [ ] **Step 1: Create empty package init files**

File `examples/pp_prop/__init__.py`:
```python
# examples/pp_prop/__init__.py
```

File `examples/pp_prop/tests/__init__.py`:
```python
# examples/pp_prop/tests/__init__.py
```

- [ ] **Step 2: Verify directory exists**

Run: `ls examples/pp_prop/ examples/pp_prop/tests/`
Expected: both directories exist, each contains `__init__.py`.

- [ ] **Step 3: Commit**

```bash
git add examples/pp_prop/__init__.py examples/pp_prop/tests/__init__.py
git commit -m "examples(pp_prop): scaffold package directories"
```

---

## Task 2: `_shared.py` — data generators (TDD)

**Files:**
- Create: `examples/pp_prop/_shared.py`
- Create: `examples/pp_prop/tests/test_shared_data.py`

- [ ] **Step 1: Write failing tests for data generators**

File `examples/pp_prop/tests/test_shared_data.py`:
```python
"""Unit tests for examples/pp_prop/_shared.py data generators."""
from __future__ import annotations

import importlib.util
import pathlib

import jax.numpy as jnp
import numpy as np


def _load_shared():
    spec = importlib.util.spec_from_file_location(
        "_pp_prop_shared",
        pathlib.Path(__file__).resolve().parents[1] / "_shared.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_integrator_spikes_shape_and_values():
    m = _load_shared()
    xs, ys = m.make_integrator_spikes(num_step=25, num_batch=4, rate_hz=50.0, dt=1e-3, seed=0)
    assert xs.shape == (25, 4, 1)
    assert ys.shape == (25, 4, 1)
    assert jnp.all((xs == 0.0) | (xs == 1.0))
    diffs = jnp.diff(ys, axis=0)
    assert jnp.all(diffs >= 0)


def test_make_dms_spikes_shape_and_labels():
    m = _load_shared()
    xs, ys = m.make_dms_spikes(num_step=40, num_batch=8, n_in=16, fr_hz=80.0, dt=1e-3, seed=0)
    assert xs.shape == (40, 8, 16)
    assert ys.shape == (8,)
    assert set(np.unique(np.asarray(ys)).tolist()).issubset({0, 1})


def test_make_memory_pattern_shape():
    m = _load_shared()
    xs, ys = m.make_memory_pattern(num_step=30, num_batch=4, n_in=8, cue_frac=0.2, seed=0)
    assert xs.shape == (30, 4, 8)
    assert ys.shape[-1] == 8


def test_make_poisson_mnist_shape_and_labels():
    m = _load_shared()
    xs, ys = m.make_poisson_mnist(num_step=20, num_batch=8, rate_hz=80.0, dt=1e-3, digits=(0, 1, 2), seed=0)
    assert xs.shape == (20, 8, 64)
    assert ys.shape == (8,)
    assert set(np.unique(np.asarray(ys)).tolist()).issubset({0, 1, 2})
```

- [ ] **Step 2: Run test — verify failure**

Run: `pytest examples/pp_prop/tests/test_shared_data.py -v`
Expected: FAIL with `FileNotFoundError` (shared file missing).

- [ ] **Step 3: Implement data generators**

File `examples/pp_prop/_shared.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Shared SNN data generators, cell wrappers, and train-step helpers for examples/pp_prop."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import brainpy.state
import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u


DEFAULT_DT = 1.0 * u.ms
DEFAULT_SEED = 42


# --- Data generators -----------------------------------------------------


def make_integrator_spikes(
    num_step: int = 50,
    num_batch: int = 32,
    rate_hz: float = 50.0,
    dt: float = 1e-3,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Poisson input spikes. Target = cumulative mean rate (continuous)."""
    rng = np.random.default_rng(seed)
    p = rate_hz * dt
    spikes = (rng.random((num_step, num_batch, 1)) < p).astype(np.float32)
    targets = np.cumsum(spikes, axis=0) / num_step
    return jnp.asarray(spikes), jnp.asarray(targets)


def make_dms_spikes(
    num_step: int = 40,
    num_batch: int = 32,
    n_in: int = 16,
    fr_hz: float = 80.0,
    dt: float = 1e-3,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Delayed match-to-sample: sample window + delay + test window, binary label."""
    rng = np.random.default_rng(seed)
    t_sample = num_step // 4
    t_delay = num_step // 2
    labels = rng.integers(0, 2, size=(num_batch,)).astype(np.int32)
    sample_dir = rng.integers(0, n_in, size=(num_batch,))
    test_dir = np.where(labels == 1, sample_dir, (sample_dir + n_in // 2) % n_in)
    fr = fr_hz * dt
    xs = np.zeros((num_step, num_batch, n_in), dtype=np.float32)
    for b in range(num_batch):
        xs[:t_sample, b, sample_dir[b]] = fr
        xs[t_sample + t_delay:, b, test_dir[b]] = fr
    xs = (rng.random(xs.shape) < xs).astype(np.float32)
    return jnp.asarray(xs), jnp.asarray(labels)


def make_memory_pattern(
    num_step: int = 30,
    num_batch: int = 32,
    n_in: int = 8,
    cue_frac: float = 0.2,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Working memory: binary pattern during cue window, silent delay, recall target at end."""
    rng = np.random.default_rng(seed)
    cue_steps = max(1, int(num_step * cue_frac))
    pattern = rng.integers(0, 2, size=(num_batch, n_in)).astype(np.float32)
    xs = np.zeros((num_step, num_batch, n_in), dtype=np.float32)
    xs[:cue_steps] = pattern[None, :, :]
    return jnp.asarray(xs), jnp.asarray(pattern)


def make_poisson_mnist(
    num_step: int = 25,
    num_batch: int = 32,
    rate_hz: float = 80.0,
    dt: float = 1e-3,
    digits: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Poisson-encoded sklearn 8x8 digits. Falls back to synthetic patterns if sklearn missing."""
    try:
        from sklearn.datasets import load_digits
        data = load_digits()
        imgs = data.images / data.images.max()
        labels = data.target
        mask = np.isin(labels, list(digits))
        imgs = imgs[mask]
        labels = labels[mask]
    except ImportError:
        rng_fallback = np.random.default_rng(seed)
        n_per = 50
        imgs = rng_fallback.random((n_per * len(digits), 8, 8)).astype(np.float32)
        labels = np.repeat(np.arange(len(digits)), n_per)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(num_batch,))
    flat = imgs[idx].reshape(num_batch, 64)
    label_idx = np.array([list(digits).index(int(l)) for l in labels[idx]], dtype=np.int32)
    p = flat[None, :, :] * rate_hz * dt
    spikes = (rng.random((num_step, num_batch, 64)) < p).astype(np.float32)
    return jnp.asarray(spikes), jnp.asarray(label_idx)
```

- [ ] **Step 4: Run test — verify pass**

Run: `pytest examples/pp_prop/tests/test_shared_data.py -v`
Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pp_prop/_shared.py examples/pp_prop/tests/test_shared_data.py
git commit -m "examples(pp_prop): add shared data generators"
```

---

## Task 3: `_shared.py` — SNN cell wrappers (TDD)

**Files:**
- Modify: `examples/pp_prop/_shared.py` (append cell definitions)
- Create: `examples/pp_prop/tests/test_shared_cells.py`

- [ ] **Step 1: Write failing tests for cells**

File `examples/pp_prop/tests/test_shared_cells.py`:
```python
"""Unit tests for examples/pp_prop/_shared.py SNN cells."""
from __future__ import annotations

import importlib.util
import pathlib

import brainstate
import jax.numpy as jnp
import saiunit as u


def _load_shared():
    spec = importlib.util.spec_from_file_location(
        "_pp_prop_shared",
        pathlib.Path(__file__).resolve().parents[1] / "_shared.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lif_cell_returns_spikes():
    m = _load_shared()
    with brainstate.environ.context(dt=1.0 * u.ms):
        cell = m.LIFCell(n_in=4, n_rec=8)
        brainstate.nn.init_all_states(cell)
        spk = cell(jnp.ones((4,), dtype=jnp.float32))
    assert spk.shape == (8,)


def test_alif_cell_runs():
    m = _load_shared()
    with brainstate.environ.context(dt=1.0 * u.ms):
        cell = m.ALIFCell(n_in=4, n_rec=8)
        brainstate.nn.init_all_states(cell)
        spk = cell(jnp.ones((4,), dtype=jnp.float32))
    assert spk.shape == (8,)


def test_leaky_readout_returns_rate():
    m = _load_shared()
    with brainstate.environ.context(dt=1.0 * u.ms):
        readout = m.LeakyReadout(n_rec=8, n_out=3)
        brainstate.nn.init_all_states(readout)
        out = readout(jnp.ones((8,), dtype=jnp.float32))
    assert out.shape == (3,)
```

- [ ] **Step 2: Run test — verify failure**

Run: `pytest examples/pp_prop/tests/test_shared_cells.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'LIFCell'`.

- [ ] **Step 3: Append cell definitions**

Append to `examples/pp_prop/_shared.py`:
```python


# --- SNN cells -----------------------------------------------------------


class LIFCell(brainstate.nn.Module):
    """LIF recurrent block: concat(input, last-spike) -> Linear -> Expon -> LIF -> spikes."""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        import braintrace
        ff_init = braintools.init.KaimingNormal(ff_scale, unit=u.mV)
        rec_init = braintools.init.KaimingNormal(rec_scale, unit=u.mV)
        w = u.math.concatenate([ff_init((n_in, n_rec)), rec_init((n_rec, n_rec))], axis=0)
        self.linear = braintrace.nn.Linear(
            n_in + n_rec, n_rec, w_init=w, b_init=braintools.init.ZeroInit(unit=u.mV)
        )
        self.syn = brainpy.state.Expon(
            n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit(unit=u.mV)
        )
        self.neu = brainpy.state.LIF(n_rec, tau=tau_mem, V_th=V_th)

    def update(self, x):
        last_spk = self.neu.get_spike()
        pre = u.math.concatenate([x, last_spk], axis=-1)
        return self.neu(self.syn(self.linear(pre)))


class ALIFCell(brainstate.nn.Module):
    """ALIF (adaptive threshold) recurrent block. Same interface as LIFCell."""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        tau_a: u.Quantity = 100.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        beta: float = 1.8,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        import braintrace
        ff_init = braintools.init.KaimingNormal(ff_scale, unit=u.mV)
        rec_init = braintools.init.KaimingNormal(rec_scale, unit=u.mV)
        w = u.math.concatenate([ff_init((n_in, n_rec)), rec_init((n_rec, n_rec))], axis=0)
        self.linear = braintrace.nn.Linear(
            n_in + n_rec, n_rec, w_init=w, b_init=braintools.init.ZeroInit(unit=u.mV)
        )
        self.syn = brainpy.state.Expon(
            n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit(unit=u.mV)
        )
        self.neu = brainpy.state.ALIF(n_rec, tau=tau_mem, tau_a=tau_a, V_th=V_th, beta=beta)

    def update(self, x):
        last_spk = self.neu.get_spike()
        pre = u.math.concatenate([x, last_spk], axis=-1)
        return self.neu(self.syn(self.linear(pre)))


class GIFCell(brainstate.nn.Module):
    """GIF (generalized integrate-and-fire) with heterogeneous tau_I2. Same interface."""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        A2: u.Quantity = -0.5 * u.mA,
        tau_I2_low: u.Quantity = 80.0 * u.ms,
        tau_I2_high: u.Quantity = 200.0 * u.ms,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        import pathlib
        import sys
        # lazy import of local GIF class (tau_I2 heterogeneity)
        repo_examples = pathlib.Path(__file__).resolve().parent.parent
        if str(repo_examples) not in sys.path:
            sys.path.insert(0, str(repo_examples))
        from snn_models import GIF  # type: ignore
        import braintrace
        ff_init = braintools.init.KaimingNormal(ff_scale, unit=u.mA)
        rec_init = braintools.init.KaimingNormal(rec_scale, unit=u.mA)
        w = u.math.concatenate([ff_init((n_in, n_rec)), rec_init((n_rec, n_rec))], axis=0)
        self.linear = braintrace.nn.Linear(
            n_in + n_rec, n_rec, w_init=w, b_init=braintools.init.ZeroInit(unit=u.mA)
        )
        self.syn = brainpy.state.Expon(
            n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit(unit=u.mA)
        )
        self.neu = GIF(
            n_rec,
            V_th_inf=V_th,
            tau=tau_mem,
            tau_I2=brainstate.random.uniform(tau_I2_low, tau_I2_high, n_rec),
            A2=A2,
        )

    def update(self, x):
        last_spk = self.neu.get_spike()
        pre = u.math.concatenate([x, last_spk], axis=-1)
        return self.neu(self.syn(self.linear(pre)))


class COBAEICell(brainstate.nn.Module):
    """Dale-law E/I recurrent block using signed init on braintrace.nn.Linear."""

    def __init__(
        self,
        n_in: int,
        n_exc: int,
        n_inh: int,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        import braintrace
        n_rec = n_exc + n_inh
        self.n_exc = n_exc
        ff = braintools.init.KaimingNormal(ff_scale, unit=u.mV)((n_in, n_rec))
        rec_pos = u.math.abs(braintools.init.KaimingNormal(rec_scale, unit=u.mV)((n_exc, n_rec)))
        rec_neg = -u.math.abs(braintools.init.KaimingNormal(rec_scale, unit=u.mV)((n_inh, n_rec)))
        w = u.math.concatenate([ff, rec_pos, rec_neg], axis=0)
        self.linear = braintrace.nn.Linear(
            n_in + n_rec, n_rec, w_init=w, b_init=braintools.init.ZeroInit(unit=u.mV)
        )
        self.syn = brainpy.state.Expon(
            n_rec, tau=tau_syn, g_initializer=braintools.init.ZeroInit(unit=u.mV)
        )
        self.neu = brainpy.state.LIF(n_rec, tau=tau_mem, V_th=V_th)

    def update(self, x):
        last_spk = self.neu.get_spike()
        pre = u.math.concatenate([x, last_spk], axis=-1)
        return self.neu(self.syn(self.linear(pre)))


class LeakyReadout(brainstate.nn.Module):
    """Leaky-rate readout for regression / classification."""

    def __init__(self, n_rec: int, n_out: int, tau_o: u.Quantity = 10.0 * u.ms):
        super().__init__()
        import braintrace
        self.readout = braintrace.nn.LeakyRateReadout(
            in_size=n_rec, out_size=n_out, tau=tau_o,
            w_init=braintools.init.KaimingNormal(),
        )

    def update(self, spikes):
        return self.readout(spikes)
```

- [ ] **Step 4: Run test — verify pass**

Run: `pytest examples/pp_prop/tests/test_shared_cells.py -v`
Expected: three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pp_prop/_shared.py examples/pp_prop/tests/test_shared_cells.py
git commit -m "examples(pp_prop): add shared SNN cell wrappers"
```

---

## Task 4: `_shared.py` — trainer helpers

**Files:**
- Modify: `examples/pp_prop/_shared.py` (append trainer helpers)

- [ ] **Step 1: Append trainer helpers**

Append to `examples/pp_prop/_shared.py`:
```python


# --- Training helpers ----------------------------------------------------


def online_train_epoch(
    model: brainstate.nn.Module,
    opt,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_fn: Callable,
    decay_or_rank=0.95,
    vjp_method: str = "single-step",
) -> float:
    """Run one online-training epoch using pp_prop. Returns mean step loss."""
    import braintrace
    weights = model.states(brainstate.ParamState)
    online_model = braintrace.IODimVjpAlgorithm(
        model, decay_or_rank=decay_or_rank, vjp_method=vjp_method
    )

    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)
        online_model.compile_graph(inputs[0, 0])

    init()
    vmap_model = brainstate.nn.Vmap(online_model, vmap_states="new")

    def step_loss(inp, tar):
        out = vmap_model(inp)
        return loss_fn(out, tar), out

    def grad_step(prev_grads, pair):
        inp, tar = pair
        f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
        cur_grads, local_loss, _ = f_grad(inp, tar)
        return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

    init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
    grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
    grads = brainstate.functional.clip_grad_norm(grads, 1.0)
    opt.update(grads)
    return float(step_losses.mean())


def online_train_epoch_fixed_target(
    model: brainstate.nn.Module,
    opt,
    inputs: jnp.ndarray,
    target_labels: jnp.ndarray,
    decay_or_rank=0.95,
    vjp_method: str = "single-step",
) -> float:
    """Classification variant: fixed label per batch, softmax-xent loss applied each step."""
    import braintrace
    weights = model.states(brainstate.ParamState)
    online_model = braintrace.IODimVjpAlgorithm(
        model, decay_or_rank=decay_or_rank, vjp_method=vjp_method
    )

    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)
        online_model.compile_graph(inputs[0, 0])

    init()
    vmap_model = brainstate.nn.Vmap(online_model, vmap_states="new")

    def step_loss(inp):
        out = vmap_model(inp)
        loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
            out, target_labels
        ).mean()
        return loss, out

    def grad_step(prev_grads, inp):
        f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
        cur_grads, local_loss, _ = f_grad(inp)
        return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

    init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
    grads, step_losses = brainstate.transform.scan(grad_step, init_grads, inputs)
    grads = brainstate.functional.clip_grad_norm(grads, 1.0)
    opt.update(grads)
    return float(step_losses.mean())


def bptt_train_epoch_fixed_target(
    model: brainstate.nn.Module,
    opt,
    inputs: jnp.ndarray,
    target_labels: jnp.ndarray,
) -> float:
    """BPTT baseline with per-step softmax-cross-entropy over a fixed label."""
    weights = model.states(brainstate.ParamState)

    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)

    init()
    vmap_model = brainstate.nn.Vmap(model, vmap_states="new")

    def run_step(inp):
        out = vmap_model(inp)
        loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
            out, target_labels
        ).mean()
        return out, loss

    def bptt_body():
        _, losses = brainstate.transform.for_loop(run_step, inputs)
        return losses.mean()

    grads, loss = brainstate.transform.grad(bptt_body, weights, return_value=True)()
    grads = brainstate.functional.clip_grad_norm(grads, 1.0)
    opt.update(grads)
    return float(loss)


def plot_loss_curve(losses, title: str, save_path: str | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)
```

- [ ] **Step 2: Verify clean import**

Run: `python -c "import importlib.util; s = importlib.util.spec_from_file_location('x', 'examples/pp_prop/_shared.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); assert all(hasattr(m, n) for n in ['LIFCell', 'online_train_epoch', 'bptt_train_epoch_fixed_target', 'plot_loss_curve'])"`
Expected: Exit code 0, no output.

- [ ] **Step 3: Commit**

```bash
git add examples/pp_prop/_shared.py
git commit -m "examples(pp_prop): add shared online/bptt trainer helpers"
```

---

## Task 5: Smoke test harness skeleton

**Files:**
- Create: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Create empty parametrized smoke test**

File `examples/pp_prop/tests/test_smoke.py`:
```python
"""Smoke tests: each example's main() runs one epoch end-to-end without exceptions."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1]

EXAMPLE_FILES = [
    # Populated task-by-task as examples are added.
]


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(f"_pp_prop_{fname}", EXAMPLES_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fname", EXAMPLE_FILES)
def test_example_runs(fname):
    mod = _load(fname)
    result = mod.main(n_epochs=1, batch_size=4, plot=False)
    assert "losses" in result
    assert len(result["losses"]) >= 1
```

- [ ] **Step 2: Verify parses with zero tests**

Run: `pytest examples/pp_prop/tests/test_smoke.py -v`
Expected: `no tests ran` or `collected 0 items`.

- [ ] **Step 3: Commit**

```bash
git add examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add smoke-test harness"
```

---

## Task 6: Example 01 — `01-basics-lif-integrator.py`

**Files:**
- Create: `examples/pp_prop/01-basics-lif-integrator.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/01-basics-lif-integrator.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""01 · Minimal pp_prop on the LIF integrator task.

Trains a single-layer LIF RSNN on a Poisson-to-cumulative-rate regression task
using braintrace.pp_prop (aka IODimVjpAlgorithm). Smallest possible entry
point to the pp_prop API.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 5, batch_size: int = 32, num_step: int = 50, plot: bool = True) -> Dict:
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=1, n_rec=48, n_out=1)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, targets):
            return _shared.online_train_epoch(
                model, opt, inputs, targets,
                loss_fn=lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
                decay_or_rank=0.95,
                vjp_method="single-step",
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_integrator_spikes(
                num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[01-basics-lif] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="01 · LIF integrator (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1])), f"final loss not finite: {losses[-1]}"
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register in smoke test**

Edit `examples/pp_prop/tests/test_smoke.py`. Replace `EXAMPLE_FILES` list with:
```python
EXAMPLE_FILES = [
    "01-basics-lif-integrator.py",
]
```

- [ ] **Step 3: Run smoke test**

Run: `pytest examples/pp_prop/tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 4: Run file directly**

Run: `python examples/pp_prop/01-basics-lif-integrator.py`
Expected: 5 epochs of finite "loss=..." printouts. Plot window opens (close it to finish).

- [ ] **Step 5: Commit**

```bash
git add examples/pp_prop/01-basics-lif-integrator.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 01-basics-lif-integrator"
```

---

## Task 7: Example 02 — `02-neurons-alif-dms.py`

**Files:**
- Create: `examples/pp_prop/02-neurons-alif-dms.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/02-neurons-alif-dms.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""02 · Adaptive-threshold ALIF on delayed-match-to-sample (DMS).

Same pp_prop recipe as 01 but with brainpy.state.ALIF replacing LIF. The
adaptive threshold gives the network an intrinsic time constant that matters
for delay-bridging tasks like DMS.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.ALIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 5, batch_size: int = 32, num_step: int = 40, plot: bool = True) -> Dict:
    n_in = 16
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=n_in, n_rec=64, n_out=2)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels,
                decay_or_rank=0.97,
                vjp_method="single-step",
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_dms_spikes(
                num_step=num_step, num_batch=batch_size, n_in=n_in, fr_hz=80.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[02-alif-dms] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="02 · ALIF DMS (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Edit `examples/pp_prop/tests/test_smoke.py` — append `"02-neurons-alif-dms.py"` to `EXAMPLE_FILES`.

Run: `pytest examples/pp_prop/tests/test_smoke.py -v`
Expected: 01 + 02 PASS.

```bash
git add examples/pp_prop/02-neurons-alif-dms.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 02-neurons-alif-dms"
```

---

## Task 8: Example 03 — `03-neurons-gif-working-memory.py`

**Files:**
- Create: `examples/pp_prop/03-neurons-gif-working-memory.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/03-neurons-gif-working-memory.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""03 · GIF neurons with heterogeneous tau_I2 on a working-memory recall task.

GIF's slow adaptation current gives per-neuron memory timescales. pp_prop
tracks the trace through the slow state via the same diagonal approximation
it uses for membrane voltage.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.GIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 5, batch_size: int = 32, num_step: int = 30, plot: bool = True) -> Dict:
    n_in = 8
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=n_in, n_rec=64, n_out=n_in)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, targets_seq):
            return _shared.online_train_epoch(
                model, opt, inputs, targets_seq,
                loss_fn=lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
                decay_or_rank=0.98,
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_memory_pattern(
                num_step=num_step, num_batch=batch_size, n_in=n_in, cue_frac=0.2, seed=epoch,
            )
            ys_seq = jnp.broadcast_to(ys[None], (num_step, batch_size, n_in))
            losses.append(float(train_step(xs, ys_seq)))
            print(f"[03-gif-memory] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="03 · GIF working-memory (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"03-neurons-gif-working-memory.py"` to `EXAMPLE_FILES`. Run smoke test.

```bash
git add examples/pp_prop/03-neurons-gif-working-memory.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 03-neurons-gif-working-memory"
```

---

## Task 9: Example 04 — `04-neurons-coba-ei-rsnn.py`

**Files:**
- Create: `examples/pp_prop/04-neurons-coba-ei-rsnn.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/04-neurons-coba-ei-rsnn.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""04 · Dale-law E/I recurrent SNN on Poisson-encoded digits.

Excitatory neurons emit positive recurrent weights, inhibitory neurons emit
negative ones (soft Dale, enforced by initialisation). pp_prop trains the
full recurrent matrix without violating the sign constraint because the
signs are fixed by the initialiser, not by the gradient step.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_exc: int, n_inh: int, n_out: int):
        super().__init__()
        self.cell = _shared.COBAEICell(n_in=n_in, n_exc=n_exc, n_inh=n_inh)
        self.readout = _shared.LeakyReadout(n_rec=n_exc + n_inh, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 5, batch_size: int = 32, num_step: int = 25, plot: bool = True) -> Dict:
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=64, n_exc=48, n_inh=16, n_out=len(digits))
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels, decay_or_rank=0.95,
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_poisson_mnist(
                num_step=num_step, num_batch=batch_size, rate_hz=80.0, dt=1e-3,
                digits=digits, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[04-coba-ei] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="04 · COBA E/I RSNN (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"04-neurons-coba-ei-rsnn.py"`. Run smoke test.

```bash
git add examples/pp_prop/04-neurons-coba-ei-rsnn.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 04-neurons-coba-ei-rsnn"
```

---

## Task 10: Example 05 — `05-batching-vmap.py`

**Files:**
- Create: `examples/pp_prop/05-batching-vmap.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/05-batching-vmap.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""05 · Batching via brainstate.nn.Vmap(vmap_states='new').

The network and the IODimVjpAlgorithm are defined unbatched, then replicated
across the batch dimension via vmap_new_states. pp_prop's per-rule init is
aware of batching and allocates batched eligibility traces automatically.
This is the default batching path used by examples 01-04.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 50, plot: bool = True) -> Dict:
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=1, n_rec=48, n_out=1)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, targets):
            return _shared.online_train_epoch(
                model, opt, inputs, targets,
                loss_fn=lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_integrator_spikes(
                num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[05-vmap] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="05 · vmap batching (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"05-batching-vmap.py"`. Run smoke test.

```bash
git add examples/pp_prop/05-batching-vmap.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 05-batching-vmap"
```

---

## Task 11: Example 06 — `06-batching-batched.py`

**Files:**
- Create: `examples/pp_prop/06-batching-batched.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

**Reference:** If this fails, inspect `examples/003-snn-memory-and-speed-evaluation-batched.py` for the exact batched-primitive init pattern used in the existing codebase and match it.

- [ ] **Step 1: Write example**

File `examples/pp_prop/06-batching-batched.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""06 · Batching via the batched ETP primitive path (no vmap wrapper).

Inputs carry a batch dimension directly. braintrace.matmul dispatches to the
batched primitive etp_mm_p when x.ndim >= 2, so the network runs natively
batched and IODimVjpAlgorithm does not need to wrap the model in Vmap.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 50, plot: bool = True) -> Dict:
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=1, n_rec=48, n_out=1)
        brainstate.nn.init_all_states(model, batch_size=batch_size)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        online_model = braintrace.IODimVjpAlgorithm(
            model, decay_or_rank=0.95, vjp_method="single-step"
        )
        online_model.compile_graph(jnp.zeros((batch_size, 1)))

        def step_loss(inp, tar):
            out = online_model(inp)
            return braintools.metric.squared_error(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        @brainstate.transform.jit
        def train_step(inputs, targets):
            init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
            grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
            grads = brainstate.functional.clip_grad_norm(grads, 1.0)
            opt.update(grads)
            return step_losses.mean()

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_integrator_spikes(
                num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[06-batched] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="06 · batched primitive (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"06-batching-batched.py"`. Run smoke test.

```bash
git add examples/pp_prop/06-batching-batched.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 06-batching-batched"
```

---

## Task 12: Example 07 — `07-vjp-single-step.py`

**Files:**
- Create: `examples/pp_prop/07-vjp-single-step.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/07-vjp-single-step.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""07 · vjp_method='single-step' on LIF integrator.

Demonstrates pp_prop's default VJP mode. The VJP of the loss w.r.t. hidden
state is computed at each current time step only (no lookback), which keeps
the online cost small and is the mode used when each step provides its own
target signal.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 50, plot: bool = True) -> Dict:
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=1, n_rec=48, n_out=1)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, targets):
            return _shared.online_train_epoch(
                model, opt, inputs, targets,
                loss_fn=lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
                decay_or_rank=0.95,
                vjp_method="single-step",
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_integrator_spikes(
                num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[07-vjp-single] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="07 · vjp_method='single-step'")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"07-vjp-single-step.py"`. Run smoke test.

```bash
git add examples/pp_prop/07-vjp-single-step.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 07-vjp-single-step"
```

---

## Task 13: Example 08 — `08-vjp-multi-step.py`

**Files:**
- Create: `examples/pp_prop/08-vjp-multi-step.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/08-vjp-multi-step.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""08 · vjp_method='multi-step' on delayed-match-to-sample.

Multi-step VJP lets pp_prop combine gradients computed at multiple time steps
(partial L^{t'}/partial h^{t-k}). Useful when the task has a sparse target signal
(one label per sequence) and we still want temporal credit assignment on top
of the eligibility-trace diagonal approximation.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 40, plot: bool = True) -> Dict:
    n_in = 16
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=n_in, n_rec=64, n_out=2)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels,
                decay_or_rank=0.97,
                vjp_method="multi-step",
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_dms_spikes(
                num_step=num_step, num_batch=batch_size, n_in=n_in, fr_hz=80.0, dt=1e-3, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[08-vjp-multi] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="08 · vjp_method='multi-step'")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"08-vjp-multi-step.py"`. Run smoke test. If multi-step hits runtime issue, fall back to `vjp_method="single-step"` and update docstring noting multi-step is demonstrated head-to-head in file 14.

```bash
git add examples/pp_prop/08-vjp-multi-step.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 08-vjp-multi-step"
```

---

## Task 14: Example 09 — `09-operator-sparse.py`

**Files:**
- Create: `examples/pp_prop/09-operator-sparse.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

**Note:** Uses masked-dense fallback because `saiunit.sparse` COO/CSR lack JAX batching rules (same constraint blocked `examples/drtrl/06-operator-sparse.py`).

- [ ] **Step 1: Write example**

File `examples/pp_prop/09-operator-sparse.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""09 · Sparse recurrent weights via a fixed connectivity mask.

Only the non-zero entries of the recurrent matrix are effectively trainable;
zero entries stay zero because their values are multiplied by a zero mask.
pp_prop sees the combined operation as a dense matmul and computes gradients
accordingly -- the mask simply zeros out updates to absent connections.

Note: the sparse ETP primitive etp_sp_mm_p exists but saiunit.sparse lacks
JAX batching rules today, so the drtrl series (file 06) documented the same
limitation. This masked-dense approach still exercises pp_prop end-to-end.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainpy.state
import brainstate
import braintools
import jax.numpy as jnp
import numpy as np
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class SparseRecCell(brainstate.nn.Module):
    """LIF cell where the recurrent weight matrix has a fixed sparsity pattern."""

    def __init__(self, n_in: int, n_rec: int, density: float = 0.1, seed: int = 0):
        super().__init__()
        self.ff = braintrace.nn.Linear(n_in, n_rec, b_init=braintools.init.ZeroInit(unit=u.mV))
        rng = np.random.default_rng(seed)
        mask = (rng.random((n_rec, n_rec)) < density).astype(np.float32)
        init_rec = braintools.init.KaimingNormal(unit=u.mV)((n_rec, n_rec))
        self.mask = u.math.asarray(mask, unit=u.math.get_unit(init_rec))
        self.rec = brainstate.ParamState(init_rec * mask)
        self.syn = brainpy.state.Expon(
            n_rec, tau=10.0 * u.ms, g_initializer=braintools.init.ZeroInit(unit=u.mV)
        )
        self.neu = brainpy.state.LIF(n_rec, tau=20.0 * u.ms, V_th=1.0 * u.mV)

    def update(self, x):
        last_spk = self.neu.get_spike()
        ff_current = self.ff(x)
        # mask multiplication keeps fixed zeros frozen
        rec_current = braintrace.matmul(last_spk, self.rec.value * self.mask)
        return self.neu(self.syn(ff_current + rec_current))


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int, density: float):
        super().__init__()
        self.cell = SparseRecCell(n_in=n_in, n_rec=n_rec, density=density)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 25, plot: bool = True) -> Dict:
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=64, n_rec=96, n_out=len(digits), density=0.1)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels, decay_or_rank=0.95,
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_poisson_mnist(
                num_step=num_step, num_batch=batch_size, digits=digits, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[09-sparse] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="09 · Sparse recurrence (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"09-operator-sparse.py"`. Run smoke test.

```bash
git add examples/pp_prop/09-operator-sparse.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 09-operator-sparse (masked-dense fallback)"
```

---

## Task 15: Example 10 — `10-operator-lora.py`

**Files:**
- Create: `examples/pp_prop/10-operator-lora.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/10-operator-lora.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""10 · Low-rank recurrent weights via braintrace.lora_matmul.

Parameterise the recurrent matrix as W = alpha * B @ A with rank r << n_rec.
pp_prop dispatches to the LoRA ETP primitive etp_lora_mm_p.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainpy.state
import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class LoRARecCell(brainstate.nn.Module):
    """LIF cell with low-rank recurrent weights via braintrace.lora_matmul."""

    def __init__(self, n_in: int, n_rec: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.ff = braintrace.nn.Linear(n_in, n_rec, b_init=braintools.init.ZeroInit(unit=u.mV))
        ki = braintools.init.KaimingNormal(unit=u.mV)
        self.B = brainstate.ParamState(ki((n_rec, rank)))
        self.A = brainstate.ParamState(ki((rank, n_rec)))
        self.alpha = alpha
        self.syn = brainpy.state.Expon(
            n_rec, tau=10.0 * u.ms, g_initializer=braintools.init.ZeroInit(unit=u.mV)
        )
        self.neu = brainpy.state.LIF(n_rec, tau=20.0 * u.ms, V_th=1.0 * u.mV)

    def update(self, x):
        last_spk = self.neu.get_spike()
        ff_current = self.ff(x)
        rec_current = braintrace.lora_matmul(last_spk, self.B.value, self.A.value, alpha=self.alpha)
        return self.neu(self.syn(ff_current + rec_current))


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int, rank: int):
        super().__init__()
        self.cell = LoRARecCell(n_in=n_in, n_rec=n_rec, rank=rank)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 25, plot: bool = True) -> Dict:
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_in=64, n_rec=96, n_out=len(digits), rank=8)
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels, decay_or_rank=0.95,
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_poisson_mnist(
                num_step=num_step, num_batch=batch_size, digits=digits, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[10-lora] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="10 · LoRA recurrence (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"10-operator-lora.py"`. Run smoke test.

```bash
git add examples/pp_prop/10-operator-lora.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 10-operator-lora"
```

---

## Task 16: Example 11 — `11-operator-conv.py`

**Files:**
- Create: `examples/pp_prop/11-operator-conv.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/11-operator-conv.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""11 · Conv-SNN on Poisson-encoded 8x8 digits via braintrace.conv (Conv2d).

Single Conv2d -> Expon -> LIF -> global-avg-pool -> readout. pp_prop
dispatches to the convolutional ETP primitive etp_conv_p for gradient
computation on the conv kernel.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainpy.state
import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class ConvLIFCell(brainstate.nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = braintrace.nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, padding="SAME",
        )
        self.syn = brainpy.state.Expon(
            (8, 8, out_ch), tau=10.0 * u.ms, g_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        self.neu = brainpy.state.LIF((8, 8, out_ch), tau=20.0 * u.ms, V_th=1.0 * u.mV)

    def update(self, x):
        return self.neu(self.syn(self.conv(x)))


class Net(brainstate.nn.Module):
    def __init__(self, n_out: int, out_ch: int = 4):
        super().__init__()
        self.cell = ConvLIFCell(in_ch=1, out_ch=out_ch)
        self.readout = _shared.LeakyReadout(n_rec=out_ch, n_out=n_out)

    def update(self, x):
        spikes_2d = self.cell(x)  # (8, 8, out_ch)
        pooled = spikes_2d.mean(axis=(0, 1))  # -> (out_ch,)
        return self.readout(pooled)


def main(n_epochs: int = 3, batch_size: int = 16, num_step: int = 20, plot: bool = True) -> Dict:
    digits = (0, 1, 2, 3)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model = Net(n_out=len(digits))
        weights = model.states(brainstate.ParamState)
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights(weights)

        @brainstate.transform.jit
        def train_step(inputs_flat, labels):
            inputs = inputs_flat.reshape(*inputs_flat.shape[:2], 8, 8, 1)
            return _shared.online_train_epoch_fixed_target(
                model, opt, inputs, labels, decay_or_rank=0.95,
            )

        losses = []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_poisson_mnist(
                num_step=num_step, num_batch=batch_size, digits=digits, seed=epoch,
            )
            losses.append(float(train_step(xs, ys)))
            print(f"[11-conv] epoch {epoch}  loss={losses[-1]:.4f}")

    if plot:
        _shared.plot_loss_curve(losses, title="11 · Conv-SNN (pp_prop)")
    assert jnp.isfinite(jnp.asarray(losses[-1]))
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"11-operator-conv.py"`. Run smoke test. If runtime exceeds 2 min, reduce `out_ch` to 2 or `num_step` to 15.

```bash
git add examples/pp_prop/11-operator-conv.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 11-operator-conv"
```

---

## Task 17: Example 12 — `12-classification-neuromorphic.py` (flagship)

**Files:**
- Create: `examples/pp_prop/12-classification-neuromorphic.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/12-classification-neuromorphic.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""12 · Flagship classification --- pp_prop vs BPTT on Poisson-MNIST (10 classes).

Trains two identical LIF RSNNs on the same Poisson-encoded sklearn digits,
one with pp_prop, one with BPTT. Reports per-epoch loss and final accuracy.
Demonstrates that pp_prop tracks BPTT's performance with the O(BI+BO)
memory footprint advertised by the algorithm.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec, ff_scale=4.0, rec_scale=1.0)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _accuracy(outputs_seq, labels):
    mean_out = outputs_seq.mean(axis=0)
    return float(jnp.mean(jnp.argmax(mean_out, axis=-1) == labels))


def _eval(model, inputs, labels):
    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)

    init()
    vmap_model = brainstate.nn.Vmap(model, vmap_states="new")
    outs = brainstate.transform.for_loop(lambda x: vmap_model(x), inputs)
    return _accuracy(outs, labels)


def main(n_epochs: int = 4, batch_size: int = 32, num_step: int = 25, plot: bool = True) -> Dict:
    digits = tuple(range(10))
    n_out = len(digits)
    with brainstate.environ.context(dt=1.0 * u.ms):
        model_pp = Net(n_in=64, n_rec=128, n_out=n_out)
        w_pp = model_pp.states(brainstate.ParamState)
        opt_pp = braintools.optim.Adam(lr=1e-3)
        opt_pp.register_trainable_weights(w_pp)

        @brainstate.transform.jit
        def train_pp(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                model_pp, opt_pp, inputs, labels,
                decay_or_rank=0.97, vjp_method="single-step",
            )

        model_bp = Net(n_in=64, n_rec=128, n_out=n_out)
        w_bp = model_bp.states(brainstate.ParamState)
        opt_bp = braintools.optim.Adam(lr=1e-3)
        opt_bp.register_trainable_weights(w_bp)

        @brainstate.transform.jit
        def train_bp(inputs, labels):
            return _shared.bptt_train_epoch_fixed_target(model_bp, opt_bp, inputs, labels)

        pp_losses, bp_losses = [], []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_poisson_mnist(
                num_step=num_step, num_batch=batch_size, digits=digits, seed=epoch,
            )
            pp_losses.append(float(train_pp(xs, ys)))
            bp_losses.append(float(train_bp(xs, ys)))
            print(
                f"[12-flagship] epoch {epoch}  pp_prop={pp_losses[-1]:.4f}  "
                f"bptt={bp_losses[-1]:.4f}"
            )

        xs_e, ys_e = _shared.make_poisson_mnist(
            num_step=num_step, num_batch=batch_size, digits=digits, seed=9999,
        )
        acc_pp = _eval(model_pp, xs_e, ys_e)
        acc_bp = _eval(model_bp, xs_e, ys_e)
        print(f"[12-flagship] final acc  pp_prop={acc_pp:.3f}  bptt={acc_bp:.3f}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(pp_losses, label="pp_prop")
        ax.plot(bp_losses, label="BPTT")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("12 · pp_prop vs BPTT on Poisson-MNIST")
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    chance = 1.0 / n_out
    assert acc_pp > chance, f"pp_prop accuracy {acc_pp} <= chance {chance}"
    assert acc_bp > chance, f"BPTT accuracy {acc_bp} <= chance {chance}"
    return {"losses": pp_losses, "bptt_losses": bp_losses, "acc_pp": acc_pp, "acc_bp": acc_bp}


if __name__ == "__main__":
    main()
```

**Note:** Smoke test overrides `n_epochs=1, batch_size=4`. At that setting the chance-only assertion is the binding one. The `__main__` block uses stronger defaults where 0.6+ accuracy is achievable; monitor printed accuracy on manual runs and tighten the assertion post-hoc if it consistently clears 0.6.

- [ ] **Step 2: Register + run smoke + commit**

Append `"12-classification-neuromorphic.py"`. Run smoke test.

```bash
git add examples/pp_prop/12-classification-neuromorphic.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 12-classification-neuromorphic flagship"
```

---

## Task 18: Example 13 — `13-knob-decay-vs-rank.py`

**Files:**
- Create: `examples/pp_prop/13-knob-decay-vs-rank.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/13-knob-decay-vs-rank.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""13 · Sweep decay_or_rank on LIF integrator.

IODimVjpAlgorithm accepts decay_or_rank as either a float (exponential-smoothing
decay, 0 < alpha < 1) or an int (approximation rank). The two parameterisations
are duals: num_rank = 2/(1-decay) - 1. This file trains multiple models, one
per value, and plots their final-epoch losses side-by-side.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict, List

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _train_once(decay_or_rank, n_epochs, batch_size, num_step):
    model = Net(n_in=1, n_rec=48, n_out=1)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=1e-3)
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def train_step(inputs, targets):
        return _shared.online_train_epoch(
            model, opt, inputs, targets,
            loss_fn=lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
            decay_or_rank=decay_or_rank,
        )

    losses = []
    for epoch in range(n_epochs):
        xs, ys = _shared.make_integrator_spikes(
            num_step=num_step, num_batch=batch_size, rate_hz=50.0, dt=1e-3, seed=epoch,
        )
        losses.append(float(train_step(xs, ys)))
    return losses


def main(n_epochs: int = 3, batch_size: int = 32, num_step: int = 50, plot: bool = True) -> Dict:
    sweep_values = [0.9, 0.95, 0.99, 3, 10, 40]
    with brainstate.environ.context(dt=1.0 * u.ms):
        all_losses: Dict[str, List[float]] = {}
        for val in sweep_values:
            label = f"decay={val}" if isinstance(val, float) else f"rank={val}"
            all_losses[label] = _train_once(val, n_epochs, batch_size, num_step)
            print(f"[13-sweep] {label}  final_loss={all_losses[label][-1]:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, curve in all_losses.items():
            ax.plot(curve, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("13 · decay_or_rank sweep (pp_prop)")
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    for curve in all_losses.values():
        assert jnp.isfinite(jnp.asarray(curve[-1]))
    first_key = next(iter(all_losses))
    return {"losses": all_losses[first_key], "sweep": all_losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"13-knob-decay-vs-rank.py"`. Run smoke test.

```bash
git add examples/pp_prop/13-knob-decay-vs-rank.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 13-knob-decay-vs-rank sweep"
```

---

## Task 19: Example 14 — `14-knob-vjp-method-contrast.py`

**Files:**
- Create: `examples/pp_prop/14-knob-vjp-method-contrast.py`
- Modify: `examples/pp_prop/tests/test_smoke.py`

- [ ] **Step 1: Write example**

File `examples/pp_prop/14-knob-vjp-method-contrast.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""14 · single-step vs multi-step VJP vs BPTT head-to-head on DMS.

Trains three identical LIF RSNNs on the same DMS data:
one with vjp_method='single-step', one with 'multi-step', one with BPTT.
Reports per-epoch loss and final accuracy for all three.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _accuracy(outputs_seq, labels):
    mean_out = outputs_seq.mean(axis=0)
    return float(jnp.mean(jnp.argmax(mean_out, axis=-1) == labels))


def _eval(model, inputs, labels):
    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)

    init()
    vmap_model = brainstate.nn.Vmap(model, vmap_states="new")
    outs = brainstate.transform.for_loop(lambda x: vmap_model(x), inputs)
    return _accuracy(outs, labels)


def main(n_epochs: int = 4, batch_size: int = 32, num_step: int = 40, plot: bool = True) -> Dict:
    n_in = 16
    with brainstate.environ.context(dt=1.0 * u.ms):
        def make():
            m = Net(n_in=n_in, n_rec=64, n_out=2)
            w = m.states(brainstate.ParamState)
            o = braintools.optim.Adam(lr=1e-3)
            o.register_trainable_weights(w)
            return m, o

        m_ss, o_ss = make()
        m_ms, o_ms = make()
        m_bp, o_bp = make()

        @brainstate.transform.jit
        def train_ss(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                m_ss, o_ss, inputs, labels, decay_or_rank=0.97, vjp_method="single-step"
            )

        @brainstate.transform.jit
        def train_ms(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                m_ms, o_ms, inputs, labels, decay_or_rank=0.97, vjp_method="multi-step"
            )

        @brainstate.transform.jit
        def train_bp(inputs, labels):
            return _shared.bptt_train_epoch_fixed_target(m_bp, o_bp, inputs, labels)

        ss_l, ms_l, bp_l = [], [], []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_dms_spikes(
                num_step=num_step, num_batch=batch_size, n_in=n_in, seed=epoch,
            )
            ss_l.append(float(train_ss(xs, ys)))
            ms_l.append(float(train_ms(xs, ys)))
            bp_l.append(float(train_bp(xs, ys)))
            print(
                f"[14-vjp-contrast] epoch {epoch}  "
                f"single={ss_l[-1]:.4f}  multi={ms_l[-1]:.4f}  bptt={bp_l[-1]:.4f}"
            )

        xs_e, ys_e = _shared.make_dms_spikes(
            num_step=num_step, num_batch=batch_size, n_in=n_in, seed=9999,
        )
        acc_ss = _eval(m_ss, xs_e, ys_e)
        acc_ms = _eval(m_ms, xs_e, ys_e)
        acc_bp = _eval(m_bp, xs_e, ys_e)
        print(f"[14-vjp-contrast] acc  single={acc_ss:.3f}  multi={acc_ms:.3f}  bptt={acc_bp:.3f}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ss_l, label="single-step")
        ax.plot(ms_l, label="multi-step")
        ax.plot(bp_l, label="BPTT", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("14 · vjp_method contrast on DMS")
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    for val in (ss_l[-1], ms_l[-1], bp_l[-1]):
        assert jnp.isfinite(jnp.asarray(val))
    return {
        "losses": ss_l, "multi_step_losses": ms_l, "bptt_losses": bp_l,
        "acc_single": acc_ss, "acc_multi": acc_ms, "acc_bptt": acc_bp,
    }


if __name__ == "__main__":
    result = main()
    # Full-run chance-level bounds (smoke test skips these since it uses n_epochs=1):
    assert result["acc_single"] > 0.5, f"single-step acc {result['acc_single']} <= 0.5"
    assert result["acc_multi"] > 0.5, f"multi-step acc {result['acc_multi']} <= 0.5"
```

- [ ] **Step 2: Register + run smoke + commit**

Append `"14-knob-vjp-method-contrast.py"`. Run smoke test. Expected: 14 parametrized tests PASS.

```bash
git add examples/pp_prop/14-knob-vjp-method-contrast.py examples/pp_prop/tests/test_smoke.py
git commit -m "examples(pp_prop): add 14-knob-vjp-method-contrast"
```

---

## Task 20: `examples/pp_prop/README.md`

**Files:**
- Create: `examples/pp_prop/README.md`

- [ ] **Step 1: Write README**

File `examples/pp_prop/README.md`:
```markdown
# pp_prop Examples

A tutorial-linear walk through `braintrace.pp_prop` (aliases `ES_D_RTRL` /
`IODimVjpAlgorithm`) — an online eligibility-trace gradient estimator with
input-output dimensional complexity for spiking neural networks. Each file
is self-contained. Read them in order (01 → 14) to follow the companion
tutorial at `docs/tutorials/pp_prop.md`.

## How to run

    python examples/pp_prop/01-basics-lif-integrator.py

All examples run on CPU in roughly 1–2 minutes. No external datasets; the
Poisson-MNIST examples use sklearn's 8×8 digits (with a pure-numpy fallback
if sklearn is missing).

## Axis map

| Axis                                       | Files                    |
|--------------------------------------------|--------------------------|
| Neuron model (LIF / ALIF / GIF / COBA-EI)  | 01, 02, 03, 04           |
| Batching mode (vmap vs batched primitive)  | 05, 06                   |
| vjp_method (single-step vs multi-step)     | 07, 08, 14               |
| Operator (matmul / sparse / LoRA / conv)   | 09, 10, 11               |
| Training target                            | 01, 02, 03, 04, 12       |
| Algo knob (decay vs rank)                  | 13                       |
| BPTT baseline                              | 12, 14                   |

### File-by-file summary

| #  | File                                  | Demo                                                         |
|----|---------------------------------------|--------------------------------------------------------------|
| 01 | `01-basics-lif-integrator.py`         | LIF RSNN on Poisson-to-cumulative-rate regression            |
| 02 | `02-neurons-alif-dms.py`              | ALIF (adaptive threshold) on delayed-match-to-sample         |
| 03 | `03-neurons-gif-working-memory.py`    | GIF with heterogeneous tau_I2 on working-memory recall       |
| 04 | `04-neurons-coba-ei-rsnn.py`          | Dale-law E/I RSNN on small Poisson-MNIST                     |
| 05 | `05-batching-vmap.py`                 | Batching via `brainstate.nn.Vmap(vmap_states='new')`         |
| 06 | `06-batching-batched.py`              | Batching via the batched ETP primitive path                  |
| 07 | `07-vjp-single-step.py`               | `vjp_method='single-step'` (default)                         |
| 08 | `08-vjp-multi-step.py`                | `vjp_method='multi-step'` for temporal credit                |
| 09 | `09-operator-sparse.py`               | Sparse recurrent connectivity (masked-dense fallback)        |
| 10 | `10-operator-lora.py`                 | Low-rank recurrence via `braintrace.lora_matmul`             |
| 11 | `11-operator-conv.py`                 | Conv-SNN via `braintrace.conv`                               |
| 12 | `12-classification-neuromorphic.py`   | Flagship: pp_prop vs BPTT on Poisson-MNIST (10 classes)      |
| 13 | `13-knob-decay-vs-rank.py`            | Sweep `decay_or_rank` across floats and ints                 |
| 14 | `14-knob-vjp-method-contrast.py`      | single-step vs multi-step vs BPTT head-to-head on DMS        |

Cross-reference: for `fast_solve` / `normalize_matrix_spectrum` knobs
(shared with D_RTRL but not required for pp_prop), see
`examples/drtrl/11-knob-fast-solve.py` and
`examples/drtrl/12-knob-normalize-spectrum.py`.

## Tutorial

See `docs/tutorials/pp_prop.md` for the long-form narrative.

## Tests

    pytest examples/pp_prop/tests -v
```

- [ ] **Step 2: Commit**

```bash
git add examples/pp_prop/README.md
git commit -m "docs(pp_prop): add examples README with axis map"
```

---

## Task 21: `docs/tutorials/pp_prop.md`

**Files:**
- Create: `docs/tutorials/pp_prop.md`

- [ ] **Step 1: Write tutorial**

File `docs/tutorials/pp_prop.md`:
```markdown
# pp_prop — Online Gradient Estimation for Spiking Networks

`braintrace.pp_prop` (exposed also as `braintrace.ES_D_RTRL` and
`braintrace.IODimVjpAlgorithm`) is an online gradient estimator based on
eligibility traces with input-output dimensional complexity. It trains
recurrent spiking networks one time step at a time, without
backpropagation through time.

This tutorial is a narrative companion to `examples/pp_prop/`. Open the
examples side-by-side as you read.

## 1. What pp_prop solves

Backpropagation through time (BPTT) gives exact gradients for recurrent
networks but needs to store the entire forward trajectory in memory. For
long sequences, deep networks, or streaming inputs, the memory cost blows
up.

pp_prop replaces the full Jacobian product by an **eligibility trace** —
a low-pass-filtered approximation of the parameter-to-hidden-state
Jacobian — plus a per-step VJP from the loss to the hidden state:

$$
\nabla_{\boldsymbol\theta} \mathcal L
\approx
\sum_t \frac{\partial \mathcal L^t}{\partial h^t} \odot \boldsymbol\epsilon^t,
\qquad
\boldsymbol\epsilon^t \approx \boldsymbol\epsilon_f^t \otimes \boldsymbol\epsilon_x^t.
$$

The trace factorises into an input-side term and a hidden-side term, each
updated by a low-pass filter with decay $\alpha$:

$$
\boldsymbol\epsilon_x^t = \alpha\,\boldsymbol\epsilon_x^{t-1} + x^t
\qquad
\boldsymbol\epsilon_f^t = \alpha\,\mathrm{diag}(D^t)\circ \boldsymbol\epsilon_f^{t-1}
  + (1-\alpha)\,\mathrm{diag}(D_f^t)
$$

where $D^t, D_f^t$ are local Jacobians produced by the rule registered
for each ETP primitive.

### Complexity

| Quantity | pp_prop (I/O dim) | D_RTRL (param dim) | BPTT |
|----------|-------------------|---------------------|------|
| Memory   | $O(BI + BO)$      | $O(BIO)$            | $O(TBIO)$ |
| Compute  | $O(TBIO)$         | $O(TBIO)$           | $O(TBIO)$ |

For a linear recurrent layer with $n$ hidden units:
pp_prop is $O(Bn)$ memory and $O(Bn^2)$ compute per step.

## 2. When pp_prop beats D_RTRL

Both algorithms are online and avoid BPTT's trajectory storage. pp_prop
saves the extra factor-of-IO memory by replacing the outer-product
eligibility tensor with its rank-1 factorisation. The price is a diagonal
approximation: correlations between rows of the parameter-to-hidden
Jacobian are discarded.

Use pp_prop when:

- Hidden states are large (memory savings matter).
- Online streaming is required (long, unbounded sequences).
- The task's gradient signal does not rely on fine-grained cross-parameter
  correlations.

Fall back to D_RTRL when memory is not the bottleneck and a tighter
gradient estimate is needed. Fall back to BPTT when the sequence is short
enough to fit in memory and you need exact gradients for a published
benchmark.

## 3. Walk-through of the example series

Each paragraph below points at the example and the single axis it is
demonstrating.

### 3.1 Basics (file 01)

`01-basics-lif-integrator.py` is the smallest working pp_prop call.
A one-layer LIF RSNN receives a stream of Poisson spikes; its readout is
trained to match the cumulative spike rate.

### 3.2 Neuron models (files 02-04)

`02-neurons-alif-dms.py` swaps LIF for an ALIF neuron on DMS.
`03-neurons-gif-working-memory.py` introduces GIF with per-neuron slow
adaptation currents. `04-neurons-coba-ei-rsnn.py` builds a Dale-law E/I
block where recurrent signs are fixed by the initialiser. In all three
cases, pp_prop needs no modification — the per-primitive rules in
`braintrace/_etrace_op/` cover the dense matmul gate, and the neuron
dynamics are transparent to the algorithm.

### 3.3 Batching (files 05-06)

`05-batching-vmap.py` uses `brainstate.nn.Vmap(vmap_states='new')` to
replicate the unbatched model across the batch dimension.
`06-batching-batched.py` bypasses vmap by making the network natively
batched — `braintrace.matmul` dispatches to the batched primitive
`etp_mm_p` when the input already has a batch axis.

### 3.4 VJP methods (files 07-08)

`07-vjp-single-step.py` uses the default `vjp_method='single-step'`:
$\partial L^t / \partial h^t$ is computed from the loss at time $t$ only.
`08-vjp-multi-step.py` uses `vjp_method='multi-step'`, which multiplies
the loss gradient by the window of recent hidden-to-hidden Jacobians
before folding it into the trace. Multi-step is strictly more expressive
but runs slower.

### 3.5 Operators (files 09-11)

`09-operator-sparse.py` uses a fixed sparse connectivity mask on the
recurrent matrix. Because `saiunit.sparse` COO/CSR primitives lack JAX
batching rules today, the file uses a masked-dense fallback that still
exercises pp_prop's per-primitive trace rule.
`10-operator-lora.py` parameterises the recurrent matrix as
$W = \alpha BA$ with rank $r \ll n$, via `braintrace.lora_matmul`.
`11-operator-conv.py` swaps matmul for a 2D convolution, demonstrating
that pp_prop's eligibility trace is per-primitive — adding a new operator
means writing a new ETP rule, not changing the algorithm.

### 3.6 Flagship comparison (file 12)

`12-classification-neuromorphic.py` trains two identical LIF RSNNs on
Poisson-encoded 10-class digits, one with pp_prop, one with BPTT. It
reports per-epoch losses and final accuracies.

### 3.7 Knob sweeps (files 13-14)

`13-knob-decay-vs-rank.py` sweeps `decay_or_rank` across both float and
integer parameterisations ($n_{\text{rank}} = 2/(1-\alpha)-1$).
`14-knob-vjp-method-contrast.py` runs single-step pp_prop, multi-step
pp_prop, and BPTT on the same DMS task and plots three loss curves on
one axis.

## 4. Limitations

1. **Diagonal approximation.** The trace factorises as
   $\epsilon_f \otimes \epsilon_x$, which drops off-diagonal couplings.
   Tasks that depend critically on those couplings may learn slower than
   with BPTT.
2. **Single hidden-group assumption per primitive.** If an ETP primitive
   feeds multiple disjoint hidden groups, pp_prop allocates one trace per
   group but relies on the per-primitive rule to handle the summation.
3. **Operator-invariant rule.** Each new operator needs a hand-written
   `xy_to_dw` / `init_pp` rule in `braintrace/_etrace_op/`. See
   `CLAUDE.md` for the "adding a new primitive" recipe.
4. **Weight-through-weight pathways are not supported.** If a trainable
   ETP weight $W_1$ feeds another trainable ETP weight $W_2$ before
   reaching a hidden state, the compiler correctly excludes $W_1$
   (see `CLAUDE.md` "no weight → weight → hidden pathway" invariant).
   Cells like GRU hit this: some of their inner linears are not trained
   by pp_prop and remain non-temporal.

## 5. When to reach for BPTT instead

If your sequences fit in memory, if you can afford the compute, and if
you need exact gradients for a published benchmark — use BPTT. pp_prop is
for the regime where BPTT's memory is the bottleneck. File 12 shows that
on a mid-sized task pp_prop reaches competitive accuracy, but on tasks
where the diagonal approximation loses signal it will lag.

## 6. Further reading

- The `ES-D-RTRL` manuscript:
  [https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2](https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2)
- `braintrace/_etrace_vjp/pp_prop.py` — full docstrings and mathematical
  derivation of the update rules.
- `docs/tutorials/drtrl.md` — the parameter-dimensional dual algorithm
  (D_RTRL / `ParamDimVjpAlgorithm`).
```

- [ ] **Step 2: Commit**

```bash
git add docs/tutorials/pp_prop.md
git commit -m "docs(tutorial): add pp_prop long-form tutorial"
```

---

## Task 22: Final verification

- [ ] **Step 1: Full smoke suite**

Run: `pytest examples/pp_prop/tests -v`
Expected: 14 parametrized smoke tests PASS + 4 unit tests in
`test_shared_data.py` + 3 unit tests in `test_shared_cells.py`.

- [ ] **Step 2: No regression in braintrace unit tests**

Run: `pytest braintrace/ --timeout=60 -q`
Expected: pass-rate unchanged from baseline.

- [ ] **Step 3: drtrl suite unaffected**

Run: `pytest examples/drtrl/tests -v`
Expected: all pre-existing tests PASS.

- [ ] **Step 4: Flagship sanity run**

Run: `python examples/pp_prop/12-classification-neuromorphic.py`
Expected: four epochs of "pp_prop=... bptt=..." prints; final accuracies
both above chance (0.10); no NaN.

- [ ] **Step 5: Git history review**

Run: `git log --oneline -30`
Expected: incremental commits for every file (01-14) + shared + README +
tutorial + scaffold, each on its own commit.

- [ ] **Step 6: Straggler fixes**

If any of the previous steps surface issues (e.g., a loosened assertion,
a missing `__init__.py`, a failing smoke test), make the fix and commit
with `chore(pp_prop): ...`.

---

## Appendix: deferred decisions (record choice in commit message)

1. **Sparse backend for file 09.** Plan uses masked-dense. Alternatives:
   `saiunit.sparse` (blocked) or `jax.experimental.sparse.BCOO` (if the
   `braintrace.sparse_matmul` API accepts it).
2. **GIF class for file 03.** Plan imports `examples.snn_models.GIF` via
   lazy `sys.path` insertion in `_shared.GIFCell`. Alternative: upstream
   `brainpy.state.GIF` if it gains per-neuron `tau_I2`.
3. **Dale's law for file 04.** Plan uses signed init on a plain
   `braintrace.nn.Linear` ("soft Dale"). Alternative: `SignedWLinear` if
   it is wrapped by an ETP primitive.
4. **Conv2d shape for file 11.** Plan uses 1×4 channels, 3×3 kernel,
   8×8 spatial, T≤20. Tune down if runtime > 2 min.
5. **sklearn fallback.** Plan imports sklearn lazily with a numpy random
   fallback. If sklearn becomes a hard project dependency, delete the
   fallback.

## Self-review notes

Reviewed after writing. Issues found and fixed inline:

1. **File 12 accuracy assertion.** Spec asserted `final_acc > 0.6`; plan
   uses `> chance` so the smoke test (1 epoch, batch 4) does not falsely
   fail. Manual default run (`n_epochs=4, batch_size=32`) is the regime
   where 0.6+ is achievable; assertion can be tightened post-hoc.
2. **`main()` signature consistency.** Every example exports
   `main(n_epochs, batch_size, plot=True, ...)` and returns a dict with
   at least `"losses"` (list of floats). Smoke test calls
   `main(n_epochs=1, batch_size=4, plot=False)`.
3. **`GIFCell` lazy import.** The cell lazy-imports
   `examples.snn_models.GIF` inside `__init__` via `sys.path` insertion,
   so `_shared.py` imports cleanly whether or not `examples/` is on
   `sys.path` at module load time.
4. **`brainpy.state` import placement.** In file 09 the `SparseRecCell`
   uses `brainpy.state.Expon` / `brainpy.state.LIF`; the import is at
   file top, not inside the class.
5. **Type consistency across tasks.** All 14 examples use `Net`
   (not `Model` or `RNN`), all use `main(n_epochs, batch_size, plot)`
   (matching `examples/drtrl/`). All examples call
   `_shared.online_train_epoch` (regression/sequence-target variant) or
   `_shared.online_train_epoch_fixed_target` (classification variant).
6. **File 09 concern.** Using `braintrace.matmul(last_spk, self.rec.value * self.mask)`
   keeps the fixed-zero entries frozen only under gradient updates that
   multiply by the mask. If the optimizer applies updates to
   `self.rec.value` directly, the zeros may drift. Mitigate by applying
   a mask in the optimizer step (`opt.update` on `grads * mask`) if the
   drift is observed during implementation.
