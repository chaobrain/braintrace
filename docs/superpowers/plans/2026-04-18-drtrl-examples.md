# D-RTRL Examples + Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 12 self-contained `D_RTRL` example programs plus one tutorial
under `examples/drtrl/` and `docs/tutorials/drtrl.md`, covering operators
(matmul, sparse, LoRA, conv), batching modes (vmap / Batching), vjp methods
(single / multi-step), training targets (regression, classification, copy,
generation), and tunable knobs (`fast_solve`, `normalize_matrix_spectrum`).

**Architecture:** All examples share a thin `_shared.py` module of data
generators and training helpers. Each example follows the same shape:
hyperparameters → model class → loss → training loop (online + optional
BPTT) → plotting guarded by `__main__`. The tutorial lives in
`docs/tutorials/drtrl.md` and points to each numbered example file for
concrete code.

**Tech Stack:** `brainstate >= 0.2.2`, `braintools`, `braintrace` (this repo),
`jax`, `saiunit`, `numpy`, `matplotlib`. `torchvision` used only by Task 10.
Pytest for smoke tests.

**Layered TDD notes for this plan:** Every example is guarded by a
tiny pytest smoke test that calls its `main(override=...)` with a one-epoch
override. Tests are written first, the example module stub second, the real
body third, then a commit. This matches the skill's RED-GREEN-COMMIT loop
even when the "unit under test" is an example script.

---

## File Structure

```
examples/drtrl/
├── README.md                              # human index
├── __init__.py                            # empty marker so tests can import
├── _shared.py                             # data generators + training helpers
├── 01-basics-integrator.py
├── 02-batching-vmap.py
├── 03-batching-batched.py
├── 04-vjp-single-step.py
├── 05-vjp-multi-step.py
├── 06-operator-sparse.py
├── 07-operator-lora.py
├── 08-operator-conv.py
├── 09-classification-mnist.py
├── 10-char-lm-generation.py
├── 11-knob-fast-solve.py
├── 12-knob-normalize-spectrum.py
└── tests/
    ├── __init__.py
    └── test_smoke.py

docs/tutorials/drtrl.md
```

**Shared contract** — every example module MUST expose:

- A top-level `def main(*, n_epochs: int = DEFAULT, batch_size: int = DEFAULT, plot: bool = True) -> dict` function.
- Return dict keys: `{"losses": list[float]}` at minimum. Flagship files add
  `"bptt_losses"`, `"accuracy"`, or `"samples"` as appropriate.
- Module-level `if __name__ == "__main__": main()` calling with defaults.
- Plotting is done inside `main()` only when `plot=True`.

The smoke test calls `main(n_epochs=1, batch_size=4, plot=False)` for each
example; the default-arg call path remains untested automatically.

---

## Important note about file names

Python filenames with digits + hyphen (`01-basics-integrator.py`) are NOT
valid import identifiers. The smoke test must import them by path via
`importlib.util`, not `import examples.drtrl.01_basics_integrator`. The plan
uses `importlib.util.spec_from_file_location` everywhere.

---

## Task 1: Create package skeleton + shared data generators

**Files:**
- Create: `examples/drtrl/__init__.py`
- Create: `examples/drtrl/_shared.py`
- Create: `examples/drtrl/tests/__init__.py`
- Create: `examples/drtrl/tests/test_shared.py`

- [ ] **Step 1: Write failing tests for `_shared.py`**

Create `examples/drtrl/tests/test_shared.py`:

```python
import importlib.util
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest


def _load_shared():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_drtrl_shared", root / "_shared.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_integrator_batch_shape():
    s = _load_shared()
    x, y = s.make_integrator_batch(num_step=25, num_batch=4)
    assert x.shape == (25, 4, 1)
    assert y.shape == (25, 4, 1)
    assert jnp.allclose(y, jnp.cumsum(x, axis=0))


def test_copy_batch_shape():
    s = _load_shared()
    x, y = s.make_copy_batch(time_lag=5, batch_size=4, seed=0)
    assert x.shape == (5 + 20, 4, 10)
    assert y.shape == (10, 4)


def test_xor_batch_shape_and_values():
    s = _load_shared()
    x, y = s.make_xor_batch(seq_len=12, delay=4, batch_size=8, seed=0)
    assert x.shape == (12, 8, 2)
    assert y.shape == (8,)
    assert set(np.asarray(y).tolist()).issubset({0, 1})


def test_sine_batch_shape():
    s = _load_shared()
    x, y = s.make_sine_batch(num_step=40, batch_size=4, seed=0)
    assert x.shape == (40, 4, 1)
    assert y.shape == (40, 4, 1)


def test_char_batches_shape():
    s = _load_shared()
    text = "abcdefghij" * 10
    vocab, x, y = s.make_char_batches(text=text, seq_len=8, batch_size=4, seed=0)
    assert len(vocab) > 0
    assert x.shape == (8, 4, len(vocab))
    assert y.shape == (8, 4)
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `pytest examples/drtrl/tests/test_shared.py -v`
Expected: ModuleNotFoundError / ImportError — `_shared.py` does not yet exist.

- [ ] **Step 3: Implement `examples/drtrl/__init__.py` and `tests/__init__.py`**

Write empty-content files:

```python
# examples/drtrl/__init__.py
# examples/drtrl/tests/__init__.py
```

- [ ] **Step 4: Implement `examples/drtrl/_shared.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Shared data generators and thin training helpers for examples/drtrl/*.py."""

from __future__ import annotations

from typing import Callable, Tuple

import brainstate
import jax
import jax.numpy as jnp
import numpy as np


def make_integrator_batch(
    num_step: int = 25,
    num_batch: int = 64,
    mean: float = 0.025,
    scale: float = 0.01,
    dt: float = 0.04,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Cumulative-sum regression task (continuous-valued)."""
    rng = np.random.default_rng(seed)
    bias_sample = rng.standard_normal((1, num_batch, 1)).astype(np.float32)
    bias = mean * 2.0 * (bias_sample - 0.5)
    noise = (scale / np.sqrt(dt)) * rng.standard_normal((num_step, num_batch, 1)).astype(np.float32)
    inputs = bias + noise
    targets = np.cumsum(inputs, axis=0)
    return jnp.asarray(inputs), jnp.asarray(targets)


def make_copy_batch(
    time_lag: int,
    batch_size: int,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Copying-memory task. 10 input symbols (0..8 content, 9 = cue)."""
    rng = np.random.default_rng(seed)
    seq_length = time_lag + 20
    ids = np.zeros((batch_size, seq_length), dtype=np.int32)
    ids[:, :10] = rng.integers(1, 9, (batch_size, 10))
    ids[:, -10:] = 9
    one_hot = np.zeros((batch_size, seq_length, 10), dtype=np.float32)
    for b in range(batch_size):
        one_hot[b, np.arange(seq_length), ids[b]] = 1.0
    x = jnp.asarray(np.transpose(one_hot, (1, 0, 2)))  # (T, B, 10)
    y = jnp.asarray(ids[:, :10].T)  # (10, B)
    return x, y


def make_xor_batch(
    seq_len: int,
    delay: int,
    batch_size: int,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Delayed XOR. Two bits shown at t=0 and t=delay, answer needed at t=seq_len-1."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=(batch_size,)).astype(np.float32)
    b = rng.integers(0, 2, size=(batch_size,)).astype(np.float32)
    x = np.zeros((seq_len, batch_size, 2), dtype=np.float32)
    x[0, :, 0] = a
    x[delay, :, 1] = b
    y = (a.astype(np.int32) ^ b.astype(np.int32)).astype(np.int32)
    return jnp.asarray(x), jnp.asarray(y)


def make_sine_batch(
    num_step: int,
    batch_size: int,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Random-frequency sine continuation regression."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.5, 2.5, size=(batch_size,)).astype(np.float32)
    t = np.arange(num_step, dtype=np.float32)[:, None] / 10.0
    signal = np.sin(2.0 * np.pi * freqs[None, :] * t)
    x = signal[:, :, None].astype(np.float32)
    y = np.roll(signal, -1, axis=0)[:, :, None].astype(np.float32)
    return jnp.asarray(x), jnp.asarray(y)


def make_char_batches(
    text: str,
    seq_len: int,
    batch_size: int,
    seed: int | None = None,
) -> Tuple[str, jnp.ndarray, jnp.ndarray]:
    """Character-level batch from a single corpus string."""
    vocab = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    data = np.asarray([char2idx[c] for c in text], dtype=np.int32)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - seq_len - 1, size=(batch_size,))
    x_ids = np.stack([data[s : s + seq_len] for s in starts], axis=1)  # (T, B)
    y_ids = np.stack([data[s + 1 : s + 1 + seq_len] for s in starts], axis=1)
    one_hot = np.eye(len(vocab), dtype=np.float32)[x_ids]
    return ''.join(vocab), jnp.asarray(one_hot), jnp.asarray(y_ids)


def accumulate_grads(
    weights,
    step_grad_fn: Callable,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> Tuple[dict, jnp.ndarray]:
    """Scan ``step_grad_fn`` over (inputs, targets) accumulating gradients."""
    init = jax.tree.map(lambda a: jnp.zeros_like(a), {k: v.value for k, v in weights.items()})

    def body(prev_grads, pair):
        inp, tar = pair
        cur_grads, loss = step_grad_fn(inp, tar)
        next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
        return next_grads, loss

    return brainstate.transform.scan(body, init, (inputs, targets))
```

- [ ] **Step 5: Run tests — verify they pass**

Run: `pytest examples/drtrl/tests/test_shared.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add examples/drtrl/__init__.py examples/drtrl/_shared.py examples/drtrl/tests/__init__.py examples/drtrl/tests/test_shared.py
git commit -m "examples(drtrl): add shared data generators and training helpers"
```

---

## Task 2: `01-basics-integrator.py` — minimal D_RTRL + BPTT baseline

**Files:**
- Create: `examples/drtrl/01-basics-integrator.py`
- Create: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Write failing smoke test**

Create `examples/drtrl/tests/test_smoke.py`:

```python
"""Smoke tests: each example's main() must run one epoch end-to-end."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(f"_drtrl_{fname}", EXAMPLES_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fname", [
    "01-basics-integrator.py",
])
def test_example_runs(fname):
    mod = _load(fname)
    result = mod.main(n_epochs=1, batch_size=4, plot=False)
    assert "losses" in result
```

- [ ] **Step 2: Run — verify failure**

Run: `pytest examples/drtrl/tests/test_smoke.py -v`
Expected: FileNotFoundError / SpecLoaderError — `01-basics-integrator.py` missing.

- [ ] **Step 3: Implement `01-basics-integrator.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""01 · Minimal D_RTRL on the integrator task.

The smallest working example. Trains a vanilla RNN on noisy cumulative-sum
regression and compares D_RTRL against BPTT on the same initialisation.
Read the inline comments top to bottom.
"""
from __future__ import annotations

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


def _train_online(n_epochs, num_step, num_batch, num_hidden, lr):
    model = RNN(1, num_hidden)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=lr, eps=1e-1)
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online_model = braintrace.D_RTRL(model)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online_model.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online_model, vmap_states='new')

        def step_loss(inp, tar):
            out = vmap_model(inp)
            return braintools.metric.squared_error(out, tar).mean(), out

        def grad_step(prev_grads, x):
            inp, tar = x
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=num_batch)
        losses.append(float(f_train(x, y)))
    return model, losses


def _train_bptt(n_epochs, num_step, num_batch, num_hidden, lr):
    model = RNN(1, num_hidden)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=lr, eps=1e-1)
    opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_predict(inputs):
        brainstate.nn.init_all_states(model, batch_size=inputs.shape[1])
        return brainstate.transform.for_loop(model.update, inputs)

    def f_loss(inputs, targets):
        preds = f_predict(inputs)
        return braintools.metric.squared_error(preds, targets).mean()

    @brainstate.transform.jit
    def f_train(inputs, targets):
        grads, l = brainstate.transform.grad(f_loss, weights, return_value=True)(inputs, targets)
        opt.update(grads)
        return l

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=num_batch)
        losses.append(float(f_train(x, y)))
    return model, losses


def main(*, n_epochs: int = 50, batch_size: int = 64, plot: bool = True) -> dict:
    num_step = 25
    num_hidden = 32
    _, online_losses = _train_online(n_epochs, num_step, batch_size, num_hidden, lr=5e-3)
    _, bptt_losses = _train_bptt(n_epochs, num_step, batch_size, num_hidden, lr=2.5e-2)
    if plot:
        plt.plot(online_losses, label='D_RTRL')
        plt.plot(bptt_losses, label='BPTT')
        plt.xlabel('epoch'); plt.ylabel('MSE')
        plt.legend(); plt.title('01 · Basics — integrator')
        plt.show()
    return {"losses": online_losses, "bptt_losses": bptt_losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test — verify pass**

Run: `pytest examples/drtrl/tests/test_smoke.py -v`
Expected: 1 passed. Runtime under 90s on CPU.

- [ ] **Step 5: Commit**

```bash
git add examples/drtrl/01-basics-integrator.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 01-basics-integrator with BPTT baseline"
```

---

## Task 3: `02-batching-vmap.py` — vmap_new_states pattern

**Files:**
- Create: `examples/drtrl/02-batching-vmap.py`
- Modify: `examples/drtrl/tests/test_smoke.py` (add to parametrize list)

- [ ] **Step 1: Add to smoke parametrize + verify failure**

Edit `test_smoke.py` — append `"02-batching-vmap.py"` to parametrize list.
Run: `pytest examples/drtrl/tests/test_smoke.py -v -k batching_vmap`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `02-batching-vmap.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""02 · Batching via ``vmap_new_states``.

Shows the per-sample-init pattern explicitly:
    1. wrap model in D_RTRL
    2. inside a vmapped new-states scope: init_all_states + compile_graph
    3. outside, wrap the online model in brainstate.nn.Vmap

Pick this pattern when every sample needs its own eligibility trace state
(the usual case).
"""
from __future__ import annotations

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
        vmap_model = brainstate.nn.Vmap(online_model, vmap_states='new')

        def step_loss(inp, tar):
            out = vmap_model(inp)
            return braintools.metric.squared_error(out, tar).mean(), out

        def grad_step(prev_grads, x):
            inp, tar = x
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('MSE')
        plt.title('02 · Batching via vmap_new_states'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k batching_vmap`
Expected: 1 passed.

```bash
git add examples/drtrl/02-batching-vmap.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 02-batching-vmap pattern"
```

---

## Task 4: `03-batching-batched.py` — Batching mode

**Files:**
- Create: `examples/drtrl/03-batching-batched.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"03-batching-batched.py"`. Run:
`pytest examples/drtrl/tests/test_smoke.py -v -k batched`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `03-batching-batched.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""03 · Batching via ``brainstate.mixin.Batching``.

Alternative batching path: the algorithm sees the batch axis directly
instead of relying on vmap. Init once with ``batch_size=...``, compile
on a batched sample.
"""
from __future__ import annotations

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
        online_model = braintrace.D_RTRL(model, mode=brainstate.mixin.Batching())
        brainstate.nn.init_all_states(model, batch_size=inputs.shape[1])
        online_model.compile_graph(inputs[0])

        def step_loss(inp, tar):
            out = online_model(inp)
            return braintools.metric.squared_error(out, tar).mean(), out

        def grad_step(prev_grads, x):
            inp, tar = x
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_integrator_batch(num_step=num_step, num_batch=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('MSE')
        plt.title('03 · Batching via brainstate.mixin.Batching'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k batched`
Expected: 1 passed.

```bash
git add examples/drtrl/03-batching-batched.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 03-batching-batched (Batching mode)"
```

---

## Task 5: `04-vjp-single-step.py` — GRU copy task, single-step

**Files:**
- Create: `examples/drtrl/04-vjp-single-step.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"04-vjp-single-step.py"`. Run:
`pytest examples/drtrl/tests/test_smoke.py -v -k single_step`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `04-vjp-single-step.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""04 · vjp_method='single-step' on the copying-memory task.

Single-step computes the VJP only at the current timestep. Cheapest mode,
introduces gradient bias over long dependencies. Good default.
"""
from __future__ import annotations

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


class GRUNet(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = braintrace.nn.GRUCell(n_in, n_rec)
        self.readout = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(*, n_epochs: int = 50, batch_size: int = 32, plot: bool = True) -> dict:
    n_in, n_rec, n_out, time_lag = 10, 64, 10, 10
    model = GRUNet(n_in, n_rec, n_out)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(1e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model, vjp_method='single-step')

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online, vmap_states='new')
        warmup_len = inputs.shape[0] - 10
        brainstate.transform.for_loop(lambda inp: vmap_model(inp), inputs[:warmup_len])

        def step_loss(inp, tar):
            out = vmap_model(inp)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, step_losses = brainstate.transform.scan(
            grad_step, init_grads, (inputs[warmup_len:], targets)
        )
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_copy_batch(time_lag=time_lag, batch_size=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('cross-entropy')
        plt.title('04 · single-step VJP — copying task'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k single_step`
Expected: 1 passed.

```bash
git add examples/drtrl/04-vjp-single-step.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 04-vjp-single-step (GRU + copy task)"
```

---

## Task 6: `05-vjp-multi-step.py` — multi-step counterpart

**Files:**
- Create: `examples/drtrl/05-vjp-multi-step.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"05-vjp-multi-step.py"`. Run:
`pytest examples/drtrl/tests/test_smoke.py -v -k multi_step`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `05-vjp-multi-step.py`**

Full file — copy verbatim:

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""05 · vjp_method='multi-step' on the copying-memory task.

Multi-step computes the VJP over a window, reducing single-step bias at the
cost of more compute and memory. Compare the loss curve here with Task 04.
"""
from __future__ import annotations

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


class GRUNet(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = braintrace.nn.GRUCell(n_in, n_rec)
        self.readout = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(*, n_epochs: int = 50, batch_size: int = 32, plot: bool = True) -> dict:
    n_in, n_rec, n_out, time_lag = 10, 64, 10, 10
    model = GRUNet(n_in, n_rec, n_out)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(1e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model, vjp_method='multi-step')

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online, vmap_states='new')
        warmup_len = inputs.shape[0] - 10
        brainstate.transform.for_loop(lambda inp: vmap_model(inp), inputs[:warmup_len])

        def step_loss(inp, tar):
            out = vmap_model(inp)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, step_losses = brainstate.transform.scan(
            grad_step, init_grads, (inputs[warmup_len:], targets)
        )
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_copy_batch(time_lag=time_lag, batch_size=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('cross-entropy')
        plt.title('05 · multi-step VJP — copying task'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k multi_step`
Expected: 1 passed.

```bash
git add examples/drtrl/05-vjp-multi-step.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 05-vjp-multi-step counterpart to 04"
```

---

## Task 7: `06-operator-sparse.py` — SparseLinear recurrent

**Files:**
- Create: `examples/drtrl/06-operator-sparse.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"06-operator-sparse.py"`. Run:
`pytest examples/drtrl/tests/test_smoke.py -v -k sparse`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `06-operator-sparse.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""06 · ``braintrace.sparse_matmul`` as the recurrent operator.

Uses ``braintrace.nn.SparseLinear`` to hold the recurrent weight as a sparse
matrix. Only the nonzero entries are trained; the sparsity pattern is fixed.

Task: delayed XOR — two bits at t=0 and t=delay, XOR label at final step.
"""
from __future__ import annotations

import pathlib
import sys

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import saiunit as u

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


def _make_sparse_mat(n_hidden: int, density: float = 0.1, seed: int = 0):
    """Build a CSR-style sparse mask (n_hidden, n_hidden)."""
    rng = np.random.default_rng(seed)
    mask = (rng.random((n_hidden, n_hidden)) < density).astype(np.float32)
    rows, cols = np.nonzero(mask)
    indices = jnp.asarray(np.stack([rows, cols], axis=1))
    data = jnp.ones(len(rows))
    return u.sparse.COO((data, indices), shape=(n_hidden, n_hidden))


class SparseRNNCell(brainstate.nn.RNNCell):
    def __init__(self, n_in: int, n_hidden: int, density: float = 0.1):
        super().__init__()
        self.out_size = n_hidden
        self.in_size = n_in
        self.input_w = braintrace.nn.Linear(n_in, n_hidden)
        self.rec = braintrace.nn.SparseLinear(_make_sparse_mat(n_hidden, density), b_init=braintools.init.ZeroInit())

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(
            braintools.init.param(braintools.init.ZeroInit(), self.out_size, batch_size)
        )

    def reset_state(self, batch_size=None, **kwargs):
        self.h.value = braintools.init.param(braintools.init.ZeroInit(), self.out_size, batch_size)

    def update(self, x):
        pre = self.input_w(x) + self.rec(self.h.value)
        self.h.value = jax.nn.tanh(pre)
        return self.h.value


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.cell = SparseRNNCell(n_in, n_hidden)
        self.readout = braintrace.nn.Linear(n_hidden, n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def main(*, n_epochs: int = 40, batch_size: int = 32, plot: bool = True) -> dict:
    seq_len, delay, n_hidden = 20, 10, 48
    model = Net(2, n_hidden, 2)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(5e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online, vmap_states='new')
        brainstate.transform.for_loop(lambda inp: vmap_model(inp), inputs[:-1])

        def final_loss():
            out = vmap_model(inputs[-1])
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean(), out

        grads, loss, _ = brainstate.transform.grad(final_loss, weights, has_aux=True, return_value=True)()
        opt.update(grads)
        return loss

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_xor_batch(seq_len=seq_len, delay=delay, batch_size=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('cross-entropy')
        plt.title('06 · sparse_matmul recurrent — delayed XOR'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k sparse`
Expected: 1 passed. If compiler warns about `input_w` non-temporal, document
that in the example docstring.

```bash
git add examples/drtrl/06-operator-sparse.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 06-operator-sparse (SparseLinear recurrent + delayed XOR)"
```

---

## Task 8: `07-operator-lora.py` — LoRA adapter on frozen base

**Files:**
- Create: `examples/drtrl/07-operator-lora.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"07-operator-lora.py"`. Run:
`pytest examples/drtrl/tests/test_smoke.py -v -k lora`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `07-operator-lora.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""07 · ``braintrace.lora_matmul`` adapter on a frozen base.

The base recurrent weight is a regular ``brainstate.ParamState`` accessed via
plain ``x @ w`` — therefore NOT part of any ETP primitive, therefore frozen
from D_RTRL's perspective. The LoRA layer uses ``braintrace.lora_matmul``
internally, so only ``lora_a``/``lora_b`` appear in the eligibility trace.

Task: random-frequency sine wave one-step-ahead prediction.
"""
from __future__ import annotations

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
        base = xh @ self.frozen_base.value   # plain matmul — excluded from ETP
        residual = self.lora(xh)             # ETP-aware via lora_matmul
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
    opt = braintools.optim.Adam(5e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online, vmap_states='new')

        def step_loss(inp, tar):
            out = vmap_model(inp)
            return braintools.metric.squared_error(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
        grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
        opt.update(grads)
        return step_losses.mean()

    losses = []
    for _ in range(n_epochs):
        x, y = _shared.make_sine_batch(num_step=num_step, batch_size=batch_size)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('MSE')
        plt.title('07 · LoRA adapter on frozen base — sine'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k lora`
Expected: 1 passed.

```bash
git add examples/drtrl/07-operator-lora.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 07-operator-lora (LoRA adapter on frozen base)"
```

---

## Task 9: `08-operator-conv.py` — Conv1d + MiniGRU

**Files:**
- Create: `examples/drtrl/08-operator-conv.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"08-operator-conv.py"`. Run:
`pytest examples/drtrl/tests/test_smoke.py -v -k conv`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `08-operator-conv.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""08 · ``braintrace.nn.Conv1d`` as an ETP operator.

Each timestep delivers a 28-pixel row of synthetic Poisson rates. A Conv1d
extracts local features; MiniGRU integrates over time; a Linear head
classifies. The Conv kernel is a standard ParamState but routes through
``etp_conv_p`` via ``braintrace.nn.Conv1d``, so it appears in the eligibility
trace.
"""
from __future__ import annotations

import pathlib
import sys

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


def _make_poisson_rows(batch_size: int, n_classes: int = 4, seed: int = 0):
    """Generate synthetic row-streams: each class has a distinct rate profile."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=(batch_size,))
    t, w = 28, 28
    profiles = rng.uniform(0.1, 0.9, size=(n_classes, w)).astype(np.float32)
    rates = profiles[labels]
    stream = rng.poisson(rates[:, None, :].repeat(t, axis=1)).astype(np.float32)
    # (T, B, 1, 28) — channel axis required by Conv1d
    x = jnp.asarray(np.transpose(stream, (1, 0, 2))[:, :, None, :])
    y = jnp.asarray(labels)
    return x, y


class ConvRNN(brainstate.nn.Module):
    def __init__(self, n_hidden: int, n_out: int):
        super().__init__()
        self.conv = braintrace.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding='SAME')
        self.rnn = braintrace.nn.MiniGRU(in_size=8 * 28, out_size=n_hidden)
        self.readout = braintrace.nn.Linear(n_hidden, n_out)

    def update(self, x):
        y = self.conv(x)
        y = y.reshape(y.shape[0], -1) if y.ndim > 2 else y.reshape(-1)
        y = self.rnn(y)
        return self.readout(y)


def main(*, n_epochs: int = 30, batch_size: int = 16, plot: bool = True) -> dict:
    n_classes, n_hidden = 4, 32
    model = ConvRNN(n_hidden, n_classes)
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(3e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vmap_model = brainstate.nn.Vmap(online, vmap_states='new')
        brainstate.transform.for_loop(lambda inp: vmap_model(inp), inputs[:-1])

        def final_loss():
            out = vmap_model(inputs[-1])
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean(), out

        grads, loss, _ = brainstate.transform.grad(final_loss, weights, has_aux=True, return_value=True)()
        opt.update(grads)
        return loss

    losses = []
    for epoch in range(n_epochs):
        x, y = _make_poisson_rows(batch_size, n_classes, seed=epoch)
        losses.append(float(f_train(x, y)))

    if plot:
        plt.plot(losses); plt.xlabel('epoch'); plt.ylabel('cross-entropy')
        plt.title('08 · Conv1d + MiniGRU'); plt.show()
    return {"losses": losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k conv`
Expected: 1 passed. If shape mismatch, inspect `braintrace.nn.Conv1d.__doc__`
and adjust the `reshape` in `update()`.

```bash
git add examples/drtrl/08-operator-conv.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 08-operator-conv (Conv1d + MiniGRU)"
```

---

## Task 10: `09-classification-mnist.py` — flagship LSTM + MNIST

**Files:**
- Create: `examples/drtrl/09-classification-mnist.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize with skip marker + verify skip**

In `test_smoke.py` parametrize list add:

```python
pytest.param("09-classification-mnist.py", marks=pytest.mark.skipif(
    True,
    reason="MNIST example is network-dependent; covered by __main__ only",
)),
```

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k mnist`
Expected: 1 skipped.

- [ ] **Step 2: Implement `09-classification-mnist.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""09 · Row-scan MNIST classification with LSTM.

Flagship example: treats each MNIST image as 28 timesteps × 28 input features
and classifies the digit. Compares D_RTRL and BPTT on matched hyperparams.

Requires ``torchvision``. First run downloads MNIST to ``examples/data/MNIST``.
"""
from __future__ import annotations

import pathlib

import brainstate
import braintools
import jax
import matplotlib.pyplot as plt
import numpy as np

import braintrace


def _load_mnist(batch_size: int):
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    root = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'MNIST'
    root.mkdir(parents=True, exist_ok=True)
    tfm = T.Compose([T.ToTensor(), lambda img: img.squeeze(0).numpy()])
    train = torchvision.datasets.MNIST(str(root), train=True, download=True, transform=tfm)
    test = torchvision.datasets.MNIST(str(root), train=False, download=True, transform=tfm)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)


class LSTMNet(brainstate.nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.cell = braintrace.nn.LSTMCell(n_in, n_hidden)
        self.readout = braintrace.nn.Linear(n_hidden, n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _to_batch(batch):
    import jax.numpy as jnp
    x, y = batch
    x_np = np.stack([np.asarray(img) for img in x], axis=0)  # (B, 28, 28)
    y_np = np.asarray(y)
    return jnp.asarray(np.transpose(x_np, (1, 0, 2))), jnp.asarray(y_np)


def main(*, n_epochs: int = 1, batch_size: int = 64, max_batches: int | None = 100, plot: bool = True) -> dict:
    n_hidden = 64
    model_online = LSTMNet(28, n_hidden, 10)
    model_bptt = LSTMNet(28, n_hidden, 10)
    for (_, a), (_, b) in zip(model_online.states(brainstate.ParamState).items(),
                              model_bptt.states(brainstate.ParamState).items()):
        b.value = jax.tree.map(lambda x: x, a.value)

    w_online = model_online.states(brainstate.ParamState)
    w_bptt = model_bptt.states(brainstate.ParamState)
    opt_online = braintools.optim.Adam(1e-3); opt_online.register_trainable_weights(w_online)
    opt_bptt = braintools.optim.Adam(1e-3); opt_bptt.register_trainable_weights(w_bptt)

    train_loader, _ = _load_mnist(batch_size)

    @brainstate.transform.jit
    def online_step(inputs, targets):
        om = braintrace.D_RTRL(model_online)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model_online)
            om.compile_graph(inputs[0, 0])

        init()
        vm = brainstate.nn.Vmap(om, vmap_states='new')
        brainstate.transform.for_loop(lambda inp: vm(inp), inputs[:-1])

        def final_loss():
            out = vm(inputs[-1])
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean(), out

        grads, loss, _ = brainstate.transform.grad(final_loss, w_online, has_aux=True, return_value=True)()
        opt_online.update(grads)
        return loss

    @brainstate.transform.jit
    def bptt_step(inputs, targets):
        brainstate.nn.init_all_states(model_bptt, batch_size=inputs.shape[1])

        def f_loss():
            out = brainstate.transform.for_loop(model_bptt.update, inputs)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out[-1], targets).mean()

        grads, loss = brainstate.transform.grad(f_loss, w_bptt, return_value=True)()
        opt_bptt.update(grads)
        return loss

    online_losses, bptt_losses = [], []
    for _ in range(n_epochs):
        for i, batch in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = _to_batch(batch)
            online_losses.append(float(online_step(x, y)))
            bptt_losses.append(float(bptt_step(x, y)))

    if plot:
        plt.plot(online_losses, label='D_RTRL')
        plt.plot(bptt_losses, label='BPTT')
        plt.xlabel('batch'); plt.ylabel('cross-entropy')
        plt.legend(); plt.title('09 · row-scan MNIST'); plt.show()

    return {"losses": online_losses, "bptt_losses": bptt_losses}


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add examples/drtrl/09-classification-mnist.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 09-classification-mnist (LSTM, BPTT baseline)"
```

---

## Task 11: `10-char-lm-generation.py` — toy char-level LM

**Files:**
- Create: `examples/drtrl/10-char-lm-generation.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"10-char-lm-generation.py"`. Run:
`pytest examples/drtrl/tests/test_smoke.py -v -k char_lm`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `10-char-lm-generation.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""10 · Toy character-level language model.

Trains a MiniGRU on a short embedded corpus string with D_RTRL and BPTT,
then autoregressively samples a short continuation from each trained model.
"""
from __future__ import annotations

import pathlib
import sys

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
) * 6


class CharRNN(brainstate.nn.Module):
    def __init__(self, vocab_size: int, n_hidden: int):
        super().__init__()
        self.cell = braintrace.nn.MiniGRU(in_size=vocab_size, out_size=n_hidden)
        self.readout = braintrace.nn.Linear(n_hidden, vocab_size)

    def update(self, x):
        return self.readout(self.cell(x))


def _sample(model, vocab: str, prompt: str, length: int) -> str:
    char2idx = {c: i for i, c in enumerate(vocab)}
    brainstate.nn.init_all_states(model)
    out_chars = []
    x = jax.nn.one_hot(jnp.asarray(char2idx.get(prompt[0], 0)), len(vocab))
    for _ in range(length):
        logits = model.update(x)
        probs = jax.nn.softmax(logits)
        next_idx = int(np.argmax(np.asarray(probs)))
        out_chars.append(vocab[next_idx])
        x = jax.nn.one_hot(jnp.asarray(next_idx), len(vocab))
    return ''.join(out_chars)


def main(*, n_epochs: int = 20, batch_size: int = 16, plot: bool = True) -> dict:
    seq_len = 32
    vocab, _, _ = _shared.make_char_batches(CORPUS, seq_len=seq_len, batch_size=batch_size, seed=0)
    vocab_size = len(vocab)

    model_online = CharRNN(vocab_size, n_hidden=64)
    model_bptt = CharRNN(vocab_size, n_hidden=64)
    w_online = model_online.states(brainstate.ParamState)
    w_bptt = model_bptt.states(brainstate.ParamState)
    for (_, a), (_, b) in zip(w_online.items(), w_bptt.items()):
        b.value = jax.tree.map(lambda x: x, a.value)

    opt_online = braintools.optim.Adam(3e-3); opt_online.register_trainable_weights(w_online)
    opt_bptt = braintools.optim.Adam(3e-3); opt_bptt.register_trainable_weights(w_bptt)

    @brainstate.transform.jit
    def online_step(inputs, targets):
        om = braintrace.D_RTRL(model_online)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model_online)
            om.compile_graph(inputs[0, 0])

        init()
        vm = brainstate.nn.Vmap(om, vmap_states='new')

        def step_loss(inp, tar):
            out = vm(inp)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f_grad = brainstate.transform.grad(step_loss, w_online, has_aux=True, return_value=True)
            cur_grads, loss, _ = f_grad(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), loss

        init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in w_online.items()})
        grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
        opt_online.update(grads)
        return step_losses.mean()

    @brainstate.transform.jit
    def bptt_step(inputs, targets):
        brainstate.nn.init_all_states(model_bptt, batch_size=inputs.shape[1])

        def f_loss():
            outs = brainstate.transform.for_loop(model_bptt.update, inputs)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(outs, targets).mean()

        grads, loss = brainstate.transform.grad(f_loss, w_bptt, return_value=True)()
        opt_bptt.update(grads)
        return loss

    online_losses, bptt_losses = [], []
    for epoch in range(n_epochs):
        _, x, y = _shared.make_char_batches(CORPUS, seq_len=seq_len, batch_size=batch_size, seed=epoch)
        online_losses.append(float(online_step(x, y)))
        bptt_losses.append(float(bptt_step(x, y)))

    online_sample = _sample(model_online, vocab, prompt=vocab[0], length=100)
    bptt_sample = _sample(model_bptt, vocab, prompt=vocab[0], length=100)

    if plot:
        plt.plot(online_losses, label='D_RTRL')
        plt.plot(bptt_losses, label='BPTT')
        plt.xlabel('epoch'); plt.ylabel('cross-entropy')
        plt.legend(); plt.title('10 · toy char-LM'); plt.show()

    return {
        "losses": online_losses,
        "bptt_losses": bptt_losses,
        "samples": {"online": online_sample, "bptt": bptt_sample},
    }


if __name__ == "__main__":
    out = main()
    print("[online] ", out["samples"]["online"])
    print("[bptt]   ", out["samples"]["bptt"])
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k char_lm`
Expected: 1 passed.

```bash
git add examples/drtrl/10-char-lm-generation.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 10-char-lm-generation (flagship, BPTT baseline)"
```

---

## Task 12: `11-knob-fast-solve.py` — fast_solve equivalence + speed

**Files:**
- Create: `examples/drtrl/11-knob-fast-solve.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"11-knob-fast-solve.py"`. Relax the smoke test assertion
(`assert "losses" in result` — no length check) to accept empty lists.
Run: `pytest examples/drtrl/tests/test_smoke.py -v -k fast_solve`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `11-knob-fast-solve.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""11 · ``fast_solve=True`` vs ``fast_solve=False``.

Demonstrates:
  * numerical equivalence (allclose on summed gradients)
  * wall-clock speedup from the einsum fast path

The fast path applies when every ETP primitive in the graph has an
elementwise ``yw_to_w`` rule (matmul, mv, element_wise). For this example
(ValinaRNN + Linear) all primitives qualify.
"""
from __future__ import annotations

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
    @brainstate.transform.jit
    def f_grad(inputs, targets):
        online = braintrace.D_RTRL(model, fast_solve=fast_solve)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vm = brainstate.nn.Vmap(online, vmap_states='new')

        def step_loss(inp, tar):
            out = vm(inp)
            return braintools.metric.squared_error(out, tar).mean(), out

        def grad_step(prev_grads, pair):
            inp, tar = pair
            f = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
            cur, l, _ = f(inp, tar)
            return jax.tree.map(lambda a, b: a + b, prev_grads, cur), l

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

    print(f"fast_solve=True  mean time/epoch: {fast_time*1000:.2f} ms")
    print(f"fast_solve=False mean time/epoch: {slow_time*1000:.2f} ms")
    print(f"max |grad_fast - grad_slow|     : {max_diff:.3e}")
    print(f"allclose (atol=1e-5, rtol=1e-4) : {allclose}")

    if plot:
        plt.bar(['fast_solve=True', 'fast_solve=False'], [fast_time * 1000, slow_time * 1000])
        plt.ylabel('ms / epoch')
        plt.title(f'11 · fast_solve runtime (max-grad-diff {max_diff:.1e})'); plt.show()

    return {
        "losses": [],
        "fast_time_ms": fast_time * 1000,
        "slow_time_ms": slow_time * 1000,
        "max_grad_diff": max_diff,
        "allclose": allclose,
    }


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k fast_solve`
Expected: 1 passed.

```bash
git add examples/drtrl/11-knob-fast-solve.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 11-knob-fast-solve (equivalence + speed)"
```

---

## Task 13: `12-knob-normalize-spectrum.py` — stability knob

**Files:**
- Create: `examples/drtrl/12-knob-normalize-spectrum.py`
- Modify: `examples/drtrl/tests/test_smoke.py`

- [ ] **Step 1: Add parametrize + verify failure**

Add `"12-knob-normalize-spectrum.py"`. Run:
`pytest examples/drtrl/tests/test_smoke.py -v -k normalize_spectrum`
Expected: FileNotFoundError.

- [ ] **Step 2: Implement `12-knob-normalize-spectrum.py`**

```python
# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""12 · ``normalize_matrix_spectrum``: trace stability knob.

Intentionally starts with a spectrally unstable recurrent weight (scale 1.5)
and shows three training curves:

  * D_RTRL, normalize_matrix_spectrum=False   (baseline — may diverge)
  * D_RTRL, normalize_matrix_spectrum=True    (branch-free trace clip)
  * BPTT                                       (true-gradient reference)

Task: delayed XOR (short sequence, still long enough to surface instability).
"""
from __future__ import annotations

import pathlib
import sys

import brainstate
import braintools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import braintrace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class UnstableRNN(brainstate.nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, scale: float):
        super().__init__()
        self.cell = braintrace.nn.ValinaRNNCell(
            in_size=n_in, out_size=n_hidden,
            w_init=lambda shape, dtype: scale * jax.random.normal(jax.random.PRNGKey(0), shape, dtype) / np.sqrt(shape[-2]),
            activation='tanh',
        )
        self.readout = braintrace.nn.Linear(n_hidden, n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _clone_weights(src, dst):
    for (_, a), (_, b) in zip(src.states(brainstate.ParamState).items(),
                              dst.states(brainstate.ParamState).items()):
        b.value = jax.tree.map(lambda x: x, a.value)


def _online_train(model, weights, iter_batches, *, normalize: bool, n_epochs: int):
    opt = braintools.optim.Adam(3e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        online = braintrace.D_RTRL(model, normalize_matrix_spectrum=normalize)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=inputs.shape[1])
        def init():
            brainstate.nn.init_all_states(model)
            online.compile_graph(inputs[0, 0])

        init()
        vm = brainstate.nn.Vmap(online, vmap_states='new')
        brainstate.transform.for_loop(lambda inp: vm(inp), inputs[:-1])

        def final_loss():
            out = vm(inputs[-1])
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out, targets).mean(), out

        grads, loss, _ = brainstate.transform.grad(final_loss, weights, has_aux=True, return_value=True)()
        opt.update(grads)
        return loss

    losses = []
    for x, y in iter_batches(n_epochs):
        losses.append(float(f_train(x, y)))
    return losses


def _bptt_train(model, weights, iter_batches, *, n_epochs: int):
    opt = braintools.optim.Adam(3e-3); opt.register_trainable_weights(weights)

    @brainstate.transform.jit
    def f_train(inputs, targets):
        brainstate.nn.init_all_states(model, batch_size=inputs.shape[1])

        def f_loss():
            out = brainstate.transform.for_loop(model.update, inputs)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(out[-1], targets).mean()

        grads, loss = brainstate.transform.grad(f_loss, weights, return_value=True)()
        opt.update(grads)
        return loss

    losses = []
    for x, y in iter_batches(n_epochs):
        losses.append(float(f_train(x, y)))
    return losses


def main(*, n_epochs: int = 30, batch_size: int = 32, plot: bool = True) -> dict:
    seq_len, delay, n_hidden = 16, 8, 32

    def iter_batches(n):
        for i in range(n):
            yield _shared.make_xor_batch(seq_len=seq_len, delay=delay, batch_size=batch_size, seed=i)

    model_base = UnstableRNN(2, n_hidden, 2, scale=1.5)
    model_fix = UnstableRNN(2, n_hidden, 2, scale=1.5)
    model_bptt = UnstableRNN(2, n_hidden, 2, scale=1.5)
    _clone_weights(model_base, model_fix)
    _clone_weights(model_base, model_bptt)

    loss_base = _online_train(model_base, model_base.states(brainstate.ParamState), iter_batches, normalize=False, n_epochs=n_epochs)
    loss_fix = _online_train(model_fix, model_fix.states(brainstate.ParamState), iter_batches, normalize=True, n_epochs=n_epochs)
    loss_bptt = _bptt_train(model_bptt, model_bptt.states(brainstate.ParamState), iter_batches, n_epochs=n_epochs)

    if plot:
        plt.plot(loss_base, label='D_RTRL (normalize=False)')
        plt.plot(loss_fix, label='D_RTRL (normalize=True)')
        plt.plot(loss_bptt, label='BPTT')
        plt.xlabel('epoch'); plt.ylabel('cross-entropy')
        plt.legend(); plt.title('12 · normalize_matrix_spectrum — delayed XOR'); plt.show()

    return {
        "losses": loss_fix,
        "baseline_losses": loss_base,
        "bptt_losses": loss_bptt,
    }


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke + commit**

Run: `pytest examples/drtrl/tests/test_smoke.py -v -k normalize_spectrum`
Expected: 1 passed.

```bash
git add examples/drtrl/12-knob-normalize-spectrum.py examples/drtrl/tests/test_smoke.py
git commit -m "examples(drtrl): add 12-knob-normalize-spectrum (stability)"
```

---

## Task 14: `examples/drtrl/README.md`

**Files:**
- Create: `examples/drtrl/README.md`

- [ ] **Step 1: Write README**

Write `examples/drtrl/README.md`:

````markdown
# D_RTRL Examples

A tutorial-linear walk through ``braintrace.D_RTRL`` — the online
eligibility-trace gradient estimator.

Each file is self-contained. Read them in order (01 → 12) to follow the
companion tutorial at `docs/tutorials/drtrl.md`.

## How to run

```bash
python examples/drtrl/01-basics-integrator.py
```

All examples run on CPU in roughly 1–2 minutes. Task 09 (MNIST) requires
`torchvision` and downloads ~11 MB on first run.

## Axis map

| Axis                                       | Files              |
|--------------------------------------------|--------------------|
| Operator (matmul / sparse / LoRA / conv)   | 06, 07, 08         |
| Target (regression / classification / ...) | 01, 04, 05, 09, 10 |
| Batching mode                              | 02, 03             |
| vjp method                                 | 04, 05             |
| fast_solve knob                            | 11                 |
| normalize_matrix_spectrum knob             | 12                 |
| BPTT baseline                              | 01, 09, 10, 12     |

## Tutorial

See `docs/tutorials/drtrl.md`.

## Tests

```bash
pytest examples/drtrl/tests -v
```
````

- [ ] **Step 2: Commit**

```bash
git add examples/drtrl/README.md
git commit -m "examples(drtrl): add README with axis map"
```

---

## Task 15: `docs/tutorials/drtrl.md` — full tutorial

**Files:**
- Create: `docs/tutorials/drtrl.md`

- [ ] **Step 1: Write tutorial**

Write `docs/tutorials/drtrl.md`:

````markdown
# D_RTRL: Online Gradient Learning via Eligibility Traces

This tutorial walks through
[`braintrace.D_RTRL`](../../braintrace/_etrace_vjp/d_rtrl.py), the online
eligibility-trace gradient estimator shipped with BrainTrace, using the
numbered examples under `examples/drtrl/`.

## 1. What is D-RTRL

D-RTRL is an online approximation of Real-Time Recurrent Learning (RTRL). It
maintains a per-parameter *eligibility trace* ε^t estimating how sensitive
the current hidden state is to each trainable parameter:

```
ε^t  ≈  D^t · ε^{t-1}  +  diag(D_f^t) ⊗ x^t
∇θ L =  Σ_t  ∂L^t/∂h^t ∘ ε^t
```

Compared with BPTT, D-RTRL:
- does not require storing the whole trajectory
- runs online: no backward pass through time
- pays with approximation error (drops off-diagonal Jacobian coupling)
- pays with memory: one trace tensor per ETP parameter

## 2. Mental model

- **Hidden state**: any `brainstate.HiddenState` on a cell is tracked.
- **ETP primitive**: `matmul`, `sparse_matmul`, `lora_matmul`, `conv`,
  `element_wise`. Plain `x @ w` is **excluded** — use it to freeze.
- **compile_graph**: one jaxpr walk discovers ETP uses, builds the trace graph.
- **Jacobian rules**: per-primitive `yw_to_w`, `xy_to_dw`, `init_drtrl`.

## 3. Minimal example — integrator

[`01-basics-integrator.py`](../../examples/drtrl/01-basics-integrator.py).
Path `ValinaRNNCell → Linear`. Both cells use `braintrace.matmul` internally,
so both parameter sets end up in the trace.

## 4. Batching patterns

- [`02-batching-vmap.py`](../../examples/drtrl/02-batching-vmap.py) — per-sample
  init + compile in a `vmap_new_states` scope, then `brainstate.nn.Vmap`.
  **Default choice.**
- [`03-batching-batched.py`](../../examples/drtrl/03-batching-batched.py) —
  `D_RTRL(..., mode=brainstate.mixin.Batching())`, init once with
  `batch_size=...`. Pick when the algorithm needs the batch axis exposed.

## 5. VJP methods

- [`04-vjp-single-step.py`](../../examples/drtrl/04-vjp-single-step.py) —
  `vjp_method='single-step'`. Cheapest, most biased.
- [`05-vjp-multi-step.py`](../../examples/drtrl/05-vjp-multi-step.py) —
  `vjp_method='multi-step'`. More expensive, less biased.

## 6. Choosing an operator

| Operator | User-facing function | Example |
|---|---|---|
| Matmul | `braintrace.matmul` | 01–05, 10 |
| Elementwise identity | `braintrace.element_wise` | internal |
| Convolution | `braintrace.conv`, `braintrace.nn.Conv1d` | 08 |
| Sparse matmul | `braintrace.sparse_matmul`, `braintrace.nn.SparseLinear` | 06 |
| LoRA matmul | `braintrace.lora_matmul`, `braintrace.nn.LoRA` | 07 |

## 7. Target types

| Target | Loss | Example |
|---|---|---|
| Regression | `braintools.metric.squared_error` | 01–03, 07, 12 |
| Classification | `softmax_cross_entropy_with_integer_labels` | 04–06, 08, 09 |
| Autoregressive generation | next-token cross-entropy | 10 |

## 8. Performance + stability knobs

- **`fast_solve` (default True)** — einsum fast path for mm/mv/elemwise
  primitives. Conv/sparse/LoRA always legacy. See
  [`11-knob-fast-solve.py`](../../examples/drtrl/11-knob-fast-solve.py).
- **`normalize_matrix_spectrum` (default False)** — branch-free
  `v / max(|v|_max, 1)` clip on the trace each step. Turn on if training
  diverges with spectral radius above 1. See
  [`12-knob-normalize-spectrum.py`](../../examples/drtrl/12-knob-normalize-spectrum.py).
- **`num_state == 1` shortcut** — automatic; drops the state-axis einsum
  for single-state hidden groups.

## 9. Limitations

- **Approximation error** — D-RTRL drops off-diagonal Jacobian terms; on
  strongly coupled recurrences the gradient diverges from true RTRL / BPTT.
- **Memory** — trace shape `(batch, in, out, num_state)` scales as
  `O(param_count × hidden_count)`. Infeasible for very wide layers.
- **Primitive coverage** — only the ETP primitives are traced. Plain
  `x @ w` is intentionally excluded, giving explicit control over what
  participates.
- **No `weight → weight → hidden` chains** — only one trainable ETP
  primitive may sit between input and hidden state. Intermediate primitives
  must be gradient-enabled (today only `element_wise` qualifies). In
  `GRUCell` this means `Wr` is excluded because its output reaches `h` only
  through `Wh`'s matmul.
- **Batching-mode split** — `brainstate.mixin.Batching()` vs
  `vmap_new_states` have different semantics; pick one per model.
- **Spectral clip off by default** — numerical stability is your
  responsibility unless `normalize_matrix_spectrum=True`.
- **`multi-step` vs `single-step`** — speed/bias tradeoff. No free lunch.
- **Long-horizon credit assignment** — traces still decay exponentially; very
  long dependencies suffer.
- **No higher-order gradients** — D-RTRL uses `custom_vjp` at the primal
  level. `jax.grad(jax.grad(...))` through `D_RTRL` is not supported.

## 10. FAQ / troubleshooting

- **NaN traces** → spectral radius likely above 1. Set
  `normalize_matrix_spectrum=True` or rescale the recurrent init.
- **`compile_graph` fails "no hidden state reachable"** → the weight is used
  only through plain `x @ w` or the ETP primitive does not reach any
  `HiddenState`.
- **Loss diverges fast** → check `vjp_method='multi-step'` window fits the
  sequence length, and no outer `jax.lax.scan` is re-init'ing state each step.
- **Conv1d shape error** → Conv1d wants a channel axis. Pre-flatten features
  before the RNN.

## 11. API reference pointers

- [`braintrace.D_RTRL`](../../braintrace/_etrace_vjp/d_rtrl.py) — alias for
  `ParamDimVjpAlgorithm`.
- [`braintrace.ES_D_RTRL` / `braintrace.pp_prop`](../../braintrace/_etrace_vjp/pp_prop.py)
  — the I/O-dimension variant (separate future tutorial).
- [`braintrace.compile_etrace_graph`](../../braintrace/_etrace_compiler/graph.py)
  — called internally by `D_RTRL.compile_graph`.
- [`braintrace.register_primitive`](../../braintrace/_etrace_op/_primitive.py)
  — add your own ETP primitive.
- [`CLAUDE.md`](../../CLAUDE.md) — architectural overview of BrainTrace.
````

- [ ] **Step 2: Commit**

```bash
git add docs/tutorials/drtrl.md
git commit -m "docs(tutorial): add D_RTRL tutorial with limitations"
```

---

## Task 16: Full smoke-test verification + plan sign-off

**Files:**
- Modify: none (verification only)

- [ ] **Step 1: Run the entire D-RTRL smoke suite**

Run: `pytest examples/drtrl/tests -v --timeout=300`
Expected: all non-skip examples pass (01–08, 10, 11, 12) + MNIST skipped.

- [ ] **Step 2: Run the full braintrace test suite (regression check)**

Run: `pytest braintrace -v`
Expected: all existing tests still pass.

- [ ] **Step 3: Verify branch is clean**

Run: `git status`
Expected: nothing to commit, working tree clean.

Run: `git log --oneline -18`
Inspect: see the task's worth of commits in order.

---

## Self-Review Results

**Spec coverage audit:**

| Spec item | Task(s) |
|---|---|
| 12 quick-running examples | 2–13 |
| BPTT baselines (01, 09, 10, 12) | 2, 10, 11, 13 |
| Shared data generators | 1 |
| Tutorial with limitations | 15 |
| Operator axis (sparse/LoRA/conv) | 7, 8, 9 |
| Target axis (reg/cls/copy/gen) | 2, 5, 6, 10, 11 |
| Batching axis | 3, 4 |
| vjp method axis | 5, 6 |
| `fast_solve` knob | 12 |
| `normalize_matrix_spectrum` knob | 13 |
| Plots guarded by `__main__` | every task |
| Smoke tests | every task adds to `test_smoke.py` |
| `torchvision` optional | 10 skips in CI |
| README index | 14 |

**Placeholder scan:** all code blocks complete; no TBD/TODO.

**Type consistency:** `main(*, n_epochs, batch_size, plot) -> dict` is
identical across all 12 examples. Weight handles always
`model.states(brainstate.ParamState)`. Loss via `braintools.metric`.

**Known wobbles to watch during execution:**

1. `braintrace.nn.Conv1d` I/O shape — if the Task 9 reshape fails, inspect
   `braintrace/nn/_conv.py`.
2. `saiunit.sparse.COO` constructor — Task 7 uses `(data, indices)` plus
   `shape=`. If `saiunit` version differs, adjust kwargs to match.
3. `_load_mnist` path in Task 10 — `torchvision.datasets.MNIST` takes `str`
   `root`.

If any wobble surfaces, fix inline — do not abandon the task.
