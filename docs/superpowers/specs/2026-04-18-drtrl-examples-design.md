# D-RTRL Examples + Tutorial — Design Spec

**Date:** 2026-04-18
**Author:** brainstorming session
**Status:** approved (awaiting written-spec review)

## Purpose

Add diverse `D_RTRL` example programs under `examples/drtrl/` and one companion
tutorial under `docs/tutorials/drtrl.md`. Together they demonstrate D-RTRL
across the library's ETP operator set, batching patterns, vjp methods, and
training targets, and teach the reader how to apply and reason about D-RTRL
(including its limitations) in their own models.

## Goals

- 10 quick-running example programs (~1-2 min CPU runtime each) covering
  different operators, targets, and batching / vjp patterns.
- 3 of the examples carry a BPTT baseline for comparison.
- 1 tutorial walking through the same examples chapter-by-chapter, including
  an explicit limitations section.
- Examples organized in tutorial-linear reading order so the tutorial reader
  follows numbered files from 01 → 10.
- Plots guarded by `if __name__ == "__main__":`; plotting never blocks CI.

## Non-Goals

- Not a benchmark suite. Examples illustrate usage, not raw numbers.
- Not paper-quality reproductions. Small epoch counts, small models.
- Not covering `ES_D_RTRL` / `pp_prop` (`IODimVjpAlgorithm`) — separate future
  spec.
- Not rewriting the two existing top-level example files (`100-*`, `101-*`).
  They remain as historical references.

## Directory Layout

```
examples/drtrl/
├── README.md                     Index + how to run
├── _shared.py                    Common data generators + training helpers
├── 01-basics-integrator.py       Minimal D_RTRL loop, BPTT baseline
├── 02-batching-vmap.py           vmap_new_states pattern (common)
├── 03-batching-batched.py        brainstate.mixin.Batching pattern
├── 04-vjp-single-step.py         vjp_method='single-step', GRU on copy task
├── 05-vjp-multi-step.py          vjp_method='multi-step', GRU on copy task
├── 06-operator-sparse.py         braintrace.sparse_matmul as recurrent op
├── 07-operator-lora.py           braintrace.lora_matmul adapter on frozen base
├── 08-operator-conv.py           braintrace.conv over input rows + MiniGRU
├── 09-classification-mnist.py    LSTM + row-scan MNIST, BPTT baseline
└── 10-char-lm-generation.py      Autoregressive toy char-LM, BPTT baseline

docs/tutorials/drtrl.md           Tutorial + limitations
```

## Per-File Specification

### `_shared.py`

Shared utilities only — not a framework. Keep small.

- Data generators:
  - `make_integrator_batch(num_step, num_batch)` — noisy signal → cumulative sum.
  - `make_copy_batch(time_lag, batch_size)` — copying memory task.
  - `make_xor_batch(seq_len, delay, batch_size)` — delayed binary XOR.
  - `make_sine_batch(num_step, batch_size)` — sine wave of random frequency.
  - `make_char_batches(text, seq_len, batch_size)` — character-level sequences.
  - `load_mnist_rows(batch_size)` — torchvision MNIST as 28 time-steps × 28 pixels.
- Training helpers (thin wrappers):
  - `online_train(model, weights, opt, data_iter, n_steps, loss_fn)`
  - `bptt_train(model, weights, opt, data_iter, n_steps, loss_fn)`
- No plotting helpers — each example handles its own figure.

### `01-basics-integrator.py`

- Task: regression, cumulative-sum of noisy input.
- Model: `ValinaRNNCell` + `braintrace.nn.Linear` readout.
- Pattern: `braintrace.D_RTRL(model)` + `vmap_new_states` for per-sample init.
- Loss: MSE.
- BPTT baseline: yes.
- Annotations: heavy inline comments — smallest working D_RTRL example.

### `02-batching-vmap.py`

- Task: same as 01 (integrator).
- Pattern: explicit `braintrace.D_RTRL(model)` → `vmap_new_states` →
  `brainstate.nn.Vmap(online_model, vmap_states='new')`.
- Comments: why per-sample `init_all_states` + `compile_graph`, then vmap.
- No BPTT.

### `03-batching-batched.py`

- Task: same as 01.
- Pattern: `ParamDimVjpAlgorithm(model, mode=brainstate.mixin.Batching())`,
  single `init_all_states(batch_size=...)`, compile on batched sample.
- Comments: contrast with 02 — when `Batching()` is the right pick.
- No BPTT.

### `04-vjp-single-step.py`

- Task: copying memory task (10 symbols, tunable lag).
- Model: `braintrace.nn.GRUCell` + `braintrace.nn.Linear` readout.
- Algorithm: `D_RTRL(model, vjp_method='single-step')`.
- Loss: softmax cross-entropy on copy phase only.
- No BPTT (paired with 05).

### `05-vjp-multi-step.py`

- Same task + model as 04, `vjp_method='multi-step'`.
- Docstring: speed / accuracy tradeoff vs single-step.

### `06-operator-sparse.py`

- Task: delayed XOR (synthetic, short sequence).
- Model: custom `brainstate.nn.RNNCell` subclass using
  `braintrace.sparse_matmul` for the recurrent weight (sparse mask stored as
  `brainstate.ParamState` data + static indices).
- Algorithm: `D_RTRL(model)`.
- Comments: how to declare a sparse ETP parameter; shape notes.

### `07-operator-lora.py`

- Task: sine-wave continuation (regression).
- Model: frozen full-rank recurrent `Linear` (`brainstate.ParamState`, excluded
  from ETP by using plain `x @ w`), plus `braintrace.lora_matmul(..., alpha=...)`
  adapter on the hidden update — adapter `A`, `B` are the only trainable weights.
- Algorithm: `D_RTRL(model)`; only LoRA params appear in ETP graph.
- Comments: why frozen base uses plain matmul, why adapter uses ETP matmul.

### `08-operator-conv.py`

- Task: row-scan Poisson-MNIST-like spike stream (synthetic Poisson rates per
  row to avoid dataset download here; keep torchvision for example 09).
- Model: `braintrace.nn.Conv1d` over each timestep's row → `MiniGRU` →
  `Linear` readout.
- Algorithm: `D_RTRL(model)` — Conv uses `etp_conv_p`.
- Comments: conv kernel as ETP parameter; how `batched=True` dispatches.

### `09-classification-mnist.py` (flagship)

- Task: row-scan MNIST — 28 time-steps × 28 input features, 10-class output.
- Data: torchvision MNIST (download cache under `examples/data/MNIST`).
- Model: `LSTMCell` + `braintrace.nn.Linear` readout over final hidden state.
- Algorithm: `D_RTRL(model)` and BPTT baseline side-by-side.
- Metrics: train loss + test accuracy curves plotted.

### `10-char-lm-generation.py` (flagship)

- Task: toy char-level language model on a small embedded corpus string
  (e.g. a paragraph of public-domain text), autoregressive generation after
  training.
- Model: `braintrace.nn.MiniGRU` + `Linear` readout over vocab.
- Algorithm: `D_RTRL(model)` and BPTT baseline.
- After training: show 200-char sampled continuation from each model.

### `README.md`

- Summary of D_RTRL and when it helps.
- Table mapping axis → file:
  | Axis            | Files |
  |-----------------|-------|
  | Operator        | 06, 07, 08 |
  | Target          | 01, 04, 05, 09, 10 |
  | Batching mode   | 02, 03 |
  | vjp method      | 04, 05 |
  | BPTT baseline   | 01, 09, 10 |
- How to run.
- Link to tutorial.

## Tutorial (`docs/tutorials/drtrl.md`)

Chapter structure:

1. **What is D-RTRL** — math sketch
   `ε^t ≈ D^t · ε^{t-1} + diag(D_f^t) ⊗ x^t`; per-parameter trace; online vs
   BPTT tradeoff.
2. **Mental model** — hidden states, ETP primitives, `compile_graph`, jacobian rules.
3. **Minimal example** → points to `01-basics-integrator.py`.
4. **Batching patterns** → `02-batching-vmap.py` vs `03-batching-batched.py`;
   selection criteria.
5. **vjp methods** → `04-vjp-single-step.py` vs `05-vjp-multi-step.py`;
   speed / accuracy tradeoff.
6. **Choosing an operator** — table: `matmul`, `sparse_matmul`, `lora_matmul`,
   `conv`, `element_wise` with example file refs.
7. **Target types** — regression, classification, autoregressive generation;
   loss and readout patterns.
8. **Performance knobs** — `fast_solve`, `normalize_matrix_spectrum`,
   `num_state==1` shortcut.
9. **Limitations** (explicit):
   - Approximation error — D-RTRL drops off-diagonal jacobian terms; diverges
     from true RTRL on strongly coupled recurrences.
   - Per-parameter trace memory — `ε` shape `(batch, in, out, num_state)` grows
     as `O(param_count × hidden_count)`; infeasible for very wide layers.
   - ETP primitive coverage — only `matmul`, `element_wise`, `conv`,
     `sparse_matmul`, `lora_matmul` are traced; plain `x @ w` is excluded
     intentionally.
   - No `weight → weight → hidden` chains — only one trainable ETP primitive
     may sit between the input and the hidden state. Tail primitives must be
     gradient-enabled (today only `element_wise` is). Consequence: in `GRUCell`
     `Wr` is excluded because its output reaches `h` only through another ETP
     matmul (`Wh`).
   - Batching mode split — `brainstate.mixin.Batching()` vs `vmap_new_states`
     have different semantics; user must pick one per model.
   - Spectral-normalization flag off by default — numerical stability is the
     user's responsibility unless `normalize_matrix_spectrum=True`.
   - `multi-step` vjp more accurate but slower; `single-step` cheaper but
     biased.
   - Long-horizon credit assignment still limited by the same
     exponential-decay issue as true RTRL (eligibility traces decay).
   - No higher-order gradients through `D_RTRL` (implemented with
     `custom_vjp` at the primal level).
10. **FAQ / troubleshooting** — NaN traces, `compile_graph` failures,
    divergence patterns.
11. **API reference pointers** — links to `ParamDimVjpAlgorithm`, `D_RTRL`
    alias, `compile_graph`, public ETP primitive constructors.

## Execution Constraints

- Every example must run on CPU in ~1-2 minutes; enforce small epoch counts,
  small hidden sizes, small batch sizes.
- No downloads inside import paths; MNIST download only when
  `09-classification-mnist.py` is run as `__main__`.
- Every example guarded by `if __name__ == "__main__":` around plotting.
- All plots use matplotlib only; no seaborn, no plotly.
- No logging library required — plain `print` for loss output matches existing
  `examples/100-*.py` and `examples/101-*.py` conventions.

## Testing

- Smoke test per example: add one pytest that imports each example module and
  calls its `main()` with a tiny override (e.g. `n_epochs=1`, `batch_size=4`)
  — covered by a single parametrized test file
  `examples/drtrl/tests/test_smoke.py` (OK for tests to live adjacent since
  this is an examples package, not a library source module).
- Tutorial is prose — no automated test.
- No coverage target for `examples/` (example code is demonstrative).

## Dependencies

- Existing: `brainstate`, `braintools`, `braintrace`, `jax`, `saiunit`, `numpy`,
  `matplotlib`.
- New: `torchvision` (only for `09-classification-mnist.py`). Document in the
  example's docstring and `examples/drtrl/README.md`.

## Open Risks

- `GRUCell` 2-of-3 ETP relation pitfall already documented in `CLAUDE.md`;
  tutorial must reference it correctly. If the 06/07/08 examples exhibit a
  similar compile-time warning, surface it in the example docstring.
- `sparse_matmul` and `lora_matmul` examples require some custom cell
  construction. If the existing `braintrace.nn` library does not already
  expose a convenient sparse/LoRA cell, design the example to subclass
  `brainstate.nn.RNNCell` and show the construction pattern — that is itself
  instructive.
- Torch/torchvision availability on reader machines — document cleanly, fall
  back gracefully if download fails.

## Out of Scope (explicit)

- Rewriting `examples/100-gru-on-copying-task.py` or `examples/101-integrator-rnn.py`.
- Benchmarking vs BPTT with published numbers.
- `ES_D_RTRL` / `pp_prop` example suite.
- Sphinx integration of the tutorial.
