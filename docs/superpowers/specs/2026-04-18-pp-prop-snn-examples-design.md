# pp_prop SNN Examples — Design Spec

**Date:** 2026-04-18
**Status:** Approved (brainstorm phase)
**Scope:** New tutorial-linear example series for `braintrace.pp_prop` (aka
`ES_D_RTRL` / `IODimVjpAlgorithm`) demonstrating its use across diverse
spiking neural networks, operators, training targets, and algorithm
configurations.

## 1. Goal

Provide a parallel to `examples/drtrl/` but focused on spiking networks and
the `pp_prop` algorithm. Each example is self-contained, CPU-only,
reproducible in under two minutes, and isolates one variation along a single
axis. Together the series covers neuron-model diversity, batching modes,
VJP methods, operator choices, training-target diversity, a BPTT
side-by-side flagship, and one algorithm-knob sweep.

Non-goals:
- No GPU, no tonic/torchvision, no NMNIST/DVS/SHD datasets.
- No modification to existing `examples/000-004-*.py` (they remain as legacy
  pp_prop SNN examples).
- No new ETP primitives or changes to `braintrace/_etrace_vjp/pp_prop.py`.
- No re-demonstration of `fast_solve` / `normalize_matrix_spectrum` knobs
  (already covered in `examples/drtrl/11`, `12`). README cross-links.

## 2. Directory layout

```
examples/pp_prop/
├── __init__.py                             # empty
├── _shared.py                              # SNN data gens + cells + helpers
├── README.md                               # axis map + run instructions
├── 01-basics-lif-integrator.py
├── 02-neurons-alif-dms.py
├── 03-neurons-gif-working-memory.py
├── 04-neurons-coba-ei-rsnn.py
├── 05-batching-vmap.py
├── 06-batching-batched.py
├── 07-vjp-single-step.py
├── 08-vjp-multi-step.py
├── 09-operator-sparse.py
├── 10-operator-lora.py
├── 11-operator-conv.py
├── 12-classification-neuromorphic.py       # flagship, BPTT side-by-side
├── 13-knob-decay-vs-rank.py
├── 14-knob-vjp-method-contrast.py
└── tests/
    └── test_smoke.py
docs/tutorials/pp_prop.md                   # long-form companion
```

## 3. `_shared.py` — shared utilities (≤ 350 lines)

### Data generators (numpy, deterministic via seed)
- `make_integrator_spikes(num_step, num_batch, rate_hz, dt, seed)` →
  `(xs, ys)` — Poisson inputs + cumulative-rate target. Used by 01, 07, 13.
- `make_dms_spikes(num_step, num_batch, n_in, fr_hz, dt, seed)` →
  `(xs, ys)` — delayed match-to-sample binary labels. Used by 02, 08, 14.
- `make_memory_pattern(num_step, num_batch, n_in, cue_frac, seed)` →
  `(xs, ys)` — cue/recall working memory. Used by 03.
- `make_poisson_mnist(num_step, num_batch, rate_hz, dt, digits, seed)` →
  `(xs, ys)` — Poisson-encoded sklearn 8×8 digits. Used by 04, 09, 10, 11,
  12.

### SNN cells (thin wrappers; neurons from `brainpy.state`)
- `LIFCell(n_in, n_rec, tau_mem, V_th, ff_scale, rec_scale)` — recurrent
  block: concat(input, last-spike) → `braintrace.nn.Linear` → `Expon` →
  `LIF`. Returns spikes.
- `ALIFCell(...)` — same shape, `brainpy.state.ALIF` (adaptive threshold).
- `GIFCell(...)` — same shape, `brainpy.state.GIF` with heterogeneous
  `tau_I2` (fallback: local GIF port from `examples/snn_models.py` if
  `brainpy.state.GIF` lacks per-neuron tau).
- `COBAEICell(n_in, n_exc, n_inh, ...)` — Dale-law E/I recurrent block via
  `braintrace.nn.SignedWLinear` (fallback: regular `Linear` with signed
  init if `SignedWLinear` not wrapped by ETP primitives).
- `LeakyReadout(n_rec, n_out, tau_o)` — thin wrapper around
  `braintrace.nn.LeakyRateReadout`.

### Trainer helpers
- `online_train_step(model, weights, opt, inputs, targets, decay_or_rank,
  vjp_method='single-step')` — jitted scan over time, returns mean loss.
- `bptt_train_step(model, weights, opt, inputs, targets)` — jitted BPTT
  baseline. Used by 12, 14.
- `plot_loss_curve(losses, title, save_path=None)` — matplotlib helper.

### Config
- `DEFAULT_DT = 1. * u.ms`
- `DEFAULT_SEED = 42`
- `set_env(dt=DEFAULT_DT)` context wrapper (thin around
  `brainstate.environ.context`).

## 4. File specs

Every file shares the following contract:
- Copyright header + 1-paragraph module docstring stating intent.
- `if __name__ == '__main__':` guard.
- CPU only, <2 min runtime.
- Reads `PP_PROP_NO_PLOT` env var → skip matplotlib when set.
- Reads `PP_PROP_SMOKE` env var → shrink `n_epochs=1`, `num_batch=2`,
  `num_step=10` for CI.
- Algorithm entry: `braintrace.IODimVjpAlgorithm(model, decay_or_rank=...,
  vjp_method=...)`.

| # | File | Cell/Op | Data | Algo config | Baseline | ~LoC |
|---|------|---------|------|-------------|----------|------|
| 01 | `01-basics-lif-integrator.py` | LIFCell + Linear | `make_integrator_spikes` | `decay=0.95`, single-step | — | 120 |
| 02 | `02-neurons-alif-dms.py` | ALIFCell | `make_dms_spikes` | `decay=0.97`, single-step | — | 140 |
| 03 | `03-neurons-gif-working-memory.py` | GIFCell (het. tau_I2) | `make_memory_pattern` | `decay=0.98` | — | 160 |
| 04 | `04-neurons-coba-ei-rsnn.py` | COBAEICell (Dale's law) | `make_poisson_mnist` (tiny) | `decay=0.95` | — | 180 |
| 05 | `05-batching-vmap.py` | LIFCell | integrator | `Vmap(model, vmap_states='new')` | — | 130 |
| 06 | `06-batching-batched.py` | LIFCell | integrator | batched primitive, no vmap | 05 cross-ref | 130 |
| 07 | `07-vjp-single-step.py` | LIFCell | integrator | `vjp_method='single-step'` | — | 130 |
| 08 | `08-vjp-multi-step.py` | LIFCell | DMS | `vjp_method='multi-step'` | 07 cross-ref | 160 |
| 09 | `09-operator-sparse.py` | LIFCell + `braintrace.sparse_matmul` | Poisson-MNIST | `decay=0.95` | dense cross-ref | 170 |
| 10 | `10-operator-lora.py` | LIFCell + `braintrace.lora_matmul` recurrence | Poisson-MNIST | `decay=0.95` | — | 160 |
| 11 | `11-operator-conv.py` | Conv2d → LIF readout via `braintrace.conv` | Poisson-MNIST as 1ch frames | `decay=0.95` | — | 170 |
| 12 | `12-classification-neuromorphic.py` | LIFCell (flagship) | Poisson-MNIST (10 digits) | `decay=0.97`, single-step | **BPTT side-by-side** | 220 |
| 13 | `13-knob-decay-vs-rank.py` | LIFCell | integrator | sweep `decay_or_rank` ∈ {0.9, 0.95, 0.99, 3, 10, 40} | — | 180 |
| 14 | `14-knob-vjp-method-contrast.py` | LIFCell | DMS | single-step vs multi-step head-to-head | BPTT ref line | 200 |

### Numeric assertions
Assertions placed inside each file's `if __name__ == '__main__':` block,
after training completes. They guard against silent regressions when
examples are run manually or via smoke tests.

- #12: `assert final_acc_pp_prop > 0.6` and both pp_prop and BPTT strictly
  above chance (0.1 for 10-class).
- #14: both VJP methods strictly above chance (0.5 for binary DMS).
- Other files: assert final loss is finite and non-NaN
  (`assert jnp.isfinite(final_loss)`).

## 5. `examples/pp_prop/README.md`

Mirror `examples/drtrl/README.md`. Contents:

```markdown
# pp_prop Examples

Tutorial-linear walk through `braintrace.pp_prop` (alias `ES_D_RTRL` /
`IODimVjpAlgorithm`) — online eligibility-trace gradient estimator with
input-output dimensional complexity. Each file is self-contained. Read 01→14
alongside the companion tutorial at `docs/tutorials/pp_prop.md`.

## How to run

    python examples/pp_prop/01-basics-lif-integrator.py

All CPU, 1-2 min each. No external datasets.

## Axis map

| Axis                                       | Files                    |
|--------------------------------------------|--------------------------|
| Neuron model (LIF/ALIF/GIF/COBA-EI)        | 01, 02, 03, 04           |
| Batching mode (vmap vs batched primitive)  | 05, 06                   |
| vjp_method (single- vs multi-step)         | 07, 08, 14               |
| Operator (matmul/sparse/LoRA/conv)         | 09, 10, 11               |
| Training target                            | 01, 02, 03, 04, 12       |
| Algo knob (decay vs rank)                  | 13                       |
| BPTT baseline                              | 12, 14                   |

Cross-reference: for `fast_solve` / `normalize_matrix_spectrum` knobs
(shared with D_RTRL) see `examples/drtrl/11-knob-fast-solve.py` and
`examples/drtrl/12-knob-normalize-spectrum.py`.

## Tests

    pytest examples/pp_prop/tests -v
```

## 6. `docs/tutorials/pp_prop.md`

Long-form companion (400-600 lines). Sections:
1. **What pp_prop solves** — online gradient, O(BI+BO) memory, O(BIO)
   compute.
2. **Mathematical rule** — ε = ε_f ⊗ ε_x; low-pass update; decay ↔ rank
   duality.
3. **When pp_prop over D_RTRL** — memory saving vs diagonal-approx
   tradeoff.
4. **Walk-through of examples 01-14** — short paragraph each pointing at
   the demonstrated axis.
5. **Limitations** — diagonal approx; single hidden-group assumption per
   primitive; operator-invariant rule requirement.
6. **Fallback to BPTT** — when pp_prop is inappropriate.

Reuse prose structure from `docs/tutorials/drtrl.md` where algorithmic
narrative overlaps.

## 7. `examples/pp_prop/tests/test_smoke.py`

Pytest file parametrized over all 14 example names. Each test:
1. Sets `PP_PROP_NO_PLOT=1` and `PP_PROP_SMOKE=1` via `monkeypatch.setenv`.
2. Loads the example via `importlib.util.spec_from_file_location`.
3. Executes the module (`spec.loader.exec_module(mod)`).
4. Asserts no exceptions raised.

Target: smoke suite <60s wall-clock.

## 8. Risks and fallbacks

| # | Risk | Fallback |
|---|------|----------|
| 09 | `saiunit.sparse` lacks JAX batching rule (drtrl 06 already blocked). | Use `jax.experimental.sparse.BCOO`-backed sparse_matmul path. If still blocked, drop to 13 files, stub 09 with README note. |
| 11 | Conv-SNN may exceed 2 min CPU budget. | 8×8 input, 1 conv layer, 1-2 output channels, T≤30 steps. |
| 04 | `SignedWLinear` may not be wrapped by ETP primitives. | Use regular `braintrace.nn.Linear` with signed init ("soft Dale"). |
| 03 | `brainpy.state.GIF` may lack per-neuron `tau_I2`. | Port local GIF class from `examples/snn_models.py`. |
| 08, 14 | `vjp_method='multi-step'` may have rough edges. | Default examples to `single-step`; multi-step shown as opt-in with explicit `MultiStepData` construction. |

## 9. Dependencies

Already in the project:
- `brainstate` ≥ 0.2.2, `braintools`, `saiunit`, `brainpy-state`, `jax`,
  `matplotlib`, `numpy`.

New transitive:
- `scikit-learn` (for 8×8 digits). Verify availability at implementation
  time. Fallback: pure-numpy synthetic data if missing.

## 10. Out-of-scope

- No modification to `braintrace/` source tree.
- No modification to existing `examples/000-004-*.py`.
- No new primitives, rules, or algorithm implementations.
- No GPU or multi-device paths.
- No persistence of trained weights.
- No CLI/argparse utilities. Plot-skip controlled exclusively via
  `PP_PROP_NO_PLOT` env var (see Section 4).

## 11. Acceptance criteria

1. All 14 Python files exist and run standalone on CPU in under two minutes
   each, using `python examples/pp_prop/NN-*.py`.
2. `pytest examples/pp_prop/tests -v` passes end-to-end in under 60s.
3. `examples/pp_prop/README.md` renders the axis map above.
4. `docs/tutorials/pp_prop.md` covers the six tutorial sections and
   cross-links to each example file.
5. File 12 (flagship) reports both pp_prop and BPTT accuracy curves on the
   same Poisson-MNIST split.
6. File 14 reports both VJP methods on the same DMS split.
7. No regressions in existing tests (`pytest braintrace/`) or drtrl example
   smoke tests.
