# D_RTRL: Online Gradient Learning via Eligibility Traces

This tutorial walks through
[`braintrace.D_RTRL`](../../braintrace/_algorithm/d_rtrl.py), the online
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
- **Jacobian rules**: per-primitive `dt_to_t`, `xy_to_dw`, `init_drtrl`.

## 3. Minimal example — integrator

> **Quick start:** most users should call `braintrace.compile(model, braintrace.D_RTRL, x0,
> batch_size=B)` (or `braintrace.compile(..., vmap=True)` for batched lanes). The sections
> below describe how `D_RTRL` works under the hood and the knobs it exposes.

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
| Sparse matmul | `braintrace.sparse_matmul`, `braintrace.nn.SparseLinear` | — (skipped; see below) |
| LoRA matmul | `braintrace.lora_matmul`, `braintrace.nn.LoRA` | 07 |

> **Sparse-matmul note:** `brainunit.sparse` COO/CSR primitives do not yet
> have JAX batching rules, which D_RTRL's internal Jacobian vmap requires.
> `06-operator-sparse.py` is therefore not shipped. Adding batching rules
> (or integrating `jax.experimental.sparse.BCOO`) is future framework work.

## 7. Target types

| Target | Loss | Example |
|---|---|---|
| Regression | `braintools.metric.squared_error` | 01–03, 07 |
| Classification | `softmax_cross_entropy_with_integer_labels` | 04–06, 08, 09 |
| Autoregressive generation | next-token cross-entropy | 10 |

## 8. Performance + stability knobs

- **`fast_solve` (default True)** — einsum fast path for mm/mv/elemwise
  primitives. Conv/sparse/LoRA always legacy. See
  [`11-knob-fast-solve.py`](../../examples/drtrl/11-knob-fast-solve.py).
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
- **`multi-step` vs `single-step`** — speed/bias tradeoff. No free lunch.
- **Long-horizon credit assignment** — traces still decay exponentially; very
  long dependencies suffer.
- **No higher-order gradients** — D-RTRL uses `custom_vjp` at the primal
  level. `jax.grad(jax.grad(...))` through `D_RTRL` is not supported.

## 10. FAQ / troubleshooting

- **NaN traces** → spectral radius likely above 1. Rescale the recurrent
  init.
- **`compile_graph` fails "no hidden state reachable"** → the weight is used
  only through plain `x @ w` or the ETP primitive does not reach any
  `HiddenState`.
- **Loss diverges fast** → check `vjp_method='multi-step'` window fits the
  sequence length, and no outer `jax.lax.scan` is re-init'ing state each step.
- **Conv1d shape error** → Conv1d wants a channel axis. Pre-flatten features
  before the RNN.

## 11. API reference pointers

- [`braintrace.D_RTRL`](../../braintrace/_algorithm/d_rtrl.py) — alias for
  `ParamDimVjpAlgorithm`.
- [`braintrace.ES_D_RTRL` / `braintrace.pp_prop`](../../braintrace/_algorithm/pp_prop.py)
  — the I/O-dimension variant (separate future tutorial).
- [`braintrace.compile_etrace_graph`](../../braintrace/_compiler/graph.py)
  — called internally by `D_RTRL.compile_graph`.
- [`braintrace.register_primitive`](../../braintrace/_op/_primitive.py)
  — add your own ETP primitive.
- [`CLAUDE.md`](../../CLAUDE.md) — architectural overview of BrainTrace.
