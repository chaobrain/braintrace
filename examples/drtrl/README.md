# D_RTRL Examples

A tutorial-linear walk through ``braintrace.D_RTRL`` — the online
eligibility-trace gradient estimator.

Each file is self-contained. Read them in order (01 → 11) to follow the
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
| Operator (matmul / sparse / LoRA / conv)   | 07, 08             |
| Target (regression / classification / ...) | 01, 04, 05, 09, 10 |
| Batching mode                              | 02, 03             |
| vjp method                                 | 04, 05             |
| fast_solve knob                            | 11                 |
| BPTT baseline                              | 01, 09, 10         |

> **Note — skipped example:** `06-operator-sparse.py` is not currently
> shipped. The ``brainunit.sparse`` COO/CSR primitives lack JAX batching
> rules, which blocks D_RTRL's internal Jacobian vmap. Adding batching
> rules (or an alternative sparse backend such as
> ``jax.experimental.sparse.BCOO``) is tracked as future framework work.

## Tutorial

See `docs/tutorials/drtrl.md`.

## Tests

```bash
pytest examples/drtrl/tests -v
```
