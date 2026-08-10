# Results: 128K-neuron sparse pp-prop benchmark, before and after

Companion to `2026-08-09-turboquant-sparse-benchmark.md`.
Date: 2026-08-09

## Configuration

```
16-configurable-sparse-benchmark.py \
  --neurons 131072 --degree 8 --updates 5 --eval-interval 5 --seed {0,1,2} \
  --max-rss-gib 32 --min-available-gib 4 --max-wall-seconds 3600
```

1,048,576 stored recurrent edges, batch 32, 30 timesteps, 5 supervised
timesteps, `sparse_backend=jax_raw`. Host: i9-12900H, 64 GiB DDR5, XLA CPU
backend, JAX 0.11.0, braintrace at `b013f61`. Each arm ran three seeds on an
otherwise idle machine; the tables report medians across seeds.

Arms:

* **baseline** -- unmodified `b013f61`.
* **contract** -- `_contract_hidden_jacobian` only.
* **full** -- `_contract_hidden_jacobian` plus the `jacrev_last_dim` rewrite. Adopted.

## Speed

| Arm | Warm update | Per seed | Validation pass | Total worker |
|---|---|---|---|---|
| baseline | 5.244 s | 5.31 / 5.23 / 5.24 | 9.353 s | 51.6 s |
| contract | 4.171 s | 4.57 / 4.14 / 4.17 | 6.730 s | 41.0 s |
| **full** | **4.192 s** | 4.25 / 4.19 / 4.19 | **6.822 s** | **41.6 s** |

Against baseline, the adopted arm is **1.25x** faster per training update,
**1.37x** faster per validation pass, and **1.24x** faster end to end. The
validation pass gains more because it is pure forward trace evolution, where the
Jacobian contraction is a larger share of the work.

`contract` and `full` are indistinguishable in time; the spread within the
baseline arm (5.23-5.31 s) is comparable to the gap between them.

## Memory

| Arm | Peak process-tree RSS |
|---|---|
| baseline | 1.771 GiB |
| contract | 1.827 GiB |
| **full** | **1.815 GiB** |

The adopted arm costs **2.5% more peak RSS**. Unrolling the contraction keeps
the 144 MiB Jacobian live across three multiply-accumulates rather than handing
it to a single `dot_general`; the `jacrev_last_dim` rewrite, which stops a 144
MiB broadcast identity being materialized per timestep, recovers part but not
all of that. RSS is sampled at 0.1 s, so 2.5% is near the resolution of the
measurement, but it is a regression in the same direction across all three
seeds and is reported as one.

`full` was adopted over `contract` on mechanism rather than on the timing gap:
it strictly removes work, materializing three broadcast one-hot cotangents in
place of a full `(32, 131072, 3, 3)` identity.

## Quality

The rewrite is arithmetically equivalent to the `einsum` it replaces, differing
only in floating-point association order. Observed:

| Seed | Loss trajectory (identical across all three arms) | Final accuracy |
|---|---|---|
| 0 | 0.899178, 88.545609, 45.191315, 0.0, 0.0 | 0.9583 |
| 1 | 1.040395, 79.180244, 34.708027, 0.699299, 1.415405 | 0.9861 |
| 2 | 0.789764, 42.128510, 0.023178, 0.025669, 0.552137 | 0.9861 |

Losses agree to all six printed digits and final accuracies agree exactly, for
every seed, across all three arms. Quality is unchanged, not merely comparable.

The trajectories themselves are violent -- loss rises by two orders of magnitude
before collapsing -- which is a property of the benchmark at this width and
learning rate, not of the change. It is also why bit-identical agreement, rather
than a statistical comparison over five updates, is the evidence being offered.

## TurboQuant compression of the live state

`examples/pp_prop/turboquant_state_study.py --neurons 131072 --degree 8
--batch-size 32 --steps 30`.

Total live float32 state is 164.0 MiB, against a 1.8 GiB peak RSS dominated by
per-timestep transients. Compressing all of it to 4 bits would save 143.5 MiB,
about 8% of peak. The distortion each tensor would pay, and why rotation helps
some and hurts others, is in section 4 of the spec.

Quantization was not adopted in the hot path. Widening int8 back to float32 was
measured at 3.81 Gelem/s against 3.03 for reading float32 outright, so the
conversion consumes the bandwidth saving; on the benchmark's own shapes the
Jacobian contraction went 19.9 to 28.4 ms and the `(nnz, batch)` reduction 13.4
to 18.4 ms when the stored operand was narrowed.

## Reproduction

```
git worktree add -b feat/turboquant-sparse-benchmark <path> b013f61
cd <path>/examples/pp_prop
PYTHONPATH=<path> python 16-configurable-sparse-benchmark.py --neurons 131072 ...
PYTHONPATH=<path> python turboquant_state_study.py --neurons 131072 ...
```

`turboquant_state_study.py` re-measures the conversion throughput on the host it
runs on. The speed conclusion in section 3 of the spec is backend-specific and
should be re-derived from that output before being carried to a GPU.
