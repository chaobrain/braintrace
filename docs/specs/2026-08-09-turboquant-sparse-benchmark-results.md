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

## GPU: the same two arms on CUDA

Date: 2026-08-10. Host: same machine, RTX 3080 Ti Laptop GPU (16 GiB), driver
595.79, JAX 0.11.0 with the CUDA 12 plugin, Python 3.12, running in a container
so the CPU results above are untouched. Both arms are the same two worktrees at
`b013f61` and `0a99c99`, mounted read-only into one container and selected by
`PYTHONPATH`; the harness `source_sha256` is identical across arms, so the only
difference is `braintrace/` itself.

Moving to GPU is worth about 23x on its own: the baseline warm update goes from
5.244 s to 0.222 s.

### The arm difference does not survive the noise

Three seeds, arms alternated within each seed:

| Seed | Order | Warm, baseline | Warm, optimized | Validation, baseline | Validation, optimized | Total, baseline | Total, optimized |
|---|---|---|---|---|---|---|---|
| 0 | baseline first | 0.2352 s | 0.2060 s | 2.436 s | 2.657 s | 13.6 s | 14.6 s |
| 1 | optimized first | 0.2218 s | 0.2106 s | 3.092 s | 2.834 s | 16.3 s | 14.4 s |
| 2 | baseline first | 0.2085 s | 0.2666 s | 3.203 s | 3.195 s | 16.8 s | 16.9 s |

Do not read medians off that table. Sorted by launch order the totals are 13.6,
14.6, 14.4, 16.3, 16.8, 16.9 s while the GPU climbs from 83 C to 89 C and the SM
clock falls from 1624 to 1590 MHz. Position in the sequence moves the total by
about 3 s; the arms differ by well under 1 s. Whichever arm ran second lost.

A second protocol was run to remove that: five alternating pairs at a fixed
seed, order flipped every pair, with a warm-up pair discarded so every measured
pair starts from a hot GPU. Differences are baseline minus optimized, so
positive favours the optimized arm.

| Metric | Paired differences | Median |
|---|---|---|
| Warm update | +0.007, -0.022, -0.003, +0.413, -0.279 s | -0.003 s |
| Validation | +0.012, -0.758, -0.258, +0.183, +0.342 s | +0.012 s |
| Total worker | +0.72, -2.19, -1.05, +1.28, +1.22 s | +0.72 s |

Two of those pairs are contaminated. Pairs 4 and 5 recorded warm updates of
1.043, 0.630 and 0.533 s against a 0.22 s norm, a factor of three to five that a
34 MHz clock drop cannot explain; something stalled, host contention or the WSL
layer, and the cause was not identified. They are left in because the median
absorbs them and dropping inconvenient runs is worse, but the two largest
differences in every row above come from them. The thermal drift described in
the previous table is a separate and much smaller effect.

Every metric straddles zero and the per-pair spread exceeds the median by an
order of magnitude. **On GPU the adopted change is indistinguishable from the
baseline.** That is the expected result, not a measurement failure: section 7 of
the spec shows the einsum and the unrolled form both running at roofline on this
device, so there is nothing for the rewrite to recover.

### Memory

| Arm | Peak device memory over idle |
|---|---|
| baseline | 1261 MiB |
| optimized | 1265 MiB |

Sampled at 0.2 s from `nvidia-smi` device totals with a 476 MiB idle floor
subtracted, so this is coarser than the CPU RSS figures and is reported only to
show the arms allocate the same. The 2.5% host-RSS regression measured on CPU
does not have a device-side counterpart worth quoting at this resolution.

### Quality

| Seed | Arm | Loss trajectory | Accuracy |
|---|---|---|---|
| 0 | baseline | 0.899855, 88.598038, 45.273769, 0.0, 0.0 | 0.9583 |
| 0 | optimized | 0.899855, 88.598198, 45.273376, 0.0, 0.0 | 0.9583 |
| 1 | baseline | 1.040726, 79.302788, 34.837658, 0.712170, 1.421714 | 0.9861 |
| 1 | optimized | 1.040726, 79.302544, 34.836983, 0.712265, 1.422137 | 0.9861 |
| 2 | baseline | 0.789559, 41.470875, 0.009855, 0.028038, 0.557275 | 0.9861 |
| 2 | optimized | 0.789559, 41.470871, 0.009855, 0.028037, 0.557274 | 0.9861 |

Final accuracies match exactly, both between arms and against the CPU table.
The losses do **not** match bit-for-bit the way they do on CPU: the arms agree to
roughly six significant figures and diverge below that, which is ordinary
GPU reduction-order variation and not a property of the change. The CPU-to-GPU
gap is larger again, third to fourth digit, which the benchmark's violent
trajectory amplifies over five updates.

### Where quantization stands on this device

Spec section 7 has the measurements. The short version is that the section 3
negative result is a CPU result only. On GPU, int8 storage makes the Jacobian
contraction 1.54x faster, the batch reduction 2.34x faster and the gather 1.41x
faster, and 4-bit Lloyd-Max packing matches int8 on the contraction while
halving the bytes again.

None of that is wired into the benchmark, and it is a weaker claim than it
looks. Those speedups are measured on per-timestep transients, while the
distortion figures earlier in this document are measured on stored state; the
two sets share no tensor, and the probe prices no rotation on either side. What
is established is the bandwidth narrow storage makes available on this device,
not that a deployable codec would keep it at acceptable distortion. The door
that is bolted shut on CPU is merely unlocked here.

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

For the GPU figures, on a host with a CUDA device and `jax[cuda12]` installed:

```
PYTHONPATH=<path> python turboquant_gpu_probe.py --require-gpu \
  --neurons 131072 --batch-size 32 --nnz 1048576
```

`--require-gpu` makes the probe refuse to report rather than silently produce
CPU numbers. The probe queues twenty launches per timing sample because a single
launch measures a 1.3 ms dispatch floor on this backend, and it interleaves the
candidates within each comparison so thermal drift is charged to both. Anything
comparing two arms end to end needs the same treatment: run them alternately and
difference within a pair, since on a thermally limited device position in the
sequence is worth more than the change under test.
