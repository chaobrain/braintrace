# TurboQuant for the 128K-neuron sparse pp-prop benchmark

Status: implemented
Date: 2026-08-09
Target: `examples/pp_prop/16-configurable-sparse-benchmark.py` at `--neurons 131072 --degree 8`

## 1. Goal

Apply the TurboQuant methodology (Zandieh et al. 2026, arXiv:2504.19874) to reduce
the resource footprint or the wall-clock cost of the sparse pp-prop benchmark at
128K neurons, without regressing learning quality.

Read the outcome carefully before attributing it. **The adopted speedup in
section 5 is a code-generation fix and owes nothing to quantization.**
TurboQuant's contributions here are section 4, a compression study showing the
randomized rotation is worth its cost on eligibility traces and worthless on
weights, and section 3, a measured negative result: on the XLA CPU backend no
quantized representation can accelerate this benchmark, because widening the
narrow form back to float32 is no faster than reading float32 to begin with.
The profiling that produced the section 5 fix was undertaken to locate the
tensors worth quantizing; the fix is what that search actually found.

## 2. Baseline characterisation

Host: i9-12900H, 64 GiB DDR5, XLA CPU backend, JAX 0.11.0, `sparse_backend=jax_raw`.
Configuration: 131072 neurons, degree 8 (1,048,576 stored edges), batch 32,
30 timesteps, 5 updates.

| Quantity | Value (median of seeds 0, 1, 2) |
|---|---|
| Warm update | 5.244 s |
| Validation pass | 9.353 s |
| Peak process-tree RSS | 1.771 GiB |
| Trainable values | 9,830,400 |

XLA cost analysis of the compiled update reports 2.13 GFLOP and 4.40 GB of memory
traffic **per timestep**. Arithmetic intensity is therefore 0.48 FLOP/byte.
A measured streaming rate of 26 GB/s predicts 169 ms per timestep against
167 ms observed, so the update is bandwidth-bound with no slack: bytes removed
convert one-for-one into time saved.

Per-timestep buffer inventory from the optimized HLO:

| Buffer | Shape | Size | Origin |
|---|---|---|---|
| CSR edge intermediates (x3) | `f32[1048576, 32]` | 128 MiB each | `brainevent.csrmm` jvp/transpose |
| Hidden Jacobian, stacked | `f32[3, 32, 131072, 3]` | 144 MiB | `jacrev_last_dim` |
| Hidden Jacobian, transposed | `f32[32, 131072, 3, 3]` | 144 MiB | `out_axes=-2` |
| Trace / signal tensors (x4) | `f32[32, 131072, 3]` | 48 MiB each | trace propagation |

## 3. Does quantization pay for speed on this backend?

Decided by one measurement: the throughput of the integer-to-float conversion a
quantized consumer must pay, against the throughput of simply reading float32.
`examples/pp_prop/turboquant_state_study.py` reports it.

| Kernel | Gelem/s |
|---|---|
| float32 elementwise | 3.03 |
| int8 elementwise, no widening | 9.94 |
| int8 widened to float32 | 3.81 |

Storing int8 cuts the bytes read by four, yet widening recovers only 1.26x over
reading float32 outright: the conversion consumes essentially the whole
bandwidth saving even in the friendliest case, a unary elementwise kernel. In
real consumers, which also read a float32 operand and write float32, it turns
into a net loss. Confirmed end to end on the benchmark's own shapes:

| Operation | float32 | int8 stored |
|---|---|---|
| Jacobian contraction `(32,131072,3,3)x(32,131072,3)` | 19.9 ms | 28.4 ms |
| Batch reduction over `(1048576, 32)` | 13.4 ms | 18.4 ms |
| Gather `(131072,32) -> (1048576,32)` | 10.6 ms | 4.1 ms |

Narrow storage wins only where the value is *moved* and never widened, as in the
gather. Every arithmetic consumer in the pp-prop step widens. **TurboQuant
cannot accelerate this benchmark on the XLA CPU backend**, and the two rewrites in
section 5 were adopted instead. The conclusion is backend-specific: a device with
native low-precision arithmetic would need this measurement repeated.

## 4. Does the randomized rotation earn its keep? Only on the input traces

The codec lives in `braintrace/_quant/`. It follows Algorithm 2 of the paper:
normalize, rotate by `H . D`, assign Lloyd-Max centroids, then optionally spend
one bit on the sign pattern of an independently rotated residual (QJL).

The full Walsh-Hadamard transform over `d = 131072` costs `log2(d) = 17` sweeps
and was measured at 19.0 ms per `(32, 131072)` array against 1.30 ms for a single
sweep, which is unaffordable in a per-step path. `rotate_blocks` therefore
restricts the butterfly to contiguous blocks that XLA emits as one fused pass:
block 16 costs 1.38 ms and block 64 costs 1.45 ms, roughly one sweep.

Measured 4-bit relative reconstruction error on live benchmark state at the
benchmark's own configuration -- 131072 neurons, degree 8, batch 32, 30
timesteps -- quantized along each tensor's widest axis, by rotation block
(block 1 means no rotation):

| Tensor | Shape | MiB | block 1 | 16 | 64 | 256 |
|---|---|---|---|---|---|---|
| `etrace_x`, recurrent relation | `(32, 131072)` | 16.0 | 0.2006 | 0.1035 | 0.0989 | **0.0978** |
| `etrace_x`, feedforward relation | `(32, 131072)` | 16.0 | 0.1995 | 0.1034 | 0.0988 | **0.0978** |
| `etrace_df`, hidden group 0 (x2) | `(32, 131072, 3)` | 48.0 | **0.0427** | 0.0684 | 0.0693 | 0.0650 |
| recurrent CSR values | `(1048576,)` | 4.0 | 0.0977 | 0.0975 | 0.0972 | 0.0978 |
| dense feedforward weight | `(64, 131072)` | 32.0 | 0.0974 | 0.0976 | 0.0975 | 0.0975 |

The 4-bit Lloyd-Max limit for Gaussian coordinates is 0.0975. Reading the table
against that number explains all three behaviours:

* **Input traces gain 2.05x.** They accumulate Poisson spike drive and are
  heavy-tailed, so unrotated scalar quantization is far off the Gaussian limit at
  0.20. Rotation Gaussianizes them onto the limit. Nearly all of the gain is
  already realised at block 16, so the cheapest rotation is also the right one.
* **Weights gain nothing.** Kaiming-normal initialisation makes them Gaussian and
  incoherent already; they sit on the limit rotated or not.
* **Output traces lose 1.6x.** They are *more* quantizer-friendly than Gaussian,
  landing at 0.0427 unrotated, because the surrogate derivatives that drive them
  are near-identical across neurons. Rotation destroys that structure and drags
  them back toward the Gaussian limit.

Rotation is therefore not a free default. It buys a real 2x on heavy-tailed state
and costs a real 1.6x on state that is already concentrated, and which of the two
applies is a property of the tensor, not of the algorithm. Note also that these
figures move with scale: the same table at 4096 neurons showed input-trace gains
of 2.8x and 5.4x, because the post-rotation coordinate distribution converges to
`N(0, 1/d)` and the unrotated baseline degrades faster with `d` than the rotated
one. Any rotation decision has to be measured at the deployed width.

On the scalar stage alone the codec matches Lloyd-Max theory to within 5%
(4-bit: 0.0971 measured against 0.0975 predicted).

The QJL stage behaves as documented only when compared at matched stage-one
width. Adding one bit of QJL on top of a 3-bit scalar stage reduces inner-product
bias from -0.916 to +0.364 and RMSE from 11.28 to 9.62. At a *matched total*
budget, 4-bit scalar beats 3-bit-plus-QJL on both, because for these
near-Gaussian post-rotation coordinates the scalar codebook is already near
optimal. Both facts are asserted in `braintrace/_quant/_turboquant_test.py`.

## 5. Adopted change: contract the hidden Jacobian without `dot_general`

Profiling attributed 63.4 ms of the 167 ms timestep to four operations around the
hidden-group Jacobian, all bandwidth-bound and all far off roofline.

`_update_IO_dim_etrace_scan_fn` contracted the Jacobian with
`jnp.einsum('...ij,...j->...i', ...)`. The contracted axis is the minor-most axis
of both operands and holds one entry per hidden state in the group, three for this
model. XLA lowers that to a batched matvec whose innermost loop is three elements
long, which cannot be vectorized; measured 19.9 ms against a 9.4 ms roofline.

`_contract_hidden_jacobian` unrolls the contraction into `num_state` full-width
multiply-accumulates over the leading, contiguous axes. Same arithmetic, same
result to float associativity, 8.2 ms. Removing the `dot_general` consumers also
let XLA fuse away the `out_axes=-2` transpose of the 144 MiB Jacobian entirely.

`jacrev_last_dim` additionally materialized a `(32, 131072, 3, 3)` broadcast
identity, 144 MiB per timestep, purely to feed `vmap` over one-hot cotangents.
It now calls the pullback once per basis vector against a broadcast one-hot,
which keeps peak RSS at the baseline level.

## 6. Results

Full tables in `docs/specs/2026-08-09-turboquant-sparse-benchmark-results.md`.
Medians over seeds 0, 1, 2 at 131072 neurons: warm update 5.244 s to 4.192 s
(**1.25x**), validation pass 9.353 s to 6.822 s (**1.37x**), total worker 51.6 s
to 41.6 s (**1.24x**). Peak RSS 1.771 GiB to 1.815 GiB, a **2.5% regression**.
Loss trajectories and final accuracies are identical to all printed digits on
every seed.

## 7. What was not pursued

The `brainevent.csrmm` jvp and transpose rules materialize three
`(nnz, batch)` buffers, 128 MiB each, and account for roughly 59% of the
timestep after the change above. The gradient for edge `e` is
`sum_b eps_f[b, post(e)] * eps_x[b, pre(e)]`, which needs no `(nnz, batch)`
materialization if the batch reduction is fused into the gather. That is a
fusion problem in a different package, not a compression problem, and is out of
scope here.
