# Sparse pp-prop learning example

## Purpose

Add one deterministic example that demonstrates learning rather than merely
executing the pp-prop API. The example must use a real dataset, native sparse
recurrence, multiple seeds, and held-out accuracy.

## Design

- Use a fixed stratified train-validation split of sklearn digits zero and one.
- Convert each image into a 30-step Poisson spike train.
- Apply cross-entropy only over the final five steps.
- Use 96 LIF neurons with eight CSR recurrent edges per neuron.
- Reset model and eligibility state before every sequence batch.
- Train with single-step pp-prop for five real epochs and three model seeds.
- Report initial and final validation accuracy, loss, and seed variation.

## Acceptance

- The recurrent parameter contains exactly 768 values.
- Every seed reaches at least 90 percent held-out accuracy.
- Mean held-out accuracy is at least 95 percent.
- Training loss decreases for every seed.
- No dense hidden-by-hidden matrix is constructed.

## Configurable benchmark

Example 15 remains the fixed learning acceptance profile. Example 16 provides
an isolated configurable worker for synthetic sparse-CSR scaling experiments.
It must not describe the classifier as connectome learning or treat adaptive
validation checks as unbiased held-out accuracy.

The benchmark exposes neuron count, recurrent degree, batch size, temporal
steps, supervised final-window length, seed, learning rate, trace decay,
gradient clipping, sparse backend, update budget, evaluation cadence, target
validation accuracy, recurrent initialization basis, memory limits, and a
wall-clock limit.

Two modes are supported:

- ``fixed-work`` runs an exact update budget and reports the first cold update
  separately from warmed updates.
- ``validation-target`` stops at the first configured validation checkpoint
  meeting the target and reports that checkpoint as an upper bound in training
  ticks.

The fixed-degree scaling profile defaults to recurrent values normalized by
degree. A legacy neuron-count basis remains selectable to reproduce the earlier
example dynamics. Poisson spike trains are generated once per example and
epoch so changing batch size does not change the sampled inputs.

Every run executes in a fresh subprocess. The supervisor records the highest
100 ms sampled process-tree RSS and terminates only the worker when either the
configured RSS ceiling or minimum available-memory floor is crossed. CPU memory
is measured; GPU allocator memory is outside this benchmark's contract.

The final result is strict schema-versioned JSON containing the effective
configuration, status, update and tick counts, examples and sample-ticks seen,
validation history, recurrent update evidence, parameter counts, setup and
evaluation timing, cold and warm update timing, total runtime, environment
versions, source fingerprint, and resource-guard telemetry.

## Configurable benchmark acceptance

- Invalid dimensions, rates, decay, targets, cadence, and memory limits fail
  before model construction.
- The reported recurrent edge count is exactly ``neurons * degree``.
- Initial validation is checked at update zero.
- Threshold ticks latch at the first passing validation checkpoint.
- A target miss is explicit and can produce a nonzero exit with
  ``--require-target``.
- Progress is written to stderr and stdout contains exactly one JSON result.
- The supervisor distinguishes success, benchmark failure, and memory-guard
  termination.
