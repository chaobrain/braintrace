# pp_prop Examples

A tutorial-linear walk through `braintrace.pp_prop` (aliases `ES_D_RTRL` /
`IODimVjpAlgorithm`) — an online eligibility-trace gradient estimator with
input-output dimensional complexity for spiking neural networks. Each file
is self-contained. Read them in order (01 → 16) to follow the companion
tutorial at `docs/tutorials/pp_prop.ipynb`.

## How to run

    python examples/pp_prop/01-basics-lif-integrator.py

The fixed examples run on CPU. The digit examples require the examples extra:

    pip install "braintrace[examples]"

Example 16 is a configurable scaling benchmark, so its runtime depends on the
requested neuron count and update budget.

## Axis map

| Axis                                      | Files              |
|-------------------------------------------|--------------------|
| Neuron model (LIF / ALIF / GIF / COBA-EI) | 01, 02, 03, 04     |
| Batching mode (vmap vs batched primitive) | 05, 06             |
| vjp_method (single-step vs multi-step)    | 07, 08, 14         |
| Operator (matmul / sparse / LoRA / conv)  | 09, 10, 11         |
| Training target                           | 01, 02, 03, 04, 12 |
| Algo knob (decay vs rank)                 | 13                 |
| BPTT baseline                             | 12, 14             |
| Held-out learning evidence                | 15                 |
| Configurable sparse scaling               | 16                 |

### File-by-file summary

| #  | File                                | Demo                                                    |
|----|-------------------------------------|---------------------------------------------------------|
| 01 | `01-basics-lif-integrator.py`       | LIF RSNN on Poisson-to-cumulative-rate regression       |
| 02 | `02-neurons-alif-dms.py`            | ALIF (adaptive threshold) on delayed-match-to-sample    |
| 03 | `03-neurons-gif-working-memory.py`  | GIF with heterogeneous tau_I2 on working-memory recall  |
| 04 | `04-neurons-coba-ei-rsnn.py`        | Dale-law E/I RSNN on small Poisson-MNIST                |
| 05 | `05-batching-vmap.py`               | Batching via `brainstate.nn.Vmap(vmap_states='new')`    |
| 06 | `06-batching-batched.py`            | Batching via the batched ETP primitive path             |
| 07 | `07-vjp-single-step.py`             | `vjp_method='single-step'` (default)                    |
| 08 | `08-vjp-multi-step.py`              | `vjp_method='multi-step'` for temporal credit           |
| 09 | `09-operator-sparse.py`             | Native CSR recurrent connectivity with SparseLinear     |
| 10 | `10-operator-lora.py`               | Low-rank recurrence via `braintrace.lora_matmul`        |
| 11 | `11-operator-conv.py`               | Conv-SNN via `braintrace.nn.Conv2d`                     |
| 12 | `12-classification-neuromorphic.py` | Small pp_prop and BPTT classifier smoke comparison      |
| 13 | `13-knob-decay-vs-rank.py`          | Sweep `decay_or_rank` across floats and ints            |
| 14 | `14-knob-vjp-method-contrast.py`    | single-step vs multi-step vs BPTT head-to-head on DMS   |
| 15 | `15-sparse-temporal-learning.py`    | Sparse LIF learning on held-out handwritten digits      |
| 16 | `16-configurable-sparse-benchmark.py` | Guarded synthetic sparse-CSR scaling and target timing |

### Configurable benchmark

Run an isolated fixed-work measurement:

    python examples/pp_prop/16-configurable-sparse-benchmark.py --neurons 131072 --degree 8 --updates 3

Measure the first validation checkpoint at or above 95 percent:

    python examples/pp_prop/16-configurable-sparse-benchmark.py --mode validation-target --neurons 4096 --target-accuracy 0.95 --json-output pp-prop-4096.json

Use ``--help`` to configure temporal steps, final supervision window, batch
size, optimizer settings, trace decay, evaluation cadence, sparse backend,
recurrent scaling basis, and resource limits. Each run uses a fresh worker
process and prints one schema-versioned JSON result. The default wall-clock
limit is 30 minutes. Progress goes to stderr.

This is a synthetic fixed-degree CSR classifier benchmark with trainable dense
input and readout projections. It is not a connectome-learning benchmark.
Time-to-target repeatedly checks the validation split and is therefore an
adaptive validation metric, not an unbiased held-out estimate. Reported memory
is the highest 100 ms sampled CPU process-tree RSS; GPU allocator memory is not
measured.

Cross-reference: for the `fast_solve` knob (shared with D_RTRL but not
required for pp_prop), see `examples/drtrl/11-knob-fast-solve.py`.

## Tutorial

See `docs/tutorials/pp_prop.ipynb` for the long-form narrative.

## Tests

    pytest examples/pp_prop -v
