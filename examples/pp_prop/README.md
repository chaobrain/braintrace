# pp_prop Examples

A tutorial-linear walk through `braintrace.pp_prop` (aliases `ES_D_RTRL` /
`IODimVjpAlgorithm`) — an online eligibility-trace gradient estimator with
input-output dimensional complexity for spiking neural networks. Each file
is self-contained. Read them in order (01 → 14) to follow the companion
tutorial at `docs/tutorials/pp_prop.md`.

## How to run

    python examples/pp_prop/01-basics-lif-integrator.py

All examples run on CPU in roughly 1–2 minutes. No external datasets; the
Poisson-MNIST examples use sklearn's 8×8 digits (with a pure-numpy fallback
if sklearn is missing).

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
| 09 | `09-operator-sparse.py`             | Sparse recurrent connectivity (masked-dense fallback)   |
| 10 | `10-operator-lora.py`               | Low-rank recurrence via `braintrace.lora_matmul`        |
| 11 | `11-operator-conv.py`               | Conv-SNN via `braintrace.nn.Conv2d`                     |
| 12 | `12-classification-neuromorphic.py` | Flagship: pp_prop vs BPTT on Poisson-MNIST (10 classes) |
| 13 | `13-knob-decay-vs-rank.py`          | Sweep `decay_or_rank` across floats and ints            |
| 14 | `14-knob-vjp-method-contrast.py`    | single-step vs multi-step vs BPTT head-to-head on DMS   |

Cross-reference: for the `fast_solve` knob (shared with D_RTRL but not
required for pp_prop), see `examples/drtrl/11-knob-fast-solve.py`.

## Tutorial

See `docs/tutorials/pp_prop.md` for the long-form narrative.

## Tests

    pytest examples/pp_prop/tests -v
