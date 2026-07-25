# Known limitations — verified findings list

Status: living document
Baseline: commit `bc153da` plus the P1 phase
Supersedes: the untracked list formerly at
`dev/superpowers/specs/2026-05-26-comprehensive-test-strategy-design.md`, which
is gitignored and absent from the repository

This is the backlog of expected-failure and improvement items that `AGENTS.md`
§ Known limitations refers to. Every entry is verified against the current test
suite, not transcribed. An entry is **active** only if a test or a documented
scope boundary pins it today.

## Observed suite state

Measured at the close of the P1 phase:

```
$ pytest braintrace/ -q -rs
2119 passed, 4 deselected, 262 warnings in 809.64s
```

The suite contains **no `skip` and no `xfail` markers at all** — `grep -rn
"mark.xfail\|mark.skip" braintrace/` returns nothing. So no finding below is
pinned by a deferral; each is pinned by a passing assertion or by a documented
scope boundary in the code. The 4 deselected tests are `diagnostic`-marked
random exploration, excluded from CI gating by the default `addopts` in
`pyproject.toml`; they pass as well (`pytest braintrace/ -m diagnostic -q` →
`4 passed, 2119 deselected`).

This matters for reading the table: "resolved" always means a test asserts the
resolution, never that a test stopped being run.

## Status legend

- **resolved** — the claim no longer holds; a passing test pins the resolution.
- **dead** — the claim described code that no longer exists.
- **active** — the claim still holds; a test or documented boundary pins it.
- **misattributed** — the observation was real, the stated cause was not.

## Findings

| ID | Claim | Status | Pinned by |
|---|---|---|---|
| F-01 / F-04 | multi-state (`num_state >= 2`) HiddenGroups mishandled | resolved | `tests/snn_model_correctness_test.py::test_multi_state_hidden_groups_are_discovered` and `::test_d_rtrl_matches_bptt_on_snn_models` (num_state 1–5) |
| F-07 / F-08 / F-09 | OTTT / OTPE approximation bias | dead | removed with those algorithms in 0.3.0 |
| F-17 | implementation facts drifted from the instruction file | resolved | `braintrace/__init___test.py` |
| F-19 / F-20 | OTTT / OSTTP exactness gaps | dead | removed with those algorithms in 0.3.0 |
| F-21 | rank / decay / random-feedback configs are exact on rate models | misattributed — the cause is the oracle path (F-23), not the model | `tests/approx_correctness_test.py::test_rank_decay_random_approximations_are_exact_on_rate_model_F21`, docstring corrected |
| F-22 | exposing approximation bias needs an SNN multi-population zoo | **retired** — premise false; a multi-population SNN model (num_state 5, 3 relations) is bitwise-exact on the same path | replaced by `tests/approx_correctness_test.py::test_approximations_are_measurable_through_a_finite_window` and `tests/snn_model_correctness_test.py::test_approximation_is_measurable_on_snn_models` |
| F-23 | the full-window multi-step oracle path is blind to every learning-rule axis | active, **by design** — documented, not a defect | `tests/axis_discrimination_test.py`, both directions; warned in `online_param_gradients`' docstring |
| F-24 | `_etrace_model_test.py` factories are non-deterministic (unseeded global RNG) | active in the layer classes; neutralised for oracle use | `oracle_models_test.py::test_snn_spec_construction_is_deterministic` |
| F-25 | SNN models are silent at unit input scale, so comparisons are vacuous | active as a property; guarded | `oracle_models_test.py::test_snn_spec_is_live`, `::test_underdriven_input_scale_is_dead`, `::test_overdriven_input_scale_is_also_dead_while_still_spiking`, and `assert_model_is_live` |
| F-26 | `pp_prop` / IODim raised on conv + trainable bias | resolved | `oracle_test.py::test_pp_prop_conv_bias_matches_bptt`; `conv_nwc_bias` restored to both pp_prop family parametrizations |
| F-27 | *(never instantiated)* — reserved during planning for "an SNN spec that cannot be made live". No such spec exists: every one of the nine is live at a recorded input scale. The phenomenon that prompted the reservation is the bounded-above liveness window, recorded under F-25. | n/a | — |
| F-28 | `EProp(feedback='random')` assumes a single, direct readout whose width equals the HiddenGroup's own | active, **documented scope boundary** | `e_prop.py:117-122` states it: the hooks only see `∂L/∂h`, which has no visibility into a separate readout layer's width, so the projection matrix is square |
| F-29 | `pp_prop(decay_or_rank=1)` is a genuine approximation | **active, misattributed as approximate** — rank 1 means decay 0, so the presynaptic EMA collapses to the exact current input | `tests/approx_correctness_test.py::test_rank1_pp_prop_is_degenerate_on_a_recurrent_only_relation` pins both sides |
| F-SCAN / F-SCAN-WEIGHT | weight inside control flow raised `KeyError` | resolved | `_compiler/base_test.py::TestCheckUnsupportedOp::test_error_message_identifies_weight_variable` |
| F-SINGLESTEP | naive single-step summation does not equal BPTT even for an exact algorithm | active as a property; documented | `online_param_gradients_singlestep_naive`' docstring; used deliberately as the maximally-sensitive window in `oracle_test.py` |

## Notes on F-23

This is the load-bearing finding, and it is a property rather than a bug. A
whole-sequence `MultiStepData` call makes the within-call gradient exact
reverse-mode, so the eligibility trace enters only at a sequence boundary that
does not exist. `chunked_online_param_gradients` has documented this all along
— "this is the oracle that actually validates trace correctness" — but the
assertions written against `online_param_gradients` did not heed it, which is
how F-21 and F-22 reached the wrong cause.

The rule that follows: an assertion whose subject is a trace factorization, a
temporal recursion, a recurrence scope, a filter or a learning signal must use a
finite window and must be guarded by `assert_gradients_differ`. An assertion
whose subject is the compiler or an ETP per-primitive rule may use the
full-window path, and that is most of the `multi-step` call sites in the suite.

## Notes on F-29

`decay_or_rank` accepts either a float decay or an int rank, and
`_format_decay_and_rank` maps an int through `decay = (rank - 1) / (rank + 1)`.
So `decay_or_rank=1` means decay 0, the presynaptic EMA
`eps^t = a * eps^{t-1} + (1 - a) * x_t` collapses to `x_t`, and no presynaptic
smearing is introduced at all.

Measured on a model whose only ETP relation is the recurrent weight — so the
relation's presynaptic input *is* the hidden state — rank 1 reproduces exact
D_RTRL to round-off through a finite window:

| variation | rank-1 rel. deviation | decay-0.5, same setup |
|---|---|---|
| chunk 1 / 2 / 4, `T` 8 / 12 / 16 | 1e-10 … 7e-9 | 9e-4 … 1.2e-2 |
| recurrent spectral radius 0.25 … 4.97 | 6e-10 … 1.6e-8 | 4e-3 … 1.5e-1 |
| `n_rec` 4 → 8 | 2e-8 | 1.2e-1 |

float32 round-off on these trees is ~1e-8, so every figure in the first column
is noise. The deviation does **not** grow with the spectral radius, which rules
out "the recurrence is too weak for history to matter" as the explanation.

It is a real approximation as soon as the presynaptic input carries an external
component: 0.55 on a variant whose ETP weight is the input weight, and 0.31 to
0.79 on all nine SNN specs, whose projections consume `concat(input, spikes)`.

**The mechanism is not established — only the boundary is.** Two consequences:
`pp_prop(decay_or_rank=1)` on a recurrent-only relation is unusable as a
positive control for approximation error, and any future axis test that wants
one must either use a float decay or a model with an external-input relation.

## Mapping from the `AGENTS.md` prose list

`AGENTS.md` described its limitations in prose. Each item maps onto a finding
above, so nothing survives only as prose:

| Prose item | Disposition |
|---|---|
| approximation-mode validity beyond shallow depth | successor to F-21 / F-23; measured through a finite window, and bounded by F-29 for one configuration |
| heterogeneous-population leak resolution | resolved — `tests/snn_model_correctness_test.py::test_heterogeneous_leaks_do_not_break_exactness` |
| target-signal threading under JIT | dead — died with `OSTTP`'s `y_target` path |
| single-readout / feedback-shape assumptions | active — F-28, a documented scope boundary in `e_prop.py` rather than a defect |
| gaps in cross-algorithm equivalence coverage | successor to F-23; `tests/axis_discrimination_test.py` covers the pairs |

## What P1 did not resolve

Stated so the list is not read as exhaustive:

- **F-29's mechanism.** Boundary measured, cause unexplained.
- **F-28.** A square hidden×hidden feedback matrix is a real restriction on
  `EProp(feedback='random')`. It is documented, not lifted, and no test
  exercises a separate readout layer of a different width.
- **F-24 in the layer classes.** The oracle specs re-seed at construction; the
  `_etrace_model_test.py` constructors still draw from the global RNG.
