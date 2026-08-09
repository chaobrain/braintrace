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
| F-07 / F-08 / F-09 | OTTT / OTPE approximation bias | dead | removed with those algorithms in 0.2.5 |
| F-17 | implementation facts drifted from the instruction file | resolved | `braintrace/__init___test.py` |
| F-19 / F-20 | OTTT / OSTTP exactness gaps | dead | removed with those algorithms in 0.2.5 |
| F-21 | rank / decay / random-feedback configs are exact on rate models | misattributed — the cause is the oracle path (F-23), not the model | `tests/approx_correctness_test.py::test_rank_decay_random_approximations_are_exact_on_rate_model_F21`, docstring corrected |
| F-22 | exposing approximation bias needs an SNN multi-population zoo | **retired** — premise false; a multi-population SNN model (num_state 5, 3 relations) is bitwise-exact on the same path | replaced by `tests/approx_correctness_test.py::test_approximations_are_measurable_through_a_finite_window` and `tests/snn_model_correctness_test.py::test_approximation_is_measurable_on_snn_models` |
| F-23 | the full-window multi-step oracle path is blind to every learning-rule axis | active, **by design** — documented, not a defect | `tests/axis_discrimination_test.py`, both directions; warned in `online_param_gradients`' docstring |
| F-24 | `braintrace/_testing/models.py` factories are non-deterministic (unseeded global RNG) | active in the layer classes; neutralised for oracle use | `oracle_models_test.py::test_snn_spec_construction_is_deterministic` |
| F-25 | SNN models are silent at unit input scale, so comparisons are vacuous | active as a property; guarded | `oracle_models_test.py::test_snn_spec_is_live`, `::test_underdriven_input_scale_is_dead`, `::test_overdriven_input_scale_is_also_dead_while_still_spiking`, and `assert_model_is_live` |
| F-26 | `pp_prop` / IODim raised on conv + trainable bias | resolved | `oracle_test.py::test_pp_prop_conv_bias_matches_bptt`; `conv_nwc_bias` restored to both pp_prop family parametrizations |
| F-27 | *(never instantiated)* — reserved during planning for "an SNN spec that cannot be made live". No such spec exists: every one of the nine is live at a recorded input scale. The phenomenon that prompted the reservation is the bounded-above liveness window, recorded under F-25. | n/a | — |
| F-28 | `EProp(feedback='random')` assumes a single, direct readout whose width equals the HiddenGroup's own | active, **documented scope boundary** | `e_prop.py:117-122` states it: the hooks only see `∂L/∂h`, which has no visibility into a separate readout layer's width, so the projection matrix is square |
| F-29 | `pp_prop(decay_or_rank=1)` is a genuine approximation | **active, misattributed as approximate** — rank 1 means decay 0, so the presynaptic EMA collapses to the exact current input | `tests/approx_correctness_test.py::test_rank1_pp_prop_is_degenerate_on_a_recurrent_only_relation` pins both sides |
| F-30 | the IO-dim f-side de-biasing correction was indexed by `update()` call count, not by trace-step count | resolved — `running_index` now counts completed timesteps and the correction uses stable `expm1` math without a cutoff | `_input_data_test.py::TestCountUpdateSteps`; `io_dim_vjp_test.py::TestBiasCorrectionTimeIndex`; `sequence_test.py::TestStateLifecycle::test_running_index_advances_by_window_length` |
| F-31 | on a hidden group whose `y -> hidden` tail is not position-preserving, *no* within-group scope reaches BPTT — saturated SnAp / full within-group RTRL included | active outside IO-factorized learners; pp-prop now fails closed at compile time when position preservation cannot be proved | `_algorithm/snap_n_test.py::TestConservativeFallbackEndToEnd::test_a_relabelling_tail_defeats_saturation_and_says_so`; `_algorithm/pp_prop_safety_test.py::test_pp_prop_rejects_a_non_position_preserving_tail` |
| F-32 | the coupled-scope transition is materialised as a dense `(*V, S, *V, S)` Jacobian, so `random_projection` costs `O((V·S)^2)` per group even though it only ever needs `D @ s_tilde` — one JVP would do | active, **deliberate**; a matrix-free `D_full` would need a JVP seam the executor does not have | `_algorithm/vjp_graph_executor_test.py::TestFullJacobianFlag::test_the_flag_changes_the_jacobian_shape` records the shape; UORO's docstring states the memory honestly |
| F-33 | under `vjp_method='single-step'`, every **plain** (non-ETP) parameter's gradient was exactly **zero** — not merely truncated | **resolved 2026-08-08** — the compiled graph now partitions ETP-routed and plain-only paths before VJP | `pp_prop_test.py::TestPpPropGradients::test_single_step_plain_gradients_match_exact_local_vjp`, `oracle_models_test.py::TestPlainAndEtpRnn::test_single_step_keeps_local_plain_gradients`, and `three_factor_test.py::TestItDoesSomethingMeasurably::test_every_plain_parameter_gets_its_exact_local_vjp_gradient` |
| F-34 | the multi-step backward-pass dictionaries did not reflect ETP/plain ownership | **resolved 2026-08-08** — the compiled graph is authoritative and the executor partitions states by relation path before VJP | `_state_management_test.py::TestSplitDictStatesV2`, `_algorithm/graph_executor_test.py::TestStatePartitioning`, and `_compiler/graph_test.py::TestCompileGraphRNN::test_compiled_graph_owns_exclusive_etrace_parameter_paths` |
| F-35 | a `train_synthetic_gradient` fit is valid only for the exact `(loss_fn, chunk_size)` pair it was given, and **nothing checks that deployment matches** — a mismatch is not degraded DNI but noise shaped like a cotangent, measurably worse than leaving DNI off | active, **documented not enforced**; the learner never sees the caller's objective, so it has nothing to compare against. Narrowed since: the helper no longer *introduces* a mismatch of its own — a sequence length that is not a multiple of `chunk_size` used to be truncated into a short final window, fitting one pair against a shorter future than any deployed window, and is now refused | `dni_test.py::TestALearnedSynthesiserHelps::test_training_on_the_wrong_window_size_is_worse_than_not_training` pins the `chunk_size` half; the `loss_fn` half is measured in B4's docstring (0.577 against 0.140 for `M ≡ 0`). Since the sequence driver, `chunk_size` is also a keyword on `etrace_grad`, so fit-vs-deploy mismatch is reachable without writing a window by hand — `sequence_test.py::TestAlgorithmInteractions::test_a_chunk_size_mismatch_degrades_a_fitted_synthesiser` fits at 1, deploys at 2 (2.0e-02 relative), with an untrained-synthesiser control at 1.3e-01 so the comparison is not between two no-ops. `reduction` is **not** a mismatch surface: it divides once after the scan and never enters the differentiated objective (`|mean − sum/T| = 7.5e-09`) |
| F-36 | path-granular routing could silently lose an occurrence when one ParamState had represented ETP ownership plus another differentiable use | **resolved 2026-08-08** — compilation rejects unrepresented plain, cross-leaf, descended-scan, and earlier chained-ETP occurrences, plus trainable inputs derived from multiple leaves | `_compiler/graph_test.py::TestCompileGraphRNN::test_mixed_etrace_and_plain_use_of_one_leaf_is_rejected`; `test_mixed_ownership_is_rejected_across_pytree_leaves`; `test_trainable_invar_from_multiple_param_states_is_rejected`; `_compiler/canonicalize_test.py::TestScanModelCompilation::test_drtrl_rejects_unrepresented_internal_etrace_paths` |
| F-37 | DNI injects only at **hidden** exits, so future credit that would arrive through a *non-hidden* persistent state (a `ShortTermState` carried across windows) is dropped: `_exit_cotangent_grads` zeroes the `oth_states` slot of the injected template | active, **structural**; a synthesiser for non-hidden state would need its own group abstraction, and `hidden_groups` is the only slab the compiler provides | `dni_test.py::TestOtherStateCreditIsNotInjected`, on a fixture that actually has one — the earlier zeroing assertion ran on an empty tree and held vacuously |
| F-38 | a **standing** `ThreeFactor.modulator` is a plain Python attribute, so under `jit` it is captured at trace time: reassigning it does not retrace and the compiled closure keeps the old value | active, **documented**; the per-call `update(..., modulator=...)` form is unaffected and is what the docstring now steers to | `three_factor_test.py::TestTheStandingModulatorUnderJit` |
| F-39 | a relation with both a direct hidden path and an indirect path through another trainable ETP primitive silently omitted the indirect instantaneous chain-rule term in every VJP eligibility-trace algorithm | **resolved 2026-08-08** — the shared VJP graph executor rejects `PathClassification.MIXED` before committing the graph; low-level relation discovery remains inspectable and independent direct relations may still share one ParamState | `vjp_graph_executor_test.py::TestETraceVjpGraphExecutor::test_mixed_relation_is_inspectable_but_not_executable`; `io_dim_vjp_test.py::TestConstruction::test_partial_direct_and_indirect_relation_is_rejected`; `param_dim_vjp_test.py::TestConstruction::test_partial_direct_and_indirect_relation_is_rejected` |
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

## Notes on F-30

Resolved on 2026-08-08. `running_index` is now the cumulative number of
successfully completed model timesteps: a six-step `MultiStepData` call advances
it by six, while six single-step calls reach the same value. The f-side warm-up
normalizer is computed as `-expm1(trace_steps * log(decay))`, where
`trace_steps` is the exact age of the trace being contracted. Multi-step VJP
contracts the window-entry trace, so its ages are `0, k, 2k, ...`; single-step
contracts the newly updated trace. Age zero and decay zero are handled exactly.
This removes both the window-length lag and the former 1000-step cutoff without
cancellation near decay one.

The regression tests cover single and multi-step counting, cumulative windowed
runs, decays `0`, `0.9`, and `0.9999`, and the boundary at 1000/1001 steps.

## Notes on F-31

The per-parameter eligibility trace indexes hidden units by *position within a
hidden group*. Every `recurrence_scope` coordinate — `diagonal`, `coupled`,
`sparse_n` — is a statement about which positions a parameter's influence is
retained for. That vocabulary presumes the group's positions survive the trip
from the mixing primitive's output back into the hidden state. When they do not,
the trace cannot represent the influence at all, and widening the neighbourhood
does not help: the missing term is not a neighbour that was dropped, it is a
relabelling the representation has no slot for.

Reproduced on a dense tanh RNN (`n_in=3`, `n_rec=5`, `T=8`, chunk 2, ETP
parameters only) whose recurrent pre-activation is rolled by one position before
it re-enters `h` — `h^t = tanh(x @ win + roll(matmul(h^{t-1}, w), 1))`. The
position graph of the mixing primitive alone has diameter 1, so `n = 2` already
saturates and the un-rolled control is the same model with `roll` deleted:

| algorithm | rel. deviation from BPTT, **with** the roll | control, **without** |
|---|---|---|
| `D_RTRL` (diagonal) | 1.0329 | 0.4041 |
| `OSTLRecurrent` (coupled) | 1.0452 | 0.2836 |
| `SnAp(n=2)` (saturated) | 1.3031 | **1.29e-07** |
| `SnAp(n=4)` (saturated) | 1.3031 | 1.29e-07 |

The control column is what gives the first column its meaning: with a
position-preserving tail, saturation *is* BPTT to float32 round-off while the
cheaper scopes are genuinely approximate — the axis behaves exactly as the
roadmap claims. Add the roll and the ordering collapses: the most expensive
coordinate is now the least accurate of the three, because retaining more
positions buys nothing when every position is mislabelled.

Two things follow, and only the second is a defect:

1. **It is detected, not silent.** IO-factorized learners now reject the
   unprovable tail during compilation. SnAp still widens its pattern to
   all-to-all and emits `SNAP_PATTERN_CONSERVATIVE`.
2. **The diagnostic's wording was wrong, and was fixed in P3.** It read "the
   gradient stays correct", which is true of the *pattern* (a superset never
   drops a real dependency) but reads as a claim about the returned gradient.
   It now says the approximation is not degraded and points here.

Not fixed: representing a non-position-preserving tail would require the trace
to carry the tail's own permutation alongside the neighbourhood, which is a
change to the trace layout rather than to any one algorithm. pp-prop is safe by
refusal; SnAp remains bounded by the elementwise-tail scope in its docstring.

## Notes on F-33

Fixed on 2026-08-08. `ETraceGraph.etrace_param_paths` is derived from compiled
relations and drives the executor's state partition. Under
`vjp_method='single-step'`, ETP-routed parameter values remain outside the local
reverse-mode argument list because their gradients come from eligibility
traces, while plain-only parameter values remain differentiable and receive
their exact current-step VJP gradients. Compilation fails closed when one
ParamState path would belong to both categories.

## Notes on F-34

Fixed by the same compiled-path partition as F-33. In both VJP modes,
`dg_etrace_params` now contains only paths owned by compiled ETP relations and
`dg_non_etrace_params` contains plain-only parameter paths. DNI still consults
the graph when deciding which paths carry eligibility traces. The compiler
enforces that these path sets are exclusive before execution begins.

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
  `braintrace/_testing/models.py` constructors still draw from the global RNG.
