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
| F-30 | the IO-dim f-side de-biasing correction is indexed by `update()` call count, not by trace-step count, so it is exact only for single-step input | active, **preserved deliberately** — fixing it moves `pp_prop` / `OSTLFeedforward` gradients | `io_dim_vjp_test.py::TestBiasCorrectionTimeIndex`, both the structural claim and the finite-window consequence |
| F-31 | on a hidden group whose `y -> hidden` tail is not position-preserving, *no* within-group scope reaches BPTT — saturated SnAp / full within-group RTRL included | active, **structural** — pre-P3, affects every `recurrence_scope`; the compiler detects the tail and warns | `_algorithm/snap_n_test.py::TestConservativeFallbackEndToEnd::test_a_relabelling_tail_defeats_saturation_and_says_so`, against the `roll=0` control; measured below |
| F-32 | the coupled-scope transition is materialised as a dense `(*V, S, *V, S)` Jacobian, so `random_projection` costs `O((V·S)^2)` per group even though it only ever needs `D @ s_tilde` — one JVP would do | active, **deliberate**; a matrix-free `D_full` would need a JVP seam the executor does not have | `_algorithm/vjp_graph_executor_test.py::TestFullJacobianFlag::test_the_flag_changes_the_jacobian_shape` records the shape; UORO's docstring states the memory honestly |
| F-33 | under `vjp_method='single-step'`, every **plain** (non-ETP) parameter's gradient is exactly **zero** — not merely truncated | active, **structural** | `oracle_models_test.py::test_single_step_zeroes_every_plain_key`, and for the preset that cannot avoid it, `three_factor_test.py::TestItDoesSomethingMeasurably::test_every_plain_parameter_gets_an_exact_zero_which_is_f_33`; consequence documented in `ThreeFactor`'s docstring |
| F-34 | in the multi-step backward pass, `dg_etrace_params` carries the within-window gradient of **every** trainable parameter, plain ones included, while `dg_non_etrace_params` is empty — so the ETP/plain split cannot be read off which dictionary a parameter arrives in | active, **naming only**; the graph is the authority | `_algorithm/vjp_base_test.py::TestTheDefaultHooksAreInert::test_etp_routed_paths_reads_the_compiled_graph`; `_etp_routed_paths()` exists because of this |
| F-35 | a `train_synthetic_gradient` fit is valid only for the exact `(loss_fn, chunk_size)` pair it was given, and **nothing checks that deployment matches** — a mismatch is not degraded DNI but noise shaped like a cotangent, measurably worse than leaving DNI off | active, **documented not enforced**; the learner never sees the caller's objective, so it has nothing to compare against. Narrowed since: the helper no longer *introduces* a mismatch of its own — a sequence length that is not a multiple of `chunk_size` used to be truncated into a short final window, fitting one pair against a shorter future than any deployed window, and is now refused | `dni_test.py::TestALearnedSynthesiserHelps::test_training_on_the_wrong_window_size_is_worse_than_not_training` pins the `chunk_size` half; the `loss_fn` half is measured in B4's docstring (0.577 against 0.140 for `M ≡ 0`). Since the sequence driver, `chunk_size` is also a keyword on `etrace_grad`, so fit-vs-deploy mismatch is reachable without writing a window by hand — `sequence_test.py::TestAlgorithmInteractions::test_a_chunk_size_mismatch_degrades_a_fitted_synthesiser` fits at 1, deploys at 2 (2.0e-02 relative), with an untrained-synthesiser control at 1.3e-01 so the comparison is not between two no-ops. `reduction` is **not** a mismatch surface: it divides once after the scan and never enters the differentiated objective (`|mean − sum/T| = 7.5e-09`) |
| F-36 | DNI's pass-2 routing is **path**-granular, so a parameter used *both* as an ETP weight and plainly loses the future credit of its plain occurrence — the whole leaf is skipped because the path is ETP-routed | active, **structural**; the gradient dictionaries are keyed by path, not by occurrence, so occurrence-level routing is not expressible without changing every engine's `_solve_weight_gradients` contract | `dni_test.py::TestMixedRoutingIsPathGranular` |
| F-37 | DNI injects only at **hidden** exits, so future credit that would arrive through a *non-hidden* persistent state (a `ShortTermState` carried across windows) is dropped: `_exit_cotangent_grads` zeroes the `oth_states` slot of the injected template | active, **structural**; a synthesiser for non-hidden state would need its own group abstraction, and `hidden_groups` is the only slab the compiler provides | `dni_test.py::TestOtherStateCreditIsNotInjected`, on a fixture that actually has one — the earlier zeroing assertion ran on an empty tree and held vacuously |
| F-38 | a **standing** `ThreeFactor.modulator` is a plain Python attribute, so under `jit` it is captured at trace time: reassigning it does not retrace and the compiled closure keeps the old value | active, **documented**; the per-call `update(..., modulator=...)` form is unaffected and is what the docstring now steers to | `three_factor_test.py::TestTheStandingModulatorUnderJit` |
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

`_solve_IO_dim_weight_gradients` undoes the warm-up bias of the f-side
exponential smoothing by dividing by `1 - decay ** (running_index + 1)`
(`io_dim_vjp.py:493`). The comment above it derives the factor correctly for
`ε_f^t = α ε_f^{t-1} + (1-α) x_t`, but `t` there is a *smoothing step* whereas
`running_index` is a *call counter*: it is incremented once per `update()`
(`vjp_base.py:319`) while the trace scan runs once per sequence element
(`io_dim_vjp.py:934-943`). The two agree only when every call carries one step.

Measured on `tanh_rnn(n_in=3, n_rec=4)` at `decay=0.9`:

| observation | value |
|---|---|
| `running_index` after a single 6-step `MultiStepData` call | 1, not 6 |
| correction applied vs. required, that call | `1 - 0.9^1 = 0.100` vs. `1 - 0.9^6 = 0.469` |
| gradient deviation, full-window multi-step | 0 — that path is exact reverse-mode (F-23) |
| gradient deviation, finite window (`T` 6, chunk 2) | 6.8e-04 |

The last two rows are the useful ones. The mis-indexing is invisible wherever
the trace is not load-bearing, so it only biases the estimator on the paths
that actually use it, and the magnitude is set by how far `decay ** k` has
decayed at the first few chunk boundaries — largest on the first call, shrinking
as both counters grow.

Not fixed in P2: the phase's acceptance criterion is that no preset's gradients
move, and this correction is on `pp_prop`'s and `OSTLFeedforward`'s hot path.
The P2 golden values in `data/p2_golden.npz` therefore freeze the biased
numbers on purpose. A fix belongs in its own change, with its own goldens.

**Reachable from the public API since the sequence driver.** `etrace_grad(...,
chunk_size=k)` advances `running_index` once per *window*, so a user who
chunks a sequence now hits this without writing a `MultiStepData` call
themselves — the window is the only thing that was ever required, and
`chunk_size` makes it a keyword argument. Pinned from that direction by
`sequence_test.py::TestAlgorithmInteractions::
test_window_mode_lags_the_io_factorized_bias_correction`, which asserts three
things rather than the index alone: `running_index == T // k`, that the driver
reproduces a hand-written window loop *exactly* (so the driver adds no bias of
its own), and that forcing the index to the true trace-step count moves the
gradient by 2.1e-03 (so the bias is real and a future fix will surface here).

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

1. **It is detected, not silent.** The position analysis rejects the `slice`
   equation `roll` lowers to, widens to all-to-all, and emits
   `SNAP_PATTERN_CONSERVATIVE`. Nothing here is assumed and later violated.
2. **The diagnostic's wording was wrong, and was fixed in P3.** It read "the
   gradient stays correct", which is true of the *pattern* (a superset never
   drops a real dependency) but reads as a claim about the returned gradient.
   It now says the approximation is not degraded and points here. The warning
   is emitted for the SnAp path only; the same tail degrades `diagonal` and
   `coupled` too, without any warning at all, because those coordinates never
   ask for a position graph.

Not fixed: representing a non-position-preserving tail would require the trace
to carry the tail's own permutation alongside the neighbourhood, which is a
change to the trace layout rather than to any one algorithm. Until then, the
usable statement is the one in `SnAp`'s docstring: saturation equals BPTT on a
single-group model with an **elementwise** `y -> hidden` tail.

## Notes on F-33

Under `vjp_method='single-step'` the residual jaxpr yields an **empty**
`dg_etrace_params` and an empty `dg_non_etrace_params`, so `dG_weights[path]`
is never written for a non-ETP parameter and stays `None` — which surfaces as an
exact zero, not as a truncated approximation. Measured on `plain_and_etp_rnn`
(ETP `w`, plain `win`, `wout`): the single-step total gradient is exactly zero
for both plain keys while `w` is finite.

This is a property of the single-step window, not of any learning rule, but it
bounds what a single-step-only rule can train. `ThreeFactor` is single-step by
construction (see its docstring for why), so a model given to it will train only
its ETP-routed parameters. The rule for callers: route what you intend to
modulate through an ETP primitive.

## Notes on F-34

The gradient dictionaries the multi-step backward pass unflattens out of the
residual jaxpr are named after the *category* they were introduced for, not the
category they actually carry:

| Name | What it holds under multi-step |
|---|---|
| `dg_etrace_params` | the within-window reverse-AD gradient of **every** trainable parameter, ETP-routed and plain alike |
| `dg_non_etrace_params` | empty |

Measured on `plain_and_etp_rnn`: `dg_etrace_params` arrives keyed
`[('w',), ('win',), ('wout',)]` and `dg_non_etrace_params` keyed `[]`.

This matters for anything that has to treat the two categories differently. DNI
does: the injected exit cotangent must reach the plain parameters and must not
reach the ETP ones. A first implementation that split them by *dictionary* was
silently a no-op — the plain parameters never moved, because they were not in the
dictionary it was adding to. `_etp_routed_paths()` exists to ask the compiled
graph instead, which is the only authority on which parameters have a trace.

Not fixed: renaming the slots would touch every engine's
`_solve_weight_gradients` signature for no behavioural gain. The hazard is
recorded here and the helper makes the correct route the easy one.

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
