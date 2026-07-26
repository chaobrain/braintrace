# Roadmap: orthogonal axes for online-learning algorithms

Status: design, awaiting review
Baseline: commit `928219b` (OTTT / OSTTP / OTPE removed)
Target release: 0.3.0 — every phase below lands in it

## Why

`braintrace` is a framework for online learning in brain simulation. A framework
ships *general* mechanisms; it should not ship learning rules that work for one
operator type and one model shape.

The Nature Communications work established the model-agnostic abstraction
(AlignPre / AlignPost), the linear-memory rule (pp-prop), and the compiler that
generates online-learning code from a user-defined SNN. The claim that followed —
that fragmented rules (e-prop, OSTL, …) can be *described, implemented, compared
and deployed* in one compiler framework — was backed by separate hand-written
classes, three of which rejected most of the operator set. Those three are gone
(see Baseline). This roadmap makes the claim structural rather than incidental:
learning rules become coordinates in an explicit axis space, and every coordinate
works for every ETP primitive.

## Removal criteria

These governed the removals already made and govern every future addition. An
algorithm earns a standalone implementation only if it passes **both**:

1. **Model-agnostic.** It must hold for any ETP primitive (dense, conv, sparse,
   lora, element-wise) and any hidden-state dynamics. An implementation that
   whitelists primitives fails this test.
2. **Mathematically independent.** It must contribute a recursion or estimator
   the general engines cannot express as a configuration. A named class whose
   coordinates coincide with an existing one contributes nothing.

## Baseline: what the code looks like now

### What was removed, and why (history)

`OTTT`, `OSTTP` and `OTPE` all failed criterion 1: each whitelisted dense-matmul
primitives (`_SUPPORTED_PRIMITIVES = {etp_mm_p, etp_mv_p}`), raised
`NotImplementedError` for lora / sparse / conv / element-wise relations, and was
single-step only. `OTPE` additionally assumed a single global time constant, was
feed-forward only, was gradient-exact for one hidden layer, and rejected
`num_state > 1` outright — ruling out ALIF and any adaptation variable; its own
docstring called its derivation *"narrower than OTTT's"*. `OSTTP` bound `B_list`
to the HiddenGroup count and threaded `y_target` through a bespoke path.

Removed with them: `PresynapticTrace` (only `OTTT` used it) and the internal
`extract_y_target` (only `OSTTP` used it). Kept deliberately: `KappaFilter`,
`FixedRandomFeedback`, and the `_get_update_aux` per-call side channel in
`vjp_base.py` — `OSTTP` was its only consumer, but it is the general hook that
P4's `modulatory` signal needs.

### Surviving algorithms

| Class | Relationship |
|---|---|
| `ParamDimVjpAlgorithm` | engine — per-parameter trace, O(P·H) |
| `IODimVjpAlgorithm` | engine — input/output-factorized trace, O(I+O) |
| `D_RTRL` | canonical name for `ParamDimVjpAlgorithm`, no overrides |
| `pp_prop` / `ES_D_RTRL` | canonical name for `IODimVjpAlgorithm`, no overrides |
| `EProp` | `ParamDimVjpAlgorithm` + κ-filter + optional random feedback; overrides `init_etrace_state`, `reset_state`, `_compute_learning_signal`, `_solve_weight_gradients` |
| `OSTLRecurrent` | `ParamDimVjpAlgorithm` + `_include_recurrent_mixing = True` |
| `OSTLFeedforward` | `pp_prop` with `decay_or_rank` defaulting to `1e-6` |

### The engine's extension surface

`ETraceVjpAlgorithm` is already a template-method class. Its override hooks are
the natural mounting points for the axes:

| Hook | What it decides |
|---|---|
| `init_etrace_state` | trace allocation → factorization |
| `_update_etrace_data` / `_make_etrace_stepper` | trace recursion in time |
| `_solve_weight_gradients` | how trace × signal contracts to a weight gradient |
| `_compute_learning_signal` | where the learning signal comes from |
| `_get_update_aux` | per-call external data (reward, target, …) side channel |
| `_get_etrace_data` / `_assign_etrace_data` | trace (de)serialisation — abstract |
| `_include_recurrent_mixing` (class attr) | how much hidden↔hidden coupling enters the trace |

### Learning-rule axes vs execution options

The constructors already carry knobs that are **not** learning-rule axes and
must stay out of `ETraceConfig`: `vjp_method` (`'single-step'` / `'multi-step'`),
`fast_solve`, `trace_dtype`, `chunked_trace`, `control_flow`. These choose *how*
a rule is executed, not *which* rule it is. Conflating them would make the
benchmark suite enumerate a configuration space that is mostly execution
variants.

### The oracle inventory

`_algorithm/oracle.py` already provides `bptt_param_gradients`,
`finite_difference_param_gradients`, `online_param_gradients`,
`chunked_online_param_gradients`, `online_param_gradients_singlestep_naive`,
`assert_param_gradients_close`, plus the direction metrics
`cosine_similarity` / `sign_agreement` / `relative_magnitude` /
`assert_direction_aligned`. This is enough to serve every acceptance paradigm
below; no new oracle family is required.

## The axis space

### Axis 1 — `trace_factorization` (spatial factorization → memory)

| Value | Trace shape | Memory | Status |
|---|---|---|---|
| `per_param` | `(param_shape, H)` | O(P·H) | exists (`ParamDimVjpAlgorithm`) |
| `io_factorized` | `ε_x ⊗ ε_f` | O(I+O) | exists (`IODimVjpAlgorithm`) |
| `random_projection` | rank-1 random factors `(s̃, θ̃)` | O(P+H), **unbiased** | new (P4) |

### Axis 2 — `temporal_recursion` (how the trace advances in time)

- `jacobian` — `ε ← D·ε + …` (D-RTRL, OSTL 'with-H', e-prop)
- `scalar_leak` — `R̂ ← λ·R̂ + …`. The additive term is the same per-primitive
  hidden→weight Jacobian contribution `jacobian` already uses, so swapping the
  recursion is primitive-agnostic. This is what generalises out of the removed
  OTPE; the restriction stack that made OTPE non-general does not come with it.
- `none` — no temporal accumulation (feedforward regime)

Under `io_factorized` this axis is a **pair** `(x-side, f-side)`.
`IODimVjpAlgorithm` currently shares one `α` across `ε_x` and `ε_f` (derived by
`_format_decay_and_rank`); splitting it into two independent decays is the only
change in this roadmap that touches a numerical path in existing code. With the
split, the removed OTTT's coordinate — x-side leak, f-side instantaneous —
becomes reachable *and* primitive-generic, which the deleted implementation
never was.

### Axis 3 — `recurrence_scope` (how much hidden↔hidden coupling enters the trace)

**This axis already exists in the code as a boolean.** `_include_recurrent_mixing`
on `ETraceVjpAlgorithm` flows into `find_hidden_groups_*` and sets
`HiddenGroup.is_diagonal_recurrence = not include_recurrent_mixing`. Axis values,
naming the two that exist today:

- `diagonal` (`_include_recurrent_mixing = False`; default for `D_RTRL`,
  `pp_prop`, `EProp`) — recurrent ETP *mixing* primitives are excluded from the
  transition, so it is position-diagonal by construction and the cheap
  column-sum Jacobian (`jacrev_last_dim`) is exact.
- `coupled` (`_include_recurrent_mixing = True`; `OSTLRecurrent`) — those
  primitives are traced into the transition, the recurrence becomes
  cross-position coupled, and the true per-position block diagonal is extracted
  explicitly (`block_diagonal_last_dim`). This retains strictly more than
  `diagonal` — the diagonal entries now include contributions routed through the
  recurrent weight, which `diagonal` drops entirely.
- `sparse_n` — the SnAp-n scale added in P3.

SnAp-n is therefore a *generalisation of an existing knob* rather than a new
representation invented from scratch, which is what makes P3 tractable.

**Settled in P3: `coupled` *is* SnAp-1, and `diagonal` sits below the scale.**
The open question was whether `coupled` lands on a point of the `n` scale or
stays a sibling beside it. It lands on `n = 1`: SnAp-1 keeps only the
instantaneous term `∂h_p/∂θ_p` and propagates it through the *self* block of
`D`, which is exactly what `coupled` computes once the recurrent mixing
primitive is in the transition and the per-position block diagonal is taken.
`SnAp(model, n=1)` therefore canonicalises to `recurrence_scope='coupled'`
rather than being a second spelling of it
(`snap_n_test.py::TestCoordinates::test_snap_1_canonicalises_to_the_coupled_coordinate`).

`diagonal` is *not* SnAp-0 or any other point on the scale. It deletes the
recurrent mixing primitive from the transition *before* differentiating, so it
is not a sparsification of `J = ∂h/∂θ` at all — it is a different transition
operator. The scale runs `coupled` (= SnAp-1) → `sparse_n(2)` → … →
`sparse_n(≥ diameter)` (= full within-group RTRL = BPTT); `diagonal` sits beside
its lower end, cheaper and structurally different.

### Axis 4 — `learning_signal` (where the signal comes from)

- `symmetric` — `∂L/∂h` back-propagated through the readout (default)
- `random_feedback` — fixed random projection; promotes `FixedRandomFeedback`
  from helper to first-class strategy
- `modulatory` — three-factor: a scalar / low-dimensional neuromodulatory signal
  (TD error, reward-prediction error) times the trace (new, P4)
- `bootstrapped` — synthetic gradient / DNI: a learned estimate of the future
  gradient (new, P4)

DRTP / target projection is deliberately **not** a value here. It was removed
with OSTTP and is not reintroduced.

### Axis 5 — `trace_filter`

`none` · `kappa` (e-prop's low-pass `ē ← κ·ē + ε`; `EProp` filters the trace
internally today, `KappaFilter` remains a separate user-facing utility)

### Axis 6 — `update_schedule`

`per_step` · `window(k)` · `sequence_end`

### Compatibility matrix

The axes are not fully orthogonal, and pretending otherwise produces silently
wrong gradients:

- `random_projection` carries UORO's own rank-1 update and normalisation; it
  cannot be paired with an arbitrary `temporal_recursion`.
- `recurrence_scope` beyond the diagonal end demands a `temporal_recursion` that
  actually propagates a Jacobian; pairing `n > 1` with `scalar_leak` or `none`
  is meaningless.
- `io_factorized` constrains the contraction in `_solve_weight_gradients`; the
  legal `(factorization, signal)` shapes the current implementations encode
  implicitly must become explicit.

`ETraceConfig` validates the combination at construction and raises an error
naming the legal pairings. The matrix is explicit data, not scattered `if`s.

## Coordinates of the surviving algorithms

Asserted field-by-field in
`_algorithm/tests/axis_acceptance_test.py::test_preset_coordinates_match_the_spec_table`,
so the table cannot drift from the code.

| Algorithm | factorization | recursion | recurrence_scope | signal | filter | decay |
|---|---|---|---|---|---|---|
| `D_RTRL` | per_param | jacobian | diagonal | symmetric | none | — |
| `pp_prop` | io_factorized | (scalar_leak, jacobian) | diagonal | symmetric | none | (α, α) |
| `OSTLRecurrent` | per_param | jacobian | **coupled** | symmetric | none | — |
| `OSTLFeedforward` | io_factorized | (scalar_leak, jacobian) | diagonal | symmetric | none | (1e-6, 1e-6) |
| `EProp` | per_param | jacobian | diagonal | symmetric \| random_feedback | **kappa** | — |
| `SnAp(n≥2)` | per_param | jacobian | **sparse_n**(n) | symmetric | none | — |
| `SnAp(n=1)` | per_param | jacobian | **coupled** | symmetric | none | — |

`SnAp` was added in P3; its two rows are pinned by
`snap_n_test.py::TestCoordinates` rather than by the P2 preset table, because
`n = 1` canonicalises onto `OSTLRecurrent`'s coordinate and the preset table
asserts a bijection between presets and coordinates.

Two entries were corrected during P2. Under `io_factorized` the recursion is a
**pair**: the x-side (presynaptic input trace) never involves a Jacobian, so the
scalar shorthand `jacobian` canonicalises to `(scalar_leak, jacobian)`.

And `OSTLFeedforward` is **not** `recursion = none`. Its default
`decay_or_rank=1e-6` leaves both recursion terms structurally present with a
negligible coefficient — which is a different coordinate from dropping them. The
exact `none` coordinate is `OSTLFeedforward(model, decay_or_rank=0.0)`. The
default was left alone: changing it would move the preset's gradients by ~1e-6
relative, and a numerical change must not ride along inside a refactor whose
acceptance criterion is that numerics do not move.

`update_schedule` is omitted from the table: every surviving algorithm is
`per_step`, so the column carries no information today. It becomes load-bearing
only once `window(k)` / `sequence_end` are implemented.

Every surviving algorithm occupies a distinct coordinate. `OSTLRecurrent` is
**not** a `D_RTRL` alias — the `recurrence_scope` column is what separates them,
and it is the column an earlier draft of this document was missing. That draft
argued `OSTLRecurrent` should be deleted as a zero-information alias; the
argument was wrong, and the class stays.

## Phases

All land in 0.3.0. Ordering reflects dependencies, not separate releases.

### P0 — Removal — **done** (`928219b`)

Deleted `ottt.py` / `osttp.py` / `otpe.py`, their tests, and their private
helpers; cleaned every reference across source, tests, docs and `AGENTS.md`.

Coverage was **repointed, not dropped**, wherever the assertion was about a
general property: the approximate-gradient descent backstop now runs on
`pp_prop(rank=1)` and `EProp(feedback='random')`; the one-step D_RTRL
equivalence tests now use `EProp(kappa_filter_decay=0)` (verified passing);
`public_api_test.py` gained a guard that the removed names stay gone. The
direction-alignment metric helpers were retained as the basis of P5.

Verified: `pytest braintrace/` → 2062 passed, 1 skipped; `mypy braintrace` clean.

### P1 — Verification harness and the limitation list — **done**

Retitled during implementation. P1 was scoped as compiler work; measurement
showed the compiler already passes every one of its targets, and that the
instrument was what failed. Full spec:
[`2026-07-25-p1-verification-harness.md`](2026-07-25-p1-verification-harness.md).

Delivered: negative-control oracle helpers (`assert_model_is_live`,
`assert_gradients_differ`) and pytree/unit-aware gradient comparison;
documented window semantics on every oracle entry point; deterministic and live
SNN model specs; an axis-discrimination meta-test; gradient-correctness tests on
realistic SNN models (multi-timescale, per-neuron heterogeneous leaks,
`num_state` 1–5, E/I populations); the conv-bias IO-dim fix (F-26); and the
in-tree findings list at
[`2026-07-25-known-limitations.md`](2026-07-25-known-limitations.md).

`hidden_group.py` was **not** modified — no defect was found in its Jacobian
path — so P3 inherits today's representation and risk 2 below is retired rather
than mitigated.

**Acceptance:** met. The reconstructed limitation list is committed under
`docs/specs/`; every item has a passing test or a documented scope boundary; the
one skipped test (F-22) is retired rather than deferred, and the suite now
carries no `skip` or `xfail` markers at all.

Verified: `pytest braintrace/` → 2119 passed, 4 deselected (`diagnostic`, which
pass separately); `mypy braintrace` clean.

### P2 — Axis decomposition — **done**

Spec: [`2026-07-25-p2-axis-decomposition.md`](2026-07-25-p2-axis-decomposition.md).

Turn the six axes into code: strategy protocols, `ETraceConfig` and its
compatibility matrix, engine hooks rewired to delegate to strategies, and the
three literature presets (`EProp`, `OSTLRecurrent`, `OSTLFeedforward`) rewritten
as thin factories over `ETraceConfig`. Includes the `IODimVjpAlgorithm` decay
split, and lifting `_include_recurrent_mixing` from a class attribute to an axis
value.

Execution options (`vjp_method`, `fast_solve`, `trace_dtype`, `chunked_trace`,
`control_flow`) stay constructor parameters and stay out of `ETraceConfig`.

**Acceptance:**
- Element-wise equality against golden values frozen *before* the refactor, for
  all five surviving algorithms — frozen and compared through a **finite-window**
  oracle path (`chunked_online_param_gradients`, `chunk_size` < `T`). The
  full-window multi-step path returns BPTT for every algorithm regardless of its
  axis coordinates (F-23), so golden values captured there would be identical
  across all five and would guard nothing. Guard each comparison with
  `assert_gradients_differ` between any two presets that differ in an axis.
- `decay_or_rank=0.9` equals the new two-sided `(0.9, 0.9)` element-wise.
- Illegal axis combinations raise a readable error naming the legal pairings,
  with test coverage.
- Each preset's coordinates are asserted against the table in this document, so
  the table cannot drift from the code.

**Delivered.** `braintrace/_algorithm/axes.py` (`ETraceConfig`, canonicalisation,
the eight-rule compatibility matrix), the Jacobian substitution and the two
feature lifts in `vjp_base.py`, the κ-filter in `param_dim_vjp.py`, the decay
split in `io_dim_vjp.py`, `EProp` / `OSTLRecurrent` / `OSTLFeedforward` reduced
to coordinates, and an `ETraceConfig` overload on `braintrace.compile`.

Tests: `axes_test.py` (64), `tests/axis_golden_test.py` (24 frozen gradients over
8 cases × 2 models, covering all three trace paths), `tests/axis_acceptance_test.py`
(29). Verified: `pytest braintrace/` → 2240 passed, 4 deselected (`diagnostic`,
as in P1); `mypy braintrace` clean.

One deviation from the spec: the κ-filter *state* is allocated in
`ParamDimVjpAlgorithm` rather than in the base. The spec put it in the base
"sized from `self._get_etrace_data()`", but matrix rule 1 already confines
`trace_filter='kappa'` to `per_param`, and the filter mirrors `etrace_bwg`'s
pytree — so allocating it beside the thing it mirrors avoids the base reaching
into an engine-specific structure. The lifecycle contract the spec actually cares
about (base concrete, engines chain via `super()` as their last statement, no
silent degradation) is unchanged.

### P3 — SnAp-n: generalise `recurrence_scope` — **done**

Spec: [`2026-07-25-p3-snap-n.md`](2026-07-25-p3-snap-n.md).

Menick et al. 2021. Derive the n-step influence sparsity pattern automatically
from the compiler's jaxpr hidden→hidden reachability graph — the thing a
compiler can do that a hand-written library cannot.

Concretely: replace the boolean `is_diagonal_recurrence` with an n-valued scope,
keep `jacrev_last_dim` as the `n = 1` fast path and `block_diagonal_last_dim` as
today's coupled path, and add the intermediate sparse representation for
`1 < n < diameter`.

**Acceptance (two-sided squeeze):**
- `n = 1` equals the current `D_RTRL` element-wise, compared through a
  finite-window path (regression guard). On the full-window path this holds for
  every algorithm and so proves nothing (F-23).
- Whatever configuration expresses `recurrence_scope = coupled` after the
  refactor equals the current `OSTLRecurrent` element-wise — again through a
  finite-window path — regardless of whether that configuration turns out to be
  a point on the `n` scale or a sibling value beside it.
  `axis_discrimination_test.py` pins that `D_RTRL` and `OSTLRecurrent` *are*
  distinguishable on that path, so this comparison has content.
- `n ≥ graph diameter` equals the **BPTT oracle** element-wise, on models whose
  hidden-state coupling the compiler fully captures. Full RTRL and BPTT compute
  the same total gradient, so `oracle.py` is the correct instrument and no
  separate full-RTRL reference is needed. (A full-RTRL reference is worth writing
  only to assert the influence matrix `dh^t/dθ` itself when localising a
  divergence between trace and learning signal — optional debugging aid.)
- Measured memory curve, monotone in n.

**Delivered.** `braintrace/_compiler/position_graph.py` (per-axis adjacency
analysis, closure, neighbourhood index, budget guard), the `snap_anchor`
capability in `_op/_registries.py` plus its nine per-primitive declarations,
`HiddenGroup.snap` / `.trace_state_width` / `widened_block_jacobian` /
`widen_instant_term` / `gather_learning_signal` in `_compiler/hidden_group.py`,
the widened-trace plumbing in `_algorithm/param_dim_vjp.py`, two guardrails in
`_algorithm/vjp_base.py`, the `sparse_n` axis value and its matrix rules in
`_algorithm/axes.py`, three new oracles (`sparse_ring_rnn`,
`sparse_ring_two_state_rnn`, `rolled_tail_rnn`), and the `braintrace.SnAp`
preset.

122 new tests: `_algorithm/snap_n_test.py` (48 acceptance),
`_compiler/position_graph_test.py` (41), plus 10 in `hidden_group_test.py`,
10 in `_registries_test.py` and 13 in `axes_test.py`.

Both curves, measured on `sparse_ring_rnn(n_rec=6)` (diameter 5) through
`chunked_online_param_gradients(chunk_size=3)`, T=8, against the BPTT oracle on
the ETP parameter only:

| n | K | trace scalars | rel. deviation from BPTT |
|---|---|---|---|
| 1 | 1 | 12 | 2.32e-01 |
| 2 | 2 | 24 | 4.96e-03 |
| 3 | 3 | 36 | 5.93e-04 |
| 4 | 4 | 48 | 2.22e-04 |
| 5 | 5 | 60 | 8.03e-08 |
| 6 | 6 | 72 | 8.03e-08 |
| 7 | 6 | 72 | 8.03e-08 |

Memory is exactly linear in `K` and saturates with it; accuracy is monotone and
reaches round-off at `n = diameter`. A *dense* recurrent weight has diameter 1
and saturates at `n = 2`, which is why the ring — and not `tanh_rnn` — is the
reference model for the scale.

Two deviations from the spec. First, the widening hooks live at the innermost
consumers in `param_dim_vjp.py` rather than behind the
`vjp_base._transform_trace_inputs` hook the spec proposed: the `df` widening and
the learning-signal gather need the `HiddenGroup` they belong to, and the base
hook does not have it without threading group identity through a signature that
every engine shares. Second, `n = 1` canonicalises to `recurrence_scope='coupled'`
instead of `sparse_n` with `sparse_n=1`; the spec left this open, and collapsing
it keeps one coordinate per rule (see Axis 3 above).

**Adversarial review.** An independent review of the finished implementation
raised twelve findings; the ones that changed code were a soundness blocker in
the reachability closure (integer overflow under-approximating the
neighbourhood), the budget being charged against the trace rather than the block
Jacobian that dominates it, the anchor check lapsing at `n = 1`, a missing
guard on the public `compile_etrace_graph` entry point, and a diagnostic whose
"the gradient stays correct" wording over-claimed. On the test side it added an
independent masked-RTRL oracle for the interior orders, gradient execution for
all eleven anchored primitives at a saturating order, and replaced two
tautological assertions. Lessons 26–30 record what generalises; F-31 in the
limitations list records the one behaviour that was confirmed as a pre-existing
property rather than a P3 regression.

### P4 — UORO, three-factor and DNI — **done**

Spec: [`2026-07-25-p4-uoro-modulatory-dni.md`](2026-07-25-p4-uoro-modulatory-dni.md).

Delivered: `RandomProjectionVjpAlgorithm` + `UORO`, the `modulatory` branch +
`ThreeFactor`, the `_inject_exit_cotangent` two-pass hook + `SyntheticGradient`,
`DNI` and `train_synthetic_gradient`. Three new findings (F-32, F-33, F-34) and
lessons 31–38 below. What each acceptance criterion actually established, and
what it did not, is recorded in the three suites' docstrings; the honest summary
is that UORO's unbiasedness and DNI's *routing* are pinned exactly, while DNI's
*estimate quality* is a demonstration rather than a property.

Three additions that share P2's axes and can proceed in parallel once P2 lands.

**UORO** (Tallec & Ollivier 2018) — adds `random_projection` with UORO's rank-1
update and normalisation, complementing the existing biased diagonal
approximations. KF-RTRL / OK are optional lower-variance siblings.

**`modulatory`** — three-factor learning: trace × neuromodulatory signal (TD
error, reward-prediction error), enabling reward-based e-prop and online policy
gradient. Uses the retained `_get_update_aux` side channel. The injection path
must not bind to HiddenGroup count or readout shape — that binding is precisely
what made OSTTP non-general.

**`bootstrapped`** — DNI / synthetic gradients (Jaderberg 2017): a learned
bootstrap for the future-loss gradient every online rule truncates away.
Requires an auxiliary network with its own training loop.

**Acceptance** — three different paradigms in one phase, do not mix them up:
- UORO: statistical. Fixed model, fixed sequence, N seeds; deviation of the mean
  gradient from BPTT shrinks as 1/√N (confidence-interval test). A single run
  asserts only shape, finiteness, absence of NaN. **This needs statistical test
  infrastructure the repo does not have** — count it in the phase's cost.
- `modulatory`: must equal `symmetric` element-wise when the modulatory signal is
  set to `∂L/∂h`. Plus a test whose HiddenGroup count differs from the signal
  dimension, pinning that the OSTTP binding mistake was not recreated.
- `bootstrapped`: must equal `symmetric` when the synthesiser output is pinned to
  the true value. Plus an end-to-end RL smoke test.

### P5 — Unified benchmark suite

Enumerate the `ETraceConfig` space and, for a fixed model, report gradient cosine
similarity and relative deviation against BPTT, peak memory, per-step wall time,
and task metrics. Machine-readable output (JSON/CSV) plus a reproducible script,
reusing the direction metrics already in `oracle.py`.

Should also retire the deferred F-22 finding: the SNN multi-population model zoo
that F-22 says is needed to expose the real bias of the IODim rank / ES decay /
random-feedback approximations is the same model zoo this suite needs.

Grows as P3 and P4 land.

## Three acceptance paradigms

Mixing these up is the most likely way to get this roadmap wrong:

| Paradigm | Applies to | Instrument |
|---|---|---|
| Element-wise equality | P2 refactor, SnAp-1, SnAp-coupled, SnAp-∞, degenerate `modulatory` / `bootstrapped` | golden values / BPTT oracle |
| Statistical convergence | UORO and unbiased siblings | 1/√N confidence interval over seeds |
| Direction + task metric | genuinely approximate configurations | cosine / sign agreement, then descent and RL smoke tests |

The `AGENTS.md` taxonomy (exact vs approximate) stays valid and gains the
statistical class.

## Risk register

1. **P2 changes numerics silently.** Mitigation: freeze reference gradients for
   all five surviving algorithms as golden values *before* touching the engine;
   assert element-wise equality after, **through a finite-window oracle path**.
   Golden values captured through the full-window path are identical across all
   five algorithms (F-23) and would not detect any change. Separately, the
   `IODim` decay split must prove `decay_or_rank=0.9` equals the new
   `(0.9, 0.9)` element-wise.
2. ~~**P1 and P3 collide in `hidden_group.py`.**~~ **Retired.** P1 found no
   defect in the hidden→hidden Jacobian path and did not modify the file, so
   there is no collision: P3 inherits today's representation directly.
3. ~~**The P1 scope is not actually written down.**~~ **Resolved by P1.** The
   list is now in-tree at
   [`2026-07-25-known-limitations.md`](2026-07-25-known-limitations.md), verified
   against the suite rather than transcribed, and the stale `dev/` docstring
   cross-reference is gone. The residual form of this risk for later phases is
   different and worse — see lesson 2 below: a criterion can be *written down*
   and still be vacuous if it names the wrong oracle path.
4. **Statistical tests are flaky in CI** (P4). Mitigation: fixed seeds, generous
   intervals, a separate slow-test marker.
5. **`modulatory` recreates OSTTP's plumbing mistake** (P4). Mitigation: the
   injection path must not bind to HiddenGroup count or readout shape; the
   mismatched-dimension test above is the guard.
6. **0.3.0 carries the whole roadmap.** All breaking changes and all new
   algorithm families ship together. Mitigation: phases merge independently
   behind the axis interfaces, and P5 runs on every merge so regressions surface
   per phase rather than at release.

## Lessons learned during implementation

Items 1–9 were recorded during P1, against commit `bc153da`; items 10–17 during
P2, against `156d058`; items 18–30 during P3, against `100b5be` (18–25 while
implementing, 26–30 while working through the adversarial review). These are the
things the roadmap got wrong or could not have known, kept here because later
phases rest on them.

1. **The instrument was the defect, not the compiler.** P1 was scoped as
   compiler work on multi-timescale and heterogeneous populations. Every one of
   those targets already passed: HiddenGroups with `num_state` up to 5,
   per-neuron heterogeneous `tau_mem` and `tau_a`, multi-timescale synapses
   (`tau_mem` / `tau_syn` / `tau_a` / `tau_std` / `tau_f` / `tau_d`), and E/I
   population splits all reproduce BPTT. Measure before scoping a phase around
   a suspected defect.

2. **A full-sequence multi-step VJP is BPTT, so it cannot see any axis**
   (F-23). `D_RTRL`, `OSTLRecurrent`, `EProp` at any `kappa_filter_decay` and
   `pp_prop` at any `decay_or_rank` return bitwise-identical gradients through
   `online_param_gradients`. This is correct semantics — no truncation, nothing
   to approximate — but it silently voids any assertion whose subject is a
   learning-rule axis. `chunked_online_param_gradients` documented this from the
   start; the tests written against the other entry point did not heed it. **Any
   acceptance criterion in this roadmap phrased as "equals X element-wise" must
   name a finite-window path.**

3. **A wrong cause propagates further than a wrong measurement.** F-21 observed
   real behaviour and attributed it to the model being a single-HiddenGroup rate
   model. F-22 then deferred an entire work item — an SNN multi-population model
   zoo — on that attribution, and P5 inherited it. Building exactly that model
   (`ALIF_ExpCo` with an E/I split, 3 relations, `num_state` 5) reproduced the
   same bitwise exactness, so no zoo could ever have helped. The fix was one
   parameter: `chunk_size`.

4. **Vacuous tests look like passing tests.** Three distinct ways a gradient
   assertion in this repository can assert nothing: the reference gradient is
   zero (a silent SNN — at unit input scale the conductance-based models never
   reach threshold); the model differs between the two sides of the comparison
   (`_etrace_model_test.py` constructors draw from the unseeded global RNG, so
   `factory()` returns a different network per call); or the oracle path cannot
   see the knob under test (F-23). Spiking is *not* sufficient for the first,
   and the live window is bounded **above** as well as below — driven hard,
   `ALIF_Delta` keeps spiking at rate 0.60 while its surrogate derivative
   saturates and the BPTT gradient returns to exactly zero. Hence
   `assert_model_is_live` keys on gradient norm rather than spike rate, and
   every new axis assertion carries `assert_gradients_differ`.

   This bit twice, and the second time it was self-inflicted: a probe written
   *while investigating F-29* silently reported 0.0 deviation for every
   configuration because the model under it had no input drive. Run the liveness
   helper inside throwaway probes too, not just inside committed tests.

5. **A threshold below round-off is not an assertion.** The first version of the
   F-22 replacement test demanded a relative deviation of only `1e-9` between an
   approximate and an exact algorithm. float32 round-off on these gradient trees
   is around `1e-8`, so the test would have passed on numerical noise for any
   configuration at all — and it failed only because one configuration was
   *even quieter* than noise. Axis assertions now use `1e-6` against deviations
   of `1e-3` to `1e-1`.

6. **"Approximate" is a claim about a configuration, not about a class**
   (F-29). `pp_prop(decay_or_rank=1)` is nominally an approximation, but an int
   rank maps through `decay = (rank - 1) / (rank + 1)`, so rank 1 means decay 0,
   the presynaptic EMA collapses to the exact current input, and on a model whose
   only ETP relation is the recurrent weight it reproduces exact `D_RTRL` to
   round-off — across chunk sizes, sequence lengths, widths and recurrent
   spectral radii from 0.25 to 4.97. It *is* a real approximation (0.31 to 0.79)
   once the relation's presynaptic input carries an external component, as every
   SNN spec's does. Consequence for P2 and P5: when picking a positive control
   for approximation error, verify the chosen configuration actually deviates on
   the chosen model instead of assuming its class name guarantees it.

7. **`xy_to_dw` rules may return un-reduced, per-position leaves.** Conv's rule
   deliberately defers the spatial sum of the bias Jacobian to `_conv_dt_to_t`,
   which only the param-dim path calls. Any new solver that consumes `xy_to_dw`
   directly must reduce produced leaves to the parameter's own shape (F-26).
   This matters for P2: the axis strategies will introduce new contraction
   paths.

8. **mypy's `exclude` does not apply to files reached by an import.** The oracle
   SNN specs wrap the layer classes in `_etrace_model_test.py` rather than
   duplicating them, which is the right call for keeping one definition of each
   model — but it means a non-excluded module imports an excluded one. mypy then
   type-checks that test file anyway and reported 52 pre-existing
   `brainpy.state.*` attribute errors, turning a clean gate red without any
   change to the file itself. Fixed with a scoped `ignore_errors` override so the
   existing "test files are outside the typed surface" policy holds however the
   file is reached. Watch for this whenever shipped code imports a `_test.py`
   module.

9. **The removed algorithms took findings with them, and left stale
   cross-references behind.** F-07/F-08/F-09/F-19/F-20 died with OTTT / OTPE /
   OSTTP, and the `AGENTS.md` prose item "target-signal threading under JIT"
   died with `OSTTP`'s `y_target` path. Meanwhile a docstring still pointed at
   `dev/superpowers/specs/...`, a path that is gitignored and absent. Findings
   lists must live in-tree.

### From P2

10. **An axis is a property of a (rule, model) pair, not of a rule.** The single
    most repeated finding of P2, hit three separate times before it was
    internalised. `temporal_recursion` is *invisible* on `tanh_rnn` under
    `per_param`: that model's only hidden→hidden path runs through its recurrent
    ETP weight, which `recurrence_scope='diagonal'` excludes from the transition
    by construction, so `D` is identically zero and `none` — which substitutes
    zeros — changes nothing. Likewise `OSTLRecurrent` collapses onto `D_RTRL` on
    `two_state_rnn` (its v/a coupling is hand-written arithmetic, not an ETP op,
    so `coupled` has no mixing primitive to trace), and `OSTLFeedforward`
    collapses onto `D_RTRL` on `tanh_rnn` (F-29). A test asserting "this knob
    changes the gradient" must name the model it was measured on, and the
    degenerate pairs must be asserted to *stay* degenerate — a pair that starts
    differing is as much a regression as one that stops.

11. **Measure the axis before writing the test that asserts it.** Every
    numerical claim in the P2 spec was measured first, and the ones that
    surprised were the valuable ones: `io_factorized` + `coupled` is legal and
    live (3.7e-04) though nothing in-tree exercised it; random feedback grafted
    onto the IO-dim engine runs and moves the gradient by 5.4e-01, which is what
    made the lift worth specifying rather than merely tidy. Writing the test
    first would have produced two plausible assertions, one of which was wrong.

12. **`leaky_linear` is the positive control the axis work needed.** Its
    recurrence `h_t = 0.9·h_{t-1} + matmul(x, w)` has hidden→hidden Jacobian
    `0.9·I` *exactly*, so substituting `scalar_leak` at the model's own leak must
    reproduce the true Jacobian bitwise. That pins both halves at once — the
    substitution is installed on this path, and the array it installs is
    numerically right. "The knob changes something" cannot distinguish a correct
    substitution from a corrupting one; this can. Later phases replacing the
    transition operator should reach for the same construction.

13. **Vary one thing.** The exactly-once substitution test first compared
    `chunked_trace=False` *with* substitution against `chunked_trace=True`
    *without*, and failed at 3.3e-08 — which is reassociation between the two
    trace paths, not a doubled substitution. Held at fixed `chunked_trace`, the
    substitution is bitwise exact on both. The mistake also produced a real
    finding: the two paths are bitwise identical on `tanh_rnn` and
    `two_state_rnn` but not on `leaky_linear`, so "bitwise" is a model-dependent
    accident and the honest assertion is agreement to round-off.

14. **A lifted feature inherits every limitation its private version had.**
    `_include_recurrent_mixing` was a private class attribute that only
    `OSTLRecurrent` set, so `scan_descent.py` hard-coding it to `False` for
    descended bodies was a defensible internal decision. The moment it became
    the public `recurrence_scope` axis it became a trap: ask for `coupled`,
    silently get `diagonal` inside the scan. P2 added a guard that raises. Before
    making anything private public, enumerate where the private version was
    quietly overridden.

15. **A configuration that cannot be honoured must fail loudly.** The obvious
    implementation of the random-feedback lift fails *silently*: an unallocated
    feedback dict is indistinguishable from `symmetric` at the hook, so the
    algorithm computes a different learning rule than the one requested and
    every test still passes. This is lesson 4's vacuity failure with a new face,
    and it is the reason the base hook now raises rather than falling through.

16. **Canonicalise before validating, and derive the canonical form from the
    arithmetic.** `α_f = 0` collapses the f-side to `none` regardless of what
    the recursion field says — because `_expon_smooth(old, new, 0)` returns
    `new`, so the Jacobian is never applied. That is a fact about the code, not
    about the vocabulary, and it was only found by reading the smoothing
    primitive. But the same collapse must *not* apply to `per_param` +
    `jacobian` + `decay=0`, which is a user error rule 4 exists to catch —
    canonicalising it away would swallow the error. Canonicalisation rules need
    a scope, not just a condition.

17. **Do not fix a bug inside a refactor.** F-30 (the IO-dim bias correction is
    indexed by `update()` call count, not trace-step count) surfaced during the
    design review of the P2 spec. It is real, reproduced, and worth ~6.8e-04 on
    the finite-window path — and it was deliberately *not* fixed, because P2's
    entire acceptance criterion is that gradients do not move. A numerical
    correction and a refactor cannot land together, or neither can be verified.
    It is recorded in the findings list with a reproduction and pinned by a test
    that asserts the current, biased behaviour.

### From P3

18. **An approximation whose error does not fall when you spend more memory on
    it is not "converging slowly" — it is wired backwards.** The first working
    SnAp-n produced a flat error curve: `2.3167e-01` at every `n` from 1 to 4,
    then exactly round-off at `n = 5`. Every structural test passed. The cause
    was the direction of the neighbourhood: `A[p, q]` reads "h_p depends on
    h_q", so the trace slot anchored at `p` must cover **column** `p` of the
    closure — the positions `p` *influences* — and the code took the row. On a
    directed graph the row is the set `p` reads *from*, whose contribution to
    `p`'s own parameter gradient is exactly zero, so the trace paid the full
    `K`-fold widening and bought nothing until the neighbourhood saturated and
    the two sets coincided.

    What makes this worth an item is that *no size-based test can see it*. `K`,
    monotonicity, nestedness, saturation, padding validity and the memory curve
    are all identical under either orientation, and the graph fixtures in use
    were symmetric often enough that even the flat-adjacency cross-check passed.
    Only membership on a strictly directed fixture separates them, which is now
    `position_graph_test.py::TestNeighbourhoodDirection`. Generalising: when an
    axis is supposed to trade memory for accuracy, plot accuracy against the
    knob *before* trusting any single-point assertion — a two-point check at the
    endpoints would have shown "SnAp-1 differs from BPTT" and "saturated SnAp
    equals BPTT", both true here, both passing, with the entire interior wrong.

19. **A whole-tree comparison against BPTT measures the truncation window, not
    the learning rule.** Under chunking, a model's *plain* (non-ETP) parameters
    receive exactly truncated BPTT — they are carried by no trace, so they
    truncate by construction at every coordinate of every axis. On the ring at
    `chunk_size=1` the plain input projection alone contributes `4.5e-01` to the
    tree-wide relative deviation while the ETP weight is exact to `7e-08`. This
    cost the longest debugging session of P3, chasing a "saturated SnAp is not
    BPTT" failure that was never a failure. BPTT comparisons now restrict to
    `spec.etp_param_keys` (`snap_n_test._rel_etp`). Algorithm-vs-algorithm
    comparisons deliberately stay on the *full* tree: two rules at the same
    window truncate identically, so agreeing everywhere is the stronger claim.

20. **A recurrent oracle without a self-edge cannot see `recurrence_scope` at
    all.** The first sparse ring used `offsets=(1,)`, a pure cycle. Then
    `∂h_p/∂h_p = 0`, the per-position block of the recurrent Jacobian is
    identically zero, and `diagonal` and `coupled` return *bit-identical*
    gradients — so the negative control that was supposed to prove the axis has
    content proved nothing. Fixed by defaulting both ring oracles to
    `offsets=(0, 1)`, which restores a `3.68e-02` separation and does not change
    `K(n)`. This is lesson 10 with a new mechanism: for a scope axis, what the
    model must supply is a non-zero *self* block, not merely recurrence.

21. **The widened trace needed zero per-primitive changes, and the reason is
    worth keeping.** `sparse_n` replaces the trace's trailing `num_state` axis
    `S` with a `(neighbour, state)` axis of width `M = K·S`, and not one of the
    nine ETP primitives' rules was touched. Two existing properties make that
    work: `_comp_recurrent_legacy` vmaps the state axes away before calling
    `dt_to_t`, so per-primitive rules never see the widened axis; and `dt_to_t`
    already implements each primitive's anchor map (hidden position → trace
    slot), which is exactly the map the widening needs. The fast paths
    (`fp.recurrent` / `instant` / `solve` / `chunk`) index the state axis by
    einsum letter and are therefore generic in its size. P4's `random_projection`
    should check whether the same two properties buy it the same freedom before
    proposing to touch kernels.

22. **A capability the compiler cannot infer must be declared, and must default
    to deny.** `sparse_n` is only meaningful if every ETP relation anchors on a
    single hidden position — a trace slot has to have one well-defined position
    its instantaneous term lands on. `etp_einsum` with a shared axis, and the
    embedding ops, have no such position, and nothing in their shapes reveals
    that. So `snap_anchor` is a declared per-primitive capability whose default
    is *no anchor*, and `sparse_n` raises naming the offending primitive and
    pointing at `coupled` / `diagonal`. Inferring it would have been silently
    wrong for exactly the primitives that matter; cf. lesson 15.

23. **An ETP output reshaped before it reaches the hidden state severs the
    relation, and the compiler only warns.** The grouped-matmul fixture first
    declared a flat `(1, n_rec)` hidden state and reshaped the `etp_gmm` output
    into it. The compiler discovered *zero* ETP relations, emitted a warning
    ("y shape=(1, 2, 3) not broadcastable with hidden shape=(1, 6)"), and the
    test passed while exercising nothing. Fixed by shaping the hidden state
    `(1, G, K)` and reshaping the *input* instead. A warning is not a failure:
    what caught it was the structural pin on `primitives`, which was `()`.

24. **`@pytest.mark.parametrize` does not parametrize a `unittest.TestCase`
    method.** pytest collects the method once and never binds the parameter. A
    test that reads as covering nine primitives covered none. Converted to an
    explicit loop over the fixture dict. Worth stating alongside lesson 8: this
    repository mixes `TestCase` classes and bare pytest classes in the same
    file, so the decorator's applicability changes from class to class.

25. **Report the conservative and the degenerate case; do not just handle
    them.** The adjacency analysis factorises per axis (`A = ⊗ A_axis`), which
    is exact when at most one axis is non-identity and a strict superset
    otherwise — safe, but it silently costs memory, so it emits
    `SNAP_PATTERN_CONSERVATIVE` with the reason. The `K == 1` case matters more:
    it is legitimate on a model with no cross-position coupling, and it is also
    exactly what a silently failing analysis produces, so it emits
    `SNAP_PATTERN_DEGENERATE` rather than quietly computing `coupled` under a
    `sparse_n` label. One tooling note for reading these back:
    `compile_etrace_graph` opens its own nested `diagnostic_context()`, so an
    outer reporter sees nothing — the records are on `graph.diagnostics`.

### From P3's adversarial review

The implementation above passed its own acceptance suite before this review ran.
These five are what a hostile second reading found anyway, which is the argument
for running one.

26. **`bool @ bool` saturates; `uint8 @ uint8` counts paths and wraps.** The
    reachability closure `A^k` was computed on `uint8` for compactness. NumPy's
    boolean matmul accumulates with logical-or, so it saturates at `True`; the
    integer one accumulates with `+` and wraps modulo 256. A position pair
    joined by exactly 256 parallel two-hop paths therefore reported
    **unreachable** — a *subset* of the true neighbourhood, which is the one
    error direction this analysis may never have, and the one no existing test
    could see because every fixture had a handful of paths, not hundreds. Fixed
    by keeping the closure boolean throughout; pinned by
    `test_many_parallel_paths_do_not_wrap`, which builds a 256-fold fan
    deliberately. The general form: a *conservative* analysis has an asymmetric
    failure cost, so its arithmetic must be chosen for the direction it may err
    in, not for its width.

27. **Budget the allocation that actually dominates, and name it in the error.**
    The first guard capped the *trace* at `P·K·S` scalars. The term that
    actually bounds usable `n` is the widened block Jacobian `Dg`, at
    `P·(K·S)²` — quadratic in the same knob. The constant became
    `DEFAULT_MAX_JACOBIAN_ELEMENTS = 2^24`, which admits `P=K=256, S=1` at
    64 MiB and rejects `P=K=512` at 512 MiB, and the message prints the computed
    element count and names the three ways out (smaller `n`, `coupled`, or a
    larger explicit ceiling). A limit whose message does not say what was
    measured is a limit users work around by guessing.

28. **A coordinate cannot always carry its own provenance.** `SnAp(n=1)`
    canonicalises to `recurrence_scope='coupled'` (lesson 18's finding), and
    `coupled` is legal on models with no anchored primitive — so the anchor
    check, keyed on the coordinate, silently stopped applying at exactly the
    order where a user is most likely to be comparing SnAp against something
    else. The preset now records `_requested_snap_order` alongside the config
    and the check keys on either. Whenever a public entry point *narrows* to a
    shared internal coordinate, ask what the narrowing threw away that a
    validation still needs.

29. **Endpoints are not evidence about the interior; write the interior's own
    oracle.** Saturated SnAp equals BPTT and SnAp-1 equals `OSTLRecurrent`,
    both pinned — and both are statements about the *ends* of the scale. The
    interior got a hand-written masked-RTRL recursion in float64 numpy
    (`influence = mask * (influence @ Dᵀ + instant)`) whose mask is derived
    independently of the compiler's, and orders 2, 3 and 4 match it to `<1e-6`
    while a deliberately narrower reference misses by `1.2e-1`. Note the
    discrimination is one-sided on a forward-only ring — a wider reference still
    matches, because widening only adds strictly-downstream positions that
    cannot feed back — so the test documents that rather than asserting a
    symmetry the model does not have.

30. **"Correct" needs a subject.** The conservative-widening diagnostic said
    "the gradient stays correct". True of the *pattern* — a superset never drops
    a real dependency — but it reads as a promise about the returned gradient,
    and that promise is false whenever the group's `y → hidden` tail relabels
    positions: there, full within-group RTRL is **not** BPTT, and the saturated
    rule is further from BPTT than the cheaper ones (1.30 vs 1.03 relative
    deviation on the `roll`-tail probe, against `1.3e-07` for the same model
    with the roll removed). The wording now says the *approximation* is not
    degraded and points at F-31 in the limitations list, where the pair and its
    control are recorded. Every axis in this roadmap describes what a trace
    retains; none of them can describe what the trace's index does not denote.

31. **"Unbiased" needs a subject as much as "correct" does (lesson 30 again, one
    axis over).** UORO is routinely described as *the* unbiased online rule, and
    the temptation is to write that down and pin it against BPTT. It is unbiased
    for the recursion **its transition defines** — the exact within-group
    influence recursion, i.e. the saturating end of the SnAp scale — and nothing
    more. It does not repair cross-group coupling, the F-31 instantaneous tail, or
    any primitive's own solve regime. So `uoro_test.py`'s reference is
    `SnAp(n=4)`, not BPTT, and the BPTT comparison appears only as a second test
    on the one fixture where the two provably coincide. Writing it the other way
    round would have produced a test that failed for the right implementation.

32. **Unbiasedness by exhaustive enumeration beats unbiasedness by sampling, when
    the draw is discrete and small.** The roadmap asked for statistical
    infrastructure and warned it would be expensive. It was built — and then the
    sharpest test turned out not to need it: with `H = 2` and two draw steps that
    still influence the gradient, there are exactly `2^4 = 16` Rademacher sign
    patterns, so the mean over *all* of them is the expectation **exactly**, to
    1e-16 rather than to a confidence interval. The sampled interval test is kept
    for the cases enumeration cannot reach and marked `slow`. Prefer an exact
    finite average whenever the randomness has small finite support.

33. **A normaliser that cancels is a variance choice, not a correctness one.**
    UORO's `rho0`/`rho1` look load-bearing; the parity argument shows they are
    not. The cross terms are odd in the draw while `rho1` is even, so *any*
    positive draw-independent `rho0` and any positive even `rho1` leave the
    estimator unbiased. Corollary that saved a debugging session: a wrong
    normaliser cannot be diagnosed by a bias test, because there is no bias to
    find — only variance. It also kills an "obvious" optimisation: antithetic
    sampling leaves the estimate **bit-identical**, because both factors flip
    together and their product is invariant.

34. **Replace, do not multiply — and let the degenerate case decide.** For
    `modulatory`, `m * dL/dh` and `m` alone both read as "three-factor". The
    tie-breaker is not taste: multiplying makes the roadmap's own degenerate
    criterion — set the modulator to `dL/dh` and recover `symmetric` element-wise
    — unsatisfiable, leaving the axis with no coordinate at which it reduces to
    the rule it generalises. When two readings of a spec differ, prefer the one
    whose degenerate case is checkable.

35. **A hook that only runs under `grad` needs its refusals hoisted out of it.**
    `_compute_learning_signal` executes inside the `custom_vjp` backward pass, so
    a malformed modulator passed to a *forward-only* `update()` was accepted in
    silence and only failed later, from inside JAX, with a traceback pointing
    nowhere useful. The validation now runs eagerly at the top of `update()`,
    against each group's declared signal shape, while the authoritative expansion
    stays where the real shapes are. Test discipline that caught it: the refusal
    test called `algo(x)` without an outer `grad`, because that is what a user
    debugging their shapes does first.

36. **Do not infer a category from which container it arrives in.** DNI's whole
    correctness claim is a split — the synthetic cotangent must reach the plain
    parameters and must not reach the ETP ones — and the backward pass appears to
    hand that split over for free, in `dg_etrace_params` versus
    `dg_non_etrace_params`. It does not (F-34): under multi-step, *every*
    trainable parameter arrives in `dg_etrace_params`, plain ones included, and
    `dg_non_etrace_params` is empty. The first implementation therefore added the
    future term to an empty dictionary and was a **perfect no-op** — every test of
    the form "M == 0 is a no-op" passed, and so did every ETP invariance test. It
    was caught only by B1's other half, "a live synthesiser must *change* the
    plain gradients", which measured a deviation of exactly `0.0`. Name-shaped
    assumptions about someone else's data structure need a test that fails when
    they are wrong, and for a *split*, that means testing both sides.

37. **Two coordinates in the same axis can want opposite execution options.**
    `modulatory` is only meaningful under `single-step` — under multi-step the
    in-window reverse-AD term stays unmodulated and the rule is half three-factor,
    half plain loss gradient. `bootstrapped` is only meaningful under
    `multi-step` — a one-step window has no exit worth estimating. Both refusals
    are in the base constructor rather than in the presets, so a config-built
    algorithm gets them too; and `update_schedule` fell out of `ThreeFactor`'s
    scope as a consequence, not as an omission.

38. **A learned auxiliary predictor needs its data counted, not its epochs.**
    DNI's synthesiser is a 20-parameter linear map; trained on a `T = 8` sequence
    at `chunk_size = 2` it sees **four** distinct boundaries, fits them almost
    perfectly (0.83 → 0.12) and generalises *worse than predicting zero* (0.412
    against 0.368 deviation from BPTT). Twenty boundaries beat the zero estimate;
    thirty were worse again. Two lessons, and the second is the uncomfortable one:
    the number of *boundaries*, not epochs, is the sample size — and a lower
    auxiliary regression loss does not monotonically buy a better gradient, so
    B3 is written and labelled as a demonstration at a stated configuration
    rather than as a property of the method. A third, cheaper trap sits next to
    it: the fit must use the same `chunk_size` the learner will be driven with,
    or it predicts the future at boundaries that do not exist.

39. **When an end-to-end criterion fails, add the oracle arm before weakening
    the claim.** B4 — "DNI beats its controls on a delayed-reward task" — failed
    three different ways, and each time the tempting move was to soften the
    assertion into something that would pass. What actually resolved it was
    inserting a fourth arm whose estimate is *exact*: the true future cotangent
    of the training objective, recomputed against the current parameters every
    epoch. That arm separates two questions the failing test had fused — *is the
    injected credit routed correctly?* and *is a learned linear map a good enough
    predictor?* — and it is cheap, because B2 already had to build the machinery
    to pin an oracle synthesiser. With the oracle arm in place the ordering came
    out `oracle < trained < M ≡ 0` on every seed, so the criterion was true all
    along and all three failures were harness defects. A criterion that fails for
    harness reasons and a criterion that fails because the method does not hold
    are indistinguishable from the assertion alone; the oracle arm is what tells
    them apart, and it belongs in the test permanently as the ceiling every other
    arm is measured against.

40. **A helper's convenient default is a trap wherever it must agree with the
    caller.** `train_synthetic_gradient(loss_fn=...)` defaults to sum-of-squares.
    B4 descended on `(out - target)²` and left the default in place, so the
    synthesiser was fitted against the gradient of a *different function at a
    different scale* — and the injected result was not degraded DNI but noise
    with the shape of a cotangent, leaving the run measurably worse than
    switching DNI off (0.577 against 0.140). This is the identical failure to the
    `chunk_size` hazard already documented one lesson above, in a second
    parameter, and it went unnoticed precisely because a default *looks* like a
    decision already made. The rule that generalises: when a helper's parameter
    has to match something the caller does elsewhere, a default is a liability
    even when it is the common case. Recorded as F-35, and both halves are now
    stated in the parameter's own docstring rather than only in prose above it.

41. **A stale approximator is not the method.** The first repair attempt fitted
    the synthesiser once, against the *initial* model, and then trained the model
    for fifteen epochs underneath it. That arm won on one seed of three. It is
    not what DNI is: the target `dL_{≥b}/dh^b` is a function of the parameters,
    so an estimator of it goes stale exactly as fast as the parameters move.
    Refitting once per epoch fixed it. Worth stating because the stale version is
    what the natural reading of a helper named `train_synthetic_gradient` — train
    it, then use it — produces, and because the failure looks like the method
    being weak rather than the protocol being wrong.

42. **A fixture's conditioning is load-bearing for any test that trains.**
    `delayed_reward_rnn` was written as `h ← leak·h + tanh(...)`, an accumulator
    bounded only by `1/(1-leak) = 20`. Every downstream scale inherits that:
    outputs `O(10)`, squared errors `O(100)`, and the hidden cotangents the
    synthesiser regresses against `O(1e5)` — whose fit diverged to `nan`, as did
    plain-SGD training of the model at every learning rate tried down to `2e-3`.
    The one-character fix is the convex form, `h ← leak·h + (1-leak)·tanh(...)`,
    which bounds `|h| ≤ 1` and leaves the credit span untouched because the span
    comes from the `leak·h` term alone. The fixture's own tests never caught it:
    they assert *ratios* of early to late credit, which are scale-invariant. A
    fixture used only for gradient comparisons can get away with poor
    conditioning; the moment a test runs an optimiser on it, conditioning becomes
    part of the fixture's contract.

43. **A comment that asserts a shape is a claim, and claims get measured.**
    The eager modulator pre-flight skipped descended hidden groups, with a comment
    explaining that a descended group's learning signal "carries a leading substep
    axis, so its runtime shape is not `(*varshape, num_state)`". Measured on
    `snn_scan_rnn(loops=40)`, which does descend, the signal is `(1, 4, 1)` — the
    group shape exactly, because `scan_descent` folds the per-substep Jacobians
    *inside* the body and the reverse pass hands out one array per group. The skip
    bought nothing and cost the pre-flight: a malformed modulator was accepted by
    a forward-only `update()` on any descended model and only failed later from
    inside JAX. Two lines of probe would have caught it at the time it was
    written; the plausible-sounding comment is what stopped anyone looking.

44. **Widening a reduction is a dtype change, and a scan carry's dtype is a
    contract.** `_tree_sq_norm` accumulates in float32-or-wider on purpose — a sum
    of squares in float16 underflows below about `1e-3` — but the resulting
    normalisers then set the dtype of `rho0 * d_s + rho1 * nu`, which *is* the scan
    carry. `jax.lax.scan` rejects that: `carry input and carry output must have
    equal types`, on the first `MultiStepData` call, for every float16 or
    (under `jax_enable_x64`) float64 model. The fix is not to narrow the reduction
    but to narrow its *result*: reduce wide, then `astype` back to the carry's own
    dtype. Note this is the second dtype bug in the same three lines — the first
    was allocating the factors at a hard-coded `float32` — which is the tell that
    "what dtype does this carry" deserved a parametrised test from the start
    rather than two point fixes.

45. **Observing that a hook was called is not observing that it was used.**
    `test_override_hook_replaces_learning_signal` captured the hook's argument and
    asserted the resulting gradient was non-zero. A base class that invoked the
    override, discarded its return value, and went on using the reverse-AD signal
    passes both assertions. What pins it is a property of the quantity itself: the
    parameter gradient is linear in the learning signal, so a hook returning
    `k · ones` must scale the gradient by exactly `k`, and must differ from the
    un-overridden run. Same shape as lesson 39 — the assertion has to be sensitive
    to the thing being claimed, not merely present when it holds.

46. **`io_callback` is not a universal observation seam.** Capturing UORO's
    production `nu` looked like the direct way to pin draw freshness, but a
    callback placed inside the stepper never fires: the stepper is traced under
    `jax.custom_vjp`, where the callback's unused result is dead and gets
    eliminated. The property was observable anyway, because the freshness does not
    live in `_draw_projection` at all — that is a deterministic function of one
    key, which is exactly what makes it a usable test seam. It lives in the
    caller's schedule (`split_key(len(groups))` per step, carried key advanced).
    Replaying that schedule in the test and feeding the *production* draw function
    pins both freshness and cross-group independence, with no tracing involved.
    When a seam resists observation, check whether the property is actually
    located where you are looking.

47. **A recursion's first step cannot pin a normaliser that degenerates there.**
    The hand-computed factor test ran one step, where `s_tilde` and `theta_tilde`
    are both zero, so `rho0 = sqrt(eps/eps) = 1` regardless of the formula:
    inverting its ratio or deleting it outright left the test green. Only step 2
    makes it live. The fixture's conditioning matters as much as the step count —
    at the suite's default input scale `rho0` lands at 0.92, an 8% departure from
    the degenerate value, so an inverted ratio would move the expectation by only
    19%. Saturating the transition (`scale=5.0`) puts `rho0` near 16, where an
    inversion is off by 264×, and the test asserts that separation explicitly so a
    later fixture change cannot quietly return it to the degenerate regime.

48. **"That rule is about the model, not about training loops" is usually a
    rationalisation.** I pushed back on a review finding that
    `train_synthetic_gradient`'s per-window Python loop violated AGENTS.md
    rule 10, on the grounds that the rule targets *driving the model over time*
    and this was a grad-plus-optimiser-step loop like any training loop. Both
    halves were true and the conclusion was still wrong: the loop body drives the
    learner — `learner(_as_window(...))` under `grad` — so it re-traced the whole
    `custom_vjp` machinery once per window. Measured with a trace counter on the
    synthesiser's `apply`: 14 traces for 4 windows, 38 for 12, against a constant
    count under `for_loop`. The repo's own quickstart already had the right
    shape — Python loop over epochs, `scan` over steps, optimiser inside — so the
    idiom argument pointed the other way too. The tell I missed: I was arguing
    about which *category* the loop belonged to instead of asking what the body
    actually did.

    Two things fell out of the conversion that were worth more than the speed.
    A ragged final window cannot exist under `for_loop`, which forced the
    question of what a short window *meant* — it fits the synthesiser against a
    shorter future than any window the learner will ever be driven with, the same
    mismatch F-35 documents, introduced by the helper rather than the caller. It
    is now refused. And `_fit_one` had to stop calling `float()` on its error,
    which is what made the per-window concretisation visible: every window was
    forcing a device sync to build a list that got averaged one line later.

