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

**Open design question for P3.** Both existing values end up producing a
per-position block-diagonal Jacobian; they differ in what enters the transition
before the block diagonal is taken. SnAp-n instead parameterises the *n-step
influence* that is retained. Whether `coupled` lands on a particular `n`, or
stays a sibling value beside the `n` scale, is a design decision P3 must settle
explicitly — this document does not assume it maps onto one.

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

### P3 — SnAp-n: generalise `recurrence_scope`

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

### P4 — UORO, three-factor and DNI

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
P2, against `156d058`. These are the things the roadmap got wrong or could not
have known, kept here because later phases rest on them.

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
