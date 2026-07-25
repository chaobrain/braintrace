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

| Algorithm | factorization | recursion | recurrence_scope | signal | filter |
|---|---|---|---|---|---|
| `D_RTRL` | per_param | jacobian | diagonal | symmetric | none |
| `pp_prop` | io_factorized | jacobian | diagonal | symmetric | none |
| `OSTLRecurrent` | per_param | jacobian | **coupled** | symmetric | none |
| `OSTLFeedforward` | io_factorized | none | diagonal | symmetric | none |
| `EProp` | per_param | jacobian | diagonal | symmetric \| random_feedback | **kappa** |

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

### P1 — Compiler: multi-timescale and heterogeneous populations

Make the surviving rules actually run on realistic brain-simulation models —
LIF + ALIF + multi-timescale synapses, heterogeneous leaks, multi-state
HiddenGroups — before widening the algorithm family.

**Scope has to be reconstructed first.** `AGENTS.md` points at a
known-limitation findings list under `dev/`, but `dev/` is gitignored and absent
from the repository. The only machine-readable remnant is a single skipped test
(`approx_correctness_test.py::test_approximations_diverge_on_snn_multipopulation_DEFERRED`,
finding F-22). Everything else on that list has either been resolved — F-19 and
F-20 died with the algorithms they described; F-SINGLESTEP was resolved into a
positive direction-alignment assertion in `oracle_test.py` — or exists only as
prose in `AGENTS.md`. **The first task of P1 is to rebuild the list in-tree**,
from `AGENTS.md`'s summary plus a sweep of the test suite, so it stops living in
an untracked directory. Expect the rebuilt list to be shorter than the original:
part of it is already done.

Known items from `AGENTS.md`: heterogeneous-population leak resolution,
multi-state HiddenGroups, approximation validity beyond shallow depth,
single-readout / feedback-shape assumptions, cross-algorithm equivalence gaps.

This phase owns `hidden_group.py`'s Jacobian path, which P3 also needs — doing
it first is what keeps P3 from fighting a moving target.

**Acceptance:** the reconstructed limitation list is committed under
`docs/specs/`; every item on it has either a passing test or an explicitly
documented scope boundary; no expected-failure item silently remains.

### P2 — Axis decomposition

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
  all five surviving algorithms.
- `decay_or_rank=0.9` equals the new two-sided `(0.9, 0.9)` element-wise.
- Illegal axis combinations raise a readable error naming the legal pairings,
  with test coverage.
- Each preset's coordinates are asserted against the table in this document, so
  the table cannot drift from the code.

### P3 — SnAp-n: generalise `recurrence_scope`

Menick et al. 2021. Derive the n-step influence sparsity pattern automatically
from the compiler's jaxpr hidden→hidden reachability graph — the thing a
compiler can do that a hand-written library cannot.

Concretely: replace the boolean `is_diagonal_recurrence` with an n-valued scope,
keep `jacrev_last_dim` as the `n = 1` fast path and `block_diagonal_last_dim` as
today's coupled path, and add the intermediate sparse representation for
`1 < n < diameter`.

**Acceptance (two-sided squeeze):**
- `n = 1` equals the current `D_RTRL` element-wise (regression guard).
- Whatever configuration expresses `recurrence_scope = coupled` after the
  refactor equals the current `OSTLRecurrent` element-wise — regardless of
  whether that configuration turns out to be a point on the `n` scale or a
  sibling value beside it (regression guard for the existing `True` branch).
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
   assert element-wise equality after. Separately, the `IODim` decay split must
   prove `decay_or_rank=0.9` equals the new `(0.9, 0.9)` element-wise.
2. **P1 and P3 collide in `hidden_group.py`.** Both rework the hidden→hidden
   Jacobian path. Mitigation: P1 owns that file and lands first; P3 consumes the
   representation P1 leaves behind rather than introducing a second one.
3. **The P1 scope is not actually written down.** The findings list lives in an
   untracked `dev/`, and the in-tree remnants are already stale — one docstring
   still cross-referenced a test renamed some time ago. Mitigation: rebuilding
   the list in-tree is P1's first task, not an assumption, and the rebuilt list
   must be verified against the test suite rather than transcribed.
4. **Statistical tests are flaky in CI** (P4). Mitigation: fixed seeds, generous
   intervals, a separate slow-test marker.
5. **`modulatory` recreates OSTTP's plumbing mistake** (P4). Mitigation: the
   injection path must not bind to HiddenGroup count or readout shape; the
   mismatched-dimension test above is the guard.
6. **0.3.0 carries the whole roadmap.** All breaking changes and all new
   algorithm families ship together. Mitigation: phases merge independently
   behind the axis interfaces, and P5 runs on every merge so regressions surface
   per phase rather than at release.
