# P2 — Axis decomposition

Status: spec, implementation in progress
Roadmap: [`2026-07-25-algorithm-axes-roadmap.md`](2026-07-25-algorithm-axes-roadmap.md) § P2
Baseline: commit `156d058` (P1 landed)
Target release: 0.3.0

## Goal

Make the roadmap's six learning-rule axes a *fact about the code* rather than a
description of it. After P2 an algorithm is a **coordinate**, not a class: the
coordinate is an `ETraceConfig` value, the engine reads its behaviour from that
value, illegal coordinates are rejected at construction with a readable error,
and the five surviving presets are thin configurations over it.

The refactor must be **numerically inert** for every existing preset. That is
the single hardest constraint here and it drives most of the design decisions
below.

## Scope

**In:**

1. `ETraceConfig` — the six axes as a validated, canonicalising, frozen value.
2. The compatibility matrix as explicit data, with errors that name the legal
   pairings.
3. `recurrence_scope` lifted from the private class attribute
   `_include_recurrent_mixing` to an axis value.
4. `temporal_recursion` made real for both engines (`jacobian` / `scalar_leak` /
   `none`), primitive-agnostically.
5. The `IODimVjpAlgorithm` decay split into an independent `(x-side, f-side)`
   pair — the change that makes the removed OTTT's coordinate reachable *and*
   primitive-generic.
6. `learning_signal='random_feedback'` and `trace_filter='kappa'` lifted out of
   `EProp` so they are engine features rather than one class's private code.
7. The presets rewritten as configurations; each exposes `.config`.
8. `ETraceConfig` accepted by `braintrace.compile` in the `algorithm` position.

**Out** (declared in the axis vocabulary, rejected by the matrix with an error
naming the phase that will deliver them):

- `trace_factorization='random_projection'` (UORO) — P4.
- `learning_signal` ∈ {`modulatory`, `bootstrapped`} — P4.
- `recurrence_scope='sparse_n'` (SnAp-n) — P3.
- `update_schedule` ∈ {`window`, `sequence_end`} — unassigned; `per_step` is the
  only implemented value.

Declaring an unimplemented value and rejecting it is deliberate. The alternative
— omitting the name until its phase lands — means the axis vocabulary silently
disagrees with the roadmap and the matrix has nothing to say about the
combinations that will matter most. A rejected value with a message naming its
phase is discoverable; an absent one is not.

**Explicitly out:** execution options (`vjp_method`, `fast_solve`,
`trace_dtype`, `chunked_trace`, `control_flow`) stay constructor parameters and
stay out of `ETraceConfig`. They choose *how* a rule runs, not *which* rule it
is.

## The axis vocabulary

`braintrace.ETraceConfig` is a frozen dataclass. Six categorical fields carry
the coordinate; four numeric fields carry the coefficients that a category
alone cannot express.

| Field | Values | Notes |
|---|---|---|
| `trace_factorization` | `per_param` · `io_factorized` · `random_projection`† | selects the engine |
| `temporal_recursion` | `jacobian` · `scalar_leak` · `none`; a **pair** `(x, f)` under `io_factorized` | structural transition operator |
| `recurrence_scope` | `diagonal` · `coupled` · `sparse_n`† | how much hidden↔hidden coupling enters the transition |
| `learning_signal` | `symmetric` · `random_feedback` · `modulatory`† · `bootstrapped`† | where the signal comes from |
| `trace_filter` | `none` · `kappa` | low-pass on the trace |
| `update_schedule` | `per_step` · `window`† · `sequence_end`† | when the gradient is emitted |
| `decay` | float in `[0, 1)`, or a pair under `io_factorized` | per-step discount of the previous trace |
| `kappa` | float in `[0, 1)` | coefficient of `trace_filter='kappa'` |
| `sparse_n` | int ≥ 1 | coefficient of `recurrence_scope='sparse_n'` |
| `window_size` | int ≥ 1 | coefficient of `update_schedule='window'` |

† rejected in 0.3.0-P2; see *Scope / Out*.

### `temporal_recursion` is a structural operator, `decay` is a coefficient

The two are separated on purpose, because a category alone cannot distinguish
`pp_prop(decay_or_rank=0.9)` from `pp_prop(decay_or_rank=1)` — and lesson 6 of
the roadmap (F-29) is precisely that those two are different *rules*, not two
spellings of one.

Define the trace recurrence, per engine, as

- `per_param`: `ε ← R·ε + instant`
- `io_factorized`, x-side: `ε_x ← α_x·ε_x + x`
- `io_factorized`, f-side: `ε_f ← α_f·(R·ε_f) + (1 − α_f)·D_f`

`temporal_recursion` chooses `R`; `decay` supplies `α` (and, under `per_param`,
supplies the scalar inside `R`):

| Value | `R` under `per_param` | `R` under `io_factorized` f-side |
|---|---|---|
| `jacobian` | `D` (the hidden→hidden Jacobian) | `D` |
| `scalar_leak` | `λ·I`, `λ = decay` | `I` (the leak lives in `α_f`) |
| `none` | `0` | `0` |

The x-side never involves a Jacobian — the input-side trace is a filtered copy
of the presynaptic input — so `x`-side `jacobian` is rejected.

**Implementation consequence.** Every value of this axis is realised by
substituting the *executor's* per-hidden-group Jacobian array `D` before it
reaches any trace-update path. **The substitution differs per engine, and the
difference is load-bearing:**

| Value | `per_param` substitution | `io_factorized` f-side substitution |
|---|---|---|
| `jacobian` | `D` (unchanged) | `D` (unchanged) |
| `scalar_leak` | `λ · I` | `I` |
| `none` | `0` | `0` |

Under `per_param` the trace roll is `ε ← D·ε + instant`
(`param_dim_vjp.py:412-425`), so the leak must be carried *by the substituted
array*: `λ·I`. Under `io_factorized` the roll applies `D` first
(`io_dim_vjp.py:401`) and `_expon_smooth` then multiplies that product by `α_f`
(`io_dim_vjp.py:413`), so substituting `λ·I` there would yield `α_f·λ` — with
`λ = α_f`, an accidental `α_f²`. The f-side leak already lives in `α_f`, so its
`scalar_leak` substitution is the bare identity.

**Shape.** Build the replacement from `D.shape`, never from `(*varshape, S, S)`:
`jnp.zeros_like(D)` for `none`, and `coef * jnp.eye(D.shape[-1], dtype=D.dtype)`
broadcast against `D.shape` for `scalar_leak`. The trailing two axes are always
`(num_state, num_state)`, but the *leading* axes vary by path — measured:
`(*varshape, S, S)` single-step, `(T, *varshape, S, S)` stacked multi-step, and
`(L, *varshape, S, S)` for descended-scan groups
(`vjp_graph_executor.py:387`). Broadcasting against `D.shape` is correct for all
three; constructing a fixed rank is not.

This is why the axis is primitive-agnostic *for free*: the fast path, the legacy
nested-vmap path, the chunk-factorised path (`suffix_products` of `λ·I` is
`λ^k·I`), the descended-scan substep fold and both engines all consume the
substituted array without knowing it was substituted. No per-primitive rule
changes.

### Canonicalisation

A coordinate must have exactly one spelling, or the "assert the preset's
coordinates against the table" acceptance criterion is meaningless.

**Order is: canonicalise first, then validate.** Every rule in the compatibility
matrix is evaluated against the canonical form, so a rule never fires on a
spelling that canonicalisation would have removed.

- `scalar_leak` with `decay == 0` canonicalises to `none` (they are the same
  rule).
- `none` forces its `decay` side to `0.0`.
- Under `io_factorized`, a scalar `temporal_recursion` expands to both sides,
  then an x-side `jacobian` is demoted to `scalar_leak` (the x-side has no
  Jacobian). So `temporal_recursion='jacobian'` under `io_factorized` means
  `('scalar_leak', 'jacobian')` — the `pp_prop` coordinate. **The demotion
  applies only to the scalar shorthand.** An explicit pair whose x-side is
  `jacobian` is a statement about the x-side and is rejected by matrix rule 3,
  not silently rewritten.
- Under `io_factorized`, a scalar `decay` expands to both sides.
- Under `io_factorized`, `α_x == 0` canonicalises the x-side to `none`, and
  `α_f == 0` canonicalises the f-side to `none`.
- `trace_filter='kappa'` with `kappa == 0` canonicalises to `none`. This matches
  `EProp`'s documented behaviour — `kappa_filter_decay=0` "reduces exactly to
  `D_RTRL`" — so the two really are one coordinate.

**`decay` is a float, never a rank.** `decay_or_rank`'s integer form is a
user-facing convenience, and `rank` / `decay` are two spellings of one number
(`decay = (rank − 1)/(rank + 1)`). Admitting both into `ETraceConfig` would
re-introduce exactly the ambiguity canonicalisation exists to remove, so the
int→float conversion happens at the **preset boundary**
(`IODimVjpAlgorithm.__init__`) and `ETraceConfig.decay` is float-only.

## The compatibility matrix

Explicit data (a list of `(predicate, message)` rules), evaluated in
`ETraceConfig.__post_init__`. Every rejection message names the offending pair
*and* the legal alternatives.

| # | Rejected combination | Why |
|---|---|---|
| 1 | `trace_filter='kappa'` with `io_factorized` | `ε̄ ← κ·ε̄ + ε_x⊗ε_f` is not rank-1, so the filtered trace cannot be stored factorised. Filtering the two factors independently is a *different* rule, not e-prop's. |
| 2 | `recurrence_scope != 'diagonal'` unless the **`D`-consuming side** uses `jacobian` — the single side under `per_param`, the **f-side** under `io_factorized` | the scope only changes *what enters* `D`; if `D` is not used the scope carries no information. |
| 3 | x-side `temporal_recursion='jacobian'` under `io_factorized` | the input-side trace never involves a Jacobian. |
| 4 | `per_param` with `decay` set while the recursion is `jacobian` | `jacobian` under `per_param` has no discount coefficient; a `decay` here would be silently ignored. |
| 5 | `per_param` + `scalar_leak` without `decay` | `λ` is the rule. |
| 6 | `io_factorized` without `decay` | `α` is the rule. |
| 7 | `kappa` set while `trace_filter='none'` (and symmetric cases for `sparse_n` / `window_size`) | a coefficient without its category is a typo, not a configuration. |
| 8 | any `†` value from the vocabulary table | not implemented; the message names the delivering phase. |

Rule 4 deserves a note: it is the one rule that exists purely to catch a user
error rather than to encode mathematics. It is worth having — `D_RTRL(model,
decay=0.9)` reads like it should do something.

Rule 2 needs the "`D`-consuming side" qualifier or it degenerates into "reject
every `io_factorized` coordinate": the x-side *never* consumes `D`, so a rule
phrased over both sides can never be satisfied there. `coupled` is meaningful
under `io_factorized` precisely because the f-side contracts `D` directly
(`io_dim_vjp.py:399-405`).

**`coupled` + `io_factorized` is legal — measured, not assumed.** Nothing in-tree
exercises this cell today, so it was measured before the matrix was written:
grafting `_include_recurrent_mixing = True` onto `pp_prop` on `tanh_rnn` runs,
produces finite gradients, and is distinguishable from `diagonal` at a relative
deviation of `3.7e-04` on the finite-window path. The matrix admits it and the
test plan pins it.

## Where each axis attaches

The engine's existing template-method hooks are the mounting points; P2 adds no
new abstract protocol.

| Axis | Attachment |
|---|---|
| `trace_factorization` | engine class selection (`ParamDimVjpAlgorithm` / `IODimVjpAlgorithm`); each engine asserts its own value |
| `temporal_recursion` | central substitution of the hidden-group Jacobians in `ETraceVjpAlgorithm._update_fn` / `_update_fn_fwd`, plus a wrapper around the fused stepper |
| `recurrence_scope` | `include_recurrent_mixing` passed to the graph executor (was the `_include_recurrent_mixing` class attribute) |
| `learning_signal` | `ETraceVjpAlgorithm._compute_learning_signal` (base implementation, was `EProp`-private) |
| `trace_filter` | `ParamDimVjpAlgorithm._solve_weight_gradients` (was `EProp`-private) |
| `update_schedule` | nothing yet — `per_step` is current behaviour |

Two lifts change *who* owns behaviour without changing *what* it computes:

- **random feedback** moves from `EProp` to `ETraceVjpAlgorithm`. It only
  touches per-hidden-group learning signals, which are factorisation-agnostic,
  so `pp_prop` gains random feedback for free. This is a real generalisation the
  axis decomposition buys, and it was **measured before being specified**:
  grafting `EProp`'s projection onto the IO-dim engine on `two_state_rnn` runs,
  stays finite, and moves the gradient by `5.4e-01` relative to `symmetric`.
- **the κ-filter** moves from `EProp` to `ParamDimVjpAlgorithm` (not to the
  base: matrix rule 1 confines it to `per_param`). `EProp`'s `_trace_filters`
  attribute name and per-key `ShortTermState` layout are preserved so existing
  tests keep their grip on it.

### The state lifecycle of the lifted features

Both lifts allocate state, and the obvious implementation of "move it to the
base" fails: `ETraceAlgorithm.init_etrace_state` **raises**
`NotImplementedError` (`base.py:395`), and both engines override it without
chaining (`param_dim_vjp.py:945`, `io_dim_vjp.py:719`). Worse, the failure is
silent for random feedback — an empty feedback dict makes
`_compute_learning_signal` fall through to the symmetric branch, so the
algorithm quietly computes a *different rule* than the one requested. That is
lesson 4's failure mode with a new face.

So the lifecycle is specified explicitly:

1. `ETraceVjpAlgorithm.init_etrace_state` becomes **concrete** (overriding the
   raising base) and allocates only the axis-side state: the κ-filter states,
   sized from `self._get_etrace_data()`, and the `FixedRandomFeedback` matrices.
2. Each engine's `init_etrace_state` calls `super().init_etrace_state(...)` **as
   its last statement**, after its own traces exist — the κ-filter mirrors the
   trace pytree and cannot be sized before it.
3. `ETraceVjpAlgorithm.reset_state` likewise becomes concrete and resets the
   κ-filter states; each engine chains to it.
4. **No silent degradation.** If `learning_signal='random_feedback'` and no
   feedback matrix was built, `_compute_learning_signal` raises instead of
   returning the symmetric signal. A configuration that cannot be honoured must
   fail loudly, not compute something else.

The trace-update path fuses into the executor's over-time scan when the input is
multi-step, in which case `_update_etrace_data` is never called and the
substitution must happen inside the stepper. The base therefore wraps any
stepper returned by `_make_etrace_stepper` before handing it to the executor,
and substitutes the returned Jacobians on the non-fused path. The engines'
*internal* calls to their own steppers (`IODimVjpAlgorithm._update_etrace_data`,
`ParamDimVjpAlgorithm._make_scan_fn`) stay unwrapped, so the substitution is
applied exactly once on every path.

## The IODim decay split

`_format_decay_and_rank` becomes `_format_decays`, accepting a scalar (both
sides) or a `(x, f)` pair, each either a float decay in `[0, 1)` or an int rank
`≥ 1` mapping through `decay = (rank − 1)/(rank + 1)`.

The single `self.decay` is replaced by `self.decay_x` / `self.decay_f`, wired to
their three existing consumers:

| Consumer | Before | After |
|---|---|---|
| `_low_pass_filter` over `etrace_xs` | `decay` | `decay_x` |
| `_expon_smooth` over `etrace_dfs` | `decay` | `decay_f` |
| bias `correction_factor = 1 − decay^(t+1)` | `decay` | `decay_f` |

The correction corrects the f-side exponential smoothing only — the x-side
low-pass has no `(1 − α)` input weight and therefore no bias to correct — so
routing it to `decay_f` is a bug fix's worth of precision, not a change: today
the two are always equal.

`self.decay` survives as a read-only property returning the shared value when
both sides agree and raising `AttributeError` (naming `decay_x` / `decay_f`)
when they differ. Existing assertions like `assert algo.decay == 0.9` keep
working; the asymmetric case is new API and has no legacy readers.

The float bound relaxes from `0 < decay < 1` to `0 <= decay < 1`, so
`temporal_recursion='none'` is expressible as a float. It was already reachable
as `decay_or_rank=1` (rank 1 → decay 0), so this adds no new numerical regime.

## Coordinates of the surviving presets

Asserted in the test suite against this table.

| Preset | factorization | recursion | recurrence_scope | signal | filter | decay |
|---|---|---|---|---|---|---|
| `D_RTRL` | `per_param` | `jacobian` | `diagonal` | `symmetric` | `none` | — |
| `OSTLRecurrent` | `per_param` | `jacobian` | **`coupled`** | `symmetric` | `none` | — |
| `EProp` | `per_param` | `jacobian` | `diagonal` | `symmetric` \| `random_feedback` | **`kappa`** when `kappa_filter_decay > 0` | — |
| `pp_prop` | `io_factorized` | `(scalar_leak, jacobian)` | `diagonal` | `symmetric` | `none` | `(α, α)` |
| `OSTLFeedforward` | `io_factorized` | `(scalar_leak, jacobian)` | `diagonal` | `symmetric` | `none` | `(1e-6, 1e-6)` |

### Correction to the roadmap's table

The roadmap lists `OSTLFeedforward` as `temporal_recursion = none`. That is
wrong under the exact semantics defined above: its default `decay_or_rank=1e-6`
leaves both recursion terms structurally present with a negligible coefficient,
which is `(scalar_leak, jacobian) @ α = 1e-6`, not `none`. The *exact* `none`
coordinate is `OSTLFeedforward(model, decay_or_rank=1)` (or `decay_or_rank=0.0`
after the bound relaxation).

The default is **not** changed to the exact coordinate. Doing so would move the
gradients by ~1e-6 relative — small, but a deliberate numerical change during a
refactor whose entire acceptance criterion is that numerics do not move. The
roadmap table is corrected instead, and the exact-`none` spelling is documented
on the class.

## Scope boundaries the axis inherits

Making a private attribute public inherits every limitation the private version
had. Two of them need to be stated, and one needs a guard.

### `coupled` is not honoured inside a descended scan

`scan_descent.py:402` hard-codes `include_recurrent_mixing=False` for
descended-scan body analysis, documented at `scan_descent.py:50-54`: the
per-substep trace fold consumes diagonal Jacobians, so tracing recurrent ETP
mixing into a body transition would have no consumer. That is a defensible
design decision while the flag is private and only `OSTLRecurrent` sets it.

Once `recurrence_scope` is a public axis it becomes a trap: a user asks for
`coupled` and silently gets `diagonal` inside the scan body. So P2 adds a guard
— after `compile_graph`, an algorithm whose scope is `coupled` raises if any
relation carries a `control_flow_context` (i.e. a descended scan), with a
message naming the limitation. No in-tree test combines `OSTLRecurrent` with
scan descent, so nothing existing breaks. Implementing `coupled` *inside*
descended scans belongs with P3, which rebuilds the recurrence representation
anyway.

### F-30: the IO-dim bias correction counts calls, not trace steps

Surfaced by review of this spec, **pre-existing and not introduced by P2**.

`_solve_IO_dim_weight_gradients` corrects the f-side exponential-smoothing bias
with `1 − decay^(running_index + 1)` (`io_dim_vjp.py:493`). `running_index`
advances once per `update()` call (`vjp_base.py:319`), but the trace scan
advances once per *sequence element* (`io_dim_vjp.py:934-943`). Under
multi-step input the two diverge: a first call carrying `T` steps has smoothed
the trace `T` times but divides by `1 − decay^1`. The correction is exact only
for single-step input, where one call is one step.

Measured on `tanh_rnn` at `decay=0.9`: after one 6-step call `running_index` is
1, so the applied correction is `0.100` where `0.469` is required. The gradient
consequence is visible only through a finite window (6.8e-04 at `T` 6, chunk 2)
— on the full-window multi-step path the trace is not load-bearing at all
(F-23), so the mis-indexing is unobservable there.

P2 does **not** fix this. The fix would move `pp_prop` and `OSTLFeedforward`
gradients under multi-step input, and P2's entire acceptance criterion is that
gradients do not move — a numerical correction and a refactor must not land in
the same change, or neither can be verified. It is recorded in
[`2026-07-25-known-limitations.md`](2026-07-25-known-limitations.md) as F-30
with a reproduction, and the golden values freeze the current (biased) behaviour
deliberately.

## Public API

Added: `braintrace.ETraceConfig`, and `.config` on every VJP algorithm.

`braintrace.compile(model, algorithm, ...)` gains an `ETraceConfig` overload in
the `algorithm` position:

```python
learner = braintrace.compile(
    model,
    braintrace.ETraceConfig(trace_factorization='io_factorized', decay=(0.9, 0.0)),
    x0,
)
```

`_resolve_algorithm` maps the config's `trace_factorization` to the engine class
and `compile` forwards `config=`. No new top-level factory function.

Removed: the private `ETraceVjpAlgorithm._include_recurrent_mixing` class
attribute. Subclasses that set it must set `recurrence_scope` instead; because
the name is private and the only in-tree setter was `OSTLRecurrent`, no
deprecation shim is provided.

Unchanged: every preset constructor signature, `decay_or_rank`,
`kappa_filter_decay`, `feedback`, `random_feedback_key`, and every execution
option.

## Test plan

### Golden values (risk 1)

Frozen **before** the refactor, from the pristine `156d058` tree, and compared
after. Captured through `chunked_online_param_gradients` with
`chunk_size < T` — the full-window path returns BPTT for every algorithm at
every hyperparameter (F-23), so golden values taken there would be identical
across all five presets and would guard nothing.

Six preset entries: the five presets plus `EProp(feedback='random')` with a
fixed `random_feedback_key`, since the random-feedback code is being moved
between classes. Two models — `tanh_rnn` (single hidden state) and
`two_state_rnn` (`num_state = 2`, exercising the trailing per-hidden-state axis)
— stored as a `.npz` beside the test.

**Golden coverage must span the trace paths, not just the presets.** The
substitution has a separate code path per trace path, and the preset defaults
exercise only some of them. Measured, the three paths are:

| Configuration | Stepper | Consumer | Jacobian shape |
|---|---|---|---|
| multi-step, `chunked_trace=True` (ParamDim default) | `None` | `_update_etrace_data` | `(T, *varshape, S, S)` |
| multi-step, `chunked_trace=False`; all IO-dim | fused | executor's in-loop stepper | `(*varshape, S, S)` |
| single-step | `None` | `_update_etrace_data` | `(*varshape, S, S)` |

`ParamDimVjpAlgorithm` defaults to `chunked_trace=True`, so a golden set built
only from preset defaults never exercises the param-dim **fused-stepper** path —
exactly the path that needs its own substitution site. The set therefore also
carries `D_RTRL(chunked_trace=False)` and a single-step case.

**The comparison is per leaf, not one joint norm.** A joint relative deviation
over the whole gradient tree can hide a large change in a small-magnitude leaf
behind a large one. The joint figure is kept as a headline diagnostic; the
assertion is per leaf.

Every golden comparison is guarded by `assert_model_is_live` on the model and by
a measured distinctness check, so a golden file of zeros or a harness that
cannot see the axes fails loudly rather than passing (lesson 4).

**Distinctness is a property of a (rule, model) pair, not of a rule.** The first
version of this test asserted that every preset differs from `D_RTRL` on every
model, and it failed — correctly. Two pairs provably collapse:

- `tanh_rnn` / `OSTLFeedforward` ≡ `D_RTRL` to `2.2e-09`. F-29 again: the
  model's only ETP relation is the *recurrent* weight, so the IO-dim input
  factor is the hidden state itself and at `α = 1e-6` the rank-1 product
  reproduces the exact per-parameter trace.
- `two_state_rnn` / `OSTLRecurrent` ≡ `D_RTRL` **bitwise**: the v/a coupling is
  hand-written arithmetic, not an ETP op, so `coupled` has no recurrent mixing
  primitive to trace into the transition and both scopes compile to the same
  Jacobian.

Both are recorded in a `DEGENERATE` table with their reason and asserted to
*stay* collapsed — a pair that starts differing is as much a regression as one
that stops. A separate test pins that no preset is degenerate on *every* model,
which would leave its axis unguarded.

**What the golden set does not prove.** It is a necessary, not sufficient,
inertness check. It does not cover non-dense primitives (conv / sparse / LoRA),
batching or units, descended scans, multiple hidden groups, the legacy
(`fast_solve=False`) solve, or `random_feedback` combined with `kappa`. Those
are covered by the pre-existing suite staying green, which is the other half of
the acceptance. It also *preserves* any pre-existing numerical bug by
construction — including F-30 below.

### The rest

1. **Decay split.** `decay_or_rank=0.9` equals `decay_or_rank=(0.9, 0.9)`
   element-wise; and the asymmetric coordinate `(0.9, 0.0)` — x-side leak,
   f-side instantaneous, the ex-OTTT coordinate — differs from both.
2. **Coordinate table.** Each preset's `.config` asserted field-by-field against
   the table above.
3. **Compatibility matrix.** Each rejected combination raises, and the message
   names the legal pairings. One test per matrix row.
4. **New coordinates run.** `per_param` + `scalar_leak` and `per_param` + `none`
   produce finite gradients, differ from `jacobian`, and differ from each other
   — asserted on each of the three trace paths, so a substitution site that is
   missed on one path cannot hide behind another.
5. **Lifted features generalise.** `pp_prop` with
   `learning_signal='random_feedback'` runs and differs from `symmetric`, and
   `io_factorized` + `coupled` runs and differs from `diagonal` — both were
   `EProp`/`per_param`-only before.
6. **Guards fail loudly.** `coupled` on a model with a descended scan raises;
   `random_feedback` with no allocated matrix raises rather than silently
   computing `symmetric`.
7. **Regression.** The whole existing suite, unchanged, plus `mypy braintrace`.

Assertion tolerances follow lesson 5: `1e-6` relative against deviations of
`1e-3`–`1e-1`, never a threshold below float32 round-off (~`1e-8`).

## Risks

1. **The refactor moves numerics silently.** Mitigated by the golden set above.
   The riskiest single edit is the decay split, because it is the only change
   that touches a live numerical path; it is covered by an explicit
   `0.9 == (0.9, 0.9)` element-wise test in addition to the golden set.
2. **The Jacobian substitution is applied twice, or zero times.** Two paths
   consume the Jacobians (fused stepper, non-fused `_update_etrace_data`) and
   the engines call their own steppers internally. Mitigated by doing the
   substitution at exactly two call sites, both in the base class — and by
   *never wrapping the virtual method itself*, only the object it returns at
   those sites. Verified by a test asserting `scalar_leak` at `λ = 0.9` differs
   from `jacobian` (it would not if the substitution were skipped) and that
   `none` gives a strictly instantaneous trace, run across all three trace paths
   in the coverage table above.
3. ~~**`recurrence_scope='coupled'` under `io_factorized` is untested
   territory.**~~ **Retired by measurement** before implementation: it runs,
   stays finite, and is distinguishable from `diagonal` at `3.7e-04`. The matrix
   admits it and a test pins it (lesson 1 — measure before scoping).
4. **Lifting the κ-filter changes `EProp`'s numerics.** The filter code moves
   verbatim, and `EProp`'s golden entry is in the frozen set.
5. **A lifted feature degrades silently instead of failing.** An unallocated
   random-feedback matrix would make `_compute_learning_signal` fall through to
   the symmetric branch and compute a different rule under the requested name.
   Mitigated by raising on that path (see *The state lifecycle of the lifted
   features*) and by a test that constructs the degraded state and asserts it
   raises.

## Out of scope

SnAp-n (P3), UORO / modulatory / DNI (P4), the benchmark suite (P5), and any
change to `hidden_group.py` — P1 found no defect there and P3 inherits it
directly.
