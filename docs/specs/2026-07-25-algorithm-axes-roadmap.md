# Roadmap: orthogonal axes for online-learning algorithms

Status: design, awaiting review. S1 (removal) is implemented.
Target release: 0.3.0 (single breaking release; all sub-projects land in it)

## Why

`braintrace` is a framework for online learning in brain simulation. A framework
ships *general* mechanisms; it should not ship learning rules that only work for
one operator type and one model shape. Three rules in Layer 4 violate that
today, and three more are named top-level classes for what are really
configurations of the two general engines.

The Nature Communications work established the model-agnostic abstraction
(AlignPre / AlignPost), the linear-memory rule (pp-prop), and the compiler that
generates online-learning code from a user-defined SNN. The claim that followed —
that fragmented rules (e-prop, OSTL, OTPE, …) can be *described, implemented,
compared and deployed* in one compiler framework — is currently backed by
separate hand-written classes, several of which reject most of the operator set.
This roadmap makes the claim structural instead of incidental: the rules become
coordinates in an explicit axis space, and any coordinate works for every ETP
primitive.

## Removal criteria

An algorithm keeps a standalone implementation only if it passes **both**:

1. **Model-agnostic.** It must hold for any ETP primitive (dense, conv, sparse,
   lora, element-wise) and any hidden-state dynamics. An implementation that
   whitelists primitives fails this test.
2. **Mathematically independent.** It must contribute a recursion or estimator
   that the general engines cannot express as a configuration. A named class
   whose coordinates coincide with an existing one contributes nothing.

## The axes

### Axis 1 — `trace_factorization` (spatial factorization → memory)

| Value | Trace shape | Memory | Origin |
|---|---|---|---|
| `per_param` | `(param_shape, H)` | O(P·H) | existing `ParamDimVjpAlgorithm` |
| `io_factorized` | `ε_x ⊗ ε_f` | O(I+O) | existing `IODimVjpAlgorithm` |
| `sparse_n` | n-step influence sparsity pattern | between the two | new (S4) |
| `random_projection` | rank-1 random factors `(s̃, θ̃)` | O(P+H), **unbiased** | new (S5) |

### Axis 2 — `temporal_recursion` (how the trace advances in time)

- `jacobian` — `ε ← D·ε + …` (D-RTRL, OSTL 'with-H', e-prop)
- `scalar_leak` — `R̂ ← λ·R̂ + …`. The additive term is the same per-primitive
  hidden→weight Jacobian contribution the `jacobian` recursion already uses, so
  swapping the recursion is primitive-agnostic. This axis value is what remains
  of OTPE after the algorithm itself was removed.
- `none` — no temporal accumulation (feedforward regime)

Under `io_factorized` this axis is a **pair** `(x-side, f-side)`. The current
`IODimVjpAlgorithm` shares one `α` across `ε_x` and `ε_f` (derived by
`_format_decay_and_rank`); splitting it into two independent decays is the only
change in this roadmap that touches a numerical path in existing code.

### Axis 3 — `learning_signal` (where the signal comes from)

- `symmetric` — `∂L/∂h` back-propagated through the readout (default)
- `random_feedback` — fixed random projection; promotes the existing
  `FixedRandomFeedback` helper to a first-class strategy
- `modulatory` — three-factor: a scalar / low-dimensional neuromodulatory signal
  (TD error, reward-prediction error) times the trace (new, S6)
- `bootstrapped` — synthetic gradient / DNI: a learned estimate of the future
  gradient (new, S6)

DRTP / target projection is **not** a value on this axis. See "Removals".

### Axis 4 — `trace_filter`

`none` · `kappa` (e-prop's low-pass `ē ← κ·ē + ε`)

### Axis 5 — `update_schedule`

`per_step` · `window(k)` · `sequence_end`

### Compatibility matrix

The axes are not fully orthogonal: `random_projection` carries UORO's own rank-1
update and normalisation and cannot be paired with an arbitrary
`temporal_recursion`. `ETraceConfig` validates the `(factorization ×
recursion)` combination at construction and raises a readable error listing the
legal pairings. The matrix is explicit data, not scattered `if` statements.

## Coordinates of the existing algorithms

| Algorithm | factorization | recursion | signal | filter |
|---|---|---|---|---|
| `D_RTRL` | per_param | jacobian | symmetric | none |
| `pp_prop` | io_factorized | jacobian | symmetric | none |
| `OSTLRecurrent` | per_param | jacobian | symmetric | none |
| `OSTLFeedforward` | io_factorized | none | symmetric | none |
| `EProp` | per_param | jacobian | symmetric \| random_feedback | **kappa** |
| ~~`OTPE`~~ (exact) | per_param | **scalar_leak** | symmetric | none |
| ~~`OTPE`~~ (F-OTPE) | io_factorized | **scalar_leak** | symmetric | none |
| ~~`OTTT`~~ | io_factorized | (x: scalar_leak, f: none) | symmetric | none |
| ~~`OSTTP`~~ | per_param | jacobian | target_projection | none |

The three struck-through rows all whitelist dense-matmul primitives and are
single-step only; the table is the proof behind the removal decisions.
`OSTLRecurrent` matches
`D_RTRL` cell for cell — and the source says so directly ("delegates entirely to
`ParamDimVjpAlgorithm`").

**Deleting the classes does not delete the capability.** After the axes exist,
OTTT's coordinate is still reachable as a configuration — and reachable for
conv / sparse / lora, which the current OTTT rejects. The framework ships only
general mechanisms while the "one framework describes them all" claim gets
*stronger*, not weaker.

## Removals

### OTTT

`ottt.py` hard-codes `_SUPPORTED_PRIMITIVES = {etp_mm_p, etp_mv_p}` and raises
`NotImplementedError` for lora / sparse / conv / element-wise relations, plus
`'OTTT v1 supports single-step only'`. Fails criterion 1. Its coordinate is
`io_factorized` with an x-side leak and an f-side that does not accumulate —
one point in the configuration space once axis 2 splits per side.

### OSTTP

`osttp.py` fails criterion 1 for engineering rather than mathematical reasons:
`y_target` has to reach every HiddenGroup through a bespoke path, and `B_list`
is hard-bound to the HiddenGroup count. DRTP itself is just `random_feedback`
applied to the target instead of the error. **Decision: remove the value as well
as the class** — axis 3 carries no `target_projection`, and no generic
`y_target` plumbing is introduced for it.

Note for S6: `modulatory` needs a general external-signal injection path. That is
what OSTTP's `y_target` was reaching for and got wrong. Build it generically.

### OTPE

`otpe.py` fails criterion 1 harder than OTTT does, and says so itself: its
docstring states the published derivation is *"narrower than OTTT's"*. It
carries the same `_SUPPORTED_PRIMITIVES = {etp_mm_p, etp_mv_p}` whitelist and
the same `'OTPE v1 supports single-step only'` guard, hard-codes the dense outer
product (`jnp.einsum('bi,bo->bio', x, df_proj)`) instead of routing through the
per-primitive rule registry, rejects `num_state > 1` outright (so ALIF and any
adaptation variable are out), and errors on any relation touching more than one
HiddenGroup. On top of that it assumes a **single global time constant**, is
**feed-forward only**, and is gradient-exact for **one hidden layer**.

For brain simulation those assumptions are backwards: heterogeneous time
constants and adaptation variables are the norm. What generalises is the
`scalar_leak` recursion, which survives as an axis-2 value; what does not is
OTPE-the-published-algorithm, which is `scalar_leak` plus that whole restriction
stack.

### Verified deletion footprint

| Target | Verification |
|---|---|
| `_algorithm/ottt.py`, `ottt_test.py` | — |
| `_algorithm/osttp.py`, `osttp_test.py` | — |
| `_algorithm/otpe.py`, `otpe_test.py` | — |
| `_common.py::PresynapticTrace` + its tests | grep confirms `ottt.py` is the only user |
| `_common.py::extract_y_target` + its tests | grep confirms `osttp.py` is the only user |
| imports / `__all__` in both `__init__.py` levels | `_algorithm/__init__.py`, `braintrace/__init__.py` |
| `docs/apis/algorithms.rst`, `docs/apis/index.rst` | — |

Referencing files that need edits but not deletion: `oracle_models.py`,
`vjp_base.py`, `_compile.py`, `_compiler/graph.py`, `__init___test.py`,
`_compile_test.py`, and in `_algorithm/tests/`: `approx_correctness_test.py`,
`exact_correctness_test.py`, `public_api_test.py`,
`transform_correctness_test.py`.

The `_get_update_aux` / auxiliary-data hook in `vjp_base.py` is **kept**. OSTTP
was its only consumer, but it is the general per-call side channel that S6's
`modulatory` signal needs; only the OSTTP-specific prose was rewritten.

Explicitly kept: `KappaFilter` (standalone utility; its docstring already notes
e-prop filters internally) and `FixedRandomFeedback` (promoted to the axis-3
`random_feedback` strategy).

## Public API shape at 0.3.0

```python
# Engines — the hosts for the axes
ETraceVjpAlgorithm          # base class; hooks delegate to strategies
ParamDimVjpAlgorithm        # factorization = per_param
IODimVjpAlgorithm           # factorization = io_factorized

# braintrace's own contributions — keep the names
D_RTRL, pp_prop

# Configuration surface
ETraceConfig(factorization=..., recursion=..., signal=..., filter=..., schedule=...)

# Strategy protocols — third-party extension points
TraceFactorization, TemporalRecursion, LearningSignalSource, TraceFilter, UpdateSchedule

# Literature presets — thin factories over ETraceConfig
EProp, OSTLRecurrent, OSTLFeedforward
```

`braintrace.compile(model, algo, x0, **kw)` keeps its signature; `algo`
additionally accepts an `ETraceConfig`.

**Mechanism vs surface.** Internally the axes are strategy objects injected into
one engine — this is isomorphic to the existing template-method hooks
(`_compute_learning_signal`, `_solve_weight_gradients`, `_update_etrace_data`,
`_make_etrace_stepper`, `init_etrace_state`), so the delta is small and each
strategy is independently testable. Externally `ETraceConfig` parses into a
strategy combination, which keeps the user-facing API simple and makes the
configuration space enumerable — S7's benchmark suite gets that for free.

A declarative learning-rule IR (rules as a compiler IR layer, third parties
adding rules without touching the engine) is the natural long-term direction but
is deliberately **out of scope**. With ~8 rules the correct abstraction boundary
is not yet observable. Revisit once S4–S6 push the count past ~15.

## Compatibility

0.3.0 is a clean break. `OTTT`, `OSTTP` and `OTPE` are removed outright with no
shim;
the changelog documents the migration path. The repo already carries a `_legacy`
deprecation layer, and stacking another one only thickens the debt.

## Sub-projects

All land in 0.3.0. Ordering below reflects the dependency structure, not
separate releases.

### S1 — Removal (do first) — **implemented**

Delete `ottt.py` / `osttp.py` / `otpe.py` and their helpers, clean every
reference. Small, mechanical, no numerical risk. Doing it before S2 shrinks the
compiler-refactor surface by three algorithms that would otherwise have to be
kept working and then deleted.

Coverage that pointed at the removed algorithms was **repointed, not dropped**,
wherever the assertion was about a general property:

- the approximate-gradient descent backstop now runs on `pp_prop(rank=1)` and
  `EProp(feedback='random')` instead of `OTTT`/`OTPE`;
- the one-step D_RTRL equivalence tests now use `EProp(kappa_filter_decay=0)`,
  which is D_RTRL's trace with no filter (verified: passes element-wise);
- `public_api_test.py` gained a guard asserting the removed names stay gone.

The direction-alignment metric helpers (`cosine_similarity`, `sign_agreement`,
`relative_magnitude`, `assert_direction_aligned`) are retained — they are the
basis of S7's benchmark suite.

**Acceptance:** full suite green; `grep -r "OTTT\|OSTTP\|OTPE\|PresynapticTrace\|extract_y_target"`
over `braintrace/` and `docs/` hits only `changelog.md`, the removal-notice
docstrings, and the guard test; mypy clean; `py.typed` intact.

### S2 — Compiler: multi-timescale and heterogeneous populations

Work through the known-limitation list already recorded in `AGENTS.md` and the
`dev/` findings: heterogeneous-population leak resolution, multi-state
HiddenGroups, approximation validity beyond shallow depth, single-readout and
feedback-shape assumptions. This comes before the algorithm work so that the
existing rules actually run on realistic brain-simulation models (LIF + ALIF +
multi-timescale synapses) before the algorithm family is widened.

The Jacobian-representation part of this sub-project is a prerequisite for S4.

**Acceptance:** each limitation item has either a passing test or an explicit,
documented scope boundary. No expected-failure item silently remains.

### S3 — Axis decomposition

Turn the five axes into code: the strategy protocols, `ETraceConfig` and its
compatibility matrix, the engine hooks rewired to delegate to strategies, and
the three literature presets (`EProp`, `OSTLRecurrent`, `OSTLFeedforward`)
rewritten as thin factories over `ETraceConfig`. Includes the
`IODimVjpAlgorithm` decay split — the one numerical change in this roadmap.

Everything from S4 onward adds values to these axes, so this is the last piece
of groundwork.

**Acceptance:**
- Element-wise equality against golden values frozen *before* the refactor, for
  all five presets (`D_RTRL`, `pp_prop`, `EProp`, `OSTLRecurrent`,
  `OSTLFeedforward`).
- `decay_or_rank=0.9` equals the new two-sided `(0.9, 0.9)` element-wise.
- Illegal axis combinations raise a readable error naming the legal pairings,
  with test coverage.
- Every preset's coordinates are asserted against the table in this document, so
  the table cannot silently drift from the code.

### S4 — SnAp-n sparsity axis

Menick et al. 2021. Derive the n-step influence sparsity pattern automatically
from the compiler's jaxpr hidden→hidden reachability graph — this is the thing a
compiler can do that a hand-written library cannot, and it sits directly on the
paper's technical moat.

**The work is in the compiler, not the algorithm.** The current machinery keeps
only the block diagonal of `D^t` (`HiddenGroup.diagonal_jacobian`,
`block_diagonal_last_dim`); SnAp-n needs off-block-diagonal terms within n steps.
Assume the existing representation does not survive n>1 and budget for a new
sparse representation, shared with S2.

**Acceptance (two-sided squeeze):**
- `n = 1` equals the existing `D_RTRL` element-wise (regression guard).
- `n ≥ graph diameter` equals the **BPTT oracle** element-wise, on models whose
  hidden-state coupling the compiler fully captures. Full RTRL and BPTT compute
  the same total gradient, so `oracle.py` is the correct instrument and no
  separate full-RTRL reference is needed. (A full-RTRL reference is worth
  writing only to assert the influence matrix `dh^t/dθ` itself when localising a
  divergence between trace and learning signal — optional debugging aid.)
- Measured memory curve, monotone in n.

### S5 — UORO (unbiased random-projection estimator)

Tallec & Ollivier 2018. Adds the `random_projection` factorization with UORO's
rank-1 update and normalisation. Complements the existing biased diagonal
approximations. KF-RTRL / OK are optional lower-variance siblings on the same
axis value.

**Acceptance is statistical, not element-wise.** An unbiased estimator will not
match BPTT on any single run. The guard is: fixed model, fixed sequence, N random
seeds; the deviation of the mean gradient from BPTT shrinks as 1/√N (confidence
interval test). A single run asserts only shape, finiteness, and absence of NaN.
This requires statistical test infrastructure the repo does not have — count it
in the sub-project's cost.

### S6 — Learning-signal axis: three-factor and DNI

- `modulatory`: three-factor learning — trace × neuromodulatory signal (TD error,
  reward-prediction error), enabling reward-based e-prop and online policy
  gradient. Needs the general external-signal injection path noted above.
- `bootstrapped`: DNI / synthetic gradients (Jaderberg 2017) — a learned
  bootstrap for the future-loss gradient that every online rule truncates away.
  Requires an auxiliary network with its own training loop.

**Acceptance is by degeneracy, plus task performance.** `modulatory` must equal
`symmetric` element-wise when the modulatory signal is set to `∂L/∂h`;
`bootstrapped` must equal `symmetric` when the synthesiser output is pinned to
the true value. Add an end-to-end RL smoke test. Neither is a "more accurate
gradient", so element-wise comparison against BPTT is the wrong instrument.

### S7 — Unified benchmark suite

Enumerate the `ETraceConfig` space and, for a fixed model, report gradient cosine
similarity and relative deviation against BPTT, peak memory, per-step wall time,
and task metrics. Machine-readable output (JSON/CSV) plus a reproducible script.
This turns the paper's "comparable" claim into something executable.

Depends on S3's axes; grows as S4–S6 land.

## Three acceptance paradigms

Mixing these up is the most likely way to get this roadmap wrong:

| Paradigm | Applies to | Instrument |
|---|---|---|
| Element-wise equality | S1 refactor, SnAp-1, SnAp-∞ | golden values / BPTT oracle |
| Statistical convergence | UORO and unbiased siblings | 1/√N confidence interval over seeds |
| Degeneracy + task metric | modulatory, bootstrapped | reduce to `symmetric`, then RL smoke test |

The existing taxonomy in `AGENTS.md` (exact vs approximate) stays valid and is
extended by the statistical class.

## Risk register

1. **Axis refactor changes numerics silently.** Mitigation: before touching the
   engine, freeze reference gradients for all five presets as golden values;
   assert element-wise equality after. Separately, the `IODim` decay split must
   prove `decay_or_rank=0.9` equals the new `(0.9, 0.9)` element-wise.
2. **The sparse Jacobian representation does not generalise** (S2/S4). Mitigation:
   design the representation once, in S2, with SnAp-n as the explicit consumer.
3. **Statistical tests are flaky in CI** (S5). Mitigation: fixed seeds, generous
   intervals, and a separate slow-test marker.
4. **`modulatory` re-creates OSTTP's plumbing mistake** (S6). Mitigation: the
   injection path must not bind to HiddenGroup count or readout shape; add a test
   with a model whose HiddenGroup count differs from the signal dimension.
5. **0.3.0 carries the whole roadmap.** All breaking changes and all new
   algorithm families ship together. Mitigation: sub-projects merge
   independently behind the axis interfaces, and the benchmark suite (S7) runs on
   every merge so regressions surface per sub-project rather than at release.
