# P1 — Verification harness: make the oracle axis-aware, rebuild the limitation list

Status: implemented
Parent: [`2026-07-25-algorithm-axes-roadmap.md`](2026-07-25-algorithm-axes-roadmap.md) § P1
Baseline: commit `bc153da`
Target release: 0.3.0

## Premise shift

The roadmap scoped P1 as compiler work: make the surviving rules run on
realistic brain-simulation models — LIF + ALIF + multi-timescale synapses,
heterogeneous leaks, multi-state HiddenGroups — and own `hidden_group.py`'s
Jacobian path before widening the algorithm family.

Scoping measurements against the actual code contradict that premise. The
compiler already passes every one of those targets. What fails is the
*instrument* used to judge it: the oracle path that nearly every gradient
assertion in the repository is written against cannot distinguish one learning
rule from another. It returns bitwise-identical BPTT gradients for `D_RTRL`,
`OSTLRecurrent`, `EProp` and `pp_prop`, at every hyperparameter setting.

P1 therefore delivers a harness that can *see* learning-rule differences, then
re-establishes the known-limitation list against that harness. The
multi-timescale and heterogeneity claims are discharged by pinning them as
passing tests, not by fixing code.

## Evidence

All measurements below were run against `bc153da` on CPU.

### The compiler handles P1's nominal targets

`braintrace/_etrace_model_test.py` holds a ten-model SNN zoo (IF / LIF / ALIF ×
Delta / ExpCu / ExpCo, with STD and STP variants) carrying units, multi-timescale
synapses (`tau_mem`, `tau_syn`, `tau_a`, `tau_std`, `tau_f`, `tau_d`) and an E/I
population split. Compiled under `D_RTRL`:

| Model | HiddenGroups | `num_state` | ETP relations |
|---|---|---|---|
| `IF_Delta_Dense_Layer` | 1 | 1 | 1 |
| `LIF_ExpCu_Dense_Layer` | 1 | 2 | 1 |
| `ALIF_ExpCu_Dense_Layer` | 1 | 3 | 1 |
| `ALIF_ExpCo_Dense_Layer` (E/I) | 1 | 5 | 3 |

With deterministic construction and an input scale that produces spikes, every
one of these — plus per-neuron heterogeneous `tau_mem` and `tau_a` — matches the
BPTT oracle. **No defect was found in `hidden_group.py`'s Jacobian path.**

### The oracle's main entry point is blind to every learning-rule axis

`tanh_rnn(n_in=3, n_rec=4, seed=0)`, `T=8`. Each cell is the relative deviation
*between two settings of the same knob* — how much the knob moves the gradient
at all:

| Oracle path | `pp_prop` 0.99 vs 0.01 | `EProp` κ 0.0 vs 0.95 |
|---|---|---|
| `online_param_gradients` (multi-step, full window) | **0.0e+00** | **0.0e+00** |
| `chunked_online_param_gradients` (`chunk_size=2`) | 2.4e−03 | 2.3e−03 |
| `online_param_gradients_singlestep_naive` | 1.2e+00 | 3.0e+00 |

Through the full-window multi-step path all four surviving algorithms return
gradients bitwise equal to BPTT. This is *correct* semantics, not a bug: a
window spanning the whole sequence has no truncation, so there is nothing left
for a trace approximation, a κ-filter or a recurrence scope to change. The
consequence is what matters — **any assertion whose subject is a learning-rule
axis is vacuous on that path.**

`online_param_gradients(..., vjp_method='single-step')` is not a workaround: it
raises `NotImplementedError` on multi-step input data. The only live paths today
are `chunked_online_param_gradients` with `chunk_size < T` and
`online_param_gradients_singlestep_naive`.

**This is already documented in the code, and that is the sharper form of the
finding.** `chunked_online_param_gradients`' docstring states it outright — the
full-sequence call is "exact reverse-mode and the trace only enters at the
sequence boundary", while chunking "makes the total depend on the eligibility
trace at every chunk boundary — this is the oracle that actually validates trace
correctness." So F-23 is not a discovery about the code; it is a discovery that
the assertions written against `online_param_gradients` do not heed what
`oracle.py` already says. `online_param_gradients` itself carries no such
warning, which is the gap P1 closes.

Scale of exposure: 95 `vjp_method='multi-step'` call sites across 20 test
modules. Most are legitimate — see [Out of scope](#out-of-scope).

### F-22's premise is false

F-22 defers exposing the IODim rank / ES decay / random-feedback bias until an
"SNN multi-population model zoo (LIF/ALIF, multi-layer random feedback)"
exists. Running precisely that model — `ALIF_ExpCo_Dense_Layer`, 3 ETP
relations, `num_state=5`, E/I split, `T=20` — still yields bitwise-exact
agreement for every algorithm, because the cause is the harness, not the model.
No model zoo can revive a dead knob.

`approx_correctness_test.py::test_rank_decay_random_approximations_are_exact_on_rate_model_F21`
records the same misattribution in its docstring: *"these nominally-approximate
configs match BPTT element-wise on a single-HiddenGroup rate model. The model
cannot stress their approximation."* The model is not the reason; all four of
its configurations pass `vjp_method='multi-step'`.

### Two defects in the SNN zoo, one in `io_dim_vjp.py`

- **Non-determinism.** The `_etrace_model_test.py` constructors build weights
  through unseeded `braintools.init.*`, which draws from the global
  `brainstate.random` stream. `factory()` therefore returns a *different* model
  on each call, so `bptt_param_gradients(factory, …)` and
  `online_param_gradients(factory, …)` compare two different networks. This
  violates the contract `oracle.py` states for its `model_factory` argument.
  `oracle_models.py` specs are unaffected — they seed explicitly via
  `jax.random.PRNGKey(seed)`.
- **Silence.** At the default input scale, 8 of 9 probed configurations never
  reach threshold: spike rate 0.00, loss 0, gradients identically zero. A
  BPTT-versus-online comparison then compares zero to zero and passes for any
  algorithm. A related sub-case exists: `ALIF_Delta_Dense_Layer` at spike rate
  0.60 still produced a zero BPTT gradient, so spiking alone is not a
  sufficient liveness criterion — the gradient norm is.
- **conv + trainable bias.** `pp_prop` / `IODimVjpAlgorithm` raises
  `ValueError: Custom VJP bwd rule …` when a conv layer carries a trainable
  bias, in either layout: the bwd rule returns the bias cotangent still shaped
  per-position `(batch, *spatial, out_ch)` instead of reduced to `(out_ch,)`.
  Already pinned by `oracle_test.py::test_pp_prop_conv_bias_known_limitation`
  under `pytest.raises`. `D_RTRL` (param-dim) handles the same model exactly.

## Deliverables

### D1 — `docs/specs/2026-07-25-known-limitations.md`

The in-tree findings list, replacing the untracked `dev/` one that `AGENTS.md`
points at. Every entry carries: ID, the claim, status, the test that pins it,
and the evidence. **Each entry is verified against the current test suite, not
transcribed.**

Reconstructed disposition:

| ID | Claim | Status | Pinned by |
|---|---|---|---|
| F-01 / F-04 | multi-state (`num_state ≥ 2`) HiddenGroups | covered, but by a vacuous assertion | `approx_correctness_test.py::test_d_rtrl_exact_on_two_state_group` — uses the full-window path; re-point |
| F-07 / F-08 / F-09 | OTTT / OTPE bias | dead — removed with the algorithms | — |
| F-17 | drifted implementation facts | resolved | `__init___test.py` Task 7 |
| F-19 / F-20 | OTTT / OSTTP exactness | dead — removed with the algorithms | — |
| F-21 | rank / decay / random-feedback exact on rate models | **misattributed** — cause is the harness, not the model | rewrite (see D2) |
| F-22 | approximation bias needs an SNN multi-population zoo | **premise false** — retire in P1, not P5 | re-point at a finite window |
| F-SCAN / F-SCAN-WEIGHT | weight inside control flow | resolved | `_compiler/base_test.py` |
| F-SINGLESTEP | single-step naive vs BPTT | resolved into a direction-alignment assertion | `oracle_test.py` |
| **F-23** | full-window multi-step oracle path is axis-blind | **active, by design** — documented in `chunked_online_param_gradients` but not heeded by the assertions | D2 |
| **F-24** | `_etrace_model_test.py` factories are non-deterministic | **new, active** | D3 |
| **F-25** | SNN zoo silent at default scale → vacuous comparisons | **new, active** | D3 |
| **F-26** | `pp_prop` / IODim raises on conv + trainable bias | **new, active** (pre-existing) | `oracle_test.py::test_pp_prop_conv_bias_known_limitation` |

This was the table as scoped. Two entries moved during implementation, and one
finding was added that the scoping did not anticipate:

| ID | Change from the table above |
|---|---|
| F-26 | **resolved**, not carried forward — the IO-dim solver now reduces produced leaves to the parameter's shape, and the pinning test became `oracle_test.py::test_pp_prop_conv_bias_matches_bptt` |
| F-27 / F-28 | F-27 was reserved for "an SNN spec that cannot be made live" and was never instantiated; F-28 (`EProp(feedback='random')` assumes a single readout of the HiddenGroup's own width) is **active as a documented scope boundary** in `e_prop.py`, not a defect |
| **F-29** | **new** — `pp_prop(decay_or_rank=1)` is not an approximation at all on a recurrent-only relation: rank 1 means decay 0, so the presynaptic EMA collapses to the exact input. Found by the D2 work, since a config held out as a positive control turned out not to deviate |

The authoritative, verified list is
[`2026-07-25-known-limitations.md`](2026-07-25-known-limitations.md); this table
records what the scoping expected so the two can be compared.

`AGENTS.md`'s prose items map on as follows, and the mapping is recorded in the
list so nothing survives only as prose:

| `AGENTS.md` prose item | Disposition |
|---|---|
| approximation-mode validity beyond shallow depth | successor to F-21 / F-23 |
| heterogeneous-population leak resolution | resolved — pinned by D5 |
| target-signal threading under JIT | dead — died with `OSTTP`'s `y_target` path |
| single-readout / feedback-shape assumptions | verify in D5; document boundary if real |
| cross-algorithm equivalence coverage gaps | successor to F-23 |

### D2 — Axis-aware oracle

`oracle.py` changes, all additive:

- **Window semantics documented per entry point.** Each of the four functions
  states what it can and cannot detect. `online_param_gradients` gains an
  explicit warning that with full-sequence input it equals BPTT for *any*
  algorithm, making it a test of the compiler + ETP-rule stack, not of the rule.
- **`assert_gradients_differ(a, b, *, min_rel)`** — negative control. Asserts
  two gradient trees are *not* equal, so a test that intends to exercise a knob
  fails loudly when the knob is dead.
- **`assert_model_is_live(model_factory, inputs, *, min_norm)`** — asserts the
  BPTT gradient norm is above threshold, so a silent network cannot make a
  comparison vacuous. Keyed on gradient norm, not spike rate (see F-25).

New `braintrace/_algorithm/tests/axis_discrimination_test.py` pins both halves
of F-23, for each pair of axis-distinct configurations:

1. the full-window multi-step path collapses them to identical gradients —
   locking the semantics as understood rather than accidental; and
2. a finite window (`chunked`, `chunk_size < T`) separates them.

This is the meta-test whose absence let F-21 be misattributed.

### D3 — Oracle-ready SNN model specs

New `ModelSpec` factories in `oracle_models.py` wrapping the existing
`_etrace_model_test.py` layers for oracle use, fixing F-24 and F-25 at the
boundary rather than mutating the layer classes:

- deterministic construction — explicit seeding, so repeated `factory()` calls
  are bitwise identical (asserted);
- a recorded input scale that drives the network to spike, with liveness
  asserted through `assert_model_is_live`;
- coverage: LIF / ALIF × ExpCu / Delta / STD / STP, an E/I `ExpCo`
  multi-population, and per-neuron heterogeneous `tau_mem` / `tau_a`.

`ModelSpec` gains an optional input-construction field so the scale travels with
the model instead of living in each test. Existing specs keep their current
behaviour by default.

This is the model zoo F-22 asked for — built because the correctness tests need
realistic models, not because it exposes approximation bias, which it does not.

### D4 — SNN correctness tests

New `braintrace/_algorithm/tests/snn_model_correctness_test.py`: `D_RTRL` exact
against BPTT across the D3 specs, each test paired with a liveness guard.

These tests use the full-window multi-step path deliberately. `D_RTRL` is an
exact algorithm, so the subject of the assertion is the compiler and the ETP
per-primitive rules on a realistic model — which is exactly what that path
tests well. F-23 forbids the path only where the subject is a learning-rule
axis, which is not the case here. Where these tests do compare *across*
algorithms, they use a finite window.

The existing `d_rtrl_test.py::test_snn_*_vjp` and
`pp_prop_test.py::test_snn_*_vjp` assert nothing about gradient values — they
`print(grads)`. They keep their role as shape/smoke coverage and are left
alone; D4 is where value assertions live.

### D5 — Multi-timescale and heterogeneity pinned as passing tests

Within D4's module: `num_state` 1 through 5, multi-timescale synapses,
per-neuron heterogeneous leaks, and E/I populations each asserted exact against
BPTT. This discharges the `AGENTS.md` prose limitations by test rather than by
fix. Also probes the single-readout / feedback-shape assumption; if it is real,
it becomes a documented scope boundary with a named finding.

### D6 — F-22 retired

Re-point F-22's assertion at a finite window and delete the skipped
`test_approximations_diverge_on_snn_multipopulation_DEFERRED`, replacing it with
a live test that measures the approximation's actual bias through
`chunked_online_param_gradients`. F-21's docstring is corrected to attribute the
exactness to the harness.

### D7 — `io_dim_vjp.py` conv-bias fix

Reduce the bias cotangent to the bias's own shape in the custom-VJP bwd rule,
then promote `test_pp_prop_conv_bias_known_limitation` from `pytest.raises` to
an exactness assertion. If the fix is not contained within the bwd rule, F-26
becomes an explicitly documented scope boundary instead — the roadmap's P1
acceptance permits either, but not silence.

### D8 — Roadmap and `AGENTS.md` updates

- P2 and P3 acceptance criteria revised to mandate a finite-window oracle path.
  As written ("equals the current `OSTLRecurrent` element-wise", "`n = 1` equals
  the current `D_RTRL` element-wise") they would pass for any algorithm and
  would not guard the refactor at all.
- A lessons-learned section in the roadmap recording the verified context above.
- `AGENTS.md` § Known limitations repointed from `dev/` to the in-tree list.

## Testing strategy

Test-first, and specifically *vacuity-first*: for each finding, the test that
exposes the vacuity is written and seen to fail before the fix. Two invariants
apply to every new gradient assertion:

- **Liveness** — the reference gradient norm is asserted nonzero. A comparison
  against a zero reference asserts nothing.
- **Discrimination** — an assertion whose subject is a learning-rule axis is
  paired with a check that the axis actually moves the gradient on the path
  used.

Verification for the phase: full `pytest braintrace/` and `mypy braintrace`
clean, matching the P0 bar (2062 passed, 1 skipped at `928219b`; the skip count
drops to 0 once D6 lands).

**Observed at completion:**

```
$ pytest braintrace/ -q -rs
2119 passed, 4 deselected, 262 warnings in 788.97s

$ pytest braintrace/ -m diagnostic -q
4 passed, 2119 deselected

$ mypy braintrace
Success: no issues found in 56 source files
```

2062 → 2119 passed (+57), and 1 skipped → 0: F-22's skip was retired rather than
deferred, and the suite now carries no `skip` or `xfail` marker anywhere. The 4
deselected are the `diagnostic`-marked tests that the default `addopts` excludes
from CI gating; they are run above and pass.

`mypy` needed one config change to stay clean, and it was caused by this work:
`exclude` does not apply to files reached by an import, so
`oracle_models.py`'s import of the `_etrace_model_test.py` layer classes
re-admitted that file and surfaced 52 pre-existing `brainpy.state.*`
attribute errors. A scoped `ignore_errors` override for that one module
restores the existing policy. See lesson 8 in the roadmap.

## Acceptance

Restating the roadmap's P1 criteria against this scope:

- the reconstructed limitation list is committed under `docs/specs/`;
- every item on it has either a passing test or an explicitly documented scope
  boundary;
- no expected-failure item silently remains — the one skipped test is retired;
- F-23 is pinned in both directions by `axis_discrimination_test.py`;
- D3's specs are asserted deterministic and live;
- multi-timescale, heterogeneous-leak, multi-state and E/I claims each have a
  named passing test;
- P2/P3 acceptance criteria in the roadmap name a finite-window path.

## Out of scope

- **Migrating the 95 existing `multi-step` call sites.** For exact algorithms
  and the `_op/` primitive tests, a full-window comparison against BPTT is a
  legitimate test of the compiler and ETP-rule stack — that is most of the 95.
  Only assertions whose *subject* is a learning-rule axis get re-pointed.
- **`hidden_group.py` Jacobian rework.** The roadmap gave P1 ownership of this
  file to keep P3 off a moving target. No defect was found, so P1 leaves it
  unchanged; P3 inherits today's representation. Risk 2 in the roadmap's
  register is thereby retired, not mitigated.
- SnAp-n, new axes, `ETraceConfig`, UORO, three-factor, DNI — P2 through P4.
- Statistical test infrastructure — P4.
- Benchmark suite — P5.
