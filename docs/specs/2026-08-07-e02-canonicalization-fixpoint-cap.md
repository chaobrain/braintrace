# E-02 — bound the control-flow canonicalization fixpoint

Status: implemented
Scope: `braintrace/_compiler/canonicalize.py`
Backlog entry: E-02 in `2026-08-07-deferred-engineering-backlog.md`
Issue: [#157](https://github.com/chaobrain/braintrace/issues/157)

## The defect

Control-flow canonicalization runs three `while True:` fixpoint loops:

| line | driver | what it repeats |
|------|--------|-----------------|
| 298  | `if_convert_conds`         | `_convert_conds_once` + `inline_jit_calls` |
| 615  | `unroll_inner_scans`       | `_unroll_scans_once` + `inline_jit_calls` |
| 1041 | `canonicalize_control_flow`| both sweeps, alternating |

Each loop exits only when a sweep reports zero rewrites. The termination
argument is "nesting depth is finite, so this terminates" — true for the
jaxprs the compiler is meant to see, but nothing *enforces* it. The size of
each sweep's input is not monotonically decreasing in any quantity the loop
checks: `_convert_conds_once` inlines branch bodies (growing the equation
list), `_unroll_scans_once` emits `length` clones of a body, and
`inline_jit_calls` can surface fresh control flow from either. A jaxpr that
regenerates convertible `cond`/`scan` equations faster than the sweeps
consume them — from a JAX version whose lowering re-wraps something the
canonicalizer just unwrapped, or from a pathological user program — makes
the compiler hang with no output, no diagnostic, and no way to tell it from
a slow-but-progressing compile.

Scan unrolling already has the analogous guard: `scan_unroll_limit` caps how
much work a *single* scan may generate, and exceeding it produces a
`SCAN_UNROLL_SKIPPED` warning naming the offending scan. The fixpoint driver
that repeats those sweeps has no equivalent.

## The fix

### 1. A policy knob

Add to `ControlFlowPolicy`:

```python
fixpoint_iteration_limit: int = 64
```

Maximum number of sweeps any one canonicalization fixpoint may run. It
bounds *loop iterations*, not the size of any single rewrite — the
per-rewrite bound is still `scan_unroll_limit`. The two are independent: a
single sweep may unroll a length-16 scan, and 64 sweeps may run over
nested control flow 64 levels deep.

Validated in `__post_init__`: must be an `int` (not `bool`) and `>= 1`;
otherwise `ValueError`, matching the `scan_descent` validation style. There
is deliberately **no** "unbounded" sentinel — the whole point of the knob is
that no configuration hangs. A user with genuinely deep nesting raises the
number.

Default 64: real ETP models nest control flow a handful of levels deep
(a `cond` in a `scan` body in a `jit`, at worst a few of those), so 64
leaves ~an order of magnitude of headroom while still failing in seconds
rather than never.

### 2. Bounded loops

All three `while True:` become `for _ in range(policy.fixpoint_iteration_limit)`
with the same body and the same early `return` on a zero-rewrite sweep.
Falling out of the loop means the last sweep still rewrote something, so the
fixpoint had not converged: raise.

### 3. A diagnosable error

Raise `braintrace.CompilationError` (already exported; already the
"jaxpr-analysis stage failed" exception) with a message that

- names the driver that failed and the iteration count / policy field,
- names the equations the **last** sweep rewrote — those are the ones still
  churning — as `primitive` + `source_info` summary + the distinguishing
  param (`branches=N` for `cond`, `length=N` for `scan`),
- states the two remedies: raise `fixpoint_iteration_limit` if the nesting
  is genuine, or turn the offending pass off (`cond='opaque'`,
  `scan_unroll_limit=0`).

Collecting the last sweep's equations needs the once-functions to report
*what* they rewrote, not just how many. Rather than widen their return tuple
again — E-02 lands on top of E-02's sibling #158, which already changed
`(closed_jaxpr, n)` to `(closed_jaxpr, n, pending)` for buffered skip
diagnostics — each takes a new keyword-only `converted_log:
Optional[List[JaxprEqn]] = None`; when given, every rewritten equation is
appended. The driver passes a fresh list per iteration, so on exhaustion it
holds exactly the final sweep's equations.

Equation description reuses `jax._src.source_info_util.summarize` via
`braintrace._compatible_imports` if available, else falls back to
`str(eqn.source_info)`-free formatting (primitive + params only) so the
error never fails while being constructed.

## Behaviour matrix

| situation | before | after |
|-----------|--------|-------|
| converging jaxpr (every real model) | terminates | identical result, identical diagnostics |
| nesting deeper than the limit | terminates | `CompilationError` naming the equations; raising the knob fixes it |
| non-converging jaxpr | hangs forever | `CompilationError` in bounded time |
| `fixpoint_iteration_limit <= 0` | n/a | `ValueError` at policy construction |

No converging compile changes: the loop body, sweep order, and returned
jaxpr are untouched, and the limit is only consulted to decide whether to
run another iteration that would otherwise have run anyway.

## Edge cases

1. **Exactly at the limit.** A jaxpr needing exactly `limit` sweeps must
   succeed; needing `limit + 1` must raise. The loop runs `limit` bodies and
   each body returns on convergence, so a jaxpr converging on the `limit`-th
   sweep returns from inside the loop.
2. **`limit == 1`.** Convergence is observed by a sweep that rewrites
   nothing, so a jaxpr whose control flow flattens in one sweep still needs
   a second sweep to confirm it. The limit therefore counts sweeps
   *including* the confirming no-op sweep, and `limit=1` accepts only
   jaxprs with no ETP-relevant control flow at all. The tests assert exactly
   this: a single ETP `cond` needs `limit=2`, `cond`-inside-`scan` needs 3.
3. **`cond='opaque'` + `scan_unroll_limit=0`.** `canonicalize_control_flow`
   does no work per iteration; the first iteration returns at `n_total == 0`.
   Never raises.
4. **Non-int / bool limit.** `fixpoint_iteration_limit=True` is an `int`
   subclass; rejected explicitly so `True` cannot silently mean 1.
5. **Error construction on an equation with no useful source info.** JAX
   equations always carry a `source_info`, but a summarize helper may return
   an empty string; the description omits the location rather than emitting
   `at `.
6. **Diagnostics already emitted before the raise.** Sweeps that ran emitted
   `COND_IF_CONVERTED` / `SCAN_UNROLLED` records. Those stay — the raise is
   an additional failure, not a rollback, and the records help diagnose what
   the loop was doing.
7. **Buffered skip diagnostics on the raise path.** Since #158 each sweep
   *buffers* its skip warnings and only the settling sweep's buffer is
   emitted. On the raise path there is no settling sweep, so the buffer is
   discarded rather than replayed: those warnings describe equations a
   non-converged sweep chose to leave opaque, which is noise next to the
   error naming the equations that would not settle.

## Tests (`braintrace/_compiler/canonicalize_test.py`)

Policy:

- `fixpoint_iteration_limit` defaults to 64.
- rejects `0`, negatives, non-int, and `True`.

`if_convert_conds`:

- a nested cond-in-cond jaxpr converts under the default limit;
- the same jaxpr under `fixpoint_iteration_limit=1` raises
  `CompilationError` whose message names `cond` and mentions the policy
  field;
- a jaxpr with no ETP-relevant cond returns unchanged even at limit 1.

`unroll_inner_scans`:

- nested scan-in-scan unrolls under the default limit;
- raises under limit 1 with `scan` and `length=` in the message.

`canonicalize_control_flow`:

- cond-inside-scan (needs both passes and more than one alternation)
  succeeds at the default limit and raises at limit 1;
- raising the limit to the exact number of sweeps the jaxpr needs makes it
  pass — proving the cap is what failed, not the rewrite.

Equivalence: every "succeeds under the default limit" case also asserts
`eval_jaxpr` of the canonicalized jaxpr matches the original function's
output, so the cap cannot silently truncate canonicalization.
