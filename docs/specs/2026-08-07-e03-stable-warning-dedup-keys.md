# E-03: stable warning dedup keys in `canonicalize.py`

Issue: [#158](https://github.com/chaobrain/braintrace/issues/158)

## Current keying

`braintrace/_compiler/canonicalize.py` runs two canonicalization passes to a
fixpoint (`if_convert_conds` / `unroll_inner_scans`, plus the joint driver
`canonicalize_control_flow`). Each sweep (`_convert_conds_once`,
`_unroll_scans_once`) walks the *top-level* equations of the current jaxpr,
rewrites what it can, and copies through what it cannot. An equation that is
ETP-relevant but not rewritable emits a warning:

| site | warning | emitted from |
| --- | --- | --- |
| `_convert_conds_once` | `COND_CONVERSION_SKIPPED` — relevant cond, unsafe to if-convert | `handle_eqn` |
| `_unroll_scans_once` | `SCAN_UNROLL_SKIPPED` — relevant scan, statically ineligible | top-level sweep loop |
| `_unroll_scans_once` | `RELATION_EXCLUDED_SLICED_WEIGHT` — relevant, eligible scan whose `xs` reach a weight | top-level sweep loop |

Because a skipped equation is copied through unchanged, every subsequent sweep
sees it again and would warn again. The passes suppressed that with a
`skip_warned: set` of `id(eqn)` values, threaded through the fixpoint loop (and
through the joint driver as `cond_skip_warned` / `scan_skip_warned`).

## The invariant it leans on

`id()` is only unique among *live* objects. The dedup is correct only because
the enclosing `Jaxpr`/`ClosedJaxpr` (or, for equations reached inside a `cond`
branch, the branch `ClosedJaxpr` held by the parent equation's params) keeps
every equation object alive for the whole fixpoint, so CPython cannot recycle
an address onto a different equation. Nothing in the code says so. A refactor
that drops the jaxpr reference between sweeps — or that keys off an equation
built and discarded inside a sweep — would silently merge unrelated warnings.

## Why the issue's suggested key does not work

Keying on the equation's index within its jaxpr is *not* stable here: the sweep
rebuilds the equation list every iteration, and rewriting one equation changes
the index of every equation after it. Concretely, with an eligible scan
followed by an ineligible one, sweep 1 warns at index 1; sweep 2 sees the
unrolled body in place of the first scan, so the ineligible scan has moved to
index *N* and warns a second time. (`test_skip_warning_once_across_fixpoint`
covers exactly that shape.) Content-based keys are worse: two genuinely
distinct equations with the same relevance/ineligibility reason produce
byte-identical messages, so a content key would over-suppress.

## Chosen fix — no key at all

Remove the dedup set. Each `_once` sweep *buffers* its skip diagnostics instead
of emitting them, and returns the buffer alongside the rewritten jaxpr. The
fixpoint driver keeps only the most recent buffer per pass and emits it once,
after the fixpoint has settled.

That is correct because the settling sweep is by construction a sweep that
rewrote nothing: it walked the final jaxpr's top-level equations, visiting each
exactly once (no branch inlining happens when nothing is converted), so the
final buffer holds exactly one entry per equation that survives
canonicalization un-rewritten. This holds for the joint driver too — its last
iteration is the one where both sweeps return zero rewrites, and both ran on
the final jaxpr.

Applied per site:

- `_convert_conds_once`: drops the `skip_warned` parameter, returns
  `(closed_jaxpr, n_converted, pending)`. `if_convert_conds` emits the last
  `pending` before returning.
- `_unroll_scans_once`: same, for both `SCAN_UNROLL_SKIPPED` and
  `RELATION_EXCLUDED_SLICED_WEIGHT`. `unroll_inner_scans` emits the last
  `pending` before returning.
- `canonicalize_control_flow`: keeps the latest cond buffer and the latest scan
  buffer, emits both when the joint fixpoint settles.

A buffered entry is just the keyword dict for `diagnostics.emit`, built at
detection time and replayed verbatim.

### Behaviour deltas

- Warning *count* per skipped equation is unchanged (exactly one).
- Skip warnings now land after the pass's `INFO` rewrite records rather than
  interleaved with them, and they describe the *final* jaxpr — if an equation's
  ineligibility reason changes as the fixpoint proceeds, the reported reason is
  the last one, which is the one the user is left with.

## Tests

`braintrace/_compiler/canonicalize_test.py` gains a `TestSkipWarningDedup`
class asserting the dedup neither over- nor under-suppresses:

- two *distinct* unsafe conds with identical messages emit two records (no
  over-suppression), and stay at one record each across a multi-sweep fixpoint;
- an ineligible scan placed *after* an eligible one (the ordering that breaks
  an index key) warns exactly once;
- two distinct ineligible scans with identical reasons emit two records;
- the same, driven through `canonicalize_control_flow`.
