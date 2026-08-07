# Deferred engineering backlog — from the 0.2.5 package audit

Status: living document
Baseline: the 0.2.5 release branch (`worktree-audit-fixes-0.2.5`, off `3f42a24`)

These items were found by the full-package audit that produced 0.2.5 and were
**deliberately not fixed in it**. Each is a real defect or a real hazard; none is
a release blocker, and each one either carries behavioural risk disproportionate
to a patch release or needs design work rather than an edit.

This list is deliberately separate from
[`2026-07-25-known-limitations.md`](2026-07-25-known-limitations.md). That
document is the **learning-rule correctness** backlog — what the algorithms do
and do not compute, each entry pinned by a test. This one is **engineering
hygiene**: packaging, robustness, API shape, and repository state. Do not merge
them.

Every entry below is filed as a GitHub issue, linked under its heading. E-01
additionally has a written implementation plan at
[`2026-08-07-e01-hidden-gradient-correspondence.md`](2026-08-07-e01-hidden-gradient-correspondence.md);
that plan is **proposed, not implemented** in 0.2.5. E-10 (the sdist lost its
tests alongside the wheel) *was* fixed in 0.2.5 and has been removed from this
list. E-02 was fixed after 0.2.5 and is kept below, marked resolved, so the
audit trail stays readable.

## E-01 — the hidden↔gradient correspondence is asserted, not checked

Tracked as [#165](https://github.com/chaobrain/braintrace/issues/165).

`braintrace/_algorithm/vjp_base.py:1047` carries
`TODO: the correspondence between the hidden states and the gradients should be
checked`, sitting directly above three `assert` statements. `python -O` strips
those asserts, so on an optimised interpreter the correspondence is checked by
nothing at all.

This is the highest-risk marker in the tree: a silent mismatch here mis-routes
cotangents between hidden groups, which produces wrong gradients rather than an
error. The fix is a real check — a raised exception, not an assert — plus a test
that constructs a mismatch and asserts the exception.

## E-02 — the cond-conversion fixpoint is unbounded — **RESOLVED**

Tracked as [#157](https://github.com/chaobrain/braintrace/issues/157). Fixed
after 0.2.5; plan in
[`2026-08-07-e02-canonicalization-fixpoint-cap.md`](2026-08-07-e02-canonicalization-fixpoint-cap.md).

`braintrace/_compiler/canonicalize.py:298,615,1041` were `while True:` fixpoint
loops. Scan unrolling is bounded by `policy.scan_unroll_limit`; the **cond**
conversion fixpoint had no equivalent cap. A pathological jaxpr therefore hung
the compiler instead of erroring with a diagnosable message.

All three loops are now bounded by the new
`ControlFlowPolicy.fixpoint_iteration_limit` (default 64, must be a positive
integer). Exhausting it raises `braintrace.CompilationError` naming the
equations the last sweep was still rewriting — primitive, distinguishing param,
and source location — and pointing at the two remedies (raise the limit, or
turn the offending pass off).

## E-03 — `id(eqn)`-keyed dedup relies on an undocumented liveness invariant

Tracked as [#158](https://github.com/chaobrain/braintrace/issues/158).

`braintrace/_compiler/canonicalize.py:457-458,885-886,928-929` use `id(eqn)`-keyed
`skip_warned` sets to suppress duplicate warnings. This is safe **only** because
the enclosing jaxpr holds every equation alive for the duration, so no `id` can
be recycled onto a different object. That invariant is incidental and written
down nowhere.

Either key on something stable (the equation's index within its jaxpr) or state
the invariant in a comment at each site so a future refactor that drops the
jaxpr reference does not silently start merging unrelated warnings.

## E-04 — `Embedding` rejects in `update()` what `__init__` accepted

Tracked as [#159](https://github.com/chaobrain/braintrace/issues/159).

`braintrace/nn/_embedding.py:68` raises `NotImplementedError` for `max_norm`,
`freeze`, `scale_grad_by_freq` and `padding_idx` — **all four of which the
constructor accepts without complaint**. The failure is deferred to the first
forward pass, which under `jit` can be well after the line that actually made
the mistake.

Move the rejection into `__init__` so the traceback points at the constructor
call. Splitting the validation is the whole change; the unsupported set does not
move.

## E-05 — `jax` is not a declared dependency

Tracked as [#160](https://github.com/chaobrain/braintrace/issues/160).

`pyproject.toml:63-69` does not list `jax`; it resolves transitively through
`brainstate`. CI exercises JAX 0.8 through latest, but that range is not
expressible from package metadata, so a resolver is free to install a version
the project has never been tested against.

Declaring it means picking a floor and committing to it, which is a compatibility
decision rather than a packaging edit — hence deferred rather than done.

## E-06 — modules with no co-located tests

Tracked as [#161](https://github.com/chaobrain/braintrace/issues/161).

Two shipped modules have no `_test.py` sibling, against `AGENTS.md` rule 9:

- `braintrace/_typing.py` — contains executable code (`as_size_tuple`), not just
  type aliases.
- `braintrace/nn/__init__.py` — a deprecation dispatcher whose
  `AttributeError` fallthrough at L119 is a **public contract**: it is what makes
  a removed name fail with a message naming its replacement instead of a bare
  attribute error. (`braintrace/__init___test.py` now covers `__dir__` and the
  exported-cell list, but not the fallthrough itself.)

## E-07 — `_state_managment.py` is misspelled

Tracked as [#162](https://github.com/chaobrain/braintrace/issues/162).
**Resolved** — renamed to `braintrace/_state_management.py` with no shim. The
decision and the full list of updated reference sites are recorded in
[`2026-08-07-e07-state-management-rename.md`](2026-08-07-e07-state-management-rename.md).
The paragraph below is left in its original wording as the record of the state
this audit found.

`braintrace/_state_managment.py` should be `_state_management.py`. It is a
private module, but renaming it is still a breaking import path for anyone who
reached in, so it wants either a deprecation shim or a deliberate decision to
accept the break. Not worth doing inside a patch release.

## E-08 — ~4.5 MB of unreferenced assets in `docs/_static/`

Tracked as [#163](https://github.com/chaobrain/braintrace/issues/163).

Only `braintrace-learning-map.svg` is referenced by the docs build. The rest —
including a 2.7 MB `.pptx` — is carried in every clone. Several entries are
already gitignored (`model-dynamics-supported.pptx`,
`etrace_op_functions.{pptx,pdf}`), which suggests the intent was already to stop
tracking them; the cleanup was never finished. Removing tracked binaries rewrites
nothing but does change what a shallow clone gets, so it is a maintainer call.

## E-09 — the `_compiler` and `_legacy` packages are outside the typing gate

Tracked as [#164](https://github.com/chaobrain/braintrace/issues/164).

`pyproject.toml`'s `disallow_untyped_defs` list was extended in 0.2.5 to cover
every module owning a public symbol. Two packages remain outside it:
`braintrace._compiler` (~54 `no-untyped-def` errors) and `braintrace._legacy`
(~39). Neither exports public API, which is why they were left; bringing them in
is mechanical but not free.
