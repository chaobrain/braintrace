# E-06 — co-located tests for `_typing.py` and `nn/__init__.py`

Status: proposed
Scope: `braintrace/_typing_test.py` (new), `braintrace/nn/__init___test.py` (new)
Backlog entry: E-06 in `2026-08-07-deferred-engineering-backlog.md`
Issue: [#161](https://github.com/chaobrain/braintrace/issues/161)

Two shipped modules have no `_test.py` sibling, against `AGENTS.md` rule 9. This
is a test-only change: no runtime behaviour is altered. Both new files pin
*current* behaviour, including the sharp edges, so that a future change to
either module is a deliberate decision rather than an accident.

## `braintrace/_typing_test.py`

`_typing.py` is mostly `TypeAlias` declarations, which have no runtime contract
worth testing, plus one executable function: `as_size_tuple`. Its job is to turn
the broad `Size` union (`int | np.integer | Sequence[int] | Sequence[np.integer]`)
into a concrete `tuple[int, ...]` so that `self.in_size = ...` assignment and
`size[-1]` lookups both type-check. Every RNN cell and the readout layer call it
on their constructor arguments, so it is on the hot path of the public layer API.

Pinned contracts:

1. **Normalisation.** Scalar `int` and numpy integer scalars become a 1-tuple;
   sequences become a tuple with order preserved. The result is always a `tuple`
   whose elements are exactly `int` — never `np.integer`, which matters because
   the values are used as shape components and in error messages.
2. **Union coverage.** Every arm of the declared `Size` union is accepted.
3. **Idempotence.** `as_size_tuple(as_size_tuple(x)) == as_size_tuple(x)`; the
   function is safe to apply to a value that already went through it, which is
   exactly what `_rnn.py` does (`_as_size_tuple(self.out_size)[-1]` on a size
   that the setter already normalised).
4. **Round-trip.** The result is assignable to `brainstate` `in_size` /
   `out_size` and survives the setter unchanged — the function's stated reason
   to exist.
5. **No validation.** Zero, negative values and the empty sequence are accepted
   and returned as-is. `as_size_tuple` normalises; it does not police shapes.
   Pinned so that adding validation later is a visible change.
6. **Rejections, by exception type.** `float` / `None` scalars and a 0-d numpy
   array raise `TypeError` (not iterable / not indexable); a nested sequence
   raises `TypeError` from `int()`; a non-numeric string raises `ValueError`
   from `int()`. The function does not normalise these into a single custom
   error, and callers see the raw builtin exception.
7. **Two sharp edges, pinned as facts rather than endorsed.**
   - A float *inside* a sequence is silently truncated toward zero:
     `as_size_tuple((2.7,)) == (2,)`. There is no "integral float" check.
   - A `str` is a `Sequence`, so a numeric string is iterated character by
     character: `as_size_tuple('12') == (1, 2)`, not `(12,)`.
8. **Alias identity.** `ArrayLike`, `DType`, `DTypeLike` and `Size` are the
   `brainstate.typing` objects themselves, not re-declarations that could drift.

Known divergence, deliberately not changed: `brainstate`'s own
`nn._module._format_size_arg` accepts a 0-d integer `np.ndarray`
(`np.array(3)` → `(3,)`), while `as_size_tuple` raises `TypeError` on it. A 0-d
array is not a member of the declared `Size` union, and the `brainstate` layers
that consume such a size fail the same way (verified against
`brainstate.nn.LeakyRateReadout`), so this is out of scope for a test-only
change. The test pins the current `TypeError`.

## `braintrace/nn/__init___test.py`

`braintrace/nn/__init__.py` is a deprecation dispatcher: 8 names forward to
`brainpy.state`, 40 to `brainstate.nn`, each with a `DeprecationWarning` naming
the replacement. `braintrace/__init___test.py` already touches `__dir__` and two
sample forwards from the package-root side; this file owns the dispatcher
itself.

Pinned contracts:

1. **Every** deprecated name resolves — all 48, parametrised, not a sample — and
   resolves to the *same object* the target package exports.
2. The warning is a `DeprecationWarning` whose message names both the old path
   and the exact replacement path (`Use brainstate.nn.X instead.`).
3. `stacklevel=2`: the warning is attributed to the caller's file, so a user
   sees their own line, not `braintrace/nn/__init__.py`.
4. The forward is not memoised — a second access warns again, and the name never
   lands in the module `__dict__`. A dispatcher that cached would warn once per
   process and go silent for every later caller.
5. Real exports (`__all__`) resolve through normal attribute lookup and never
   reach `__getattr__`, so they emit no warning.
6. `__dir__` is sorted, duplicate-free, and is exactly the union of `__all__`
   and the two deprecated tuples.
7. The two tuples are disjoint from each other and from `__all__`. An overlap
   would make a tuple entry dead code that never dispatches.
8. **The fallthrough** (`__init__.py` L119). A module-level `__getattr__` that
   falls off the end returns `None`, so `braintrace.nn.Typo` would silently be
   `None` and fail much later with an unrelated error. The explicit `raise` is
   what makes it fail at the access:
   - an unknown name raises `AttributeError`, and never returns `None`;
   - the message names both the module and the attribute;
   - it raises without warning — the fallthrough is not a deprecation path;
   - `hasattr` is `False` and `from braintrace.nn import Nope` raises
     `ImportError`, both of which depend on the raise;
   - dunder/private probes (`__wrapped__`, `_ipython_canary_method_should_not_exist_`)
     raise too, which is what keeps `inspect`, `copy` and REPL completion from
     seeing a bogus `None` attribute.
9. **Allowlist, not blanket forwarder.** A name that exists in `brainstate.nn`
   but is not listed (e.g. `Sequential`) still raises `AttributeError`. The
   dispatcher forwards a fixed, reviewed set.

### A correction to the issue text

Issue #161 describes the fallthrough as "what makes a removed name fail with a
message naming its replacement". That is not what L119 does. Names with a
replacement never reach L119 — they are handled by the two branches above it,
which *succeed* with a warning naming the replacement. L119 is the case with no
replacement, and its message is the standard
`module 'braintrace.nn' has no attribute 'X'`. Its value is that it raises at
all rather than yielding `None`. The tests pin that reading.

## Out of scope

Changing any behaviour of either module, including the `as_size_tuple`
rejections and the 0-d-array divergence above.
