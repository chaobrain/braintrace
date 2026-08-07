# E-04 — `Embedding` validates unsupported options at construction

Status: proposed
Scope: `braintrace/nn/_embedding.py`
Backlog entry: E-04 in `2026-08-07-deferred-engineering-backlog.md`
Issue: [#159](https://github.com/chaobrain/braintrace/issues/159)

## The defect

`braintrace.nn.Embedding` subclasses `brainstate.nn.Embedding` and inherits its
constructor verbatim. That constructor accepts `padding_idx`, `max_norm`,
`scale_grad_by_freq` and `freeze` and stores each as a plain instance
attribute. `braintrace`'s `update()` then refuses all four:

```python
if (self.max_norm is not None or self.freeze
        or self.scale_grad_by_freq or self.padding_idx is not None):
    raise NotImplementedError(...)
```

So construction succeeds and the *first forward pass* fails. Under
`brainstate.transform.jit` — the shape of every driver in this package — the
first forward pass happens at trace time, arbitrarily far from the constructor
call, inside a stack of transform frames. The traceback points at the trace, not
at the line that passed the option.

The unsupported set is right: all four modify the lookup or its gradient
*outside* the ETP primitive that online learning traces (`max_norm` and `freeze`
insert `stop_gradient`; `scale_grad_by_freq` and `padding_idx` live in the
parent's `custom_vjp` backward rule, which the ETP `embedding` primitive
replaces). Only the *timing* is wrong.

## New behaviour

`__init__` is overridden. It forwards every argument to the parent unchanged,
then validates. Any of the four options set to a non-default value raises
`NotImplementedError` from the constructor call itself.

Validation runs *after* `super().__init__` so the parent keeps ownership of its
own argument validation and its error messages continue to win where they are
more specific. `padding_idx=99` on a 10-row table still raises the parent's
`ValueError`, not our `NotImplementedError` — an out-of-range index is a
different mistake from an unsupported feature, and the more precise diagnosis
should be the one the user sees.

### Error contract

- Type: `NotImplementedError` — unchanged from the `update()` check, so any
  existing `pytest.raises(NotImplementedError)` keeps passing.
- Message: names *which* options were actually passed, then states why. E.g.
  for `Embedding(10, 4, freeze=True, padding_idx=0)`:

  ```
  braintrace.nn.Embedding does not support: freeze=True, padding_idx=0. These
  modify the lookup or its gradient outside the ETP primitive that online
  learning traces. Use braintrace.nn.Embedding with default values for
  max_norm, freeze, scale_grad_by_freq and padding_idx, or use
  brainstate.nn.Embedding if you do not need online-learning traces.
  ```

  Options are listed in a fixed order (`max_norm`, `freeze`,
  `scale_grad_by_freq`, `padding_idx`) so the message is deterministic. The
  previous message named the whole unsupported set regardless of what was
  passed; naming only the offenders is strictly more informative.

### The `update()` check is kept, not deleted

The issue text suggests the `update()` check becomes dead. It does not. The four
options are **plain, public, mutable instance attributes** set by the parent
constructor (`self.freeze = bool(freeze)` and friends) — verified empirically:

```python
layer = braintrace.nn.Embedding(10, 4)   # constructs fine
layer.freeze = True                       # no descriptor, no guard
layer(idx)                                # still reaches update() unsupported
```

Removing the `update()` check would therefore make post-init mutation silently
produce wrong semantics (a frozen table that trains anyway, a `padding_idx` that
receives gradient). The check stays as the second gate. `__init__` and
`update()` share one `_reject_unsupported_options` helper so the set and the
message cannot drift apart.

### `from_pretrained`

The inherited `from_pretrained` classmethod defaults to `freeze=True` and calls
`cls(...)`, so `braintrace.nn.Embedding.from_pretrained(w)` now raises from the
`from_pretrained` call instead of from the first forward pass. This is the same
failure moved earlier — it already failed before, just later. Callers who want
pretrained weights pass `freeze=False` explicitly, which works.

## Docstring

The class currently reuses the parent docstring with a `brainstate` →
`braintrace` string substitution. That text documents `padding_idx` and
`max_norm` as working features and demonstrates both in runnable examples that
now raise. It is replaced with a docstring written for this class: the four
options are documented as *accepted for signature compatibility with*
`brainstate.nn.Embedding` *and rejected at construction*, with a `Raises`
section and examples that only exercise the supported path.

## Tests

`braintrace/nn/_embedding_test.py` (already exists; extended).

- Each of the four options raises `NotImplementedError` at construction — one
  test per option, each asserting the message names that option.
- Several at once: the message names every offender and no others.
- **The regression that matters**: constructing raises, and no forward pass is
  needed to provoke it. Asserted by constructing inside `pytest.raises` with no
  call, and by a companion test that a `jit`-wrapped forward is never reached.
- Explicit defaults (`max_norm=None`, `freeze=False`, `scale_grad_by_freq=False`,
  `padding_idx=None`) construct and run a forward pass.
- `norm_type` is *not* in the unsupported set and is accepted (it is inert
  without `max_norm`).
- Parent `ValueError` for an out-of-range `padding_idx` still wins.
- Post-init mutation of each attribute is still caught by `update()`.
- `from_pretrained` raises at the call with the default `freeze=True` and
  succeeds with `freeze=False`.

## Out of scope

Implementing any of the four options. The unsupported set does not change.
