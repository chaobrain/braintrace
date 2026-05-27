# Design: `braintrace.compile()` entry point + module-level deprecation of v0.1.x legacy shims

Date: 2026-05-27
Status: Approved (brainstorming) — pending implementation plan

## Context

Two related ergonomic changes to the BrainTrace public API, motivated by the
0.2.0 release that moved the package from a class-based operator/parameter API
(`MatMulOp`, `ETraceParam`, ...) to a primitive-based ETP user-API (`matmul`,
`conv`, ...) and a family of algorithm classes (`D_RTRL`, `EProp`, `OTTT`, ...).

1. **Deprecation surface.** The v0.1.x legacy classes are still eagerly imported
   into `braintrace/__init__.py` and listed in `__all__`, and they only warn at
   *instantiation* time (via a private `_deprecate()` inside `_legacy/`). We want
   accessing any of them through the package root to emit a clear, migration-
   oriented `DeprecationWarning`, and we want them out of the advertised public
   surface (`__all__`) so new users are guided to the primitive API.

2. **One-call setup.** Today a user must (a) init model states, (b) construct an
   algorithm instance `Algorithm(model, **opts)`, then (c) call
   `learner.compile_graph(*example_inputs)` before the first `.update()`
   (calling `.update()` first raises "not compiled"). We want a single discoverable
   entry point `etrace_model = braintrace.compile(model, algorithm, *example_inputs, **options)`
   that constructs the learner, eagerly builds the graph, and returns a ready-to-
   `update` model.

Intended outcome: a cleaner public API where the primitive functions + `compile()`
are the obvious path, and the v0.1.x surface is clearly deprecated with actionable
migration messages.

## Current state (verified)

- `braintrace/__init__.py` eagerly does `from ._legacy import (ConvOp, ElemWiseOp,
  ElemWiseParam, ETraceOp, ETraceParam, FakeElemWiseParam, FakeETraceParam, LoraOp,
  MatMulOp, NonTempParam, SpMatMulOp)` and lists all 11 in `__all__`.
- `braintrace/_misc.py` already defines `deprecation_getattr(module, deprecations)`:
  given `{name: (message, fn)}`, returns a `__getattr__` that warns and returns `fn`
  (or raises `AttributeError` if `fn is None`). Currently unused by the package root.
- Each legacy class calls a private `_deprecate(cls_name, replacement)` in its
  `__init__` (`_legacy/_ops.py`, `_legacy/_params.py`), deduped per-process via a
  `_warned` set — i.e. warnings fire on **construction**, not attribute access.
- Algorithms are `brainstate.nn.Module` subclasses constructed as
  `Algorithm(model, name=None, vjp_method='single-step', **algo_specific)`:
  - `ParamDimVjpAlgorithm`/`D_RTRL`: `fast_solve`, `normalize_matrix_spectrum`, `trace_dtype`.
  - `IODimVjpAlgorithm`/`pp_prop` (a.k.a. `ES_D_RTRL`): vjp options.
  - `EProp`: `feedback`, `kappa_filter_decay`, `random_feedback_key`.
  - `OTTT`/`OTPE`: required `leak`, plus `mode`.
  - `OSTLRecurrent` / `OSTLFeedforward`: vjp options (no bare `OSTL` factory — removed in 0.2.0).
  - `OSTTP`: `B_list`, `target_timing`.
- `ETraceAlgorithm.compile_graph(*args)` builds the graph + splits states + inits
  trace state; it is idempotent (`if not self.is_compiled`). `.update(*args)` calls
  `_assert_compiled()` and raises if `compile_graph` was not called.

## Part A — Module-level `__getattr__` deprecation of all v0.1.x legacy shims

### Scope
All 11 legacy names: operator classes `ETraceOp`, `MatMulOp`, `ElemWiseOp`,
`ConvOp`, `SpMatMulOp`, `LoraOp`; parameter classes `ETraceParam`, `ElemWiseParam`,
`NonTempParam`, `FakeETraceParam`, `FakeElemWiseParam`.

### Changes to `braintrace/__init__.py`
1. **Remove** the eager `from ._legacy import (...)` block and remove all 11 names
   from `__all__`.
2. **Add** a module-level `__getattr__(name)` that lazily imports the class from
   `._legacy` and warns with a per-name migration message, then returns it. Reuse
   `_misc.deprecation_getattr` by building a `_DEPRECATED` dict:
   ```python
   def _legacy_attr(import_name):
       def fn():
           import braintrace._legacy as _legacy
           return getattr(_legacy, import_name)
       return fn
   ```
   (Or import lazily inside `__getattr__`.) Each entry maps `name -> (message, class)`.
3. **Add** `__dir__()` returning `__all__ + list(_DEPRECATED)` so tab-completion still
   surfaces the names.
4. **Add** a `if typing.TYPE_CHECKING:` block re-importing the 11 names from `._legacy`
   so mypy / IDEs still resolve them (the package is PEP 561 / mypy-checked as of 0.2.0;
   silent removal would break type-checking of code that references them).
5. **Remove** the instantiation-time `_deprecate()` calls in `_legacy/_ops.py` and
   `_legacy/_params.py` (and the now-unused `_deprecate`/`_warned` plumbing), so the
   single access-time warning at the package root is the one source of truth and the
   common `braintrace.MatMulOp()` path does not double-warn.

### Migration messages (per name)
Format: `"braintrace.<Name> is deprecated since 0.2.0 and will be removed in a future
release; use <replacement> instead."`

| Deprecated | Replacement text |
| --- | --- |
| `MatMulOp` | `braintrace.matmul` (with a `brainstate.ParamState`) |
| `ElemWiseOp` | `braintrace.element_wise` |
| `ConvOp` | `braintrace.conv` |
| `SpMatMulOp` | `braintrace.sparse_matmul` |
| `LoraOp` | `braintrace.lora_matmul` |
| `ETraceOp` | the `braintrace` ETP primitive functions (`matmul`, `conv`, ...) |
| `ETraceParam` | `brainstate.ParamState` + an ETP primitive function |
| `ElemWiseParam` | `brainstate.ParamState` + `braintrace.element_wise` |
| `NonTempParam` | `brainstate.ParamState` with plain JAX ops (keeps the weight out of the ETP graph) |
| `FakeETraceParam` / `FakeElemWiseParam` | plain objects with plain JAX ops |

### Behavior
- `braintrace.MatMulOp` and `from braintrace import MatMulOp` both still work, now
  warning at access time. Default Python warning filters show each distinct message
  once. Returned object is the real (functional) class.
- `braintrace.<unknown>` still raises `AttributeError`.
- The private `braintrace._legacy` submodule keeps the classes available for any
  internal/back-compat use without warnings (it is underscore-private, not a public path).

## Part B — `braintrace.compile(model, algorithm, *example_inputs, **options)`

### Location & export
New module `braintrace/_compile.py` defining `compile`; exported from
`braintrace/__init__.py` and added to `__all__`. (Shadowing the `compile` builtin is
acceptable at the package's public namespace, like `jax.jit`; the internal module uses
the name only for the public function.)

### Signature & semantics
```python
def compile(model, algorithm, *example_inputs, **options):
    """Construct an online-learning algorithm for ``model`` and eagerly build its
    eligibility-trace graph, returning a ready-to-``update`` learner."""
    cls = _resolve_algorithm(algorithm)        # class or registered string
    if not example_inputs:
        raise ValueError(...)                  # eager build needs example inputs
    learner = cls(model, **options)            # options forwarded to the constructor
    learner.compile_graph(*example_inputs)     # trace jaxpr, split states, init trace
    return learner
```
- `model`: `brainstate.nn.Module` whose states are already initialized by the caller
  (`brainstate.nn.init_all_states(model)`), consistent with the current usage pattern.
- `algorithm`: a class (subclass of `ETraceAlgorithm`) used directly, **or** a
  registered string name (case-insensitive).
- `*example_inputs`: positional; forwarded verbatim to `compile_graph` (arrays /
  `SingleStepData` / `MultiStepData`, matching what `.update()` will receive).
- `**options`: keyword; forwarded to the algorithm constructor (`vjp_method`, `leak`,
  `fast_solve`, `trace_dtype`, `feedback`, `B_list`, ...).
- Returns the constructed algorithm instance (itself a `brainstate.nn.Module` with
  `.update()`), so `compile(...)` is exactly equivalent to manual construction +
  `compile_graph`, in one call.

Positional example-inputs vs keyword options make the two groups unambiguous, e.g.:
```python
m = braintrace.compile(model, 'D_RTRL', x0, vjp_method='multi-step', trace_dtype=jnp.bfloat16)
m = braintrace.compile(model, braintrace.OTTT, x0, leak=0.9)
y = m.update(x0)
```

### Algorithm name registry
A module-level dict mapping canonical lowercase names (+ a few aliases) to classes:

| String(s) | Class |
| --- | --- |
| `d_rtrl` | `D_RTRL` |
| `pp_prop`, `es_d_rtrl`, `esd_rtrl` | `pp_prop` |
| `eprop`, `e_prop` | `EProp` |
| `ostl_recurrent` | `OSTLRecurrent` |
| `ostl_feedforward` | `OSTLFeedforward` |
| `otpe` | `OTPE` |
| `ottt` | `OTTT` |
| `osttp` | `OSTTP` |

No bare `ostl` alias — the ambiguous factory was deliberately removed in 0.2.0, so
callers must pick recurrent vs feedforward explicitly. Lookup normalizes via
`name.strip().lower()`.

### `_resolve_algorithm(algorithm)`
- `isinstance(algorithm, type) and issubclass(algorithm, ETraceAlgorithm)` → return it.
- `isinstance(algorithm, str)` → registry lookup; unknown → `ValueError` listing
  available names.
- otherwise → `TypeError` (e.g. an instance, or an unrelated class).

### Error handling
- Unknown string name → `ValueError` with the sorted list of valid names.
- Non-`ETraceAlgorithm` class or wrong type → `TypeError`.
- No `example_inputs` → `ValueError` (eager build chosen; inputs required).
- Missing required option (e.g. `OTTT` without `leak`) → propagated from the
  constructor with its existing message.
- Uninitialized model states → surfaced by `compile_graph` (existing behavior);
  `compile` does not call `init_all_states` (caller's responsibility).

## Out of scope (YAGNI)
- No deferred/lazy `compile()` variant (eager build was chosen).
- No auto `init_all_states` inside `compile`.
- No removal of the legacy classes themselves this release (only deprecation).
- No changes to algorithm constructors or `compile_graph` internals.

## Testing
Part A:
- Accessing each of the 11 names via `braintrace.<Name>` emits `DeprecationWarning`
  (use `pytest.warns`) and returns the same class object as `braintrace._legacy.<Name>`.
- The 11 names are absent from `braintrace.__all__`; the primitive functions remain present.
- `from braintrace import MatMulOp` works and warns.
- A constructed legacy op still functions (no behavior regression) and does not warn a
  second time at construction (instantiation-time warning removed).
- `braintrace.does_not_exist` raises `AttributeError`.

Part B:
- `_resolve_algorithm` resolves a class, each canonical string, and each alias; unknown
  string → `ValueError`; instance/wrong type → `TypeError`.
- No example inputs → `ValueError`.
- End-to-end: `compile(model, 'D_RTRL', x)` returns a learner whose `.update(x)` output
  equals that of a manually built `D_RTRL(model); .compile_graph(x); .update(x)` (clone
  state values across two model instances per the project's deterministic-comparison
  pattern).
- Options are forwarded (e.g. `trace_dtype` reflected on the returned learner;
  `OTTT` requires `leak`).

## Critical files
- `braintrace/__init__.py` — Part A (`__getattr__`, `__dir__`, `TYPE_CHECKING`,
  remove eager legacy import + `__all__` entries) and Part B export.
- `braintrace/_misc.py` — reuse `deprecation_getattr` (no change expected).
- `braintrace/_legacy/_ops.py`, `braintrace/_legacy/_params.py` — remove
  instantiation-time `_deprecate()` plumbing.
- `braintrace/_compile.py` — new: `compile` + `_resolve_algorithm` + registry.
- Tests: `braintrace/_compile_test.py` (new) and a deprecation test (new or folded
  into an existing top-level API test).
