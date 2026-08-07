# E-09 — bring `braintrace._compiler` and `braintrace._legacy` into the typing gate

Status: implemented
Scope: `pyproject.toml` (`[[tool.mypy.overrides]]` `disallow_untyped_defs` list),
`braintrace/_compiler/*.py`, `braintrace/_legacy/*.py`
Backlog entry: E-09 in `2026-08-07-deferred-engineering-backlog.md`
Issue: [#164](https://github.com/chaobrain/braintrace/issues/164)

## The defect

`pyproject.toml` carries one `[[tool.mypy.overrides]]` block whose job is to make
"every public API is typed" a property mypy enforces rather than a convention
people remember. In 0.2.5 that block's module list was extended to cover every
module owning a name in `braintrace.__all__`. Two whole packages stayed outside
it:

- `braintrace._compiler` — the jaxpr-analysis layer (Layer 2 of the architecture
  in `AGENTS.md`), 13 modules.
- `braintrace._legacy` — the frozen v0.1.x back-compat shim, 3 modules.

Neither owns a public symbol, which is the letter of the rule the 0.2.5 list was
written against. But `_compiler` is the layer *every* algorithm depends on, and
an ungated package is not merely "currently unannotated" — it is a place where an
untyped def can silently reappear at any time. The gate is only as strong as its
weakest package.

## Baseline

Measured by adding both packages to the `disallow_untyped_defs` module list
*first*, then running `python -m mypy` (mypy 2.1.0) with no source change:

```
Found 93 errors in 13 files (checked 62 source files)
```

All 93 are `[no-untyped-def]`. Per package:

| package | errors |
| --- | --- |
| `braintrace._compiler` | 54 |
| `braintrace._legacy` | 39 |

Per file:

| file | errors |
| --- | --- |
| `_legacy/_ops.py` | 29 |
| `_compiler/hidden_group.py` | 11 |
| `_legacy/_params.py` | 10 |
| `_compiler/module_info.py` | 9 |
| `_compiler/canonicalize.py` | 7 |
| `_compiler/hid_param_op.py` | 6 |
| `_compiler/graph.py` | 5 |
| `_compiler/base.py` | 4 |
| `_compiler/jaxpr_graph.py` | 4 |
| `_compiler/hidden_pertubation.py` | 3 |
| `_compiler/position_graph.py` | 2 |
| `_compiler/scan_descent.py` | 2 |
| `_compiler/report.py` | 1 |

By mypy's own phrasing: 35 "missing a type annotation" (neither params nor
return), 31 "missing a return type annotation", 27 "missing a type annotation for
one or more parameters".

These match the counts quoted in the issue (~54 / ~39) exactly.

## Constraints this sweep worked under

1. **Typing-only.** No runtime behaviour changes. Where an annotation appeared to
   demand a code change, the annotation was the thing that got revised, not the
   code. See "What the sweep exposed" below for the one place where the
   annotation revealed a real (pre-existing, benign) inconsistency.
2. **The explicit-module-list style is preserved.** The existing block's comment
   says "Explicit module list (no wildcards) so internal / test-helper modules
   are not dragged in", so both packages are listed module by module rather than
   as `braintrace._compiler.*`. The `_compiler/tests/` subpackage is left out for
   that reason (its three files are all `*_test.py` and already fall under the
   top-level `exclude`, but the explicit list means a future non-test helper
   dropped in there would not be gated by accident).
3. **Python floor is 3.11.** `requires-python >= 3.11`, so no PEP 695 generics
   and no `type X = ...` alias statements. Everything uses `typing.TypeAlias` /
   `typing.TypeVar`, matching `braintrace/_typing.py`.
4. **`_legacy` is frozen.** Annotated to satisfy the gate and nothing else: no
   refactor, no style update, no docstring rewrite.
5. **PR #174's guards are untouched.** `_compiler/hidden_group.py`,
   `_compiler/hidden_pertubation.py` and `_algorithm/vjp_base.py` carry explicit
   `if ... raise` shape/dtype correspondence checks that must survive `python
   -O`. None of them were weakened, removed, or converted to `assert`.

## Categories of annotation added

**A. Jaxpr plumbing.** The recurring types come from
`braintrace._compatible_imports` (`Jaxpr`, `ClosedJaxpr`, `JaxprEqn`, `Var`,
`Literal`, `Primitive`), imported there rather than from `jax.core` /
`jax.extend.core` so the version-compat indirection stays in one place — this is
what the already-gated modules do. A jaxpr *atom* (an equation input, which is
either a `Var` or a `Literal`) has no single JAX-side type, so it is spelled
`Union[Var, Literal]` where the code really accepts both.

**B. Existing `_typing.py` aliases.** `Path`, `PyTree`, `Inputs`, `HiddenInVar`,
`HiddenOutVar`, `VarID` and friends were used wherever they already describe the
value, rather than re-spelling `Tuple[str, ...]` etc. at each site.

**C. Nested closures.** A large share of the baseline errors are inner helper
functions (`resolve`, `res`, `splice`, `inline_branch`, `_push`, …) defined
inside an already-annotated outer function. `disallow_untyped_defs` applies to
these too. They are annotated in place; none of them is part of any interface.

**D. `__init__` / dunder returns.** `-> None` on constructors and
`__post_init__`.

**E. Legacy op/param shims.** The `_legacy` weight pytrees are dicts keyed by
`'weight'` / `'bias'` / `'A'` / `'B'`, so the recurring parameter type is
`Dict[str, Any]`, and the arrays flowing through them are `Any` (see below).

**F. One shared alias.** `jaxpr_graph.Atom = Union[Var, Literal]` is introduced
and reused by `canonicalize.py`, which already imports from `jaxpr_graph`. It is
a `typing.TypeAlias` assignment, not a PEP 695 `type` statement, so it works on
the declared 3.11 floor.

**G. Downstream body checks.** Annotating a def also switches mypy's body
checking on for it. Five errors surfaced that way, all at the `brainstate`
boundary and all resolved without touching runtime behaviour — three narrowing
annotations on local bindings, one `cast`, two `# type: ignore[...]` with error
codes and an explanatory comment (the pattern `module_info.py` already used for
its `get_out_treedef_by_cache(...).unflatten` call).

## Where precision was deliberately given up

These are the interesting cases — every place the sweep settled for `Any`,
a `Union`, or a broad container instead of asserting a precise type, and why.

| Site | Chosen type | Why not something tighter |
| --- | --- | --- |
| `_compiler/*` — every `debug_info` parameter (`hidden_group.py` `_simplify_hid2hid_tracer` / `write_jaxpr_of_hidden_group_transition`, `hidden_pertubation.py`) | `Any` | The value is JAX's `core.DebugInfo`, whose import path has moved between JAX versions (`jax.core` → `jax._src.core` → `jax.api_util`). `_compatible_imports` deliberately does not re-export it, and pinning a private JAX path in a signature is exactly the version coupling that module exists to prevent. The value is only ever passed straight back into a JAX constructor. |
| `_compiler/jaxpr_graph.py` `splice` / `resolve_top` / `res`, `canonicalize.py` `resolve` / `inline_branch` / `handle_eqn` — the "atom" parameters | `Union[Var, Literal]` (returns `List[Any]` where the list is a jaxpr atom list) | A jaxpr equation's `invars` are typed `list[Atom]` in recent JAX but `Atom` is a private alias that is not re-exported and has changed shape across versions. The union spells the same thing without importing a private name. The `List[Any]` returns are lists that JAX itself types loosely at the boundary. |
| `_compiler/module_info.py` `_model_that_not_allow_param_assign(model, *args_, **kwargs_)` | `model: brainstate.nn.Module`, `*args_: Any`, `**kwargs_: Any`, `-> Any` | It is a pass-through that calls the model and returns whatever the model returns; the model's return type is by construction unknown (any user pytree). |
| `_compiler/module_info.py` `_check_in_out_consistent_units` | `Sequence[PyTree]` for the invar/outvar trees | These are `jax.tree`-shaped mirrors of the state list whose leaves are `Var`. `PyTree` (itself `Any`) is the alias the codebase already uses for exactly this, so it is at least a *named* imprecision. |
| `_compiler/module_info.py` `ModuleInfo.split_state_outvars` | `Tuple[PyTree, PyTree, PyTree]` | Same: three `jax.tree`-shaped `Var` trees, whose structure mirrors the model's state pytree and is not statically knowable. |
| `_compiler/graph.py` `compiler_context` | returns `Iterator[None]` under `@contextlib.contextmanager` | Precise, listed here only because it is the one place where the *decorated* type and the *def* type differ and a reader may expect `ContextManager`. |
| `_compiler/report.py` `show(file=...)` | `Optional[IO[str]]` | The parameter is forwarded verbatim to `print`, whose stub wants `SupportsWrite[str]`. That protocol lives in `_typeshed` and is not importable at runtime, so `IO[str]` is the closest importable supertype. |
| `_compiler/hid_param_op.py` `HiddenParamOpRelation.y_to_hidden_groups` | returns `List[Any]`; local `hidden_vals: Any` | The list holds a `list[Array]` per group when `concat_hidden_vals=False` and a single `Array` when `True` — the return type is switched by an argument value, which only an `@overload` pair could express, and the method has one internal caller. The local annotation exists because the same name is rebound from list to array. |
| `_compiler/module_info.py` `abstractify_model` second return | `brainstate.util.FlattedDict` via `cast` | `brainstate.graph.states()` is declared `FlattedDict | tuple[FlattedDict, ...]`; the tuple form only occurs when `*filters` are passed, and none are. `cast` is a runtime identity, so this stays typing-only. |
| `_compiler/module_info.py` `split_state_outvars` unpack | `# type: ignore[misc]` | `_state_management.sequence_split_state_values` returns a 2-tuple when `include_weight=False` and a 3-tuple otherwise. Expressing that needs `@overload` on a module outside this issue's scope, so the call site suppresses the unpack error with a comment naming the reason. |
| `_compiler/module_info.py` `get_out_shapes_by_cache(...)[0]` | `# type: ignore[index]` | `brainstate` types the cached out-shapes as its opaque `PyTree` class, which mypy does not consider indexable; at runtime it is the `(out_shapes, state_shapes)` pair. Mirrors the pre-existing `# type: ignore[attr-defined]` on the sibling `get_out_treedef_by_cache(...).unflatten` call. |
| `_legacy/_ops.py` — every `w` / `weights` parameter | `Dict[str, Any]` | The values are weight pytrees whose leaves may be `jax.Array`, numpy arrays or `brainunit.Quantity`. `brainunit` is configured `follow_imports = "skip"` in `pyproject.toml`, so `Quantity` is `Any` to mypy anyway and a narrower leaf type would be a fiction. |
| `_legacy/_ops.py` `general_y2w`, `ETraceOp.xy_to_dw` and every `xw_to_y` / `raw_xw_to_y` return | `Any` | Same reason: the return is produced by `brainunit`/`jax` calls that mypy already sees as `Any`, and the legacy op contract genuinely allows a `Quantity` or a bare array. Declaring `jax.Array` would be a false promise. |
| `_legacy/_ops.py` `ConvOp.__init__(padding=...)`, `dimension_numbers=...` | `Any` | Forwarded to `jax.lax.conv_general_dilated`, which accepts a string, an int, a sequence of pairs, or a `ConvDimensionNumbers`. The pre-existing docstring already says "Padding specification passed to the convolution". |
| `_legacy/_ops.py` `SpMatMulOp.__init__(sparse_mat=...)` | `Any` | The declared type is `brainunit.sparse.SparseMatrix`, which is `Any` under `follow_imports = "skip"`. The `isinstance` check in the body is the real contract and is untouched. |
| `_legacy/_params.py` — every `weight` / `value` parameter and every `execute` return | `Any` | Same weight-pytree reasoning; the values are arbitrary pytrees whose leaves may be `Quantity`. |
| `_legacy/_params.py` `NonTempParam.__init__(**_kwargs)` | `Any` | Deliberately-ignored legacy kwargs; the name already says so. |
| `_legacy/_params.py` `FakeElemWiseParam.op` | `Callable` (i.e. `Callable[..., Any]`) | The attribute is bound either to `ElemWiseOp.raw_xw_to_y` (two arguments) or to a user callable (one), and `execute` dispatches between the two on `self._is_etrace_op`. A precise arity would make one of the two live call sites a type error. |

One extra annotation is *narrowing* rather than widening and is worth naming for
the same reason: `_legacy/_params.py` adds a bare class-level `op: ElemWiseOp` to
`ElemWiseParam`, narrowing the `op: ETraceOp` it inherits from `ETraceParam`. The
constructor already guarantees an `ElemWiseOp`, whose `__call__` takes the weight
alone, and without the narrowing `execute`'s one-argument call is a `call-arg`
error. A bare annotation creates no class attribute at runtime.

The pattern across most of these is the same: `brainunit` is configured
`follow_imports = "skip"` in `pyproject.toml`, so every value that originates
from it — which is every array in the legacy op path, because those paths call
`u.math.*` — is `Any` regardless of what a signature claims. Annotating those
boundaries as `jax.Array` would look more precise while checking nothing, and
would be actively wrong for the `Quantity` paths. Where a value is genuinely
ours — jaxpr vars, paths, hidden groups, relations, diagnostics — a precise type
is asserted.

## What the sweep exposed

No runtime bug: the ETP graph, its Jacobians and its gradients are byte-identical
before and after. Two things are worth recording, because they are the kind of
thing this gate exists to surface.

**1. `ETraceGraph.call_hidden_perturb` documented the wrong return value.** Its
docstring said it returns "the processed model outputs, in the same structure
produced by a normal forward call". It actually returns the four-element
`(outputs, etrace_state_vals, other_state_vals, temp_data)` tuple that
`ModuleInfo._process` produces — which is what its one caller,
`_algorithm/vjp_graph_executor.py`, unpacks. Writing the annotation is what made
the mismatch visible. The annotation is the four-tuple and the docstring's
`Returns` section was corrected to match; no code changed. This is a
documentation defect rather than a behaviour defect, so it is fixed in place.

**2. `_legacy` overrides rename their parameters.** `ETraceOp.xw_to_y(self,
inputs, weights)` is overridden by every concrete subclass as `xw_to_y(self, x,
w)`. The parameter *names* differ, so a keyword call `op.xw_to_y(inputs=...,
weights=...)` works on the base class and raises `TypeError` on every subclass.
mypy does not check parameter-name compatibility on overrides, so the gate does
not flag it, and renaming is exactly the kind of "improvement" constraint 4 above
forbids in a frozen shim. It is recorded here and left alone: no caller in the
package, in the tests, or in the docs uses the keyword form.

Separately, `ElemWiseOp.__call__(self, weights)` drops an argument relative to
`ETraceOp.__call__(self, inputs, weights)`. That *is* a Liskov violation mypy
reports, and it is pre-existing and load-bearing (the element-wise op has no
input). It carries a `# type: ignore[override]`, matching the one
`ElemWiseParam.execute` already had for the identical reason.

## Verification

- `python -m mypy` (mypy 2.1.0, local Python 3.13, `python_version = "3.14"` in
  config) → `Success: no issues found in 62 source files`.
- `python -m pytest braintrace/_compiler/ braintrace/_legacy/ -q` →
  `618 passed`. Both packages have co-located tests: `_compiler` has 12
  `*_test.py` siblings plus the three files in `_compiler/tests/`, and `_legacy`
  has `_ops_test.py` and `_params_test.py`.
- `python -m pytest braintrace/ -q` (the whole suite, which includes
  `braintrace/_algorithm/` — the primary consumer of the annotated compiler
  layer) → `2902 passed, 4 deselected` in 35 min.
- Every changed file re-parsed with `ast.parse(..., feature_version=(3, 11))` to
  confirm nothing depends on syntax newer than the `requires-python` floor.
