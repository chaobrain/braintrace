# Fix the CI type check (mypy)

## Problem

The `typecheck_and_build` CI job fails on `main`: `python -m mypy braintrace`
reports 36 errors in 10 files. Almost all of them come from the JAX stubs,
which now type:

- `Var.aval` as the abstract base `AbstractValue`, which has no `shape`,
  `dtype` or `ndim`;
- `JaxprEqn.invars` / `Jaxpr.outvars` as `list[Var | Literal]`;
- `Var(...)` with a single argument (the pre-0.6.2 two-argument form is a
  `call-arg` error).

The other error is `FlattedDict.split`, whose stub returns a `NestedDict`
while it returns one `FlattedDict` per filter at runtime.

## Fix

Type-only, apart from the latent bug below.

- `_compatible_imports.shaped_aval(atom)`: a `cast` to
  `jax.core.ShapedArray`. Every atom of a traced array program carries one.
  It is used at every `.aval.shape` / `.dtype` / `.ndim` site mypy flags.
- The membership helpers `find_matched_vars` and
  `find_element_exist_in_the_set` accept `Var | Literal`. They already
  filtered with `isinstance(..., Var)`.
- `hid_param_op`: trainable invars are `Var | Literal`. A literal never
  traces to a `ParamState` (its path is `None`), and the loop narrows on
  that.
- Explicit annotations or justified casts for the remaining sites: the
  detach list in `_detach_invars`, the scan descent body maps (hidden carry
  inits are guarded to be `Var`s), the module temporaries, and the
  `FlattedDict.split` result.
- `new_var`: `# type: ignore[call-arg]` on the old-JAX branch only.

## Latent bug fixed on the way

`IODimVjpAlgorithm.get_etrace_of` tested `etrace_x_key(relation.x_var) is not
None`. `etrace_x_key` returns `id(x_var)` and is never `None`, so for an
element-wise relation (`x_var is None`) it looked up `id(None)` and raised
`KeyError`. It now checks `relation.x_var is not None`.

## Tests

- `python -m mypy braintrace`: `Success: no issues found`.
- `io_dim_vjp_test.py::TestStateLifecycle::test_get_etrace_of_elementwise_weight`
  fails with `KeyError` before the fix and passes after.
- The full suite passes unchanged.
