# JAX 0.11 `ClosedJaxpr`/`Jaxpr` merge — compatibility

Status: implemented
Date: 2026-08-18
Scope: `braintrace/_compatible_imports.py`, `braintrace/_compiler/*`,
`braintrace/_algorithm/vjp_graph_executor.py`

## Problem

Installing `jax`/`jaxlib` 0.11.1 breaks 690 of the 2902 tests in the suite.
682 of those failures are the same exception, raised the first time the
executor evaluates a compiler-built transition jaxpr:

```
ValueError: foreach() argument 2 is shorter than argument 1
```

raised from `jax.core.eval_jaxpr`.

### Root cause

JAX 0.11 merged `ClosedJaxpr` into `Jaxpr`. Where the two used to be distinct
types — an *open* `Jaxpr` carrying `constvars` (symbols) and `invars`
(symbols), and a `ClosedJaxpr` pairing an open jaxpr with `consts` (values) —
there is now a single class holding one `all_invars` symbol list plus an
optional list of attached const *values*:

```python
class Jaxpr:
    @property
    def constvars(self): return self._all_invars[:len(self._consts)]
    @property
    def invars(self):    return self._all_invars[len(self._consts):]
```

The const/invar boundary is therefore no longer a property of the *symbols*;
it is derived from how many const **values** are attached. `ClosedJaxpr(jaxpr,
consts)` still round-trips correctly, because attaching `consts` restores the
boundary. But an open jaxpr built with symbolic constvars and no values —

```python
Jaxpr(constvars=[c0, c1], invars=[h], outvars=..., eqns=...)
```

— now reports `constvars == []` and `invars == [c0, c1, h]`.

This is precisely the shape the braintrace compiler builds for every
*transition jaxpr*: a program whose `invars` are the differentiated inputs
(the hidden state at `t-1`, or the ETP primitive's output `y`) and whose
`constvars` are the surrounding intermediates, whose values are supplied at
execution time from the forward pass's captured environment. Three such
builders exist:

| Site | Produces |
|---|---|
| `_compiler/hidden_group.py` (`_simplify_hid2hid_tracer`) | `Hidden2GroupTransition.transition_jaxpr` |
| `_compiler/hidden_group.py` (group transition builder, zero-recurrence fallback) | `HiddenGroup.transition_jaxpr` |
| `_compiler/hid_param_op.py` (`_build_transition_jaxpr`) | `HiddenParamOpRelation.y_to_hidden_group_jaxprs` |

Consumers then recovered the split by reading `jaxpr.constvars` back off the
object. Under 0.11 that returns `[]`, so `eval_jaxpr(jaxpr, consts, *args)`
is handed too few values for `all_invars` and fails arity checking.

### Secondary manifestations

Beyond the arity error, the collapsed split leaks into two other places:

- **SnAp-n position analysis.** `analyze_position_adjacency` seeds its
  reachability walk from `transition_jaxpr.invars` when no explicit
  `hidden_invars` is passed. `build_snap_pattern` was called without it, so
  under 0.11 the seed set silently widened to include every constvar, making
  the derived adjacency conservative (or simply wrong).
- **Error-message assertions.** Three tests assert on a specific raised
  message; the arity `ValueError` pre-empted the expected exception.

## Constraints

`pyproject.toml` declares `jax>=0.8.0` and CI exercises 0.8.0 / 0.9.0 / 0.10.0
/ latest. The fix must therefore work unchanged across 0.8 → 0.11. Raising the
floor to 0.11 is not acceptable: it would drop three supported versions for a
purely internal representation change.

## Design

**Stop recovering the const/invar split from the jaxpr object.** Under 0.11
that information is genuinely not stored there, so any accessor-based
workaround (a weak side table keyed on the jaxpr, a version-branching shim)
would be reconstructing state JAX no longer keeps — fragile across the
`.replace()` and rebuild paths the compiler already uses.

Instead, rely on the fact that every consumer *already knows* the invars of
the jaxpr it holds, because the invars are what it is about to feed in:

- a `y_to_hidden_group_jaxprs` entry has exactly one invar, the relation's
  `y_var`;
- a `HiddenGroup.transition_jaxpr` has one invar per `hidden_invars` entry;
- a `Hidden2GroupTransition.transition_jaxpr` has exactly one, `hidden_invar`.

Given that count, the constvars are the complement — the leading prefix of
`[*constvars, *invars]`. That concatenation is identical on every supported
JAX version (on 0.11 it is `all_invars`; before, it is the two lists), so the
derivation needs no version branch.

### Helpers

Added to `braintrace/_compatible_imports.py`:

```python
def jaxpr_all_invars(jaxpr) -> List[Var]
    """[*constvars, *invars] — the full positional input list."""

def split_jaxpr_invars(jaxpr, num_invars) -> Tuple[List[Var], List[Var]]
    """Split that list into (constvars, invars) given the known invar count."""

def jaxpr_constvars(jaxpr, num_invars) -> List[Var]
    """The constvars alone."""
```

`split_jaxpr_invars` validates `0 <= num_invars <= len(all_invars)` and raises
`ValueError` otherwise, so a miscounted call fails loudly at the compiler
boundary instead of producing a silently misaligned `eval_jaxpr` argument
list.

### Call-site changes

| File | Change |
|---|---|
| `_compiler/hid_param_op.py` | `y_to_hidden_groups` derives constvars with `jaxpr_constvars(jaxpr, 1)` |
| `_compiler/hidden_group.py` | `transition_jaxpr_constvars` populated from the locally-known constvars list, not `jaxpr.constvars`; zero-recurrence fallback passes `[outvar]` literally |
| `_compiler/hidden_group.py` | `build_snap_pattern` call passes `hidden_invars=group.hidden_invars` |
| `_algorithm/vjp_graph_executor.py` | const-var collection uses `jaxpr_constvars(j, 1)` |
| `_compiler/graph.py`, `_compiler/scan_descent.py` | `invars + constvars` → `jaxpr_all_invars(...)` (same set; states the intent) |

`Hidden2GroupTransition` already stored its constvars explicitly
(`other_invars`, assigned from the builder's local variable), so it needed no
change — it is the pattern the other two sites are being brought in line with.

Nothing in the `ClosedJaxpr(...)`-wrapped construction paths
(`canonicalize.py`, `jaxpr_graph.py`, `module_info.py`,
`hidden_pertubation.py`, `scan_descent.py`) changes: attaching values keeps
those splits intact on 0.11.

## Testing

- `braintrace/_compatible_imports_test.py` covers the helpers directly:
  round-tripping an open jaxpr built with a const/invar split, the
  zero-constvar and all-constvar edges, and the out-of-range `ValueError`.
- `braintrace/_compiler/hidden_group_test.py` and
  `braintrace/_compiler/hid_param_op_test.py` gain regression tests asserting
  that a compiled group's `transition_jaxpr_constvars` matches the jaxpr's
  actual leading inputs and that `HiddenGroup.transition` /
  `HiddenParamOpRelation.y_to_hidden_groups` evaluate — these are the two
  assertions that fail on 0.11 before the fix and would catch a regression on
  any version.
- CI gains a `0.11.0` entry in the `jax-version` matrix so the merged-`Jaxpr`
  representation is exercised on every push, not only via "latest".

## Alternatives rejected

- **Weak side table** mapping jaxpr → `num_consts`, populated by a
  `make_open_jaxpr` wrapper. Centralises the fix to one construction helper,
  but the annotation is lost whenever a jaxpr is rebuilt or `.replace()`d, and
  it reintroduces state JAX deliberately removed.
- **New explicit fields on `HiddenParamOpRelation`.** Honest, but
  `y_to_hidden_group_jaxprs` invars are always `[y_var]`, so the field would
  be pure redundancy, and adding a `NamedTuple` field is a breaking change for
  positional construction in tests and downstream code.
- **Raising the `jax` floor to 0.11.** Drops 0.8–0.10 support for an internal
  representation change that is straightforward to abstract over.
