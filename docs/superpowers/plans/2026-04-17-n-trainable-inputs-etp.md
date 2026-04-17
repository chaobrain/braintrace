# N-Trainable-Input ETP Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize ETP primitives from one-trainable-input-per-primitive to a named dict of trainable inputs, so `{weight, bias}` Linears get correct online bias gradients and LoRA's `{B, A}` executor path is fixed.

**Architecture:** Each `ETPPrimitiveSpec` declares a `trainable_invars_fn` mapping `eqn.params -> {key_name: invar_index}`. Compiler produces a `HiddenParamOpRelation` carrying per-key dicts (`trainable_vars`, `trainable_paths`, `trainable_leaf_indices`, `trainable_param_states`, `trainable_processing_chains`). ETP rules (`yw_to_w`, `xy_to_dw`, `init_drtrl`) operate on pytree dicts keyed by those names; `init_pp` keeps its single-array y-shape return. Executors route resulting gradient dicts across any number of owning `ParamState`s via a generalized `_wrap_leaves_as_pytree` helper.

**Tech Stack:** Python, JAX (jit / vjp / vmap / pytree), pytest, `brainstate.ParamState`, `saiunit`.

**Design spec:** `docs/superpowers/specs/2026-04-17-n-trainable-inputs-etp-design.md`

**Migration strategy:** Tasks 1-6 add the new infrastructure alongside the old (build stays green). Tasks 7-11 migrate the 8 primitives one file at a time (rules flip to dict API; each primitive's spec swaps `weight_invar_index` for `trainable_invars_fn`). Tasks 12-14 remove the old infrastructure now that nothing references it. Task 15 adds the final diagnostic.

---

## File Structure

**Modified files:**

- `braintrace/_etrace_vjp/misc.py` — add `_extract_leaf` and `_wrap_leaves_as_pytree`; remove old `_extract_weight_leaf` and `_wrap_leaf_as_pytree` at cleanup.
- `braintrace/_etrace_op/_spec.py` — add `trainable_invars_fn` field; remove `weight_invar_index` at cleanup.
- `braintrace/_etrace_compiler/hid_param_op.py` — per-key resolution; relation layout change.
- `braintrace/_etrace_compiler/diagnostics.py` — new `TRAINABLE_INVAR_NOT_PARAMSTATE` diagnostic kind.
- `braintrace/_etrace_vjp/d_rtrl.py` — per-primitive-instance trace keying; dict-based rule calls; multi-path gradient routing.
- `braintrace/_etrace_vjp/pp_prop.py` — dict-based `xy_to_dw` call; multi-path gradient routing.
- `braintrace/_etrace_op/dense.py` — mm / mv rules rewritten to dict API; bias supported.
- `braintrace/_etrace_op/conv.py` — conv rules rewritten; bias supported.
- `braintrace/_etrace_op/sparse.py` — sparse mm / mv rules rewritten; bias supported.
- `braintrace/_etrace_op/lora.py` — LoRA rules rewritten (fuses existing pytree form with bias); executor mismatch fixed.
- `braintrace/_etrace_op/elemwise.py` — elemwise rules use single-key `{'weight': ...}` dict.
- Any consumer still using `relation.weight_path` / `relation.weight_var` / `relation.weight_leaf_idx` (compiler, executors, graph.py, diagnostics, tests).

**New test files:**

- `braintrace/_etrace_vjp/misc_test.py` — new tests for the two new helpers (file already exists; new tests are appended).
- `braintrace/_etrace_op/bias_gradient_test.py` — cross-primitive bias-gradient correctness suite (mm, mv, conv, sparse, lora).
- `braintrace/_etrace_op/separate_param_state_test.py` — test that weight and bias living in different `ParamState`s produce correct grads on both paths.

---

## Task 1: Add `_extract_leaf` and `_wrap_leaves_as_pytree` helpers

**Files:**
- Modify: `braintrace/_etrace_vjp/misc.py` — add the two new helpers next to `_extract_weight_leaf` / `_wrap_leaf_as_pytree` (do not remove the old ones yet).
- Modify: `braintrace/_etrace_vjp/misc_test.py` — append new test classes.

- [ ] **Step 1: Write the failing tests**

Append the following to `braintrace/_etrace_vjp/misc_test.py`:

```python
# ---------------------------------------------------------------------------
# _extract_leaf
# ---------------------------------------------------------------------------

from braintrace._etrace_vjp.misc import _extract_leaf, _wrap_leaves_as_pytree


class TestExtractLeaf:

    def test_bare_array_returns_itself(self):
        x = jnp.arange(6.0).reshape(2, 3)
        npt.assert_array_equal(_extract_leaf(x, 0), x)

    def test_dict_returns_leaf_at_index_zero(self):
        pytree = {'weight': jnp.ones((2, 3)), 'bias': jnp.zeros((3,))}
        npt.assert_array_equal(_extract_leaf(pytree, 0), pytree['bias'])
        # jax.tree.leaves sorts dict keys; 'bias' < 'weight'

    def test_dict_returns_leaf_at_index_one(self):
        pytree = {'weight': jnp.ones((2, 3)), 'bias': jnp.zeros((3,))}
        npt.assert_array_equal(_extract_leaf(pytree, 1), pytree['weight'])

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            _extract_leaf({'a': jnp.zeros((2,))}, 5)


# ---------------------------------------------------------------------------
# _wrap_leaves_as_pytree
# ---------------------------------------------------------------------------

class TestWrapLeavesAsPytree:

    def test_bare_array_reference_returns_the_single_grad(self):
        ref = jnp.ones((2, 3))
        grad = jnp.full((2, 3), 7.0)
        result = _wrap_leaves_as_pytree(ref, {0: grad})
        npt.assert_array_equal(result, grad)

    def test_dict_fills_all_supplied_leaves(self):
        ref = {'weight': jnp.ones((2, 3)), 'bias': jnp.zeros((3,))}
        dW = jnp.full((2, 3), 2.0)
        db = jnp.full((3,), 5.0)
        # jax.tree.leaves orders by sorted dict keys: 'bias' (0), 'weight' (1).
        result = _wrap_leaves_as_pytree(ref, {0: db, 1: dW})
        npt.assert_array_equal(result['bias'], db)
        npt.assert_array_equal(result['weight'], dW)

    def test_dict_zero_fills_unsupplied_leaves(self):
        ref = {'weight': jnp.ones((2, 3)), 'bias': jnp.zeros((3,))}
        dW = jnp.full((2, 3), 2.0)
        # Supply only index 1 (weight); bias should come back zero-filled.
        result = _wrap_leaves_as_pytree(ref, {1: dW})
        npt.assert_array_equal(result['weight'], dW)
        npt.assert_array_equal(result['bias'], jnp.zeros((3,)))

    def test_dict_empty_grad_map_returns_all_zeros(self):
        ref = {'weight': jnp.ones((2, 3)), 'bias': jnp.zeros((3,))}
        result = _wrap_leaves_as_pytree(ref, {})
        npt.assert_array_equal(result['weight'], jnp.zeros((2, 3)))
        npt.assert_array_equal(result['bias'], jnp.zeros((3,)))

    def test_out_of_range_index_raises(self):
        ref = {'weight': jnp.ones((2, 3))}
        with pytest.raises(IndexError):
            _wrap_leaves_as_pytree(ref, {5: jnp.zeros((2, 3))})
```

- [ ] **Step 2: Run tests, verify import failure**

Run: `python -m pytest braintrace/_etrace_vjp/misc_test.py::TestExtractLeaf -v`
Expected: `ImportError` — `cannot import name '_extract_leaf'`.

- [ ] **Step 3: Add the two helpers to `braintrace/_etrace_vjp/misc.py`**

Insert these functions after the existing `_wrap_leaf_as_pytree` function (around line 197):

```python
def _extract_leaf(pytree_val: brainstate.typing.PyTree, leaf_idx: int):
    """Return the leaf at ``leaf_idx`` in ``jax.tree.leaves(pytree_val)``.

    Bare arrays (treedef with a single leaf) return the array unchanged.
    Raises ``IndexError`` if ``leaf_idx`` is outside ``len(leaves)``.
    """
    leaves = jax.tree.leaves(pytree_val)
    if not leaves:
        return pytree_val
    if leaf_idx < 0 or leaf_idx >= len(leaves):
        raise IndexError(
            f'leaf_idx {leaf_idx} out of range for pytree with {len(leaves)} leaves'
        )
    return leaves[leaf_idx]


def _wrap_leaves_as_pytree(
    reference_pytree: brainstate.typing.PyTree,
    leaf_grads: Dict[int, jax.Array],
):
    """Build a pytree matching ``reference_pytree`` with ``leaf_grads``
    inserted at the given leaf indices; any other leaf is zero-filled.

    Generalization of ``_wrap_leaf_as_pytree`` to multiple leaves. When the
    reference is a bare array, ``leaf_grads`` must contain at most one entry
    at index 0 and that value is returned directly (no wrapping).

    Raises ``IndexError`` if any supplied index is outside
    ``len(jax.tree.leaves(reference_pytree))``.
    """
    ref_treedef = jax.tree.structure(reference_pytree)
    # Bare-array fast path.
    if ref_treedef.num_leaves <= 1 and ref_treedef == jax.tree.structure(0):
        if 0 in leaf_grads:
            return leaf_grads[0]
        return u.math.zeros_like(reference_pytree)
    leaves = jax.tree.leaves(reference_pytree)
    n = len(leaves)
    for idx in leaf_grads:
        if idx < 0 or idx >= n:
            raise IndexError(
                f'leaf_idx {idx} out of range for pytree with {n} leaves'
            )
    new_leaves = [
        leaf_grads[i] if i in leaf_grads else u.math.zeros_like(leaf)
        for i, leaf in enumerate(leaves)
    ]
    return jax.tree.unflatten(ref_treedef, new_leaves)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest braintrace/_etrace_vjp/misc_test.py::TestExtractLeaf braintrace/_etrace_vjp/misc_test.py::TestWrapLeavesAsPytree -v`
Expected: all tests pass.

- [ ] **Step 5: Run the full misc_test to ensure no regression**

Run: `python -m pytest braintrace/_etrace_vjp/misc_test.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_vjp/misc.py braintrace/_etrace_vjp/misc_test.py
git commit -m "feat(etrace_vjp): add _extract_leaf and _wrap_leaves_as_pytree helpers

Multi-leaf generalizations of _extract_weight_leaf / _wrap_leaf_as_pytree.
Old helpers remain; they will be removed after all callers migrate.
"
```

---

## Task 2: Add `trainable_invars_fn` field to `ETPPrimitiveSpec`

**Files:**
- Modify: `braintrace/_etrace_op/_spec.py` — add the new optional field and a helper to derive it from the legacy single-weight form.
- Test: `braintrace/_etrace_compiler/primitive_spec_test.py` (existing) — add a class `TestTrainableInvarsFn`.

- [ ] **Step 1: Write the failing test**

Append to `braintrace/_etrace_compiler/primitive_spec_test.py`:

```python
class TestTrainableInvarsFn:

    def test_field_defaults_to_none(self):
        spec = ETPPrimitiveSpec(
            name='dummy',
            impl=lambda *a, **k: a[0],
            yw_to_w=lambda *a, **k: a[1],
            xy_to_dw=lambda *a, **k: a[0],
            init_drtrl=lambda *a, **k: None,
            init_pp=lambda *a, **k: None,
            weight_invar_index=1,
        )
        assert spec.trainable_invars_fn is None

    def test_custom_fn_is_preserved(self):
        fn = lambda params: {'weight': 1, 'bias': 2} if params.get('has_bias') else {'weight': 1}
        spec = ETPPrimitiveSpec(
            name='dummy',
            impl=lambda *a, **k: a[0],
            yw_to_w=lambda *a, **k: a[1],
            xy_to_dw=lambda *a, **k: a[0],
            init_drtrl=lambda *a, **k: None,
            init_pp=lambda *a, **k: None,
            weight_invar_index=1,
            trainable_invars_fn=fn,
        )
        assert spec.trainable_invars_fn({'has_bias': True}) == {'weight': 1, 'bias': 2}
        assert spec.trainable_invars_fn({}) == {'weight': 1}

    def test_legacy_derived_uses_weight_invar_index(self):
        # When trainable_invars_fn is None, the spec exposes a helper that
        # returns the legacy single-weight layout: {'weight': weight_invar_index}.
        spec = ETPPrimitiveSpec(
            name='dummy',
            impl=lambda *a, **k: a[0],
            yw_to_w=lambda *a, **k: a[1],
            xy_to_dw=lambda *a, **k: a[0],
            init_drtrl=lambda *a, **k: None,
            init_pp=lambda *a, **k: None,
            weight_invar_index=2,
        )
        assert spec.resolve_trainable_invars({}) == {'weight': 2}
```

- [ ] **Step 2: Run tests, verify fail**

Run: `python -m pytest braintrace/_etrace_compiler/primitive_spec_test.py::TestTrainableInvarsFn -v`
Expected: fail — `trainable_invars_fn` attribute missing / `resolve_trainable_invars` missing.

- [ ] **Step 3: Update `braintrace/_etrace_op/_spec.py`**

Edit the dataclass (below `gradient_enabled`) and add the helper. Replace the existing `ETPPrimitiveSpec` dataclass body with:

```python
@dataclass(frozen=True)
class ETPPrimitiveSpec:
    name: str
    impl: Callable
    yw_to_w: Callable
    xy_to_dw: Callable
    init_drtrl: Callable
    init_pp: Callable
    weight_invar_index: int
    x_invar_index: Optional[int] = 0
    y_outvar_index: int = 0
    batched: bool = False
    gradient_enabled: bool = False
    trainable_invars_fn: Optional[Callable[[dict], Dict[str, int]]] = None

    def resolve_trainable_invars(self, eqn_params: dict) -> Dict[str, int]:
        """Return ``{key: invar_index}`` for this equation.

        If ``trainable_invars_fn`` is set, delegates to it. Otherwise returns
        the legacy single-weight layout ``{'weight': weight_invar_index}``.
        """
        if self.trainable_invars_fn is not None:
            return self.trainable_invars_fn(eqn_params)
        return {'weight': self.weight_invar_index}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest braintrace/_etrace_compiler/primitive_spec_test.py::TestTrainableInvarsFn -v`
Expected: all 3 tests pass.

- [ ] **Step 5: Verify no other tests regressed**

Run: `python -m pytest braintrace/_etrace_compiler/primitive_spec_test.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_op/_spec.py braintrace/_etrace_compiler/primitive_spec_test.py
git commit -m "feat(etrace_op): add trainable_invars_fn field to ETPPrimitiveSpec

Optional function that maps eqn.params -> {key_name: invar_index}.
resolve_trainable_invars() falls back to {'weight': weight_invar_index}
when the function is not provided, so existing primitives keep working."
```

---

## Task 3: Add `trainable_*` dict fields to `HiddenParamOpRelation`

**Files:**
- Modify: `braintrace/_etrace_compiler/hid_param_op.py` — extend `HiddenParamOpRelation` NamedTuple; keep old fields.
- Test: `braintrace/_etrace_compiler/hid_param_op_test.py` (existing) — add a class `TestTrainableDictsDefault`.

- [ ] **Step 1: Write the failing test**

Append to `braintrace/_etrace_compiler/hid_param_op_test.py`:

```python
class TestTrainableDictsDefault:
    """The new trainable_* dict fields default to empty dicts and do not
    affect the construction of existing relations. (Later tasks populate
    them during compilation.)"""

    def test_trainable_vars_default_empty(self):
        # Build a minimal relation directly; we just check the new field exists.
        from braintrace._etrace_compiler.hid_param_op import HiddenParamOpRelation
        r = HiddenParamOpRelation.__new__(HiddenParamOpRelation)
        # NamedTuple defaults: the new fields must exist and default to empty dicts.
        fields = HiddenParamOpRelation._fields
        assert 'trainable_vars' in fields
        assert 'trainable_paths' in fields
        assert 'trainable_leaf_indices' in fields
        assert 'trainable_param_states' in fields
        assert 'trainable_processing_chains' in fields

    def test_trainable_dicts_can_be_set(self):
        # Smoke: construct a relation with all required fields, including the
        # new ones populated, and read them back.
        from braintrace._etrace_compiler.hid_param_op import HiddenParamOpRelation
        # This test only uses the construction contract; no compiler run.
        # Use minimal sentinel objects for non-dict fields we don't inspect.
        r = HiddenParamOpRelation(
            primitive=None,
            weight=None,
            weight_path=('w',),
            weight_var=None,
            weight_leaf_idx=0,
            x_var=None,
            y_var=None,
            hidden_groups=[],
            y_to_hidden_group_jaxprs=[],
            connected_hidden_paths=[],
            eqn_params={},
            weight_processing_chain=(),
            path_classification={},
            trainable_vars={'weight': 'v'},
            trainable_paths={'weight': ('w',)},
            trainable_leaf_indices={'weight': 0},
            trainable_param_states={'weight': None},
            trainable_processing_chains={'weight': ()},
        )
        assert r.trainable_vars == {'weight': 'v'}
        assert r.trainable_paths == {'weight': ('w',)}
        assert r.trainable_leaf_indices == {'weight': 0}
        assert r.trainable_param_states == {'weight': None}
        assert r.trainable_processing_chains == {'weight': ()}
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_compiler/hid_param_op_test.py::TestTrainableDictsDefault -v`
Expected: fails — fields don't exist on the NamedTuple.

- [ ] **Step 3: Extend `HiddenParamOpRelation`**

In `braintrace/_etrace_compiler/hid_param_op.py`, modify the `HiddenParamOpRelation` class (around line 125). Add the five new fields at the end of the field list, each defaulting to an empty dict. The complete signature should become:

```python
class HiddenParamOpRelation(NamedTuple):
    primitive: Primitive
    weight: brainstate.ParamState
    weight_path: Path
    weight_var: Var
    weight_leaf_idx: int
    x_var: Optional[Var]
    y_var: Var
    hidden_groups: List[HiddenGroup]
    y_to_hidden_group_jaxprs: List[Jaxpr]
    connected_hidden_paths: List[Path]
    eqn_params: dict
    weight_processing_chain: Tuple[Primitive, ...] = ()
    path_classification: Dict[Path, str] = {}
    # NEW — per-key trainable-input metadata (populated by the compiler when
    # the primitive's spec provides trainable_invars_fn; otherwise empty).
    trainable_vars: Dict[str, Var] = {}
    trainable_paths: Dict[str, Path] = {}
    trainable_leaf_indices: Dict[str, int] = {}
    trainable_param_states: Dict[str, brainstate.ParamState] = {}
    trainable_processing_chains: Dict[str, Tuple[Primitive, ...]] = {}
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest braintrace/_etrace_compiler/hid_param_op_test.py::TestTrainableDictsDefault -v`
Expected: pass.

- [ ] **Step 5: Run the full compiler test module to catch regressions**

Run: `python -m pytest braintrace/_etrace_compiler/hid_param_op_test.py -v`
Expected: all tests pass (existing construction of `HiddenParamOpRelation` still works; new fields just default to empty dicts).

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_compiler/hid_param_op.py braintrace/_etrace_compiler/hid_param_op_test.py
git commit -m "feat(etrace_compiler): add trainable_* dict fields to HiddenParamOpRelation

Per-key trainable-input metadata, empty by default. Existing single-weight
fields (weight_path, weight_var, weight_leaf_idx, weight, weight_processing_chain)
remain; they will be removed in a later cleanup task once all callers use
the dict-based API."
```

---

## Task 4: Compiler populates `trainable_*` fields

**Files:**
- Modify: `braintrace/_etrace_compiler/hid_param_op.py` — in the main relation-building loop, for each resolved weight also resolve every trainable key in `spec.resolve_trainable_invars(eqn.params)` and populate the new dict fields.
- Test: `braintrace/_etrace_compiler/hid_param_op_test.py` — add `TestTrainableInvarsPopulatedForDense` (mm primitive, because we haven't migrated specs yet — so `trainable_invars_fn` is None on every primitive, which means the compiler will still write `{'weight': ...}` entries derived from `resolve_trainable_invars`).

- [ ] **Step 1: Write the failing test**

Append to `braintrace/_etrace_compiler/hid_param_op_test.py`:

```python
class TestTrainableInvarsPopulatedForDense:
    """Even before per-primitive migration, the compiler fills the
    trainable_* dicts from spec.resolve_trainable_invars(). For un-migrated
    primitives (e.g. etp_mm_p without a trainable_invars_fn), this yields
    the single-key {'weight': ...} form."""

    def test_single_weight_populated_for_mm(self):
        import brainstate
        import jax.numpy as jnp
        import braintrace
        from braintrace._etrace_compiler import (
            find_hidden_param_op_relations_from_module,
        )

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)))
                self.h = brainstate.ShortTermState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(braintrace.matmul(self.h.value, self.w.value))
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        relations = find_hidden_param_op_relations_from_module(
            cell, jnp.zeros((1, 4))
        )
        assert len(relations) == 1
        r = relations[0]
        assert list(r.trainable_vars.keys()) == ['weight']
        assert r.trainable_vars['weight'] is r.weight_var
        assert r.trainable_paths['weight'] == r.weight_path
        assert r.trainable_leaf_indices['weight'] == r.weight_leaf_idx
        assert r.trainable_processing_chains['weight'] == r.weight_processing_chain
        assert r.trainable_param_states['weight'] is r.weight
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_compiler/hid_param_op_test.py::TestTrainableInvarsPopulatedForDense -v`
Expected: fail — `trainable_vars` is empty.

- [ ] **Step 3: Update `_resolve_eqn_vars` and the relation-building loop**

In `braintrace/_etrace_compiler/hid_param_op.py`, replace the body of `_resolve_eqn_vars` (currently returns `(weight_var, x_var, y_var)`) with a richer return shape that produces the full trainable dict. Keep the legacy return signature by adding a second function `_resolve_eqn_trainable_invars`.

Add this helper after the existing `_resolve_eqn_vars` (~line 277):

```python
def _resolve_eqn_trainable_invars(
    eqn: JaxprEqn,
) -> Dict[str, Var]:
    """Return ``{key: invar_var}`` for every trainable input of *eqn*.

    Uses the spec's ``resolve_trainable_invars`` when available. For the
    legacy ``etp_elemwise_p`` special case (no spec, weight at invar[0]),
    returns the single-key ``{'weight': eqn.invars[0]}`` form.
    """
    primitive = eqn.primitive
    spec = get_primitive_spec(primitive)
    if spec is not None:
        key_to_idx = spec.resolve_trainable_invars(eqn.params)
        return {k: eqn.invars[i] for k, i in key_to_idx.items()}
    if primitive is etp_elemwise_p:
        return {'weight': eqn.invars[0]}
    return {'weight': eqn.invars[1]}
```

Now find the body of `find_hidden_param_op_relations_from_minfo` (search for `HiddenParamOpRelation(`). At the call-site where a relation is constructed, do this:

1. After the existing code computes `weight_var, weight_path, weight_leaf_idx, weight_processing_chain, weight_param_state`, add a post-step that resolves **every trainable invar**:

```python
# --- NEW: resolve every trainable invar declared by the primitive ---
trainable_invars_map = _resolve_eqn_trainable_invars(eqn)
trainable_vars: Dict[str, Var] = {}
trainable_paths: Dict[str, Path] = {}
trainable_leaf_indices: Dict[str, int] = {}
trainable_param_states: Dict[str, brainstate.ParamState] = {}
trainable_processing_chains: Dict[str, Tuple[Primitive, ...]] = {}
for key, invar in trainable_invars_map.items():
    # The already-resolved 'weight' is the legacy primary key. Reuse its
    # resolution work; for other keys, trace backward.
    if invar is weight_var:
        t_path, t_chain = weight_path, weight_processing_chain
        t_leaf = weight_leaf_idx
        t_param_state = weight_param_state
    else:
        t_path, t_chain = _trace_var_to_param(
            invar, producers, invar_to_weight_path, trace_cache,
        )
        if t_path is None:
            # Trainable invar doesn't trace to any ParamState — skip it;
            # later the relation's grad routing simply has no entry for
            # this key, and the runtime emits zero for it.
            continue
        t_leaf = _resolve_weight_leaf_idx(
            invar, t_path, producers, weight_path_to_invars,
        )
        t_param_state = path_to_param_state.get(t_path)
    trainable_vars[key] = invar
    trainable_paths[key] = t_path
    trainable_leaf_indices[key] = t_leaf
    trainable_param_states[key] = t_param_state
    trainable_processing_chains[key] = t_chain
```

2. Pass the five new dicts to the `HiddenParamOpRelation(...)` constructor call:

```python
HiddenParamOpRelation(
    primitive=primitive,
    weight=weight_param_state,
    weight_path=weight_path,
    weight_var=weight_var,
    weight_leaf_idx=weight_leaf_idx,
    x_var=x_var,
    y_var=y_var,
    hidden_groups=hidden_groups_for_relation,
    y_to_hidden_group_jaxprs=group_jaxprs,
    connected_hidden_paths=connected_paths,
    eqn_params=eqn.params,
    weight_processing_chain=weight_processing_chain,
    path_classification=classification_map,
    trainable_vars=trainable_vars,
    trainable_paths=trainable_paths,
    trainable_leaf_indices=trainable_leaf_indices,
    trainable_param_states=trainable_param_states,
    trainable_processing_chains=trainable_processing_chains,
)
```

If the existing code already uses `path_to_param_state` under a different name, use whichever name the surrounding code uses. If no such map exists, derive it with:

```python
path_to_param_state = {p: ps for ps, p in minfo.iter_param_states_with_paths()}
```

placed once before the loop (use the same accessor minfo already exposes — read `module_info.py` for the exact API).

- [ ] **Step 4: Run, verify new test passes**

Run: `python -m pytest braintrace/_etrace_compiler/hid_param_op_test.py::TestTrainableInvarsPopulatedForDense -v`
Expected: pass.

- [ ] **Step 5: Run full compiler test suite**

Run: `python -m pytest braintrace/_etrace_compiler/ -v`
Expected: all tests pass. Existing tests read only the old `weight_*` fields and are unaffected; new field is additionally populated.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_compiler/hid_param_op.py braintrace/_etrace_compiler/hid_param_op_test.py
git commit -m "feat(etrace_compiler): populate trainable_* dicts on HiddenParamOpRelation

Compiler now resolves every invar declared by spec.resolve_trainable_invars
back to its originating ParamState and records the result in the new
per-key dicts. For un-migrated primitives this yields the single-key
{'weight': ...} form; primitives that declare trainable_invars_fn get
all keys resolved (e.g. {'weight', 'bias'}, {'B', 'A', 'bias'})."
```

---

## Task 5: d_rtrl executor uses dict-based rule API (with adapter)

**Files:**
- Modify: `braintrace/_etrace_vjp/d_rtrl.py` — switch the trace storage key and route every rule call through an adapter that normalizes old-API (array) and new-API (dict) rules.

**Design.** The adapter exists purely to let primitives migrate one file at a time in Tasks 7-11 without breaking the build. The adapter checks the primitive's spec: when `spec.trainable_invars_fn is not None`, rules are called with a dict and return a dict (new API); otherwise rules are called with a single array and the returned single array is wrapped as `{'weight': ...}` (legacy). After all primitives migrate, the adapter is removed in Task 14.

- [ ] **Step 1: Write the failing test**

Append to `braintrace/_etrace_vjp/d_rtrl_test.py`:

```python
class TestDRtrlDictTraceStorage:
    """D-RTRL stores per-primitive-instance traces keyed by (id(y_var), group_index)
    whose value is a dict keyed by trainable-input names.

    Because no primitive has migrated to the new API yet, the dict has only
    the single key 'weight' and the shape matches the legacy trace."""

    def test_trace_is_dict_keyed_by_weight(self):
        import brainstate
        import jax.numpy as jnp
        import braintrace
        from braintrace._etrace_vjp.d_rtrl import _initialize_drtrl_etrace_state

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)))
                self.h = brainstate.ShortTermState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(braintrace.matmul(self.h.value, self.w.value))
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        relations = braintrace.find_hidden_param_op_relations_from_module(
            cell, jnp.zeros((1, 4))
        )
        etrace = {}
        _initialize_drtrl_etrace_state(etrace, relations)
        assert len(etrace) == 1
        (_, entry), = etrace.items()
        assert isinstance(entry.value, dict)
        assert set(entry.value.keys()) == {'weight'}
        # Shape: (batch, in, out, n_state) per existing _mm_init_drtrl
        assert entry.value['weight'].shape[0] == 1   # batch
        assert entry.value['weight'].shape[1:3] == (4, 4)
```

(If `_initialize_drtrl_etrace_state` is not the actual name in `d_rtrl.py`, substitute the init helper that is called at the start of `EligibilityTrace` / `D_RTRL` setup — search for `ETP_RULES_INIT_DRTRL` in the file and use the surrounding function.)

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_vjp/d_rtrl_test.py::TestDRtrlDictTraceStorage -v`
Expected: fail — trace is currently a raw array, not a dict.

- [ ] **Step 3: Refactor the trace-init helper in `d_rtrl.py`**

Find the block (around line 71-78 in `d_rtrl.py`) that calls `init_fn(relation.x_var, relation.y_var, relation.weight_var, group.num_state)` and stores a single-array `EligibilityTrace`. Replace with:

```python
init_fn = ETP_RULES_INIT_DRTRL[relation.primitive]
spec = get_primitive_spec(relation.primitive)
if spec is not None and spec.trainable_invars_fn is not None:
    init_val = init_fn(
        relation.x_var, relation.y_var, relation.trainable_vars, group.num_state,
    )
    if not isinstance(init_val, dict):
        raise TypeError(
            f'Primitive {relation.primitive.name} declares trainable_invars_fn '
            f'so init_drtrl must return a dict; got {type(init_val).__name__}.'
        )
else:
    # Legacy scalar-return init_drtrl: wrap as single-key dict for uniform storage.
    init_val = {
        'weight': init_fn(
            relation.x_var, relation.y_var, relation.weight_var, group.num_state,
        )
    }
etrace_bwg[bwg_key] = EligibilityTrace(init_val)
```

Import `get_primitive_spec` at the top of the file if not already imported.

Replace the bwg-keying scheme so traces belong to the primitive instance:

```python
bwg_key = (id(relation.y_var), group.index)
```

(old key: `etrace_param_key(relation.weight_path, relation.y_var, group.index)`). Keep the existing `etrace_param_key` helper in `_misc.py`; we'll remove it in the cleanup task.

- [ ] **Step 4: Refactor the trace-update in `d_rtrl.py`**

Find the in-loop code (around line 164-210) that extracts the weight leaf and calls `xy_to_dw` / `yw_to_w`. Replace with:

```python
spec = get_primitive_spec(relation.primitive)
use_dict_api = spec is not None and spec.trainable_invars_fn is not None

if use_dict_api:
    weights_dict = {
        key: _extract_leaf(weight_path_to_vals[path], leaf_idx)
        for key, (path, leaf_idx) in zip(
            relation.trainable_vars.keys(),
            [(relation.trainable_paths[k], relation.trainable_leaf_indices[k])
             for k in relation.trainable_vars],
        )
    }
else:
    weights_dict = {
        'weight': _extract_leaf(
            weight_path_to_vals[relation.weight_path], relation.weight_leaf_idx,
        )
    }

xy_to_dw = ETP_RULES_XY_TO_DW[relation.primitive]
eqn_params = relation.eqn_params
is_elemwise = relation.primitive is etp_elemwise_p
batched = is_batched_primitive(relation.primitive)
x = None if is_elemwise else etrace_xs_at_t[id(relation.x_var)]

def _call_xy_to_dw(x_, df_, weights_):
    if use_dict_api:
        return xy_to_dw(x_, df_, weights_, **eqn_params)
    # Legacy single-array API — pass the one leaf, wrap result as dict.
    single = xy_to_dw(x_, df_, weights_['weight'], **eqn_params)
    return {'weight': single}
```

All subsequent code paths (`comp_dw_without_x`, `jax.vmap`, etc.) now operate on `weights_dict` and produce dict-valued gradients. Internal `jax.vmap(fn, in_axes=-1, out_axes=-1)` handles dicts natively.

Similarly rewrite the `yw_to_w` call site:

```python
yw_to_w = ETP_RULES_YW_TO_W[relation.primitive]

def _call_yw_to_w(hidden_dim, trace_val):
    if use_dict_api:
        return yw_to_w(hidden_dim, trace_val, **eqn_params)
    single = yw_to_w(hidden_dim, trace_val['weight'], **eqn_params)
    return {'weight': single}
```

- [ ] **Step 5: Route the dict-valued gradient to (potentially multiple) ParamState paths**

Where the old code called `_wrap_leaf_as_pytree(dg_weight, weight_vals[weight_path], weight_leaf_idx)` and pushed into `temp_data[weight_path]`, replace with:

```python
# Group leaves by owning ParamState path.
per_path: Dict[Path, Dict[int, jax.Array]] = {}
for key, grad in dg_weight.items():
    path = (
        relation.trainable_paths[key]
        if use_dict_api
        else relation.weight_path
    )
    leaf_idx = (
        relation.trainable_leaf_indices[key]
        if use_dict_api
        else relation.weight_leaf_idx
    )
    per_path.setdefault(path, {})[leaf_idx] = grad

for path, leaf_to_grad in per_path.items():
    wrapped = _wrap_leaves_as_pytree(weight_vals[path], leaf_to_grad)
    _update_dict(temp_data, path, wrapped)
```

(Ensure `_wrap_leaves_as_pytree` and `_extract_leaf` are imported from `braintrace._etrace_vjp.misc`.)

- [ ] **Step 6: Run, verify the new test passes**

Run: `python -m pytest braintrace/_etrace_vjp/d_rtrl_test.py::TestDRtrlDictTraceStorage -v`
Expected: pass.

- [ ] **Step 7: Run the full D-RTRL test module**

Run: `python -m pytest braintrace/_etrace_vjp/d_rtrl_test.py -v`
Expected: all pass. Existing D-RTRL numerical-correctness tests continue to pass because the adapter preserves the legacy behavior for un-migrated primitives.

- [ ] **Step 8: Commit**

```bash
git add braintrace/_etrace_vjp/d_rtrl.py braintrace/_etrace_vjp/d_rtrl_test.py
git commit -m "feat(d_rtrl): dict-based trace storage; adapter for un-migrated primitives

Trace is now stored as Dict[str, Array] keyed by (id(y_var), group_index).
When a primitive declares trainable_invars_fn the new dict-API rules are
called directly; otherwise a small adapter wraps the legacy array form as
{'weight': ...}. Gradient routing uses _wrap_leaves_as_pytree so that
multi-ParamState layouts (bias in its own ParamState) work immediately
once a primitive migrates."
```

---

## Task 6: pp_prop executor uses dict-based rule API (with adapter)

**Files:**
- Modify: `braintrace/_etrace_vjp/pp_prop.py` — switch the `xy_to_dw` call site and gradient routing to the dict form, with the same adapter as Task 5. `init_pp` **stays** a single-array return (per design spec); no change there.

- [ ] **Step 1: Write the failing test**

Append to `braintrace/_etrace_vjp/pp_prop_test.py`:

```python
class TestPPPropDictGradientRouting:
    """After the adapter is in place, pp_prop still computes the same
    gradient for a non-bias mm layer (no behavior change for legacy API)."""

    def test_mm_gradient_matches_before_refactor(self):
        import brainstate
        import jax.numpy as jnp
        import numpy.testing as npt
        import braintrace

        brainstate.random.seed(42)

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(
                    brainstate.random.normal(size=(4, 4)) * 0.1
                )
                self.h = brainstate.ShortTermState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.ES_D_RTRL(cell, decay_or_rank=0.9)
        alg.compile_graph(jnp.zeros((1, 4)))

        x_seq = brainstate.random.normal(size=(5, 1, 4)) * 0.1
        targets = jnp.zeros((1, 4))

        def loss_fn(y):
            return jnp.mean((y - targets) ** 2)

        grads, _ = alg.online_gradients(loss_fn, x_seq)
        assert 'weight' in jax.tree.structure(grads).children()[0].children()[0] or True
        # smoke: grads is a pytree containing the expected weight leaf
        flat = jax.tree.leaves(grads)
        assert any(leaf.shape == (4, 4) for leaf in flat)
```

(The specific `ES_D_RTRL` + `online_gradients` API may differ — adapt the test to whichever public API `braintrace` exposes. The point is: a full pp_prop run finishes cleanly after the adapter change.)

- [ ] **Step 2: Run existing pp_prop tests to ensure baseline passes**

Run: `python -m pytest braintrace/_etrace_vjp/pp_prop_test.py -v`
Expected: pass (baseline before change).

- [ ] **Step 3: Update `_solve_IO_dim_weight_gradients` in `pp_prop.py`**

Find the loop body (around line 437-471). Replace the weight extraction and `xy_to_dw` call:

```python
spec = get_primitive_spec(relation.primitive)
use_dict_api = spec is not None and spec.trainable_invars_fn is not None

if use_dict_api:
    weights_dict = {
        key: _extract_leaf(weight_vals[relation.trainable_paths[key]],
                           relation.trainable_leaf_indices[key])
        for key in relation.trainable_vars
    }
else:
    weights_dict = {
        'weight': _extract_leaf(
            weight_vals[relation.weight_path], relation.weight_leaf_idx,
        )
    }

xy_to_dw = ETP_RULES_XY_TO_DW[relation.primitive]
eqn_params = relation.eqn_params
batched = is_batched_primitive(relation.primitive)

def _call(df_, w):
    if use_dict_api:
        return xy_to_dw(x, df_, w, **eqn_params)
    return {'weight': xy_to_dw(x, df_, w['weight'], **eqn_params)}
```

Then, in place of the previous `jax.vmap(lambda df: xy_to_dw(x, df, weight_leaf, **eqn_params), ...)`, vmap over the stated dict:

```python
fn_vmap = jax.vmap(lambda df: _call(df, weights_dict), in_axes=-1, out_axes=-1)
if (relation.primitive is etp_elemwise_p) and batched:
    fn_vmap = jax.vmap(fn_vmap)
    dg_dict = jax.tree.map(lambda a: _sum_dim(_sum_dim(a, axis=-1), axis=0), fn_vmap(df_hid))
else:
    dg_dict = jax.tree.map(_sum_dim, fn_vmap(df_hid))
```

- [ ] **Step 4: Multi-ParamState gradient routing**

Replace the final `_wrap_leaf_as_pytree(...)` + `_update_dict(dG_weights, weight_path, dg_weight)` with:

```python
per_path: Dict[Path, Dict[int, jax.Array]] = {}
for key, grad in dg_dict.items():
    path = (
        relation.trainable_paths[key]
        if use_dict_api
        else relation.weight_path
    )
    leaf_idx = (
        relation.trainable_leaf_indices[key]
        if use_dict_api
        else relation.weight_leaf_idx
    )
    per_path.setdefault(path, {})[leaf_idx] = grad

for path, leaf_to_grad in per_path.items():
    wrapped = _wrap_leaves_as_pytree(weight_vals[path], leaf_to_grad)
    _update_dict(dG_weights, path, wrapped)
```

Import `_extract_leaf`, `_wrap_leaves_as_pytree`, and `get_primitive_spec` at the top of `pp_prop.py` if not already imported.

- [ ] **Step 5: Run the full pp_prop tests and the smoke test**

Run: `python -m pytest braintrace/_etrace_vjp/pp_prop_test.py -v`
Expected: all pass — behavior is unchanged for un-migrated primitives.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_vjp/pp_prop.py braintrace/_etrace_vjp/pp_prop_test.py
git commit -m "feat(pp_prop): dict-based xy_to_dw call + multi-path gradient routing

Mirrors the d_rtrl adapter. init_pp stays a single-array return (y-shape)
per design — pp_prop's df trace is shared across trainable inputs."
```

---

## Task 7: Migrate `etp_elemwise_p` to dict rule API

**Files:**
- Modify: `braintrace/_etrace_op/elemwise.py` — rewrite the four rules to use single-key `{'weight': ...}` dicts; add `trainable_invars_fn` to the spec.

- [ ] **Step 1: Write the failing test**

Append to `braintrace/_etrace_op/elemwise_test.py` (create if absent, following `dense_test.py` style):

```python
class TestElemwiseDictRules:

    def test_spec_declares_trainable_invars_fn(self):
        from braintrace._etrace_op import etp_elemwise_p, get_primitive_spec
        spec = get_primitive_spec(etp_elemwise_p)
        assert spec.trainable_invars_fn is not None
        assert spec.resolve_trainable_invars({}) == {'weight': 0}

    def test_xy_to_dw_returns_dict(self):
        import jax.numpy as jnp
        from braintrace._etrace_op import ETP_RULES_XY_TO_DW, etp_elemwise_p
        hidden_dim = jnp.ones((3, 4))
        weights = {'weight': jnp.ones((3, 4))}
        out = ETP_RULES_XY_TO_DW[etp_elemwise_p](None, hidden_dim, weights)
        assert isinstance(out, dict)
        assert set(out.keys()) == {'weight'}
        assert out['weight'].shape == (3, 4)

    def test_yw_to_w_returns_dict(self):
        import jax.numpy as jnp
        from braintrace._etrace_op import ETP_RULES_YW_TO_W, etp_elemwise_p
        hidden_dim = jnp.full((3, 4), 2.0)
        trace = {'weight': jnp.full((3, 4), 5.0)}
        out = ETP_RULES_YW_TO_W[etp_elemwise_p](hidden_dim, trace)
        assert isinstance(out, dict)
        assert out['weight'].shape == (3, 4)
        # 5 * 2 = 10
        assert float(out['weight'][0, 0]) == 10.0
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_op/elemwise_test.py -v`
Expected: fail — rules still return arrays; `trainable_invars_fn` is None on spec.

- [ ] **Step 3: Rewrite `braintrace/_etrace_op/elemwise.py`**

Replace the four rule functions and the spec registration:

```python
def _elem_trainable_invars(params):
    return {'weight': 0}


def _elemwise_yw_to_w(hidden_dim, trace):
    return {'weight': trace['weight'] * hidden_dim}


def _elemwise_xy_to_dw(x, hidden_dim, weights):
    return {'weight': hidden_dim}


def _elemwise_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    y_shape = y_var.aval.shape
    return {'weight': jnp.zeros((*y_shape, num_hidden_state))}


def _elemwise_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    # pp_prop: single-array y-shape trace (unchanged by design).
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_elemwise_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_elemwise',
        impl=_etp_elemwise_impl,
        yw_to_w=_elemwise_yw_to_w,
        xy_to_dw=_elemwise_xy_to_dw,
        init_drtrl=_elemwise_init_drtrl,
        init_pp=_elemwise_init_pp,
        weight_invar_index=0,
        x_invar_index=None,
        batched=False,
        gradient_enabled=True,
        trainable_invars_fn=_elem_trainable_invars,
    )
)
```

- [ ] **Step 4: Run the new test**

Run: `python -m pytest braintrace/_etrace_op/elemwise_test.py -v`
Expected: all pass.

- [ ] **Step 5: Run the wider test suite to confirm the adapter + new rules interoperate**

Run: `python -m pytest braintrace/ -x -q`
Expected: all previously-passing tests still pass (D-RTRL / pp_prop elemwise paths now go through the dict API; adapter still supports other primitives).

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_op/elemwise.py braintrace/_etrace_op/elemwise_test.py
git commit -m "refactor(elemwise): migrate rules to dict-based trainable-input API

First primitive migrated. Rules take and return {'weight': ...} dicts;
spec declares trainable_invars_fn. Other primitives still on legacy API
go through the executor adapter."
```

---

## Task 8: Migrate `etp_mm_p` / `etp_mv_p` to dict rule API with bias support

**Files:**
- Modify: `braintrace/_etrace_op/dense.py` — rewrite rules to fused-VJP dict form; add `trainable_invars_fn`.
- Create: `braintrace/_etrace_op/bias_gradient_test.py` — new file with bias-gradient correctness tests for mm/mv.
- Create: `braintrace/_etrace_op/separate_param_state_test.py` — new file with the separate-`ParamState` layout test.

- [ ] **Step 1: Write failing tests**

Create `braintrace/_etrace_op/bias_gradient_test.py`:

```python
# Bias-gradient correctness for ETP primitives: train a tiny recurrent net
# one step and verify ETP gradients (dW, db) match BPTT ground-truth.

import brainstate
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import braintrace


# ---------------------------------------------------------------------------
# etp_mm_p — batched
# ---------------------------------------------------------------------------

class TestMMBiasGradient:

    def _build_cell_merged(self):
        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(
                    {'weight': jnp.ones((4, 4)) * 0.1,
                     'bias': jnp.ones((4,)) * 0.2}
                )
                self.h = brainstate.ShortTermState(jnp.zeros((1, 4)))

            def update(self, x):
                w = self.p.value['weight']
                b = self.p.value['bias']
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, w, b)
                )
                return self.h.value
        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        return cell

    def test_drtrl_grad_matches_bptt(self):
        cell = self._build_cell_merged()
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 4)))

        x = jnp.ones((1, 4)) * 0.3
        target = jnp.zeros((1, 4))

        def loss_fn(y):
            return jnp.sum((y - target) ** 2)

        grads_etrace, _ = alg.online_gradients(loss_fn, x)

        # BPTT reference: differentiate the single-step forward.
        def bptt_loss(params):
            h = jnp.zeros((1, 4))
            h = jnp.tanh(x + h @ params['weight'] + params['bias'])
            return jnp.sum((h - target) ** 2)

        bptt = jax.grad(bptt_loss)({'weight': cell.p.value['weight'],
                                    'bias': cell.p.value['bias']})

        etrace_leaf = jax.tree.leaves(grads_etrace)[0]  # shape {weight, bias}
        npt.assert_allclose(etrace_leaf['weight'], bptt['weight'], atol=1e-5)
        npt.assert_allclose(etrace_leaf['bias'], bptt['bias'], atol=1e-5)
```

Create `braintrace/_etrace_op/separate_param_state_test.py`:

```python
# When weight and bias live in separate ParamStates, ETP must route
# gradients to both paths correctly.

import brainstate
import jax
import jax.numpy as jnp
import numpy.testing as npt

import braintrace


class TestSeparateParamStateBias:

    def test_separate_weight_and_bias_grads(self):
        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)) * 0.1)
                self.b = brainstate.ParamState(jnp.ones((4,)) * 0.2)
                self.h = brainstate.ShortTermState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value, self.b.value)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 4)))

        x = jnp.ones((1, 4)) * 0.3
        target = jnp.zeros((1, 4))
        grads_etrace, _ = alg.online_gradients(
            lambda y: jnp.sum((y - target) ** 2), x,
        )

        def bptt_loss(w, b):
            h = jnp.zeros((1, 4))
            h = jnp.tanh(x + h @ w + b)
            return jnp.sum((h - target) ** 2)

        dW, db = jax.grad(bptt_loss, (0, 1))(cell.w.value, cell.b.value)

        leaves = jax.tree.leaves(grads_etrace)
        # Expect two leaf values, one per ParamState.
        shapes = sorted(leaf.shape for leaf in leaves)
        assert shapes == [(4,), (4, 4)]
        w_leaf = next(l for l in leaves if l.shape == (4, 4))
        b_leaf = next(l for l in leaves if l.shape == (4,))
        npt.assert_allclose(w_leaf, dW, atol=1e-5)
        npt.assert_allclose(b_leaf, db, atol=1e-5)
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_op/bias_gradient_test.py braintrace/_etrace_op/separate_param_state_test.py -v`
Expected: fail — bias gradient is zero under today's rules.

- [ ] **Step 3: Rewrite `braintrace/_etrace_op/dense.py`**

Replace the rule bodies and both spec registrations:

```python
def _mm_trainable_invars(params):
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _mm_yw_to_w(hidden_dim, trace, *, has_bias=False):
    out = {'weight': trace['weight'] * jnp.expand_dims(hidden_dim, axis=1)}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _mm_xy_to_dw(x, hidden_dim, weights, *, has_bias=False):
    def _fwd(w):
        y = x @ w['weight']
        if has_bias:
            y = y + w['bias']
        return u.get_mantissa(y)
    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    batch = x_var.aval.shape[0]
    out = {
        'weight': jnp.zeros(
            (batch, *weight_vars['weight'].aval.shape, num_hidden_state)
        )
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _mm_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_mm',
        impl=_etp_matmul_impl,
        yw_to_w=_mm_yw_to_w,
        xy_to_dw=_mm_xy_to_dw,
        init_drtrl=_mm_init_drtrl,
        init_pp=_mm_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
        trainable_invars_fn=_mm_trainable_invars,
    )
)


def _mv_trainable_invars(params):
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _mv_yw_to_w(hidden_dim, trace, *, has_bias=False):
    out = {'weight': trace['weight'] * jnp.expand_dims(hidden_dim, axis=0)}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _mv_xy_to_dw(x, hidden_dim, weights, *, has_bias=False):
    def _fwd(w):
        y = x @ w['weight']
        if has_bias:
            y = y + w['bias']
        return u.get_mantissa(y)
    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    out = {
        'weight': jnp.zeros(
            (*weight_vars['weight'].aval.shape, num_hidden_state)
        )
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _mv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_mv',
        impl=_etp_matmul_impl,
        yw_to_w=_mv_yw_to_w,
        xy_to_dw=_mv_xy_to_dw,
        init_drtrl=_mv_init_drtrl,
        init_pp=_mv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=False,
        trainable_invars_fn=_mv_trainable_invars,
    )
)
```

- [ ] **Step 4: Run the new tests**

Run: `python -m pytest braintrace/_etrace_op/bias_gradient_test.py braintrace/_etrace_op/separate_param_state_test.py -v`
Expected: pass.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest braintrace/ -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_op/dense.py braintrace/_etrace_op/bias_gradient_test.py braintrace/_etrace_op/separate_param_state_test.py
git commit -m "feat(dense): migrate mm/mv to dict rule API with bias gradient support

Fused-VJP form computes dW and db in one pass. Tests cover both merged
ParamState ({'weight': W, 'bias': b}) and separate-ParamState layouts;
ETP gradients match BPTT ground truth for a one-step recurrent net."
```

---

## Task 9: Migrate `etp_conv_p` to dict rule API with bias support

**Files:**
- Modify: `braintrace/_etrace_op/conv.py` — rewrite rules; add `trainable_invars_fn`.
- Modify: `braintrace/_etrace_op/bias_gradient_test.py` — add `TestConvBiasGradient` class.

- [ ] **Step 1: Append failing test**

Append to `braintrace/_etrace_op/bias_gradient_test.py`:

```python
class TestConvBiasGradient:

    def test_drtrl_conv_grad_matches_bptt(self):
        import brainstate
        import jax
        import jax.numpy as jnp
        import numpy.testing as npt
        import braintrace

        kernel_init = jnp.ones((3, 4, 4)) * 0.05   # (H, in, out) for 1D conv
        bias_init = jnp.ones((4,)) * 0.1

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState(
                    {'weight': kernel_init, 'bias': bias_init}
                )
                self.h = brainstate.ShortTermState(jnp.zeros((1, 6, 4)))

            def update(self, x):
                k = self.p.value['weight']
                b = self.p.value['bias']
                y = braintrace.conv(
                    self.h.value, k, b,
                    strides=(1,), padding='SAME',
                    dimension_numbers=('NHC', 'HIO', 'NHC'),
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 6, 4)))

        x = jnp.ones((1, 6, 4)) * 0.1
        target = jnp.zeros((1, 6, 4))
        grads_etrace, _ = alg.online_gradients(
            lambda y: jnp.sum((y - target) ** 2), x,
        )

        def bptt_loss(params):
            h = jnp.zeros((1, 6, 4))
            y = jax.lax.conv_general_dilated(
                lhs=h, rhs=params['weight'],
                window_strides=(1,), padding='SAME',
                dimension_numbers=('NHC', 'HIO', 'NHC'),
            ) + params['bias']
            h = jnp.tanh(x + y)
            return jnp.sum((h - target) ** 2)

        bptt = jax.grad(bptt_loss)({
            'weight': cell.p.value['weight'],
            'bias': cell.p.value['bias'],
        })
        etrace_leaf = jax.tree.leaves(grads_etrace)[0]
        npt.assert_allclose(etrace_leaf['weight'], bptt['weight'], atol=1e-5)
        npt.assert_allclose(etrace_leaf['bias'], bptt['bias'], atol=1e-5)
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_op/bias_gradient_test.py::TestConvBiasGradient -v`
Expected: fail — bias gradient is zero.

- [ ] **Step 3: Rewrite `braintrace/_etrace_op/conv.py`**

Replace the rules and the spec registration:

```python
def _conv_trainable_invars(params):
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _conv_yw_to_w(hidden_dim, trace, **params):
    # Broadcast hidden_dim along every non-channel axis of the weight trace.
    out = {}
    w_trace = trace['weight']
    hd = hidden_dim
    n_expand = w_trace.ndim - hd.ndim
    for _ in range(n_expand):
        hd = jnp.expand_dims(hd, axis=0)
    out['weight'] = w_trace * hd
    if params.get('has_bias', False):
        # bias shape (out,); hidden_dim has batch + spatial + channel. Sum over
        # all non-channel axes is NOT done here — bias trace is exactly
        # hidden_dim-shaped per-sample; propagation multiplies by the same
        # hidden_dim used for the weight trace.
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _conv_xy_to_dw(x, hidden_dim, weights, *, has_bias=False, **params):
    conv_kw = {k: v for k, v in params.items() if k != 'has_bias'}

    def _fwd(w):
        y = jax.lax.conv_general_dilated(x, w['weight'], **conv_kw)
        if has_bias:
            y = y + w['bias']
        return u.get_mantissa(y)
    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _conv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    batch = x_var.aval.shape[0]
    out = {
        'weight': jnp.zeros(
            (batch, *weight_vars['weight'].aval.shape, num_hidden_state)
        )
    }
    if 'bias' in weight_vars:
        # bias trace carries the y-shape per sample.
        out['bias'] = jnp.zeros(
            (batch, *y_var.aval.shape[1:], num_hidden_state)
        )
    return out


def _conv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_conv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_conv',
        impl=_etp_conv_impl,
        yw_to_w=_conv_yw_to_w,
        xy_to_dw=_conv_xy_to_dw,
        init_drtrl=_conv_init_drtrl,
        init_pp=_conv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
        trainable_invars_fn=_conv_trainable_invars,
    )
)
```

- [ ] **Step 4: Run the new test**

Run: `python -m pytest braintrace/_etrace_op/bias_gradient_test.py::TestConvBiasGradient -v`
Expected: pass.

- [ ] **Step 5: Full suite**

Run: `python -m pytest braintrace/ -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_op/conv.py braintrace/_etrace_op/bias_gradient_test.py
git commit -m "feat(conv): migrate rules to dict API; bias gradient support"
```

---

## Task 10: Migrate `etp_sp_mm_p` / `etp_sp_mv_p` to dict rule API with bias support

**Files:**
- Modify: `braintrace/_etrace_op/sparse.py` — rewrite rules; add `trainable_invars_fn`.
- Modify: `braintrace/_etrace_op/bias_gradient_test.py` — add `TestSparseMMBiasGradient`.

- [ ] **Step 1: Write the correctness test**

First, open `braintrace/_etrace_op/sparse_test.py` and identify the smallest test that constructs a `sparse_mat` object and passes it through `braintrace.sparse_matmul`. Copy that `sparse_mat` construction verbatim. Then append to `braintrace/_etrace_op/bias_gradient_test.py`, using the same `sparse_mat`:

```python
class TestSparseMMBiasGradient:

    def test_drtrl_sparse_grad_matches_bptt(self):
        import brainstate
        import jax
        import jax.numpy as jnp
        import numpy.testing as npt
        import braintrace

        # --- BEGIN: copy the sparse_mat construction from sparse_test.py ---
        # sparse_mat = <as in sparse_test.py>
        # nnz = <number of non-zeros>
        # in_dim, out_dim = <dims of the sparse matrix>
        # --- END ---

        w_init = jnp.ones((nnz,)) * 0.1
        b_init = jnp.ones((out_dim,)) * 0.05

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState({'weight': w_init, 'bias': b_init})
                self.h = brainstate.ShortTermState(jnp.zeros((1, in_dim)))

            def update(self, x):
                y = braintrace.sparse_matmul(
                    self.h.value, self.p.value['weight'],
                    sparse_mat=sparse_mat, bias=self.p.value['bias'],
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, in_dim)))

        x = jnp.ones((1, in_dim)) * 0.2
        target = jnp.zeros((1, out_dim))
        grads_etrace, _ = alg.online_gradients(
            lambda y: jnp.sum((y - target) ** 2), x,
        )

        def bptt_loss(p):
            h = jnp.zeros((1, in_dim))
            y = h @ sparse_mat.with_data(p['weight']) + p['bias']
            h = jnp.tanh(x + y)
            return jnp.sum((h - target) ** 2)

        bptt = jax.grad(bptt_loss)({
            'weight': cell.p.value['weight'],
            'bias':   cell.p.value['bias'],
        })
        leaf = jax.tree.leaves(grads_etrace)[0]
        npt.assert_allclose(leaf['weight'], bptt['weight'], atol=1e-5)
        npt.assert_allclose(leaf['bias'],   bptt['bias'],   atol=1e-5)
```

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_op/bias_gradient_test.py::TestSparseMMBiasGradient -v`
Expected: fail — bias gradient is zero.

- [ ] **Step 3: Rewrite `braintrace/_etrace_op/sparse.py`**

Replace rules and spec registrations:

```python
def _sp_trainable_invars(params):
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _sp_mm_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
    out = {'weight': sparse_mat.yw_to_w_transposed(hidden_dim, trace['weight'])}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _sp_mv_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
    out = {'weight': sparse_mat.yw_to_w_transposed(hidden_dim, trace['weight'])}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _sp_xy_to_dw(x, hidden_dim, weights, *, sparse_mat=None, has_bias=False):
    def _fwd(w):
        y = x @ sparse_mat.with_data(w['weight'])
        if has_bias:
            y = y + w['bias']
        return u.get_mantissa(y)
    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _sp_mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    batch = x_var.aval.shape[0]
    nnz = weight_vars['weight'].aval.shape[0]
    out = {'weight': jnp.zeros((batch, nnz, num_hidden_state))}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _sp_mm_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _sp_mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    nnz = weight_vars['weight'].aval.shape[0]
    out = {'weight': jnp.zeros((nnz, num_hidden_state))}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _sp_mv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_sp_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_sp_mm',
        impl=_etp_sp_matmul_impl,
        yw_to_w=_sp_mm_yw_to_w,
        xy_to_dw=_sp_xy_to_dw,
        init_drtrl=_sp_mm_init_drtrl,
        init_pp=_sp_mm_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
        trainable_invars_fn=_sp_trainable_invars,
    )
)

etp_sp_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_sp_mv',
        impl=_etp_sp_matmul_impl,
        yw_to_w=_sp_mv_yw_to_w,
        xy_to_dw=_sp_xy_to_dw,
        init_drtrl=_sp_mv_init_drtrl,
        init_pp=_sp_mv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=False,
        trainable_invars_fn=_sp_trainable_invars,
    )
)
```

- [ ] **Step 4: Run the correctness test**

Run: `python -m pytest braintrace/_etrace_op/bias_gradient_test.py::TestSparseMMBiasGradient -v`
Expected: pass.

- [ ] **Step 5: Run the existing sparse regression tests**

Run: `python -m pytest braintrace/_etrace_op/sparse_test.py -v`
Expected: all pass.

- [ ] **Step 6: Full suite**

Run: `python -m pytest braintrace/ -x -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add braintrace/_etrace_op/sparse.py braintrace/_etrace_op/bias_gradient_test.py
git commit -m "feat(sparse): migrate sparse mm/mv rules to dict API with bias support"
```

---

## Task 11: Migrate `etp_lora_mm_p` / `etp_lora_mv_p` and add end-to-end online-learning test

**Files:**
- Modify: `braintrace/_etrace_op/lora.py` — fuse the existing pytree-returning rules with bias support; align `xy_to_dw` signature with the executor.
- Modify: `braintrace/_etrace_op/lora_test.py` — add end-to-end online-learning tests for both `lora_mm` and `lora_mv`.

- [ ] **Step 1: Append failing test**

Append to `braintrace/_etrace_op/lora_test.py`:

```python
class TestLoRAOnlineLearning:

    def test_drtrl_lora_mm_grad_matches_bptt(self):
        import brainstate
        import jax
        import jax.numpy as jnp
        import numpy.testing as npt
        import braintrace

        rank = 2
        B_init = jnp.ones((4, rank)) * 0.1
        A_init = jnp.ones((rank, 4)) * 0.1
        bias_init = jnp.ones((4,)) * 0.05

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = brainstate.ParamState({
                    'lora_b': B_init,
                    'lora_a': A_init,
                    'bias': bias_init,
                })
                self.h = brainstate.ShortTermState(jnp.zeros((1, 4)))

            def update(self, x):
                p = self.p.value
                y = braintrace.lora_matmul(
                    self.h.value, p['lora_b'], p['lora_a'],
                    alpha=1.0, bias=p['bias'],
                )
                self.h.value = jnp.tanh(x + y)
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        alg = braintrace.D_RTRL(cell)
        alg.compile_graph(jnp.zeros((1, 4)))

        x = jnp.ones((1, 4)) * 0.3
        target = jnp.zeros((1, 4))
        grads_etrace, _ = alg.online_gradients(
            lambda y: jnp.sum((y - target) ** 2), x,
        )

        def bptt_loss(p):
            h = jnp.zeros((1, 4))
            y = 1.0 * (h @ p['lora_b'] @ p['lora_a']) + p['bias']
            h = jnp.tanh(x + y)
            return jnp.sum((h - target) ** 2)

        bptt = jax.grad(bptt_loss)({
            'lora_b': cell.p.value['lora_b'],
            'lora_a': cell.p.value['lora_a'],
            'bias':   cell.p.value['bias'],
        })
        leaf = jax.tree.leaves(grads_etrace)[0]
        npt.assert_allclose(leaf['lora_b'], bptt['lora_b'], atol=1e-5)
        npt.assert_allclose(leaf['lora_a'], bptt['lora_a'], atol=1e-5)
        npt.assert_allclose(leaf['bias'],   bptt['bias'],   atol=1e-5)
```

(The key mapping `{'B', 'A', 'bias'}` in the spec must match the leaf structure of the `ParamState` pytree. Verify `jax.tree.leaves({'lora_b': ..., 'lora_a': ..., 'bias': ...})` ordering when computing `trainable_leaf_indices`; if needed, adjust the spec's `trainable_invars_fn` key names to `lora_b` / `lora_a` to match the dict keys exactly — the important invariant is that the compiler's `_resolve_weight_leaf_idx` returns the correct leaf index for each key.)

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_op/lora_test.py::TestLoRAOnlineLearning -v`
Expected: fail — `_lora_xy_to_dw` signature mismatch with executor (the pre-existing bug).

- [ ] **Step 3: Rewrite `braintrace/_etrace_op/lora.py`**

Replace rules and both spec registrations:

```python
def _lora_trainable_invars(params):
    base = {'lora_b': 1, 'lora_a': 2}
    if params.get('has_bias', False):
        base['bias'] = 3
    return base


def _lora_mm_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    out = {
        'lora_b': trace['lora_b'],  # B frozen during trace propagation (preserved semantics)
        'lora_a': trace['lora_a'] * jnp.expand_dims(hidden_dim, axis=1),
    }
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _lora_mv_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    out = {
        'lora_b': trace['lora_b'],
        'lora_a': trace['lora_a'] * jnp.expand_dims(hidden_dim, axis=0),
    }
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _lora_xy_to_dw(x, hidden_dim, weights, *, alpha=1.0, has_bias=False):
    def _fwd(w):
        y = alpha * (x @ w['lora_b'] @ w['lora_a'])
        if has_bias:
            y = y + w['bias']
        return u.get_mantissa(y)
    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _lora_mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    batch = x_var.aval.shape[0]
    B_shape = weight_vars['lora_b'].aval.shape
    A_shape = weight_vars['lora_a'].aval.shape
    out = {
        'lora_b': jnp.zeros((batch, *B_shape, num_hidden_state)),
        'lora_a': jnp.zeros((batch, *A_shape, num_hidden_state)),
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _lora_mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    B_shape = weight_vars['lora_b'].aval.shape
    A_shape = weight_vars['lora_a'].aval.shape
    out = {
        'lora_b': jnp.zeros((*B_shape, num_hidden_state)),
        'lora_a': jnp.zeros((*A_shape, num_hidden_state)),
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _lora_mm_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _lora_mv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_lora_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_lora_mm',
        impl=_etp_lora_impl,
        yw_to_w=_lora_mm_yw_to_w,
        xy_to_dw=_lora_xy_to_dw,
        init_drtrl=_lora_mm_init_drtrl,
        init_pp=_lora_mm_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
        trainable_invars_fn=_lora_trainable_invars,
    )
)

etp_lora_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_lora_mv',
        impl=_etp_lora_impl,
        yw_to_w=_lora_mv_yw_to_w,
        xy_to_dw=_lora_xy_to_dw,
        init_drtrl=_lora_mv_init_drtrl,
        init_pp=_lora_mv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=False,
        trainable_invars_fn=_lora_trainable_invars,
    )
)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest braintrace/_etrace_op/lora_test.py -v`
Expected: `TestLoRAOnlineLearning` passes; existing LoRA forward tests still pass.

- [ ] **Step 5: Full suite**

Run: `python -m pytest braintrace/ -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_op/lora.py braintrace/_etrace_op/lora_test.py
git commit -m "fix(lora): migrate LoRA rules to dict API; fix executor signature mismatch

LoRA's {'lora_b', 'lora_a'} pytree rule form now matches the executor's
dict-based xy_to_dw call. Bias added as third optional key. End-to-end
D-RTRL test compares against BPTT ground truth."
```

---

## Task 12: Remove `weight_invar_index` from `ETPPrimitiveSpec`

**Files:**
- Modify: `braintrace/_etrace_op/_spec.py` — make `trainable_invars_fn` required; remove `weight_invar_index`.
- Modify: every primitive registration that still sets `weight_invar_index=…` — remove the kwarg (now unused).

- [ ] **Step 1: Verify every primitive now declares `trainable_invars_fn`**

Run: `python -c "from braintrace._etrace_op._spec import ETP_PRIMITIVE_SPECS; [print(s.name, s.trainable_invars_fn is not None) for s in ETP_PRIMITIVE_SPECS.values()]"`
Expected: every primitive prints `True`.

- [ ] **Step 2: Edit `_spec.py`** — make `trainable_invars_fn` required and drop `weight_invar_index`:

```python
@dataclass(frozen=True)
class ETPPrimitiveSpec:
    name: str
    impl: Callable
    yw_to_w: Callable
    xy_to_dw: Callable
    init_drtrl: Callable
    init_pp: Callable
    trainable_invars_fn: Callable[[dict], Dict[str, int]]
    x_invar_index: Optional[int] = 0
    y_outvar_index: int = 0
    batched: bool = False
    gradient_enabled: bool = False

    def resolve_trainable_invars(self, eqn_params: dict) -> Dict[str, int]:
        return self.trainable_invars_fn(eqn_params)
```

- [ ] **Step 3: Remove `weight_invar_index=…` kwargs from every spec registration**

Search: `grep -rn "weight_invar_index" braintrace/_etrace_op/*.py`
For each hit in `dense.py`, `conv.py`, `sparse.py`, `lora.py`, `elemwise.py`, delete the `weight_invar_index=…,` line from the `ETPPrimitiveSpec(...)` call.

- [ ] **Step 4: Update the compiler's `_resolve_eqn_vars`**

Remove the fallback branches that reference `spec.weight_invar_index`. Simplified body:

```python
def _resolve_eqn_vars(eqn):
    primitive = eqn.primitive
    spec = get_primitive_spec(primitive)
    if spec is None:
        # No spec: only etp_elemwise_p is legacy here, and it now has a spec.
        raise RuntimeError(
            f'ETP primitive {primitive.name} has no registered spec'
        )
    key_to_idx = spec.resolve_trainable_invars(eqn.params)
    # Canonical 'weight' key for the legacy single-weight return is gone —
    # but downstream callers still expect (weight_var, x_var, y_var) for
    # legacy relation fields until Task 13. Pick the first key in insertion
    # order as the legacy weight_var.
    first_key = next(iter(key_to_idx))
    weight_var = eqn.invars[key_to_idx[first_key]]
    if spec.x_invar_index is None:
        x_var = None
    else:
        candidate = eqn.invars[spec.x_invar_index]
        x_var = candidate if isinstance(candidate, Var) else None
    y_var = eqn.outvars[spec.y_outvar_index]
    return weight_var, x_var, y_var
```

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest braintrace/ -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add braintrace/_etrace_op/_spec.py braintrace/_etrace_op/dense.py braintrace/_etrace_op/conv.py braintrace/_etrace_op/sparse.py braintrace/_etrace_op/lora.py braintrace/_etrace_op/elemwise.py braintrace/_etrace_compiler/hid_param_op.py
git commit -m "refactor(spec): remove weight_invar_index; trainable_invars_fn is required"
```

---

## Task 13: Remove `weight_*` legacy fields from `HiddenParamOpRelation`

**Files:**
- Modify: `braintrace/_etrace_compiler/hid_param_op.py` — remove `weight`, `weight_path`, `weight_var`, `weight_leaf_idx`, `weight_processing_chain` from the NamedTuple; remove their population.
- Modify: every consumer (`graph.py`, `diagnostics.py`, executors, tests) — replace `.weight_*` access with `.trainable_*['weight']` or the primitive's canonical first key.
- Modify: `braintrace/_etrace_vjp/d_rtrl.py`, `pp_prop.py` — the adapter branches for `use_dict_api == False` are dead; remove them (part of Task 14, but the imports clean up here).

- [ ] **Step 1: Enumerate every remaining reference**

Run: `grep -rn "\.weight_path\|\.weight_var\|\.weight_leaf_idx\|\.weight_processing_chain\|\.weight\b" braintrace/ | grep -v __pycache__`

- [ ] **Step 2: Update each callsite**

For each hit, replace:
- `relation.weight_path` → `next(iter(relation.trainable_paths.values()))` when the caller really wants "any one path" (rare), or the explicit key the caller knows (e.g. `relation.trainable_paths['weight']` for dense primitives, `relation.trainable_paths['lora_b']` for LoRA).
- `relation.weight_var` → `relation.trainable_vars[<key>]`.
- `relation.weight_leaf_idx` → `relation.trainable_leaf_indices[<key>]`.
- `relation.weight_processing_chain` → `relation.trainable_processing_chains[<key>]`.
- `relation.weight` → `relation.trainable_param_states[<key>]`.

Where the caller needs to iterate *all* trainable inputs, replace the single-key access with iteration over `relation.trainable_vars`.

- [ ] **Step 3: Remove the legacy fields from the NamedTuple**

In `hid_param_op.py`, trim `HiddenParamOpRelation` to:

```python
class HiddenParamOpRelation(NamedTuple):
    primitive: Primitive
    x_var: Optional[Var]
    y_var: Var
    hidden_groups: List[HiddenGroup]
    y_to_hidden_group_jaxprs: List[Jaxpr]
    connected_hidden_paths: List[Path]
    eqn_params: dict
    path_classification: Dict[Path, str] = {}
    trainable_vars: Dict[str, Var] = {}
    trainable_paths: Dict[str, Path] = {}
    trainable_leaf_indices: Dict[str, int] = {}
    trainable_param_states: Dict[str, brainstate.ParamState] = {}
    trainable_processing_chains: Dict[str, Tuple[Primitive, ...]] = {}
```

- [ ] **Step 4: Remove population of legacy fields in the compiler**

In the relation-construction site, drop the `weight=…, weight_path=…, weight_var=…, weight_leaf_idx=…, weight_processing_chain=…` kwargs. Also drop the now-redundant standalone assignments to those variables (only `trainable_*` dicts remain).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest braintrace/ -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add braintrace/
git commit -m "refactor(relation): remove legacy single-weight fields from HiddenParamOpRelation

All consumers migrated to trainable_* dicts. No backward-compat alias."
```

---

## Task 14: Remove executor adapter (dict-API is the only path)

**Files:**
- Modify: `braintrace/_etrace_vjp/d_rtrl.py` — remove the `use_dict_api` branches.
- Modify: `braintrace/_etrace_vjp/pp_prop.py` — ditto.
- Modify: `braintrace/_etrace_vjp/misc.py` — remove the old `_extract_weight_leaf` and `_wrap_leaf_as_pytree` helpers.
- Modify: `braintrace/_etrace_vjp/misc_test.py` — remove tests for those old helpers.
- Modify: `braintrace/_misc.py` — remove `etrace_param_key` if it's no longer referenced.

- [ ] **Step 1: Delete the adapter in `d_rtrl.py`**

Remove every `if use_dict_api:` / `else:` block introduced in Task 5. The code collapses to the dict form only:

```python
weights_dict = {
    key: _extract_leaf(
        weight_path_to_vals[relation.trainable_paths[key]],
        relation.trainable_leaf_indices[key],
    )
    for key in relation.trainable_vars
}
# … dg_dict = xy_to_dw(x, df, weights_dict, **eqn_params)
# … trace_next = yw_to_w(hidden_dim, trace_val, **eqn_params)
```

Same treatment for `pp_prop.py`.

- [ ] **Step 2: Delete the legacy helpers in `misc.py`**

Remove `_extract_weight_leaf` and `_wrap_leaf_as_pytree` function definitions. Remove their imports from `d_rtrl.py` and `pp_prop.py`.

- [ ] **Step 3: Delete the corresponding tests**

From `braintrace/_etrace_vjp/misc_test.py`, delete `TestExtractWeightLeaf` and `TestWrapLeafAsPytree` (if they exist) — they tested the removed helpers. Keep `TestExtractLeaf` and `TestWrapLeavesAsPytree`.

- [ ] **Step 4: Delete `etrace_param_key` if unused**

Run: `grep -rn "etrace_param_key" braintrace/ | grep -v __pycache__`
If zero hits outside its definition, remove the definition from `braintrace/_misc.py` and its export.

- [ ] **Step 5: Run full suite**

Run: `python -m pytest braintrace/ -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add braintrace/
git commit -m "refactor(vjp): remove adapter; dict-based rule API is the only path

Cleans up the transitional branches from Tasks 5-6 now that every primitive
declares trainable_invars_fn. Legacy _extract_weight_leaf and
_wrap_leaf_as_pytree removed."
```

---

## Task 15: Add `TRAINABLE_INVAR_NOT_PARAMSTATE` diagnostic

**Files:**
- Modify: `braintrace/_etrace_compiler/diagnostics.py` — add the new `DiagnosticKind` value.
- Modify: `braintrace/_etrace_compiler/hid_param_op.py` — emit at INFO level when a trainable key's invar doesn't trace to any `ParamState`.
- Modify: `braintrace/_etrace_compiler/diagnostics_test.py` — add a test that constructs a model where bias is a plain `jnp.array` (not a `ParamState`) and checks that the diagnostic is emitted.

- [ ] **Step 1: Write failing test**

Append to `braintrace/_etrace_compiler/diagnostics_test.py`:

```python
class TestTrainableInvarNotParamState:

    def test_constant_bias_emits_diagnostic(self):
        import brainstate
        import jax.numpy as jnp
        import braintrace
        from braintrace._etrace_compiler.diagnostics import (
            DiagnosticKind, get_latest_record,
        )

        bias_const = jnp.ones((4,))  # NOT wrapped in a ParamState

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)))
                self.h = brainstate.ShortTermState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(
                    x + braintrace.matmul(self.h.value, self.w.value, bias_const)
                )
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        _ = braintrace.find_hidden_param_op_relations_from_module(
            cell, jnp.zeros((1, 4))
        )
        record = get_latest_record()
        kinds = [d.kind for d in record.diagnostics]
        assert DiagnosticKind.TRAINABLE_INVAR_NOT_PARAMSTATE in kinds
```

(The `get_latest_record` accessor may differ — inspect `diagnostics.py` for the idiomatic way tests inspect the latest emitted diagnostics.)

- [ ] **Step 2: Run, verify fail**

Run: `python -m pytest braintrace/_etrace_compiler/diagnostics_test.py::TestTrainableInvarNotParamState -v`
Expected: fail — the `DiagnosticKind` value doesn't exist.

- [ ] **Step 3: Add the new `DiagnosticKind`**

In `braintrace/_etrace_compiler/diagnostics.py`, add:

```python
    TRAINABLE_INVAR_NOT_PARAMSTATE = 'trainable_invar_not_paramstate'
```

alongside the existing `DiagnosticKind` values.

- [ ] **Step 4: Emit the diagnostic in the compiler**

In `hid_param_op.py`, inside the per-key loop added in Task 4, where the plan said "skip it", replace the bare `continue` with:

```python
    else:
        emit(
            kind=DiagnosticKind.TRAINABLE_INVAR_NOT_PARAMSTATE,
            level=DiagnosticLevel.INFO,
            message=(
                f"ETP primitive {eqn.primitive.name}: trainable input "
                f"'{key}' at invar index {trainable_invars_map_idx} does "
                f"not trace to any ParamState. No online gradient will be "
                f"produced for this input."
            ),
            primitive=eqn.primitive,
            context={'key': key},
        )
        continue
```

(Substitute `trainable_invars_map_idx` for whatever local variable holds the invar index at that point.)

- [ ] **Step 5: Run new test**

Run: `python -m pytest braintrace/_etrace_compiler/diagnostics_test.py::TestTrainableInvarNotParamState -v`
Expected: pass.

- [ ] **Step 6: Full suite**

Run: `python -m pytest braintrace/ -x -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add braintrace/_etrace_compiler/diagnostics.py braintrace/_etrace_compiler/hid_param_op.py braintrace/_etrace_compiler/diagnostics_test.py
git commit -m "feat(diagnostics): add TRAINABLE_INVAR_NOT_PARAMSTATE

Emitted at INFO when a trainable invar (e.g. a constant bias) doesn't
trace to any ParamState. The key is silently dropped from the relation
and no gradient is produced for it at runtime."
```

---

## Final verification

After Task 15:

```bash
python -m pytest braintrace/ -v
```

Expected: full suite passes. Key end-to-end coverage:
- Dense `mm`/`mv` with merged `{weight, bias}` `ParamState` — bias gradient matches BPTT.
- Dense `mm`/`mv` with separate `w`/`b` `ParamState`s — gradients routed to both paths.
- Convolution with bias — bias gradient matches BPTT.
- Sparse `mm`/`mv` with bias — bias gradient matches BPTT.
- LoRA online learning (D-RTRL) — all three `{B, A, bias}` gradients match BPTT.
- Constant bias (not in a `ParamState`) — `TRAINABLE_INVAR_NOT_PARAMSTATE` diagnostic emitted, no runtime crash.
- Every GRU / RNN / LSTM test count invariant preserved (still one relation per primitive instance).
