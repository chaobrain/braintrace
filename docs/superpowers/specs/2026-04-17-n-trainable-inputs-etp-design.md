# Design: N-trainable-input support for ETP primitives

**Date:** 2026-04-17
**Status:** Approved for implementation planning

## Problem

ETP primitives currently assume each primitive has exactly one trainable input ("the weight"). This assumption leaks into the spec (`weight_invar_index: int`), the relation (`weight_var`, `weight_leaf_idx`, `weight_path`), and the executor (`_extract_weight_leaf` returns one leaf; `_wrap_leaf_as_pytree` zero-fills the rest).

The assumption has two concrete failures:

1. **Bias is silently dropped.** For a `ParamState({'weight': W, 'bias': b})` (e.g. `braintrace.nn.Linear`), `W` learns via ETP online learning but `b` receives a zero gradient every step. The forward pass correctly adds bias (`_etp_matmul_impl` with `has_bias=True`), but none of the four ETP rules (`yw_to_w`, `xy_to_dw`, `init_drtrl`, `init_pp`) produce bias gradients, and `_wrap_leaf_as_pytree` in `_etrace_vjp/misc.py` zero-fills the bias slot with an explicit comment that bias gradients "are simply absent." With `b_init=ZeroInit()` as the default, this failure is silent.

2. **LoRA is already broken.** `_lora_xy_to_dw(x, hidden_dim, w_B, w_A, ...)` takes four positional args and returns a `{'B': ..., 'A': ...}` dict, but the executor calls every `xy_to_dw` as `xy_to_dw(x, df, single_weight_leaf, **params)` — a single positional weight. LoRA's online-learning path is inconsistent and currently untested end-to-end.

The real issue is not bias per se. A `ParamState` can be an arbitrary pytree and a primitive can consume any number of its leaves. Bias is one instance of this pattern; LoRA's `{B, A}` is another; third-party primitives may need more.

## Goal

Refactor ETP so that a primitive has **a named dict of trainable inputs**, with keys chosen by the primitive author. All ETP rules, the compiler relation, and the executor operate on pytrees keyed by those names. Bias, LoRA's (B, A), and future multi-array trainable structures all fall out of the same general mechanism.

Forward-pass semantics, public user API (`braintrace.matmul`, `braintrace.lora_matmul`, `braintrace.conv`, etc.), hidden-group discovery, and algorithm selection are unchanged.

## Core abstraction

Each ETP primitive declares a function from its static equation parameters to an ordered dict `{key_name: invar_index}`. Keys are per-primitive and chosen by the primitive author. Two keys in the same primitive may point to the same `ParamState` (e.g. `{weight, bias}` merged Linear; `{B, A}` LoRA) or to different `ParamState`s (separate weight and bias objects). The compiler resolves this at compile time.

Key inventory across current primitives:

| Primitive                       | Keys (no bias)   | Keys (with bias)         |
|---------------------------------|------------------|--------------------------|
| `etp_mm_p` / `etp_mv_p`         | `{'weight'}`     | `{'weight', 'bias'}`     |
| `etp_conv_p`                    | `{'weight'}`     | `{'weight', 'bias'}`     |
| `etp_sp_mm_p` / `etp_sp_mv_p`   | `{'weight'}`     | `{'weight', 'bias'}`     |
| `etp_lora_mm_p` / `etp_lora_mv_p` | `{'B', 'A'}`   | `{'B', 'A', 'bias'}`     |
| `etp_elemwise_p`                | `{'weight'}`     | n/a                      |

## Spec changes

`ETPPrimitiveSpec` (in `braintrace/_etrace_op/_spec.py`) replaces `weight_invar_index` with a function-valued field:

```python
@dataclass(frozen=True)
class ETPPrimitiveSpec:
    name: str
    impl: Callable
    yw_to_w: Callable
    xy_to_dw: Callable
    init_drtrl: Callable
    init_pp: Callable

    # maps eqn.params -> {key_name: invar_index}
    trainable_invars_fn: Callable[[dict], Dict[str, int]]

    x_invar_index: Optional[int] = 0
    y_outvar_index: int = 0
    batched: bool = False
    gradient_enabled: bool = False
```

`weight_invar_index` is removed. No framework-level "bias" or "weight" concept — keys are strings chosen by the primitive. The function form lets each primitive decide which invars are trainable from its own static params (e.g. `has_bias`, or user flags in a third-party primitive).

## Rule signatures

All four rule types operate on pytrees keyed by the primitive's trainable-input names (same keys `trainable_invars_fn` returns for the given `eqn.params`).

```python
# propagate existing trace through hidden-Jacobian diagonal
yw_to_w(hidden_dim, trace: Dict[str, Array], **eqn_params) -> Dict[str, Array]

# compute instantaneous gradient w.r.t. each trainable input
xy_to_dw(x, hidden_dim, weights: Dict[str, Array], **eqn_params) -> Dict[str, Array]

# D-RTRL init: per-parameter trace, one Array per trainable key
init_drtrl(x_var, y_var, weight_vars: Dict[str, Var], num_hidden_state) -> Dict[str, Array]

# pp_prop init: y-side df trace — shape determined by y_var only. Single Array.
init_pp(x_var, y_var, weight_vars: Dict[str, Var], num_hidden_state) -> Array
```

**Rule asymmetry — why `init_pp` returns an Array, not a dict.** `init_drtrl` stores a per-parameter trace (shape fans out per trainable input). `init_pp` stores a per-output df trace — shape depends only on `y_var` and is shared across all trainable inputs at gradient-compute time (each input's Jacobian multiplies the same df).

Every existing rule in `_etrace_op/` is rewritten to the new signature as part of the same change. No backward compatibility layer.

## Compiler changes

### `HiddenParamOpRelation` — new layout

The old single-weight fields are removed; no backward-compat properties.

```python
class HiddenParamOpRelation(NamedTuple):
    primitive: Primitive

    trainable_vars: Dict[str, Var]              # key -> jaxpr Var at that invar
    trainable_paths: Dict[str, Path]            # key -> owning ParamState's module path
    trainable_leaf_indices: Dict[str, int]      # key -> jax.tree.leaves() index in that ParamState
    trainable_param_states: Dict[str, brainstate.ParamState]
    trainable_processing_chains: Dict[str, Tuple[Primitive, ...]]
    # per-key: primitives traversed backward from the invar to the ParamState (e.g. mask*W)

    x_var: Optional[Var]
    y_var: Var
    hidden_groups: List[HiddenGroup]
    y_to_hidden_group_jaxprs: List[Jaxpr]
    connected_hidden_paths: List[Path]
    eqn_params: dict
    path_classification: Dict[Path, str] = {}
```

### Compiler pass (`hid_param_op.py`)

For each ETP-primitive equation found in the jaxpr:

1. Look up the spec via `get_primitive_spec(eqn.primitive)`.
2. Call `spec.trainable_invars_fn(eqn.params)` to get `{key: invar_index}`.
3. For each `(key, idx)`: trace `eqn.invars[idx]` backward through the jaxpr until it resolves to a `ParamState` invar. Reuses the existing backward-tracing machinery.
4. Record `(param_state, path, leaf_idx, processing_chain)` per key in the dicts above.
5. If a trainable invar doesn't trace to any `ParamState` (e.g. a constant bias passed as a plain array), omit that key from `trainable_*` dicts and emit `TRAINABLE_INVAR_NOT_PARAMSTATE` at INFO level. Forward still runs; that input simply isn't registered for online learning.
6. If **all** trainable keys fail to resolve to a `ParamState`, no relation is built for that equation (matches today's behavior when `weight_var` has no backing `ParamState`).
7. Build exactly one `HiddenParamOpRelation` per primitive instance regardless of how many `ParamState`s the surviving trainable keys span. Same-`ParamState` and separate-`ParamState` layouts differ only in whether entries of `trainable_paths` share a value.
8. Key order is preserved from `trainable_invars_fn(eqn.params)`'s insertion order throughout the relation and the rule calls.

The existing `weight → weight → hidden` exclusion (`RELATION_EXCLUDED_WEIGHT_TO_WEIGHT`) now runs per-key — a key whose only path to `h` crosses another non-gradient-enabled ETP primitive is dropped from the relation (or excludes the whole relation if every key is blocked, matching today's behavior for the single-weight case).

## Executor changes (`_etrace_vjp/d_rtrl.py`, `pp_prop.py`)

### Trace storage keys

Trace is per-primitive-instance, not per-`ParamState`. Today `etrace_bwg` is keyed by `(weight_path, y_var, group.index)`, which conflates primitive identity with `ParamState` identity. New key: `(id(relation.y_var), group.index)`. Value: `Dict[str, Array]` matching the relation's trainable keys.

`pp_prop`'s `etrace_dfs` is already keyed by `(y_var, group.index)` with a single-array value — unchanged. `etrace_xs` is keyed by `id(x_var)` — unchanged.

### Extracting weights for rule calls

```python
weights_dict = {
    key: _extract_leaf(weight_path_to_vals[path], leaf_idx)
    for key, path, leaf_idx in zip(
        relation.trainable_vars.keys(),
        relation.trainable_paths.values(),
        relation.trainable_leaf_indices.values(),
    )
}

grads_dict = xy_to_dw(x, df, weights_dict, **eqn_params)   # Dict[str, Array]
trace_next = yw_to_w(hidden_dim, trace_dict, **eqn_params) # Dict[str, Array]
```

`jax.vmap` over the state axis handles pytree values natively — no extra wiring.

### Gradient routing across `ParamState`s

After computing `grads_dict`, split by owning `ParamState` path:

```python
per_path: Dict[Path, Dict[int, Array]] = defaultdict(dict)
for key, grad in grads_dict.items():
    path = relation.trainable_paths[key]
    leaf_idx = relation.trainable_leaf_indices[key]
    per_path[path][leaf_idx] = grad

for path, leaf_to_grad in per_path.items():
    wrapped = _wrap_leaves_as_pytree(weight_path_to_vals[path], leaf_to_grad)
    _update_dict(dG_weights, path, wrapped)
```

`_wrap_leaves_as_pytree(reference, {leaf_idx: grad, ...})` generalizes today's `_wrap_leaf_as_pytree`: walks `jax.tree.leaves(reference)`, inserts each provided grad at its index, zero-fills any leaf not supplied. Handles:

- **Same `ParamState` for all keys** (merged `{weight, bias}`): one entry in `per_path`; all leaves filled by the rule.
- **Separate `ParamState`s**: one entry per path; each wrap call fills only the leaves it owns, zero-filling any non-ETP siblings.
- **Partial coverage**: `ParamState` leaves the primitive doesn't consume stay zero (unchanged from today).

### Initialization

`init_drtrl` now returns `Dict[str, Array]` and is stored directly under the new primitive-instance-keyed `etrace_bwg` entry. `init_pp` returns a single y-shaped array — unchanged.

`_extract_weight_leaf` and `_wrap_leaf_as_pytree` in `_etrace_vjp/misc.py` are replaced by the generalized `_extract_leaf` / `_wrap_leaves_as_pytree`. Old names removed.

## Primitive rule migration

All 8 primitives (`etp_mm_p`, `etp_mv_p`, `etp_conv_p`, `etp_sp_mm_p`, `etp_sp_mv_p`, `etp_lora_mm_p`, `etp_lora_mv_p`, `etp_elemwise_p`) are rewritten to the dict-based signature.

### Example: `etp_mm_p` (batched)

```python
def _mm_trainable_invars(params):
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base

def _mm_yw_to_w(hidden_dim, trace, *, has_bias=False):
    # hidden_dim: (batch, out)
    out = {'weight': trace['weight'] * jnp.expand_dims(hidden_dim, axis=1)}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out

def _mm_xy_to_dw(x, hidden_dim, weights, *, has_bias=False):
    def fwd(w):
        y = x @ w['weight']
        if has_bias:
            y = y + w['bias']
        return u.get_mantissa(y)
    _, vjp_fn = jax.vjp(fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])

def _mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    batch = x_var.aval.shape[0]
    out = {'weight': jnp.zeros((batch, *weight_vars['weight'].aval.shape, num_hidden_state))}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros((batch, *weight_vars['bias'].aval.shape, num_hidden_state))
    return out

def _mm_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)
```

`etp_mv_p` mirrors `etp_mm_p` without the batch dim. `etp_conv_p` uses `jax.vjp` over `(kernel, bias)` the same way. Sparse primitives likewise — VJP through the sparse-matmul impl.

### `etp_lora_mm_p` / `etp_lora_mv_p`

```python
def _lora_trainable_invars(params):
    base = {'B': 1, 'A': 2}
    if params.get('has_bias', False):
        base['bias'] = 3
    return base

def _lora_xy_to_dw(x, hidden_dim, weights, *, alpha=1.0, has_bias=False):
    def fwd(w):
        y = alpha * (x @ w['B'] @ w['A'])
        if has_bias:
            y = y + w['bias']
        return u.get_mantissa(y)
    _, vjp_fn = jax.vjp(fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])

def _lora_mm_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    out = {
        'B': trace['B'],  # B frozen during trace propagation (existing behavior)
        'A': trace['A'] * jnp.expand_dims(hidden_dim, axis=1),
    }
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out
```

The fused-VJP form for `xy_to_dw` also fixes the current signature mismatch with the executor — one dict argument, no unpaired `w_B, w_A` positional args.

### `etp_elemwise_p`

Single-key dict `{'weight': ...}` throughout. `_elem_trainable_invars = lambda p: {'weight': 0}`.

## Diagnostics

- **New:** `TRAINABLE_INVAR_NOT_PARAMSTATE` at INFO — emitted when `trainable_invars_fn` yields a key whose invar doesn't trace to any `ParamState`. Forward runs; the key is dropped from the relation and produces no gradient.
- **Existing per-key:** `RELATION_EXCLUDED_WEIGHT_TO_WEIGHT` and path-classification diagnostics now evaluate each trainable key independently.

## Testing plan

1. **Migrate existing tests.** Every `relation.weight_path` / `.weight_var` / `.weight_leaf_idx` access is rewritten to `trainable_paths['weight']` / `trainable_vars['weight']` / `trainable_leaf_indices['weight']` — and to the appropriate key for LoRA. Counts like `len(relations) == 2` in GRU tests are unchanged (still one relation per primitive instance).
2. **Bias-gradient correctness** for `mm`, `mv`, `conv`, `sparse_mm`, `sparse_mv`: small recurrent net with `ParamState({'weight': W, 'bias': b})`; verify D-RTRL and ES-D-RTRL produce the same per-step `dW` and `db` as BPTT (reference oracle).
3. **Separate-`ParamState` layout:** `w = ParamState(W)`, `b = ParamState(b0)`; same matmul call; verify both gradients routed to the correct paths.
4. **LoRA end-to-end online learning** for both `etp_lora_mm_p` and `etp_lora_mv_p`, with and without bias — closes the silent gap today.
5. **Diagnostic test:** constant bias (not from a `ParamState`) → expect `TRAINABLE_INVAR_NOT_PARAMSTATE` info and zero bias gradient at runtime.
6. **Unit tests** for `_wrap_leaves_as_pytree`: multi-leaf fill, partial fill (zero for unsupplied leaves), bare-array fast path, index-out-of-range error.

## Out of scope

- Forward-pass semantics (impl functions, eqn params unchanged).
- Public user API (`braintrace.matmul`, `braintrace.lora_matmul`, `braintrace.conv`, `braintrace.sparse_matmul`, `braintrace.element_wise` signatures unchanged).
- Hidden-group discovery.
- Algorithm selection (`D_RTRL`, `ES_D_RTRL`).

## Files touched (anticipated)

- `braintrace/_etrace_op/_spec.py` — new `trainable_invars_fn` field; remove `weight_invar_index`.
- `braintrace/_etrace_op/_primitive.py` — minor; no API change.
- `braintrace/_etrace_op/{dense,conv,sparse,lora,elemwise}.py` — all 8 primitives' rules rewritten.
- `braintrace/_etrace_compiler/hid_param_op.py` — multi-key resolution; relation layout change.
- `braintrace/_etrace_compiler/diagnostics.py` — new diagnostic kind.
- `braintrace/_etrace_vjp/d_rtrl.py` — new trace keying; dict-based rule calls; multi-leaf gradient routing.
- `braintrace/_etrace_vjp/pp_prop.py` — dict-based `xy_to_dw` call; multi-leaf gradient routing.
- `braintrace/_etrace_vjp/misc.py` — `_extract_leaf`, `_wrap_leaves_as_pytree` replace old helpers.
- Tests in `braintrace/_etrace_compiler/`, `braintrace/_etrace_vjp/`, `braintrace/_etrace_op/` — migration + new correctness tests.
