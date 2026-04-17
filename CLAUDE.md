# braintrace

Online learning for recurrent networks via Eligibility Trace Propagation (ETP). Version 0.2.0.


1. Before write any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before write code.
3. After write code, list edge cases + suggest test cases.
4. Task touch >3 files? Stop, break into smaller tasks first.
5. Bug? Write test that reproduce it, then fix until test pass.
6. Every correction: reflect on mistake, plan to avoid repeat.


## What this package does

Online learning algos (D-RTRL, ES-D-RTRL, postsynaptic propagation) for RNNs via JAX custom primitives. Models use `braintrace.matmul(x, w)` instead of old `ETraceParam(w, op=MatMulOp()).execute(x)`. Compiler ID ETP primitives by type (`eqn.primitive in ETP_PRIMITIVES`) not string-match JIT names.

## Package structure

```
braintrace/
├── __init__.py                    Public API exports
├── _etrace_operators.py           ETPPrimitive, rule registries, user API functions
├── _etrace_algorithms.py          EligibilityTrace, ETraceAlgorithm
├── _etrace_graph_executor.py      ETraceGraphExecutor (forward pass + Jacobians)
├── _etrace_input_data.py          SingleStepData, MultiStepData
├── _etrace_compiler/              Jaxpr analysis & graph compilation
│   ├── base.py                    JaxprEvaluation, check_unsupported_op
│   ├── hid_param_op.py            HiddenParamOpRelation — find ETP primitives in jaxpr
│   ├── hidden_group.py            HiddenGroup — group related hidden states
│   ├── hidden_pertubation.py      HiddenPerturbation — perturbation analysis
│   ├── module_info.py             ModuleInfo, extract_module_info
│   └── graph.py                   ETraceGraph, compile_etrace_graph
├── _etrace_vjp/                   VJP-based online learning algorithms
│   ├── base.py                    ETraceVjpAlgorithm
│   ├── d_rtrl.py                  ParamDimVjpAlgorithm, D_RTRL
│   ├── pp_prop.py                 IODimVjpAlgorithm, ES_D_RTRL, pp_prop
│   ├── graph_executor.py          ETraceVjpGraphExecutor
│   └── misc.py                    Utility functions
├── nn/                            Neural network layers (Linear, Conv, RNN, Norm, etc.)
├── _compatible_imports.py         JAX version compatibility layer
├── _grad_exponential.py           GradExpon
├── _misc.py                       NotSupportedError, CompilationError
├── _state_managment.py            State management utilities
├── _typing.py                     Type aliases
└── _version.py                    Version: 0.2.0
```

## Architecture layers

```
Layer 1: _etrace_operators.py    ETPPrimitive + rule registries + user API
Layer 2: _etrace_compiler/       Jaxpr walk → find primitives → connect to hidden states
Layer 3: _etrace_graph_executor  Forward pass + h2w/h2h Jacobian computation
Layer 4: _etrace_vjp/            D-RTRL, ES-D-RTRL, postsynaptic propagation algorithms
         _etrace_algorithms.py   EligibilityTrace (top-level orchestrator)
```

## ETP primitives (`_etrace_operators.py`)

### Registered primitives

| Primitive | Operation | Batched? | Key invars |
|---|---|---|---|
| `etp_mm_p` | y = x @ w (+ b) | Yes | x (batch, in), w (in, out) |
| `etp_mv_p` | y = x @ w (+ b) | No | x (in,), w (in, out) |
| `etp_elemwise_p` | y = fn(w) | No | processed weight y |
| `etp_conv_p` | y = conv(x, kernel) (+ b) | Yes | x, kernel |
| `etp_sp_mm_p` / `etp_sp_mv_p` | sparse matmul | Yes/No | x, weight_data |
| `etp_lora_mm_p` / `etp_lora_mv_p` | y = alpha * x @ B @ A (+ b) | Yes/No | x, B, A |

### ETPPrimitive & rule registries

`register_primitive()` returns `ETPPrimitive` (subclass of JAX `Primitive`) with built-in rule registration methods. Rules stored in four global constant dicts:

| Registry | Method on `ETPPrimitive` | Signature |
|---|---|---|
| `ETP_RULES_YW_TO_W` | `register_yw_to_w(fn)` | `(hidden_dim, trace, **params) -> trace` |
| `ETP_RULES_XY_TO_DW` | `register_xy_to_dw(fn)` | `(x, hidden_dim, w, **params) -> dw` |
| `ETP_RULES_INIT_DRTRL` | `register_init_drtrl(fn)` | `(x_var, y_var, weight, num_hidden_state) -> zeros` |
| `ETP_RULES_INIT_PP` | `register_init_pp(fn)` | `(x_var, y_var, weight, num_hidden_state) -> zeros` |

Convenience method `register_etp_rules(*, yw_to_w, xy_to_dw, init_drtrl, init_pp)` register multiple rules one call.

### User API functions

- `matmul(x, weight, bias=None)` — auto-dispatch mm/mv by `x.ndim`
- `element_wise(weight, fn=lambda w: w)` — element-wise marker
- `conv(x, kernel, bias, ...)` — convolution, full param support
- `sparse_matmul(x, weight_data, *, sparse_mat, bias=None)` — sparse matmul
- `lora_matmul(x, B, A, *, alpha=1.0, bias=None)` — LoRA decomposition

All functions handle `saiunit` quantities (split mantissa/unit, recombine after).

## Key design decisions

### Primitives are thin markers, not reimplementations

Each primitive's `impl` delegates to standard JAX ops (`x @ w`, `lax.conv_general_dilated`). All JAX rules (JVP, transpose, batching, abstract_eval, lowering) auto-derived via `register_primitive()`, returns `ETPPrimitive` instance. No hand-written derivative formulas.

### Batched vs unbatched dispatch

Dense matmul and sparse/LoRA matmul each have two primitives (mm/mv). User API auto-dispatch on `x.ndim >= 2`. Replace old `brainstate.mixin.Batching` mode checks — primitive type encode whether computation batched.

### ParamState not ETraceParam

Compiler map ALL `brainstate.ParamState` invars. Selection **primitive-based**: parameter in ETP iff used through ETP primitive. Parameters used with regular JAX ops excluded — no special class needed.

```python
import brainstate, jax, braintrace

class MyRNN(brainstate.nn.Module):
    def __init__(self):
        self.w_rec = brainstate.ParamState(...)   # want ETP
        self.w_in = brainstate.ParamState(...)     # do NOT want ETP
        self.h = brainstate.ShortTermState(...)

    def update(self, x):
        # regular matmul → w_in excluded from ETP
        inp = x @ self.w_in.value

        # braintrace.matmul → w_rec included in ETP
        self.h.value = jax.nn.tanh(inp + braintrace.matmul(self.h.value, self.w_rec.value))
        return self.h.value
```

| Aspect | Old system | New system |
|---|---|---|
| Include in online learning | Use `ETraceParam` | Use `braintrace.matmul(x, w)` |
| Exclude from online learning | Use `brainstate.ParamState` | Use `x @ w` (regular JAX op) |
| Selection mechanism | Parameter class type | Operation primitive type |

## Compiler algorithm (`_etrace_compiler/hid_param_op.py`)

For each equation in jaxpr:
1. Check `eqn.primitive in ETP_PRIMITIVES` (type identity, not string match)
2. Extract weight var from invars (index 1 for matmul/conv, index 0 for elemwise)
3. Trace weight var **backward** through jaxpr to find originating `ParamState`
4. BFS **forward** from output var to find reachable hidden-state outvars — **stops at any other non-gradient-enabled ETP primitive** (see invariant below)
5. Filter by shape compatibility (broadcast check)
6. Build transition jaxpr: y → h — **equations belong to other non-gradient-enabled ETP primitives treated as constvar boundaries**, not recursed into

Result: `HiddenParamOpRelation` — connect `ParamState` to reachable hidden states via ETP primitive.

### Invariant: no "weight → weight → hidden" pathway

Each ETP primitive's rules (`xy_to_dw`, `yw_to_w`) assume `h = g(y)` where `g` contain **no other trainable ETP weights**. If primitive `W1`'s output flow through another non-gradient-enabled ETP primitive `W2` before reach `h`, `W1` must **not** be recorded as relation:

1. `W2` already own gradient of its input (depend on `W1`) — register `W1` too = double-count.
2. `W2`'s `xy_to_dw` assume its `x` externally-supplied data, not function of another trainable ETP weight.
3. Only correct decomposition bundle `W1` and `W2` together, which per-primitive ETP cannot express.

Filter enforced in `_find_reachable_hidden_outvars` and `_build_transition_jaxpr` via `is_etp_enable_gradient_primitive`. Gradient-enabled primitives (only `etp_elemwise_p` today; see `register_primitive(..., gradient_enabled=True)` in `_etrace_operators.py`) identity-like and *may* sit on tail.

**Concrete consequence — `GRUCell` has 3 Linears but only 2 ETP relations** (`Wz`, `Wh`). `Wr`'s output reach `h` only via `Wh`'s matmul (`r = sigmoid(Wr(xh)); rh = r * old_h; h = activation(Wh(concat([x, rh])))`), so `Wr` correctly excluded, warned non-temporal. Tests relying on these counts (`hid_param_op_test.py::test_gru_one_layer`, `graph_test.py::test_gru_one_layer`) bake in `len(relations) == 2`. When add or modify RNN cell, walk each parameter's path to `h`, count only those with non-parametric tail.

## Adding a new primitive

```python
import braintrace

def _my_impl(x, w, *, my_param=1):
    return some_jax_op(x, w, my_param)

my_p = braintrace.register_primitive('etp_my_op', _my_impl, batched=True)
my_p.register_yw_to_w(lambda hidden_dim, trace, **p: trace * hidden_dim[None, :])
my_p.register_xy_to_dw(lambda x, hidden_dim, w, **p: jax.vjp(lambda w: _my_impl(x, w, **p), w)[1](hidden_dim)[0])
my_p.register_init_drtrl(lambda x_var, y_var, weight, ns: jnp.zeros((x_var.aval.shape[0], *jnp.shape(weight.value), ns)))
my_p.register_init_pp(lambda x_var, y_var, weight, ns: jnp.zeros((*y_var.aval.shape, ns), dtype=y_var.aval.dtype))
```

All JAX rules (jit, grad, vmap, jvp) auto-derived. Only four ETP-specific rules need hand-writing via `ETPPrimitive` methods.

## Public API (`braintrace.__init__`)

```python
import braintrace

# ETP primitives (user API)
braintrace.matmul, braintrace.element_wise, braintrace.conv
braintrace.sparse_matmul, braintrace.lora_matmul

# ETP primitive registration
braintrace.ETPPrimitive         # Primitive subclass with register_* methods
braintrace.register_primitive   # create ETPPrimitive + auto-derive JAX rules

# Algorithms
braintrace.EligibilityTrace, braintrace.ETraceAlgorithm
braintrace.D_RTRL, braintrace.ES_D_RTRL, braintrace.pp_prop

# Compiler
braintrace.compile_etrace_graph, braintrace.ETraceGraph
braintrace.ModuleInfo, braintrace.extract_module_info
braintrace.HiddenGroup, braintrace.HiddenParamOpRelation, braintrace.HiddenPerturbation

# Execution
braintrace.ETraceGraphExecutor, braintrace.ETraceVjpGraphExecutor

# Input data
braintrace.SingleStepData, braintrace.MultiStepData

# Neural network layers
braintrace.nn  # Linear, Conv1d/2d/3d, BatchNorm, LayerNorm, RNN, GRU, LSTM, etc.
```

## Running tests

```bash
python -m pytest braintrace/ -v
```

~700 test functions across 21 test files cover primitives, compiler, algorithms, VJP, NN layers.

## Dependencies

- **brainstate** >= 0.2.2 — state management + NN base classes
- **saiunit** (imported as `u`) — physical unit support, mantissa/unit splitting
- **JAX** — core computation framework
- **braintools**, **brainpy-state** — additional utilities

## File dependency flow

```
_etrace_operators.py (ETPPrimitive + rules + user API)
    ↓
_etrace_compiler/ (jaxpr analysis & graph compilation)
    ├── hid_param_op.py ← _etrace_operators.ETP_PRIMITIVES
    ├── hidden_group.py
    ├── module_info.py
    └── graph.py
        ↓
_etrace_graph_executor.py (forward pass + Jacobians)
    ↓
_etrace_vjp/ (online learning algorithms)
    ├── d_rtrl.py
    ├── pp_prop.py
    └── base.py
        ↓
_etrace_algorithms.py (EligibilityTrace orchestrator)
```