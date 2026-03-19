# braintrace

Online learning for recurrent networks via Eligibility Trace Propagation (ETP). Version 0.2.0.

## What this package does

Implements online learning algorithms (D-RTRL, ES-D-RTRL, postsynaptic propagation) for recurrent neural networks using JAX custom primitives. Models use `braintrace.matmul(x, w)` instead of the old `ETraceParam(w, op=MatMulOp()).execute(x)`. The compiler identifies ETP primitives by type (`eqn.primitive in ETP_PRIMITIVES`) instead of string-matching JIT function names.

## Package structure

```
braintrace/
├── __init__.py                    Public API exports
├── _etrace_operators.py           ETP primitives, rule registries, user API functions
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
Layer 1: _etrace_operators.py    Primitives + rule registries + user API
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

### Three rule registries

Each primitive registers handlers in three global dicts:

- `etp_rules_yw_to_w[primitive]` — D-RTRL trace propagation: `(hidden_dim, trace, **p) -> trace`
- `etp_rules_xy_to_dw[primitive]` — weight gradient: `(x, hidden_dim, w, **p) -> dw`
- `etp_rules_init_state[primitive]` — trace initialization: `(x_var, y_var, weight, num_hidden_state) -> zeros`

### User API functions

- `matmul(x, weight, bias=None)` — auto-dispatches mm/mv based on `x.ndim`
- `element_wise(weight, fn=lambda w: w)` — element-wise marker
- `conv(x, kernel, bias, ...)` — convolution with full parameter support
- `sparse_matmul(x, weight_data, *, sparse_mat, bias=None)` — sparse matmul
- `lora_matmul(x, B, A, *, alpha=1.0, bias=None)` — LoRA decomposition

All functions handle `saiunit` quantities (split mantissa/unit, recombine after computation).

## Key design decisions

### Primitives are thin markers, not reimplementations

Each primitive's `impl` delegates to standard JAX ops (`x @ w`, `lax.conv_general_dilated`). All JAX rules (JVP, transpose, batching, abstract_eval, lowering) are auto-derived via `register_primitive()`. No hand-written derivative formulas.

### Batched vs unbatched dispatch

Dense matmul and sparse/LoRA matmul each have two primitives (mm/mv). The user API auto-dispatches based on `x.ndim >= 2`. This replaces the old `brainstate.mixin.Batching` mode checks — the primitive type encodes whether the computation is batched.

### ParamState not ETraceParam

The compiler maps ALL `brainstate.ParamState` invars. Selection is **primitive-based**: a parameter participates in ETP if and only if it is used through an ETP primitive. Parameters used with regular JAX ops are excluded — no special class needed.

```python
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

For each equation in the jaxpr:
1. Check `eqn.primitive in ETP_PRIMITIVES` (type identity, not string match)
2. Extract weight var from invars (index 1 for matmul/conv, index 0 for elemwise)
3. Trace weight var **backward** through jaxpr to find originating `ParamState`
4. BFS **forward** from output var to find reachable hidden-state outvars
5. Filter by shape compatibility (broadcast check)
6. Build transition jaxpr: y → h (for computing df = dh/dy)

Result: `HiddenParamOpRelation` — connects a `ParamState` to its reachable hidden states via an ETP primitive.

## Adding a new primitive

```python
from braintrace._etrace_operators import (
    register_primitive, etp_rules_yw_to_w, etp_rules_xy_to_dw, etp_rules_init_state
)

def _my_impl(x, w, *, my_param=1):
    return some_jax_op(x, w, my_param)

my_p = register_primitive('etp_my_op', _my_impl, batched=True)
etp_rules_yw_to_w[my_p] = lambda hidden_dim, trace, **p: trace * hidden_dim[None, :]
etp_rules_xy_to_dw[my_p] = lambda x, hidden_dim, w, **p: jax.vjp(lambda w: _my_impl(x, w, **p), w)[1](hidden_dim)[0]
etp_rules_init_state[my_p] = lambda x_var, y_var, weight, ns: jnp.zeros((x_var.aval.shape[0], *jnp.shape(weight.value), ns))
```

All JAX rules (jit, grad, vmap, jvp) are auto-derived. Only the three ETP-specific rules need hand-writing.

## Public API (`braintrace.__init__`)

```python
import braintrace

# ETP primitives (user API)
braintrace.matmul, braintrace.element_wise, braintrace.conv
braintrace.sparse_matmul, braintrace.lora_matmul

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

~700 test functions across 21 test files covering primitives, compiler, algorithms, VJP, and neural network layers.

## Dependencies

- **brainstate** >= 0.2.2 — state management and neural network base classes
- **saiunit** (imported as `u`) — physical unit support for mantissa/unit splitting
- **JAX** — core computation framework
- **braintools**, **brainpy-state** — additional utilities

## File dependency flow

```
_etrace_operators.py (primitives + rules + user API)
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
