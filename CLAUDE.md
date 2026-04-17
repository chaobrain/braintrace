# braintrace

Online learning for recurrent networks via Eligibility Trace Propagation (ETP). Version 0.2.0.


## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.




## What this package does

Implements online learning algorithms (D-RTRL, ES-D-RTRL, postsynaptic propagation) for recurrent neural networks using JAX custom primitives. Models use `braintrace.matmul(x, w)` instead of the old `ETraceParam(w, op=MatMulOp()).execute(x)`. The compiler identifies ETP primitives by type (`eqn.primitive in ETP_PRIMITIVES`) instead of string-matching JIT function names.

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

`register_primitive()` returns an `ETPPrimitive` (subclass of JAX `Primitive`) with built-in rule registration methods. Rules are stored in four global constant dicts:

| Registry | Method on `ETPPrimitive` | Signature |
|---|---|---|
| `ETP_RULES_YW_TO_W` | `register_yw_to_w(fn)` | `(hidden_dim, trace, **params) -> trace` |
| `ETP_RULES_XY_TO_DW` | `register_xy_to_dw(fn)` | `(x, hidden_dim, w, **params) -> dw` |
| `ETP_RULES_INIT_DRTRL` | `register_init_drtrl(fn)` | `(x_var, y_var, weight, num_hidden_state) -> zeros` |
| `ETP_RULES_INIT_PP` | `register_init_pp(fn)` | `(x_var, y_var, weight, num_hidden_state) -> zeros` |

A convenience method `register_etp_rules(*, yw_to_w, xy_to_dw, init_drtrl, init_pp)` registers multiple rules in one call.

### User API functions

- `matmul(x, weight, bias=None)` — auto-dispatches mm/mv based on `x.ndim`
- `element_wise(weight, fn=lambda w: w)` — element-wise marker
- `conv(x, kernel, bias, ...)` — convolution with full parameter support
- `sparse_matmul(x, weight_data, *, sparse_mat, bias=None)` — sparse matmul
- `lora_matmul(x, B, A, *, alpha=1.0, bias=None)` — LoRA decomposition

All functions handle `saiunit` quantities (split mantissa/unit, recombine after computation).

## Key design decisions

### Primitives are thin markers, not reimplementations

Each primitive's `impl` delegates to standard JAX ops (`x @ w`, `lax.conv_general_dilated`). All JAX rules (JVP, transpose, batching, abstract_eval, lowering) are auto-derived via `register_primitive()`, which returns an `ETPPrimitive` instance. No hand-written derivative formulas.

### Batched vs unbatched dispatch

Dense matmul and sparse/LoRA matmul each have two primitives (mm/mv). The user API auto-dispatches based on `x.ndim >= 2`. This replaces the old `brainstate.mixin.Batching` mode checks — the primitive type encodes whether the computation is batched.

### ParamState not ETraceParam

The compiler maps ALL `brainstate.ParamState` invars. Selection is **primitive-based**: a parameter participates in ETP if and only if it is used through an ETP primitive. Parameters used with regular JAX ops are excluded — no special class needed.

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

For each equation in the jaxpr:
1. Check `eqn.primitive in ETP_PRIMITIVES` (type identity, not string match)
2. Extract weight var from invars (index 1 for matmul/conv, index 0 for elemwise)
3. Trace weight var **backward** through jaxpr to find originating `ParamState`
4. BFS **forward** from output var to find reachable hidden-state outvars — **stops at any other non-gradient-enabled ETP primitive** (see invariant below)
5. Filter by shape compatibility (broadcast check)
6. Build transition jaxpr: y → h — **equations belonging to other non-gradient-enabled ETP primitives are treated as constvar boundaries**, not recursed into

Result: `HiddenParamOpRelation` — connects a `ParamState` to its reachable hidden states via an ETP primitive.

### Invariant: no "weight → weight → hidden" pathway

Each ETP primitive's rules (`xy_to_dw`, `yw_to_w`) assume `h = g(y)` where `g` contains **no other trainable ETP weights**. If primitive `W1`'s output flows through another non-gradient-enabled ETP primitive `W2` before reaching `h`, `W1` must **not** be recorded as a relation:

1. `W2` already owns the gradient of its input (which depends on `W1`) — registering `W1` too would double-count.
2. `W2`'s `xy_to_dw` assumes its `x` is externally-supplied data, not a function of another trainable ETP weight.
3. The only correct decomposition bundles `W1` and `W2` together, which per-primitive ETP cannot express.

The filter is enforced in `_find_reachable_hidden_outvars` and `_build_transition_jaxpr` via `is_etp_enable_gradient_primitive`. Gradient-enabled primitives (only `etp_elemwise_p` today; see `register_primitive(..., gradient_enabled=True)` in `_etrace_operators.py`) are identity-like and *may* sit on the tail.

**Concrete consequence — `GRUCell` has 3 Linears but only 2 ETP relations** (`Wz`, `Wh`). `Wr`'s output reaches `h` only via `Wh`'s matmul (`r = sigmoid(Wr(xh)); rh = r * old_h; h = activation(Wh(concat([x, rh])))`), so `Wr` is correctly excluded and warned as non-temporal. Tests relying on these counts (`hid_param_op_test.py::test_gru_one_layer`, `graph_test.py::test_gru_one_layer`) bake in `len(relations) == 2`. When adding or modifying an RNN cell, walk each parameter's path to `h` and count only those whose tail is non-parametric.

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

All JAX rules (jit, grad, vmap, jvp) are auto-derived. Only the four ETP-specific rules need hand-writing via `ETPPrimitive` methods.

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

~700 test functions across 21 test files covering primitives, compiler, algorithms, VJP, and neural network layers.

## Dependencies

- **brainstate** >= 0.2.2 — state management and neural network base classes
- **saiunit** (imported as `u`) — physical unit support for mantissa/unit splitting
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
