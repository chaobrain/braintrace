# Design and Methodology of the ETrace Compiler

> Publication-style design document for the relation-extraction and
> graph-compilation subsystem of `braintrace` (`braintrace/_etrace_compiler/`).
> This document accompanies the source and is the canonical reference for
> the algorithmic invariants, the diagnostic taxonomy, and the testing
> framework.

## Abstract

`braintrace` is an online-learning library for recurrent networks built
around the idea of *Eligibility Trace Propagation* (ETP). The runtime
algorithms (D-RTRL, ES-D-RTRL) all assume a
specific graph-level decomposition of the model: every trainable weight
that should be learned online must sit at the *head* of an ETP primitive,
and the *tail* between the primitive's output `y` and the affected
hidden state `h` must contain no other trainable ETP weight.

The compiler in `braintrace/_etrace_compiler/` is the static analysis
that takes a `brainstate.nn.Module` and the inputs that drive one
update step, traces the model into a JAX `Jaxpr`, and produces three
artefacts: (i) a list of `HiddenGroup`s, (ii) a list of
`HiddenParamOpRelation`s connecting each ETP weight to the hidden states
it influences, and (iii) a structured stream of
`CompilationRecord`s explaining every decision the compiler took.

This document describes the algorithmic design of that compiler, the
invariants it preserves, the structured-diagnostic interface, the
determinism guarantees, the test framework that pins down its
correctness, and a brief discussion of limitations and future work.

## 1. Background

### 1.1 The online-learning problem

Backpropagation through time (BPTT) requires holding the full unrolled
graph in memory and is impractical for long sequences and continual
learning. Real-Time Recurrent Learning (RTRL) avoids the unroll but is
*O(N³)* in the hidden-state size *N*. ETP-class algorithms accept a
controlled approximation by *factoring* the gradient
`d L / d θ = (d L / d h) · (d h / d y) · (d y / d θ)` and propagating
only the small per-weight pieces forward in time.

### 1.2 Why the compiler

The factorisation only holds when the model's computation graph respects
specific structural conditions. Hand-checking those conditions is
error-prone in real models with many layers, masked weights, weight
tying, and pytree-valued state. The compiler examines the JAX `Jaxpr`
that abstractly represents one model step, identifies every ETP
primitive, and decides — per primitive — whether its weight participates
in ETP and, if so, against which hidden states.

### 1.3 The non-parametric-tail invariant

Each ETP primitive's rules (`xy_to_dw`, `yw_to_w`) assume that the path
from the primitive's output `y` to the affected hidden state `h`
contains no other *trainable* ETP weight. If a weight `W₁`'s output flows
through another non-gradient-enabled ETP primitive `W₂` before reaching
`h`, then `W₂` already owns the gradient of its input with respect to
`h` (and that input depends on `W₁`). Recording `W₁` as an independent
relation would double-count the gradient. The decomposition only
becomes valid if `W₁` and `W₂` are bundled — which the per-primitive
ETP design cannot express. The compiler therefore *excludes* such `W₁`s
from ETP.

This is the central correctness condition the compiler enforces. Every
algorithmic choice in the rest of this document derives from it.

## 2. Compiler Algorithm

The compiler is layered:

```
extract_module_info        →   ModuleInfo (jaxpr, weight/hidden invars)
find_hidden_groups         →   HiddenGroup list, hid_path → group dict
find_hidden_param_op_     →   HiddenParamOpRelation list, diagnostics
  relations
add_hidden_perturbation    →   HiddenPerturbation (optional)
```

This document focuses on the third layer (`hid_param_op.py`), which
performs the relation extraction and graph compilation that are the
structural core of the system.

### 2.1 Primitive identification (type identity, not strings)

The compiler looks for `eqn.primitive in ETP_PRIMITIVES`. ETP primitives
are first-class JAX primitive instances created by
`register_primitive`/`register_primitive_spec`; identity comparison is
therefore stable across imports and immune to renaming. A new primitive
is automatically picked up as soon as it is registered — *no compiler
edit is required*.

### 2.2 Weight resolution (backward trace to a `ParamState`)

For each ETP equation, the compiler reads the spec
(`get_primitive_spec(primitive)`) to find the weight invar index, then
walks backward through the producing equations until it reaches a JAX
variable that is registered as a `ParamState` invar. The walk records
the primitive types of every equation it traverses; that list is exposed
as `relation.weight_processing_chain`. A non-empty chain reveals
intervening ops such as `mul` (mask multiplication) or
`convert_element_type` (dtype coercions in `weight_fn`).

When the owning `ParamState` is pytree-valued (e.g.
`{'W': ..., 'b': ...}`), the compiler resolves which leaf the primitive
actually consumed and exposes its index as `weight_leaf_idx`. If the
resolution is ambiguous (multiple leaves reach the primitive through
disjoint paths and none of them matches the traced source), the compiler
emits `PYTREE_WEIGHT_LEAF_AMBIGUOUS` and falls back to leaf 0 — the
diagnostic is the user-actionable signal that something is wrong.

### 2.3 Hidden-state reachability (forward BFS, group-scoped)

From the ETP primitive's output `y`, the compiler runs a forward
breadth-first search on the consumer map. It collects every hidden-state
outvar that `y` can reach, restricted to the *home* hidden group: the
group that owns the first hidden outvar encountered. This restriction
prevents a weight in layer 0 of a stacked RNN from spuriously registering
relations against the hidden states of layer 1, layer 2, …  See
`StackedDeepRNN` for the canonical scenario.

The BFS uses an insertion-ordered `dict` (not a `set`) so that iteration
order is stable across runs. This is the foundational determinism
guarantee — every higher-level data structure inherits it.

### 2.4 Path classification (the W → W → h enforcement)

The forward BFS comes in two modes:

* **Restricted** — refuses to cross any non-gradient-enabled ETP
  primitive. The set of hidden states reachable in this mode is the set
  of relations the compiler will register.
* **Unrestricted** — crosses every consumer edge. Used in *path
  classification*.

For each `(weight, hidden)` candidate the compiler then classifies the
set of paths between them as one of three categories:

| Classification           | Definition                                   | Outcome                                                     |
| ------------------------ | -------------------------------------------- | ----------------------------------------------------------- |
| `ALL_DIRECT`             | Every path avoids other non-grad ETP        | Relation included; emits `RELATION_INCLUDED` (info)        |
| `ALL_THROUGH_OTHER_ETP`  | Every path traverses another non-grad ETP   | Relation excluded; emits `RELATION_EXCLUDED_WEIGHT_TO_WEIGHT` (warn) |
| `MIXED`                  | Some paths direct, others through other ETP | Relation included (preserves prior behavior); emits `RELATION_PARTIAL_PATH` (warn) |

The classification is derived from a forward-and-backward intersection of
reachable variables: for any equation that lies in the intersection
(other than `eqn` itself), if it is a non-gradient-enabled ETP primitive
then at least one path passes through another ETP weight. Whether the
*direct* path also exists is determined by the restricted BFS.

The MIXED case deserves special note. The current behavior, preserved
by user requirement, is to include the relation. The user will, however,
want to know that ETP only captures the direct contribution to
`dh/dy`; the `RELATION_PARTIAL_PATH` diagnostic surfaces this with
machine-readable structure so downstream tooling can decide whether to
fall back to BPTT for that weight.

### 2.5 Transition-Jaxpr construction

For each `(weight, group)` pair the compiler emits a small `Jaxpr`
mapping `y → group.hidden_outvars`. It is constructed by a *backward
slice* from the hidden outvars: equations producing variables in the
slice are included; equations producing values external to the slice
become *constvars*; the equation that produces `y` itself is excluded
because `y` is the slice's *invar*. Critically, equations whose primitive
is a non-gradient-enabled ETP primitive are also excluded — their
output becomes a constvar, capturing the value at compile time so
`dh/dy` does not double-count through them.

Why exclude the equation producing `y`? Without that exclusion, a
gradient-enabled ETP primitive (such as `etp_elemwise_p`) on the tail
would re-execute when the transition jaxpr runs, ignoring the supplied
`y` invar and yielding an apparent zero Jacobian. This subtle bug was
caught by the numerical-oracle test on `ElemwiseOnlyRNN`.

### 2.6 Control-flow descent

The Jaxpr scanner descends into

* `jit` / `pjit` — transparent inlining (with a structural diagnostic).
* `scan` — body jaxpr is walked.
* `while` — cond and body jaxprs are walked.
* `cond` — every branch jaxpr is walked.

ETP primitives discovered inside any of `scan`/`while`/`cond` are *not*
lifted as relations. The carry-variable lineage required to expose
their outputs across the control-flow boundary is not yet supported. A
`PRIMITIVE_INSIDE_CONTROL_FLOW` diagnostic is emitted instead so the
user can locate the problematic call and either lift the weight out of
the body or learn it via BPTT.

### 2.7 Multi-output primitives

`ETPPrimitiveSpec` carries a `y_outvar_index` field (default 0). The
compiler reads `eqn.outvars[spec.y_outvar_index]` for `y_var`. When a
primitive has more than one output the compiler emits
`MULTI_OUTPUT_PRIMITIVE_DETECTED` (info) so the registration is visible
in the diagnostic stream.

## 3. Invariants and correctness sketch

The compiler enforces the following invariants on every successful
compilation:

1. **Type identity.** A `Jaxpr` equation participates in ETP if and
   only if `eqn.primitive in ETP_PRIMITIVES`.
2. **Weight provenance.** Every registered relation's `weight_path`
   resolves to a `ParamState` reachable from the model's state tree.
3. **Non-parametric tail.** For every registered relation, the
   transition `Jaxpr` from `y_var` to the relation's hidden group
   contains no equation whose primitive is a non-gradient-enabled ETP
   primitive.
4. **Group scope.** A relation's `connected_hidden_paths` are all in
   the home hidden group of the first hidden outvar reached during the
   forward BFS.
5. **Shape compatibility.** `y_var.aval.shape` is broadcastable with
   every `hidden_outvar.aval.shape` in the relation.
6. **Determinism.** Two compilations of structurally-identical models
   with identical inputs produce relation lists in identical order, with
   identical `path_classification` dictionaries and identical
   diagnostic streams.

Conditions (1)–(5) can be read off the relation list directly. Condition
(6) is verified by the property tests in
`compiler_property_test.py::TestIdempotence`.

## 4. Diagnostics

Every compiler decision emits a `CompilationRecord`. The records form a
machine-readable explanation that does not depend on parsing warning
strings. Each record carries `kind`, `level`, `message`, `primitive`,
`weight_path`, `hidden_paths`, and a free-form `context` dict.

### 4.1 Diagnostic kinds

```
RELATION_INCLUDED                       (info)
RELATION_EXCLUDED_NO_PARAMSTATE         (warn)
RELATION_EXCLUDED_NON_TEMPORAL          (warn)
RELATION_EXCLUDED_SHAPE_MISMATCH        (warn)
RELATION_EXCLUDED_WEIGHT_TO_WEIGHT      (warn)
RELATION_PARTIAL_PATH                   (warn)
PRIMITIVE_INSIDE_NESTED_JIT             (warn)
PRIMITIVE_INSIDE_CONTROL_FLOW           (warn)
MULTI_OUTPUT_PRIMITIVE_DETECTED         (info)
PYTREE_WEIGHT_LEAF_AMBIGUOUS            (warn)
TRANSITION_TAIL_BOUNDED                 (info, reserved)
HIDDEN_GROUP_MERGED                     (info, reserved)
STATE_MISMATCH                          (warn)
WEIGHT_IN_CONTROL_FLOW                  (warn, reserved)
```

The reserved kinds are wired in the `DiagnosticKind` enum and emitted by
neighbouring subsystems (e.g. hidden-group construction); they are
listed here so downstream tools can switch on the full taxonomy.

### 4.2 Structured access via `ETraceGraph.explain`

```python
graph = compile_etrace_graph(model, *inputs)
for record in graph.explain(kind=DiagnosticKind.RELATION_PARTIAL_PATH):
    print(record.weight_path, record.hidden_paths)
```

Filtering by `weight_path`, `hidden_path`, and/or `kind` is supported.

## 5. Determinism

Determinism is a property the compiler must preserve, not just achieve
incidentally. The implementation discipline is:

1. Every traversal data structure that escapes a function as part of the
   compiler's output must be insertion-ordered. `set`s are restricted to
   internal membership tests where the data does not influence output
   order.
2. Equation iteration follows the order in `jaxpr.eqns`.
3. Var iteration follows insertion order (Python's `dict`).
4. The home-group restriction commits to a *single* group as soon as
   the first hidden outvar is encountered; no subsequent hidden outvar
   can change the choice.

The property test
`compiler_property_test.py::TestIdempotence::test_unbatched_mv_rnn_is_idempotent`
compiles random-shape models twice and asserts a structural fingerprint
match including diagnostic order.

## 6. Testing framework

The test framework has three layers, each pinning down a different
property of the compiler.

### 6.1 Discriminative scenario catalog (`scenario_catalog_test.py`)

The catalog contains 30+ scenarios grouped into 14 categories (A–N)
covering single-primitive baselines, chain traversal, fan-in/fan-out,
exclusion paths, canonical recurrences (GRU, LSTM), determinism,
structural smoke tests, pytree weights, masked weights, stacked deep
networks, shared/tied weights, mixed batching, partial paths, and
control flow. Each scenario asserts on the *exact* set of registered
`(weight_path, hidden_path)` pairs, on primitive type identity (using
`is`), on the structured diagnostic emitted, and (where applicable) on
the path classification. This is the canonical answer to "what should
the compiler do for X?".

The reusable model fixtures live in
`scenario_catalog.py`. New scenarios should be added there so the
property and oracle tests can also import them.

### 6.2 Property tests (`compiler_property_test.py`)

Hypothesis-driven tests that pin down invariants across a parameter
space rather than at fixed examples:

* Idempotence of compilation for `UnbatchedMvRNN` and `PartialPathRNN`
  across random shape choices.
* `StackedDeepRNN(depth=k)` produces exactly `k` relations, each scoped
  to its own layer.
* `SharedTiedWeightRNN` produces exactly two relations (one per call
  site).
* A chain of `k` matmuls in series registers exactly the last weight
  and emits exactly `k − 1` `RELATION_EXCLUDED_WEIGHT_TO_WEIGHT`
  records.
* `PartialPathRNN` always classifies `w1` as `MIXED` and `w2` as
  `ALL_DIRECT` regardless of input/output sizes.

### 6.3 Numerical oracle (`compiler_oracle_test.py`)

Treats the compiler as a black box and verifies that the transition
jaxpr `y → h` is *semantically* correct by:

1. Running `jax.jacfwd` on the transition function to obtain `dh/dy`.
2. Computing `dh/dy` analytically from the model's known forward
   expression (`tanh` tail → `diag(1 − h²)`).
3. Asserting agreement to `rtol=atol=1e-5`.

A finite-difference cross-check (`atol=1e-3`) verifies the autograd
result against the same forward function evaluated at perturbed inputs.

The `ElemwiseOnlyRNN` oracle test caught a real compiler bug where the
equation producing `y_var` was not excluded from the transition jaxpr,
making `dh/dy` silently zero for any gradient-enabled ETP primitive on
the tail.

## 7. Limitations and future work

* **ETP primitives inside control-flow bodies** are detected and
  reported but cannot yet be lifted to relations. Carry-variable
  lineage across `scan`/`while`/`cond` boundaries is required and is
  left to future work.
* **MIXED-path gradient capture.** When a path from `w` to `h` traverses
  both a direct tail and another ETP primitive, the relation is
  registered but the indirect contribution is silently dropped from the
  ETP estimate. The structured `RELATION_PARTIAL_PATH` record surfaces
  this; downstream consumers must decide whether to fall back to BPTT
  for that particular weight.
* **Multi-output primitives** are supported via the
  `y_outvar_index` field, but rules (`yw_to_w`, `xy_to_dw`,
  `init_drtrl`, `init_pp`) for non-trivial multi-output primitives are
  not yet validated end-to-end in the test suite.

## 8. Glossary

| Term                          | Meaning                                                   |
| ----------------------------- | --------------------------------------------------------- |
| *ETP primitive*              | A JAX primitive registered via `register_primitive[_spec]`, listed in `ETP_PRIMITIVES`. |
| *gradient-enabled primitive* | An ETP primitive whose output value is identity-like; the compiler may walk through it on the tail (currently only `etp_elemwise_p`). |
| *hidden group*                | A set of hidden states sharing a recurrent transition jaxpr. |
| *home group*                 | For one ETP primitive, the hidden group of the first hidden outvar reached by its forward BFS. |
| *non-parametric tail*         | The path from a primitive's `y` to a hidden state `h` containing no other trainable ETP weight. |
| *relation*                   | A `HiddenParamOpRelation`: one ETP primitive's call site, its weight, the hidden states it influences, and the transition jaxpr from `y` to those hidden states. |
| *transition jaxpr*           | A small `Jaxpr` mapping `y → h` for one relation; computed by a backward slice with non-grad ETP primitives lifted to constvars. |

## 9. References

The relevant source files (within `braintrace/_etrace_compiler/`):

* `hid_param_op.py` — relation extraction.
* `hidden_group.py` — hidden-group analysis.
* `module_info.py` — model abstraction → `ModuleInfo`.
* `graph.py` — top-level orchestrator (`compile_etrace_graph`).
* `diagnostics.py` — `DiagnosticKind`, `CompilationRecord`,
  `diagnostic_context`, `emit`.
* `scenario_catalog.py` — reusable test fixtures.
* `scenario_catalog_test.py` — discriminative scenarios (categories A–N).
* `compiler_property_test.py` — Hypothesis-driven property tests.
* `compiler_oracle_test.py` — numerical oracles.
