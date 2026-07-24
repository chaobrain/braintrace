# braintrace

Online learning for recurrent networks via Eligibility Trace Propagation (ETP).

> This file holds **durable, abstract rules**: architecture, invariants, design
> principles, and process requirements. It deliberately avoids concrete file
> paths, exact symbol names, signatures, and counts — those drift. For the
> current concrete layout, read the source: `braintrace/__init__.__all__` is the
> source of truth for the public API; the package directory tree is the source of
> truth for module structure.


## Working agreement

1. Before writing any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before writing code.
3. After writing code, list edge cases + suggest test cases.
4. Bug? Write a test that reproduces it, then fix until the test passes.
5. Every correction: reflect on the mistake, plan to avoid repeating it.
6. All updates must be happened on the worktree branch, not main. 
7. Write spec and plan under `/mnt/d/codes/projects/brainmass/dev/superpowers` as gitignored files before implementation. This makes them available for reference during implementation, but not clutter the repo history.
8. Tests should >90% coverage, but focus on meaningful tests that cover edge cases and critical paths, not just trivial lines. 
9. Co-locate tests with the code under test: each module `foo.py` has its tests in a sibling `foo_test.py` (suffix style — never a separate `tests/` directory, never the `test_*.py` prefix). 
10. **Never drive a model with a bare Python `for`/`while` loop when it runs repeatedly.** Python loops execute op-by-op (dispatch overhead, no fusion) and trace fresh each step; the `brainstate.transform` primitives lower the whole loop into one compiled XLA program, tracing the body only once. Pick by shape of the work:
    - **Single step or one-shot call** → `brainstate.transform.jit` — wrap the step/model call so it compiles once and reuses the trace.
    - **Many steps, collect outputs** → `brainstate.transform.for_loop` — repeat a step `length` times or map over `xs`; `State` is carried automatically and stacked outputs are returned.
    - **Many steps with an explicit carry** → `brainstate.transform.scan` — when threading a carry value alongside `State` (`f(carry, x) -> (carry, y)`).
    - **Long rollout under autograd (backprop through time)** → `brainstate.transform.checkpointed_for_loop` / `brainstate.transform.checkpointed_scan` — same semantics as above but rematerialize activations on the backward pass (tune `base`) to bound peak memory at the cost of recomputation.

    Compose them freely (e.g. `jit` an outer driver that calls a `for_loop`/`scan`). Reach for the checkpointed variants only when reverse-mode gradients through a long simulation would otherwise exhaust memory — otherwise prefer plain `for_loop`/`scan`.


## What this package does

Online learning algorithms (D-RTRL, ES-D-RTRL, and a family of SNN algorithms)
for RNNs, built on JAX custom primitives. Models mark trainable operations with
ETP user-API ops (e.g. an ETP `matmul`) rather than wrapping parameters in a
special class. A compiler walks the jaxpr, identifies ETP primitives, and
connects parameters to the hidden states they influence.

## Architecture (layered)

```
Layer 1  ETP operators      Custom JAX primitives + per-primitive rule registries + user-facing ops
Layer 2  ETP compiler       Jaxpr analysis: find ETP primitives, connect parameters to hidden states
Layer 3  Graph executor     Forward pass + hidden→weight / hidden→hidden Jacobian computation
Layer 4  Algorithms         Orchestrators (D-RTRL, pp_prop/ES-D-RTRL, EProp/OSTL/OTPE/OTTT/OSTTP)
```

Dependency direction is strictly downward: operators know nothing of the
compiler; the compiler depends on the operator registry; algorithms depend on
the compiler and executor. Legacy back-compat shims are a side branch nothing
else depends on.

## Core design principles

These are the load-bearing rules. Preserve them across refactors.

### Primitives are thin markers, not reimplementations

Each ETP primitive's implementation delegates to a standard JAX op. All standard
JAX rules (JVP, transpose, batching, abstract eval, lowering) are auto-derived
from that implementation. **Never hand-write derivative formulas for standard
rules.** Only the small set of *ETP-specific* rules is hand-written per
primitive (trace propagation, the input/hidden→weight-gradient rule, and the
trace-state initializers). Adding a primitive means supplying those few rules;
everything else is free.

### Selection is primitive-based, not class-based

A parameter participates in online learning **iff it is used through an ETP
primitive**. The same parameter used through a regular JAX op is excluded. There
is no special parameter class controlling this — the *operation* decides. To
include a parameter, route it through an ETP op; to exclude it, use a plain JAX
op. The compiler considers all ordinary parameter states and filters by how each
is used.

### Identify primitives by type, not by name

The compiler recognizes ETP primitives by primitive-type identity, never by
string-matching op or trace names. Keep it that way.

### Batched vs unbatched is encoded in the primitive

Operations that have both batched and unbatched forms expose two primitives; the
user-facing op dispatches by input rank. Do not reintroduce runtime
batching-mode flags — the primitive identity carries that information.

### Invariant: no "weight → weight → hidden" pathway

Each ETP primitive's hand-written rules assume the map from its output to the
hidden state contains **no other trainable ETP weight**. If one trainable ETP
op's output flows through *another* (non-gradient-enabled) ETP op before
reaching the hidden state, the first op must **not** be recorded as a relation —
otherwise its contribution is double-counted, and per-primitive rules cannot
express the correct joint decomposition. The compiler enforces this by stopping
forward reachability at such primitives and treating them as boundaries.

A small set of identity-like, explicitly *gradient-enabled* primitives are
exempt and may sit on the tail of such a path.

**Practical consequence:** a cell can have more trainable linear maps than ETP
relations, because some parameters reach the hidden state only *through another
trainable map* and are correctly excluded (and flagged non-temporal). When
adding or modifying an RNN cell, trace each parameter's path to the hidden
state and count only those whose tail to the hidden state is non-parametric.
Tests that assert relation counts encode this invariant — update them
deliberately, not reflexively.

### Quantity (unit) support

ETP user-API ops accept physical-unit quantities: split mantissa/unit, compute,
recombine. New ops must preserve this.

### SNN learning-signal axis invariant

SNN learning signals carry a trailing per-hidden-state axis. Every learning-
signal hook, every per-algorithm weight-gradient solver, and every new SNN
algorithm must thread this axis explicitly (einsum/broadcast/collapse-expand).
Misuse produces shape errors or silently wrong gradients. Treat this axis layout
as a contract.

## Algorithm taxonomy (for correctness reasoning)

- **Exact algorithms** compute the same total gradient as backprop-through-time
  (BPTT), just forward instead of backward. They must match a BPTT oracle
  element-wise.
- **Approximate algorithms** deliberately drop or factor part of the
  computation. They match BPTT *only* in the degenerate regime their math
  guarantees; elsewhere they are expected to diverge. Correctness for them means
  "exact in the guaranteed regime" + "bounded, well-behaved divergence + descent"
  generally — not element-wise equality everywhere.

Know which class an algorithm belongs to before asserting anything about its
gradients.

## Dependencies (abstract)

- A state-management + NN base-class library (provides parameter/hidden state
  abstractions and cell base classes).
- A physical-units library (mantissa/unit handling).
- JAX as the core computation framework.
- Additional brain-ecosystem utilities.

Pin exact names/versions in `pyproject.toml` / requirements files,.

## Known limitations

First-cut SNN algorithms pass smoke/cross-checks but carry approximation edges
and rough spots (e.g. approximation-mode validity beyond shallow depth,
heterogeneous-population leak resolution, target-signal threading under JIT,
single-readout/feedback-shape assumptions, and gaps in cross-algorithm
equivalence coverage). These are enumerated and mapped to concrete improvement
actions in the test-strategy findings list under `dev/`. Treat that list as the
backlog of expected-failure / improvement items rather than duplicating it here.


## Rules

1. Use `brainstate.random` instead of `jax.random` directly for all random number generation. 
2. Place superpowers spec and plans in `/mnt/d/codes/projects/braintrace/dev/superpowers` directory.

## Docstring style (NumPy-doc)

All public classes, methods, functions must use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). Canonical section order:

1. **Short summary** – one-line imperative description (no blank line before).
2. **Extended summary** – optional, follow blank line after short summary.
3. **Parameters** – each entry: `name : type` on own line, description indented below.
4. **Returns** / **Yields** – same format as Parameters.
5. **Raises** – exception type and when raised.
6. **See Also** – related functions / classes.
7. **Notes** – implementation details, math, references.
8. **References** – numbered bibliography entries (`.. [1]`).
9. **Examples** – runnable, doctestable code snippets.

#### Rules for the Examples section

- Wrap example code in `.. code-block:: python` directive so Sphinx render with syntax highlighting.
- Prefix every input line with `>>>` (continuation lines with `...`) for `doctest` compatibility.
- Show expected output on line immediately after statement, **without** prompt prefix.
- Separate distinct scenarios with blank `>>>` line.
- Always include necessary imports (`import brainunit as u`, etc.) at top of example block so self-contained.

