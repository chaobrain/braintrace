# braintrace

Online learning for recurrent networks via Eligibility Trace Propagation (ETP).

## Working agreement

1. Before writing any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before writing code.
3. After writing code, list edge cases + suggest test cases.
4. Bug? Write a test that reproduces it, then fix until the test passes.
5. Every correction: reflect on the mistake, plan to avoid repeating it.
6. All updates must be happened on the worktree branch, not main. 
7. Write specs under `docs/specs` before implementation.
8. Tests should >90% coverage, but focus on meaningful tests that cover edge cases and critical paths, not just trivial lines. 
9. Co-locate tests with the code under test: each module `foo.py` has its tests in a sibling `foo_test.py` (suffix style — never a separate `tests/` directory, never the `test_*.py` prefix). 
10. **Never drive a model with a bare Python `for`/`while` loop when it runs repeatedly.** Python loops execute op-by-op (dispatch overhead, no fusion) and trace fresh each step; the `brainstate.transform` primitives lower the whole loop into one compiled XLA program, tracing the body only once. Pick by shape of the work:
    - **Single step or one-shot call** → `brainstate.transform.jit` — wrap the step/model call so it compiles once and reuses the trace.
    - **Many steps, collect outputs** → `brainstate.transform.for_loop` — repeat a step `length` times or map over `xs`; `State` is carried automatically and stacked outputs are returned.
    - **Many steps with an explicit carry** → `brainstate.transform.scan` — when threading a carry value alongside `State` (`f(carry, x) -> (carry, y)`).
    - **Long rollout under autograd (backprop through time)** → `brainstate.transform.checkpointed_for_loop` / `brainstate.transform.checkpointed_scan` — same semantics as above but rematerialize activations on the backward pass (tune `base`) to bound peak memory at the cost of recomputation.

    Compose them freely (e.g. `jit` an outer driver that calls a `for_loop`/`scan`). Reach for the checkpointed variants only when reverse-mode gradients through a long simulation would otherwise exhaust memory — otherwise prefer plain `for_loop`/`scan`.
11. Use `brainstate.random` instead of `jax.random` directly for all random number generation. 


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
Layer 4  Algorithms         Orchestrators (D-RTRL, pp_prop/ES-D-RTRL, EProp/OSTL)
```

Dependency direction is strictly downward: operators know nothing of the
compiler; the compiler depends on the operator registry; algorithms depend on
the compiler and executor. Legacy back-compat shims are a side branch nothing
else depends on.

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

## Known limitations

First-cut SNN algorithms pass smoke and cross-checks but carry approximation
edges and rough spots. These are enumerated, verified against the test suite,
and mapped to concrete improvement actions in the findings list at
`docs/specs/2026-07-25-known-limitations.md`. Treat that list as the backlog of
expected-failure and improvement items rather than duplicating it here.

One rule from that list is load-bearing enough to state here: a gradient
assertion whose subject is a **learning-rule property** — a trace
factorization, a temporal recursion, a recurrence scope, a filter, a learning
signal — must be measured through a *finite-window* oracle path
(`chunked_online_param_gradients`). A whole-sequence multi-step VJP has no
truncation left to approximate and returns BPTT for every algorithm at every
hyperparameter, so such an assertion passes vacuously there. Assertions about
the compiler or an ETP per-primitive rule may use the whole-sequence path.


## Docstring style (NumPy-doc)

All public classes, methods, functions must use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). Rules for the Examples section: 

- Wrap example code in `.. code-block:: python` directive so Sphinx render with syntax highlighting.
- Prefix every input line with `>>>` (continuation lines with `...`) for `doctest` compatibility.
- Show expected output on line immediately after statement, **without** prompt prefix.
- Separate distinct scenarios with blank `>>>` line.
- Always include necessary imports (`import brainunit as u`, etc.) at top of example block so self-contained.

