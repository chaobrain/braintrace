# Warning Visibility and Map Initialization

## Status

Approved for implementation.

## Motivation

BrainTrace warnings are part of the public diagnostic surface. Library code,
tests, examples, and documentation must not suppress them with
`warnings.catch_warnings` or `warnings.filterwarnings`.

Mapped state initialization must use the state-management abstraction provided
by BrainState. The supported flow is:

```python
model = brainstate.nn.Map(model, init_map_size=batch_size)
model.init_all_states()
```

This replaces executable uses of
`brainstate.transform.vmap_new_states`. Historical changelog entries may retain
the old symbol when they describe behavior from an earlier release.

## Requirements

1. Remove every executable use of `warnings.catch_warnings`,
   `warnings.filterwarnings`, and bare `filterwarnings`.
2. Do not add replacement warning filters or warning-suppression helpers.
3. Let BrainTrace warnings reach users and test output unchanged.
4. Replace mapped-state discovery and initialization with
   `brainstate.nn.Map(model, init_map_size=...)` followed by
   `model.init_all_states()`.
5. Compile mapped algorithms against the complete batched example input and
   return the algorithm object directly.
6. Keep mapped model states discoverable by the compiler without duplicating
   state paths.
7. Preserve ETP primitives and ETP-specific batching behavior under
   `brainstate.nn.Map`.
8. Update affected tests, examples, and tutorials to use the supported Map
   workflow.

## Non-goals

- Do not change algorithm equations, optimization behavior, or public callable
  signatures unrelated to mapped initialization.
- Do not suppress third-party compatibility warnings.
- Do not address the separate `braintools`/`saiunit` quantity compatibility
  failures.
- Do not add custom documentation CSS, JavaScript, Sphinx hooks, or static API
  pages.

## Verification

1. Search the repository for prohibited warning filters and executable
   `vmap_new_states` calls.
2. Run focused compiler, mapped-state, convolution batching, and public API
   tests.
3. Build the documentation with Sphinx warnings treated as errors.
4. Run the complete test suite and allow it to finish naturally.
5. Report dependency-related failures separately from regressions caused by
   this change.
