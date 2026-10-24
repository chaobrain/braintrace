# Issue #187: actionable error for `ParamState` writes inside the online step

Issue: [#187](https://github.com/chaobrain/braintrace/issues/187)

## Problem

`braintrace.compile` (through `abstractify_model`) wraps the model in
`_compiler/module_info.py::_model_that_not_allow_param_assign`, which raises
`NotSupportedError` when any `brainstate.ParamState` is written while the step
is traced. The rule is intended: the eligibility-trace recursion assumes every
parameter is constant within one step, and the trained weights are updated
outside the step from the gradients `etrace_grad` returns. The error message,
though:

- prints the state's repr (`ParamState(f32[3,4])`), not its path in the model;
- reports only the first written parameter;
- does not say that the rule also covers parameters that are *not* trained
  online (e.g. a weight updated by a local plasticity rule);
- does not point at a supported pattern.

The limitations page (`docs/advanced/limitations.ipynb`) does not mention the
rule.

## Fix

1. `_model_that_not_allow_param_assign` collects **every** written
   `ParamState`, resolves each to its path through
   `brainstate.graph.states(model)` (falling back to the repr for a state that
   is not reachable from the model, e.g. a global), and raises one
   `NotSupportedError` that lists them and gives the two supported patterns:

   - trained weights: update them outside the step, from the gradients returned
     by `etrace_grad` (an optimizer step);
   - locally-updated weights (plasticity): hold them in a non-param state such
     as `brainstate.LongTermState`; they are then not trained online.

2. A new section in `docs/advanced/limitations.ipynb` states the rule with the
   failing and the working pattern.

Behaviour is unchanged: the same exception type is raised in the same cases.

## Edge cases

- Several written parameters: all listed, in trace order, deduplicated.
- A parameter of a nested sub-module: the full path (`('layer', 'w')`).
- A `ParamState` that is read but not written: no error.
- A `ParamState` not attached to the model: listed by repr.
- `LongTermState` written in the step: compiles.

## Tests (`_compiler/module_info_test.py`)

- message names the path and both patterns;
- two written params are both named;
- nested sub-module path;
- unattached global `ParamState` falls back to repr;
- `LongTermState` variant compiles.
