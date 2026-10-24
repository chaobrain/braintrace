# Issue #186: weight-free `while` whose result reaches a hidden state indirectly

Issue: [#186](https://github.com/chaobrain/braintrace/issues/186)

## Problem

Under the default `ControlFlowPolicy(while_hidden='opaque-fwd')` a weight-free
`jax.lax.while_loop` that produces a hidden state is supported. The loop is
kept as an opaque forward node: its hidden-to-hidden Jacobian is taken in
forward mode, and its inputs are detached in the perturbed jaxpr so that
`jax.vjp` of the step never has to transpose the `while`.

The detach fires only when one of the loop's *own* outvars is a hidden outvar.
When the result passes through further equations before it is written
(`V <- while(...)(V) + I`, `- stop_gradient(spk) * th`, ...), the loop is left
in place, and `etrace_grad` fails with JAX's "Reverse-mode differentiation
does not work for lax.while_loop" `ValueError`. The same happens for a loop
fed only by the input whose result reaches a hidden state
(`V <- 0.9 V + matmul(while(...)(x), w)`). This was checked on `main`.

## Fix

`JaxprEvalForHiddenPerturbation.compile` computes, in one backward pass over
the top-level equations (`_while_eqns_reaching`), the set of `while` equations
whose outputs reach any hidden outvar. `_eval_while` detaches exactly those.
The direct case is a subset, so its behaviour is unchanged. The detach
emits the existing WARNING `CONTROL_FLOW_OPAQUE_FWD` (`site='perturbation'`).

The hidden-to-hidden Jacobian already switches to forward mode for any group
transition that contains a `while` (`hidden_group._transition_contains_while`),
so no change is needed there.

A `while` that reaches **no** hidden state (e.g. on the readout path only) is
still left in place. Detaching it would silently zero `dL/dh` for the hidden
states it reads. The existing loud `ValueError` is preferable there.

## Semantics

The same as the direct case: the loop's own hidden group keeps an exact
learning signal (the perturbation add is outside the detach), and its temporal
credit flows through the forward-mode `D^t`. A same-step reverse path *through*
the loop (e.g. to an upstream layer feeding the loop) is zero, as documented
for the direct case.

## Tests

- `_compiler/hidden_pertubation_test.py`:
  - an indirectly-reaching loop (`while(h) + x`) is detached, and the VJP of
    the perturbed step is traceable;
  - an input-fed loop reaching the hidden state is detached;
  - a readout-only loop is untouched;
  - one WARNING per detached loop.
- `_algorithm/tests/while_support_test.py`:
  - the issue's `while_leak` cell under D-RTRL equals its Python-unrolled twin
    (`V = 0.9**K V + I`) element-wise, both through `etrace_grad` and the
    single-step oracle;
  - the input-fed loop equals its twin;
  - the twin itself equals BPTT (finite-window oracle), which anchors the
    comparison to ground truth;
  - a readout-only loop still raises.
