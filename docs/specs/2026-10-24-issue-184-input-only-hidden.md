# Issue #184: `KeyError` for an input-only hidden state inside a merged group

Issue: [#184](https://github.com/chaobrain/braintrace/issues/184)

## Problem

A `HiddenState` whose new value depends only on inputs/weights (an input latch
`latch <- x @ w`) and whose *old* value is read by another hidden state
(`V <- 0.9 V + latch`) compiles, but the first online step raises
`KeyError: Var(...)` in `HiddenGroup.transition`.

## Root cause

`_compiler/hidden_group.py::_post_check` builds each group from
`[outvar(transition.hidden_invar)] + transition.connected_hidden_outvars`.
The transition starting at `latch^{t-1}` reaches `V^t`, so `latch` and `V`
merge into one group whose outvars include `latch^t`.
`write_jaxpr_of_hidden_group_transition` collects only the equations of the
member transitions, i.e. those reachable from the group invars, and derives the
constvars from their inputs. No collected equation produces `latch^t`
(it is computed from `x` and `w` only), and it is not a constvar either. The
group jaxpr thus has an outvar that is never bound, and `eval_jaxpr` raises.

The zero-recurrence fallback ("fourth-b step") handles the same situation for
a hidden state that sits in no group, by binding the outvar as a constvar. It
does not apply here because the merge has already covered `latch`.

## Fix

In `write_jaxpr_of_hidden_group_transition`, after deriving `other_invars`,
append every hidden outvar of the group that the collected equations do not
produce and that is not one of the group's hidden invars, in group order.
`HiddenGroup.transition_jaxpr_constvars` (read off the jaxpr by
`jaxpr_constvars`) then includes it, the graph executor binds its forward value,
and its rows of the hidden-to-hidden Jacobian are exactly zero. This is the
mathematically correct `d latch^t / d h^{t-1} = 0`.

A hidden outvar that *is* one of the group's invars is already bound as an
invar and gives the identity row; it is left alone.

## Related case found while fixing: pure copies (`h2^t = h1^{t-1}`)

A shift register / delay line writes a hidden state as a pure copy of another
hidden state's previous value. There is no equation, so no transition records
the edge. `h2` then falls into a zero-recurrence fallback group, which drops
`d h2^t / d h1^{t-1} = 1`, and the online gradient of the weight feeding `h1`
is silently zero:

| model (D-RTRL, `etrace_grad`) | main | BPTT |
|---|---|---|
| delay line, copy read on the next step | `KeyError` (this issue) | 0.584477 |
| shift register, copy read in the same step | 0.0 | 1.231248 |

The `new + 0 * old` workaround used downstream hides this case as well, so it
is fixed here too.

- `_post_check` merges each pure copy `h2` (outvar is `h1`'s invar) into
  `h1`'s group. The group transition maps that invar straight to `h2^t`,
  which gives the identity row. A delay line read on the next step then trains
  exactly (matches BPTT).
- A *same-step* read of the copy (e.g. `return tanh(self.h2.value)` right
  after `self.h2.value = old_h1`) cannot be attributed to `h2^t` at all: in
  the jaxpr `h2^t` and `h1^{t-1}` are one variable. Such a read carries no
  online learning signal. `_warn_copy_read_in_same_step` detects it by
  forward reachability from the copy variable to a model output or a
  non-hidden state output, stopping at new hidden values. It emits a WARNING
  `HIDDEN_COPY_SAME_STEP_READ` that names both states and suggests reading the
  copy on the next step or computing the output from a new hidden value.

## Edge cases

- `latch <- x` (no weight): the outvar is a top-level jaxpr invar.
- `h2 <- h1^{t-1}` shift register: outvar is a group invar (unchanged path).
- Two input-only latches feeding one state.
- Constvar order is deterministic (group outvar order).
- Pure copy read on the next step: merged, exact.
- Pure copy read in the same step: WARNING, no silent change.
- A copy inside a descended scan is left to the scan-body analysis.

## Tests

- `_compiler/hidden_group_test.py`: the group contains both states, the
  transition evaluates, the latch outvar is a constvar, and the diagonal
  Jacobian equals `[[0, 0], [1, 0.9]]` per position. Plus the edge cases above.
- `_algorithm/d_rtrl_test.py`: the issue model trains under `d_rtrl` and
  `eprop`. Its gradients equal those of the `new + 0 * old` workaround, and
  D-RTRL equals BPTT through the finite-window oracle
  (`chunked_online_param_gradients`, chunk size 1 and 3). The same checks are
  run for the delay-line model.
- `_compiler/hidden_group_test.py::TestHiddenCopySameStepRead`: the warning
  fires for a same-step read and not for a next-step read.
