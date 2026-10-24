# Issue #185: same-step gradient of parameters whose path to the output bypasses new hidden values

Issue: [#185](https://github.com/chaobrain/braintrace/issues/185)

## Problem

Under `vjp_method='single-step'` (the default), the gradient of every
`ParamState` is assembled from two sources:

- the eligibility trace: `dL/dh^t · e^t`, where `dL/dh^t` is read off the
  hidden perturbation `h^t = fresh + p` (`_compiler/hidden_pertubation.py`);
- the transposed step jaxpr (`dg_etrace_params` in
  `VjpAlgorithm._update_fn_bwd`), which **must be empty**. The parameters are
  closed over as constants in `ETraceVjpGraphExecutor.solve_h2w_h2h_l2h_jacobian`
  (`etrace_param_vals = {}`), because `split_dict_states_v2` classes *every*
  `ParamState` as an eligibility-trace parameter.

A parameter can also reach the step output **without passing through any new
hidden value**. Its same-step contribution then appears in neither source and
is dropped silently: no error, no warning. Measured on `main` (D-RTRL,
`etrace_grad`, |grad| summed):

| model | online | BPTT |
|---|---|---|
| issue LIF, output `sg(V_pre - th)` (ETP weight through a pre-reset value) | 0.000000 | 1.079097 |
| plain readout inside the step, `tanh(V) @ w_out` (`w_out`) | 0.0 | 16.27 |

The second row is the same defect, and probably the more common one: any
readout weight applied inside the compiled step gets no gradient.

## Fix: add the exact same-step bypass term

For a parameter `w` the true gradient decomposes, per step, into paths that
cross a new hidden value (handled by the trace, unchanged) and paths that do
not, the *bypass*. The bypass contribution is a plain same-step reverse-mode
term with no temporal component, so it is exact for every algorithm, exact or
approximate, and is added the same way for all of them.

1. **Cone detection (compile time, perturbation pass).** Forward reachability
   from every parameter leaf invar through the perturbed step. Propagation stops
   at hidden-state output slots and at `stop_gradient`. A parameter whose cone
   reaches a model output, or a non-hidden, non-param state output, is a
   *bypass parameter*. An INFO diagnostic `OUTPUT_BYPASSES_HIDDEN` names them.
2. **Cone duplication (compile time).** For each bypass parameter the perturbed
   jaxpr gets one extra invar per leaf (a differentiable copy of the value).
   Every equation in the copies' cone is duplicated, consuming the copies
   (or earlier duplicates) for cone inputs and `stop_gradient` of the original
   variable for every other input. The original equations are kept unchanged,
   so values, the hidden perturbation and every other gradient are untouched.
   At each merge slot (model outputs and non-hidden state outputs) the
   duplicate `d` is attached as a pure tangent:
   `out = select_n(is_finite(d), out, out + (d - stop_gradient(d)))`.
   The forward value is bitwise `out`, and the `select_n` guard keeps an
   overflowing output from turning into NaN. Hidden slots and the extra
   Jacobian temporaries are never merged.
3. **Executor.** In single-step mode, `etrace_param_vals` holds the bypass
   parameters (instead of `{}`) and is fed to the copies only. The real
   parameter values still come from the states, as constants. The VJP
   therefore returns exactly the bypass gradient in `dg_etrace_params`.
4. **Backward.** `_update_fn_bwd` accepts `dg_etrace_params` whose keys are the
   compiled bypass parameters (anything else still raises) and passes them as
   `dl_to_etws_at_t`. Every `_solve_weight_gradients` already adds this term
   (`param_dim_vjp`, `io_dim_vjp`, `random_projection_vjp`). For parameters
   with no ETP relation it becomes their only gradient. Bootstrapped DNI is
   unaffected: the future cotangent enters at hidden exits, which the cone
   never crosses.
5. **Unfixable remainder: WARNING.** A cone equation that cannot be
   duplicated is cut, with a WARNING (`OUTPUT_BYPASS_DROPPED`) naming the
   parameters. That covers a `while` (not reverse-differentiable) or an
   equation with effects (duplicating it would repeat the effect).

No change when there is no bypass: the perturbed jaxpr and the executor are
exactly as before. Multi-step mode differentiates the parameters directly
and already includes the bypass. A test checks that, and that path is not
changed.

## Edge cases

- Output reads the new hidden value: no bypass, jaxpr unchanged.
- Mixed output (bypass + hidden path in one output): both terms, no double
  counting (the cone is cut at hidden slots).
- Weight feeds another ETP primitive on the readout path (`W2(W1 x)` with no
  hidden state): the cone runs through `W2`'s primitive. The ETP JVP rules are
  used, and both get bypass gradients.
- Output is the parameter itself (`return w`).
- Pytree / multi-leaf parameter (weight + bias).
- Integer or boolean outputs: not merged.
- Output overflow (`inf`): the forward value stays `inf`, not NaN.
- `while` on the bypass path: dropped with a WARNING.
- Other (non-hidden) state written from a weight (`ShortTermState`): merged
  slot, its incoming cotangent is honoured.
- vmap / batched learners.

## Tests

- `_compiler/hidden_pertubation_test.py`:
  - detection: bypass and no-bypass;
  - structure: no extra invars without a bypass;
  - forward values bitwise unchanged;
  - the gradient w.r.t. the copies equals a hand-written bypass VJP;
  - `while` on the cone gives a WARNING;
  - `inf` output stays `inf`.
- `_algorithm/d_rtrl_test.py`, `_algorithm/e_prop_test.py`, `pp_prop_test.py`:
  - the issue's LIF and the plain-readout model: D-RTRL equals BPTT,
    through `etrace_grad` (issue repro path) and through the finite-window
    oracle with chunk size 1;
  - mixed output: no double count;
  - EProp / pp_prop: the bypass term is present and the algorithms keep
    their own trace terms. Checked against the same algorithm on the
    `new hidden value` twin model, via the finite-window oracle;
  - multi-step mode already equals BPTT on these models (regression pin);
  - batched (`batch_size=2`) learner.
- `_algorithm/vjp_base_test.py`: the guard still raises for unexpected ETP
  gradients under single-step.
