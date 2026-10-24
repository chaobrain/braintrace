# Issue #185: same-step gradient along paths to the output that bypass new hidden values

Issue: [#185](https://github.com/chaobrain/braintrace/issues/185)

## Problem

Under `vjp_method='single-step'` (the default), the gradient of every
`ParamState` is assembled from two sources:

- the eligibility trace: `dL/dh^t · e^t`, where `dL/dh^t` is read off the
  hidden perturbation `h^t = fresh + p` (`_compiler/hidden_pertubation.py`);
- the transposed step jaxpr (`dg_etrace_params` in
  `VjpAlgorithm._update_fn_bwd`), which **had to be empty**. The parameters
  are closed over as constants in
  `ETraceVjpGraphExecutor.solve_h2w_h2h_l2h_jacobian`
  (`etrace_param_vals = {}`), because `split_dict_states_v2` classes *every*
  `ParamState` as an eligibility-trace parameter.

A value can reach the step output **without passing through any new hidden
value**, and then its contribution appears in neither source. It is dropped
silently: no error, no warning. Two terms go missing:

1. **Parameter bypass**: `∂L^t/∂w` along such a path (a plain readout weight
   applied inside the step, a weight feeding a pre-reset value).
2. **Hidden bypass**: `∂L^t/∂h^{t-1}` along such a path, which RTRL pairs
   with the previous trace `e^{t-1} = ∂h^{t-1}/∂w`. In the issue's LIF,
   `spk = sg(0.9 V^{t-1} + x @ w - th)` is read from `V^{t-1}` directly. The
   perturbation at `V^t` cannot see that read.

Measured on `main` (D-RTRL, `etrace_grad`, |grad| summed):

| model | online | BPTT |
|---|---|---|
| issue LIF, output `sg(V_pre - th)` | 0.000000 | 1.079097 |
| plain readout inside the step, `tanh(V) @ w_out` (`w_out`) | 0.0 | 16.27 |

With only the parameter term, the LIF gives 0.234932. The hidden term
supplies the rest.

## Fix: add the exact same-step bypass terms

### Compile time (`_compiler/output_bypass.py`)

`add_output_bypass(eqns, outvars, sources=..., cut_vars=..., merge_positions=...)`
rewrites the perturbed step. The sources are keyed `('param', path)` (the
parameter leaf invars) and `('hidden', path)` (the hidden invars, i.e.
`h^{t-1}`).

1. **Cones.** Forward reachability from every source. Propagation stops at
   the hidden-state output slots (`cut_vars`, whose same-step credit the
   trace already carries) and at `stop_gradient`. A source whose cone
   reaches a merge slot has a bypass. The merge slots are the model outputs
   and the non-hidden, non-param state outputs; integer and boolean slots
   are skipped. An INFO diagnostic `OUTPUT_BYPASSES_HIDDEN` names the
   parameters and the hidden states.
2. **Duplication.** Each bypass source gets one new invar per leaf, a
   differentiable copy. Every equation of the copies' cone is duplicated.
   A duplicate consumes the copies (or earlier duplicates) for cone inputs,
   and `stop_gradient` of the original variable for every other input. The
   original equations are untouched, so values, the hidden perturbation and
   every other gradient stay exactly as before.
3. **Merge.** At each merge slot the duplicate `d` is attached as a pure
   tangent: `select_n(is_finite(d), o, o + (d - stop_gradient(d)))`. The
   forward value is `o`, and an overflowing output stays `inf` rather than
   becoming `inf - inf = nan`.
4. **Unfixable remainder: WARNING.** A cone equation that cannot be
   duplicated is cut: a `while` anywhere inside it (no reverse rule) or an
   equation with effects. `OUTPUT_BYPASS_DROPPED` names the sources.

`HiddenPerturbation` gains `bypass_param_paths` / `bypass_vars`,
`bypass_hidden_paths` / `bypass_hidden_vars`, and
`eval_jaxpr(inputs, perturb_data, bypass_data=(), bypass_hidden_data=())`.
The new invars follow the perturbation invars. With no bypass, the
perturbed jaxpr is exactly as before.

### Executor (`_algorithm/vjp_graph_executor.py`)

In single-step mode, `etrace_param_vals` holds the bypass parameters and
feeds the copies only (the real parameter values still come from the states
as constants). A new VJP primal `hidden_bypass_vals` holds `h^{t-1}` of the
bypass hidden states. The VJP therefore returns the parameter bypass
gradient in `dg_etrace_params` and the hidden bypass signal in the new
cotangent slot.

### Backward (`_algorithm/vjp_base.py`)

- The single-step guard now requires `dg_etrace_params` to carry exactly the
  compiled bypass parameters (anything else still raises). They pass on as
  `dl_to_etws_at_t`, which every `_solve_weight_gradients` already adds.
- `_update_fn_fwd` keeps `e^{t-1}` in the residuals when there is a hidden
  bypass. `_update_fn_bwd` turns the hidden bypass cotangents into per-group
  arrays (zeros for group members without a bypass). It passes them through
  `_compute_learning_signal` (so `random_feedback` projects them too) and
  contracts them with `e^{t-1}` through a new pure hook,
  `_solve_previous_trace_gradients`. The hook runs *before* the main
  contraction, which may advance the ParamDim trace filters.
- The hook is implemented for each engine:
  - ParamDim: the previous *filtered* trace when filters are active;
  - IO-dim: the warm-up bias correction is indexed by `running_index - 1`;
  - random projection: the same contraction as the main term.
- `learning_signal='modulatory'` replaces `dL/dh` entirely, so the graph is
  compiled without the rewrite (`compile_etrace_graph(include_output_bypass=False)`).
- Bootstrapped DNI: the second (future) pass ignores the bypass slots, like
  the perturbation slots. The learning signal comes from pass 1 only.

Multi-step mode differentiates the parameters and the hidden values directly
and already includes both terms. It builds no perturbation and is unchanged.

## Consequences

- A same-step read of a pure hidden copy (`h2^t = h1^{t-1}`, #184) is now
  credited exactly, as the hidden bypass of `h1^{t-1}`: the same value with
  the same trace. The `HIDDEN_COPY_SAME_STEP_READ` warning added in #190 is
  removed.
- A loss that reads a descended scan's stacked ys instead of its carry
  (`scan_descent.py` note) now gets the same gradient as a carry read. The
  limitation pin is changed to an equality test.

## Edge cases

- Output reads the new hidden value: no bypass, jaxpr unchanged.
- Mixed output (bypass + hidden path in one output): both terms, no double
  counting (the cone is cut at hidden slots).
- Two hidden states in one group, only one with a bypass: zero-filled.
- Weight feeds a second ETP op on the readout path (`W2(tanh(W1 x))`): the
  cone runs through both primitives, and both get bypass gradients.
- A parameter applied directly to the output (`... + b`).
- A pytree parameter (`Linear` weight + bias).
- Integer or boolean outputs: not merged.
- Output overflow (`inf`): the forward value stays `inf`, not NaN.
- `while` on the bypass path: dropped with a WARNING.
- A non-hidden state written from a weight (`ShortTermState`): a merge slot,
  so its same-step credit is exact. Recursion *through* such a state carries
  no trace (it is not a hidden state), as before.
- vmap / per-lane learners.

## Tests

- `_compiler/output_bypass_test.py`:
  - detection: the pre-reset output has a parameter and a hidden bypass; a
    readout from the new hidden value has none, and its jaxpr is unchanged;
    a plain readout has a parameter bypass only; `include_output_bypass=False`;
  - `while` on the cone gives one WARNING naming both sources;
  - forward values unchanged; gradients w.r.t. the copies equal the
    hand-written partials; `inf` stays `inf`; length checks;
  - `add_output_bypass` on hand-written jaxprs: `stop_gradient`, integer
    outputs, cut vars, merge positions.
- `_algorithm/tests/output_bypass_test.py`:
  - D-RTRL and EProp (exact on these element-wise cells) equal BPTT through
    `etrace_grad` for:
    - the pre-reset LIF and the mixed output;
    - a two-state group;
    - the pure-copy same-step read;
    - the plain readout and the readout variants (ETP `Linear`, direct
      parameter, ETP chain, non-hidden state);
  - the issue's numbers;
  - multi-step D-RTRL equals BPTT through the finite-window oracle (chunk 1
    and 4);
  - ThreeFactor compiles without a bypass;
  - pp_prop and UORO get both bypasses; their hooks equal the main
    contraction of the previous trace; the ParamDim hook reads the filtered
    trace without advancing it;
  - vmapped lanes equal per-sample BPTT.
- `_algorithm/tests/scan_descent_support_test.py`: the ys readout equals the
  carry readout.
