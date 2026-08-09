# pp-prop correctness and sparse readiness

## Objective

Make pp-prop safe to use without silent gradient loss, window-dependent bias,
trace-key collisions, or accidental dense scaling. Preserve the existing JAX
and brainstate architecture and public constructor signatures.

## Required behavior

- Single-step execution gives ETP parameters eligibility-trace gradients and
  plain parameters exact current-step gradients. Routing is exclusive for an
  entire ParamState path, including every pytree leaf. Compilation rejects
  every unrepresented differentiable occurrence, including mixed ETP/plain
  use, earlier ETP occurrences hidden behind later trainable ETP primitives,
  descended-scan boundaries, and trainable inputs derived from multiple leaves.
- `running_index` counts completed timesteps. Windowed and unwindowed execution
  produce the same trace age after processing the same sequence.
- The f-trace correction is stable for every valid decay and has no arbitrary
  timestep cutoff.
- Transformed x traces are relation-specific. Untransformed consumers of the
  same x value continue to share one trace.
- Every trainable parameter in a relation is queryable with `get_etrace_of`.
  Elementwise relations return an empty x-trace instead of failing.
- Public validation uses stable exceptions and rejects invalid, non-finite,
  boolean, or conflicting configuration values.
- pp-prop rejects hidden tails whose position preservation cannot be proved.
- pp-prop accepts only same-shape, position-preserving
  `element_wise(weight_fn=...)` transforms.
- pp-prop rejects parameter preprocessing outside an ETP primitive because its
  chain rule is not represented; equivalent primitive transform hooks remain
  supported.
- Every VJP eligibility-trace algorithm rejects a relation that reaches the
  same hidden state through both a direct path and an indirect path through
  another trainable ETP primitive. Raw relation discovery remains available
  for inspection, and independent direct ETP relations may share a ParamState.
- Sparse examples and supported installs use native CSR storage without dense
  masks or dense conversion.
- Any path that would materialize an oversized full hidden Jacobian fails at
  compile time before JAX lowering or allocation.

## Compatibility

- Keep `pp_prop`, `IODimVjpAlgorithm`, and constructor signatures available.
- Keep `running_index` public but change its meaning from calls to timesteps.
- Keep the existing recurrence-scope options for models within the configured
  Jacobian element ceiling.
- Keep automatic sparse backend dispatch. Platform extras install the matching
  brainevent backend dependencies, and `jax_raw` remains an explicit fallback.

## Verification

- Compare plain single-step gradients with a one-step reverse-mode oracle.
- Reject `element_wise(w) + 2 * w`, cross-leaf mixed ownership, and an ETP
  trainable input derived from two parameter leaves, including outer plain use
  of a parameter owned by an ETP relation inside a descended scan and repeated
  internal ETP uses whose earlier occurrences cannot become relations.
- Compare the final trace from one six-step roll with three two-step rolls,
  verify each solver sees the age of the trace it contracts, and check the
  exact warm-up factor at decays 0, 0.9, and 0.9999.
- Exercise shared raw inputs, two differently shaped embeddings, elementwise
  relations, and multi-parameter sparse or dense relations.
- Verify invalid configuration under normal and optimized Python execution.
- Verify pp-prop rejects reversal, reshape, and mixing element-wise transforms
  and external parameter scaling or masking while accepting ordinary scalar
  element-wise functions and equivalent primitive transform hooks.
- Verify `PartialPathRNN` is rejected by both pp-prop and D_RTRL, MGUCell and
  MinimalRNNCell fail closed for the same mixed-path structure, and a tied
  ParamState used by two independent direct ETP relations remains supported.
- Verify native CSR forward and pp-prop gradient in a clean CPU installation.
- Verify a 139,255-neuron, 1,000,000-edge graph has only O(N + nnz) persistent
  leaves and that oversized coupled recurrence is rejected before allocation.
- Run the complete source tests, example tests, type check, and package build.

## Non-goals

- Do not change connectome-agent, its PyTorch plasticity path, or checkpoint
  formats.
- Do not add a Torch/JAX bridge or submit an upstream pull request.
- Do not change pp-prop because of the algebraic Jacobian Conjecture; the
  algorithm uses local derivative products and assumes no global inverse.
