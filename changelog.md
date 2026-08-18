# Release Notes


## Version 0.2.6

A compatibility release: `braintrace` now works with **JAX / jaxlib 0.11**,
which merged `ClosedJaxpr` into `Jaxpr` and, in doing so, silently changed
what a compiler-built transition jaxpr reports about its own inputs. On 0.11.1
that broke 690 of the 2902 tests in the suite; all of them pass again.

Support for JAX 0.8 through 0.10 is unchanged — the fix derives the
information JAX stopped storing rather than branching on a version, so there is
one code path across every supported release. No public API changes.

### Fixes

- **Restored compatibility with JAX / jaxlib 0.11.** JAX 0.11 merged
  `ClosedJaxpr` into `Jaxpr`: a jaxpr now holds a single positional input list
  and derives the `constvars` / `invars` boundary from how many constant
  *values* are attached, instead of storing it alongside the symbols. The ETP
  compiler builds *transition jaxprs* — programs whose `invars` are the
  differentiated inputs (a hidden state at `t-1`, or an ETP primitive's output
  `y`) and whose `constvars` are surrounding intermediates bound from the
  forward pass at execution time. Those carry symbols but no attached values,
  so on 0.11 they began reporting `constvars == []` and folding the constvars
  into `invars`.

  Every consumer that recovered the split by reading `jaxpr.constvars` back
  therefore passed too few values to `jax.core.eval_jaxpr`, which failed
  arity checking with `ValueError: foreach() argument 2 is shorter than
  argument 1` the first time any algorithm evaluated a transition. Since that
  is the shared entry point for hidden→hidden and hidden→weight Jacobians, it
  took down essentially every gradient path: D-RTRL, ES-D-RTRL / pp_prop,
  EProp, OSTL, SnAp, UORO, DNI and all of the BPTT oracle cross-checks, across
  every ETP primitive family (dense, LoRA, sparse, convolutional,
  element-wise, embedding).

  The split is now derived from the invar count — which every caller already
  knows, since the invars are what it is about to feed in — via three helpers
  in `braintrace._compatible_imports`: `jaxpr_all_invars`,
  `split_jaxpr_invars` and `jaxpr_constvars`. `split_jaxpr_invars` range-checks
  its argument, so a miscounted call raises at the compiler boundary rather
  than producing a misaligned argument list. The full analysis is in
  `docs/specs/2026-08-18-jax-011-jaxpr-merge-compat.md`.

- **Fixed SnAp-n position analysis under JAX 0.11.** `build_snap_pattern` let
  `analyze_position_adjacency` seed its reachability walk from the transition
  jaxpr's own `invars`. Under the merged representation that set silently
  widened to include every constvar, so the derived neighbourhood was inflated
  — typically all the way to the conservative all-positions-couple fallback,
  which is correct but costs orders of magnitude in trace size. The group's
  `hidden_invars` are now passed explicitly.

### Internal

- `HiddenGroup.transition_jaxpr_constvars` is populated from the constvars the
  builder actually used rather than read back off the constructed jaxpr,
  matching how `Hidden2GroupTransition.other_invars` has always been handled.
  Const-var collection in the graph executor, the compiled-graph output
  registration and structured scan descent go through the new helpers.

- The `jax-version` CI matrix pins `0.11.0` alongside `0.8.0` / `0.9.0` /
  `0.10.0` and `latest`, so the merged-`Jaxpr` representation stays covered
  once `latest` moves past it.

- New regression coverage for the split itself: helper-level tests in
  `braintrace/_compatible_imports_test.py` (round-trip, zero- and
  all-constvar edges, out-of-range rejection) and compiler-level tests
  asserting that a compiled group's recorded constvars match its jaxpr's
  leading inputs and that the transition still evaluates.


## Version 0.2.5

> **This patch release removes public API.** Despite the patch version number,
> `OTTT`, `OSTTP`, `OTPE` and `PresynapticTrace` are gone and
> `IODimVjpAlgorithm.decay` is now read-only. A pin of `braintrace>=0.2,<0.3`
> will pick these changes up, so read the *Breaking changes* section below
> before upgrading.

It is also the largest feature release since 0.2.0: ten new public symbols,
headlined by the **sequence-driver API** (`etrace_grad` / `etrace_evolve`),
which removes the hand-written scan-and-accumulate loop from every call site,
and by five new learning rules (`SnAp`, `UORO`, `ThreeFactor`, `DNI`,
`RandomProjectionVjpAlgorithm`) expressed as coordinates in the new
`ETraceConfig` axis space rather than as bespoke implementations.

Closing the release, a nine-item hardening pass (E-01 … E-09) worked through
the backlog opened by the pre-release package audit. Its theme is that a
failure should be visible: four paths that could hang the compiler, mis-attribute
a gradient, or refuse a constructor argument several transforms too late now
raise where the mistake is, `jax` is a declared dependency instead of a borrowed
one, and the `mypy` typing gate covers the last two packages that were outside
it. Those items appear under *Correctness and robustness*, *Improvements*,
*Documentation* and *Internal* below.

### Breaking changes

- **Removed `OTTT`, `OSTTP` and `OTPE`.** `braintrace` is a framework for online
  learning in brain simulation, and a framework should ship *general*
  mechanisms. These three rules were not model-agnostic: all of them whitelisted
  dense-matmul primitives (`_SUPPORTED_PRIMITIVES = {etp_mm_p, etp_mv_p}`) and
  raised `NotImplementedError` for lora / sparse / convolutional /
  element-wise relations, and all of them were single-step only. `OTPE`
  additionally assumed a single global time constant, was feed-forward only,
  was gradient-exact for one hidden layer, and rejected `num_state > 1`
  outright — ruling out ALIF and any adaptation variable. `OSTTP` bound
  `B_list` to the HiddenGroup count and threaded `y_target` through a bespoke
  path.

  Their coordinates remain reachable: the planned axis decomposition
  (`trace_factorization` × `temporal_recursion` × `learning_signal` ×
  `trace_filter` × `update_schedule`) expresses each of them as a configuration
  that works for **every** ETP primitive, not just dense matmul. See
  `docs/specs/2026-07-25-algorithm-axes-roadmap.md`.

  Also removed with them: `PresynapticTrace` (used only by `OTTT`) and the
  internal `extract_y_target` helper (used only by `OSTTP`). `KappaFilter` and
  `FixedRandomFeedback` are unaffected. The `'ottt'`, `'osttp'` and `'otpe'`
  names no longer resolve in `braintrace.compile`.

  **Migration:** `OTTT` → `pp_prop` (same `io_factorized` trace, keeps the
  temporal term instead of dropping it); `OTPE` → `D_RTRL` or `pp_prop`;
  `OSTTP` → `EProp(feedback='random')`, which is random feedback on the error
  rather than on the target.

- **`IODimVjpAlgorithm.decay` is now a read-only property.** The x-side and
  f-side decays became independent (see `ETraceConfig` below), so a single
  `decay` attribute is only meaningful when the two agree. Reading it when they
  differ raises `AttributeError` naming `decay_x` / `decay_f`; assigning to it
  is no longer possible. `decay_or_rank=0.9` remains element-wise identical to
  the new `(0.9, 0.9)`.

- **`decay_or_rank=0.0` is now accepted** by `IODimVjpAlgorithm` (the bound
  relaxed from `0 < decay < 1` to `0 <= decay < 1`). Zero is the coordinate for
  "no temporal accumulation on this side" and canonicalises to
  `temporal_recursion='none'`; it was previously rejected as invalid input.

- **Renamed the private module `braintrace._state_managment` to
  `braintrace._state_management`** (the old name was missing an `e`). No
  deprecation shim and no `DeprecationWarning` were left behind: the module is
  private, none of its helpers (`assign_dict_state_values`,
  `assign_state_values_v2`, `sequence_split_state_values`,
  `split_dict_states_v2`) is re-exported from `braintrace`, and a shim would
  keep the misspelling importable — and therefore greppable and
  copy-pasteable — indefinitely. Public API is unaffected; only code that
  imported the private path directly needs to change:

  ```python
  # before
  from braintrace._state_managment import assign_state_values_v2
  # after
  from braintrace._state_management import assign_state_values_v2
  ```

  Resolves [#162](https://github.com/chaobrain/braintrace/issues/162); see
  `docs/specs/2026-08-07-e07-state-management-rename.md`.

### New features

- **Sequence drivers: `etrace_grad` and `etrace_evolve`.** Every algorithm now
  carries two methods (via the new `braintrace.SequenceDriverMixin`) that drive
  a whole sequence, so the scan-and-accumulate loop that every call site used to
  hand-write is gone. `etrace_grad` accumulates online gradients over a
  sequence; `etrace_evolve` advances hidden states and eligibility traces
  without computing any gradient.

  ```python
  # before -- hand-written, repeated in 19 example files
  grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
  def body(carry, xs_t):
      x, y = xs_t
      g, loss = brainstate.transform.grad(step, weights, return_value=True)(x, y)
      return jax.tree.map(jnp.add, carry, g), loss
  grads, losses = brainstate.transform.scan(body, grads, (xs, ys))

  # after
  grads, losses = learner.etrace_grad(xs, ys, step_fn=step, return_value=True)
  ```

  `step_fn` is keyword-only, so any number of sequences can be passed
  positionally and are sliced in lockstep. Supporting options: `mask` (weights
  the loss per step while still evolving the trace on masked steps),
  `chunk_size` (windowed drive), `weights`, `reduction` (`'mean'` over unmasked
  steps by default, or `'sum'`), `loss_output` (`'per_step'` / `'masked'` /
  `'scalar'`), `has_aux` and `return_value`. `chunk_size` and `vjp_method` are
  independent axes: chunking sets how many steps each window covers, the VJP
  method sets how the window is differentiated.

  `braintrace.compile(..., vmap=True)` now returns a `braintrace.ETraceVmap` —
  a `brainstate.nn.Vmap` subclass carrying the same two methods — so batched and
  unbatched call sites are identical. It remains a `brainstate.nn.Vmap` for
  `isinstance` purposes; only `type(x) is brainstate.nn.Vmap` changes. Note that
  reaching through `.module` is *not* equivalent:
  `learner.module.etrace_grad(...)` drives the unbatched learner and silently
  produces per-lane-wrong results.

  See `docs/specs/2026-07-27-sequence-driver-api.md`. All examples, tutorials
  and docstrings were migrated onto `compile` / `etrace_grad` / `etrace_evolve`;
  the remaining manual loops in `examples/` are BPTT baselines and benchmark
  instrumentation, each carrying an in-file `# kept manual:` rationale.

- **`SnAp` — sparse *n*-step approximation.** `recurrence_scope` generalises
  from the two-valued `'diagonal'` / `'coupled'` to an *n*-step neighbourhood:
  `SnAp(model, n=k)` keeps hidden→hidden influence out to `k` steps in the
  position graph and drops the rest, so `n=1` is the diagonal rule and larger
  `n` interpolates toward the fully coupled one. Reachable by name as
  `'snap'`, and the neighbourhood is computed from the compiled graph, so it
  works for every ETP primitive rather than dense matmul only.
  See `docs/specs/2026-07-25-p3-snap-n.md`.

- **`UORO` — unbiased online recurrent optimization.** A rank-1 random
  projection of the influence matrix, giving an *unbiased* (but higher-variance)
  gradient estimate at O(P) memory instead of D-RTRL's O(P·H). Built on the new
  `RandomProjectionVjpAlgorithm` engine, which is also public. Reachable as
  `'uoro'`.

- **`ThreeFactor` — modulated learning.** `learning_signal='modulatory'`
  replaces the backpropagated error with an externally supplied scalar (or
  per-group) modulator, the neuromodulation-style third factor. Requires
  `vjp_method='single-step'`, which is enforced at construction rather than
  discovered at run time. Reachable as `'three_factor'`.

- **`DNI` — decoupled neural interfaces / synthetic gradients.** `DNI` learns a
  synthesiser `M(h)` that predicts the future loss gradient `dL_{>=t}/dh_t`,
  removing the dependence on a full backward pass. Ships with
  `SyntheticGradient` (the synthesiser module, sized from the compiled graph via
  `algo.group_signal_shapes()`) and `train_synthetic_gradient` (a training
  helper driven by the learner's own returned hidden cotangent, so the
  regression target is exact rather than bootstrapped). Reachable as `'dni'`.
  See `docs/specs/2026-07-25-p4-uoro-modulatory-dni.md`.

- **`ETraceConfig` — learning rules as explicit axis coordinates.** The new
  `braintrace.ETraceConfig` describes a learning rule as a point in a six-axis
  space (`trace_factorization`, `temporal_recursion`, `recurrence_scope`,
  `learning_signal`, `trace_filter`, `update_schedule`) plus the coefficients
  those axes need (`decay`, `kappa`, `feedback_scale`, `sparsity`). Illegal
  combinations are rejected at construction with an error naming the legal
  pairings, and coordinates that mean the same rule are canonicalised to one
  form (e.g. a zero decay collapses to `temporal_recursion='none'`).

  `braintrace.compile` accepts a config wherever it accepts an algorithm name,
  so a rule with no name is as constructible as one with a name:

  ```python
  # an x-side leak with an instantaneous f-side
  learner = braintrace.compile(
      model,
      braintrace.ETraceConfig(trace_factorization='io_factorized',
                              temporal_recursion=('scalar_leak', 'none'),
                              decay=(0.9, 0.0)),
      x0,
  )
  ```

  The named algorithms are now thin factories over coordinates rather than
  separate implementations: `D_RTRL`, `pp_prop`, `EProp`, `OSTLRecurrent` and
  `OSTLFeedforward` all construct a config and delegate. Their gradients are
  unchanged — the migration is pinned by 24 frozen golden gradients spanning
  all three trace paths (chunked, fused multi-step, single-step).

- **`temporal_recursion` works for every ETP primitive.** The recursion is
  realised by substituting the executor's per-hidden-group hidden→hidden
  Jacobian (`λ·I` for `scalar_leak`, zeros for `none`) rather than by
  special-casing operators, so it applies to dense, conv, sparse, lora and
  element-wise relations alike. The removed `OTTT`'s coordinate — x-side leak,
  f-side instantaneous — is reachable again this way, and is now
  primitive-generic, which the deleted implementation never was.

- **Random feedback and the κ-filter are no longer `EProp`-only.**
  `learning_signal='random_feedback'` and `trace_filter='kappa'` moved onto the
  base engine, so random feedback now composes with the O(I+O) `io_factorized`
  trace as well as the O(P·H) one. A configuration that cannot be honoured
  (random feedback requested but no feedback matrices allocated) now raises
  rather than silently computing the symmetric rule.

- **`recurrence_scope` is a public axis.** What was the private
  `_include_recurrent_mixing` class attribute is now `recurrence_scope`, and
  asking for a scope wider than the model supports — e.g. `'coupled'` on a model
  whose compilation descends into a `scan` — raises instead of silently
  degrading to `'diagonal'`. Beyond the original two values it accepts an
  integer *n*-step neighbourhood; see `SnAp` above.

- **New algorithm names in `braintrace.compile`.** `'snap'`, `'uoro'`,
  `'three_factor'` and `'dni'` now resolve, alongside the existing `'d_rtrl'`,
  `'pp_prop'`, `'e_prop'` and the OSTL names. (`'ottt'`, `'osttp'` and `'otpe'`
  no longer do — see *Breaking changes*.)

- **`braintrace.nn.CFNCell` is exported.** The Chaos-Free Network cell
  (`arXiv:1612.06212`) was implemented but never added to `braintrace.nn`'s
  `__all__`, so the call its own docstring advertised raised `AttributeError`.
  Exporting it surfaced a second defect, fixed here: the constructor sized the
  input projection `out_size -> out_size`, but `update()` feeds it the input, so
  every forward pass with `in_size != out_size` raised a `dot_general` shape
  error. It is now sized `in_size -> out_size`, matching the paper's
  `h_t = theta_t * phi(h_{t-1}) + eta_t * phi(W x_t)`.

- **`braintrace.nn.__dir__`** now advertises the ~50 names that
  `__getattr__` forwards to `brainstate.nn` / `brainpy.state`. They always
  resolved, but were invisible to `dir()` and tab-completion, unlike the
  top-level `braintrace` namespace which already had this.

### Correctness and robustness

- **The hidden↔gradient correspondence is checked, not asserted**
  ([#165](https://github.com/chaobrain/braintrace/issues/165)). Three
  `assert` statements in `_algorithm/vjp_base.py` were meant to guarantee that
  the backward pass's cotangents line up with the compiled hidden states. They
  failed at that twice over: `python -O` strips an `assert`, so on an optimised
  interpreter nothing checked anything; and even enabled, they compared
  cardinalities and one key set, never that cotangent *i* belongs to hidden
  state *i*. A misattributed cotangent yields a wrong gradient, not an error.

  Every guard is now an explicit `raise`.
  `vjp_base._check_hidden_gradient_correspondence` verifies totality, absence
  of strays, and per-index shape/dtype agreement, and is called from both
  branches. `HiddenGroup.concat_hidden` — whose `zip` against
  `self.hidden_states` silently truncated a short value list into a
  too-narrow slab that surfaced later in unrelated trace math, or never —
  raises unless it receives exactly one value per hidden state;
  `split_hidden` gets the mirror guard on its trailing-axis width.
  `HiddenPerturbation.perturb_data_to_hidden_group_data` raises on a length
  disagreement and names the group and the missing path instead of a bare
  `KeyError`. All comparisons are on Python-level metadata, so nothing is
  added to the traced graph. See
  `docs/specs/2026-08-07-e01-hidden-gradient-correspondence.md`.

- **Control-flow canonicalization can no longer hang the compiler**
  ([#157](https://github.com/chaobrain/braintrace/issues/157)). The three
  fixpoint loops in `braintrace/_compiler/canonicalize.py` — cond
  if-conversion, inner-scan unrolling, and the joint driver — were
  `while True:` loops whose termination rested on an unenforced assumption
  that control-flow nesting is finite. A jaxpr that regenerated convertible
  `cond`/`scan` equations as fast as the sweeps consumed them spun forever,
  with no output and no way to distinguish it from a slow compile.

  All three are now bounded by the new
  `ControlFlowPolicy.fixpoint_iteration_limit` (default 64, must be a
  positive integer — there is no "unbounded" setting). Exhausting it raises
  `braintrace.CompilationError` naming the equations the last sweep was
  still rewriting (primitive, branch count or scan length, and source
  location) and pointing at the remedies: raise the limit if the nesting is
  genuine, or turn the offending pass off with
  `ControlFlowPolicy(cond='opaque')` / `ControlFlowPolicy(scan_unroll_limit=0)`.

  The limit bounds loop *iterations*, not the size of any single rewrite —
  that remains `scan_unroll_limit`. Compiles that converged before converge
  identically now.

- **`braintrace.nn.Embedding` rejects its unsupported options at construction**
  ([#159](https://github.com/chaobrain/braintrace/issues/159)). `max_norm`,
  `freeze`, `scale_grad_by_freq` and `padding_idx` were accepted by the
  inherited `__init__` and refused only by `update()` — which, under `jit`, is
  a trace arbitrarily far from the line that passed the option, so the
  traceback pointed at the transform rather than at the mistake. `__init__`
  now forwards every argument to the parent unchanged and then validates, so
  the error is raised from the constructor call; running validation *after*
  `super().__init__` keeps the parent's more specific diagnoses (an
  out-of-range `padding_idx` is still its `ValueError`). The `update()` check
  is kept rather than deleted, because the four options are plain public
  attributes that can be set after construction, and the message now names
  only the options actually passed. The class docstring no longer inherits
  the parent's text, which documented two of the four as working and
  demonstrated them in examples that raise.

- **Canonicalization warning dedup no longer rests on an undocumented
  invariant** ([#158](https://github.com/chaobrain/braintrace/issues/158)).
  Skip diagnostics were suppressed with `skip_warned` sets keyed on
  `id(eqn)`, sound only because the enclosing jaxpr happened to keep every
  equation object alive across sweeps. Keying on the equation's index is no
  better (each sweep rebuilds the list, so one rewrite shifts every later
  index), and content keys over-suppress. The key is gone instead: each sweep
  buffers its diagnostics and returns them, and the fixpoint driver emits only
  the buffer from the settling sweep — which rewrote nothing, so it visited
  each surviving equation exactly once and holds exactly one entry per skip.

### Improvements

- **JAX 0.11 compatibility.** Ported the scan handling onto JAX's flattree
  representation. CI now runs the suite against JAX 0.8, 0.9, 0.10 and latest.
- **`jax>=0.8.0` is now a declared dependency**
  ([#160](https://github.com/chaobrain/braintrace/issues/160)). Every module
  imports `jax` directly, but `[project].dependencies` never said so. The floor
  a resolver saw was not absent, it was *borrowed*: `brainstate` declares `jax`
  only under extras, so the constraint that made installs work came from
  `brainevent`'s metadata, and would have moved without a braintrace commit.
  The floor is `0.8.0` because that is the lowest entry in the CI matrix, and
  it is deliberately uncapped — a cap published today would constrain JAX
  releases that do not exist yet, for every artifact already on PyPI, while the
  daily scheduled run against unpinned `jax` surfaces a breaking release within
  24 hours. No install that works today gains a constraint; ownership of the
  floor simply moves to the matrix that tests it. The accelerator extras are
  unaffected (`braintrace[cuda12]` still resolves one `jax` satisfying
  `>=0.8.0` *with* the cuda12 extra, verified against the built wheel with
  `pip install --dry-run --report`), and they stay unversioned so the floor
  lives in exactly one place. The previously untested `braintrace/_version.py`
  gained a co-located test pinning the declared floor to `min(CI matrix)`, the
  no-cap rule, and the `requirements.txt` sync.
- **Conv bias IO-dim fix**, plus an axis-aware verification harness and an
  in-tree limitation list (`docs/specs/2026-07-25-known-limitations.md`), which
  is now the tracked backlog of known approximation edges.
- **`train_synthetic_gradient`'s window loop is compiled** rather than traced
  once per window, via `brainstate.transform.for_loop`.
- **The wheel no longer ships the test suite.** This project co-locates tests
  (`foo.py` / `foo_test.py`), and `setuptools`' `packages.find` `exclude`
  matches package names, not loose modules — so all 76 `*_test.py` files, the
  `_algorithm/tests/` and `_compiler/tests/` subpackages, and the fixture
  modules were being copied into every install: 1.63 MB of a 2.90 MB wheel,
  56%. The fixture cluster (reference models, the BPTT oracle, the compiler
  scenario catalog) moved to a new `braintrace._testing` package, excluded by
  name, and a `build_py` subclass in `setup.py` drops `*_test.py`. The wheel is
  now **1.24 MB across 68 files**, down from 2.90 MB across 148.

  This also removed a layering violation: `_algorithm/oracle_models.py` (shipped
  code) imported layer classes from `braintrace/_etrace_model_test.py` (a
  pytest-collected module that contained no tests). Both now live in
  `braintrace._testing`, on the same side of the ship/no-ship line, and the
  mypy `ignore_errors` special case that existed only to paper over that import
  is gone.

  **The sdist keeps the full test payload**, because that is what downstream
  packagers build and run the suite from. Splitting the two artifacts is why
  the pruning lives in `setup.py` rather than in `packages.find`: exclusion at
  discovery time happens before either artifact exists and so hits both. The
  wheel filter is applied in `build_py.find_package_modules` and switched off
  for `get_source_files`, the list `sdist` builds its manifest from. This only
  holds with `include-package-data = false` (now set): left on, setuptools
  re-adds every file in the sdist manifest to the wheel as *package data*,
  which put the whole payload back. Verified end-to-end — the extracted sdist
  runs its own suite in a clean virtualenv, `braintrace._testing` included.
- **The "every public API is typed" mypy gate now actually covers them.** Twelve
  modules owning names in `braintrace.__all__` were missing from the
  `disallow_untyped_defs` list — including `_algorithm/sequence.py`, which owns
  `etrace_grad` / `etrace_evolve`. They are now listed and annotated. Three dead
  entries naming the removed `otpe` / `ottt` / `osttp` modules were deleted;
  mypy ignores unmatched module patterns silently, so they never errored.
- **`braintrace._op` honours its facade claim.** The module docstring promised
  that every name exported from the underlying registries is available on the
  package, but seven were not — which is why `_algorithm/param_dim_vjp.py` had
  to reach past the facade into `braintrace._op._registries`. All seven
  (`BATCHED_COUNTERPARTS`, `register_batched_counterpart`,
  `get_batched_counterpart`, `ETP_RULES_INSTANT_DRTRL`, `ETP_RULES_SOLVE_DRTRL`,
  `get_instant_drtrl_rule`, `get_solve_drtrl_rule`) are now re-exported.
- **`braintrace._compiler` declares an `__all__`**, the last package facade in
  the tree without one.
- **Public surfaces no longer recommend `jax.random`.** The
  `random_feedback_key` error message, the `FixedRandomFeedback` doctest and
  `EProp`'s parameter documentation all told users to build keys with
  `jax.random.PRNGKey`, against the project's own rule to use
  `brainstate.random`. They now point at `brainstate.random.split_key()`.
- **A bad algorithm name gives a clean error.** `braintrace.compile(model,
  'nosuch')` raised its actionable `ValueError` chained onto the internal
  `KeyError`, burying the message under "During handling of the above
  exception". The chain is now suppressed.
- **`compile(..., vmap=True)` validates `example_inputs`.** A scalar or
  wrong-batch leaf previously produced a raw `TypeError` from `a[0]` naming
  neither `compile` nor the offending leaf.

- **`sparse_matmul` migrates off brainevent's deprecated trace protocol.**
  `braintrace._op.sparse` now calls `brainevent.DataRepresentation.dt2t` /
  `.dt2t_transposed` directly instead of the deprecated `.yw_to_w` /
  `.yw_to_w_transposed` aliases (brainevent renamed its own trace-propagation
  protocol to match braintrace's `DT_TO_T` terminology). The minimum
  `brainevent` version is raised to **0.1.2** (the release that introduces
  `dt2t`/`dt2t_transposed`) in `pyproject.toml` / `requirements.txt`.
- **Fixed a JAX-internal `linear_util.wrap_init` `DeprecationWarning`** raised
  by the single-step VJP residual construction in `vjp_graph_executor.py`.
  The call now threads a `DebugInfo` object through the ecosystem-standard
  `brainstate._compatible_import.wrap_init` shim (re-exported as
  `braintrace._compatible_imports.wrap_init`), matching the pattern already
  used elsewhere in the `brainstate`/`saiunit` stack. No behavior change.
- **Fixed stale API references in the documentation notebooks.** Three
  notebooks (`docs/advanced/etp_primitives.ipynb`,
  `docs/advanced/customizing_primitive_transforms.ipynb`,
  `docs/advanced/limitations.ipynb`) still referenced braintrace's own
  pre-#130 `yw_to_w` / `ETP_RULES_YW_TO_W` rule naming instead of the current
  `dt_to_t` / `ETP_RULES_DT_TO_T`; two executable cells in
  `etp_primitives.ipynb` raised `ImportError` / `TypeError` if re-run.
  Incidentally, `etp_primitives.ipynb` and `docs/quickstart/concepts.ipynb`
  also still called `element_wise(weight, fn=...)`, predating that
  parameter's rename to `weight_fn`. All affected notebooks were re-executed
  end-to-end to confirm they now run cleanly.
- **CI runs the example suites.** `examples/tests/` and
  `examples/pp_prop/tests/` both existed and both passed locally, but the CI
  test job ran `pytest braintrace/` only, so neither was ever executed — which
  is how the two `AttributeError` / `TypeError` bugs fixed in #153 reached
  `main` in the first place. They now run in a dedicated `Examples` job, kept
  out of the four-way JAX matrix because multiplying a 13-minute integration
  smoke run across four JAX versions buys nothing. The files were also renamed
  from the `test_*.py` prefix to the `*_test.py` suffix the rest of the repo
  uses. The job earned its place immediately: it caught that
  `examples/003-snn-memory-and-speed-evaluation-*.py` import `psutil` on the
  CPU backend without anything declaring it — green locally only because
  developers happen to have it installed. It is now in `requirements-dev.txt`.

### Documentation

- **Tutorials reorganised into learning paths.** The flat tutorial list became
  four hubs — *Online training*, *Algorithm tutorials*, *Foundations*,
  *Compiler & runtime* — with the deeper material moved from
  `docs/tutorials/` to a new `docs/advanced/` section. The hierarchy is native
  RST rather than synthesised.
- **The Algorithms API reference is complete**, covering every public algorithm
  including the five added in this release.
- **Every docstring is NumPy-style.** The remaining Google-style docstrings were
  converted, so the whole public surface renders consistently.
- **`ScaledWSLinear` is documented.** It was exported from `braintrace.nn` but
  appeared on no API page, which `nitpicky = True` would eventually flag.
- **`README.md` now opens with a runnable quickstart** — `compile` plus
  `etrace_grad` — where it previously carried no code at all.
- **The docs build is warning-free again.** The convolution layers now close the
  bullet lists they inherit from the upstream `brainstate` docstrings
  structurally, rather than by matching one exact sentence of upstream wording
  that had since changed; `docs/conf.py` gained the `brainstate.random.split_key`
  nitpick exemption the ecosystem convention already used for `brainstate`
  classes.
- **The audit's deferred findings are written down.** Ten engineering-hygiene
  items surfaced by the pre-release audit are recorded in
  `docs/specs/2026-08-07-deferred-engineering-backlog.md`, kept separate from
  the learning-rule correctness backlog in
  `docs/specs/2026-07-25-known-limitations.md`. Nine of the ten (E-01 … E-09)
  were resolved before this release shipped; each carries an implementation
  spec of its own under `docs/specs/`.
- **`docs/_static` lost 599 KiB of unreferenced assets**
  ([#163](https://github.com/chaobrain/braintrace/issues/163)). Seven files
  that no docs page, README, docstring, example or workflow referenced were
  deleted from every clone's checkout. Two files the issue implied were dead
  are kept, both load-bearing: `braintrace-learning-map.svg` is used by
  `docs/index.rst`, and `braintrace.png` is hotlinked by the READMEs frozen on
  the PyPI pages for 0.1.1 and 0.1.2 — deleting it would permanently break
  their header image. Three stale per-file `.gitignore` rules naming files
  that are already absent were replaced with one documented glob for the
  editable figure masters. Verified with a real `sphinx-build -W --keep-going`
  over all 114 pages.

### Internal

- **`braintrace._compiler` and `braintrace._legacy` are now inside the typing
  gate** ([#164](https://github.com/chaobrain/braintrace/issues/164)). The
  `disallow_untyped_defs` module list in `pyproject.toml` is what makes "every
  shipped def is annotated" a property `mypy` enforces rather than a
  convention. Two packages were still outside it — the whole jaxpr-analysis
  layer that every algorithm depends on, and the frozen v0.1.x back-compat
  shim. Both are now listed and fully annotated: 93 `no-untyped-def` errors
  cleared (54 in `_compiler`, 39 in `_legacy`). No runtime behaviour changed;
  the only non-annotation edit is a corrected `Returns` docstring on
  `ETraceGraph.call_hidden_perturb`, which claimed to return the model outputs
  when it returns the same four-element tuple as a normal forward call. See
  `docs/specs/2026-08-07-e09-type-gate-compiler-legacy.md`.
- **The last two modules without co-located tests have them**
  ([#161](https://github.com/chaobrain/braintrace/issues/161)).
  `braintrace/_typing_test.py` pins `as_size_tuple`'s normalisation contract
  across every arm of the `Size` union, its idempotence, the round-trip
  through a `brainstate` size setter that motivates the helper, and each
  rejection by exception type — including two sharp edges recorded as facts
  rather than changed (a float inside a sequence truncates toward zero; a
  numeric string iterates character by character, so `'12'` becomes `(1, 2)`).
  `braintrace/nn/__init___test.py` pins the deprecation dispatcher: all 48
  forwarded names, `stacklevel=2` attribution, non-memoisation, dispatch-table
  disjointness, a sorted `__dir__`, and the `AttributeError` fallthrough that
  stops an unknown name from silently evaluating to `None`. Test-only; no
  runtime behaviour is altered.


## Version 0.2.4

This release makes eligibility-trace online learning work *through* JAX control
flow. A new compiler canonicalization + descent pipeline lets ETP operations
inside `vmap`, `cond`, `scan` / `for_loop`, and weight-free `while` bodies
participate in online learning (Phases 0–4), so recurrent cells built with
control flow no longer silently drop parameters from the trace graph. The
operator layer gains three new ETP ops — `grouped_matmul`, `embedding`, and
`einsum` — each with a matching `braintrace.nn` layer; the D-RTRL multi-step
trace update is chunk-factorized for a 2.4–4.5× speedup on multi-step windows;
and a full `_op` / `_algorithm` audit closes 24 correctness findings. The
compiler itself is now deterministic across processes and transparently inlines
user `jax.jit` bodies. One internal ETP rule is renamed (see Breaking changes).

### Highlights

#### New: `grouped_matmul`, `embedding`, and `einsum` ETP operators

- **Three new ETP operators join the operator layer**, each with hand-written
  ETP rules (`dt_to_t`, `xy_to_dw`, trace initializers), a closed-form D-RTRL
  fast path where applicable, public exports, and single-step BPTT-oracle
  coverage:
  - **`braintrace.grouped_matmul`** — a grouped matmul exposing both batched
    and unbatched primitives (`etp_gmm` / `etp_gmv`) and a closed-form D-RTRL
    fast path; D-RTRL matches BPTT element-wise and `pp_prop` is directionally
    aligned. Wrapped by the new **`braintrace.nn.GroupedLinear`** layer.
  - **`braintrace.embedding`** — an ETP embedding lookup with a broadcast
    `dt_to_t` and a scatter-add `xy_to_dw`. Because the input is integer token
    indices, the IO-dim (`pp_prop` / `ES_D_RTRL`) input trace cannot low-pass
    the raw indices; a new optional per-primitive `ETP_RULES_PP_X_REPR` registry
    lets `embedding` filter the linear one-hot representation
    (`y = onehot(idx) @ T`) instead, and `xy_to_dw` dispatches on the x dtype
    (integer indices → gather-VJP scatter-add; float one-hot → contraction VJP).
    Wrapped by the new **`braintrace.nn.Embedding`** layer.
  - **`braintrace.einsum`** — an equation-parsed ETP einsum with axis
    classification; diagonal-class and shared-axis equations are D-RTRL
    BPTT-exact (maxdiff 0.0), while the genuinely lossy regime (output positions
    collapsing into a smaller hidden state) fails loudly at compile time with a
    cotangent-shape error rather than silently emitting wrong gradients.

#### New: structured scan descent — long ETP scans compile and learn online (Phase 4)

- **A third compile path for ETP-relevant `scan`s above the unroll limit.**
  Previously an ETP-relevant `lax.scan`/`for_loop` whose static length
  exceeded `ControlFlowPolicy.scan_unroll_limit` was a dead end
  (`NotImplementedError`). Under the new default
  `ControlFlowPolicy(scan_descent='auto')`, the compiler *descends* such a
  scan: relations and hidden groups are discovered inside the scan body
  with the same flat finders, the equation is rewritten to emit stacked
  per-substep values as extra ys (leading substep axis `L`), and the graph
  executor computes stacked per-substep Jacobians by vmapping over that
  axis — the compiled program stays a single scan equation, so compile size
  is independent of the loop length (an `L=100` inner loop compiles in
  under 60 equations). A `SCAN_DESCENT_APPLIED` INFO diagnostic records
  each descent; blocked scans get `SCAN_DESCENT_SKIPPED`. Set
  `ControlFlowPolicy(scan_descent='off')` to restore the pre-Phase-4 error.
- **Param-dim algorithms fold the eligibility trace over the substep
  axis.** `D_RTRL` (the `ParamDimVjpAlgorithm` family) applies its trace
  update per substep with an inner `jax.lax.scan`
  (`eps <- D_tau * eps + x_tau (x) df_tau`), declaring
  `_supports_scan_descent = True`. The fold is values-only and
  stop-gradient'd — never differentiated, so no checkpointing is needed.
  The learning signal stays one-per-outer-step (`(*varshape, num_state)`;
  the SNN learning-signal axis contract is unchanged). The io-dim family
  (`pp_prop` / `ES_D_RTRL`) and other algorithms without the flag reject
  descended graphs with an actionable `NotImplementedError` at
  `compile_graph`.
- **Exactness contract (pinned by oracle tests).** For diagonal-recurrence
  bodies (elementwise hidden-to-hidden substep path — the SNN class),
  descended D-RTRL is exact: whole-sequence, chunked (3-step and 1-step
  chunks, where the gradient depends on the folded trace at every chunk
  boundary), and one-step single-step gradients all match BPTT / the
  unrolled twin element-wise, including a two-hidden-state
  (`num_state == 2`) group through the fold. For bodies that mix the hidden
  state through an ETP matmul, whole-sequence multi-step gradients remain
  BPTT-exact; chunked gradients approximate cross-substep credit (the same
  approximation class as the unroll path — documented divergence).
- **Algorithm-level `control_flow` kwarg.** `D_RTRL`, `pp_prop`
  (`IODimVjpAlgorithm`), `OTPE`, and `OTTT` now accept
  `control_flow=ControlFlowPolicy(...)` and thread it through their graph
  executors into compilation.
- **v1 restrictions** (each blocks descent for that scan, with a
  diagnostic): reverse scans, nested control flow inside the body,
  trainable weights scanned over as xs, and an *outer* ETP relation
  targeting a hidden state carried by a descended scan (raises with
  restructuring guidance). `jit` bodies nested inside control-flow
  equations are now inlined during extraction so descent sees a flat body.
- **Single-step readout limitation.** The per-step hidden perturbation is
  added to a descended scan's *carry* outvar; a loss that reads the hidden
  state through the scan's stacked ys (e.g. `for_loop(...)[-1]`) bypasses
  it, dropping the same-step learning signal (pinned by test; parallels
  the Phase 3 while-hidden limitation). Read the state after the loop
  (`self.h.value`) instead — multi-step VJP is unaffected either way.

#### New: `while`-loop policy — weight-free opaque-forward support (Phase 3)

- **Weight-free `lax.while_loop`s that read/update hidden state now
  compile.** Under the new default policy knob
  `ControlFlowPolicy(while_hidden='opaque-fwd')`, a `while` whose inputs
  carry no trainable ETP weight is kept as an *opaque forward node*: the
  compiler registers relations whose `y`→hidden tail crosses the loop,
  emits a `CONTROL_FLOW_OPAQUE_FWD` INFO diagnostic, and extracts
  hidden-to-hidden Jacobians for any hidden group whose transition contains
  a `while` in **forward mode** (`jax.jvp`-based `jacfwd_last_dim` /
  `jacfwd` block extraction) — reverse mode through `while_loop` is
  structurally unsupported by JAX. Set
  `ControlFlowPolicy(while_hidden='error')` to reject such loops instead.
- **Perturbation detach keeps the VJP reverse-traceable.** The hidden
  perturbation pass rewires every hidden-producing `while` to consume
  `stop_gradient` copies of its inputs in the *perturbed jaxpr only*; the
  `h = fresh + ε` add stays outside the detach, so the single-step learning
  signal of the loop's **own** hidden group (taken exclusively from the
  perturbation cotangents) is exact. Verified: D-RTRL single-step gradients
  on a while-settle model match its hand-composed no-`while` twin
  element-wise, and the twin matches the BPTT oracle.
  **Limitation:** the detach zeroes every same-step reverse path *through*
  the loop, so a parameter or other hidden group whose only same-step path
  to the loss crosses the loop — e.g. the weights of an upstream layer
  feeding a while-hidden layer — receives a **zero** learning signal (a
  WARNING-level `CONTROL_FLOW_OPAQUE_FWD` diagnostic records each detach;
  the zero-upstream-gradient behavior is pinned by test).
  `vjp_method='multi-step'` on a `while`-hidden model still raises JAX's
  reverse-through-`while_loop` `ValueError` (documented limitation — use
  the default single-step path).
- **A weight used inside a `while` is now a hard, actionable error**
  (`WEIGHT_IN_WHILE` ERROR diagnostic + `NotImplementedError`): move the
  weight application outside the loop so the loop consumes only its result
  (subject to the same-step limitation above), or use a fixed-length
  scan/`for_loop` (which the compiler unrolls).
- **Breaking: ETP primitives left inside an un-flattened `scan`/`while`/
  `cond` body now raise** instead of being silently warned-and-excluded
  (`etp_in_control_flow='error'`, the new default). Pass
  `ControlFlowPolicy(etp_in_control_flow='exclude')` to restore the old
  warn-and-exclude behavior.
- **Position-mixing guard**: a `while`/opaque control-flow body that applies
  `dot_general`/`conv_general_dilated` to hidden-derived values (recurrent
  weight mixing inside the loop) cannot be expressed as a per-position
  Jacobian; the compiler treats it as a boundary, emits a
  `CONTROL_FLOW_RECURRENT_MIXING` WARNING, and falls back to the
  zero-recurrence (e-prop-style) group.

#### New: inner-`scan` unrolling (compiler canonicalization, Phase 2)

- **ETP operations inside `lax.scan` / `brainstate.transform.for_loop`
  bodies now participate in online learning.** A new canonicalization pass
  (`unroll_inner_scans` in `_compiler/canonicalize.py`) runs at extraction
  time and replaces every ETP-relevant, statically short scan with its
  unrolled body: one cloned copy per iteration with fresh variables, `xs`
  sliced per step, consumed `ys` re-stacked via `broadcast_in_dim` +
  `concatenate`, and `reverse=True` respected. The unrolled program is
  value- and Jacobian-identical to the scan, so exact algorithms (D-RTRL,
  full-rank pp_prop, EProp(k=0), OSTLRecurrent) match BPTT element-wise on
  scan-body models — verified against hand-flattened twins and the BPTT
  oracle. Cond and scan canonicalization now run as a joint fixpoint
  (`canonicalize_control_flow`), so a `cond` inside a scan body (and an
  eligible scan inside a `cond` branch) both flatten.
- **Relation counts follow the weight→weight→hidden invariant**: in an
  unrolled inner loop only the *last* sub-step's ETP ops become relations —
  earlier sub-steps reach the hidden state through another trainable ETP op
  and are excluded (with the usual no-relation warning).
- **Eligibility gates**: only scans whose static `length` is ≤
  `ControlFlowPolicy.scan_unroll_limit` (default 16) and that carry no
  effects and contain no `while` are unrolled. An ETP-relevant scan that
  fails a gate emits a `SCAN_UNROLL_SKIPPED` warning and keeps today's
  hard-error behavior; unrolls are recorded as `SCAN_UNROLLED` INFO
  diagnostics on `ETraceGraph.diagnostics`. Scans that scan *over* a
  trainable weight (weights as `xs`) are never unrolled
  (`RELATION_EXCLUDED_SLICED_WEIGHT` warning) — per-slice trace lineage is
  deferred.
- **Cond gate revision**: a branch containing a scan no longer blocks
  if-conversion when that scan is itself unrollable; `scan_unroll_limit=0`
  disables unrolling and restores the exact Phase 1 gating.
- **Tied-weight invariant locked**: one `ParamState` consumed by several
  ETP call sites (which unrolling multiplies) is keyed per relation
  instance with per-path gradient accumulation — verified BPTT-exact and
  now covered by regression tests.

#### New: `cond` if-conversion (compiler canonicalization, Phase 1)

- **ETP operations inside `lax.cond` branches now participate in online
  learning.** A new canonicalization pass (`_compiler/canonicalize.py`)
  runs at extraction time (after user-`jit` inlining) and rewrites every
  ETP-relevant `cond` equation into the inlined bodies of *all* branches
  followed by one `select_n` per output. `select_n`'s index semantics and
  JVP match `cond` exactly, so for finite branches values and Jacobians —
  and therefore exact algorithms such as D-RTRL — are unchanged. Weights used inside `cond`
  branches previously raised `NotImplementedError` (or were silently
  excluded when only ETP primitives appeared inside).
- **Semantics note**: on the canonicalized graph **both branches execute
  every step** and the dead branch's value is discarded by `select_n`.
  Values and forward-mode derivatives are unaffected by dead-branch
  NaN/Inf. **Reverse-mode gradients are not**: if the dead branch's local
  Jacobian is NaN/Inf (e.g. a `cond` protecting a `sqrt` domain), its VJP
  multiplies the exact-zero cotangent by that Jacobian (`0 * nan = nan`)
  and contaminates gradients of shared inputs — the classic single-`where`
  pitfall. Keep such domain-guard conds opaque
  (`ControlFlowPolicy(cond='opaque')`) or guard the operand inside the
  branch.
- **Gates**: conds that touch no ETP primitive, weight, or hidden state
  stay opaque at zero cost. Conds with effects or containing `while`/`scan`
  in a branch are never converted; an ETP-relevant one that is skipped this
  way emits a `COND_CONVERSION_SKIPPED` warning and keeps today's behavior.
  Conversions are recorded as `COND_IF_CONVERTED` INFO diagnostics on
  `ETraceGraph.diagnostics`.
- **Opt-out**: `braintrace.ControlFlowPolicy(cond='opaque')` via the new
  `control_flow` keyword on `compile_etrace_graph` / `extract_module_info`
  restores the previous behavior.

#### New: vmap identity preservation (operator layer)

- **vmap identity preservation (operator layer)**: `jax.vmap` over an
  unbatched ETP op (`matmul`, `lora_matmul`, `sparse_matmul` with vector
  input) now re-binds the batched ETP primitive (`etp_mm` / `etp_lora_mm` /
  `etp_sp_mm`) instead of decomposing into standard JAX ops. Models that
  vmap per-sample ETP operations inside `update()` now compile with full
  eligibility-trace relations. When promotion is impossible (batched
  weights, `etp_conv`, nested vmap), the op decomposes as before but emits
  a `UserWarning` instead of silently dropping the parameter from online
  learning. Note: when this warning appears from a `compile(..., vmap=True)`
  learner's execution trace (e.g. conv models), it is expected and benign —
  the eligibility-trace graph was already compiled per-sample before the
  learner was vmapped, so no parameter is dropped.

#### New: user-`jit` inlining and a deterministic compiler

- **ETP operations inside a user `jax.jit` now compile.** `extract_module_info`
  inlines user `jax.jit` bodies before any analysis, so `jit` boundaries are
  transparent to hidden-group discovery and relation finding. Previously a
  weight used inside a `jit` raised `NotImplementedError` and bare ETP
  primitives were silently skipped (#123).
- **Deterministic, reproducible compilation.** Hidden-group discovery,
  transition bookkeeping, and group merging now use insertion-ordered maps and a
  canonical compiled-state ordering instead of `set`s keyed by object identity,
  so group membership and ordering are stable across processes. Every built
  group is validated with `check_consistent_varshape`, and merges emit an
  INFO-level `HIDDEN_GROUP_MERGED` diagnostic (#123).
- **Directly-fed fan-out fix.** A single ETP op feeding two independent
  recurrent states now registers relations to *both* groups — the forward BFS
  previously locked onto whichever hidden state it reached first and dropped the
  rest. Relation gating also resolves all keys before excluding a relation, so a
  constant-weight / `ParamState`-bias matmul still registers with the bias as
  its trainable key (#123).
- **Robust perturbation pass.** The single-step perturbation now handles
  multi-output equations and read-only hidden states (synthesizing the
  `h^t = h^{t-1} + p` passthrough) and preserves the source jaxpr's effect set,
  instead of falling through to an unexplained-hidden error (#123).

### Performance

- **Chunk-factorized multi-step D-RTRL trace update.** Multi-step trace updates
  for `D_RTRL` now factor the per-step decay into suffix products and apply the
  trace update per chunk instead of step-by-step, giving a **2.4–4.5× speedup**
  on multi-step windows for the dense (`etp_mm` / `etp_mv`) and elementwise
  (`etp_elemwise`) kernels. Exposed as a `chunked_trace` knob on `D_RTRL` /
  `braintrace.compile` (#132).

### Correctness

- **ETP `_op` / `_algorithm` audit — 24 findings closed (4 Critical, 6 High,
  6 Medium, 8 Minor).** Highlights: exact `conv` (C1) and `lora_matmul` (C2)
  gradients under param-dim D-RTRL (per-position kernel trace + effective-weight
  trace, backed by new optional instant / solve D-RTRL rule registries); a fix
  for the batched sparse D-RTRL crash (C3) via a hashable CSR wrapper; and
  OSTTP's always-zero learning signal (C4) via `custom_vjp` residual threading.
  Also resolved: `trace_dtype` gate mismatch, conv bias broadcast, EProp
  kappa-filter cross-state contamination and random-feedback scale invariance,
  OTTT / OTPE dropped bias gradients and missing guards, int/bool autodiff
  guards, rank guards with nn-layer axis folding, and a corrected OSTL exactness
  claim. Adds a cross-family single-step BPTT oracle suite and first-principles
  rule tests (`6c7796a`).

### Breaking changes

- **ETP rule rename: `YW_TO_W` → `DT_TO_T`.** The recurrent trace-propagation
  rule computes `D^t * ε^{t-1}` (the `Dᵗ`-times-previous-trace term of the
  D-RTRL update), so `DT_TO_T` names it accurately; `YW_TO_W` never matched what
  the rule computes. Custom primitives that register this rule (via
  `register_etp_rules` / `register_primitive`) must use the new name. This is
  unrelated to `brainevent`'s external `DataRepresentation.yw_to_w` /
  `yw_to_w_transposed` protocol methods, which are untouched (#130).

### Internal

- **`mypy` CI gate repaired.** Cleared 40 accumulated type errors across 8 files
  (annotation-only, no behavior change), restoring a green `typecheck_and_build`
  job (#133).
- Signature cleanups: a readability pass on `_etp_sp_matmul_impl` and removal of
  unused keyword arguments across several modules.


## Version 0.2.3

This release adds optional, shape-preserving parameter-transform hooks to the
eligibility-trace (ETP) operators, so a trainable weight (or bias) can be passed
through an elementwise / standardizing function *before* it enters the operation
while the eligibility trace and gradient remain with respect to the **raw**
stored parameter. These hooks are threaded through the `braintrace.nn` linear
layers and demonstrated in a new tutorial. The release also hardens the public
API with inline type annotations behind an enforced `mypy` gate, corrects the
`weight_fn` / `bias_fn` gradients on the closed-form fast path, relocates the
fast-path kernels into the operator layer, and tightens the `sparse_matmul`
input contract. Two public APIs are renamed and one operand type is now required
(see Breaking changes).

### Highlights

#### New: parameter-transform hooks on ETP operators

- Add transform hooks to the ETP ops, computing
  `y = x @ weight_fn(w) (+ bias_fn(b))` (and per-op equivalents), with the
  eligibility trace and gradient kept with respect to the **raw** parameter:
  - **`braintrace.matmul`** / **`braintrace.sparse_matmul`** — `weight_fn`,
    `bias_fn`.
  - **`braintrace.conv`** — `kernel_fn`, `bias_fn`.
  - **`braintrace.lora_matmul`** — `b_fn`, `a_fn`, `bias_fn`.
  - **`braintrace.element_wise`** — `weight_fn` (see Breaking changes).

  Each transform is applied *inside* the ETP primitive; the per-parameter
  Jacobian is recovered exactly once (via `jax.vjp`) in the weight-gradient rule,
  while the trace-propagation rule is unchanged — so the forward-mode eligibility
  trace stays exact and is never double-counted. D-RTRL matches
  backprop-through-time element-wise for non-identity transforms (verified with
  `tanh`, `w**2`, and `abs`). Omitting a transform is bit-identical to the
  previous behavior.

#### New / Improved: `braintrace.nn` linear layers

- **`braintrace.nn.Linear`** (with `w_mask`), **`braintrace.nn.SignedWLinear`**,
  and **`braintrace.nn.ScaledWSLinear`** now route their weight masking / sign /
  standardization through the new `matmul(weight_fn=...)` hook, so the masked /
  signed / standardized weight participates in eligibility-trace learning with
  the gradient kept w.r.t. the raw weight leaf. (For `ScaledWSLinear`, `gain` and
  `bias` are applied as post-operations and are therefore non-temporal for the
  online trace, though still recovered exactly by the multi-step VJP oracle.)
- **Export `braintrace.nn.ScaledWSLinear`** (previously importable only by its
  fully-qualified module path).

#### New: typed public API with an enforced `mypy` gate

- Inline type annotations now cover the public surface — ETP operators and their
  rule functions, `ETPPrimitive` / `register_primitive`, the `braintrace.compile`
  entry point and package accessors, input-data containers, the `braintrace.nn`
  linear / conv / recurrent cells, and the algorithm base classes, executors, and
  concrete algorithms. A new `WeightFn` alias names the transform-hook signature.
- An enforced `mypy` gate guards the public API, so type regressions fail the
  build (#119).

### Improvements

- **Correct `weight_fn` / `bias_fn` gradients on the fast path.** The
  transform Jacobian `f'(W)` is now applied on the param-dim D-RTRL closed-form
  fast path (it lives solely in `xy_to_dw`; `dt_to_t` stays transform-free), so
  transformed-parameter gradients match the slow path. Also fixes an
  `element_wise` slow-path batched-cotangent crash (#120).
- **Operator-layer fast-path kernels.** The closed-form fast-path kernels
  (instant / recurrent / solve) move into the operator layer as a per-primitive
  `FastPathRules` bundle behind an `ETP_FAST_PATH_RULES` registry, and the
  algorithm-layer string-match gate is replaced by a per-primitive
  `applicable()` predicate — keeping primitive knowledge in the operator layer
  per the layered design (#120).

### Documentation

- **New tutorial: customizing primitive transforms**
  (`docs/tutorials/customizing_primitive_transforms.ipynb`), plus transform-hook
  docstrings on the ETP operators (#120).

### Breaking changes

- **`braintrace.element_wise`**: the `fn` parameter is renamed to **`weight_fn`**
  and is now **keyword-only**, and the transform is applied *inside* the ETP
  primitive (previously it was applied to the weight outside the primitive).
  Migrate `element_wise(w, fn=g)` to `element_wise(w, weight_fn=g)`. Forward
  results are unchanged; only the call signature and the internal
  trace-factorization point differ.
- **`braintrace.sparse_matmul`**: the weight parameter is renamed from
  `weight_data` to **`weight`** for a cleaner, more consistent API. All in-tree
  call sites pass it positionally and are unaffected; update any keyword callers
  (#116).
- **`braintrace.sparse_matmul`**: the sparse operand (`sparse_mat`) must now be a
  **`brainevent.DataRepresentation`** and is enforced with a strict runtime
  `isinstance` check (raising `TypeError`). `DataRepresentation` supplies the ETP
  online-learning protocol the compiler / executor require (`with_data`,
  `yw_to_w`, `yw_to_w_transposed`); `brainunit` sparse types (`u.sparse`) lack
  these and are no longer accepted. **`brainevent` is now a runtime dependency.**
  Migrate sparse weights to `brainevent` (e.g. `brainevent.CSR`) (#121).

### Dependencies

- Add **`brainevent`** as a runtime dependency (`pyproject.toml`,
  `requirements.txt`) (#121).
- Bump `codecov/codecov-action` from 5 to 7 (#117).


## Version 0.2.2

This release introduces a unified `braintrace.compile` entry point for building
eligibility-trace online learners, adds a recurrent mixing mode to the
graph-construction compiler, and fixes eligibility-trace convergence under
`vmap` / `brainstate.mixin.Batching()`. It also migrates unit handling from
`saiunit` to `brainunit`, modernizes the toolchain (Python 3.14,
`brainstate` >= 0.5.2, Codecov), and ships broad documentation, example, and
test improvements. Internal modules were renamed for brevity; no documented
0.2.x public API is removed.

### Highlights

#### New: unified `braintrace.compile` entry point

- **`braintrace.compile(model, algorithm, example_input, ...)`** is now the
  canonical, single-call way to build a compiled online learner. It always
  initializes states, accepts a `seed`, applies model guardrails, and can emit a
  verbose compilation report — replacing the manual
  `init_states` / `learner.compile_graph(x0)` triad.
- **`vmap=` parameter** for per-sample vmap state initialization. With
  `vmap=True`, states are built via `vmap_new_states(state_tag='new', ...)` and
  the learner is wrapped in `brainstate.nn.Vmap(vmap_states='new')`, so
  eligibility-trace models compose with brainstate's per-sample vmap scheme.
- **`CompilationReport`**, a structured view over the eligibility-trace graph
  (relation/weight counts, `etrace_weights`, `excluded_weights`, `report.show()`
  with verbosity levels). It is exposed via `ETraceAlgorithm.report` and now
  backs `show_graph`.

#### New: recurrent mixing mode for graph construction

- Add a recurrent mixing mode to eligibility-trace graph construction, broadening
  the set of cell topologies the compiler can connect (#108).

### Improvements

#### Dependencies and toolchain

- **Replace `saiunit` with `brainunit`** for all unit handling across source,
  tests, examples, and docs. `brainunit` re-exports `saiunit` internally, so
  this is a drop-in change (#106).
- Raise the `brainstate` floor to **>= 0.5.2**, required by the
  `compile(vmap=True)` path, and drop a duplicate dependency declaration.
- Update the supported Python version to **3.14** and adjust the CI JAX version
  matrix.
- Add **Codecov** coverage reporting and raise source coverage to 93%, with new
  tests for previously-untested modules (#109).

#### Refactoring

- **Rename internal module packages** for brevity: `_etrace_op` → `_op`,
  `_etrace_compiler` → `_compiler`, and `_etrace_algorithms` → `_algorithm`.
  These are private modules; imports were updated package-wide with
  word-boundary-anchored replacement (#111).
- Remove the unused `ParamState` from state management.
- Remove the per-step spectral-normalization path
  (`normalize_matrix_spectrum`) from D-RTRL, E-Prop, and the OSTL trace scan; it
  ran `jnp.linalg.eigvals` on every hidden-group Jacobian, was off by default,
  and was costly.

### Fixes

- **Eligibility-trace convergence under `vmap` batching.** Defer graph
  compilation during the `vmap_new_states` discovery probe so the executor binds
  to the real batched states (fixes a `BatchAxisError` when writing batched
  values), correctly handle models that mix batched and unbatched ETP primitives
  in the param-dim VJP solve, and align convolution eligibility traces under
  `brainstate.nn.Vmap(vmap_states='new')`. Restores convergence for the
  conv-based SNN/RNN training examples.
- **Element-wise eligibility traces under `brainstate.mixin.Batching()`.** Size
  the trace from the (batch-aware) hidden group and sum out the leading batch
  axis in the solver, fixing a scan-carry type mismatch and a custom-VJP
  backward shape mismatch. This unblocks the default SHD batch trainer, where
  every LIF leak is an element-wise weight.
- **`braintrace.nn.LoRA` now routes its forward through the ETP `lora_matmul`
  primitive**, so LoRA factors participate in eligibility-trace learning (fixes
  the zero-relations bug) and the factor order is corrected.
- Resolve pre-existing `mypy` errors in the compiler's `report.py` (#112) and
  treat `brainunit` / `saiunit` as untyped for `mypy` to clear spurious
  `attr-defined` errors from their re-export chain.
- Convert legacy `xfail` tests to positive assertions, silence the `core.Jaxpr`
  `DebugInfo` deprecation warning, and migrate deprecated `brainstate` APIs
  (`brainstate.augment` → `brainstate.transform`, `brainstate.functional` →
  `brainstate.nn`) (#113).

### Documentation and examples

- Make `braintrace.compile` the canonical entry point in every docstring,
  tutorial, notebook, and example, and fix broken examples (e.g. self-contained
  RNNs, consistent batch axes); each documented example is now backed by an
  executable test (#114).
- Document `CompilationReport` in the API reference and migrate the onboarding
  guides, quickstart, and tutorials to the unified compile flow.
- Add a smoke-test harness and a testable `main()` entry point to the
  standalone examples; repair all docs notebooks so they execute cleanly.

### Notes

- The internal module renames (`_etrace_*` → `_*`), the removal of `ParamState`,
  and the removal of `normalize_matrix_spectrum` touch private/internal surfaces
  only; the documented 0.2.x public API is unchanged.
- Verified locally: the full CPU test suite is green (1604 passed, 3 skipped).


## Version 0.2.1

This is a maintenance release that restores compatibility with the latest
brain-ecosystem dependencies and toolchain. It contains no functional or
public-API changes — code written against 0.2.0 continues to work unchanged —
and exists to keep BrainTrace green against `brainstate` 0.5, `saiunit`/
`brainunit` 0.5.1, and `pytest` 9.1.

### Fixes

#### Dependency Compatibility

- **`brainstate` 0.5 typed API**: adopted `brainstate`'s PEP 561 `py.typed`
  surface throughout the source — routed `PyTree` through BrainTrace's existing
  type alias, centralized an `as_size_tuple()` helper in `_typing`, dropped
  `FlattedDict` subscripts, and added boundary asserts/casts. This clears the
  154 mypy errors newly exposed by the upstream typing, with minimal
  `# type: ignore` only where `brainstate`'s typing makes it unavoidable.
- **`brainstate` 0.5.0 convolution validation**: updated convolution test
  expectations for the hardened validation (bare `assert` → `ValueError`) and
  the new one-value-per-spatial-dimension padding-tuple semantics.
- **`pytest` 9.1.0 collection**: removed trailing commas in single-argument
  `parametrize` ids that `pytest` 9.1.0 mis-parses as two values, fixing a
  collection-time `GraphNodeMeta has no len()` error.

### Notes

- All changes are BrainTrace-side. A related upstream `saiunit` issue is
  resolved in `saiunit`/`brainunit` 0.5.1 and requires no change here.
- Verified locally: full suite 1367 passed (2 xfailed), mypy clean across 51
  files, and wheel + sdist build with `py.typed` shipped (PEP 561).


## Version 0.2.0

This release is a major step for BrainTrace. It adds a family of spiking neural
network (SNN) online-learning algorithms, rewrites the eligibility-trace
compiler around primitive-type dispatch, generalizes every ETP primitive to
support multiple trainable inputs (fixing a silent bias-gradient drop),
delivers substantial performance gains for D-RTRL and multi-step rollouts, and
hardens the package with PEP 561 typing and a BPTT-oracle-backed test suite.

### Major Changes

#### New: SNN Online-Learning Algorithms

- **Added five SNN online-learning algorithms** as flat `ETraceVjpAlgorithm`
  subclasses: `EProp`, `OSTL` (`OSTLRecurrent` / `OSTLFeedforward`), `OTPE`,
  `OTTT`, and `OSTTP`. All are exported at the top level.
- **Added a `_compute_learning_signal` hook** to `ETraceVjpAlgorithm` to support
  target-projection algorithms (`OSTTP`) without disrupting the existing D-RTRL
  and pp-prop paths.
- **Added supporting trace helpers**: `PresynapticTrace`, `KappaFilter`,
  `FixedRandomFeedback`, and target-signal extraction utilities.
- Algorithms are cross-checked for regime equivalence and verified to decrease
  loss in integration smoke tests.

#### ETP Compiler Rewrite

- **Rewrote the eligibility-trace compiler to dispatch on primitive-type
  identity** rather than string-matching op or trace names, with structured,
  leveled diagnostics (`DiagnosticKind`, `DiagnosticLevel`,
  `CompilationRecord`) replacing ad-hoc warnings.
- **Added compile-time diagnostics** that surface previously silent issues —
  e.g. `TRAINABLE_INVAR_NOT_PARAMSTATE` flags a trainable input (such as a
  constant bias) that does not trace to a `ParamState`, so users can wrap it
  intentionally instead of silently losing its gradient.

#### Multi-Trainable-Input ETP Primitives (Bias Gradients)

- **Generalized every ETP primitive from a single-"weight" assumption to an
  arbitrary named dict of trainable inputs.** This fixes a silent bias-gradient
  drop and a LoRA executor signature mismatch in one coherent refactor.
- Migrated all built-in primitives (`elemwise`, dense `mm`/`mv`, `conv`,
  `sparse` `mm`/`mv`, and `lora`) to the dict-based rule API with first-class
  **bias gradient support**, each verified element-wise against a BPTT oracle.
- **Fixed layout-aware axis handling in conv** primitives (1D/2D, NHWC/NCHW,
  OIHW/HWIO kernel layouts) that previously corrupted gradients on non-default
  layouts, and **fixed non-square dense weight broadcasting** in `_mm_dt_to_t`.
- Eligibility traces are now stored as per-key dicts; the transitional
  legacy-array adapter has been fully removed.

#### Performance

- **D-RTRL einsum fast path** (`fast_solve=True`, default on): replaces nested
  `vmap`-of-`vjp` and per-step `lax.cond` overhead with direct einsum kernels
  for `mm`/`mv`/`elemwise`; conv/sparse/LoRA fall back to the legacy path.
- **Reduced-precision trace storage** (`trace_dtype`, e.g. bf16/fp16) halves the
  dominant `B*N^2` trace bandwidth on GPU/TPU while keeping Jacobians, learning
  signals, and final gradients in fp32. Default `None` preserves exact behavior.
- **Multi-step trace fusion**: the per-step eligibility-trace roll for exact
  algorithms (D-RTRL, pp-prop) is now threaded into the graph executor's forward
  scan, eliminating an `O(T × Jacobian)` HBM round-trip (traced scan count drops
  3 → 2). Opt-in and multi-step-only; single-step/SNN paths are unchanged.
- Branch-free spectrum/vector normalization to restore XLA fusion across steps.

#### Primitive Registration Simplification

- **Removed `ETPPrimitiveSpec`** and the spec-based registration layer; invar/
  outvar layout metadata (`trainable_invars_fn`, `x_invar_index`,
  `y_outvar_index`) now lives in internal registries populated directly through
  `register_primitive` keyword arguments.

#### Package Restructuring

- **Consolidated the eligibility-trace code into a single flat
  `_etrace_algorithms` package**, merging the former `_etrace_vjp/`,
  `_etrace_algorithms.py`, `_etrace_graph_executor.py`, and `_snn_algorithms/`
  modules. The top-level public API is unchanged.
- **Split the algorithm base hierarchy into dedicated modules**:
  `ParamDimVjpAlgorithm` (D-RTRL) and `IODimVjpAlgorithm` (pp-prop) now live in
  their own files, with `D_RTRL`/`pp_prop` as thin subclasses.
- Removed the experimental hybrid online-learning method.

#### Typing & Packaging

- **The package is now PEP 561 compliant**: ships a `py.typed` marker so
  downstream users receive inline type hints.
- Added a pragmatic `mypy` configuration and wired type checking plus packaging
  verification (`python -m build`, `py.typed` presence) into CI.

#### Testing

- **Added a BPTT gradient oracle and a layered correctness test suite** (P2–P8):
  per-operator rule oracles, public-API contract tests, exact-class
  element-wise equivalence with BPTT, approximate-class direction-alignment
  checks, transform/integration invariance, and per-cell compiler relation
  guardrails tied to the cell registry.

#### Documentation

- **Converted all public-API docstrings to NumPy-doc style** with math,
  references, and runnable examples.
- Documentation is now self-hosted at `brainx.chaobrain.com/braintrace/`, with
  refreshed RTD links and a WebP logo.

#### Dependencies & Tooling

- **Replaced `brainunit` with `saiunit`** throughout for unit handling.
- Numerous CI/CD upgrades (checkout, setup-python, artifact actions, sphinx and
  theme requirements); docs deploy on release publication.

### Deprecations

The entire v0.1.x **class-based** operator/parameter API is deprecated in favor
of the new **primitive-based** ETP user-API. The legacy classes still work —
they are thin back-compatibility shims that route through the new primitives —
but each emits a `DeprecationWarning` (once per class, per process) on first
use, and they will be removed in a future release. Migrate at your convenience.

**Deprecated operator classes** → new primitive functions:

| Deprecated (v0.1.x) | Use instead (v0.2.0) |
| --- | --- |
| `MatMulOp` | `braintrace.matmul` |
| `ElemWiseOp` | `braintrace.element_wise` |
| `ConvOp` | `braintrace.conv` |
| `SpMatMulOp` | `braintrace.sparse_matmul` |
| `LoraOp` | `braintrace.lora_matmul` |
| `ETraceOp` (base) | the ETP primitive functions above |

**Deprecated parameter classes** → `brainstate.ParamState` + a primitive:

| Deprecated (v0.1.x) | Use instead (v0.2.0) |
| --- | --- |
| `ETraceParam` | `brainstate.ParamState` + an ETP primitive function (e.g. `braintrace.matmul`) |
| `ElemWiseParam` | `brainstate.ParamState` + `braintrace.element_wise` |
| `NonTempParam` | `brainstate.ParamState` + plain JAX ops (`x @ w`) — keeps the weight out of the ETP graph |
| `FakeETraceParam`, `FakeElemWiseParam` | plain objects with plain JAX ops |

The `stop_param_gradients` context manager and the `general_y2w` helper are kept
as no-op compatibility shims and have no effect on the new primitive path.

### Breaking Changes

1. **OSTL factory removed** — use `OSTLRecurrent` or `OSTLFeedforward` directly
   instead of the former `OSTL` factory function.

2. **`OTTT` and `OTPE` require an explicit `leak`** — the membrane leak is no
   longer inferred from `model.states()` (it silently picked a wrong value on
   heterogeneous/multi-population models). Both now also reject hidden groups
   with `num_state > 1` at compile time, as collapsing the `num_state` axis has
   no theoretical basis for these LIF-derived rules. `OTPE` additionally
   documents a narrower feed-forward / single-layer / global-scalar-leak regime.

3. **Unit dependency change** — code relying on `brainunit` internals should
   migrate to `saiunit`.

4. **`ETPPrimitiveSpec` removed** — custom primitives must register layout
   metadata via `register_primitive` keyword arguments
   (`trainable_invars_fn`, `x_invar_index`, `y_outvar_index`).

### Migration Guide

#### OSTL
```python
# Old
algo = OSTL(model, ...)         # factory

# New — choose the regime explicitly
algo = OSTLRecurrent(model, ...)
# or
algo = OSTLFeedforward(model, ...)
```

#### OTTT / OTPE
```python
# Old
algo = OTTT(model, ...)               # leak inferred from model.states()

# New — pass the postsynaptic membrane leak explicitly
algo = OTTT(model, leak=0.9, ...)
```

#### Custom ETP primitives
```python
# Old: register_primitive_spec(ETPPrimitiveSpec(...))
# New: pass layout metadata directly
register_primitive(
    prim,
    trainable_invars_fn=...,
    x_invar_index=...,
    y_outvar_index=...,
)
```

#### Deprecated class-based API → primitive-based API
```python
# Old (v0.1.x): wrap the weight in an ETraceParam bound to an op
self.w = braintrace.ETraceParam({'weight': w}, braintrace.MatMulOp())
y = self.w.execute(x)

# New (v0.2.0): a plain ParamState + the ETP primitive function
self.w = brainstate.ParamState({'weight': w})
y = braintrace.matmul(x, self.w.value)
```

The element-wise case is analogous (`ElemWiseParam`/`ElemWiseOp` →
`brainstate.ParamState` + `braintrace.element_wise`); to keep a weight out of
the eligibility-trace graph, use a plain `brainstate.ParamState` with ordinary
JAX ops instead of `NonTempParam` / `FakeETraceParam`.

### Version
- Bumped version from `0.1.3` to `0.2.0`


## Version 0.1.2

### Major Changes

#### Import Path Migration
- **Updated dependency from `brainpy` to `brainpy.state`**: Migrated all imports to use the more specific `brainpy.state` module
  - Updated `braintrace/nn/_readout.py`: Changed neuron model imports from `brainpy` to `brainpy.state`
  - Updated all documentation notebooks (12 files): Concepts, RNN/SNN online learning, batching, state management, and graph visualization tutorials
  - Updated example scripts (4 files): COBA EI RSNN, SNN evaluation, feedforward conv SNN, and SNN models
  - Updated `requirements.txt` and `pyproject.toml` to specify `brainpy-state` as dependency
  - Total: 19 files changed with improved module structure and consistency

#### New Algorithms
- **Added PP-Prop (Pseudo-Prospective Propagation) algorithm**: New eligibility trace algorithm in VJP-based methods
  - Added `pp_prop` to `braintrace/_etrace_vjp/esd_rtrl.py`
  - Updated `docs/apis/algorithms.rst` to include PP-Prop in algorithm documentation

#### Python 3.14 Support
- **Added Python 3.14 compatibility**: Updated project metadata to officially support Python 3.14
  - Updated `pyproject.toml` classifiers to include Python 3.14

#### Bug Fixes
- **Fixed version info tuple creation**: Corrected the version info structure in `braintrace/__init__.py`
  - Ensures proper version tuple formatting for compatibility checks

#### CI/CD Improvements
- **Updated GitHub Actions workflow**: Bumped `actions/upload-artifact` from v5 to v6
  - Modernized CI/CD pipeline with latest GitHub Actions versions
  - Improved artifact upload reliability and performance

#### Documentation Updates
- **Updated documentation links**: Refreshed links in concept documentation for better navigation
  - Updated `docs/quickstart/concepts-en.ipynb` (116 lines modified)
  - Updated `docs/quickstart/concepts-zh.ipynb` (104 lines modified)

### Breaking Changes

**Dependency Change:**
1. **Dependency name change**: The project now requires `brainpy-state` instead of `brainpy`
   - Update your `requirements.txt` or installation commands accordingly

```bash
# Old (0.1.1)
pip install brainpy

# New (0.1.2)
pip install brainpy-state
```

2. **Import path update**: Update neuron model imports to use `brainpy.state`

```python
# New (0.1.2)
from brainpy.state import IF, LIF, ALIF
```

### Migration Guide

#### Update Dependencies
Replace `brainpy` with `brainpy-state` in your project dependencies:

```bash
pip uninstall brainpy
pip install brainpy-state
```

#### Update Import Statements
If you have custom code importing neuron models, update to use `brainpy.state`:

```python
# Find and replace in your codebase
# from brainpy import → from brainpy.state import
```

### Version
- Bumped version from `0.1.1` to `0.1.2`




## Version 0.1.1

### Major Changes

#### Project Rename: BrainScale → BrainTrace
- **Renamed the entire project from `brainscale` to `braintrace`**: This change reflects the project's focus on eligibility trace-based learning algorithms
  - Package directory renamed from `brainscale/` to `braintrace/`
  - All internal imports updated from `brainscale` to `braintrace`
  - Updated all 95 files including source code, tests, documentation, and examples
  - Updated `pyproject.toml` with new project name and metadata
  - Updated README with new project branding and citation information

#### VJP-Based Eligibility Trace Algorithms
- **Added new VJP-based eligibility trace module** (`_etrace_vjp/`): Comprehensive implementation of vector-Jacobian product based algorithms
  - `base.py`: Core base classes and utilities for VJP operations (671 lines)
  - `d_rtrl.py`: Diagonal Real-Time Recurrent Learning implementation (756 lines)
  - `esd_rtrl.py`: Efficient Sparse Diagonal RTRL implementation (847 lines)
  - `hybrid.py`: Hybrid approaches combining multiple techniques (604 lines)
  - `graph_executor.py`: Graph-based execution for VJP computations
  - `misc.py`: Miscellaneous utilities including matrix spectrum normalization

- **Refactored VJP algorithm structure**: Migrated from monolithic `_etrace_vjp_algorithms.py` (2,888 lines) to modular architecture
  - Better separation of concerns
  - Improved testability with dedicated test files (`d_rtrl_test.py`, `esd_rtrl_test.py`, `graph_executor_test.py`)

#### Logo and Branding
- Updated logo format from JPG to PNG for consistency
- Updated logo across documentation

### Breaking Changes

**Package Rename:**
1. **Import path change**: All imports must now use `braintrace` instead of `brainscale`

```python
# Old (0.1.0)
import brainscale
from brainscale import EligibilityTrace
from brainscale.nn import Linear, GRUCell

# New (0.1.1)
import braintrace
from braintrace import EligibilityTrace
from braintrace.nn import Linear, GRUCell
```

2. **Installation**: Package name changed from `brainscale` to `braintrace`

```bash
# Old
pip install brainscale

# New
pip install braintrace
```

### Migration Guide

#### Update Import Statements
Replace all occurrences of `brainscale` with `braintrace`:

```python
# Find and replace in your codebase
# brainscale → braintrace
```

#### VJP Algorithm Usage
The new VJP-based algorithms are now available through the modular interface:


### Version
- Bumped version from `0.1.0` to `0.1.1`


## Version 0.1.0

### Major Changes

#### State Management Refactoring
- **Renamed `ETraceState` to `HiddenState`**: All eligibility trace state management now uses the more general `HiddenState` naming convention
  - Updated across `_etrace_algorithms.py`, `_etrace_concepts.py`, `_state_managment.py`
  - Added deprecation warnings for `ETraceState` to guide users to `brainstate.HiddenState`
  - Updated all documentation and examples to reflect the new naming

- **Renamed `ETraceGroupState` to `HiddenGroupState`**: Improved consistency in hidden state handling
  - Updated in `_etrace_compiler_hidden_group.py`
  - Added deprecation warnings for backward compatibility

- **Added deprecation handling**: Implemented `__getattr__` in main `__init__.py` to provide helpful warnings when using deprecated names:
  - `ETraceState` → `brainstate.HiddenState`
  - `ETraceGroupState` → `brainstate.HiddenGroupState`
  - `ETraceTreeState` → `brainstate.HiddenTreeState`

#### Neural Network Module Reorganization

- **Consolidated neural network modules**: Removed standalone neuron, synapse, and activation modules, migrating them to `brainstate` and `brainpy` ecosystems
  - **Deleted files**:
    - `brainscale/nn/_neurons.py` (IF, LIF, ALIF now in `brainpy.state`)
    - `brainscale/nn/_synapses.py` (Expon, Alpha, DualExpon, STP, STD now in `brainpy.state`)
    - `brainscale/nn/_elementwise.py` (activation functions now in `brainstate.nn`)
    - `brainscale/nn/_poolings.py` (pooling layers now in `brainstate.nn`)

- **Renamed `_rate_rnns.py` to `_rnn.py`**: Simplified module naming for better clarity

- **Added comprehensive deprecation warnings in `nn.__getattr__`**: Automatically redirects users to the correct modules:
  - Neuron models (IF, LIF, ALIF) → `brainpy.state`
  - Synapse models (Expon, Alpha, DualExpon, STP, STD) → `brainpy.state`
  - Activation functions (ReLU, Sigmoid, etc.) → `brainstate.nn`
  - Pooling layers (MaxPool, AvgPool, etc.) → `brainstate.nn`
  - Dropout layers → `brainstate.nn`

#### API Improvements

- **Normalization parameter standardization**: Renamed `normalized_shape` to `in_size` across all normalization layers for consistency
  - Updated in `_normalizations.py` for LayerNorm, GroupNorm, InstanceNorm, etc.
  - Improved clarity and consistency with other layer APIs

- **Enhanced input dimension validation**: Improved error checking in convolutional layers to catch dimension mismatches early

- **Refactored imports for consistency**: Updated all internal imports to use `braintools` for optimization and initialization utilities consistently across the codebase

#### Testing Infrastructure

- **Added comprehensive unit tests** for neural network modules:
  - `_conv_test.py`: 868 lines of tests for convolutional layers (Conv1d, Conv2d, Conv3d, ConvTranspose)
  - `_linear_test.py`: 658 lines of tests for linear layers (Linear, Identity)
  - `_normalizations_test.py`: 695 lines of tests for normalization layers (LayerNorm, BatchNorm, GroupNorm, etc.)
  - `_readout_test.py`: 763 lines of tests for readout layers (LeakyRateReadout, LeakySpikeReadout)
  - `_rnn_test.py`: 710 lines of tests for RNN cells (VanillaRNNCell, GRUCell, LSTMCell, MGUCell, etc.)
  - Total: 3,694 lines of new test coverage

#### Documentation Updates

- **Streamlined API documentation**: Updated `docs/apis/nn.rst` to remove redundant sections and enhance RNN overview
- **Updated tutorials and examples**: All 16 tutorial notebooks and 11 example scripts updated to reflect new APIs:
  - Concepts tutorials (en/zh)
  - RNN and SNN online learning guides
  - Batching strategies documentation
  - ETrace state management examples
  - Graph visualization tutorials

#### Code Quality Improvements

- **Removed redundant docstrings**: Cleaned up duplicate documentation in `LeakyRateReadout` and `LeakySpikeReadout`
- **Improved code organization**: Streamlined `__all__` definitions across all modules
- **Enhanced readability**: Consistent import structure and better code formatting throughout

#### Dependency Updates

- **Updated `requirements.txt`**: Refined dependency specifications to ensure compatibility with latest `brainstate` and `brainpy` versions
- **Updated `pyproject.toml`**: Bumped version to 0.1.0 and updated project metadata


### Breaking Changes

**API Changes:**
1. **State class renaming** (with deprecation warnings):
   - `ETraceState` → Use `brainstate.HiddenState` instead
   - `ETraceGroupState` → Use `brainstate.HiddenGroupState` instead
   - `ETraceTreeState` → Use `brainstate.HiddenTreeState` instead

2. **Neural network component migration** (with deprecation warnings):
   - Neuron models (IF, LIF, ALIF) → Use `brainpy.state` module
   - Synapse models (Expon, Alpha, etc.) → Use `brainpy.state` module
   - Activation functions → Use `brainstate.nn` module
   - Pooling layers → Use `brainstate.nn` module

3. **Normalization parameter rename**:
   - `normalized_shape` → `in_size` (for LayerNorm, GroupNorm, etc.)

4. **Module file reorganization**:
   - `nn/_rate_rnns.py` → `nn/_rnn.py`
   - Removed: `_neurons.py`, `_synapses.py`, `_elementwise.py`, `_poolings.py`

### Migration Guide

#### For State Management:
```python
# Old (0.0.11)
from brainscale import ETraceState, ETraceGroupState

# New (0.1.0)
from brainstate import HiddenState, HiddenGroupState
```

#### For Neural Network Components:
```python
# Old (0.0.11)
from brainscale.nn import IF, LIF, Expon, ReLU, MaxPool2d

# New (0.1.0)
from brainpy.state import IF, LIF, Expon
from brainstate.nn import ReLU, MaxPool2d
```

#### For Normalization Layers:
```python
# Old (0.0.11)
norm = LayerNorm(normalized_shape=(128,))

# New (0.1.0)
norm = LayerNorm(in_size=128)
```

**Note**: All deprecated APIs include automatic warnings that will guide you to the correct replacements. The old APIs will continue to work in 0.1.0 but will be removed in a future release.

### Version
- Bumped version from `0.0.11` to `0.1.0`



## Version 0.0.11

### Major Changes

#### Import Refactoring
- **Migrated imports from `brainstate` to `braintools`**: All initialization-related imports now use `braintools.init` instead of `brainstate.init`
  - Updated imports in:
    - `brainscale/nn/_neurons.py`: Changed `from brainstate import init` to `from braintools import init`
    - `brainscale/nn/_linear.py`: Changed `from brainstate import init` to `from braintools import init`
    - `brainscale/nn/_conv.py`: Updated initialization imports
    - `brainscale/nn/_synapses.py`: Updated initialization imports
    - `brainscale/nn/_readout.py`: Updated initialization imports

- **Migrated neural network model imports from `brainstate.nn` to `brainpy`**: Updated base classes for neuron models
  - `IF`, `LIF`, `ALIF` now inherit from `brainpy` instead of `brainstate.nn`
  - Maintained API compatibility while using the new `brainpy` backend

- **Updated functional API calls**: Changed from `brainstate.functional.sigmoid` to `brainstate.nn.sigmoid` in RNN cells

#### Dependency Updates
- **Added `brainpy` as a required dependency** in `requirements.txt`

#### Documentation Enhancements
- **Improved docstring formatting across the codebase**:
  - Enhanced parameter documentation with proper type annotations using NumPy-style docstrings
  - Added missing "Returns" sections to property and method docstrings
  - Converted inline examples to proper "Examples" sections with code blocks
  - Updated documentation in:
    - `brainscale/_etrace_algorithms.py`: Enhanced `EligibilityTrace` and `ETraceAlgorithm` documentation
    - `brainscale/_etrace_compiler_base.py`: Improved parameter and return type documentation
    - `brainscale/_etrace_compiler_module_info.py`: Enhanced module documentation

#### Core Algorithm Updates
- **RNN State Management**: Updated all RNN cells to use `braintools.init.param` for state initialization and reset
  - `ValinaRNNCell`: Updated `init_state()` and `reset_state()` methods
  - `GRUCell`: Updated state management and activation functions
  - `CFNCell`: Updated forget and input gate implementations
  - `MGUCell`: Updated minimal gated unit state handling

#### Test Updates
- **Refactored test imports**: Updated test files to use new import paths
  - `brainscale/_etrace_model_test.py`: Updated with new import structure
  - `brainscale/_etrace_vjp_algorithms_test.py`: Aligned with new API

#### Version
- Bumped version from `0.0.10` to `0.0.11`

### Files Changed (17 files)
- `.gitignore`: Added new patterns
- `brainscale/__init__.py`: Updated version number
- `brainscale/_etrace_algorithms.py`: Enhanced documentation and imports
- `brainscale/_etrace_compiler_base.py`: Improved documentation
- `brainscale/_etrace_compiler_graph.py`: Minor updates
- `brainscale/_etrace_compiler_hidden_group.py`: Minor updates
- `brainscale/_etrace_compiler_module_info.py`: Enhanced documentation
- `brainscale/_etrace_model_test.py`: Updated test imports
- `brainscale/_etrace_vjp_algorithms_test.py`: Updated test imports
- `brainscale/_etrace_vjp_graph_executor.py`: Updated imports
- `brainscale/nn/_conv.py`: Migrated to braintools imports
- `brainscale/nn/_linear.py`: Migrated to braintools imports
- `brainscale/nn/_neurons.py`: Migrated to brainpy and braintools
- `brainscale/nn/_rate_rnns.py`: Migrated to braintools and updated functional APIs
- `brainscale/nn/_readout.py`: Updated imports
- `brainscale/nn/_synapses.py`: Updated imports
- `requirements.txt`: Added brainpy dependency

### Breaking Changes
None. All changes maintain backward compatibility at the API level.

### Migration Guide
If you have custom code using brainscale:
- No changes required for end users
- If extending brainscale internally, note that initialization utilities now come from `braintools` instead of `brainstate`


