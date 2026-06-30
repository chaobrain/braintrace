# Release Notes


## Version 0.2.3

This release adds optional, shape-preserving parameter-transform hooks to the
eligibility-trace (ETP) operators, so a trainable weight (or bias) can be passed
through an elementwise / standardizing function *before* it enters the operation
while the eligibility trace and gradient remain with respect to the **raw**
stored parameter. It also threads these hooks through the `braintrace.nn` linear
layers. One public API is renamed (see Breaking changes).

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

### Breaking changes

- **`braintrace.element_wise`**: the `fn` parameter is renamed to **`weight_fn`**
  and is now **keyword-only**, and the transform is applied *inside* the ETP
  primitive (previously it was applied to the weight outside the primitive).
  Migrate `element_wise(w, fn=g)` to `element_wise(w, weight_fn=g)`. Forward
  results are unchanged; only the call signature and the internal
  trace-factorization point differ.


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
  layouts, and **fixed non-square dense weight broadcasting** in `_mm_yw_to_w`.
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


