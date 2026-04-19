# SNN Online Learning Algorithms as ETraceAlgorithm Classes

**Status:** Draft — design approved via brainstorming session 2026-04-19.
**Scope:** Six online learning algorithms for spiking neural networks implemented as
flat subclasses of `braintrace.ETraceVjpAlgorithm`.

## Background

braintrace v0.2 ships two online-learning algorithm classes:

- `D_RTRL` / `ParamDimVjpAlgorithm` — per-parameter trace of shape `(*w.shape, num_hid)`, O(P·H) memory.
- `pp_prop` / `IODimVjpAlgorithm` — factored I/O trace, O((I+O)·H) memory.

Both assume the trace update rule
`eps^t = D^t · eps^{t-1} + diag(D_f^t) ⊗ x^t` and compute the learning signal
`dL/dh^t` via JAX reverse-AD of the loss through the hidden states.

The research report `research/snn-online/report.md` documents six online learning
algorithms for SNNs (e-prop, OSTL, OTPE, OTTT, OSTTP, ETLP) which the community
commonly cites. Each has paper-specific details (κ-filters, regime flags, target
projections, event-driven gating) that do not naturally surface from a generic
`D_RTRL` invocation. Users today must hand-wire these details per paper; this
design adds six first-class classes that encapsulate the paper-faithful logic
and expose paper-named public symbols.

## Goals

1. Ship six public classes: `EProp`, `OSTL`, `OTPE`, `OTTT`, `OSTTP`, `ETLP`.
2. Each class is self-contained, paper-faithful, and independently documented.
3. Reuse existing `ETPPrimitive` + compiler infrastructure; no new primitives,
   no compiler invariant changes.
4. Match existing `D_RTRL` conventions: `compile_graph()`, `update()`,
   `reset_state()`, `get_etrace_of()`, `ParamState`-keyed gradient output.
5. Support both batched (`etp_mm_p`) and unbatched (`etp_mv_p`) dispatch via
   existing `braintrace.matmul` auto-routing.

## Non-goals

- **MultiStepData input** deferred to v2; v1 handles `SingleStepData` only.
- **All paper variants** not covered; each class ships canonical + one knob.
- **Compiler extension** for OTPE cross-layer R_hat not pursued; OTPE stores
  R_hat as model-side `ShortTermState` and computes the cross-layer propagation
  inside `_solve_weight_gradients`, bypassing the W→W→h invariant.
- **New ETP primitive** not introduced; all six classes reuse `etp_mm_p`,
  `etp_mv_p`, `etp_conv_p`, `etp_elemwise_p`.

## Architecture

New top-level subpackage parallel to `_etrace_vjp/`:

```
braintrace/
  _snn_algorithms/
    __init__.py          # public re-exports
    _common.py           # PresynapticTrace, FixedRandomFeedback, KappaFilter, extract_y_target
    e_prop.py            class EProp(ETraceVjpAlgorithm)
    ostl.py              class OSTL(ETraceVjpAlgorithm)
    otpe.py              class OTPE(ETraceVjpAlgorithm)
    ottt.py              class OTTT(ETraceVjpAlgorithm)
    osttp.py             class OSTTP(ETraceVjpAlgorithm)
    etlp.py              class ETLP(ETraceVjpAlgorithm)
    tests/
      test_e_prop.py
      test_ostl.py
      test_otpe.py
      test_ottt.py
      test_osttp.py
      test_etlp.py
      test_integration.py
      test_cross_check.py
      fixtures/
        bptt_gradients_tiny_lsnn.pkl   # reference BPTT gradient dumps
```

Public API (`braintrace.__init__`):

```python
from braintrace._snn_algorithms import EProp, OSTL, OTPE, OTTT, OSTTP, ETLP
__all__ += ['EProp', 'OSTL', 'OTPE', 'OTTT', 'OSTTP', 'ETLP']
```

### Base class extension

Add one new overridable hook on `ETraceVjpAlgorithm` (in `_etrace_vjp/base.py`):

```python
def _compute_learning_signal(
    self,
    dl_to_hidden_from_autodiff: Sequence[jax.Array],
    args: tuple,
) -> Sequence[jax.Array]:
    """Override to replace reverse-AD learning signal with alternative (e.g., target projection)."""
    return dl_to_hidden_from_autodiff
```

Called inside `_update_fn_bwd` after `dl2h_at_t_or_t_minus_1` is computed but before
`_solve_weight_gradients`. Default returns the autodiff result unchanged.
`EProp` / `OSTL` / `OTPE` / `OTTT` inherit default. `OSTTP` / `ETLP` override.

### Update signature

Extended on the base `update()`:

```python
def update(self, x, y_target=None, teaching_spike=None):
    ...
```

Default values mean existing callers unaffected. EProp/OSTL/OTPE/OTTT call
`algo.update(x)`. OSTTP calls `algo.update(x, y_target)`. ETLP calls
`algo.update(x, y_target, teaching_spike)`.

## Per-class responsibility matrix

| Class | Trace shape (per layer) | `_update_etrace_data` | `_solve_weight_gradients` | `_compute_learning_signal` | Extra model-side state |
|---|---|---|---|---|---|
| **EProp** | `(batch, I, O, num_hid)` via D_RTRL; κ-filter sidecar | inherit D_RTRL + append κ-filter step | inherit + apply κ-filtered trace `ē_ji` | default; knob for random-feedback readout swap | none |
| **OSTL** | `pp_prop`-shape (FF / w-o-H) or D_RTRL-shape (w-H) | inherit corresponding base | inherit | default | none |
| **OTPE** | canonical: `(batch, I, O)` R_hat; approx: `(batch, I)` + `(batch, O)` | custom: `R̂ ← λ · R̂ + ∂s/∂θ_local` (additive, **hid2hid_jac ignored**) | custom: `(L · W_next^T) ⊗ R_hat` | default | per-layer `R_hat` `ShortTermState` (+ `ḡ`, `ẑ` for approx) |
| **OTTT** | `(batch, I)` λ-accumulator | custom: `â ← λ â + x_t` (both jacobians ignored) | custom: `outer(â, L · σ'(u))` | default | per-layer `â` `ShortTermState` |
| **OSTTP** | OSTL trace (D_RTRL shape) | inherit D_RTRL | inherit (`L_l ⊙ e_l`) | **override: `[B_l @ y_target for each HiddenGroup]`** | `B_list` held on algo (frozen `jnp` buffers) |
| **ETLP** | `(batch, I)` ε_pre + `(batch, I, O)` ε_adapt (ALIF) | custom: `ε_pre ← α ε_pre + x`; ALIF coupled `ε_adapt` recursion using φ stashed by neuron cell | custom: `outer(ε_pre − θ ε_adapt, I · φ)` | **override: `(B_l @ y_target) · teaching_spike`** | per-layer `ε_pre` + optional `ε_adapt` |

### Canonical knob per class

| Class | Constructor flag | Values | Default |
|---|---|---|---|
| EProp | `feedback` | `'symmetric'`, `'random'` | `'symmetric'` |
| OSTL | `regime` | `'with-H'`, `'without-H'` | `'with-H'` |
| OTPE | `mode` | `'full'`, `'approx'` | `'full'` |
| OTTT | `mode` | `'A'` (accumulated), `'O'` (per-step) | `'A'` |
| OSTTP | `target_timing` | `'per-step'`, `'sequence-end'` | `'per-step'` |
| ETLP | `neuron` | `'ALIF'`, `'LIF'` | `'ALIF'` |

### Shared helpers (`_common.py`)

- `class PresynapticTrace(brainstate.ShortTermState)` — λ-accumulator factory. Used by OTTT, ETLP, OTPE-Approx.
- `class FixedRandomFeedback` — wraps a frozen `jnp` buffer `B_l` with `jax.lax.stop_gradient` guard; constructor takes `(n_target, n_layer, key, init_scale)`. Used by OSTTP, ETLP, EProp-random.
- `class KappaFilter(brainstate.ShortTermState)` — low-pass output-side filter `x_filt ← (1-κ) x + κ x_filt`. Used by EProp's `ē = F_κ(e)` step.
- `def extract_y_target(args) -> Optional[jax.Array]` — extract from `update()` positional args, return `None` if absent.
- `def extract_teaching_spike(args) -> Optional[jax.Array]` — same for ETLP.

## Data flow

### Forward pass (all 6 classes)

User model calls `braintrace.matmul(x, w)`. Compiler registers
`HiddenParamOpRelation`(s) via `etp_mm_p` (or `etp_mv_p` unbatched).
Base `ETraceVjpAlgorithm.update()` wraps `jax.custom_vjp` which runs
`solve_h2w_h2h_jacobian()` → yields `hid2weight_jac` (x, df) and `hid2hid_jac` (D).
No change to existing flow.

### Trace update

`_update_etrace_data(running_index, hist_etrace_vals, hid2weight_jac, hid2hid_jac, weight_vals, input_is_multi_step)`
called per step. Class-specific behavior:

- **EProp / OSTL / OSTTP** — inherit from `D_RTRL._update_etrace_data` or `pp_prop._update_etrace_data`.
- **OTPE** — ignore `hid2hid_jac`; compute `R̂ ← λ · R̂ + hid2weight_jac_local`. λ pulled from model (neuron leak) via constructor arg or discovered from first `ShortTermState` with attribute `leak`.
- **OTTT** — ignore both jacobians; update `â ← λ â + x_t` using raw input spike from `args`. λ pulled same way as OTPE: constructor arg `leak: float`, or auto-discovered from the first `ShortTermState` on the neuron cell with attribute `leak`; fail fast with `ValueError` if neither resolves.
- **ETLP** — ignore `hid2hid_jac`; use `hid2weight_jac[0]` (x) for `ε_pre`; ALIF coupled recurrence using φ stashed by neuron cell in a `ShortTermState` field named `phi_last`.

### Learning signal routing

Inside `_update_fn_bwd`, base computes `dl2h_at_t` via reverse-AD. New hook
`_compute_learning_signal(dl2h_at_t, args)` runs next. Default returns
`dl2h_at_t` unchanged. OSTTP / ETLP override to return
`[B_l @ y_target for each HiddenGroup]`. Reverse-AD result discarded for those
two (wasted compute accepted — XLA may optimize away since the unused output
feeds a downstream op whose gradient is zero).

### Event-driven gating (ETLP only)

Teaching-spike stream enters via third positional arg to `update()`. Consumed
inside `_compute_learning_signal` — output `signal = B_l @ y_target *
teaching_spike`. When teaching-spike is 0, signal is 0 → zero gradient.

### Compiler relations

- EProp / OSTL / OSTTP / ETLP — compiler relations unchanged; each ETP weight reaches its own hidden state via one non-gradient-enabled primitive.
- OTTT — same; `â` is model input to matmul, not a hidden output of ETP primitive.
- OTPE — **bypasses compiler** for cross-layer R_hat routing. OTPE stores R_hat per-layer as model-side `ShortTermState` updated inside the model's `update()` (not via ETP machinery). Compiler registers standard one-hop relations per weight. Cross-layer coupling math happens inside `OTPE._solve_weight_gradients`: when computing `ΔW^{l-1}`, OTPE reaches through `W_l^T` to layer-l learning signal, then outer-products with `R_hat[l-1]`.

### Compiler invariant

No relaxation required. Each ETP weight reaches its own hidden state via one
non-gradient-enabled primitive. Cross-layer coupling in OTPE is a higher-level
algorithm concern, not a compiler concern.

## Error handling and edge cases

### Construct-time validation

| Class | Check | Error |
|---|---|---|
| all | `model` is `brainstate.nn.Module` | `TypeError` |
| OSTL | `regime='with-H'` requires at least one recurrent `braintrace.matmul(y_prev, H)` | `CompilationError` during `compile_graph()` |
| OTPE | per-layer λ leak provided (constructor arg or discovered from model) | `ValueError` if `None` and auto-discovery fails |
| OTPE | `mode='approx'` + > 1 hidden layer | `UserWarning` ("F-OTPE variant recommended; bias compounds with depth") |
| OTTT | subtraction-reset LIF required; neuron has `stop_gradient` on spike feedback | runtime warning if `hid2hid_jac` contains non-λ diagonal structure |
| OSTTP | `len(B_list) == len(HiddenGroup)` | `ValueError` |
| OSTTP | each `B_l.shape == (n_target, n_l)` | `ValueError` |
| ETLP | `len(B_list) == number of hidden layers` (output layer uses MSE-derived signal) | `ValueError` |
| ETLP | `neuron='ALIF'` requires `ε_adapt` state on layer | `ValueError` |

### Runtime validation (`update()`)

- OSTTP / ETLP: `y_target is None` → `ValueError('target required for target-projection algorithm')`.
- ETLP: `teaching_spike is None` → default to 1.0 (always-on teaching) + one-shot `UserWarning`.
- Shape mismatch `y_target.shape[-1] != n_target` → `ValueError`.

### Compile-time OTPE invariant

OTPE's `compile_graph()` verifies no `HiddenParamOpRelation` spans more than one
layer (i.e., each relation's reachable hidden states all live within a single
`HiddenGroup`). If user mistakenly wires recursive `braintrace.matmul` chains
that trigger a multi-group relation, emit `CompilationError('OTPE requires
per-layer one-hop weight-to-hidden relations; found cross-layer relation X')`.

### Soft-reset spike-leakage diagnostic

Best-effort jaxpr inspection, not a hard guarantee. Added to
`_etrace_compiler/diagnostics.py` as new `DiagnosticKind.SPIKE_RESET_LEAKAGE`:

1. After `compile_graph()`, each `HiddenParamOpRelation`'s transition jaxpr is walked.
2. For each eqn whose input var is another `HiddenState` value, check: does that var originate from a `stop_gradient_p` eqn (or pre-stop-gradient'd source)?
3. If **not** stop-gradient'd AND the originating state's upstream jaxpr contains a `custom_vjp_call_jaxpr` primitive whose forward jaxpr has a comparison op (`gt_p`, `ge_p`), emit warning via `CompilationRecord`.

**Limits:**
- False negatives — smooth surrogate without threshold-comparison op missed.
- False positives — legitimate direct-spike-coupling still warns (accept, since author should stop-gradient anyway).
- Cannot catch runtime-dynamic leakage (doesn't exist — jaxpr static).

Each of the six SNN classes enables this diagnostic by default via constructor
flag `check_spike_reset_leakage: bool = True`.

### Numerical stability knobs (opt-in)

- `normalize_matrix_spectrum: bool = False` — reuse D_RTRL's branch-free spectral clip. Applicable: EProp, OSTL, OSTTP.
- `trace_clip_abs: Optional[float] = None` — elementwise `clip(t, -c, c)` per step. Applicable: OTPE (λ near 1), ETLP (long-sequence ε_pre saturation).
- `kappa_filter_decay: float = 0.0` — EProp κ-filter; 0 disables.

### Other edge cases

1. **Soft-reset** — neuron cell must `jax.lax.stop_gradient(self.z.value)` on reset term. Best-effort detected by jaxpr diagnostic (above). Documented in each class docstring + example.
2. **saiunit quantities** — each trace-update helper must call `u.split_mantissa_unit` / recombine. Reuse pattern from `_etrace_vjp/d_rtrl.py::_remove_units`.
3. **Batch=1 dispatch** — `x.ndim < 2` auto-routes through `etp_mv_p`. Trace shape drops leading batch axis. All six classes tested for both paths.
4. **Readout exclusion** — EProp / OSTL / OTPE / OTTT document: readout `W_out` should use plain `x @ w`, NOT `braintrace.matmul`. OSTTP / ETLP readout can be ETP-traced. Enforced by example only, not by code (user responsibility).
5. **GRUCell caveat** — 3 matmuls → 2 relations (Wz, Wh). Document in OSTL docstring that recurrent cells with Wr-style branches get implicit filter. Reuse existing braintrace behavior.
6. **Reset semantics** — `reset_state(batch_size)` resets `running_index` to 0 and zeros all algo-owned `ShortTermState`. Default from base class covers this; subclasses extend only if they add model-side state (OTPE R_hat, OTTT â, ETLP ε_pre/ε_adapt).

## Testing

Per-class unit tests (`_snn_algorithms/tests/test_<algo>.py`) — cross-cutting matrix:

| Test | EProp | OSTL | OTPE | OTTT | OSTTP | ETLP |
|---|---|---|---|---|---|---|
| Shape: trace allocation matches paper Table 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Forward pass matches paper toy model (LIF/ALIF/SNU/LIF-sub) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Gradient agreement vs BPTT on tiny net (2-neuron, 5-step) | cosine > 0.95 (bias expected) | cosine ≈ 1.0 (single-layer exact) | cosine > 0.6 (vs OSTL > 0.2) | cosine > 0.5 | — | — |
| Target-projection signal matches `B @ y_target` exactly | — | — | — | — | ✓ | ✓ |
| Event-driven gating (teaching_spike=0 → zero grad) | — | — | — | — | — | ✓ |
| Batched ↔ unbatched numerical equivalence | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `reset_state()` clears all algo state | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Knob flag exercises both modes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| saiunit quantities preserved through trace update | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Spike-reset leakage diagnostic fires when neuron lacks `stop_gradient` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Compile-time error: missing `y_target` at update() | — | — | — | — | ✓ | ✓ |
| Compile-time error: OTPE cross-layer relation rejected | — | — | ✓ | — | — | — |

**Integration tests** (`tests/test_integration.py`):
- Each of 6 classes trains small SHD-like toy task for 100 steps; loss decreases monotonically after warmup.
- Swap classes with same model definition: ensure public API identical except optional args.

**Cross-check tests** (`tests/test_cross_check.py`):
- `OSTL(regime='without-H')` with linear (identity-surrogate) neuron == `D_RTRL` on same model (bitwise match expected).
- `EProp(feedback='symmetric', num_hid=1)` with LIF neuron == `D_RTRL` up to κ-filter (within float tolerance).
- `OTPE(mode='full')` with `λ=0` reduces to `OSTL` (R_hat degenerates to `∂s/∂θ` per step).

**Reference-gradient fixtures** (`tests/fixtures/`):
- Pickled BPTT gradient dumps for tiny LSNN on 10-step sequence (produced once via `brainstate.transform.grad` through full unroll).
- Each algo test compares against fixture with documented cosine-similarity threshold per paper's reported values.

**Documentation tests** (doctest):
- Each class docstring has a ≤5-line runnable example.
- `pytest --doctest-modules braintrace/_snn_algorithms/` passes.

**Test budget**: ~40 unit + 6 integration + 3 cross-check ≈ 50 tests. Matches density of existing `_etrace_vjp/tests/`.

## Example usage

### EProp (simplest case — symmetric feedback LSNN)

```python
import brainstate, braintrace, braintools, jax.numpy as jnp

class LSNN(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.w_rec = brainstate.ParamState(braintools.init.KaimingNormal()((n_in + n_rec, n_rec)))
        self.lif = braintrace.nn.LIFCell(n_rec)   # assume LIFCell exists or user-defined
        self.w_out = brainstate.ParamState(braintools.init.KaimingNormal()((n_rec, n_out)))

    def update(self, x):
        pre = jnp.concatenate([x, jax.lax.stop_gradient(self.lif.z.value)], axis=-1)
        I = braintrace.matmul(pre, self.w_rec.value)
        z = self.lif(I)
        return z @ self.w_out.value       # readout excluded from ETP

net = LSNN(n_in=80, n_rec=200, n_out=10)
algo = braintrace.EProp(net, feedback='symmetric', kappa_filter_decay=0.9)
algo.compile_graph(sample_x)
```

### OSTTP (target-projection, per-layer B)

```python
B_list = [jax.random.normal(key, (n_out, n_rec)) * 0.1]
algo = braintrace.OSTTP(net, B_list=B_list, target_timing='per-step')
algo.compile_graph(sample_x, sample_y)
# Training loop:
for x_t, y_t in stream:
    algo.update(x_t, y_target=y_t)
```

### ETLP (event-driven ALIF)

```python
algo = braintrace.ETLP(alif_net, B_list=[B], neuron='ALIF')
algo.compile_graph(sample_x, sample_y, sample_teaching)
for x_t, y_t, teach_t in stream:
    algo.update(x_t, y_target=y_t, teaching_spike=teach_t)
```

## Implementation phases

1. **Base class extension** — add `_compute_learning_signal` hook to `ETraceVjpAlgorithm`; wire into `_update_fn_bwd`. Update existing `D_RTRL`/`pp_prop` tests to confirm no regression.
2. **`_common.py` helpers** — `PresynapticTrace`, `FixedRandomFeedback`, `KappaFilter`, arg-extraction utilities.
3. **Tier-1 classes** — `OSTL` (thin D_RTRL/pp_prop wrapper with regime flag), `EProp` (D_RTRL + κ-filter + feedback knob). Unit tests + cross-check.
4. **Tier-2 easy** — `OTTT` (pre-only trace; clean break from D_RTRL shape). `OSTTP` (inherits OSTL trace, overrides `_compute_learning_signal`).
5. **Tier-2 harder** — `ETLP` (pre + ALIF adapt trace, event gating). `OTPE` (leaky-additive trace, cross-layer routing inside `_solve_weight_gradients`, compile-time invariant check).
6. **Integration + cross-check tests** + doctest enablement + public re-exports.
7. **Docs** — tutorial showing each class side-by-side on SHD toy task; migration note for users currently hand-wiring e-prop via `D_RTRL`.

## Open questions (resolved during brainstorming)

- **Scope**: all 6 classes in one spec. ✓
- **API shape**: 6 flat subclasses, no mixin composition. ✓
- **Variants**: canonical + one knob per class. ✓
- **Multi-step**: single-step only in v1. ✓
- **Batching**: both batched + unbatched. ✓
- **Target routing**: extra positional arg `update(x, y_target)`. ✓
- **Layout**: new top-level `braintrace/_snn_algorithms/`. ✓
- **Spike-reset leakage detection**: best-effort jaxpr diagnostic. ✓

## Out of scope for this spec

- SHD / N-MNIST full-scale benchmark validation against paper numbers (future benchmarking effort).
- Hardware-specific fixed-point quantization for ETLP FPGA deployment.
- Multi-layer extensions for ETLP (paper-acknowledged as single-hidden-layer-only).
- `MultiStepData` support — deferred to v2.
- Conv variants of ETLP / OSTTP — deferred; the underlying `etp_conv_p` primitive already supports the forward pass.
