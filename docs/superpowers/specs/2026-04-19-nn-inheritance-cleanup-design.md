# Design: `braintrace/nn/` inheritance cleanup

**Date:** 2026-04-19
**Scope:** `braintrace/nn/_conv.py` and `braintrace/nn/_normalizations.py`
**Pattern:** Mirror the thin-shim style established in `braintrace/nn/_linear.py` — subclass `brainstate.nn` counterparts and override only the ETP-routing hook.

## Background

`braintrace/nn/_linear.py` subclasses `brainstate.nn.Linear` (and `SignedWLinear`, `ScaledWSLinear`, `SparseLinear`, `LoRA`) and overrides only `update` to route the matmul through `braintrace.matmul` (an ETP primitive). This is compact (~40 LOC of logic) and keeps behavioural parity with upstream.

Four sibling files could, in principle, adopt the same pattern. This design documents which are in scope and why.

| File | In scope? | Reason |
|---|---|---|
| `_linear.py` | n/a | already done |
| `_conv.py` | **yes** | full parent-class coverage in `brainstate.nn` |
| `_normalizations.py` | **yes** | cosmetic rewrite; already inherits but via a redundant `_BatchNormETrace` intermediate |
| `_readout.py` | **no** | `brainstate.nn.LeakyRateReadout` is deprecated; `brainpy.state.LeakyRateReadout` uses `in_size` for `tau` shape where braintrace uses `out_size` — would propagate a latent shape bug |
| `_rnn.py` | **no** | `brainstate.nn.{GRU,LSTM,URLSTM}Cell` use fused-gate Linear layouts; braintrace uses per-gate Linears. Adopting fused layouts would change ETP primitive-relation math and break attribute names (`Wz`, `Wr`, `Wi`, `Wg`, `Wf`, `Wo` → `Wrz`, `W`) |

## Non-goals

- Re-examining ETP primitive rules.
- Changing semantic behaviour of convolution or normalization layers (defaults preserved).
- Adding new features (e.g. exposing `channel_first` to convolutions — it becomes incidentally available via parent, but not advertised as a feature of this change).

## Design: `_conv.py`

### Current

`_conv.py` (397 LOC) defines a private `_Conv(brainstate.nn.Module)` base plus `Conv1d/2d/3d`. It re-implements dimension-number construction (`to_dimension_numbers`), kernel-shape inference, per-element `replicate`, padding normalization, and a batched-only forward path that calls `braintrace.conv` (the ETP primitive).

### Proposed

Delete `_Conv`, `to_dimension_numbers`, `replicate`. Subclass `brainstate.nn.Conv{1,2,3}d` and override only the `_conv_op(self, x, params)` hook — the exact extensibility point exposed by brainstate's convolution classes.

```python
# braintrace/nn/_conv.py  (target ≈ 60 LOC incl. headers)
import brainstate

from braintrace._etrace_op import conv as etp_conv

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


class Conv1d(brainstate.nn.Conv1d):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.Conv1d.__doc__.replace('brainstate', 'braintrace')

    def _conv_op(self, x, params):
        w = params['weight']
        if self.w_mask is not None:
            w = w * self.w_mask
        b = params.get('bias')
        return etp_conv(
            x, w, b,
            strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.rhs_dilation,
            feature_group_count=self.groups,
            dimension_numbers=self.dimension_numbers,
        )


class Conv2d(brainstate.nn.Conv2d):
    # identical body, docstring rewritten from parent
    ...

class Conv3d(brainstate.nn.Conv3d):
    ...
```

### Why this works

- `brainstate.nn.Conv{1,2,3}d` store parameters in a single `self.weight: ParamState` whose value is a dict with key `'weight'` and optional `'bias'`. `etp_conv(x, w, b=None, ...)` accepts `bias=None`, so `params.get('bias')` composes cleanly.
- Parent's `update` already handles batched/unbatched dispatch (`_check_input_dim` + `expand_dims` + `squeeze`), so no custom forward pass is needed.
- All public attributes the test suite reads (`in_channels`, `kernel_size`, `dimension_numbers`, `stride`, `out_size`, `w_mask`, `groups`, `padding`) are already provided by the parent with the same names and semantics.
- `dimension_numbers` produced by parent matches current braintrace layout (verified by probe: 2D gives `rhs_spec=(3, 2, 0, 1)`, kernel shape `(H, W, in, out)`).

### Breaking changes

| Old attribute | New access path |
|---|---|
| `self.kernel` (`ParamState`) | `self.weight.value['weight']` |
| `self.bias` (`ParamState` or `None`) | `self.weight.value.get('bias')` |

`grep` across `braintrace/` finds no reads of `.kernel` or `.bias` on a conv instance (tests read `in_channels`/`kernel_size`/`dimension_numbers`/`stride`/shapes, never the parameter attributes directly).

### Edge cases

- `b_init=None` → brainstate parent omits `'bias'` from the weight dict (verified); `params.get('bias')` returns `None`; `etp_conv` receives `bias=None`.
- `w_mask` propagation preserved by applying inside `_conv_op`, same as before.
- Grouped / depthwise convolution parameters (`groups`, `feature_group_count`) unchanged.
- Unbatched input — parent's `update` expands/squeezes around `_conv_op`, so our override receives always-batched `x`.
- `channel_first` — parent accepts it; braintrace does not advertise it, but passing it through is harmless.

## Design: `_normalizations.py`

### Current

`_normalizations.py` (464 LOC) defines a `_BatchNormETrace(_BatchNorm)` intermediate whose only behaviour is to forward kwargs into parent with `param_type=NormalizationParamState`. `BatchNorm0d/1d/2d/3d` set `num_spatial_dims` on `_BatchNormETrace`. `LayerNorm`, `RMSNorm`, `GroupNorm` already inherit brainstate equivalents and pass `param_type=NormalizationParamState` the same way. Each class carries a copy of the upstream docstring.

### Proposed

Collapse to thin subclasses, one per class, no intermediate:

```python
# braintrace/nn/_normalizations.py  (target ≈ 80 LOC)
import brainstate
from brainstate.nn._normalizations import NormalizationParamState

__all__ = [
    'BatchNorm0d', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
    'LayerNorm', 'RMSNorm', 'GroupNorm',
]

_NORM_PARAM = NormalizationParamState


class BatchNorm0d(brainstate.nn.BatchNorm0d):
    __module__ = 'braintrace.nn'
    __doc__ = brainstate.nn.BatchNorm0d.__doc__.replace('brainstate', 'braintrace')

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('param_type', _NORM_PARAM)
        super().__init__(*args, **kwargs)


# BatchNorm1d / BatchNorm2d / BatchNorm3d — identical body over the matching parent
# LayerNorm / RMSNorm / GroupNorm — same pattern
```

### Why

- `_BatchNormETrace` adds no semantic value: its `__init__` is a pure pass-through with one default. `kwargs.setdefault('param_type', _NORM_PARAM)` at each leaf expresses the same contract in fewer lines.
- Duplicated long docstrings are replaced with `__doc__ = parent.__doc__.replace('brainstate', 'braintrace')`, matching `_linear.py`.
- Norm layers do not route through ETP primitives, so `param_type=NormalizationParamState` is still the only required ETP-specific setting.

### Breaking changes

None. Public constructor signatures, attribute names, and runtime behaviour are preserved. `NormalizationParamState` is still the `param_type`.

### Edge cases

- Users passing `param_type=` explicitly still override the default (`setdefault` respects user input).
- brainstate's LayerNorm/RMSNorm use `use_bias`/`use_scale` (not `affine`) — already handled by current `*args, **kwargs` forwarding, preserved.
- Import path for `NormalizationParamState` remains `brainstate.nn._normalizations` (private module, same as today).

## Out-of-scope files (decision record)

### `_readout.py` — keep standalone

`brainpy.state.LeakyRateReadout` sets `self.tau = param(tau, self.in_size)`, but `decay * self.r.value` needs the `out_size` shape. If `in_size != out_size`, the parent's shape is wrong. The current `_readout.py` uses `out_size`, which is correct. Inheriting from `brainpy.state` would silently propagate the bug, so we keep the standalone class.

### `_rnn.py` — keep standalone

`brainstate.nn.GRUCell/LSTMCell/URLSTMCell` use fused-gate Linears (`Wrz`, `W`). braintrace uses per-gate Linears (`Wz`/`Wr`, `Wi`/`Wg`/`Wf`/`Wo`, `Wu`/`Wf`/`Wr`/`Wo`). Adopting the fused layout would:

1. Rename public attributes (breaking).
2. Change ETP per-primitive gradient rules (different weight shape per primitive, different `xy_to_dw` / `yw_to_w` inputs).
3. Require re-analysing the per-cell reachability invariant documented in `CLAUDE.md` ("GRUCell has 3 Linears but only 2 ETP relations").

The risk/reward is not justified by the line-count reduction.

## Testing

Existing tests remain authoritative:

- `braintrace/nn/_conv_test.py` (868 LOC): exercises shape inference, dimension numbers, padding, stride, dilation, groups, bias on/off, batched/unbatched dispatch, gradient/ETP integration. All attribute reads are on shared names (`in_channels`, `kernel_size`, `dimension_numbers`, `stride`, `out_size`, `w_mask`). No test reads `.kernel` or `.bias`.
- `braintrace/nn/_normalizations_test.py` (695 LOC): constructor signatures and runtime behaviour only — unaffected by the refactor.

### New edge cases to add (minimal)

- Conv: assert that `.weight.value['weight']` is a `ParamState` accessible after construction, and that `.weight.value.get('bias')` returns `None` when `b_init=None`.
- Conv: round-trip a conv through `etp_conv` rules — confirm ETP compilation produces the same per-primitive relations as the pre-refactor implementation on a small reference model (e.g. single `Conv2d` into a `HiddenState`).

## Rollout

Single PR. No deprecation shims — `self.kernel` / `self.bias` are not documented public attributes (the docstring advertises `out_channels`, `kernel_size`, etc., not parameter container names), and no callers read them.

Sequence:

1. Rewrite `_conv.py` to the thin-shim form.
2. Rewrite `_normalizations.py` to remove `_BatchNormETrace`.
3. Run `python -m pytest braintrace/nn/ -v` — fix regressions inline.
4. Run full `python -m pytest braintrace/ -v` — confirm compiler-side tests still see the same ETP primitive relations.
5. Commit.

## Open questions

None remaining — clarifying questions resolved:

- `_rnn.py` — excluded (user: "skip rewrite/update").
- `_readout.py` — excluded (tau-shape bug in parent).
- `_normalizations.py` — thin shims without `_BatchNormETrace`.
- `_conv.py` — inherit + override `_conv_op` hook.
