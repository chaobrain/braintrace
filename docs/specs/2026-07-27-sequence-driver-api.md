# Sequence driver API — `etrace_grad` and `etrace_evolve`

Status: spec, awaiting implementation
Baseline: commit `8b7cdc7`
Target release: 0.3.0

## Goal

Give users a first-class way to run a compiled learner over a **sequence** —
accumulating online gradients when a loss applies, and advancing hidden state
plus eligibility trace when it does not — without hand-writing a
`scan`-accumulate loop.

Today every training script re-implements the same block. It appears 25 times
across `examples/`, in `examples/drtrl/*`, `examples/pp_prop/*`,
`examples/00x`–`examples/10x`, and again in the docs notebooks:

```python
def step_loss(inp, tar):
    out = learner(inp)
    return braintools.metric.squared_error(out, tar).mean(), out

def grad_step(prev_grads, x):
    inp, tar = x
    f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
    cur_grads, local_loss, _ = f_grad(inp, tar)
    return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
opt.update(grads)
```

Sixteen lines of transform plumbing that say nothing about the model, the task,
or the learning rule. Worse, the block is easy to get subtly wrong: `scan`
*sums* the per-step gradients while the reported objective is `losses.mean()`,
so `examples/100-gru-on-copying-task.py:166` carries a hand-written
`grads = jax.tree.map(lambda g: g / losses.shape[0], grads)` correction and a
paragraph of comment explaining it. That correction is exactly the kind of thing
a library should own.

The no-loss half is duplicated too, as a bare `for_loop` over the learner —
`examples/100-gru-on-copying-task.py:178` warms the hidden states and traces for
`n_seq + 10` steps before the learning window opens.

## Scope

**In:**

1. `ETraceAlgorithm.etrace_grad` — accumulated online gradients over a sequence,
   with per-step loss masking and multi-step VJP windows.
2. `ETraceAlgorithm.etrace_evolve` — drive the model and the trace forward with
   no gradient computation anywhere.
3. `braintrace.ETraceVmap` — a `brainstate.nn.Vmap` subclass carrying the same
   two methods, returned by `compile(..., vmap=True)` so the call site is
   identical in batched and unbatched mode.
4. Migration of all 25 call sites in `examples/` and the docs.

**Out:**

- Any optimizer involvement. `etrace_grad` returns gradients; the caller writes
  `opt.update(grads)`. `braintools.optim` does not become a dependency of this
  path.
- Gradient clipping. `brainstate.nn.clip_grad_norm(grads, 1.0)` is already one
  line and orthogonal.
- A stateful `Trainer` object owning learner + optimizer + loss.
- Any change to `update()`, to the trace algebra, or to any algorithm's numerics.
  This is a driver layer; it composes existing public behaviour and must not
  alter it.

## Public API

### `etrace_grad`

```python
learner.etrace_grad(
    loss_fn,
    inputs,
    targets=None,
    *,
    mask=None,
    chunk_size=1,
    weights=None,
    reduction='mean',
    loss_output='per_step',
    has_aux=False,
    return_value=False,
)
```

**`loss_fn`** — called **per step**, regardless of `chunk_size`:

- `loss_fn(out_t, target_t) -> scalar` when `targets` is given;
- `loss_fn(out_t) -> scalar` when `targets` is `None` (a fixed label is closed
  over — see `examples/pp_prop/_shared.online_train_epoch_fixed_target`);
- returns `(scalar, aux)` instead when `has_aux=True`.

That the signature does not change with `chunk_size` is the load-bearing
property of this design. Users tune the window size — a real
accuracy/memory knob — without touching their loss.

**`inputs`** — a pytree whose leaves share a leading length `T`. A `tuple` is
unpacked as several positional model arguments; anything else is passed as a
single argument. Inputs that are constant across time must be broadcast to `T`
by the caller or closed over.

**`targets`** — a pytree sliced along the leading axis in lockstep with
`inputs`, or `None`.

**`mask`** — a `(T,)` array of 0/1 (or bool). `None` means all-ones. It gates
**only the loss**. The model and the eligibility trace are driven at every step
regardless — which is the entire point: a masked step still shapes the trace
that later unmasked steps consume, so masking is *not* the same as shortening
the sequence.

**`chunk_size`** — steps per window. At `k > 1` the learner is called once per
window with a `MultiStepData`, which is the multi-step VJP path. `T` must be a
multiple of `k`.

**`weights`** — the `ParamState`s to differentiate. Defaults to the learner's
own `param_states`, which is verified key-identical *and object-identical* to
`model.states(brainstate.ParamState)`, so an optimizer registered the usual way
accepts the returned tree unchanged. Pass explicitly to freeze a subset.

**`reduction`** — defines the scalar objective that is differentiated:

| value | objective |
| --- | --- |
| `'mean'` (default) | `sum_t mask_t · loss_t / max(sum_t mask_t, 1)` |
| `'sum'` | `sum_t mask_t · loss_t` |

`'mean'` divides by the number of **unmasked** steps, not by `T`. It is the
default because it is the scale people actually report and compare against BPTT,
and it retires the hand-written `grads / losses.shape[0]` correction.

**`loss_output`** — what `return_value=True` hands back:

| value | returns |
| --- | --- |
| `'per_step'` (default) | `(T,)` of the raw `loss_fn` values, **pre-mask** |
| `'masked'` | `(T,)` of `mask_t · loss_t` |
| `'scalar'` | the reduced objective itself |

`loss_output` is ignored when `return_value=False`.

`'per_step'` keeps a masked step's real loss visible, which is what you want for
monitoring a held-out span. `'scalar'` mirrors `brainstate.transform.grad`'s
`return_value` — the value *of the differentiated function* — and removes the
trailing `.mean()` from the call site.

**Returns**, mirroring `brainstate.transform.grad` exactly:

| `return_value` | `has_aux` | returns |
| --- | --- | --- |
| `False` | `False` | `grads` |
| `True` | `False` | `(grads, losses)` |
| `False` | `True` | `(grads, aux)` |
| `True` | `True` | `(grads, losses, aux)` |

`aux` is stacked with leading axis `T`.

### `etrace_evolve`

```python
learner.etrace_evolve(
    inputs,
    *,
    chunk_size=1,
    return_outputs=False,
)
```

Drives the learner over the sequence under `brainstate.transform.for_loop`, with
no gradient transform anywhere. Hidden states and eligibility traces advance
exactly as they do inside `etrace_grad`; no gradient is computed and none is
returned. `return_outputs=False` (the default) returns `None` and stacks
nothing, so a long warm-up costs no output memory.

### `ETraceVmap`

`compile(..., vmap=True)` currently returns a `brainstate.nn.Vmap`, which has no
`etrace_grad`. It will instead return `braintrace.ETraceVmap`, a subclass
carrying both methods. Because it *is* a `brainstate.nn.Vmap`, every existing use
of the `vmap=True` return value keeps working; only the added methods are new.

Reaching into `.module` is not an option: `learner.module.etrace_grad(...)` would
drive the **unbatched** learner and silently produce per-lane-wrong results.

## Semantics

### Masking and the trace

For an unmasked-step set `U = {t : mask_t = 1}`, `etrace_grad` computes

```
grads = d/dw [ (1/|U|) · sum_{t in U} loss_t ]        (reduction='mean')
```

where each `loss_t` is evaluated on the output of a learner that has been driven
through **every** step `0..t`, masked or not. Consequently:

```python
learner.etrace_evolve(xs[:a])
g1 = learner.etrace_grad(loss_fn, xs[a:], ys[a:])

# is exactly
g2 = learner.etrace_grad(loss_fn, xs, ys, mask=concat([zeros(a), ones(T - a)]))
```

including the `reduction='mean'` normalizer, which is `T - a` on both sides. The
equivalence holds exactly when `a` is a multiple of `chunk_size`; otherwise the
two runs place their window boundaries differently and the multi-step VJP spans
differ, which is a real numerical difference and not a bug.

This equivalence is the spec's sharpest test, and it is the reason the two
methods belong in the same module.

### Masked steps still pay

A masked step multiplies its loss by zero *after* `loss_fn` runs, so the VJP
backward still executes and the product is discarded. That is free for
correctness and not free for compute.

The guidance, which the docstrings must state: use `etrace_evolve` for a long
**contiguous** free-running prefix; use `mask` for **sparse or interleaved**
supervision, where there is no contiguous span to hoist out.

A `lax.cond` that skips the backward on masked steps was considered and
rejected for this version. Gating the loss alone is impossible — the learner
call sits *inside* the differentiated function — so the cond would have to wrap
the whole grad step, with both branches driving the learner to keep the state
writes identical. That doubles trace time and complexity to optimize a case the
`etrace_evolve` split already covers.

### Chunking

With `chunk_size = k` and `n = T // k`:

1. Every leaf of `inputs`, `targets` and `mask` is reshaped `(T, ...) -> (n, k, ...)`.
2. Each window calls the learner once with `MultiStepData`, yielding a stacked
   `(k, ...)` output.
3. `loss_fn` is `jax.vmap`-ed over the window's step axis, giving `(k,)` losses.
4. The window contributes `sum_j mask_j · loss_j` to the objective.
5. Windows are walked by `brainstate.transform.scan`, carrying the gradient
   accumulator. Hidden and trace states thread through automatically.
6. The `'mean'` normalizer is applied **once, after the scan**, using the global
   `mask.sum()`. Because the objective is linear in the per-step losses and
   gradients accumulate additively, dividing at the end is exact — it is *not*
   an average of per-window means.

`k == 1` keeps the current single-step call shape (`learner(x_t)`, unstacked
output, `loss_fn` called directly rather than vmapped) rather than routing
through a length-1 `MultiStepData`. This is deliberate: every existing example
and test relies on the single-step path, and unifying the two would silently
change which internal branch runs.

## Implementation

New module `braintrace/_algorithm/sequence.py`, holding:

- `SequenceDriverMixin` — both methods, written once. It depends on two hooks
  the host class supplies:
  - `_seq_call` — the callable driving one window. `self` in both cases, but for
    `ETraceVmap` that resolves to the *vmapped* call, which is the whole point;
  - `_seq_param_states` — the default differentiation set.
- `ETraceVmap(SequenceDriverMixin, brainstate.nn.Vmap)` — supplies `self` and
  `self.module.param_states`.

`ETraceAlgorithm` (in `_algorithm/base.py`) mixes it in and supplies `self` and
`self.param_states`. `sequence.py` imports nothing from `base.py`, so there is no
cycle; `_compile.py` swaps `brainstate.nn.Vmap(...)` for `ETraceVmap(...)`.

Both methods are pure `brainstate.transform` compositions — `scan` for
`etrace_grad` (it needs the gradient carry), `for_loop` for `etrace_evolve` —
so the body is traced once, per AGENTS.md rule 10. Neither method calls `jit`
itself; both are traceable inside an outer `jit`, and both compile their own
loop when called eagerly.

The `aux` output is threaded through the scan only when `has_aux=True`. The
`(T,)` loss vector is always carried — it is negligible — and simply discarded
when `return_value=False`.

### Known risk: gradient accumulator initialization

The accumulator is built as
`jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})`, the
same construction the existing examples use. For `brainunit`-valued parameters
this produces a zero *with the parameter's unit*, which is not in general the
unit of that parameter's gradient. The construction demonstrably works today in
the unit-carrying SNN examples (`examples/pp_prop/_shared.py:350`, whose weights
carry `u.mA`), so it is adopted unchanged.

If the units test below fails, the fallback is to derive the accumulator's
structure from `jax.eval_shape` of the window-gradient function rather than from
the parameter values. Implement the simple version first and let the test decide.

## Errors and edge cases

| Condition | Behaviour |
| --- | --- |
| `chunk_size < 1` | `ValueError` |
| `T % chunk_size != 0` | `ValueError` — refused, never truncated, matching `train_synthetic_gradient`'s existing contract |
| `mask` shape not `(T,)` | `ValueError` naming both shapes |
| `reduction` / `loss_output` not a legal value | `ValueError` listing the legal ones |
| `has_aux=True` and `loss_fn` returns a non-pair | error from the underlying `grad`, not silently mis-unpacked |
| all-zero `mask` | exactly-zero gradients, no NaN — the `'mean'` denominator is `max(mask.sum(), 1)` |
| learner not compiled | the existing `RuntimeError` from `_assert_compiled` |
| `inputs` leaves with mismatched leading lengths | `ValueError` naming the offending leaf |
| `T == 0` | `ValueError` — an empty sequence has no defined objective |

## Test plan

Co-located at `braintrace/_algorithm/sequence_test.py` (AGENTS.md rule 9).

**Equivalence to the status quo**

1. `chunk_size=1, mask=None, reduction='sum'` reproduces the hand-written
   scan-accumulate block element-wise.
2. `reduction='mean'` equals `'sum'` scaled by `1/T` at `mask=None`, and by
   `1/mask.sum()` otherwise.
3. `chunk_size=T, reduction='sum', mask=None` with
   `loss_fn = lambda out: (out ** 2).sum()` matches
   `oracle.online_param_gradients`, which differentiates exactly that objective
   in one whole-sequence call.

**Masking**

4. The evolve/mask equivalence stated above, element-wise.
5. A mid-sequence mask is *not* equal to concatenating gradients from two
   independent sequences — the negative control proving the trace crosses the
   masked span.
6. All-zero mask → exactly zero, finite.

**Chunking**

7. `chunk_size` divides `T` is enforced; a ragged length raises.
8. `loss_fn` is called with per-step shapes at `k=1` and at `k>1` — asserted by a
   loss that records the shape it saw.
9. Gradients at `k=1` and `k=T` differ for an approximate algorithm (they must —
   the window size is a real knob) and agree for an exact one under the regime
   its math guarantees. Per AGENTS.md, the approximate assertions go through the
   finite-window path, never a whole-sequence VJP.

**Surface**

10. All four `(return_value, has_aux)` combinations return the documented arity
    and shapes; `aux` stacks to leading `T`.
11. All three `loss_output` values return the documented thing, including that
    `'per_step'` reports a real number on a masked step where `'masked'` reports
    zero.
12. `weights=` restricted to a subset returns only those keys and leaves the rest
    untouched.
13. `ETraceVmap.etrace_grad` matches the unbatched learner's per-lane gradients.
14. `compile(..., vmap=True)` still satisfies `isinstance(..., brainstate.nn.Vmap)`.

**Robustness**

15. `brainunit`-valued weights survive the accumulator, the mask multiply and the
    mean division (the risk flagged above).
16. `inputs` as a tuple is unpacked into multiple model arguments.
17. `targets=None` calls `loss_fn` with one argument.
18. Each error-table row raises the stated type with a message naming the
    offending value.
19. `etrace_evolve(return_outputs=True)` stacks to `(T, ...)`, and
    `return_outputs=False` returns `None`.
20. Both methods work inside an outer `brainstate.transform.jit` and eagerly.

## Migration

25 call sites. Each collapses to roughly:

```python
@brainstate.transform.jit
def f_train(inputs, targets):
    grads, loss = learner.etrace_grad(
        lambda out, tar: braintools.metric.squared_error(out, tar).mean(),
        inputs, targets,
        loss_output='scalar', return_value=True,
    )
    opt.update(grads)
    return loss
```

- `examples/drtrl/_shared.accumulate_grads` is deleted — it exists only to wrap
  the block this API replaces.
- `examples/pp_prop/_shared.online_train_epoch` and
  `online_train_epoch_fixed_target` collapse to a few lines each; the latter
  exercises the `targets=None` path.
- `examples/100-gru-on-copying-task.py` loses both the manual `grads / T`
  correction (now `reduction='mean'`) and the manual warm-up `for_loop` (now
  `etrace_evolve`).
- BPTT baselines are **not** touched. They have no learner and no trace.
- `docs/quickstart/rnn_online_learning.ipynb`,
  `docs/quickstart/snn_online_learning.ipynb`, `docs/tutorials/batching.ipynb`,
  `docs/tutorials/drtrl.md` and `docs/tutorials/pp_prop.md` are updated to teach
  the new API first. `docs/apis/algorithms.rst` documents both methods and
  `ETraceVmap`.
- `examples/tests/test_smoke.py`, `test_compile_modes.py` and the per-family
  smoke suites must keep passing. The migration is behaviour-preserving except
  for the gradient *scale*, which is the one thing to watch: a site that
  previously summed keeps its tuned learning rate by passing `reduction='sum'`,
  while a site that hand-divided by `T` switches to the default `'mean'` and
  drops the division. Each migrated site declares which of the two it was, and
  the smoke assertion (loss decreases) is the check that it declared right.

`braintrace/__init__.__all__` gains `ETraceVmap`. The two methods need no export.

## Open questions

None. Argument order, `reduction`, `loss_output`, per-step `loss_fn`, and the
`ETraceVmap` subclass were each settled during design.
