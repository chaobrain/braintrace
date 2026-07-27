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
def step_loss(inp, tar):                          # <- meaningful: the user's model + objective
    out = learner(inp)
    return braintools.metric.squared_error(out, tar).mean(), out

def grad_step(prev_grads, x):                     # <- plumbing
    inp, tar = x
    f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
    cur_grads, local_loss, _ = f_grad(inp, tar)
    return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})   # <- plumbing
grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
opt.update(grads)
```

Only the second half is boilerplate. `step_loss` is the part that says something
about the model and the task, and the `learner(inp)` call inside it is the one
line a trace library should *not* hide — it is where the eligibility trace
advances. **This API removes the plumbing and keeps the step function.**

The plumbing is also easy to get subtly wrong. `scan` *sums* the per-step
gradients while the reported objective is `losses.mean()`, so
`examples/100-gru-on-copying-task.py:166` carries a hand-written
`grads = jax.tree.map(lambda g: g / losses.shape[0], grads)` correction and a
paragraph of comment explaining it. That correction is exactly the kind of thing
a library should own.

The no-loss half is duplicated too, as a bare `for_loop` over the learner —
`examples/100-gru-on-copying-task.py:178` warms the hidden states and traces for
`n_seq + 10` steps before the learning window opens.

## Scope

**In:**

1. `ETraceAlgorithm.etrace_grad` — accumulated online gradients over a sequence,
   with per-step loss masking and optional multi-step VJP windows.
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

## Measured facts this design rests on

Three probes, run against the baseline commit on a small `ValinaRNNCell` +
`Linear` model, `D_RTRL`, `T = 6`:

| probe | result |
| --- | --- |
| `MultiStepData` under `vjp_method='single-step'` | `NotImplementedError` from `vjp_graph_executor.solve_h2w_h2h_l2h_jacobian` — refused outright |
| plain single-step call vs. length-1 `MultiStepData`, both under `'multi-step'` | max abs diff `2.4e-07` — the same computation, to float32 round-off |
| `'single-step'` vs `'multi-step'`, both plain call | max abs diff `6.0` — genuinely different learning rules |

Consequences, all load-bearing below:

1. **Windowing requires `vjp_method='multi-step'`.** The API validates this up
   front rather than letting the executor's error surface three frames down.
2. **Windowing was never a free knob.** Raising it already means editing
   `compile(...)` and adopting a materially different rule. So a step function
   whose shape depends on whether windowing is on costs far less than it would
   if `chunk_size` were a free performance dial — which is why the step-function
   design below is affordable.
3. **`chunk_size=None` and `chunk_size=1` must agree to round-off** on a
   multi-step learner. That is a free consistency test, not a coincidence.

## Public API

### `etrace_grad`

```python
learner.etrace_grad(
    step_fn,
    *sequences,
    mask=None,
    chunk_size=None,
    weights=None,
    reduction='mean',
    loss_output='per_step',
    has_aux=False,
    return_value=False,
)
```

**`step_fn`** — the user's own step function. It **runs the model itself** and
returns the loss:

```python
def step_fn(inp, tar):
    out = learner(inp)
    return braintools.metric.squared_error(out, tar).mean()
```

It receives one slice of each sequence, positionally, in the order the sequences
were passed. It returns `(loss, aux)` instead when `has_aux=True`.

Handing the model call to the user is what makes the API general. A rate
regularizer reading hidden states, a multi-head model where only one head is
supervised, an auxiliary objective, a per-call `modulator=` — all are ordinary
Python inside `step_fn`, and none is expressible if the API owns the model call
and only hands back an output.

**`*sequences`** — one or more pytrees whose leaves share a leading length `T`,
sliced in lockstep. There is no distinguished `targets` argument: targets are
simply the second sequence, and a task with three aligned streams passes three.
At least one sequence is required, since it defines `T`.

**`chunk_size`** — selects the driving mode. The two modes are different
contracts, not a parameterization of one:

| `chunk_size` | slice handed to `step_fn` | `step_fn` returns | requires |
| --- | --- | --- | --- |
| `None` (default) | `seq[t]` | a **scalar** | nothing — works with the default `vjp_method='single-step'` |
| `int k >= 1` | `seq[t:t+k]`, shape `(k, ...)` | a **`(k,)`** vector of per-step losses | `vjp_method='multi-step'`, and `T % k == 0` |

In window mode the user wraps their own model inputs:
`learner(braintrace.MultiStepData(x_win))`. The API does **not** wrap, because
it cannot know which of the passed sequences are model inputs and which are
targets — that is `step_fn`'s business, and the price of it deciding.

A window-level objective is expressible as
`jnp.broadcast_to(value / k, (k,))`; the `(k,)` contract is kept strict so that
it is statically checkable and so that per-step mask granularity survives.

**`mask`** — a `(T,)` array of 0/1 (or bool). `None` means all-ones. It gates
**only the loss**. The model and the eligibility trace are driven at every step
regardless — which is the entire point: a masked step still shapes the trace
that later unmasked steps consume, so masking is *not* the same as shortening
the sequence.

**`weights`** — the `ParamState`s to differentiate. Defaults to the learner's
own `param_states`, verified key-identical *and object-identical* to
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
| `'per_step'` (default) | `(T,)` of the raw `step_fn` losses, **pre-mask** |
| `'masked'` | `(T,)` of `mask_t · loss_t` |
| `'scalar'` | the reduced objective itself |

`'per_step'` keeps a masked step's real loss visible, which is what you want for
monitoring a held-out span. `'scalar'` mirrors `brainstate.transform.grad`'s
`return_value` — the value *of the differentiated function* — and removes the
trailing `.mean()` from the call site. `loss_output` is ignored when
`return_value=False`.

**Returns**, mirroring `brainstate.transform.grad` exactly:

| `return_value` | `has_aux` | returns |
| --- | --- | --- |
| `False` | `False` | `grads` |
| `True` | `False` | `(grads, losses)` |
| `False` | `True` | `(grads, aux)` |
| `True` | `True` | `(grads, losses, aux)` |

`losses` is always `(T,)`: in window mode the `(n, k)` stack reshapes cleanly
precisely because the `(k,)` return contract is strict.

`aux` stacks over **windows**, leading axis `T // k` (which is `T` when
`chunk_size=None`). No reshape magic is applied to it — the API cannot know
whether a user's aux carries a step axis. A user wanting per-step aux in window
mode returns a `(k, ...)` aux and reshapes it themselves.

### `etrace_evolve`

```python
learner.etrace_evolve(
    *sequences,
    step_fn=None,
    chunk_size=None,
    return_outputs=False,
)
```

Drives the learner over the sequence under `brainstate.transform.for_loop`, with
no gradient transform anywhere. Hidden states and eligibility traces advance
exactly as they do inside `etrace_grad`; no gradient is computed and none is
returned.

`step_fn=None` (the default) calls `learner(*slices)` directly — and here the
API *does* wrap in `MultiStepData` under window mode, because with no `step_fn`
every sequence is by definition a model input. So the warm-up stays a one-liner:

```python
learner.etrace_evolve(xs)
```

while custom invocation stays available:

```python
learner.etrace_evolve(xs, ctx, step_fn=lambda x, c: learner(x, modulator=c))
```

`return_outputs=False` returns `None` and stacks nothing, so a long warm-up costs
no output memory. `return_outputs=True` stacks whatever `step_fn` returned, with
leading axis `T // k`.

### `ETraceVmap`

`compile(..., vmap=True)` currently returns a `brainstate.nn.Vmap`, which has no
`etrace_grad`. It will instead return `braintrace.ETraceVmap`, a subclass
carrying both methods. Because it *is* a `brainstate.nn.Vmap`, every existing use
of the `vmap=True` return value keeps working; only the added methods are new.

Reaching into `.module` is not an option: `learner.module.etrace_grad(...)` would
drive the **unbatched** learner and silently produce per-lane-wrong results.
Note that `step_fn` closes over whatever `compile` returned, which is the
vmapped object — the correct one — so this footgun does not reappear inside the
user's step function.

## Semantics

### Masking and the trace

For an unmasked-step set `U = {t : mask_t = 1}`, `etrace_grad` computes

```
grads = d/dw [ (1/|U|) · sum_{t in U} loss_t ]        (reduction='mean')
```

where each `loss_t` is evaluated on a learner that has been driven through
**every** step `0..t`, masked or not. Consequently:

```python
learner.etrace_evolve(xs[:a])
g1 = learner.etrace_grad(step_fn, xs[a:], ys[a:])

# is exactly
g2 = learner.etrace_grad(step_fn, xs, ys, mask=concat([zeros(a), ones(T - a)]))
```

including the `reduction='mean'` normalizer, which is `T - a` on both sides. In
window mode the equivalence holds when `a` is a multiple of `chunk_size`;
otherwise the two runs place their window boundaries differently and the
multi-step VJP spans differ, which is a real numerical difference and not a bug.

This equivalence is the spec's sharpest test, and it is the reason the two
methods belong in the same module.

### Masked steps still pay

A masked step multiplies its loss by zero *after* `step_fn` runs, so the VJP
backward still executes and the product is discarded. That is free for
correctness and not free for compute.

The guidance, which the docstrings must state: use `etrace_evolve` for a long
**contiguous** free-running prefix; use `mask` for **sparse or interleaved**
supervision, where there is no contiguous span to hoist out.

A `lax.cond` that skips the backward on masked steps was considered and
rejected for this version. Gating the loss alone is impossible — the learner
call sits *inside* `step_fn`, which is *inside* the differentiated function — so
the cond would have to wrap the whole grad step, with both branches driving the
learner to keep the state writes identical. That doubles trace time and
complexity to optimize a case the `etrace_evolve` split already covers.

### Window mode

With `chunk_size = k` and `n = T // k`:

1. Every leaf of every sequence and of `mask` is reshaped `(T, ...) -> (n, k, ...)`.
2. Each window calls `step_fn` once with the `(k, ...)` slices; `step_fn` calls
   the learner once, with `MultiStepData`.
3. The window contributes `sum_j mask_j · loss_j` to the objective.
4. Windows are walked by `brainstate.transform.scan`, carrying the gradient
   accumulator. Hidden and trace states thread through automatically.
5. The `'mean'` normalizer is applied **once, after the scan**, using the global
   `mask.sum()`. Because the objective is linear in the per-step losses and
   gradients accumulate additively, dividing at the end is exact — it is *not*
   an average of per-window means.

## Implementation

New module `braintrace/_algorithm/sequence.py`, holding:

- `SequenceDriverMixin` — both methods, written once. It depends on two hooks
  the host class supplies:
  - `_seq_call` — the learner callable used by the *default* `step_fn` and by
    the `vjp_method` validation. `self` in both cases, but for `ETraceVmap` that
    resolves to the *vmapped* call, which is the whole point;
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

The window-level objective is differentiated with
`brainstate.transform.grad(..., has_aux=True)` regardless of the user's
`has_aux`, since the internal aux channel is how the per-step losses leave the
grad transform. The user's aux is threaded through that same channel only when
`has_aux=True`.

The `vjp_method` check reads `getattr(learner, 'vjp_method', None)` — defensively,
since not every algorithm has one (`OSTLFeedforward` does not) — and refuses
window mode only when it is explicitly `'single-step'`.

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
| no sequences passed | `ValueError` — `T` is undefined |
| sequence leaves with mismatched leading lengths | `ValueError` naming the offending leaf |
| `T == 0` | `ValueError` — an empty sequence has no defined objective |
| `chunk_size` an int `< 1` | `ValueError` |
| `chunk_size` an int and `T % chunk_size != 0` | `ValueError` — refused, never truncated, matching `train_synthetic_gradient`'s existing contract |
| `chunk_size` an int and `learner.vjp_method == 'single-step'` | `ValueError` naming the fix (`compile(..., vjp_method='multi-step')`), raised **before** any tracing |
| `chunk_size=None` and `step_fn` returns a non-scalar | `ValueError` stating the scalar contract |
| `chunk_size=k` and `step_fn` returns anything but shape `(k,)` | `ValueError` naming both the expected and the received shape |
| `mask` shape not `(T,)` | `ValueError` naming both shapes |
| `reduction` / `loss_output` not a legal value | `ValueError` listing the legal ones |
| `has_aux=True` and `step_fn` returns a non-pair | error from the underlying `grad`, not silently mis-unpacked |
| all-zero `mask` | exactly-zero gradients, no NaN — the `'mean'` denominator is `max(mask.sum(), 1)` |
| learner not compiled | the existing `RuntimeError` from `_assert_compiled` |

## Test plan

Co-located at `braintrace/_algorithm/sequence_test.py` (AGENTS.md rule 9).

**Equivalence to the status quo**

1. `chunk_size=None, mask=None, reduction='sum'` reproduces the hand-written
   scan-accumulate block element-wise.
2. `reduction='mean'` equals `'sum'` scaled by `1/T` at `mask=None`, and by
   `1/mask.sum()` otherwise.
3. `chunk_size=T, reduction='sum', mask=None` with a sum-of-squares `step_fn`
   matches `oracle.online_param_gradients`, which differentiates exactly that
   objective in one whole-sequence call.

**The two driving modes**

4. On a `vjp_method='multi-step'` learner, `chunk_size=None` and `chunk_size=1`
   agree to float32 round-off (measured `2.4e-07` at baseline). This pins the
   claim that the mode split is a contract difference, not a numerical one.
5. On a `vjp_method='single-step'` learner, `chunk_size=1` raises `ValueError`
   before tracing — not the executor's `NotImplementedError`.
6. `chunk_size` divides `T` is enforced; a ragged length raises.
7. `step_fn` sees `seq[t]` under `None` and `(k, ...)` under `k` — asserted by a
   `step_fn` that records the shape it was handed.
8. A `step_fn` returning the wrong rank for its mode raises the documented
   `ValueError`.
9. Gradients at `chunk_size=None` and `chunk_size=k>1` differ for an approximate
   algorithm (they must — the window is a real knob) and agree for an exact one
   under the regime its math guarantees. Per AGENTS.md, the approximate
   assertions go through the finite-window path, never a whole-sequence VJP.

**Masking**

10. The evolve/mask equivalence stated above, element-wise.
11. A mid-sequence mask is *not* equal to concatenating gradients from two
    independent sequences — the negative control proving the trace crosses the
    masked span.
12. All-zero mask → exactly zero, finite.

**Surface**

13. All four `(return_value, has_aux)` combinations return the documented arity
    and shapes; `losses` is `(T,)` in both modes; `aux` stacks to leading `T // k`.
14. All three `loss_output` values return the documented thing, including that
    `'per_step'` reports a real number on a masked step where `'masked'` reports
    zero.
15. `weights=` restricted to a subset returns only those keys and leaves the rest
    untouched.
16. Three sequences are sliced in lockstep and passed positionally in order.
17. A `step_fn` that reads hidden states (a firing-rate regularizer) and one that
    supervises only one head of a two-head model both work — the generality this
    design exists for.
18. `ETraceVmap.etrace_grad` matches the unbatched learner's per-lane gradients.
19. `compile(..., vmap=True)` still satisfies `isinstance(..., brainstate.nn.Vmap)`.

**Robustness**

20. `brainunit`-valued weights survive the accumulator, the mask multiply and the
    mean division (the risk flagged above).
21. Each error-table row raises the stated type with a message naming the
    offending value.
22. `etrace_evolve` with `step_fn=None` drives the learner and, under window
    mode, wraps in `MultiStepData` itself; with a custom `step_fn` it does not.
23. `etrace_evolve(return_outputs=True)` stacks to leading `T // k`, and
    `return_outputs=False` returns `None`.
24. Both methods work inside an outer `brainstate.transform.jit` and eagerly.

## Migration

25 call sites. Each keeps its step function verbatim and drops the plumbing:

```python
def step_loss(inp, tar):                       # unchanged from today
    out = learner(inp)
    return braintools.metric.squared_error(out, tar).mean()

@brainstate.transform.jit
def f_train(inputs, targets):
    grads, loss = learner.etrace_grad(
        step_loss, inputs, targets,
        loss_output='scalar', return_value=True,
    )
    opt.update(grads)
    return loss
```

- `examples/drtrl/_shared.accumulate_grads` is deleted — it exists only to wrap
  the block this API replaces.
- `examples/pp_prop/_shared.online_train_epoch` and
  `online_train_epoch_fixed_target` collapse to a few lines each; the latter
  exercises a `step_fn` closing over a fixed label with only one sequence passed.
- `examples/101-integrator-rnn.py` keeps its L2 term inside `step_loss`
  unchanged — under the previous out-based design it would have needed
  restructuring.
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

## Rejected alternatives

**`loss_fn(out_t, target_t)`, with the API calling the model.** The first draft
of this spec. It buys chunk-size transparency — the API can `jax.vmap` a
per-step loss over the window axis, so the user's function never changes shape.
It was rejected on two grounds. First, generality: it cannot express a
regularizer reading hidden states, a multi-head model, or a per-call
`modulator=`, and it hides the `learner(...)` call that is the whole subject of
the library. Second, the transparency it buys is worth less than it appears —
window mode already requires `vjp_method='multi-step'`, a materially different
learning rule (measured `6.0` max abs gradient difference), so `chunk_size` was
never a knob a user could turn without editing `compile(...)` anyway.

**Always driving through `MultiStepData`, including length-1 windows**, to make
the two modes one. Impossible: the default `vjp_method='single-step'` refuses
`MultiStepData` outright.

**`chunk_size=1` meaning the plain single-step path**, with window mode starting
at 2. Rejected in favour of `None`, which makes the mode a *type* distinction
rather than a special case carved out of the integer range — and which leaves
`chunk_size=1` free to mean the genuine degenerate window that test 4 exercises.

## Open questions

None. The step-function contract, argument order, `chunk_size` mode split,
`reduction`, `loss_output`, and the `ETraceVmap` subclass were each settled
during design.
