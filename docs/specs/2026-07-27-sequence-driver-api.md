# Sequence driver API — `etrace_grad` and `etrace_evolve`

Status: spec, revised after external review
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
`examples/100-gru-on-copying-task.py:178` and
`examples/drtrl/05-vjp-multi-step.py:59` both warm the hidden states and traces
before the learning window opens.

## Scope

**In:**

1. `ETraceAlgorithm.etrace_grad` — accumulated online gradients over a sequence,
   with per-step loss weighting and optional multi-step VJP windows.
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
- **Window mode under `ETraceVmap`** — refused, see "The vmap × window axis
  collision" below.
- **Lane-aware (per-sample) masks** — refused, see "Declared limitations".
- Any change to `update()`, to the trace algebra, or to any algorithm's numerics.
  This is a driver layer; it composes existing public behaviour and must not
  alter it.

## Measured facts this design rests on

Six probes, run against the baseline commit. The first three use a small
`ValinaRNNCell` + `Linear` model with `D_RTRL`, `T = 6`:

| # | probe | result |
| --- | --- | --- |
| P1 | `MultiStepData` **under a gradient**, `vjp_method='single-step'` | `NotImplementedError` from `vjp_graph_executor.solve_h2w_h2h_l2h_jacobian` |
| P2 | plain single-step call vs. length-1 `MultiStepData`, both `'multi-step'`, under a gradient | max abs diff `2.4e-07` — the same computation, to float32 round-off |
| P3 | `'single-step'` vs `'multi-step'`, both plain call | max abs diff `6.0` — genuinely different learning rules |
| P4 | `MultiStepData` with **no gradient transform** (evolve-style) | **succeeds under both `vjp_method`s**, output `(T, B, ...)` |
| P5 | grep for `MultiStepData` across `examples/` | **zero hits** — windowed driving is unexercised in the repo; `05-vjp-multi-step.py` uses `vjp_method='multi-step'` with single-step driving |
| P6 | gradient unit vs parameter unit for a `u.mA`-valued `braintrace.nn.Linear` weight | gradient is **also `mA`**; `zeros_like(param) + grad` succeeds |

Consequences, all load-bearing below:

1. **Windowed *gradients* require `vjp_method='multi-step'`** (P1). The API
   validates this up front rather than letting the executor's error surface
   three frames down.
2. **Windowed *evolution* requires nothing** (P4). `etrace_evolve` therefore
   gets its own compatibility matrix, not a copy of `etrace_grad`'s.
3. **Windowing was never a free knob** (P3). Raising it already means editing
   `compile(...)` and adopting a materially different rule — which is why the
   step-function design's shape change is affordable.
4. **`chunk_size=None` and `chunk_size=1` must agree to round-off** on a
   multi-step learner (P2). A free consistency test, not a coincidence.
5. **The accumulator may be built from parameter values** (P6). See "Gradient
   accumulator initialization".
6. **Refusing window mode under `ETraceVmap` breaks no existing code** (P5).

## Public API

### `etrace_grad`

```python
def etrace_grad(
    self,
    *sequences,
    step_fn,
    mask=None,
    chunk_size=None,
    weights=None,
    reduction='mean',
    loss_output='per_step',
    has_aux=False,
    return_value=False,
):
```

Both methods share one signature shape — `(*sequences, step_fn=..., <mode args>)`
— so the two read the same at every call site. `step_fn` is a **required
keyword-only** parameter here (there is no default loss) and optional in
`etrace_evolve`. Omitting it raises Python's own
`TypeError: etrace_grad() missing 1 required keyword-only argument: 'step_fn'`.

Because it is keyword-only, every call site passes it by name:

```python
grads = learner.etrace_grad(inputs, targets, step_fn=step_loss)
```

**`step_fn`** — the user's own step function. It **runs the model itself** and
returns the loss:

```python
def step_loss(inp, tar):
    out = learner(inp)
    return braintools.metric.squared_error(out, tar).mean()
```

It receives one slice of each sequence, positionally, in the order the sequences
were passed. It returns `(loss, aux)` instead when `has_aux=True`.

Handing the model call to the user is what makes the API general. A multi-head
model where only one head is supervised, an auxiliary objective, an L2 term over
weights, a per-call `modulator=` — all are ordinary Python inside `step_fn`, and
none is expressible if the API owns the model call and only hands back an
output.

#### `step_fn` preconditions

`step_fn` participates in state mutation, so it carries obligations the driver
cannot check. Violating any of these silently corrupts trace advancement rather
than raising:

1. **Exactly one learner call per invocation.** Zero calls leave the trace
   un-advanced for that step while still consuming a slice; two calls advance it
   twice. Conditional calls (`if pred: learner(x)`) make the trace depend on data
   in a way the mask cannot express — use `mask` instead.
2. **It must call *this* learner** — the object whose `etrace_grad` was invoked.
   Calling a different learner, or `self.module` of an `ETraceVmap`, drives the
   wrong states.
3. **Ordering.** Any other state mutation the user performs must not be
   interleaved in a way that changes what the learner reads; do model-independent
   work before or after the learner call, not between its state reads and writes.
4. **Purity otherwise.** `step_fn` runs inside `scan`, so Python-level side
   effects (appending to a list, printing) execute once at trace time, not per
   step.

The driver documents these; it does not enforce them. A `step_fn` that violates
(1) produces a plausible-looking gradient that is wrong, which is the worst
failure mode in this API and the reason the preconditions are stated this
prominently.

**`*sequences`** — one or more pytrees whose leaves share a leading length `T`,
sliced in lockstep. There is no distinguished `targets` argument: targets are
simply the second sequence, and a task with three aligned streams passes three.
At least one sequence is required, since it defines `T`.

A sequence may **not** be a `SingleStepData` or `MultiStepData` wrapper. Those
describe how one *slice* is fed to the model, which is `step_fn`'s decision, and
the driver slices along axis 0 — slicing a wrapper would decompose the wrapper
rather than the data. Passing one raises `TypeError` naming the fix (wrap inside
`step_fn`). Model arguments that are constant across time are closed over by
`step_fn` rather than passed as sequences.

**`chunk_size`** — selects the driving mode. The two modes are different
contracts, not a parameterization of one:

| `chunk_size` | slice handed to `step_fn` | `step_fn` returns | requires |
| --- | --- | --- | --- |
| `None` (default) | `seq[t]` | a **scalar** | nothing — legal on either `vjp_method` |
| `int k >= 1` | `seq[t:t+k]`, shape `(k, ...)` | a **`(k,)`** vector of per-step losses | `vjp_method='multi-step'`; `T % k == 0`; not an `ETraceVmap` |

Window mode changes five things at once, which is a real cost of the knob and is
listed here rather than discovered:

| what changes | `chunk_size=None` | `chunk_size=k` |
| --- | --- | --- |
| slice shape into `step_fn` | `seq[t]` | `(k, ...)` |
| `step_fn` return | scalar | `(k,)` |
| model call | `learner(x)` | `learner(MultiStepData(x))` — the user wraps |
| admissible learners | any | `vjp_method='multi-step'`, non-vmapped |
| learning rule | per P3, a different rule from the other column |

In window mode the user wraps their own model inputs. The API does **not** wrap,
because it cannot know which of the passed sequences are model inputs and which
are targets — that is `step_fn`'s business, and the price of it deciding.

**`mask`** — a `(T,)` array of **per-step loss weights**. `None` means all-ones.
Values need not be binary: `mask` is a weight vector, `0` and `1` are simply its
most common values, and `reduction='mean'` normalizes by `mask.sum()` so a
weighted mask stays a weighted mean. Only the shape is validated; negative
weights are the user's business.

It gates **only the loss**. The model and the eligibility trace are driven at
every step regardless — which is the entire point: a zero-weighted step still
shapes the trace that later weighted steps consume, so masking is *not* the same
as shortening the sequence.

**`weights`** — the `ParamState`s to differentiate. Defaults to the learner's
own `param_states`, verified key-identical *and object-identical* to
`model.states(brainstate.ParamState)`, so an optimizer registered the usual way
accepts the returned tree unchanged. Pass explicitly to freeze a subset.

**`reduction`** — how the per-step losses combine into the reduced objective:

| value | reduced objective |
| --- | --- |
| `'mean'` (default) | `sum_t mask_t · loss_t / max(sum_t mask_t, 1)` |
| `'sum'` | `sum_t mask_t · loss_t` |

`'mean'` divides by the **total mask weight**, not by `T`. It is the default
because it is the scale people actually report and compare against BPTT, and it
retires the hand-written `grads / losses.shape[0]` correction.

**`loss_output`** — what `return_value=True` hands back:

| value | returns | shape |
| --- | --- | --- |
| `'per_step'` (default) | the raw `step_fn` losses, **pre-mask** | `(T,)` |
| `'masked'` | `mask_t · loss_t` | `(T,)` |
| `'scalar'` | the reduced objective itself | scalar |

`'per_step'` keeps a zero-weighted step's real loss visible, which is what you
want for monitoring a held-out span. `'scalar'` mirrors
`brainstate.transform.grad`'s `return_value` — the value *of the reduced
objective* — and removes the trailing `.mean()` from the call site.
`loss_output` is ignored when `return_value=False`.

**Returns**, mirroring `brainstate.transform.grad` exactly:

| `return_value` | `has_aux` | returns |
| --- | --- | --- |
| `False` | `False` | `grads` |
| `True` | `False` | `(grads, losses)` |
| `False` | `True` | `(grads, aux)` |
| `True` | `True` | `(grads, losses, aux)` |

`losses` is `(T,)` under `loss_output='per_step'` or `'masked'`, and a scalar
under `'scalar'`. In window mode the `(n, k)` stack reshapes to `(T,)` cleanly
precisely because the `(k,)` return contract is strict.

`aux` stacks over **windows**, leading axis `T // k` (which is `T` when
`chunk_size=None`). No reshape magic is applied to it — the driver cannot know
whether a user's aux carries a step axis. A user wanting per-step aux in window
mode returns a `(k, ...)` aux and reshapes it themselves.

#### What `grads` is, precisely

`grads` is **the learner's online-gradient estimate of the reduced objective** —
not, in general, its mathematical derivative.

For an exact algorithm driven inside its valid regime the two coincide. For every
approximate rule they deliberately do not: a factorized trace, a scalar-leak
temporal recursion, a truncated recurrence scope or a random-feedback learning
signal each returns an estimate that differs from the true derivative by
construction, and that difference is the algorithm's whole content. The
`reduction` and `mask` parameters define the *objective the estimate is aimed
at*; they do not promise that the returned tree is `∂/∂w` of it.

Wording matters here because this repository already measures the gap — see
`docs/specs/2026-07-25-known-limitations.md` F-SINGLESTEP and F-23, and
`tests/axis_discrimination_test.py`. The driver must not launder an approximate
estimate into an exactness claim, and its docstrings must use the same language.

### `etrace_evolve`

```python
def etrace_evolve(
    self,
    *sequences,
    step_fn=None,
    chunk_size=None,
    return_outputs=False,
):
```

Drives the learner over the sequence under `brainstate.transform.for_loop`, with
no gradient transform anywhere. Hidden states and eligibility traces advance
exactly as they do inside `etrace_grad`; no gradient is computed and none is
returned.

`step_fn=None` (the default) calls `learner(*slices)` directly — and here the
driver *does* wrap in `MultiStepData` under window mode, because with no
`step_fn` every sequence is by definition a model input. So the warm-up stays a
one-liner:

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

**`etrace_evolve` has its own compatibility matrix.** It performs no loss VJP, so
per P4 it accepts windowed driving on a single-step learner, which `etrace_grad`
refuses:

| | `vjp_method='single-step'` | `vjp_method='multi-step'` |
| --- | --- | --- |
| `chunk_size=None` | legal | legal |
| `chunk_size=k` | **legal** — P4; contrast `etrace_grad`, which refuses | legal |

Under `ETraceVmap`, window mode is refused for both methods (next section).

The `step_fn` preconditions apply unchanged, except that a custom `step_fn` in
window mode must do its own `MultiStepData` wrapping — supplying `step_fn` opts
out of the driver's wrapping entirely.

### `ETraceVmap`

`compile(..., vmap=True)` currently returns a `brainstate.nn.Vmap`, which has no
`etrace_grad`. It will instead return `braintrace.ETraceVmap`, a subclass
carrying both methods. Because it *is* a `brainstate.nn.Vmap`, every existing use
of the `vmap=True` return value keeps working; only the added methods are new.

Reaching into `.module` is not an option: `learner.module.etrace_grad(...)` would
drive the **unbatched** learner and silently produce per-lane-wrong results.
`step_fn` closes over whatever `compile` returned, which is the vmapped object —
the correct one — so this footgun does not reappear inside the user's step
function.

#### The vmap × window axis collision

**Window mode is refused under `ETraceVmap`, for both methods, with a
`ValueError`.**

`compile(vmap=True)` builds `brainstate.nn.Vmap(learner, ...)` with the default
`in_axes=0` (`_compile.py:315`). In window mode a slice has shape `(k, B, ...)`,
so axis 0 is **time**, and the wrapper would map the time axis as the batch axis:
each of `k` lanes would receive a `(B, ...)` array where the per-lane compiled
graph expects `(...)`. That is usually a loud shape error, but when `k == B` the
shapes line up and it silently trains on transposed data.

A transpose adapter — swap to `(B, k, ...)` before the call, swap the output back
— would fix the layout, but the driver cannot apply it: `step_fn` owns the model
call, so the driver does not know which sequences to transpose. Requiring the
*user* to transpose would put a subtle, silently-wrong-if-omitted step into the
contract.

Refusing costs nothing today: per P5 no example in the repository uses
`MultiStepData` at all, let alone with `vmap=True`. Users needing windowed
gradients on a batch use the batched (non-vmap) mode, which carries the batch
axis inside the compiled graph rather than outside it, and the error message says
so.

Lifting this restriction — via an explicit transpose adapter and an `in_axes`
that names the batch axis — is future work, and would need a test at `B != k`
specifically, since `B == k` is the case that hides the bug.

## Semantics

### State lifecycle

Both methods are **continuations, not sessions**:

- They begin from the learner's *current* hidden, trace and `running_index`
  state. Neither resets anything; `brainstate.nn.init_all_states` remains the
  caller's tool for that.
- They leave the final state installed, so consecutive calls compose —
  `evolve(a); grad(b)` drives `a` then `b` over one continuous trajectory. This
  is what makes the warm-up idiom work.
- `running_index` advances once per `update()` call, i.e. `T` times under
  `chunk_size=None` and `T // k` times under window mode. This asymmetry is the
  subject of the F-30 interaction below.
- **On failure, state is not restored.** A `step_fn` that raises mid-`scan`
  leaves the learner partially advanced, because `brainstate.transform.scan`
  writes state as it goes. Callers who need transactional behaviour snapshot
  state themselves. The driver does not add a rollback, which would mean copying
  every trace state on every call.

### Masking and the trace

For per-step weights `mask_t`, `etrace_grad` returns the learner's online
estimate of the gradient of

```
(1 / sum_t mask_t) · sum_t mask_t · loss_t          (reduction='mean')
```

where each `loss_t` is evaluated on a learner that has been driven through
**every** step `0..t`, zero-weighted or not. Consequently, for two learners
`A` and `B` built identically from the same seed:

```python
A.etrace_evolve(xs[:a])
g1 = A.etrace_grad(xs[a:], ys[a:], step_fn=step_fn)

g2 = B.etrace_grad(xs, ys, step_fn=step_fn,
                   mask=concat([zeros(a), ones(T - a)]))

# g1 == g2
```

The two learners are essential: a single learner would be advanced by the first
pair of calls before the second ran, so the comparison would not be of the same
trajectory. The `reduction='mean'` normalizer is `T - a` on both sides.

In window mode the equivalence holds when `a` is a multiple of `chunk_size`;
otherwise the two runs place their window boundaries differently and the
multi-step VJP spans differ, which is a real numerical difference and not a bug.

This equivalence is the spec's sharpest test, and it is the reason the two
methods belong in the same module.

### Zero-weighted steps still pay

A zero-weighted step multiplies its loss by zero *after* `step_fn` runs, so the
VJP backward still executes and the product is discarded. That is free for
correctness and not free for compute.

The guidance, which the docstrings must state: use `etrace_evolve` for a long
**contiguous** free-running prefix; use `mask` for **sparse or interleaved**
supervision, where there is no contiguous span to hoist out.

A `lax.cond` that skips the backward on zero-weighted steps was considered and
rejected for this version. Gating the loss alone is impossible — the learner
call sits *inside* `step_fn`, which is *inside* the differentiated function — so
the cond would have to wrap the whole grad step, with both branches driving the
learner to keep the state writes identical. That doubles trace time and
complexity to optimize a case the `etrace_evolve` split already covers.

### Window mode mechanics

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

#### Expressing a window-level objective

A `step_fn` with a genuinely window-level objective `value` must still return
`(k,)`, and the correct spreading depends on `reduction`:

| `reduction` | return | gives the window a total contribution of |
| --- | --- | --- |
| `'sum'` | `jnp.broadcast_to(value / k, (k,))` | `value` |
| `'mean'` | `jnp.broadcast_to(value / k, (k,))` | `value / sum(mask)` — i.e. `value` averaged over **steps**, not windows |
| `'mean'`, wanting the mean over **windows** | `jnp.broadcast_to(value * k / k, (k,)) = jnp.full((k,), value)` | `value · k / sum(mask)`, which for an all-ones mask is `value / n` |

The asymmetry is intrinsic: `'mean'` normalizes per step, and a window-level
objective has no per-step decomposition to normalize. Users mixing the two should
prefer `reduction='sum'` and divide themselves.

### F-30: window mode and the IO-factorized bias correction

`docs/specs/2026-07-25-known-limitations.md` F-30 records that the IO-dim f-side
de-biasing correction is **indexed by `update()` call count, not by trace-step
count**, so it is exact only for single-step input — and that this is *preserved
deliberately*, because fixing it moves `pp_prop` / `OSTLFeedforward` gradients.

Window mode makes that latent condition reachable from a single keyword. A
`k`-step window advances the trace `k` times but increments `running_index` once,
so the correction lags the trace by a factor of `k`.

Resolution: **document, do not refuse.** F-30 is a deliberate property of those
engines, not a driver defect, and refusing would make the driver stricter than
the algorithms it drives. Concretely:

- `etrace_grad`'s docstring names F-30 under window mode for IO-factorized
  engines (`pp_prop`, `ES_D_RTRL`, `OSTLFeedforward`, and any
  `trace_factorization='io_factorized'` config).
- A driver-level regression test pins the consequence, so a future F-30 fix is
  detected here rather than only in `io_dim_vjp_test.py`.
- F-30's row in known-limitations gains a sentence noting that
  `etrace_grad(chunk_size=k)` is now a first-class way to reach it.

### DNI: matching the synthesizer's deployment contract

F-35 records that a `train_synthetic_gradient` fit is valid only for the exact
`(loss_fn, chunk_size)` pair it was trained on, that **nothing checks that
deployment matches**, and that a mismatch is not degraded DNI but noise shaped
like a cotangent — measurably worse than leaving DNI off.

This driver adds new ways to mismatch, because it introduces `reduction`, `mask`
and `loss_output` between the user's `step_fn` and the objective the synthesizer
was fit against. The correspondence, which the docstrings must state:

| `train_synthetic_gradient` | `etrace_grad` |
| --- | --- |
| `chunk_size` | `chunk_size` — must be **equal**, including `None` vs `1` |
| `loss_fn(output)` | the objective `step_fn` actually descends, *after* `reduction` and `mask` |

The trap specific to this driver: fitting against a plain summed loss and then
deploying with `reduction='mean'` rescales the cotangent by `1/sum(mask)`, and a
`mask` that was not present during fitting changes which steps contribute at all.
Both are F-35 mismatches wearing new clothes.

The driver does not enforce the match — per F-35 the learner never sees the
caller's objective — but `dni_test.py` gains a driver-level case in the shape of
`TestALearnedSynthesiserHelps::test_training_on_the_wrong_window_size_is_worse_than_not_training`,
covering a `reduction` mismatch introduced through `etrace_grad`.

## Declared limitations

Stated here rather than discovered later:

1. **Masks are time-only.** `mask` is `(T,)` and cannot express different valid
   lengths per batch lane, which is the standard padded-batch requirement.
   Users with ragged batches fold the per-lane validity into `step_fn` — the
   loss it returns is already a scalar reduced over lanes, so a per-lane weight
   applied there is well-defined — and accept that `reduction='mean'`'s
   normalizer counts steps, not valid lane-steps. Lane-aware masks are future
   work; they would need the mask to enter *inside* the vmapped region.
2. **Per-step hidden-state history is unavailable in window mode.** After one
   `MultiStepData` call, hidden states hold the **final** state of the window,
   not a `(k, ...)` history. A per-step firing-rate regularizer that reads hidden
   states therefore cannot generally produce the required `(k,)` losses. The
   generality claim for `step_fn` is scoped to `chunk_size=None`; in window mode
   an objective must be computable from the stacked `(k, ...)` **outputs**.
3. **Window mode is refused under `ETraceVmap`** — see above.
4. **No transactional state.** A raising `step_fn` leaves the learner partially
   advanced.

## Implementation

New module `braintrace/_algorithm/sequence.py`, holding:

- `SequenceDriverMixin` — both methods, written once. It depends on three hooks
  the host class supplies:
  - `_seq_call` — the callable driving one step or window, used by the default
    `step_fn`;
  - `_seq_param_states` — the default differentiation set;
  - `_seq_vjp_method` — the learner's `vjp_method`, or `None` if it has none.
- `ETraceVmap(SequenceDriverMixin, brainstate.nn.Vmap)` — supplies `self`,
  `self.module.param_states`, and `getattr(self.module, 'vjp_method', None)`.

`ETraceAlgorithm` (in `_algorithm/base.py`) mixes it in and supplies `self`,
`self.param_states`, and `getattr(self, 'vjp_method', None)`.
`sequence.py` imports nothing from `base.py`, so there is no cycle;
`_compile.py` swaps `brainstate.nn.Vmap(...)` for `ETraceVmap(...)`.

**`_seq_vjp_method` is a hook, not a `getattr` on the callable.**
`brainstate.nn.Vmap` defines no `__getattr__` (verified), so it does not forward
`vjp_method` from `.module` — reading the attribute off the driver object would
silently yield `None` for every vmapped learner and bypass the validation
entirely.

**Window mode requires `_seq_vjp_method == 'multi-step'` exactly.** A learner
reporting `None` is refused, not admitted. The earlier draft admitted `None` on
the stated grounds that `OSTLFeedforward` has no `vjp_method`; that claim was
false — its MRO is `OSTLFeedforward → pp_prop → IODimVjpAlgorithm →
ETraceVjpAlgorithm`, which sets `self.vjp_method`, and `ostl.py:96,169` documents
it. No shipped algorithm lacks the attribute, so `None` means "an unknown
subclass whose windowing support is unverified", and refusing is the safe
default.

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

### Gradient accumulator initialization

The accumulator is built as
`jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})`.

This is specified, not provisional. The concern it raises — that a gradient's
units may differ from its parameter's, making a parameter-shaped zero
dimensionally wrong — does not arise in this codebase. Measured (P6): for a
`braintrace.nn.Linear` whose weight and bias carry `u.mA`, the returned gradient
carries `mA` as well, and `zeros_like(param) + grad` succeeds. `brainunit`'s
gradient convention preserves the operand's unit rather than producing its
reciprocal.

Test 20 pins this. Should a future `brainunit` change break it, the replacement
is an accumulator whose structure comes from `jax.eval_shape` of the
window-gradient function rather than from parameter values — not "use the first
computed gradient", which cannot work because `scan` fixes the carry structure
before the loop runs.

## Errors and edge cases

| Condition | Behaviour |
| --- | --- |
| no sequences passed | `ValueError` — `T` is undefined |
| a sequence is a `SingleStepData` / `MultiStepData` | `TypeError` naming the fix: wrap inside `step_fn` |
| sequence leaves with mismatched leading lengths | `ValueError` naming the offending leaf |
| `T == 0` | `ValueError` — an empty sequence has no defined objective |
| `chunk_size` an int `< 1` | `ValueError` |
| `chunk_size` an int and `T % chunk_size != 0` | `ValueError` — refused, never truncated, matching `train_synthetic_gradient`'s existing contract |
| `chunk_size` an int and `_seq_vjp_method != 'multi-step'` (**`etrace_grad` only**) | `ValueError` naming the fix (`compile(..., vjp_method='multi-step')`), raised **before** any tracing |
| `chunk_size` an int on an `ETraceVmap` (**both methods**) | `ValueError` naming the batched non-vmap mode as the alternative |
| `chunk_size=None` and `step_fn` returns a non-scalar | `ValueError` stating the scalar contract |
| `chunk_size=k` and `step_fn` returns anything but shape `(k,)` | `ValueError` naming both the expected and the received shape |
| `mask` shape not `(T,)` | `ValueError` naming both shapes |
| `mask` non-binary | **accepted** — `mask` is a weight vector |
| `reduction` / `loss_output` not a legal value | `ValueError` listing the legal ones |
| `has_aux=True` and `step_fn` returns a non-pair | error from the underlying `grad`, not silently mis-unpacked |
| all-zero `mask` | zero gradients **for finite, differentiable step losses**; the `max(mask.sum(), 1)` denominator prevents division by zero but cannot prevent `0 · NaN = NaN`, nor a NaN arriving through a zero cotangent on an undefined derivative |
| `step_fn` raises mid-loop | propagates; learner left partially advanced (see State lifecycle) |
| learner not compiled | the existing `RuntimeError` from `_assert_compiled` |

## Test plan

Co-located at `braintrace/_algorithm/sequence_test.py` (AGENTS.md rule 9).

**Equivalence to the status quo**

1. `chunk_size=None, mask=None, reduction='sum'` reproduces the hand-written
   scan-accumulate block element-wise.
2. `reduction='mean'` equals `'sum'` scaled by `1/T` at `mask=None`, and by
   `1/mask.sum()` otherwise, including a non-binary mask.
3. `chunk_size=T, reduction='sum', mask=None` with a sum-of-squares `step_fn`
   matches `oracle.online_param_gradients`, which differentiates exactly that
   objective in one whole-sequence call.

**The two driving modes**

4. On a `vjp_method='multi-step'` learner, `chunk_size=None` and `chunk_size=1`
   agree to float32 round-off (measured `2.4e-07`, P2). This pins the claim that
   the mode split is a contract difference, not a numerical one.
5. On a `vjp_method='single-step'` learner, `etrace_grad(chunk_size=1)` raises
   `ValueError` before tracing — not the executor's `NotImplementedError`.
6. On the same learner, `etrace_evolve(chunk_size=1)` **succeeds** (P4). The two
   matrices differ, and this is the test that says so.
7. A learner whose `_seq_vjp_method` is `None` is refused in window mode.
8. `chunk_size` divides `T` is enforced; a ragged length raises.
9. `step_fn` sees `seq[t]` under `None` and `(k, ...)` under `k` — asserted by a
   `step_fn` that records the shape it was handed.
10. A `step_fn` returning the wrong rank for its mode raises the documented
    `ValueError`.
11. Gradients at `chunk_size=None` and `chunk_size=k>1` differ for an approximate
    algorithm (they must — the window is a real knob) and agree for an exact one
    under the regime its math guarantees. Per AGENTS.md, the approximate
    assertions go through the finite-window path, never a whole-sequence VJP.

**Masking**

12. The evolve/mask equivalence, **using two identically seeded learners**,
    element-wise.
13. A mid-sequence mask is *not* equal to concatenating gradients from two
    independent sequences — the negative control proving the trace crosses the
    zero-weighted span.
14. All-zero mask on a finite, differentiable loss → exactly zero, finite.
15. A non-binary mask reweights as documented.

**vmap**

16. `ETraceVmap.etrace_grad` matches the unbatched learner's per-lane gradients
    at `chunk_size=None`.
17. `ETraceVmap.etrace_grad(chunk_size=k)` and `etrace_evolve(chunk_size=k)` both
    raise `ValueError`, **tested at `B != k` and at `B == k`** — the latter being
    the case that would otherwise pass silently.
18. `compile(..., vmap=True)` still satisfies `isinstance(..., brainstate.nn.Vmap)`.

**Surface**

19. All four `(return_value, has_aux)` combinations return the documented arity;
    `losses` is `(T,)` under `'per_step'`/`'masked'` and **scalar** under
    `'scalar'`; `aux` stacks to leading `T // k`.
20. All three `loss_output` values return the documented thing, including that
    `'per_step'` reports a real number on a zero-weighted step where `'masked'`
    reports zero.
21. `weights=` restricted to a subset returns only those keys and leaves the rest
    untouched.
22. Three sequences are sliced in lockstep and passed positionally in order.
23. A `SingleStepData` / `MultiStepData` passed as a sequence raises `TypeError`.
24. A `step_fn` supervising one head of a two-head model works at
    `chunk_size=None`; a hidden-state-reading regularizer works at
    `chunk_size=None` and is documented as unsupported in window mode
    (limitation 2).

**State lifecycle**

25. `evolve(a); grad(b)` composes into one trajectory: equal to `grad` over the
    concatenation with a zero-weight prefix (this is test 12 read from the other
    side, kept separate because it pins *composition*, not masking).
26. Repeated `etrace_grad` calls continue rather than reset; `running_index`
    advances `T` times under `None` and `T // k` times under window mode.

**Algorithm interactions**

27. F-30: on an IO-factorized engine, `chunk_size=k` produces the lagged bias
    correction the limitation describes. A regression test, so a future F-30 fix
    surfaces here.
28. F-35: a `reduction` mismatch between `train_synthetic_gradient` and
    `etrace_grad` degrades DNI, in the shape of the existing window-size test.

**Robustness**

29. `brainunit`-valued weights survive the accumulator, the mask multiply and the
    mean division — pinning P6.
30. Each error-table row raises the stated type with a message naming the
    offending value.
31. `etrace_evolve` with `step_fn=None` drives the learner and, under window
    mode, wraps in `MultiStepData` itself; with a custom `step_fn` it does not.
32. `etrace_evolve(return_outputs=True)` stacks to leading `T // k`, and
    `return_outputs=False` returns `None`.
33. Both methods work inside an outer `brainstate.transform.jit` and eagerly.

## Migration

25 call sites. Each keeps its step function verbatim and drops the plumbing:

```python
def step_loss(inp, tar):                       # unchanged from today
    out = learner(inp)
    return braintools.metric.squared_error(out, tar).mean()

@brainstate.transform.jit
def f_train(inputs, targets):
    grads, loss = learner.etrace_grad(
        inputs, targets, step_fn=step_loss,
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
  unchanged — under an out-based `loss_fn` design it would have needed
  restructuring.
- `examples/100-gru-on-copying-task.py` loses both the manual `grads / T`
  correction (now `reduction='mean'`) and the manual warm-up `for_loop` (now
  `etrace_evolve`). `examples/drtrl/05-vjp-multi-step.py:59` loses its warm-up
  `for_loop` the same way.
- **Every migrated site stays at `chunk_size=None`.** Per P5 no example uses
  `MultiStepData` today, and migration is not the place to change a learning
  rule. Window mode therefore ships exercised only by `sequence_test.py`, which
  is why tests 4–11 carry it.
- BPTT baselines are **not** touched. They have no learner and no trace.
- `docs/quickstart/rnn_online_learning.ipynb`,
  `docs/quickstart/snn_online_learning.ipynb`, `docs/tutorials/batching.ipynb`,
  `docs/tutorials/drtrl.md` and `docs/tutorials/pp_prop.md` are updated to teach
  the new API first. `docs/apis/algorithms.rst` documents both methods and
  `ETraceVmap`.
- `docs/specs/2026-07-25-known-limitations.md` F-30 and F-35 each gain a sentence
  noting the driver as a new route to the condition.
- `examples/tests/test_smoke.py`, `test_compile_modes.py` and the per-family
  smoke suites must keep passing. The migration is behaviour-preserving except
  for the gradient *scale*, which is the one thing to watch: a site that
  previously summed keeps its tuned learning rate by passing `reduction='sum'`,
  while a site that hand-divided by `T` switches to the default `'mean'` and
  drops the division. Each migrated site declares which of the two it was, and
  the smoke assertion (loss decreases) is the check that it declared right.

`braintrace/__init__.__all__` gains `ETraceVmap`. The two methods need no export.

## Rejected alternatives

**`loss_fn(out_t, target_t)`, with the API calling the model.** The first draft.
It buys chunk-size transparency — the API can `jax.vmap` a per-step loss over the
window axis, so the user's function never changes shape. Rejected on two grounds.
First, generality: it cannot express a multi-head model, an auxiliary objective,
or a per-call `modulator=`, and it hides the `learner(...)` call that is the
whole subject of the library. Second, the transparency it buys is worth less than
it appears — window mode already requires `vjp_method='multi-step'`, a materially
different learning rule (P3), so `chunk_size` was never a knob a user could turn
without editing `compile(...)` anyway.

**Always driving through `MultiStepData`, including length-1 windows**, to make
the two modes one. Impossible for gradients: the default
`vjp_method='single-step'` refuses it (P1). Note this is *not* true for evolution
(P4), which is why the two methods have separate matrices.

**`chunk_size=1` meaning the plain single-step path**, with window mode starting
at 2. Rejected in favour of `None`, which makes the mode a *type* distinction
rather than a special case carved out of the integer range — and which leaves
`chunk_size=1` free to mean the genuine degenerate window that test 4 exercises.

**A user-applied transpose to make window mode work under `ETraceVmap`.**
Rejected: silently wrong if omitted, and undetectable when `B == k`. Refusal
with a message is better than a contract users can forget.

**Deriving the accumulator from `jax.eval_shape`** rather than from parameter
values. Held in reserve rather than adopted: P6 shows the simple construction is
dimensionally correct here, and `eval_shape` through `brainstate`'s State
machinery is unverified complexity for a problem that does not currently exist.

## Open questions

**One, for the spec's owner:** should `chunk_size` be renamed?

The review's remaining objection is that one keyword changes five things at once
(now tabulated under "Public API") and that `None` versus `1` reads as a
degenerate case rather than a mode switch. A name like `window_size` or
`vjp_window_size` would say what it selects; `chunk_size` says only how much.
The rename is cosmetic and cheap now, expensive after release. It is left open
because the current name was chosen deliberately, and the substance of the
objection is answered by the table regardless of the name.
