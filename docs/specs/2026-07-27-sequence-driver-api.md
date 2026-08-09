# Sequence driver API — `etrace_grad` and `etrace_evolve`

Status: spec, revised after external review
Baseline: commit `8b7cdc7`
Target release: 0.2.5

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
4. `braintrace.SequenceDriverMixin` — exported because it is where the two
   methods' docstrings live, so `docs/apis/algorithms.rst` has something to
   point `autosummary` at. Added after the draft, which listed only
   `ETraceVmap`: leaving it private would have meant the reference docs could
   not render the API they document.
5. Migration of all 25 call sites in `examples/` and the docs.

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

And two more, established while building the driver:

| # | fact | source |
| --- | --- | --- |
| P7 | `chunk_size` is **already** the public name for this quantity, and **`chunk_size=1` already means the plain single-step path** — `_as_window` returns `seq[0]` unwrapped at 1 and `MultiStepData(seq)` above it | `dni.py:410` (`train_synthetic_gradient(..., chunk_size: int = 1)`), `dni.py:663-667`; also `oracle.py:145` |
| P8 | with a fitted DNI synthesizer attached, `reduction='mean'` equals `reduction='sum'` divided by `T` **exactly** (`|mean − sum/T| = 0`), while a synthesizer fit at `chunk_size=1` and deployed at `chunk_size=2` moves the gradient 1.9% relative | measured against the implementation; see "DNI" below |
| P9 | a model whose ETP op receives an *unbatched* input while its hidden state is batched has a trace state that changes shape on first `update()` (e.g. `(3,4,2) → (1,3,4,2)`), so it cannot be driven under `scan`/`for_loop` **at all** — the hand-written block fails identically | `oracle_models.unit_weight_rnn` under `D_RTRL`; verified the status-quo block fails the same way |
| P10 | a `step_fn` that raises mid-loop leaves the learner **untouched**, not partially advanced: `running_index` stayed `0` and the hidden state was unchanged after an exception at step 3, and the learner was still usable | measured on a real learner; functional `scan` semantics — see "State lifecycle" |
| P11 | `compile(vmap=True)` cannot batch a model that allocates its `HiddenState` in `__init__` (`BatchAxisError` on the first call); the fixture must defer state creation to `init_state`, as `braintrace.nn.ValinaRNNCell` does | `oracle_models.tanh_rnn` fails; `ValinaRNNCell` + a plain readout works |
| P12 | under `vjp_method='single-step'` every **plain** (non-ETP) parameter's gradient is exactly zero (F-33), so any fixture asserting on a plain key must run `'multi-step'` or the assertion is vacuous | `docs/specs/2026-07-25-known-limitations.md` F-33; reproduced through the driver (`win` = `0.0` vs BPTT `0.65`) |

**Status 2026-08-08:** P12/F-33 is resolved. Single-step execution now gives
plain-only parameter paths their exact current-step VJP gradients; ETP-owned
paths continue to receive eligibility-trace gradients. The row above is retained
as the observation that shaped this historical design.

Consequences, all load-bearing below:

1. **Windowed *gradients* require `vjp_method='multi-step'`** (P1). The API
   validates this up front rather than letting the executor's error surface
   three frames down.
2. **Windowed *evolution* requires nothing** (P4). `etrace_evolve` therefore
   gets its own compatibility matrix, not a copy of `etrace_grad`'s.
3. **Windowing was never a free knob** (P3). Raising it already means editing
   `compile(...)` and adopting a materially different rule — which is why the
   step-function design's shape change is affordable.
4. **`chunk_size=1` means the plain path here too, and window mode starts at
   `k >= 2`** (P7). Diverging would give one public parameter name two opposite
   meanings at the same value, at exactly the F-35 boundary where users are
   instructed to match the two APIs. Nothing is lost: P2 shows a length-1 window
   is the same computation as the plain call anyway, so the configuration the
   driver stops exposing was never one worth choosing — and it would have been
   refused on single-step learners for no gain.
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
| `None` (default) or `1` | `seq[t]` | a **scalar** | nothing — legal on either `vjp_method` |
| `int k >= 2` | `seq[t:t+k]`, shape `(k, ...)` | a **`(k,)`** vector of per-step losses | `vjp_method='multi-step'`; `T % k == 0`; not an `ETraceVmap` |

Window mode changes five things at once, which is a real cost of the knob and is
listed here rather than discovered:

| what changes | `chunk_size` of `None` or `1` | `chunk_size=k >= 2` |
| --- | --- | --- |
| slice shape into `step_fn` | `seq[t]` | `(k, ...)` |
| `step_fn` return | scalar | `(k,)` |
| model call | `learner(x)` | `learner(MultiStepData(x))` — the user wraps |
| admissible learners | any | `vjp_method='multi-step'`, non-vmapped |
| learning rule | per P3, a different rule from the other column |

In window mode the user wraps their own model inputs. The API does **not** wrap,
because it cannot know which of the passed sequences are model inputs and which
are targets — that is `step_fn`'s business, and the price of it deciding.

**Why `1` is the plain path and not a length-1 window.** `chunk_size` is not a
new parameter: `train_synthetic_gradient` already takes one, defaulted to `1`,
and `dni.py:663-667` resolves `chunk_size == 1` to `seq[0]` — the unwrapped,
plain single-step call — reserving `MultiStepData` for `k >= 2` (P7). The driver
adopts that encoding exactly. The alternative — `1` meaning a genuine length-1
`MultiStepData` window — would give one public name two opposite meanings at the
same value, and the seam would fall on F-35, the one place this spec instructs
users to match `chunk_size` across the two APIs. Someone fitting at the default
`chunk_size=1` and driving at `etrace_grad(chunk_size=1)` must be right to
believe they matched.

`None` is retained as the default because it says *no chunking* more clearly
than `1` does, and because it is the value every migrated call site passes
implicitly. The two spellings are exactly synonymous; `None` is what the
docstrings use.

Note that `oracle.chunked_online_param_gradients` does **not** special-case `1`
(it always wraps, and tolerates a short final chunk). It is test support, per its
own docstring, and the user-facing DNI encoding is the one to match. The two
agree for every `k >= 2`, which is the range where the driver uses windows.

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

`aux` stacks over **windows**, leading axis `T // k` (which is `T` in plain
mode). No reshape magic is applied to it — the driver cannot know whether a
user's aux carries a step axis. A user wanting per-step aux in window mode
returns a `(k, ...)` aux and reshapes it themselves.

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
| `chunk_size` of `None` or `1` | legal | legal |
| `chunk_size=k >= 2` | **legal** — P4; contrast `etrace_grad`, which refuses | legal |

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

Examples 08 and 14 now demonstrate the supported route: users needing windowed
gradients on a batch use the batched (non-vmap) mode, which carries the batch
axis inside the compiled graph rather than outside it.

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
- `running_index` advances once per completed timestep: by one in plain mode
  and by `k` for each `k`-step window, reaching `T` in both modes.
- **On failure, state is left untouched.** A `step_fn` that raises mid-loop
  aborts before anything is written back: the body is traced into a functional
  `brainstate.transform.scan`, so state reaches the learner only when the whole
  transform completes. Measured — an exception at step 3 of a real learner left
  `running_index == 0` and the hidden state unchanged, and the learner was
  still usable afterwards.

  This is the opposite of the draft's claim that the learner is left
  "partially advanced", which assumed the scan wrote state as it went. The
  driver needs no rollback to get transactional behaviour; it inherits it.
  (P10)

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

With `chunk_size = k >= 2` and `n = T // k`:

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

F-30 was resolved on 2026-08-08. IO-factorized learners now index the f-side
warm-up correction by the exact age of the trace being contracted. A `k`-step
window advances the stored trace and `running_index` by `k`; its backward pass
contracts the window-entry trace at age `0, k, 2k, ...`. Driver regressions
require the corresponding ages and compare one six-step trace roll with three
two-step rolls.

### DNI: matching the synthesizer's deployment contract

F-35 records that a `train_synthetic_gradient` fit is valid only for the exact
`(loss_fn, chunk_size)` pair it was trained on, that **nothing checks that
deployment matches**, and that a mismatch is not degraded DNI but noise shaped
like a cotangent — measurably worse than leaving DNI off.

The driver's new parameters are **not** all new ways to mismatch, and an earlier
draft of this section was wrong about which are. Measured (P8): with a fitted
synthesizer attached, `reduction='mean'` and `reduction='sum'` differ by exactly
`1/T` — `|mean − sum/T| = 0`, not merely small. `reduction` divides the
*accumulated* gradient after the scan; it never enters the differentiated
objective, so it cannot desynchronize a synthesizer. It is a learning-rate
rescale, not an F-35 surface.

What does remain an F-35 surface:

- **`chunk_size`**, the documented one. Measured (P8): a synthesizer fit at
  `chunk_size=1` and deployed at `chunk_size=2` moves the gradient by 1.9%
  relative against one fit at `chunk_size=2`.
- **`mask`**, which *does* enter the differentiated objective — a zero-weighted
  step changes the loss the synthesizer's cotangent is predicting the future of.

The correspondence, which the docstrings must state:

| `train_synthetic_gradient` | `etrace_grad` |
| --- | --- |
| `chunk_size=1` (the default) | `chunk_size=None` (the default) or `1` — synonyms |
| `chunk_size=k >= 2` | `chunk_size=k` — the same integer, meaning the same thing |
| `loss_fn(output)` | the objective `step_fn` actually descends, *after* `reduction` and `mask` |

The `chunk_size` correspondence is **literal equality at every value passable to
both**, which is the entire reason the driver adopts DNI's encoding rather than
its own (P7). A user who reads "match the `chunk_size` you fit with" and types
the same number is correct, with no mental mapping at the exact point where
being wrong is silent.

The trap specific to this driver is therefore a `mask` that was not present
during fitting: it changes which steps contribute to the objective at all, so
the synthesizer's cotangent predicts the future of a loss that is no longer the
one being descended. That is an F-35 mismatch wearing new clothes.

The driver does not enforce the match — per F-35 the learner never sees the
caller's objective. `sequence_test.py` pins both halves: that a `chunk_size`
mismatch moves the gradient, and that `reduction` does **not** — the latter being
the more valuable regression, since moving the reduction inside the
differentiated objective would be a natural-looking refactor that silently
breaks DNI.

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
   generality claim for `step_fn` is scoped to plain mode; in window mode an
   objective must be computable from the stacked `(k, ...)` **outputs**.
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

Test 32 pins this. Should a future `brainunit` change break it, the replacement
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
| `chunk_size` neither `None` nor an int | `TypeError` |
| `chunk_size` an int `< 1` | `ValueError` — same bound and message shape as `dni.py:528` |
| `chunk_size=1` | **accepted, and identical to `None`** — the plain path (P7) |
| `chunk_size >= 2` and `T % chunk_size != 0` | `ValueError` — refused, never truncated, matching `train_synthetic_gradient`'s existing contract |
| `chunk_size >= 2` and `_seq_vjp_method != 'multi-step'` (**`etrace_grad` only**) | `ValueError` naming the fix (`compile(..., vjp_method='multi-step')`), raised **before** any tracing |
| `chunk_size >= 2` on an `ETraceVmap` (**both methods**) | `ValueError` naming the batched non-vmap mode as the alternative |
| plain mode and `step_fn` returns a non-scalar | `ValueError` stating the scalar contract |
| `chunk_size=k >= 2` and `step_fn` returns anything but shape `(k,)` | `ValueError` naming both the expected and the received shape |
| `mask` shape not `(T,)` | `ValueError` naming both shapes |
| `mask` non-binary | **accepted** — `mask` is a weight vector |
| `reduction` / `loss_output` not a legal value | `ValueError` listing the legal ones |
| `has_aux=True` and `step_fn` returns a non-pair | error from the underlying `grad`, not silently mis-unpacked |
| all-zero `mask` | zero gradients **for finite, differentiable step losses**; the `max(mask.sum(), 1)` denominator prevents division by zero but cannot prevent `0 · NaN = NaN`, nor a NaN arriving through a zero cotangent on an undefined derivative |
| `step_fn` raises mid-loop | propagates; learner state **untouched** — the functional `scan` never writes back (see State lifecycle) |
| learner not compiled | the existing `ValueError` (`"The etrace algorithm has not been compiled"`); the driver adds no guard of its own |

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

4. `chunk_size=1` is **exactly** `chunk_size=None` — identical gradients,
   identical losses, identical learner state afterwards — on a single-step *and*
   a multi-step learner. Not "agrees to round-off": the same code path. This is
   the test that pins the P7 alignment, and the one that fails loudly if someone
   later re-splits the two.
5. The driver's plain path agrees to float32 round-off with a length-1
   `MultiStepData` **constructed directly**, outside the driver, on a multi-step
   learner (measured `2.4e-07`, P2). This is the evidence that the configuration
   the driver stops exposing was never worth exposing; it is a test *about* the
   library, not about the driver's surface.
6. On a `vjp_method='single-step'` learner, `etrace_grad(chunk_size=2)` raises
   `ValueError` before tracing — not the executor's `NotImplementedError` — while
   `etrace_grad(chunk_size=1)` does **not** raise. The pair is what makes the
   `1`-is-plain encoding observable.
7. On the same learner, `etrace_evolve(chunk_size=2)` **succeeds** (P4). The two
   matrices differ, and this is the test that says so.
8. A learner whose `_seq_vjp_method` is `None` is refused at `chunk_size >= 2`.
9. `chunk_size` divides `T` is enforced at `k >= 2`; a ragged length raises.
   `chunk_size=1` can never trigger it, and is asserted not to.
10. `chunk_size=0`, a negative `chunk_size`, and a non-int `chunk_size` each
    raise, with the same bound as `dni.py:527-528`.
11. `step_fn` sees `seq[t]` under both `None` and `1`, and `(k, ...)` under
    `k >= 2` — asserted by a `step_fn` that records the shape it was handed.
12. A `step_fn` returning the wrong rank for its mode raises the documented
    `ValueError`.
13. Gradients at plain mode and `chunk_size=k >= 2` differ for an approximate
    algorithm (they must — the window is a real knob) and agree for an exact one
    under the regime its math guarantees. Per AGENTS.md, the approximate
    assertions go through the finite-window path, never a whole-sequence VJP.

**Masking**

14. The evolve/mask equivalence, **using two identically seeded learners**,
    element-wise.
15. A mid-sequence mask is *not* equal to concatenating gradients from two
    independent sequences — the negative control proving the trace crosses the
    zero-weighted span.
16. All-zero mask on a finite, differentiable loss → exactly zero, finite.
17. A non-binary mask reweights as documented.

**vmap**

18. `ETraceVmap.etrace_grad` matches the unbatched learner's per-lane gradients
    in plain mode.
19. `ETraceVmap.etrace_grad(chunk_size=k)` and `etrace_evolve(chunk_size=k)` both
    raise `ValueError` at `k >= 2`, **tested at `B != k` and at `B == k`** — the
    latter being the case that would otherwise pass silently. At `chunk_size=1`
    neither raises, since that is plain mode.
20. `compile(..., vmap=True)` still satisfies `isinstance(..., brainstate.nn.Vmap)`.

**Surface**

21. All four `(return_value, has_aux)` combinations return the documented arity;
    `losses` is `(T,)` under `'per_step'`/`'masked'` and **scalar** under
    `'scalar'`; `aux` stacks to leading `T // k`.
22. All three `loss_output` values return the documented thing, including that
    `'per_step'` reports a real number on a zero-weighted step where `'masked'`
    reports zero.
23. `weights=` restricted to a subset returns only those keys and leaves the rest
    untouched.
24. Three sequences are sliced in lockstep and passed positionally in order.
25. A `SingleStepData` / `MultiStepData` passed as a sequence raises `TypeError`.
26. A `step_fn` supervising one head of a two-head model works in plain mode; a
    hidden-state-reading regularizer works in plain mode and is documented as
    unsupported in window mode (limitation 2).

**State lifecycle**

27. `evolve(a); grad(b)` composes into one trajectory: equal to `grad` over the
    concatenation with a zero-weight prefix (this is test 14 read from the other
    side, kept separate because it pins *composition*, not masking).
28. Repeated `etrace_grad` calls continue rather than reset; `running_index`
    advances by `T` in both plain and window mode.

**Algorithm interactions**

29. F-30: one six-step trace roll and three two-step rolls produce the same
    stored final traces and completed-step count; the multi-step solver receives
    each window-entry trace age (`0, 2, 4` for the latter partition).
30. F-35: a `reduction` mismatch between `train_synthetic_gradient` and
    `etrace_grad` degrades DNI, in the shape of the existing window-size test.
31. The F-35 `chunk_size` correspondence is an identity: fitting with
    `train_synthetic_gradient(chunk_size=c)` and driving with
    `etrace_grad(chunk_size=c)` is the matched case for every `c`, including
    `c=1` against the driver's `None`. This is the test that would catch a future
    divergence between the two encodings.

**Robustness**

32. `brainunit`-valued weights survive the accumulator, the mask multiply and the
    mean division — pinning P6.
33. Each error-table row raises the stated type with a message naming the
    offending value.
34. `etrace_evolve` with `step_fn=None` drives the learner and, under window
    mode, wraps in `MultiStepData` itself; with a custom `step_fn` it does not.
35. `etrace_evolve(return_outputs=True)` stacks to leading `T // k`, and
    `return_outputs=False` returns `None`.
36. Both methods work inside an outer `brainstate.transform.jit` and eagerly.

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
  is why tests 4–13 carry it.
- BPTT baselines are **not** touched. They have no learner and no trace.
- `docs/quickstart/rnn_online_learning.ipynb`,
  `docs/quickstart/snn_online_learning.ipynb`, `docs/tutorials/batching.ipynb`,
  `docs/tutorials/drtrl.md` and `docs/tutorials/pp_prop.md` are updated to teach
  the new API first. `docs/apis/algorithms.rst` documents both methods and
  `ETraceVmap`.
- `docs/specs/2026-07-25-known-limitations.md` F-30 and F-35 each gain a sentence
  noting the driver as a new route to the condition.
- `examples/tests/smoke_test.py`, `compile_modes_test.py` and the per-family
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

**`chunk_size=1` meaning a genuine length-1 `MultiStepData` window**, with `None`
as the only spelling of the plain path. This was the previous draft's choice, on
the grounds that `None`-versus-int makes the mode a *type* distinction rather
than a special case carved out of the integer range. Reversed on reading
`dni.py:663-667`, where `chunk_size == 1` already resolves to the unwrapped
plain call in the shipped public API (P7). Keeping the distinction would have
given one parameter name two opposite meanings at the same value, precisely at
the F-35 boundary where users are told to match the two APIs. The `None`-versus-
int type distinction survives anyway — `None` is still the default and still the
spelling the docstrings use; `1` is now a synonym for it rather than a rival
meaning. What is genuinely lost is the ability to request a length-1 window
*through the driver*, which P2 shows is the same computation as the plain call
and which would have been refused on single-step learners for no gain. Test 5
keeps that measurement by constructing the wrapper directly.

**Renaming `chunk_size` to `window_size`.** The review's objection — that the
name says how much rather than what it selects, and understates that window mode
changes the learning rule — is fair on its own terms and was rejected on
codebase-consistency grounds: `chunk_size` is already the public name for this
quantity in `train_synthetic_gradient` (`dni.py:410`) and
`chunked_online_param_gradients` (`oracle.py:145`), and `dni_test.py:422` is
titled after the trap the parameter exists to prevent. A second name for one
concept would put the seam on F-35, where a mismatch is silent and produces
noise shaped like a cotangent — and `dni.py`'s `chunk_size` carries the same
"changes the learning rule" property while being documented, not renamed. The
objection is answered by the "what window mode changes" table instead.

**A user-applied transpose to make window mode work under `ETraceVmap`.**
Rejected: silently wrong if omitted, and undetectable when `B == k`. Refusal
with a message is better than a contract users can forget.

**Deriving the accumulator from `jax.eval_shape`** rather than from parameter
values. Held in reserve rather than adopted: P6 shows the simple construction is
dimensionally correct here, and `eval_shape` through `brainstate`'s State
machinery is unverified complexity for a problem that does not currently exist.

## Open questions

None. The naming question is resolved in "Rejected alternatives": the name stays
`chunk_size`, matching `train_synthetic_gradient` and
`chunked_online_param_gradients`, and the encoding is aligned to DNI's so that
`chunk_size=1` means the plain path in both.

Deferred to future work, each with its reason recorded above rather than left
undecided:

- Window mode under `ETraceVmap`, which needs an explicit transpose adapter and
  an `in_axes` naming the batch axis, plus a test at `B != k`.
- Lane-aware masks, which need the mask to enter inside the vmapped region.
- Skipping the VJP backward on zero-weighted steps, which the `etrace_evolve`
  split already covers for the contiguous case.
