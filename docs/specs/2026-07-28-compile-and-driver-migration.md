# Migration onto `compile`, `etrace_grad` and `etrace_evolve`

Status: spec, describing work carried out
Baseline: commit `f935856` (squash-merge of PR #151)
Target release: 0.3.0

## Goal

Make the repository teach **one** way to set a learner up and **one** way to run
it over a sequence.

`braintrace.compile` has been the documented entry point since before 0.2, and
PR #151 landed the two sequence drivers. Neither migration finished. #151's own
spec scoped a list of call sites, `main` moved underneath it twice (#148
reorganized the tutorials, #150 rewrote the API reference per-class), and
`examples/` has no CI coverage at all — `.github/workflows/CI.yml` runs
`pytest braintrace/`, nothing else. The result is that the answer to "how do I
train a model with braintrace?" depends on which file the reader opens:

| what the reader finds | where |
|---|---|
| `compile(...)` + `etrace_grad(...)` | `examples/drtrl/01`, `docs/advanced/batching.ipynb` |
| `compile(...)`, then a hand-written `scan`-accumulate | `docs/quickstart/quickstart.ipynb`, `docs/tutorials/drtrl.ipynb` |
| `Algo(model)` + `init_all_states` + `compile_graph` + `Vmap` | `examples/004`, `examples/100-gru` `'batch'` arm |

Two of those un-migrated sites did not merely look dated — they raised on the
first call. Neither was caught, because nothing runs them.

## Scope

**In scope.** Every remaining `examples/` site that constructs a learner the old
way or drives a sequence by hand; the docs notebooks; the class-level `Examples`
blocks in the library, which #150 turned into the per-class API reference pages
and which therefore *are* the API documentation for each algorithm.

**Out of scope.** Changing any algorithm's numerics. Changing `compile` or the
drivers themselves — the limitations found while auditing are recorded in
"Declared limitations" below and left in place. Dated spec documents under
`docs/specs/` other than this one. A user-facing migration page: this document
is a record of a completed change, not a guide, and the migration is small
enough that the class docstrings and `docs/advanced/batching.ipynb` carry the
teaching load.

## Measured facts this migration rests on

**M1. Both driver substitutions are bit-exact.** Checked against the code as
merged, not assumed from the #151 spec:

- `etrace_evolve(xs, return_outputs=True)` returns outputs **identical** to
  `brainstate.transform.for_loop(learner, xs)`.
- `etrace_grad(..., reduction='mean')` returns a gradient tree whose maximum
  absolute difference from the hand-written `scan`-accumulate-then-divide loop
  is **0.0**.

This is what licenses re-executing the notebooks and *requiring* the printed
numbers to be unchanged. Drift would mean the rewrite is not the equivalence
claimed, and would be investigated rather than accepted.

**M2. `compile` inside `jit` is fine.** `examples/100-gru-on-copying-task.py`
called `compile` inside `@brainstate.transform.jit` in one branch of an `if`
while the sibling branch carried a comment claiming compile "must live outside
jit". Both branches now run, in the same jitted function, and both pass.

**M3. `compile(vmap=True)` *is* the manual vmap expansion.** `_compile.py:308-330`
is literally `vmap_new_states(state_tag='new')` + `init_all_states` +
`compile_graph` on the axis-stripped sample + `ETraceVmap(learner,
vmap_states='new')` — the four steps `examples/drtrl/02-batching-vmap.py` writes
out by hand.

**M4. `jax.ShapeDtypeStruct` is not subscriptable.** `compile`'s vmap branch
strips the batch axis with `jax.tree.map(lambda a: a[0], example_inputs)`, so a
shape-only example input raises `TypeError` there. The `003-*` benchmarks build
their graph from a shape, never from data.

## The three canonical replacements

| old | new |
|---|---|
| `Algo(model)` + `init_all_states(model, batch_size=B)` + `compile_graph(x0)` | `compile(model, Algo, x0, batch_size=B)` |
| `Algo(model)` + `vmap_new_states(state_tag='new')` + `compile_graph(x0[0])` + `Vmap(..., vmap_states='new')` | `compile(model, Algo, x0, batch_size=B, vmap=True)` |
| `for_loop(learner, xs)` | `learner.etrace_evolve(xs)`, or `etrace_evolve(xs, return_outputs=True)` to keep the stack |
| `scan`-accumulate, then `grads / T` | `learner.etrace_grad(..., reduction='mean')` |
| `scan`-accumulate, no divide | `learner.etrace_grad(..., reduction='sum')` |

In the vmap row, `x0` carries the batch axis in **both** columns: `compile`
strips axis 0 itself to recover the per-sample example. Passing `x0[0]` to
`compile` is the one easy mistake, and it is why `docs/advanced/batching.ipynb`
says so in a comment at the call.

The last two rows are the same method; the distinction is only which reduction
reproduces the site's existing gradient scale. **Every migrated site declares
which of the two it was** — that declaration is the reviewable claim, since
choosing wrong rescales the effective learning rate by `T` and the smoke
assertion (loss decreases) would not necessarily catch it.

## Two bugs the migration fixes

Both shipped. Both are invisible to CI, which does not run `pytest examples/`.

**B1. `examples/drtrl/02-batching-vmap.py`** raised `AttributeError: 'Vmap'
object has no attribute 'etrace_grad'`. #151 migrated its training loop to
`etrace_grad` but left the wrapper as `brainstate.nn.Vmap`; only
`braintrace.ETraceVmap` carries the drivers. Fixed by switching the wrapper —
the file deliberately keeps the manual expansion (see below).

**B2. `examples/004-feedforward-conv-snn.py:302`** raised `TypeError:
ParamDimVjpAlgorithm.__init__() got multiple values for argument 'model'`. The
line was `D_RTRL(self.target, self.decay_or_rank, model=brainstate.mixin.Batching())`
— the model passed both positionally and by keyword, plus a `decay_or_rank`
positional that `D_RTRL` does not take (that belongs to `ES_D_RTRL`, whose
commented-out line directly above it was the original). It survived because
`OnlineBatchTrainer` is never instantiated in `main()` and `004` is skipped in
the smoke tests, which need the NMNIST dataset.

B2 is fixed by the migration itself: `compile(model, D_RTRL, inputs[0],
batch_size=B)` has nowhere to put either bad argument. The sibling batched paths
(`100-gru`'s `'batch'` arm, `003-batched`) pass no `Batching()` mixin either.

Because `004` is skipped, neither arm is reachable from any test. Both were
exercised directly against a synthetic batch, with `tonic` (the dataset library,
not installed) stubbed at import. That run is the only evidence B2 is *fixed*
rather than merely rewritten, and it also confirms both trainers compile the
same eligibility-trace graph (identical hidden groups and weight associations).

## What stays manual, and why

Five sites carried a `# kept manual:` comment. Checked against `_compile.py`,
**all five reasons were false or imprecise.** The code at three of them was
migrated; at the other two the code stays and the reason is corrected. Stating a
wrong reason is worse than stating none: it tells the next reader that a working
API does not work.

- **`examples/drtrl/02-batching-vmap.py`** — claimed the `vmap_states='new'` path
  was "not yet covered by `compile()`". By M3 it is exactly what `compile`
  covers. The file stays manual because it *is* the worked expansion: it is the
  one place a reader can see which state gets the per-sample axis and on which
  unbatched sample the trace graph is built. The docstring now says so and
  names the one-call form.
- **`examples/pp_prop/05-batching-vmap.py`** — carried the same false claim, and
  its docstring was stale on top of it: the file routes through
  `_shared.online_train_epoch`, which has used `compile(..., vmap=True)` since
  #151. The note is dropped and the docstring retitled to what the file does.
- **`examples/003-snn-memory-and-speed-evaluation-{all,batched,vmap}.py`** — the
  reasons were true but vague. They are now stated as the precise limitations
  (below): `-all` and `-batched` use a third state scheme `compile` does not
  offer, and all three pass a `ShapeDtypeStruct` example input, which M4 rules
  out of the vmap branch. These files also keep their **Python step loop**, which
  is not a `compile` question at all: `get_mem_usage()` is sampled between
  steps, and a fused `etrace_grad` scan would leave nowhere to sample. The loop
  is the measurement, not an un-migrated leftover.

BPTT baselines and eval-only re-init blocks keep their annotations unchanged.
They construct no learner and hold no trace, so there is nothing to migrate.

## Declared limitations

Found while auditing. Recorded, not fixed — each is a real gap that a caller can
hit, and naming it here is cheaper than the reader rediscovering it.

1. **`ETraceVmap` does not forward the introspection surface.** It carries
   `etrace_grad` / `etrace_evolve` and the `Vmap` call, but not `show_graph`,
   `graph`, `report` or `param_states`. A vmapped caller wanting a post-compile
   diagnostic must reach through `.module`. This is safe and is what
   `examples/004`'s vmap trainer now does, with a comment: `ETraceVmap`'s
   docstring warns against `.module`, but that warning is about **driving** the
   unbatched learner, which would give per-lane-wrong results. A read-only
   diagnostic is not driving.
2. **`compile(**options)` cannot carry an option named `model`.** It collides
   with the positional parameter. No current algorithm needs one, so this is a
   latent constraint on the option namespace rather than a present blocker.
3. **`compile` has no path for the `003-*` state scheme** —
   `vmap_init_all_states(state_tag='new')` for the per-sample states,
   `compile_graph` on a **batched** example, no wrapper at all, and an explicit
   `brainstate.transform.vmap(in_states=...)` used only for the reset. `compile`
   offers two schemes; this is a third.
4. **`compile(vmap=True)` cannot take a `jax.ShapeDtypeStruct` example input**
   (M4). Building a graph from a shape rather than from data is what a benchmark
   does, so the gap and limitation 3 are usually hit together.

## Sites changed

**Initialization → `compile`** (3 sites): `examples/100-gru-on-copying-task.py`
`'batch'` arm; `examples/004-feedforward-conv-snn.py` `OnlineVmapTrainer` and
`OnlineBatchTrainer` (the latter fixing B2).

**Hand-driven sequence → drivers** (2 sites): `examples/drtrl/08-operator-conv.py`
and `09-classification-mnist.py`, whose warm-up `for_loop` becomes
`etrace_evolve`. Both compute their loss at the final step only, so the gradient
below stays a single-step `grad` rather than an `etrace_grad` — the warm-up is
the only part the driver replaces.

**Docs notebooks.** One `scan`-accumulate block, byte-identical in
`docs/quickstart/quickstart.ipynb`, `docs/tutorials/drtrl.ipynb` and
`docs/tutorials/pp_prop.ipynb`, becomes `etrace_grad(..., reduction='mean')` —
`'mean'` because the block ended in `grads / inputs.shape[0]`. Their `evaluate()`
becomes `etrace_evolve(..., return_outputs=True)`. `docs/quickstart/concepts.ipynb`
collapses `online_sequence_gradient` to one `etrace_grad(..., reduction='sum')`,
`'sum'` because it accumulated without dividing.
`docs/tutorials/neural_network_layers.ipynb` swaps a `for_loop(learner, ...)` for
`etrace_evolve`. `import jax` becomes unused in three notebooks once `jax.tree`
does, and is dropped.

**Docs prose.** Four passages still named `brainstate.transform.scan` as the
mechanism — `quickstart` "What happened", the `drtrl` and `pp_prop` section
headings, `concepts`'s BPTT comparison table, and `batching`'s numbered
walkthrough. This matters as much as the code: prose describing the old
mechanism is why a reader would write a `scan` after reading migrated code.

**Docstrings.** Nine `Examples` blocks built a learner and stopped at one
forward call, so the per-class API pages never showed a driver. Each gains the
same tail, adapted to its own shapes. Two differ on purpose: `ThreeFactor` rides
its per-step `modulator` as a **second sequence**, which is the clearest
demonstration of why `step_fn` owns the model call — there is nowhere the
modulator needs threading through. `UORO` is multi-step by construction, so its
tail is the only documented example of **window mode**, with `chunk_size=k`,
`MultiStepData` inside `step_fn`, and a `(k,)` return.

`braintrace/__init__.py`'s layer-4 bullet now names the drivers, and its `Notes`
paragraph says what the one `compile` call replaces.

## Verification

`braintrace/_docs_examples_test.py`, which executed docstring snippets, was
deleted by #150. **Nothing in the repo runs them now**, so:

1. `pytest examples/ -q` — the only coverage these files have. Reaches
   `drtrl/02`, `08`, `09`, both `100-gru` arms, and `003-all`.
2. `examples/004`'s two trainers driven directly on a synthetic batch with
   `tonic` stubbed, since the file is skipped. This is the only proof of B2.
3. Every changed notebook re-executed with `nbclient`. By M1 the printed numbers
   must be **unchanged**; the two notebooks that ship without stored outputs are
   executed in a copy so the migration does not add outputs the docs never had.
4. All docstring snippets extracted and executed, sharing a namespace per file
   so blocks that continue each other across a `.. code-block::` directive
   resolve their names.
5. `python -m mypy braintrace` (the CI gate); every
   `docs/apis/algorithm_details/braintrace.*` toctree entry still resolves.
6. `pytest braintrace/ -n 6 -q` for regression.

The standing gap this exercise exposed is that items 1, 2 and 4 are all manual.
`examples/` and the docstring snippets are documentation that can break silently,
and both B1 and B2 are what that costs. Wiring either into CI is future work,
and is the change that would stop this document from needing a sequel.
