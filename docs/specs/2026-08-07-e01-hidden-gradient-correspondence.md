# E-01 — check the hidden↔gradient correspondence for real

Status: proposed
Scope: `braintrace/_algorithm/vjp_base.py`, `braintrace/_compiler/hidden_group.py`,
`braintrace/_compiler/hidden_pertubation.py`
Backlog entry: E-01 in `2026-08-07-deferred-engineering-backlog.md`

## The defect

`braintrace/_algorithm/vjp_base.py:1046` opens the branch that turns the
backward pass's raw cotangents into the per-hidden-group learning signal
`dl2h`. It carries a standing TODO:

```python
if self.graph_executor.is_single_step_vjp:
    # TODO: the correspondence between the hidden states and the gradients
    #       should be checked.
    #
    assert len(dg_etrace_params) == 0
    assert self.graph.hidden_perturb is not None
    assert len(self.graph.hidden_perturb.perturb_vars) == len(dg_hid_perturb_or_dl2h)
    ...
else:
    assert len(dg_last_hiddens) == len(self.hidden_states)
    assert set(dg_last_hiddens.keys()) == set(self.hidden_states.keys()), (...)
```

Two distinct problems, which is why the TODO has outlived several refactors.

**1. Every guard here is an `assert`, so `python -O` removes all of them.** On an
optimised interpreter the correspondence is checked by nothing at all. This is
not a hypothetical interpreter mode: `-O` is what `PYTHONOPTIMIZE=1` sets, and
it is common in container images and in some benchmark harnesses.

**2. Even with asserts enabled, none of them checks a *correspondence*.** They
check *cardinalities* and, in the multi-step branch, a *key set*. What actually
has to hold is that cotangent *i* belongs to hidden state *i* — matching shape,
matching dtype, matching position. Nothing verifies that.

The consequence is the reason this is the highest-risk marker in the tree: a
cotangent attributed to the wrong hidden state produces a **wrong gradient, not
an error**. There are two concrete paths to it:

- `HiddenPerturbation.perturb_data_to_hidden_group_data`
  (`hidden_pertubation.py:185-198`) maps perturbation data onto paths by
  `zip(self.perturb_hidden_paths, perturb_data)` — positional. If the two ever
  disagree in order, every downstream trace is silently mis-attributed.
- `HiddenGroup.concat_hidden` (`hidden_group.py:441-447`) zips the value list
  against `self.hidden_states`. **`zip` truncates to the shorter argument**, so a
  short value list does not raise: it produces a concatenated array with a
  too-narrow trailing axis, and the error surfaces later as a shape mismatch in
  unrelated trace math, or not at all when the widths happen to coincide.

## What "correspondence" means here

For each hidden group `g` and each position `i` in `g.hidden_paths`, the
cotangent paired with `g.hidden_states[i]` must have:

- the same shape as that hidden state's value (i.e. `varshape`, plus the
  trailing `num_state` axis for a `HiddenGroupState`), and
- a dtype that matches the state's, up to the mantissa extraction
  `u.get_mantissa` already performs.

And the mapping from paths to cotangents must be **total**: every path a group
needs must be present in the cotangent collection, and the collection must not
carry paths no group claims.

## Changes

### 1. `HiddenGroup.concat_hidden` — close the `zip` truncation

Raise `ValueError` when `len(splitted_hid_vals) != len(self.hidden_states)`,
naming the group and both counts. This is the deepest guard and protects every
caller, not just the two in `vjp_base`.

Add the same check to `split_hidden`'s inverse assumption where it is cheap.

### 2. `HiddenPerturbation.perturb_data_to_hidden_group_data` — raise, and be total

- Convert the length `assert` (`hidden_pertubation.py:180`) to a `ValueError`.
- Replace the bare `path_to_perturb_data[path]` `KeyError` with an explicit
  check that names the group, the missing path, and the paths that *were*
  perturbed.

### 3. `vjp_base.py` — a real correspondence check, not asserts

Introduce a module-level helper:

```python
def _check_hidden_gradient_correspondence(groups, path_to_cotangent, *, source) -> None
```

which raises `ValueError` when the path set is not exactly what the groups
need, or when any cotangent's shape/dtype does not match its hidden state.
`source` names which branch produced the mapping, so the message says
`single-step hidden perturbation` or `multi-step last-hidden gradients` rather
than leaving the reader to guess.

Then, at both branches of `vjp_base.py:1046`:

- replace `assert len(dg_etrace_params) == 0` with a `ValueError` explaining
  that under `vjp_method='single-step'` the ETP weights are updated by the
  RTRL recursion, so a non-empty `dg_etrace_params` means the graph and the
  executor disagree;
- replace `assert self.graph.hidden_perturb is not None` with a `ValueError`
  pointing at `compile_graph`;
- call the helper before building `dl2h`, in both branches;
- delete the TODO.

The multi-step branch's existing key-set `assert` already carries a good
message; it becomes a `raise ValueError` with the same text.

## Cost

These run once per backward pass, on Python-level metadata (shapes, dtypes,
dict keys) — not on array data — so they are traced away entirely and add
nothing to the compiled program. No `jax` operations are introduced.

## Tests

Co-located per AGENTS.md rule 9. All are compiler/executor-contract assertions,
not learning-rule assertions, so per the `AGENTS.md` note they do not need the
finite-window oracle path.

`braintrace/_compiler/hidden_group_test.py`

- `concat_hidden` raises `ValueError` on a value list shorter than
  `hidden_states` — the case that currently truncates silently. Pin the
  pre-change behaviour in the test's docstring so the regression is legible.
- `concat_hidden` raises on a longer list too.
- `concat_hidden` still succeeds, unchanged, on the exact-length list.

`braintrace/_compiler/hidden_pertubation_test.py`

- `perturb_data_to_hidden_group_data` raises `ValueError` (not
  `AssertionError`) on a wrong-length `perturb_data`.
- It names the offending path when a group needs a path that was not perturbed.

`braintrace/_algorithm/vjp_base_test.py`

- The helper accepts a well-formed mapping.
- It raises on a missing path, an extra path, a shape mismatch, and a dtype
  mismatch — one test each, each asserting the message names the path.
- **The `-O` test**: run the shape-mismatch case in a `python -O` subprocess and
  assert it still raises `ValueError`. This is the assertion that actually pins
  E-01's core complaint; without it the fix could silently regress to asserts.

## Out of scope

Constructing a genuine end-to-end mis-attribution requires corrupting the
compiler's output, which no public API allows. The tests therefore drive the
checkers directly. That is the right level: the checkers are the contract, and
an end-to-end test would only re-verify that the compiler currently satisfies
it — which the existing oracle suite already does.
