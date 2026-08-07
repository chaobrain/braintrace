# E-05: declare `jax` as a core dependency

Issue: [#160](https://github.com/chaobrain/braintrace/issues/160)

## What is missing

`[project].dependencies` in `pyproject.toml` lists `brainstate`, `brainunit`,
`brainevent`, `brainpy.state` and `braintools` — but not `jax`, which every
module in the package imports directly (`braintrace/_compatible_imports.py:16`
is the first line of the compiler's own compatibility layer). The package is
therefore installable in a state where its own metadata never says which JAX it
needs, and the version range CI actually exercises is invisible to a resolver.

The decision is to declare it with a floor and no cap. This spec records the
evidence for that floor, the argument against a cap, and what the change does
(and does not do) to the existing accelerator extras.

## Correcting the issue's premise

The issue says `jax` "resolves transitively through `brainstate`". That is not
where it comes from. `brainstate` 0.5.2's core requirements are `numpy`,
`brainunit` and `brainevent`; it declares JAX **only under extras**
(`jax[cpu]>=0.7.0; extra == "cpu"`, and the cuda12/cuda13/tpu equivalents).
Installing `brainstate` without an extra installs no JAX at all.

The dependencies that actually drag JAX in as a *core* requirement today are:

| distribution | JAX requirement (core) |
| --- | --- |
| `brainevent` (declared `>=0.1.2`) | `jax>=0.8.0` |
| `braintools` | `jax` (no floor) |
| `brainpy.state` | `jax` (no floor) |
| `brainstate` | none (extras only) |
| `brainunit` | none (extras only) |

So the floor a resolver sees today is `>=0.8.0`, and it holds only because
`brainevent` happens to declare it. If `brainevent` relaxed that line, or if a
future release dropped the constraint, braintrace's effective floor would move
without a single braintrace commit. That is the real defect: the floor is not
absent, it is *borrowed*.

## The floor: `jax>=0.8.0`

Three independent lines of evidence converge on 0.8.0.

**1. It is the lowest version CI tests.** `.github/workflows/CI.yml:37`:

```yaml
jax-version: [ "0.8.0", "0.9.0", "0.10.0", "" ]  # "" means latest version
```

Four matrix entries run the full suite; 0.8.0 is the floor of that set. A
declared floor below 0.8.0 would be an untested claim, which is precisely the
failure mode the issue objects to.

**2. Nothing in the source requires more than 0.8.0.** The JAX-facing
compatibility code is written to span versions rather than pin one:

- `scan_num_consts_carry` and `scan_params_add_ys`
  (`braintrace/_compatible_imports.py:91-148`) handle the JAX 0.11 "flattree"
  scan refactor — which removed the `num_consts` / `num_carry` scan params — by
  *capability* detection (`if 'num_consts' in params`, `if 'ft_out' not in
  params`), not by a version comparison. Both sides of each branch are live on
  the tested range.
- `new_jaxpr_eqn` is imported from `jax.extend.core` with a `jax.core` fallback
  for the versions that still expose it there.
- `stop_gradient_p` is imported from `jax._src.ad_util` with a trace-based
  recovery path that raises an explicitly actionable `ImportError` if a future
  JAX moves it.

None of these needs a version newer than the CI floor, so the code does not
push the floor above 0.8.0.

**3. Nothing in the source supports less than 0.8.0 either.** Two version
guards exist for older JAX — `jax.__version_info__ < (0, 6, 2)` in `new_var`
and `< (0, 7, 0)` in `is_jit_primitive` (`_compatible_imports.py:66,73`). They
are not evidence for a lower floor: no CI entry ever takes those branches, so
the claim "braintrace works on JAX 0.6" is unverified. They are cheap
defensive code, not a support commitment, and a floor is a commitment.

`jax>=0.8.0` therefore adds **no new constraint to any install that works
today** (the transitive floor from `brainevent` is already 0.8.0). What it
changes is ownership: the floor becomes a braintrace decision, backed by the
braintrace test matrix, instead of a side effect of a sibling package's
metadata.

## Why no upper cap

A cap was considered and rejected.

- **A cap published today is a claim about software that does not exist yet.**
  Capping at the current latest (`<0.12`) forbids the next JAX minor for every
  braintrace release already on PyPI, and makes a braintrace release a
  prerequisite for adopting each JAX release. The failure it prevents (a
  breaking JAX bump) is recoverable by a patch release; the failure it causes
  (a stale published cap) is not, because old artifacts cannot be edited.
- **The code is written to survive a bump.** Every JAX-version-sensitive site
  is capability-detected or guarded by `try`/`except ImportError` with a
  fallback, as enumerated above. The design assumption is already "JAX will
  move"; a cap would contradict it.
- **CI already detects the breakage a cap would guess at.** `CI.yml` runs on a
  daily `schedule` (`cron: '0 0 * * *'`) with an unpinned `pip install jax`, so
  a JAX release that breaks the compiler surfaces within 24 hours, against a
  real failure rather than a pre-emptive guess.
- **Caps on a leaf package poison a shared environment.** braintrace is one of
  several JAX consumers a user installs alongside each other. An upper cap here
  does not merely constrain braintrace; it constrains the whole environment's
  JAX, and pip reports the resulting conflict against braintrace.

If a specific JAX release is ever found to break braintrace, the answer is an
exclusion for that release (`jax>=0.8.0,!=X.Y.Z`) plus a fix — targeted at the
version that actually failed, not at every version after it.

## Interaction with the accelerator extras

This is the part worth stating explicitly, because it is the one place the
change could plausibly do harm.

```toml
[project.optional-dependencies]
cpu = ["jax[cpu]"]
cuda12 = ["jax[cuda12]"]
cuda13 = ["jax[cuda13]"]
tpu = ["jax[tpu]"]
```

**Adding bare `jax>=0.8.0` to the core list does not change what these extras
install.** Extras of the *same* distribution are **additive, not alternative**.
When a user runs `pip install braintrace[cuda12]`, the resolver collects two
requirements naming the one project `jax`:

- `jax>=0.8.0` (core)
- `jax[cuda12]` (extra)

It intersects the version specifiers (`>=0.8.0`) and unions the extra sets
(`{cuda12}`). Exactly one `jax` is installed, at a version satisfying the
floor, with the `cuda12` extra's own requirements pulled in alongside it.

The reason there is no CPU-vs-CUDA conflict to begin with is that **the
accelerator choice is not encoded in the `jax` wheel**. From JAX 0.11's
metadata:

- `jax`'s *core* requirements already include `jaxlib` (plus `ml_dtypes`,
  `numpy`, `opt_einsum`, `scipy`). A bare `jax` install is already runnable.
- `jax[cuda12]` **adds** `jax-cuda12-plugin[with-cuda]`; `jax[cuda13]` adds the
  cuda13 plugin; `jax[tpu]` adds `libtpu`.
- `jax[cpu]` contributes **no** `Requires-Dist` line at all in 0.11 — the `cpu`
  extra is empty and exists for symmetry. `jax[cpu]` is literally `jax`.

So there is no "CPU build" of `jax` that a bare requirement could force a user
onto. Backends are selected by *additional plugin packages*, which the extras
still contribute unchanged. `braintrace[cuda12]` keeps getting the CUDA wheel.

The `testing` and `dev` extras list `jax[cpu]` for the same reason and are
unaffected by the same argument.

### Why the floor is not repeated in the extras

The extras stay unversioned (`jax[cuda12]`, not `jax[cuda12]>=0.8.0`). The core
requirement already constrains the same distribution, so the floor applies to
every install path including every extra. Writing it once means it cannot drift
between the five places it would otherwise appear.

## Changes

1. `pyproject.toml` — add `"jax>=0.8.0"` to `[project].dependencies`.
2. `requirements.txt` — already lists a bare `jax`; raise it to `jax>=0.8.0` so
   the dev/CI install path states the same floor as the published metadata.
   (This does not disturb CI's matrix: `CI.yml` installs `requirements-dev.txt`
   and *then* `pip install jax==0.8.0`, which still satisfies `>=0.8.0`.)
3. `setup.py` — no change. It carries no dependency list; it exists only as the
   `build_py` shim that prunes the test payload from the wheel. All install
   metadata lives in `pyproject.toml`.
4. `requirements-dev.txt` / `requirements-doc.txt` — no change; both begin with
   `-r requirements.txt` and inherit the floor.

## Verification

The built wheel's `METADATA` carries the requirement in the core (unconditional)
block, with every extra unchanged:

```
Requires-Dist: jax>=0.8.0
Requires-Dist: brainstate>=0.5.2
...
Requires-Dist: jax[cuda12]; extra == "cuda12"
```

The "does this force a CPU build?" question was answered by resolution rather
than by reading, via `pip install --dry-run --report` against the built wheel
with `[cuda12]`: the resolution installs `jax` / `jaxlib` 0.11.0 **plus**
`jax-cuda12-plugin`, `jax-cuda12-pjrt` and five `nvidia-*-cu12` packages. The
CUDA install path is intact.

## Tests

`braintrace/_version_test.py` (new — the module had no co-located test, against
`AGENTS.md` rule 9) pins the invariants that would otherwise regress silently:

- `__version__` / `__version_info__` agree, and `__version__` is what
  `[tool.setuptools.dynamic]` reads.
- `[project].dependencies` declares `jax` with a `>=` floor.
- **The declared floor equals the lowest JAX in the CI matrix.** This is the
  issue's actual complaint expressed as an assertion: if someone adds a lower
  entry to the matrix, or raises the floor without extending the matrix, the
  metadata and the tested range have diverged and the test fails.
- `requirements.txt` states the same floor as `pyproject.toml`.
- The accelerator extras still request `jax[<backend>]`, so a future edit
  cannot quietly turn `braintrace[cuda12]` into a plain `jax` install.

Both `pyproject.toml` and `requirements.txt` ship in the sdist (which is what
downstream packagers run the suite from) but not in the wheel, so the file-based
tests skip when the files are absent. The CI-matrix cross-check additionally
skips outside a repo checkout, since `.github/` is in neither artifact.

### Edge cases considered

- **Wheel-only install** — no `pyproject.toml` next to the package; tests skip
  rather than fail.
- **Stale installed `dist-info`** — the tests deliberately read the repo's
  `pyproject.toml` rather than `importlib.metadata`, because a source checkout
  commonly shadows an older installed distribution and the metadata would
  describe the wrong version.
- **Matrix `""` entry** — the empty string means "latest" and is excluded from
  the minimum computation.
- **A future matrix entry with a pre-release or trailing suffix** — the parser
  compares dotted numeric tuples and ignores non-numeric matrix values.
