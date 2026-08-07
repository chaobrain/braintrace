# E-08 — Prune unreferenced assets from `docs/_static/`

- **Issue**: [#163](https://github.com/chaobrain/braintrace/issues/163)
- **Date**: 2026-08-07
- **Scope**: repository hygiene only; no source, API, or docs-content change.

## Problem

The 0.2.5 package audit flagged `docs/_static/` as carrying several megabytes of
binary assets that no page in the docs build references. Every clone pays for
them. Some entries were already listed in `.gitignore` while (allegedly) still
being tracked, which is contradictory and suggested an unfinished cleanup.

## Method

A deletion of tracked binaries is irreversible from the point of view of a
consumer of a shallow clone, and a wrong deletion silently breaks a docs page or
the rendered README. So every candidate was checked against **four** independent
reference surfaces, and a file was deleted only if all four came back empty:

1. **Repo-wide literal grep**, per file, over the bare filename, the stem
   without extension, and the `_static/<name>` form. Covered `docs/` (`.rst`,
   `.md`, `.ipynb`, `conf.py`, `_templates/`), `README.md` and the other
   top-level Markdown, `braintrace/` docstrings, `examples/`, `.github/`,
   `pyproject.toml`, `setup.py`, `.readthedocs.yml`.
2. **`docs/conf.py` read in full**, not just grepped — an asset can be pulled in
   implicitly by `html_logo`, `html_favicon`, `latex_logo`, or a
   `html_theme_options` logo key without its filename ever appearing in a page.
3. **Packaging surface** — whether an sdist/wheel ships `docs/_static/`.
4. **External raw-URL surface** — whether any *immutable* published artifact
   (a PyPI release page renders the README frozen at release time) points at
   `raw.githubusercontent.com/.../docs/_static/<name>`. Deleting such a file
   from `main` breaks an already-published page.

### What surfaces 2–4 returned

`docs/conf.py` sources both the logo and the favicon from a remote URL, not from
`_static`, so no asset is implicitly live:

```
docs/conf.py:165: html_logo = "https://brainx.chaobrain.com/images/braintrace.webp"
docs/conf.py:169: html_favicon = "https://brainx.chaobrain.com/images/braintrace.webp"
```

There is no `latex_logo`, and `html_theme_options` carries only
`show_toc_level`. Packaging sets `include-package-data = false`
(`pyproject.toml:12`) and no `package_data`/`MANIFEST.in` names `docs/`, so no
distribution ships these files.

The external surface produced the one genuine surprise, recorded below.

## Reference table

| File | Size (bytes) | Referenced by | Verdict |
| --- | ---: | --- | --- |
| `braintrace-learning-map.svg` | 2,641 | `docs/index.rst:125` — `.. image:: _static/braintrace-learning-map.svg` | **KEEP** |
| `braintrace.png` | 1,111,047 | PyPI release pages for **0.1.1** and **0.1.2**, whose frozen README renders `<img src="https://raw.githubusercontent.com/chaobrain/braintrace/main/docs/_static/braintrace.png">` | **KEEP** |
| `architecture-rnn-ltr.png` | 15,273 | nothing | PROVEN UNREFERENCED — delete |
| `etraceop-xw2y.png` | 25,437 | nothing | PROVEN UNREFERENCED — delete |
| `etraceop-yw2w.png` | 53,980 | nothing | PROVEN UNREFERENCED — delete |
| `expo-synapse.png` | 24,877 | nothing | PROVEN UNREFERENCED — delete |
| `model-dynamics-supported.png` | 355,399 | nothing | PROVEN UNREFERENCED — delete |
| `neuron-spike.gif` | 8,641 | nothing | PROVEN UNREFERENCED — delete |
| `rnn-applications.png` | 129,540 | nothing | PROVEN UNREFERENCED — delete |

The only in-repo hits for any of the seven deleted names are `.gitignore:199`
(the `.pptx` ignore rule, see below) and the audit backlog text itself at
`docs/specs/2026-08-07-deferred-engineering-backlog.md:119-121`. Neither is a
docs reference.

**Total reclaimed: 613,147 bytes (~599 KiB).**

## Findings that correct the issue text

Two claims in #163 do not hold against the current tree, and both make the
change smaller and safer than the issue implies.

### 1. The 2.7 MB `.pptx` is already gone

`docs/_static/model-dynamics-supported.pptx` is neither in `git ls-files` nor on
disk, and the same is true of `etrace_op_functions.{pptx,pdf}`. The "gitignored
yet still tracked" contradiction the issue describes does not exist: those
`.gitignore` lines guard files that are correctly untracked. The headline
"~4.4 MB" figure therefore over-counts by the size of a file that is not in the
repository; the real recoverable amount is ~599 KiB.

### 2. `braintrace.png` is live, not dead

The issue asserts that only `braintrace-learning-map.svg` is referenced. That is
true of the *docs build*, but `braintrace.png` has a load-bearing reference
outside the repository.

Between commit `aa11f2c` (2025-12-02, project rename to BrainTrace) and
`9e8c376` (2026-05-24, "serve logo from brainx.chaobrain.com webp instead of
github raw"), `README.md` embedded the header image as
`https://raw.githubusercontent.com/chaobrain/braintrace/main/docs/_static/braintrace.png`.
Releases **0.1.1** (2025-12-02) and **0.1.2** (2025-12-25) were published inside
that window, and a PyPI project page renders the README as it was at release
time — it is immutable. Confirmed directly against the API:

```console
$ curl -s https://pypi.org/pypi/braintrace/0.1.2/json | jq -r .info.description | grep raw.githubusercontent
  	<img alt="Header image of braintrace." src="https://raw.githubusercontent.com/chaobrain/braintrace/main/docs/_static/braintrace.png" width=40%>
```

Deleting `docs/_static/braintrace.png` from `main` would break the header image
on those two PyPI pages permanently. It is the single largest file in the
directory and the most tempting deletion, and it is exactly the one that must be
kept. Releases 0.2.0 onward use the `brainx.chaobrain.com` URL and do not depend
on the repository.

The same audit was run against the predecessor `brainscale` distribution; its
frozen README references only `docs/_static/brainscale.png`, a file removed long
ago, and none of the seven deleted names.

## `.gitignore` reconciliation

`.gitignore` previously listed three specific editable masters
(`model-dynamics-supported.pptx`, `etrace_op_functions.pptx`,
`etrace_op_functions.pdf`) scattered among unrelated entries. Replaced with one
documented rule covering the class rather than the three instances:

```gitignore
# Editable masters for docs figures (PowerPoint/PDF). ...
/docs/_static/*.pptx
/docs/_static/*.pdf
```

Verified that neither kept asset is ignored — `git check-ignore -v
docs/_static/braintrace.png docs/_static/braintrace-learning-map.svg` exits `1`
with no output, so a future `git add docs/_static` cannot silently drop them.
The glob deliberately does not cover `.png`/`.svg`/`.gif`: a rendered figure
that a page references must stay trackable.

## Verification

A real build was run in this worktree after the deletions:

```console
$ python -m sphinx -b html docs docs/_build/html -W --keep-going
...
copying images... [ 17%] _static/braintrace-learning-map.svg
copying images... [ 33%] _build/jupyter_execute/921ab9c0...png
...
build finished with problems, 4 warnings (with warnings treated as errors).
```

All 114 pages rendered. The four warnings are pre-existing and unrelated to this
change — every one is a `py:class`/`py:obj` nitpick miss on
`brainstate.nn.Embedding` from `braintrace/nn/_embedding.py`, which
`nitpick_ignore_regex` in `docs/conf.py` does not yet cover. This change touches
no Python and no docs page, so it can neither introduce nor fix them.

The build output is also independent corroboration of the reference table: the
"copying images" phase names exactly one file from `_static`,
`braintrace-learning-map.svg`. Every other image Sphinx copied is a notebook
execution artifact under `_build/jupyter_execute/`. If any deleted asset had
been referenced by a page, `-W` would have turned the missing-image warning into
an error.

Post-deletion grep for all nine original filenames and stems returns no hit from
any docs source, `README.md`, `braintrace/`, `examples/`, or `.github/`.

## Risks and non-goals

- **Not** rewriting history. The blobs stay in the object graph; only the
  working tree and future clones' checkouts shrink. Full-history clone size is
  unchanged by design — rewriting would break every existing checkout and is not
  worth ~599 KiB.
- If a future docs page wants one of the deleted figures, it is recoverable with
  `git checkout <commit>^ -- docs/_static/<name>`.
