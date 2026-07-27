# Documentation Review Follow-up

## Scope

Apply the maintainer's follow-up review to pull request 148 without changing
the approved visual-only Tutorial grouping.

## Requirements

- Keep the `braintrace-tutorial-group` metadata and the server-side grouping
  hook in `docs/conf.py`. Group titles remain non-page visual navigation.
- Delete the three repository-local documentation test modules:
  `_docs_examples_test.py`, `_docs_rendering_test.py`, and
  `_docs_structure_test.py`.
- Move the RNN and SNN online-learning notebooks from `quickstart` to
  `tutorials`.
- Move the custom ETP primitives notebook from `tutorials` to `advanced`.
- Update every navigation entry and internal cross-reference to the new paths.
- Do not commit BrainX-generated header or footer CSS/JavaScript assets.
- Keep the BrainX Sphinx extension as the integration point for centrally
  supplied branding.

## Verification

- A clean, strict Sphinx HTML build completes with `-n -W`.
- No documentation source references the three former notebook paths.
- The rendered sidebar keeps the four visual Tutorial groups.
- The rendered Advanced navigation contains Custom ETP Primitives.
- No BrainX-generated CSS or JavaScript file is tracked or left for commit.
- `git diff --check` reports no patch errors.
