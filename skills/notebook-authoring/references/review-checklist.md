# Maintained Notebook Review Checklist

Use this checklist after the notebook content is substantially complete. Apply
items that are relevant to the notebook; do not add artificial material only
to satisfy a checklist item.

## Teaching and scope

- The title names the workflow rather than only the underlying class.
- The introduction identifies the purpose, audience, and expected outcomes.
- The notebook has one coherent primary lesson.
- Explanations precede the operations they motivate.
- Domain-specific choices are explained at the level needed by a scientific
  Python user who is new to that part of aenet-python.
- Descriptive text uses neutral, subject-centered wording and does not address
  the reader directly or promise what the reader will know.
- Demonstration-scale settings and scientific limitations are explicit.
- The conclusion states what was demonstrated and what a realistic workflow
  would change.

## Data and scientific claims

- Input data provenance and size are clear.
- File discovery is sorted and deterministic.
- Random splitting, initialization, and sampling have explicit seeds where
  reproducibility matters.
- Training, validation, and held-out test roles are not conflated.
- Reported metrics use appropriate units and normalization.
- Evaluation data supports the claims made in the narrative.
- Force-free or energy-only examples state that limitation.
- Uncertainty indicators are not described as calibrated error estimates
  without supporting calibration evidence.
- Toy results are not presented as deployable potentials.

## Notebook state and code quality

- The notebook executes from a fresh kernel in top-to-bottom order.
- Execution does not depend on variables created by omitted or reordered cells.
- There are no empty, stale debugging, or accidental scratch cells.
- Imports, variable names, and configurations are readable.
- The executable core follows one direct, human-readable path without
  unnecessary fallback branches or configuration switches.
- Data preparation, validation, and infrastructure do not dominate the
  scientific operation being taught.
- Each code cell can be understood locally without tracing substantial
  conditional state across the notebook.
- Each substantial code cell advances one conceptual step.
- Important invariants have compact checks where a silent failure would be
  misleading.
- Every substantive code cell has an appropriate stored output unless it
  intentionally produces none.
- Stored outputs come from a clean execution of the current notebook, match
  the current code and parameters, and have ordered execution counts.
- Outputs are concise, readable without re-execution, and interpreted when
  their meaning is not obvious.
- Plots have readable labels, units, legends where needed, and a stated
  takeaway.

## Paths, artifacts, and dependencies

- Paths work from both the repository root and `notebooks/`, unless the
  notebook clearly documents a narrower execution contract.
- Generated files stay in a notebook-specific output directory.
- Markdown, code, metadata, and stored outputs contain no absolute local paths,
  usernames, home directories, hostnames, local environment names, temporary
  directories, credentials, or other machine-local details.
- Displayed paths are repository-relative when paths are pedagogically useful.
- Stored outputs omit progress bars, verbose logs, and incidental timing or
  hardware-dependent details unless they are part of the lesson.
- Required optional dependencies and compiled backends are declared near the
  beginning.
- Optional sections do not create hidden prerequisites for the core workflow.
- CPU execution is available for the core of a PyTorch notebook unless the
  notebook explicitly teaches GPU-only behavior.
- Runtime and storage are reasonable for routine local execution and, when
  applicable, CI.

## Maintained surfaces

- `notebooks/README.md` lists the notebook accurately.
- Relevant Sphinx pages link to it or are updated for changed behavior.
- Related notebooks do not contain stale cross-references.
- The CI notebook matrix includes it when its dependencies and runtime fit the
  base notebook environment.
- Any intentional CI exclusion is recorded in the relevant developer
  documentation.
- The associated local or global issue reflects the implemented scope.

## Final validation

- Execute with `jupyter nbconvert --to notebook --execute` in a disposable
  repository copy or worktree, using a temporary output directory.
- Use a timeout consistent with `.github/workflows/ci.yml` when the notebook is
  CI-maintained.
- Inspect the rendered executed notebook for layout, clipped output, unreadable
  figures, warnings, and confusing transitions.
- Retain the validated and sanitized outputs in the maintained source notebook
  after disposable execution succeeds.
- Confirm that execution did not leave unintended files in the source tree.
- Confirm that notebook-generated artifact directories are not accidentally
  included in the proposed commit.
- Run relevant tests and focused Ruff checks.
- Build relevant Sphinx documentation with warnings treated as errors when
  documentation changed.
- Report validation, dependency limitations, remaining work, and a proposed
  commit message without committing.
