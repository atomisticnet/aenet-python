---
name: notebook-authoring
description: Create or substantially revise maintained Jupyter example notebooks for aenet-python, including their pedagogy, reproducibility, documentation links, and execution validation. Use for notebooks under notebooks/; do not use for exploratory or personal notebooks under notebooks-work/ unless they are being promoted into official examples.
---

# Notebook Authoring

Create maintained examples that teach one coherent aenet-python workflow and
execute reproducibly from a fresh kernel. Match the scientific-Python level of
the neighboring official notebooks without copying their incidental legacy
inconsistencies.

Follow the repository workflow in `AGENTS.md`, including its planning,
issue-tracking, testing, and approval requirements.

## Plan the teaching contract

Before editing, inspect:

- `ISSUES.md` and `LOCAL_ISSUES.md` when present;
- `notebooks/README.md` and the closest related notebooks;
- relevant pages under `docs/source/`;
- `.github/workflows/ci.yml` when the notebook may be maintained in CI; and
- the example data, models, optional dependencies, and backends needed by the
  workflow.

State in the plan:

- the intended audience and assumed background;
- one primary learning objective and a small set of observable outcomes;
- how the notebook complements rather than duplicates existing examples;
- which parts form the executable core and which are optional extensions; and
- the expected runtime, generated artifacts, and validation path.

Prefer an official notebook when the material is tutorial-shaped, file-heavy,
or benefits from inspecting intermediate results. Keep compact API usage in
the Sphinx documentation and link to the notebook for the longer workflow.

## Build a coherent narrative

Use a pedagogical arc appropriate to the topic rather than a mandatory cell
template. A complete workflow will usually include:

1. A descriptive title, purpose, and learning outcomes.
2. Prominent dependency or backend requirements.
3. Reproducible data and output-path setup.
4. Concepts and rationale immediately before substantive operations.
5. Small code cells that each advance one conceptual step.
6. Inspection or visualization of meaningful intermediate results.
7. Evaluation of the final result on data appropriate to the claim.
8. Interpretation, caveats, common mistakes, and a concise summary.

Explain aenet- and MLIP-specific decisions such as descriptors, reference
energies, data splits, force weighting, and uncertainty semantics. Assume
familiarity with ordinary scientific Python; do not narrate routine syntax.
Explain why consequential parameters are chosen. Clearly label deliberately
small datasets, networks, committees, or iteration counts used to keep the
example fast, and do not imply that the resulting potential is production
quality.

Use neutral, subject-centered explanatory prose. Do not address the reader
directly or promise what the reader will know after completing the notebook.
Prefer formulations such as "This notebook demonstrates ..." and "The
following section compares ...".

## Make execution reproducible

- Sort discovered input paths and seed stochastic initialization, splitting,
  sampling, and shuffling when they affect the lesson.
- Use explicit train, validation, and held-out test semantics. Do not evaluate
  generalization on training data or an order-dependent slice without saying
  so and justifying it.
- Make paths work when execution starts from either the repository root or the
  `notebooks/` directory, following the pattern in maintained PyTorch
  notebooks.
- Write generated files beneath a notebook-specific output directory. Do not
  use user-specific paths or depend on pre-existing hidden state.
- Keep the core workload small enough for routine execution and the CI timeout.
- Ensure a clean, top-to-bottom run from a fresh kernel. Remove empty cells,
  stale debugging cells, and dependencies on out-of-order execution.

Use imports near first use unless a small common setup cell makes the workflow
clearer. Prefer named configuration objects and readable keyword arguments.
After consequential steps, show compact diagnostics or assert important
invariants so failures are understandable.

Treat notebook code as a teaching surface rather than production
orchestration. Prefer one concrete, linear happy path that a reader can follow
top to bottom and adapt without tracing conditional state across distant
cells. Keep setup proportional to the lesson: move alternative pipelines,
defensive validation for tracked inputs, reusable infrastructure, and advanced
configuration into tested helpers, supporting documentation, or clearly
optional follow-up material. Do not add configuration switches that change the
meaning or data flow of later cells merely to make one notebook cover more use
cases.

Favor locally understandable cells over generalized abstractions. A reader
should be able to identify the scientific operation taught by a cell without
first understanding archive-management machinery, fallback branches, or
several layers of configuration. Repetition may be consolidated when the
result remains more obvious than a helper abstraction, but minimizing line
count is not itself the goal.

## Handle optional capabilities explicitly

- State PyTorch and matching PyG extension requirements prominently in
  PyTorch-specific notebooks.
- State when aenet executables or `libaenet` are required.
- Do not add a notebook to base CI if its executable core requires unavailable
  compiled aenet components, external services, GPUs, or undeclared data.
- Give optional GPU sections a CPU-compatible core path unless GPU behavior is
  itself the subject.
- Keep unavailable or expensive extensions clearly optional; they must not be
  hidden prerequisites for later cells.

Respect the project's goal of keeping PyTorch optional outside PyTorch-specific
functionality.

## Make outputs teach

Maintained notebooks should retain representative outputs from a clean,
top-to-bottom execution so they remain useful when viewed without running
them. Stored outputs must correspond to the current source cells and
parameters. Re-execute the notebook before finalizing whenever code or
consequential parameters change.

Choose each output to answer a question:

- use a small table for exact comparisons;
- use a plot for trends, distributions, errors, or relationships;
- use concise text for paths, sizes, seeds, and selected settings; and
- include units in displayed quantities, labels, and captions.

Interpret non-obvious outputs in the following markdown cell. Avoid large
object dumps, verbose progress logs, redundant displays, and plots without a
stated takeaway.

Treat stored outputs and notebook metadata as publishable content. Inspect
them as well as markdown and code for absolute local paths, usernames, home
directories, hostnames, local environment names, temporary directories,
credentials, and other machine-local details. Display repository-relative
paths when paths are pedagogically useful. Suppress progress bars, verbose
logs, and incidental timing or hardware-dependent output unless performance
is part of the lesson. Stored numerical results are illustrative and need not
be identical across supported platforms.

For training and evaluation notebooks, distinguish optimization diagnostics
from held-out predictive performance. Discuss dataset representativeness,
energy normalization, force inclusion or omission, and other limitations that
affect scientific interpretation. Describe heuristic uncertainty indicators as
such; do not present raw model spread as calibrated uncertainty without
evidence.

## Keep maintained surfaces synchronized

When the notebook is added or its scope changes, review and update as needed:

- `notebooks/README.md`;
- relevant Sphinx usage, API, or developer pages;
- cross-links from related notebooks; and
- the maintained notebook matrix in `.github/workflows/ci.yml`.

Do not add optional-backend notebooks to the base CI matrix merely for
symmetry. Record the reason when a maintained notebook cannot run there.

## Validate and review

Before finalizing, read and follow
[the review checklist](references/review-checklist.md). Execute the notebook
with `nbconvert --execute` in a disposable copy or worktree and write the
executed notebook to a temporary output directory so validation does not
contaminate the source tree. Inspect that executed copy, then retain its
validated and sanitized outputs in the maintained source notebook. Confirm
that execution-created artifact directories are not accidentally committed.

Validation should be proportional to the change and normally include relevant
tests, focused Ruff checks for modified Python code, and Sphinx validation when
documentation changes. Inspect the rendered executed notebook as well as its
exit status.

In the final handoff, summarize the teaching scope, important design choices,
validation performed, dependency or CI limitations, generated untracked
artifacts, remaining limitations, and a proposed commit message. Do not commit
without user confirmation.
