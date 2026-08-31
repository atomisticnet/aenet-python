---
name: code-review
description: Review aenet-python commits, branches, diffs, or working-tree changes for correctness, issue compliance, maintainability, test quality, documentation, dependencies, and scope. Use when asked for a code review or pre-merge assessment; do not modify the reviewed implementation unless fixes are separately requested.
---

# Code Review

Review changes as a maintainer of `aenet-python`. Follow `AGENTS.md`, including
its issue-tracking and validation conventions. A review is read-only unless the
user separately requests implementation of fixes.

## Establish the review contract

Resolve the exact review boundary: named commits, a branch relative to its
merge base, a pull-request diff, staged or unstaged changes, or another range
specified by the user. Include subsequent refinement commits when requested.
Do not silently review only the latest snapshot when commit history is relevant
to understanding the change.

Before judging the implementation, read:

- the governing global or local issue and its acceptance criteria, when one
  exists;
- relevant repository instructions and configuration;
- the complete diff plus enough neighboring code to understand its behavior;
- tests, public exports, documentation, and call sites affected by the change;
  and
- related earlier behavior when compatibility or regression risk matters.

Treat issue compliance as necessary but not sufficient. Flag both missing
requirements and implementation that exceeds the agreed scope.

## Review the implementation

Evaluate observable correctness, boundary and failure behavior, public API and
CLI compatibility, optional-dependency boundaries, portability, performance,
and repository hygiene. Look for stale or dead code, accidental generated
files, hard-coded local paths, duplicated records, secrets, and unrelated
changes.

Apply a deliberate simplicity pass. AI-generated changes often add defensive
branches, generalized helpers, configuration switches, diagnostics, or
abstractions beyond the current requirement. Ask whether each new concept is
needed by an acceptance criterion or a demonstrated supported use case. Prefer
a direct, readable implementation when it is equally correct. Do not report a
mere stylistic preference as a defect; explain the concrete maintenance,
correctness, or usability cost.

Review names, function boundaries, duplication, and control flow in context.
Flag lengthy functions when they combine distinct responsibilities or make the
behavior difficult to verify, not solely because of a line-count threshold.

## Review tests critically

New or changed behavior should normally have focused unit tests developed in a
test-driven workflow. Assess whether tests cover the public contract, relevant
edge cases, error paths, regressions, and optional integrations without
coupling unnecessarily to implementation details.

Do not equate passing tests or line coverage with correctness. Check that a
test would fail for the bug or missing behavior it claims to protect. Inspect
mocks, monkeypatches, fixtures, skips, expected failures, fallback data,
conditional assertions, and broadened tolerances for workarounds that bypass
the production path or prevent a real defect from surfacing. Tests must not
compensate for incorrect production behavior merely to make the suite pass.

Read and apply [the review checklist](references/review-checklist.md) before
finalizing findings.

## Apply a high dependency threshold

Treat every new package dependency, including optional dependencies, as a
long-term maintenance decision. First determine whether the standard library
or an existing project dependency can meet the requirement without undue
complexity.

A new dependency is normally justified only when it dramatically simplifies a
correct implementation or materially improves performance. Verify that the
benefit is relevant to the requested scope and supported by concrete evidence,
not assertion. Also assess package stability, maintenance, license suitability,
supported-platform availability, installation reliability, transitive weight,
version compatibility, security exposure, and CI/documentation cost. Prefer an
optional extra when only a narrow capability needs the package. A stable,
widely relied-on, easy-to-install package may be acceptable when the value
clearly outweighs these costs.

## Check maintained documentation

When public API, behavior, defaults, configuration, dependencies, CLI behavior,
units, shapes, exceptions, or supported workflows change, use the repository
`documentation` skill to assess docstrings, Sphinx pages, and examples. Use the
`notebook-authoring` skill when maintained notebooks are part of the change.
Check that the issue, documentation, tests, and implementation describe the
same contract.

## Validate proportionally

Run the narrowest checks that establish the reviewed behavior, then broaden
when risk warrants it. This normally includes relevant pytest tests and focused
Ruff checks, plus documentation or notebook validation when those surfaces
changed. Do not modify production code or tests to obtain a passing result.
Record commands that could not run and distinguish environmental limitations
from implementation failures.

## Report findings first

Present actionable findings before summaries. Order them by severity and give
each a precise file and line location, the failing scenario or violated
contract, and the concrete impact. Use these priorities:

- **P0:** immediate catastrophic or security-critical impact;
- **P1:** release-blocking correctness, data-loss, or major compatibility bug;
- **P2:** substantive defect, missing requirement, regression risk, or
  maintainability problem that should be fixed; and
- **P3:** worthwhile low-risk improvement with a concrete benefit.

Separate findings from open questions and assumptions. After the findings,
summarize validation performed and residual risks or coverage gaps. If there
are no actionable findings, state that explicitly; do not invent minor issues
to populate the report. Do not provide a fix plan unless requested.
