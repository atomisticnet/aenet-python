---
name: issue-workflow
description: Plan, implement, review, validate, and close work tracked by aenet-python global or local issues. Use whenever a request asks to address, continue, fix, or close an issue; do not use for untracked exploratory questions or read-only issue triage.
---

# Issue Workflow

Own the requested issue from current evidence through a review-ready commit
handoff. Follow `AGENTS.md`; it remains authoritative for repository scope,
issue storage, approval, closure, and commit permissions.

Before planning or implementing, read and apply
[the shared engineering standards](../references/engineering-standards.md).

## Establish the issue contract

Read the complete governing issue, related global and local issues, relevant
development notes, neighboring implementation, tests, public call sites, and
documentation. Confirm that the issue still matches current behavior. If it
does not, propose a precise issue update rather than silently implementing a
stale contract.

Identify acceptance criteria, dependencies, compatibility constraints,
optional backends, persisted formats, documentation surfaces, and explicit
out-of-scope work. Treat acceptance criteria as necessary but not sufficient:
the result must also remain correct and maintainable in its surrounding code.

## Size the work

Treat one local issue as one coherent unit that can normally be reviewed and
committed together. Split work into ordered local subissues when the issue:

- spans multiple independently reviewable architectural layers;
- combines infrastructure, migration, documentation, and integration work;
- needs intermediate contracts before downstream work can be planned safely;
- is unlikely to fit a typical focused commit; or
- has validation or uncertainty that should be resolved separately.

Global issues commonly need local subissues, but local issues may also have
children. Give every child its own problem statement, acceptance criteria, and
dependency relationship. Keep the parent open until all children are closed
and the combined result has been checked against the parent's criteria. Do not
create subissues merely to mirror implementation steps.

## Plan and obtain approval

Always begin issue implementation with a planning stage. Do not modify code,
tests, documentation, or issue status during that stage. A plan should state:

- the behavioral outcome and boundaries;
- the intended implementation units and their order;
- tests to write first or regressions to reproduce;
- API, compatibility, dependency, persistence, and migration decisions;
- documentation or notebook changes; and
- the validation needed before closure.

Present the plan to the user and obtain explicit approval before implementation.
Approval of a parent plan also approves a child sequence only when that plan
describes each child's implementation and validation in enough detail to make
the approval informed. Otherwise, present and obtain approval for each child
plan. Ask again when a later discovery materially changes scope, behavior,
risk, or public contracts. A direct instruction to proceed with an already
proposed plan counts as approval.

## Implement one logical unit

Use test-driven development: establish a failing test or other reproducible
check before changing behavior when practical. Make the smallest coherent
change that satisfies the approved contract. Update tests, complete public
docstrings, Sphinx documentation, examples, issue text, and persisted-format
documentation in the same unit when their contracts change.

Use the repository `documentation` skill for substantive Sphinx or public API
documentation and `notebook-authoring` for maintained notebooks. Do not infer
permission for adjacent cleanup, dependency additions, migrations, or external
actions. Pause for the user when a necessary decision would materially alter
the approved plan.

## Apply the pre-commit review gate

Before declaring the unit ready to commit, review the complete diff as a
maintainer. Read and apply the
[code-review checklist](../code-review/references/review-checklist.md),
including issue compliance, correctness, compatibility, simplicity, test
integrity, dependencies, documentation, and repository hygiene. A passing test
suite is evidence, not a substitute for this review.

Resolve every known P0--P2 finding within the approved scope. Resolve P3
findings when proportionate or record them as explicit follow-up work. If a
finding requires a material scope expansion, create or propose an issue and
request direction. Report unresolved limitations honestly; do not claim that
future reviewers are guaranteed to agree.

## Validate and close

Run focused tests first, then broaden according to regression risk. Run focused
Ruff checks and any required Sphinx, notebook, optional-dependency, format,
platform, or integration validation. Re-read every acceptance criterion
against the final implementation and actual evidence.

Close a local issue after validation, immediately before the intended commit,
when its work is review-ready. Move it to `CLOSED_LOCAL_ISSUES.md` with a
completion receipt covering implementation, validation, limitations, related
issue status, and the absence of a commit hash when applicable.

For a global issue, leave it open while any acceptance criterion or promised
deliverable remains. Before merging a completed issue branch, finalize and move
its issue file as required by `AGENTS.md`, including its completion receipt.
Closing a child issue never implies that its parent is complete.

## Hand off the commit

Summarize the outcome, validation, remaining limitations, and issue status.
Propose a focused commit message, referencing the global issue ID when
applicable. Do not commit until the user confirms.
