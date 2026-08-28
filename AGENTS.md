# Context

- This project (aenet-python) aims to provide a Python interface to the
  aenet software package for machine-learned interatomic potentials
  (MLIPs).
- It contains routines and modules for interacting with aenet binaries,
  the compiled aenet library, and for other tasks related to MLIPs, such
  as structure conversion/manipulation, parsing of output files, and data
  analysis.
- Some functionality has PyTorch implementations, but PyTorch should
  remain an optional dependency wherever practical.
- Some functionality relies on the availability of compiled aenet
  binaries or the aenet library.

# Development environment

- Use `pytest` for testing.
- Use `ruff` for linting and PEP 8 compliance.
- Use `sphinx` for documentation.
- Repository-wide defaults belong in `./config/defaults.toml`.
- User- or site-specific paths and environment settings belong in
  `./config/local.toml`, which is not tracked by Git.
- Use `./config/local.toml.example` as the template for local configuration.
- Do not hard-code user-specific paths or environment locations in this file.
- Install required development tools in the configured development environment
  as needed.

# Development workflow

- Use a test-driven development workflow.
- Always begin with a planning stage.
- Do not modify code during the planning stage.
- Review relevant global and local issues before planning implementation.
- For procedural workflows such as debugging, building, or testing, follow the
  corresponding instructions in `./skills/`, when available.
- Ask before proceeding from planning to implementation unless explicitly
  instructed otherwise.

## During implementation

- Code edits are permitted.
- Add or modify unit tests for new or changed functionality.
- Keep changes scoped to the current logical unit of work.
- Make sure all relevant tests pass.
- Run the relevant `ruff` checks.
- Ask questions when requirements are ambiguous or progress is blocked.
- Make sure docstrings in modified code sections are complete and up to date.
- Revise the Sphinx documentation when APIs or documented behavior change.

## When finalizing a task

- Run the relevant tests and validation checks.
- Summarize the work performed.
- Record any remaining limitations or follow-up work.
- Update or close related issues as appropriate.
- Propose a Git commit message. When closing a global issue, reference
  the relevant issue ID.
- Do not commit to the Git repo without confirmation from the user.

# Global issues

- Use the tracked `./ISSUES.md` file as the high-level index of active project
  issues.
- Each issue should detail an appropriate subset of: problem, impact,
  current evidence, hypotheses or proposed approach, and acceptance
  criteria
- Store substantial issue descriptions as separate files under `./issues/`.
- Assign stable integer issue IDs.
- Substantive global issues should normally be developed on separate branches.
- Commit messages should reference the corresponding issue ID when applicable,
  for example: `Fix symmetry handling (#67)`
- Before merging the issue branch, finalize the issue file and move it
  to `closed-issues/[issue-ID]-[short-description].md`.  Include a
  completion receipt with the branch, relevant commits, validation, and
  resolution.  After merge, the merge date/hash need not be recorded in
  the file because Git already provides them.

# Local issues

- Use the untracked `./LOCAL_ISSUES.md` file for fine-grained technical work
  currently in progress.
- Assign integer issue IDs with leading "L" for "local" (e.g., "L12").
- Use `./LOCAL_ISSUES.md.example` as the template for this file.
- Local issues may reference global issue IDs.
- Treat each local issue as one logical unit of work; it may, but need
  not, map one-to-one to a Git commit.
- Move completed local issues to the untracked file
  `CLOSED_LOCAL_ISSUES.md` after the corresponding work has been
  committed.  This file is not archival and may occasionally be purged.

# Development notes

- Use the untracked `./dev-notes/` directory for extended development notes,
  diagnostics, experiments, and intermediate findings.
- Create this directory if it does not locally exist.
- Development notes are non-authoritative and may become stale.
- Promote durable findings into issue files, tests, documentation, or code
  comments as appropriate.
- Cross-reference relevant development notes from issue files when useful.
