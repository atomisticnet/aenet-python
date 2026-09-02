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

# Repository skills

Use the applicable workflow under `./skills/`:

- `issue-workflow`: Plan, implement, review, validate, and close work tracked
  by global or local issues.
- `code-review`: Review commits, branches, diffs, or working-tree changes.
- `documentation`: Create or substantially revise Sphinx documentation,
  public API docstrings, and maintained documentation examples.
- `notebook-authoring`: Create or substantially revise maintained notebooks.

Read the selected skill's `SKILL.md` completely before starting its workflow.
When multiple skills apply, use `issue-workflow` as the coordinating workflow
and the specialized skill for the relevant deliverable.

# Development workflow

- Always begin development work with a planning stage. Do not modify code in
  that stage, and obtain user approval before implementation unless the user
  has explicitly approved the proposed plan.
- Use a test-driven development workflow.
- For work tracked by a global or local issue, follow
  `./skills/issue-workflow/SKILL.md`. It governs issue review, scope splitting,
  planning and approval, implementation, the pre-commit review gate,
  validation, closure, and commit handoff.
- For procedural workflows such as debugging, building, or testing, follow the
  corresponding instructions in `./skills/`, when available.

## During implementation

- Keep changes scoped to the current logical unit of work.
- Ask questions when requirements are ambiguous or progress is blocked.
- Apply `./skills/references/engineering-standards.md` for implementation,
  tests, documentation, dependencies, and repository hygiene.

## When finalizing a task

- Complete the validation and review required by the applicable workflow.
- Summarize the work and record remaining limitations or follow-up work.
- Update or close related issues only when their acceptance criteria and
  closure requirements are satisfied.
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
  `CLOSED_LOCAL_ISSUES.md` after validation, when the corresponding work is
  ready to commit. Include a completion receipt describing the implementation
  and validation. The final commit hash need not be known in advance. This
  file is not archival and may occasionally be purged.

# Development notes

- Use the untracked `./dev-notes/` directory for extended development notes,
  diagnostics, experiments, and intermediate findings.
- Create this directory if it does not locally exist.
- Development notes are non-authoritative and may become stale.
- Promote durable findings into issue files, tests, documentation, or code
  comments as appropriate.
- Cross-reference relevant development notes from issue files when useful.
