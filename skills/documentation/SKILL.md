---
name: documentation
description: Create or substantially revise aenet-python Sphinx pages, public API docstrings, and runnable documentation examples. Use for maintained documentation under docs/ and documentation affected by public API or behavior changes; use notebook-authoring instead for tutorials and long examples.
---

# Documentation

Write concise, accurate documentation for the `aenet-python` package and the
broader ænet infrastructure. Follow the repository workflow in `AGENTS.md`,
including its planning, issue-tracking, approval, testing, and finalization
requirements.

Before editing, inspect the relevant issues, neighboring documentation,
implementation, tests, and current configuration in `docs/source/conf.py` and
`pyproject.toml`. Treat those configuration files as authoritative rather than
copying their settings into documentation policy.

## Choose the maintained surface

- Use a usage page for concepts, supported workflows, configuration, and
  short task-oriented examples.
- Use an API page selectively for core public functionality whose behavioral
  contract or explanation adds value beyond inspecting the code.
- Use a developer page for architecture, algorithms, design constraints, and
  maintenance procedures.
- Use the `notebook-authoring` skill for tutorials, sequential or stateful
  workflows, data- or visualization-heavy examples, and other long examples.
- Document every public Python object with a complete NumPy-style docstring,
  including public objects that are not selected for a rendered API page.

An object intentionally exported as supported functionality is public. An
object used by a maintained tutorial notebook is also public; confirm that it
is intentionally exported. Do not expose internal or semi-public objects in
the Sphinx API reference merely because their Python names lack a leading
underscore.

For detailed writing, terminology, docstring, backend, and formatting policy,
read [the style guide](references/style-guide.md). Read the sections relevant
to the documentation being changed.

## Keep documentation useful and synchronized

Prefer a curated explanation over an exhaustive generated inventory. Use
autodoc or autosummary only where the selected public interface belongs in the
main user-facing API reference. Keep volatile facts, exhaustive option lists,
and behavioral contracts in one authoritative location; repeat brief context
only when it makes another page independently understandable without creating
a likely source of stale content.

Update documentation when a supported API, default, configuration option,
unit, shape, return value, exception, dependency, command-line interface, or
backend behavior changes. Link related usage pages, selected API entries, and
maintained notebooks where the connection is useful.

State scientifically meaningful assumptions and limitations. Qualify
performance and accuracy claims, demonstration-scale results, heuristic
uncertainty measures, and backend comparisons. Do not imply that a small
example produces a deployment-ready interatomic potential.

## Make examples maintainable

Use the smallest example that establishes the documented behavior:

- use a Sphinx doctest for a short, deterministic, self-contained snippet;
- use a normal Python code block backed by pytest when fixtures, files, or
  several setup steps would make doctest markup distracting;
- use a maintained notebook for a tutorial or long workflow; and
- mark pseudocode and intentionally partial fragments clearly as
  non-executable.

Python snippets should normally be exercised in CI. An untested snippet needs
a concrete reason, such as unavailable compiled infrastructure, an optional
dependency outside the maintained CI environment, or deliberately illustrative
pseudocode. Keep imports explicit, paths portable, randomness controlled, and
units and array shapes visible when they affect interpretation.

Mention a prerequisite at the top of a page when all documented features
require it. Otherwise, state the requirement before the first affected
section or example. Keep PyTorch optional outside PyTorch-specific
functionality, and distinguish the PyTorch backend, Fortran backend, compiled
ænet executables, and compiled ænet library when the distinction matters.

## Validate the result

Before finalizing, read and apply
[the review checklist](references/review-checklist.md). Validation should be
proportional to the change and should exercise the maintained behavior rather
than only the source text. It normally includes relevant tests, focused Ruff
checks for modified Python and docstrings, Sphinx doctest when executable
documentation changed, and a warning-clean Sphinx HTML build. Inspect affected
rendered pages for structure, code formatting, equations, tables, links, and
admonitions.

In the final handoff, summarize the documentation scope, important policy or
content choices, validation performed, intentionally untested examples or
backend limitations, remaining work, and a proposed commit message. Do not
commit without user confirmation.
