# Shared Engineering Standards

Apply these standards during issue implementation and code review. They guide
judgment rather than requiring changes in every category.

## Correctness and contracts

- Implement observable behavior, boundary cases, and failure behavior defined
  by the issue and surrounding public contracts.
- Validate inputs only as strictly as supported behavior requires. Surface real
  failures rather than masking them with broad fallbacks.
- Preserve determinism, ordering, units, shapes, numerical precision, and
  ownership semantics where they matter.
- Treat API signatures, exports, defaults, return types, exceptions, CLI and
  configuration fields, stored formats, and filenames as compatibility
  surfaces. Change them deliberately and document migration consequences.
- Keep optional dependencies out of unrelated imports and workflows wherever
  practical.
- Account for expected data size, concurrency, filesystem behavior, and
  supported platforms before choosing algorithms or materialization patterns.

## Design and organization

- Prefer the smallest direct design that meets current requirements.
- Give modules, functions, classes, and helpers coherent responsibilities and
  use established project and domain terminology.
- Introduce abstractions, options, defensive branches, and extension points
  only for a current requirement or demonstrated supported use case.
- Avoid duplicated business logic, stale compatibility implementations, dead
  code, and unrelated cleanup.
- Use comments for non-obvious rationale, invariants, units, or constraints—not
  to restate syntax.
- Keep repository-wide defaults in `config/defaults.toml` and machine-specific
  settings in untracked local configuration. Never hard-code user paths.

## Tests

- Add focused unit or integration tests for changed behavior and meaningful
  failure paths. A regression test should fail without the corresponding fix.
- Exercise supported public boundaries rather than duplicating the production
  algorithm in the test.
- Make assertions establish meaningful outcomes, not only successful execution
  or object existence.
- Use mocks only at genuine external boundaries. Do not mock away the behavior
  under test or add production branches solely for tests.
- Keep tests deterministic and independent. Narrowly justify skips, retries,
  expected failures, fallbacks, conditional assertions, and relaxed tolerances.
- Leave no unintended files, processes, caches, or external state behind.

## Documentation

- Give supported public Python objects complete NumPy-style docstrings covering
  parameters, returns, exceptions, units, shapes, ownership, and consequential
  semantics.
- Update relevant Sphinx pages, examples, notebooks, indexes, and issue text
  when behavior, APIs, defaults, dependencies, or supported workflows change.
- Use supported public imports in runnable examples and test them when
  practical. Clearly label pseudocode and intentionally untested examples.
- Bound scientific, accuracy, performance, and uncertainty claims to available
  evidence.

## Dependencies

- Prefer the standard library or an existing dependency when it provides a
  clear, maintainable solution.
- Add a package only when its benefit justifies long-term integration,
  installation, security, compatibility, license, CI, and documentation costs.
- Support performance-based dependency decisions with representative evidence.
- Keep narrowly used dependencies optional and lazily imported where practical,
  with complete packaging metadata and actionable missing-package errors.

## Repository hygiene and validation

- Preserve unrelated user changes and keep each diff scoped to its logical
  issue unit.
- Do not track credentials, private data, machine-local paths, generated caches,
  stale outputs, accidental binaries, or duplicate change records.
- Run the narrowest tests and checks that establish the behavior, then broaden
  in proportion to risk. Use pytest, Ruff, Sphinx, and notebook execution as
  appropriate to the changed surfaces.
- Inspect the final diff and relevant generated or rendered artifacts. Record
  validation gaps as environmental limitations or follow-up work rather than
  treating an unrun check as passing.
