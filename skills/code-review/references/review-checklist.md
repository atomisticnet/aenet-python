# Code Review Checklist

Apply relevant items to the actual review scope. The checklist guides judgment;
it is not a requirement to manufacture findings in every category.

## Scope and issue compliance

- The review boundary includes every requested commit and refinement.
- The governing issue and applicable local issues have been read.
- Each acceptance criterion is implemented and supported by evidence.
- The implementation does not silently narrow, reinterpret, or exceed the
  agreed behavior.
- Unrelated changes, speculative features, and obsolete code are identified.
- Issue status and completion evidence match the actual state of the work.

## Correctness and compatibility

- Normal, boundary, degenerate, and failure behavior match the contract.
- Inputs are validated only as strictly as the supported behavior requires.
- Error handling exposes real failures rather than masking them with fallbacks.
- Determinism, random-state behavior, ordering, units, shapes, and numerical
  tolerances are correct where relevant.
- Public signatures, exports, defaults, return types, exceptions, CLI options,
  configuration, and stored formats remain compatible or change deliberately.
- Optional dependencies remain optional outside their documented capability.
- Platform, filesystem, concurrency, and resource assumptions are reasonable.
- Expensive scans, copies, repeated work, or poor scaling are justified by the
  expected data size and use case.

## Simplicity and maintainability

- Every abstraction, helper, option, and branch serves a current requirement
  or demonstrated supported use case.
- A shorter direct implementation would not be equally clear and correct.
- Defensive code handles realistic failures rather than hypothetical inputs
  outside the contract.
- Functions have coherent responsibilities and locally understandable control
  flow; length is judged by complexity rather than a fixed line count.
- Names use established project and domain terminology and reveal intent.
- Duplication, dead code, stale compatibility paths, and premature extension
  points have been removed.
- Comments explain non-obvious rationale rather than restating the code.

## Test sufficiency and integrity

- New or changed behavior has focused unit or integration coverage at the
  appropriate boundary.
- Regression tests reproduce the original failure through the real supported
  path and would fail without the production fix.
- Assertions establish meaningful outcomes rather than only execution success,
  object existence, or self-consistency with the implementation.
- Edge cases and failures that materially affect users are covered.
- Tests do not duplicate the production algorithm and compare it with itself.
- Mocks and monkeypatches isolate true external boundaries without replacing
  the behavior being tested.
- Fixtures and synthetic inputs preserve the conditions needed to expose the
  bug; they do not sanitize the problematic case away.
- Skips, expected failures, exception swallowing, fallbacks, conditional
  assertions, broadened tolerances, and retries are narrowly justified and do
  not hide defects or flaky behavior.
- Production code has not gained test-only branches or behavior.
- Tests are deterministic, independent, and leave no unintended artifacts.

## Dependencies

- The requirement cannot be met reasonably with the standard library or an
  existing dependency.
- The new package dramatically simplifies the implementation or materially
  improves measured performance for the required use case.
- The value assessment accounts for implementation code removed as well as
  integration, adaptation, and error-handling code added.
- Performance claims have representative evidence when performance is the
  justification.
- The package is stable, maintained, appropriately licensed, dependable, and
  easy to install on supported platforms.
- Binary wheels, compiled components, system libraries, and transitive
  dependencies do not create disproportionate installation risk.
- Version bounds, security exposure, release cadence, and compatibility with
  supported Python versions are acceptable.
- A narrowly used dependency is optional and lazily imported where practical.
- Packaging metadata, installation guidance, CI coverage, and missing-package
  errors are complete and consistent.

## Documentation and repository hygiene

- Changed public objects have complete, accurate NumPy-style docstrings.
- Relevant Sphinx pages, examples, notebooks, and indexes reflect changed
  behavior and dependencies.
- Runnable examples and notebooks use supported public paths and are tested.
- Scientific, performance, and uncertainty claims are appropriately bounded.
- Tracked files contain no user-specific paths, credentials, stale outputs,
  generated caches, accidental binaries, or duplicated change records.
- Removed or renamed functionality has no stale imports, links, or references
  that imply it remains available.

## Validation and report

- Relevant pytest and Ruff checks were run without altering the reviewed code.
- Sphinx and notebook checks were run when their maintained surfaces changed.
- Validation gaps and environmental failures are reported precisely.
- Findings are actionable, prioritized, and tied to tight file locations.
- Each finding explains a concrete scenario and impact rather than a personal
  preference.
- Questions, assumptions, and residual risks are separated from findings.
- A clean review explicitly states that no actionable findings were found.
