# Documentation Review Checklist

Apply the relevant items after the documentation is substantially complete.
Do not add artificial material merely to satisfy the checklist.

## Scope and audience

- The content is on the appropriate usage, API, developer, or notebook
  surface.
- Long or tutorial-style material is kept in a maintained notebook, while the
  related Sphinx page remains independently useful.
- The text assumes scientific Python, atomistic-simulation, and basic machine-
  learned-interatomic-potential knowledge without assuming ænet, PyTorch, or
  advanced machine-learning knowledge.
- Prose is concise, neutral, subject-centered, and does not address the reader.
- Active and passive constructions are chosen for clarity rather than by a
  mechanical rule.

## Terminology and technical content

- Rendered prose uses `aenet-python` for the package and ænet for the broader
  infrastructure; code and identifiers use ASCII `aenet`.
- Structure/configuration, model/interatomic potential, features/descriptors,
  and reference/train/validation/test terminology fit their contexts.
- Acronyms and advanced concepts are defined when required by the intended
  audience.
- Units, array shapes, conventions, prerequisites, and backend distinctions
  are explicit where they affect use or interpretation.
- Scientific, accuracy, performance, and uncertainty claims are supported and
  appropriately qualified.
- Demonstration-scale settings are not presented as production guidance.

## Public API and docstrings

- Every changed public object has a complete, current NumPy-style docstring.
- An object used in a maintained tutorial notebook is intentionally public and
  its export has been confirmed.
- Class construction is documented on the class; alternative constructors
  have their own docstrings.
- Parameters, returns, public attributes, shapes, units, side effects,
  actionable failures, and limitations are documented where relevant.
- Internal and semi-public objects have not been added to the Sphinx API
  reference accidentally.
- Generated API content is limited to core user-facing interfaces for which it
  adds value beyond code inspection.
- Deprecations identify a supported replacement, have warning behavior, and do
  not promise an uncommitted removal version.

## Examples

- Each Python snippet is appropriately classified as doctest, pytest-backed
  code block, notebook material, or clearly marked non-executable content.
- Python snippets normally run in CI; each intentional exception has a
  concrete reason.
- Runnable snippets include necessary imports and do not rely on hidden state.
- Paths are portable, randomness is controlled where relevant, and generated
  files do not contaminate the source tree.
- Training, validation, and test roles are distinct and scientifically
  appropriate.
- Output is included only when it helps establish or interpret behavior.
- Optional dependencies and compiled components are stated before they are
  required.

## Structure and maintainability

- Headings, cross-references, roles, tables, equations, and admonitions render
  clearly.
- Internal links use stable Sphinx references where practical.
- Repeated content provides useful local context without duplicating volatile
  facts or behavioral contracts.
- Related usage pages, curated API entries, developer pages, and notebooks are
  synchronized.
- Changed defaults, configuration, units, return behavior, exceptions,
  dependencies, CLI behavior, and backend behavior are reflected where needed.
- Documentation contains no user-specific paths, local environment names,
  hostnames, credentials, stale output, or accidental placeholders.

## Validation and handoff

- Relevant pytest tests, including tests backing documentation examples, pass.
- Relevant Sphinx doctests pass when executable documentation changed.
- A warning-clean Sphinx HTML build succeeds for substantive documentation
  changes.
- Focused Ruff checks pass for modified Python and docstrings.
- Affected rendered pages have been inspected for layout, navigation, code,
  equations, tables, output, and warnings.
- Validation is proportional to the change; any omitted check and its reason
  are recorded.
- The associated local or global issue reflects the work when applicable.
- The final handoff records scope, validation, intentional limitations,
  remaining work, and a proposed commit message without committing.
