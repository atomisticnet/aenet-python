# Documentation Style Guide

Apply these conventions to maintained Sphinx pages, public API docstrings, and
documentation examples. Preserve useful local patterns unless they conflict
with this guide.

## Audience and voice

Assume familiarity with ordinary scientific Python, atomistic simulations,
and the basics of constructing and using machine-learned interatomic
potentials. Do not assume familiarity with ænet, PyTorch, or advanced machine-
learning concepts such as committees.

Use concise, neutral, subject-centered prose. Do not address the reader
directly or describe what "you" will do, see, or learn. Prefer clear active
constructions; use passive constructions when the actor is irrelevant or when
they avoid an artificial reference to the reader. Remove introductions,
transitions, summaries, and repeated explanations that do not improve
completeness or navigation.

Not every page needs an introduction or a fixed sequence of sections. Organize
the page around the information needed for its purpose and keep heading depth
shallow.

## Project and domain terminology

- Use `aenet-python` for the repository and Python package.
- Use ænet in rendered prose for the broader infrastructure, including the
  Fortran backend. Use ASCII `aenet` in code, identifiers, commands, and paths.
- Use *atomic structure* for species, coordinates, lattice information, and
  associated physical data considered as one object. *Configuration* is often
  clearer for members of a sampled collection, trajectory, or data set,
  especially when several configurations derive from the same base structure.
  Choose according to context rather than treating either term as mandatory.
- Prefer *machine-learned interatomic potential* to the acronym *MLIP*. Use
  *model* in machine-learning contexts such as training and validation. Use
  *interatomic potential* in simulation contexts; do not shorten it to
  *potential* when that would be the full technical term.
- Use *features* in machine-learning contexts and *descriptors* when physical
  interpretation is central. Use *fingerprints* only after explaining their
  relationship to features or descriptors.
- Use *reference data* for the complete data set before a training,
  validation, and test split. Name the resulting subsets explicitly.
- Prefer *energy per atom* to the less explicit *normalized energy*.

Define ænet-specific and advanced machine-learning concepts at first
substantive use. Define committee semantics, aggregation behavior, and
uncertainty limitations rather than assuming they are familiar.

## Backends and prerequisites

Document both supported backends when they are relevant. The PyTorch backend
is generally recommended for training because it exposes more training
functionality. The Fortran backend is generally recommended for efficient
inference. State their compatibility while qualifying any specific limitations
instead of promising unconditional interchangeability.

Use the terms *PyTorch backend* and *Fortran backend*. Distinguish compiled
ænet executables from the compiled ænet library when installation or runtime
behavior differs.

State a prerequisite at the top of a page when every feature on the page
requires it. Otherwise, state it before the first affected section or example.
Identify requirements for PyTorch, matching PyTorch Geometric extensions,
compiled ænet components, GPUs, or other optional packages without making
them appear to be base-package requirements.

The standard Sphinx HTML build must succeed without PyTorch, PyTorch Geometric
extensions, or other optional backends installed. Mock optional imports for
autodoc where necessary. Backend-neutral examples must not import an optional
backend. PyTorch-specific pages may show and import PyTorch, but their
executable examples belong in the optional-backend test environment rather
than the standard documentation-build environment. Adding an optional package
to the standard documentation environment requires an explicit, documented
justification.

## Public API and docstrings

Every supported public function, class, method, property, and alternative
constructor requires a complete NumPy-style docstring. Public status is an
intentional support decision, not merely the absence of a leading underscore.
Objects used in maintained tutorial notebooks are public and should have their
intentional exports confirmed.

Document construction in the class docstring rather than in `__init__`:

- summarize what an instance represents;
- document constructor arguments under `Parameters`;
- document meaningful public state under `Attributes`, without listing every
  internal field;
- use `Notes` for important invariants, lifecycle behavior, algorithms, or
  backend distinctions; and
- include `Examples` only when a short example materially clarifies use.

The `__init__` method normally has no separate docstring. Public alternative
constructors such as `from_file` require their own complete docstrings.

Do not repeat obvious annotation information solely to fill a section.
Document information that annotations cannot express adequately, including:

- accepted forms and semantic constraints;
- array shapes, axes, dtypes, devices, and units;
- return semantics and ownership;
- meaningful side effects and overwritten files;
- optional-dependency or backend requirements;
- actionable failure behavior; and
- scientifically important assumptions and limitations.

Document exceptions callers can reasonably handle. Do not enumerate incidental
implementation exceptions.

Curate rendered API pages. Include core public procedures intended for the
main user base when explanation adds value beyond source inspection. Do not
generate Sphinx pages for internal or semi-public objects, and do not treat
inclusion in autosummary or autodoc as a substitute for explanatory content.

## Deprecation

Keep deprecation guidance visible and lightweight:

- mark the interface with the standard Sphinx `deprecated` directive;
- name the supported replacement and any required migration;
- emit and test the appropriate Python deprecation warning;
- remove the deprecated interface from primary examples; and
- avoid promising a removal version unless the project has committed to it.

Do not add broad version-history sections to ordinary pages. Use version-added
or version-changed information only when it materially helps users understand
the supported interface.

## Examples and code

Follow PEP 8, the repository Ruff configuration, and the repository formatter
for Python snippets. Prefer complete, directly runnable examples with explicit
non-obvious imports, readable names, and keyword arguments that clarify
scientific meaning. Use `pathlib.Path` for portable file-path examples.

Keep each snippet focused on one behavior. Put conceptual explanation before
the code and use comments sparingly. Include output only when it establishes an
important result, interpretation, shape, unit, or failure mode.

Control random state when it affects the documented result. Use explicit
training, validation, and held-out test semantics. State when data, network
sizes, iteration counts, or other settings are deliberately reduced for the
example. Distinguish heuristic uncertainty indicators from calibrated error
estimates.

Use doctest for short, deterministic, self-contained examples. Use a pytest-
backed code block when temporary files, fixtures, optional setup, or assertions
would make doctest markup obscure the example. Use a maintained notebook for
sequential, stateful, data-heavy, visualization-heavy, or tutorial-shaped
material. A Sphinx page may contain several short task-oriented examples and
should retain enough explanation to remain useful independently of a linked
notebook.

Omit shell prompt characters so commands can be copied directly. Do not embed
user-specific paths, local environment names, hostnames, credentials, or
unexplained placeholders.

## Sphinx structure and formatting

Use sentence-style headings and keep nesting shallow. Prefer Sphinx roles and
cross-references for Python objects, commands, files, documents, and sections
over fragile hard-coded internal links. Link usage pages, selected API entries,
and maintained notebooks when the relationship helps navigation.

Use paragraphs and definition lists by default. Use a table when exact
comparison across several repeated fields is clearer. Define symbols near
equations, state conventions, and include units in prose, tables, plot labels,
and displayed results.

Use semantic admonitions sparingly: `note` for important context, `warning` for
risks or consequential traps, and `deprecated` for deprecated interfaces. Do
not use admonitions decoratively.

Allow source lines to follow the repository's existing readable style. Do not
force wrapping that damages code, tables, equations, URLs, or Sphinx markup.

## Accuracy, duplication, and sources of truth

State assumptions and limitations that affect scientific interpretation,
including periodicity, species, units, reference energies, data
representativeness, force inclusion, and backend-specific behavior. Qualify
accuracy and performance claims with the relevant workload, hardware, data,
and measurement context.

Repeat brief context when it makes a page independently understandable. Keep
volatile facts, exhaustive option lists, compatibility details, and behavioral
contracts in one authoritative location and cross-reference them elsewhere.
Do not copy configuration values from `pyproject.toml` or
`docs/source/conf.py` into policy text when those files are the source of
truth.
