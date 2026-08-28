# Issue 35: [Feature] Implement reference-compound-based atomic reference solving

**Priority**: Medium
**Status**: Completed
**Created**: 2026-04-13

## Problem

Users also need a direct way to derive atomic reference energies from chosen
reference compounds. This is a different workflow from regression: the helper
should identify the relevant reference compositions in the supplied
structures, choose appropriate energies for them, and solve the resulting
linear equation system.

This workflow can also be underdetermined when the chosen reference
compositions do not span the full species space unless one or more species
energies are fixed by the user.

## Proposed Solution

Implement `ReferenceEnergies.from_reference_compounds(...)` in
`aenet.reference_energies`.

Initial API requirements:

- accept lightweight ``(composition, energy)`` samples as input
- accept a user-specified list of reference compositions or compounds
- accept `fixed_atomic_energies={...}` constraints
- expose the resolved mapping via `.atomic_energies`
- expose provenance/diagnostic metadata describing which samples were used

Suggested API shape:

- `ReferenceEnergies.from_reference_compounds(samples,
  reference_compounds=[...], fixed_atomic_energies=None, ...)`

## Resolution

Implemented on 2026-04-13.

- Added `ReferenceEnergies.from_reference_compounds(...)` to
  `aenet.reference_energies`, keeping the public input model aligned with the
  regression helper by consuming lightweight ``(composition, energy)``
  samples.
- Added internal normalization for requested reference compounds so callers
  can provide either simple formula strings such as ``"TiO2"`` or explicit
  composition mappings.
- Implemented deterministic lowest-energy selection per requested reference
  composition, using a single streaming pass over the sample iterable and
  retaining the minimum-energy match for each requested compound.
- Factored the constrained linear solve into a shared internal helper so
  regression and reference-compound workflows use the same
  `fixed_atomic_energies` semantics and the same rank-deficiency checks.
- Added metadata describing the requested reference compounds, how many
  candidate samples were found for each one, and which selected
  composition-energy samples entered the solve.
- Added focused tests covering exact solves, lowest-energy selection,
  constrained underdetermined systems, missing reference compounds, and
  duplicate/equivalent compound requests.
- Updated the Sphinx API and training docs to describe the new factory and
  the reference-compound selection semantics.

# Issue 34: [Feature] Implement regression-based atomic reference estimation

**Priority**: Medium
**Status**: Completed
**Created**: 2026-04-13

## Problem

Large or chemically focused training sets often need a reproducible way to
estimate mean atomic reference energies from total-energy data. Users
currently had to perform this regression outside the API, and the
underdetermined cases were easy to mishandle.

Examples include composition spaces such as `A(x)B(1-x)C2`, where the
concentration of one species does not vary. In those cases, the regression is
underdetermined unless the user fixes one or more species energies.

## Proposed Solution

Implement `ReferenceEnergies.from_regression(...)` in
`aenet.reference_energies`.

Initial API requirements:

- accept lightweight ``(composition, energy)`` samples as input
- accept `fixed_atomic_energies={...}` constraints
- support estimation on the full input set or on a random subset
- expose the resolved mapping via `.atomic_energies`
- expose useful fit diagnostics and provenance metadata

Suggested API shape:

- `ReferenceEnergies.from_regression(samples, fixed_atomic_energies=None,
  subset_size=None, subset_fraction=None, random_seed=None, ...)`

## Resolution

Implemented on 2026-04-13.

- Added the top-level module `aenet.reference_energies` with the
  `ReferenceEnergies` value object and a first implementation of
  `ReferenceEnergies.from_regression(...)`.
- Centered the regression workflow on lightweight ``(composition, energy)``
  samples so callers can stream data without materializing full structure
  objects in memory.
- Added `iter_composition_energy_samples_from_files(...)` as a lazy adapter
  that uses `aenet.io.structure.read` and yields one sample per structure
  frame.
- Added constrained regression support via `fixed_atomic_energies` so
  underdetermined composition spaces can be solved explicitly when users pin
  one or more species energies.
- Added deterministic random-subset support through `subset_size`,
  `subset_fraction`, and `random_seed`, with reservoir sampling for
  streaming-friendly `subset_size` selection on lazy iterators.
- Added metadata/diagnostics on the resulting object, including sample counts,
  fixed/free species, solver rank, residual statistics, and subset settings.
- Added focused unit tests covering exact recovery, constrained
  underdetermined systems, deterministic subset selection, invalid subset
  arguments, lazy sample iterators, and file-backed sample extraction.
- Added a Sphinx API page plus a usage note pointing users to
  `ReferenceEnergies` for programmatic `atomic_energies` construction.

# Issue 33: [Feature] Add committee training support to the PyTorch backend

**Priority**: Medium
**Status**: Completed
**Created**: 2026-04-08

## Problem

Issue 32 added committee inference and uncertainty estimation for the
Fortran-backed inference interfaces, but the PyTorch backend still has no
first-class way to produce those independently trained committee members.

Today, users can train multiple PyTorch models manually, but that leaves
several workflow gaps:

- no dedicated API for training a committee of models with shared
  hyperparameters
- no explicit seed/split controls to make committee diversity and
  reproducibility well defined
- no built-in orchestration for running multiple members in parallel
- no standardized output layout or metadata for later committee inference

This is especially relevant for active-learning workflows, where committee
uncertainty is only as usable as the training path that produces the member
models.

## Proposed Solution

Add first-class committee training support to the PyTorch backend, built on
top of the existing single-model `TorchANNPotential` trainer rather than
replacing it.

The first design and implementation pass should:

- add explicit seed control for single-model PyTorch training, including
  separate controls for data splitting and per-run stochastic behavior
- introduce a committee-level training API that launches multiple
  independently initialized PyTorch members with shared architecture and
  training settings
- support bounded parallel execution for committee members, with explicit
  device assignment semantics for CPU and CUDA workflows
- define a stable per-member output layout for saved models, checkpoints, and
  summary metadata
- provide a committee-level inference path or compatibility layer that can
  aggregate member predictions using the existing ensemble uncertainty
  semantics
- include a committee-level convenience function to export all trained members
  to `.nn.ascii` format together, so PyTorch-trained ensembles can be used
  directly with the existing Fortran-backed committee inference interfaces
- document the workflow and add tests covering reproducibility, parallel
  orchestration, and committee inference behavior

## Resolution

Implemented on 2026-04-09.

- Added `TorchCommitteeConfig`, `TorchCommitteePotential`, and structured
  committee result objects for orchestrating seeded PyTorch committee runs.
- Added explicit `seed` and `split_seed` controls to
  `TorchTrainingConfig`, with deterministic trainer-owned split, shuffle,
  sampling, and force-resampling behavior.
- Added stable committee output metadata plus per-member directories for saved
  models, histories, summaries, and optional checkpoints.
- Added committee-side reload, aggregated prediction, and
  `.nn.ascii` export support so PyTorch-trained committees interoperate with
  the existing ensemble inference interfaces from Issue 32.
- Added tests covering reproducibility, sequential and parallel committee
  orchestration, committee inference/export, and docs-backed examples.
- Updated the PyTorch training and inference documentation and refreshed
  `notebooks/example-05-torch-training.ipynb` with the committee workflow.

# Issue 32: [Feature] Add committee inference and uncertainty estimates to the Fortran-backed interfaces

**Priority**: Medium
**Status**: Completed
**Created**: 2026-04-08

## Problem

Active-learning workflows need Fortran-backed inference that reports not only
energies and forces, but also committee-based uncertainty estimates derived
from multiple independently trained potentials.

The repository already had a rudimentary Python-side post-processing path via
`aenet.io.predict.PredictOutAnalyzer`, but that left the fast direct
`libaenet` interfaces without a convenient committee API. In practice, this
was especially limiting for ASE-driven simulations, where the
`AenetCalculator` path is materially more efficient than subprocess-based
prediction because it reuses ASE's neighbor list.

## Proposed Solution

Implement committee inference at the interface layer, without extending the
upstream Fortran/C API:

- add a direct `libaenet` ensemble interface for `AtomicStructure` objects
- add an ASE ensemble calculator that keeps ASE's efficient neighbor-list path
- make ensemble-mean energy and forces the default reported output
- support an optional reference-member reporting mode for continuity with an
  existing deployed model
- expose committee statistics including energy spread, force-component spread,
  and per-atom force uncertainty
- update the Fortran-backed docs and notebook example accordingly

## Resolution

Implemented on 2026-04-08.

- Added `AenetEnsembleInterface` for committee inference on top of the direct
  `libaenet` path.
- Added `AenetEnsembleCalculator` for committee inference on top of ASE's
  neighbor-list-backed calculator path.
- Added a shared `AenetEnsembleResult` container for aggregated committee
  outputs and uncertainty metrics.
- Refactored prepared-structure evaluation so the interface and calculator
  backends can reuse neighbor-list preparation across ensemble members.
- Added tests covering mean aggregation, reference-member aggregation, and
  zero-uncertainty duplicate-member committees.
- Updated the Fortran-backed inference docs and refreshed
  `notebooks/example-08-libaenet-interface.ipynb` with an ensemble example.

# Issue 1: [Docs] Implement Automated Documentation Testing

**Priority**: Medium
**Status**: Completed (Base Python/PyTorch docs-testing baseline landed; Fortran and other optional paths deferred)
**Created**: 2025-11-24

## Problem

Currently, all code examples in the Sphinx documentation are manually verified. This creates several risks:

1. **Code rot**: Examples may become outdated as the API evolves
2. **Copy-paste errors**: Manual verification may miss typos or syntax errors
3. **Platform differences**: Examples may work on one system but fail on others
4. **Time intensive**: Manual verification is slow and error-prone

## Proposed Solution

Implement automated testing for supported runnable examples across the
documentation and example notebooks using one or more of these methods:

### Option 1: Sphinx Doctest
Enable `sphinx.ext.doctest` extension and mark code blocks with `.. doctest::` directive:

```rst
.. doctest::

   >>> from aenet.geometry.transformations import AtomDisplacementTransformation
   >>> transform = AtomDisplacementTransformation(displacement=0.05)
   >>> transform.displacement
   0.05
```

**Pros**: Built into Sphinx, simple integration
**Cons**: Requires careful setup of test fixtures, may be verbose

### Option 2: pytest-sphinx
Use `pytest-sphinx` plugin to automatically discover and test code examples.

**Pros**: Integrates with existing pytest suite
**Cons**: Additional dependency

### Option 3: Custom Test Script
Write a custom script to extract code blocks and execute them.

**Pros**: Full control over test environment
**Cons**: Maintenance overhead

## Scope

Supported runnable examples should be tested:
- API reference examples (already in docstrings)
- Usage guides (`docs/source/usage/`)
- Developer guides (`docs/source/dev/`)
- Example notebooks in `notebooks/`

Narrative examples that intentionally use placeholders, external user data,
or environment-specific tooling do not need to be executed verbatim, but the
same behavior should be covered elsewhere when practical.

For this issue, "supported runnable examples" means the current base CI subset:
CPU-only and PyTorch-backed docs pages plus maintained notebook-first examples
that do not require external ænet / `libaenet` executables or other
environment-specific tooling. Fortran-backed workflows, `pymatgen`-specific
examples, and other conditional paths remain follow-on work.

## Implementation Steps

1. Choose testing method (recommend Option 1 or 2)
2. Update `docs/source/conf.py` to enable doctest
3. Keep short API-adjacent snippets in `.rst` and migrate longer workflows to
   notebooks when that improves maintainability
4. Add doctest markers and pytest-backed example tests for retained docs
   snippets
5. Add notebook execution checks for supported notebooks
6. Create shared fixtures/helpers for common scenarios
7. Integrate into CI/CD pipeline
8. Document testing process in developer guide

## Detailed Rollout Plan

A phased implementation plan has been recorded in
`dev-notes/DOCS_SNIPPET_TESTING_PLAN.md`.

Summary of the recommended rollout:

1. Establish docs-testing policy and inventory current examples
2. Add minimal doctest/pytest infrastructure
3. Pilot on `docs/source/usage/transformations_basic.rst`
4. Add shared helpers and fixtures
5. Expand to CPU-only PyTorch docs pages
6. Decide which workflow examples should live in notebooks instead of `.rst`
7. Integrate docs and notebook execution into CI and add contributor guidance

## Progress Update

Completed so far:

- [x] Established docs-testing policy and page inventory in the dev notes
- [x] Enabled `sphinx.ext.doctest` in `docs/source/conf.py`
- [x] Added a pytest-backed docs example pilot for
  `docs/source/usage/transformations_basic.rst`
- [x] Converted the shortest `transformations_basic` examples to inline,
  runnable doctests
- [x] Added a second mixed doctest + pytest page-level pilot for
  `docs/source/dev/neighbor_lists.rst`
- [x] Kept the real torch stack available during Sphinx doctest builds while
  preserving the lightweight mock-based HTML/autodoc path
- [x] Refreshed `notebooks/example-07-neighbor-list.ipynb` so the periodic
  examples distinguish Cartesian from fractional coordinate usage
- [x] Added a third mixed doctest + pytest page-level pilot for
  `docs/source/usage/torch_featurization.rst`
- [x] Refreshed `notebooks/example-04-torch-featurization.ipynb` so it remains
  the notebook-first home for file-based input, GPU, gradient, and extended
  batch-featurization workflows
- [x] Added a fourth mixed doctest + pytest page-level pilot for
  `docs/source/usage/torch_datasets.rst`
- [x] Refreshed `notebooks/example-05-torch-training.ipynb` so it remains the
  notebook-first home for file-backed training, explicit
  `CachedStructureDataset` usage, fixed dataset splits, and
  dataset-backed prediction
- [x] Added a fifth mixed doctest + pytest page-level pilot for
  `docs/source/usage/torch_training.rst`
- [x] Trimmed `torch_training.rst` back to compact CPU-only training/config
  snippets while leaving file-backed, checkpoint, cached-dataset, and plotting
  workflows in `notebooks/example-05-torch-training.ipynb`
- [x] Added a sixth pytest-backed page-level pilot for
  `docs/source/usage/torch_inference.rst`
- [x] Trimmed `torch_inference.rst` back to compact saved-model and
  dataset-backed API snippets while leaving the longer TiO2, batch, and GPU
  walkthrough in `notebooks/example-06-torch-inference.ipynb`
- [x] Added a seventh pytest-backed page-level pilot for
  `docs/source/api/trainset.rst`
- [x] Trimmed `trainset.rst` back to compact inspection, compatibility, and
  guarded neighbor-info API snippets while leaving end-to-end featurization,
  HDF5 generation, PyTorch compatibility, and optional GPU workflows in
  `notebooks/example-01-featurization.ipynb`
- [x] Added developer guidance for running and writing docs example tests
- [x] Added an initial GitHub Actions CI workflow for pull requests and pushes
- [x] Added a dedicated docs job covering `pytest -m docs_examples`, Sphinx
  doctest, and warning-clean Sphinx HTML builds
- [x] Added a notebook execution matrix for the maintained notebook-first
  examples (`example-04`, `05`, and `07`) without mutating tracked notebook
  files
- [x] Added a general unit-test CI job covering the broader pytest suite
  outside the docs-example marker
- [x] Fixed an unstable cache-invalidation neighbor-list test so the first
  general pytest CI baseline is green
- [x] Re-ran `notebooks/example-05-torch-training.ipynb` in the
  `aenet-torch` environment after shifting more workflow ownership to the
  notebook

Follow-on backlog after the initial baseline:

- [x] Decide whether four completed pages justify shared helpers for future
  docs-example pages
- [x] Expand coverage to the next CPU-only PyTorch page:
  `docs/source/usage/torch_training.rst`
- [x] Expand coverage to the next PyTorch page:
  `docs/source/usage/torch_inference.rst`
- [x] Expand coverage to the next API page:
  `docs/source/api/trainset.rst`
- [x] Add CI jobs for docs and notebook example execution
- [ ] Continue trimming notebook-shaped PyTorch workflows out of long `.rst`
  pages when maintained notebooks already exist
- [x] Decide whether any completed docs-example pages are ready to run in the
  first base Python CI docs job before adding Bucket C coverage
- [ ] Decide how to extend CI beyond the first Python 3.11 baseline
  (additional Python versions, lint ratchets, and conditional Bucket C jobs)

## Acceptance Criteria

- [x] All supported runnable examples in docs and notebooks are automatically tested
- [x] Tests run in CI/CD on each pull request
- [x] Test failures clearly indicate which example broke
- [x] Documentation includes guidance on writing testable examples
- [x] Test environment handles the supported optional torch / PyG stack and
  documents deferred optional paths (`pymatgen`, Fortran / `libaenet`)

## Notes

**Optional Dependencies**: The current base docs-testing environment installs
the supported torch / PyG stack (`torch`, `torch-scatter`,
`torch-cluster`). Pages or tests that need those dependencies can run in base
CI, and local pytest-backed docs tests may skip gracefully when the matching
PyG extension is unavailable.

`pymatgen`-specific examples remain optional for now and are not part of the
current base CI docs-example slice. Fortran-backed workflows and notebooks that
depend on external ænet / `libaenet` tooling also remain outside the base CI
baseline and should be revisited as conditional coverage in a later phase.

**Docs vs. notebooks**: Long workflow examples, file-heavy tutorials, and
multi-step examples may be better maintained in `notebooks/`, with the `.rst`
docs keeping only compact API-adjacent snippets plus links to the relevant
notebook. After the `torch_inference.rst` pilot, page-local setup is still
cheaper than a shared docs-example helper layer because each completed page has
distinct fixture needs. After the `trainset.rst` pilot, `example-01` should
remain the notebook-first home for end-to-end featurization and HDF5
generation workflows rather than duplicating them on the API page.

**CI strategy**: The first CI baseline now uses three separate jobs on pull
requests and pushes:
- a general pytest job for the wider unit-test suite
- a docs job for `pytest -m docs_examples`, Sphinx doctest, and warning-clean
  HTML builds
- a notebook matrix for the supported pure-Python / PyTorch notebook-first
  examples (`example-04`, `05`, and `07`)

The GitHub Actions environment now pins `torch==2.9.0` and installs
`torch-scatter` / `torch-cluster` from the matching CPU wheel index at
`https://data.pyg.org/whl/torch-2.9.0+cpu.html`. This is required because the
project's `torch` extra provides core `torch`, while PyG-backed featurization
and neighbor-list features need the additional PyG extension packages.

The base CI environment does not provide the external ænet / `libaenet`
installation needed by `src/aenet/mlip/tests/` or by the Fortran-backed
sections of `notebooks/example-01-featurization.ipynb` and
`notebooks/example-06-torch-inference.ipynb`, so those are currently excluded
from the GitHub Actions workflow.

This keeps failures localized and preserves a fast feedback path for docs-page
regressions without combining notebook execution into the base docs job.

**Lint strategy**: Repo-wide `ruff check .` is not yet a suitable required CI
gate because the repository still has a large backlog of legacy lint
violations. Lint should be phased in later via narrower directory targets or a
ratcheting cleanup plan rather than blocking the initial test/docs CI path.

**Performance**: Documentation testing should complete in < 5 minutes to avoid slowing down CI/CD.

## References

- Sphinx doctest: https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html
- pytest-sphinx: https://github.com/thisch/pytest-sphinx
- NumPy doctest guide: https://numpydoc.readthedocs.io/en/latest/format.html#doctests

# Issue 2: [Feature] Structure read functions should accept Path objects

**Priority**: Medium
**Status**: Completed
**Created**: 2026-03-18

## Problem

API constraint: `aenet.io.structure.read()` wants string paths, not Path objects.

## Propoposed Solution

Extend the API to accept string paths and Path objects.

## Resolution

Implemented on 2026-03-19.

- `aenet.io.structure.read()` and `read_safely()` now accept `str` and
  `Path`/`PathLike` inputs.
- The same normalization was applied to the write path so
  `AtomicStructure.to_file()` and `aenet.io.structure.write()` also accept
  `Path` objects when format inference is needed.
- Added regression tests covering `read()`, `read_safely()`,
  `AtomicStructure.from_file()`, and `AtomicStructure.to_file()` with
  `pathlib.Path` inputs.

# Issue 3: [Feature] Add Fortran-style training data sampling policies to PyTorch backend

**Priority**: Medium
**Status**: Completed
**Created**: 2026-03-19

## Problem

The PyTorch training backend used standard PyTorch batching and shuffling
behavior plus the existing train/test split and optional force-set
resampling, but it did not support the Fortran backend's notion of a
training schedule with non-uniform structure sampling.

That prevented reproduction of established aenet workflows based on:

- current training error
- low energy / energetic proximity to the ground state
- other structure-level priorities

It also created terminology confusion because "scheduler" in the PyTorch
backend already referred to learning-rate scheduling, not data-sampling
policy.

## Resolution

Implemented on 2026-03-28.

- Added an explicit trainer-side `sampling_policy` API to
  `TorchTrainingConfig`, separate from both the learning-rate scheduler and
  `force_sampling`.
- Implemented `uniform`, `energy_weighted`, and `error_weighted`
  structure-sampling policies in the PyTorch trainer.
- Defined `energy_weighted` sampling in terms of referenced cohesive or
  formation energy per atom rather than raw total energy.
- Preserved training with missing `atomic_energies` by warning once and
  falling back to all-zero atomic references, so externally referenced
  energy labels continue to work.
- Implemented adaptive `error_weighted` sampling with uniform bootstrap in
  epoch 0 and epoch-to-epoch weight updates from observed per-structure
  training loss.
- Kept validation sampling uniform and deterministic.
- Clarified the epoch semantics for all policies: `uniform` samples without
  replacement, while non-uniform policies sample with replacement for
  `len(train_split)` draws per epoch.
- Updated the torch-training documentation so the behavior of all
  non-uniform policies is explicit, including replacement semantics, epoch
  meaning, interaction with `force_sampling`, and current resume behavior.

## Validation

- `/Users/aurban/.local/bin/micromamba run -n aenet-torch pytest src/aenet/torch_training/tests/test_config.py src/aenet/torch_training/tests/test_trainer_smoke.py src/aenet/torch_training/tests/test_force_training.py src/aenet/torch_training/tests/test_checkpoint_resume.py src/aenet/torch_training/tests/test_hdf5_dataset.py src/aenet/torch_training/tests/test_docs_torch_training.py`
  (`125 passed`)

# Issue 4: [Performance] Replace dense force contraction with sparse local derivative contraction

**Priority**: High
**Status**: Completed (Closed; sparse runtime path validated with real TiO2 fixture forces and benchmark artifact)
**Created**: 2026-03-24

## Problem

The current PyTorch force-training path caches neighbor lists and optional
CSR/triplet metadata, but it still materializes dense feature gradients of
shape `(N_force_atoms, N_features, N_force_atoms, 3)` before contracting them
to forces. This scales poorly in both memory and runtime and is the main
bottleneck in mixed energy/force training.

The legacy `aenet-pytorch` implementation (in `./external/BAK-aenet-pytorch/`)
avoided this by working with local descriptor derivatives for each center atom
and its neighbors, then combining those contributions into final forces via
precomputed index mappings.

## Proposed Solution

Refactor the PyTorch force-loss path to operate on sparse local derivative
blocks rather than dense all-atom gradient tensors.

The target design is:

- compute `dE/dG` from the network exactly as today
- represent descriptor derivatives as local center and neighbor blocks
- contract `dE/dG` directly with those local derivative blocks
- scatter-add the resulting contributions into force tensors without creating
  `(N, F, N, 3)` intermediates
- keep the current graph/triplet cache path as the source of local geometry
  information

This should preserve the modern PyTorch API while removing the main scaling
problem in force training.

## Implementation Tasks

- define a compact local-derivative representation for radial and angular
  descriptor terms
- add descriptor-side APIs that return local derivative blocks and the
  associated center/neighbor index mappings
- implement a sparse force-contraction kernel in the training loss path
- keep the existing dense path behind a fallback or debug path until the new
  implementation is validated
- update dataset/collate plumbing only as needed to pass local derivative
  metadata efficiently
- add regression tests that compare dense and sparse force predictions and
  losses on small systems
- add performance-oriented tests or benchmarks to confirm reduced memory and
  better runtime on force training workloads
- update notebook and Sphinx documentation where the force-training behavior
  or guidance changes

## Development Plan

The detailed phased plan is recorded in
`dev-notes/TORCH_FORCE_TRAINING_OPTIMIZATION_PLAN.md`.

Issue 4 corresponds to Workstream A in that note and should be completed
before the HDF5 derivative-caching work in Issue 5 is implemented on top.

## Acceptance Criteria

- [x] `compute_force_loss()` no longer requires dense
      `(N, F, N, 3)` feature-gradient tensors on the primary path
- [x] force losses and force predictions match the existing implementation to
      numerical tolerance on representative tests
- [x] mixed energy/force training remains compatible with cached neighbors and
      cached triplets
- [x] the new sparse path is covered by unit tests and an explicit regression
      test for the TiO2 force-training workflow
- [x] notebook and docs guidance for force training remain correct

## Completion Notes

- The TiO2 sparse force-training smoke test now consumes the real fixture
  forces from `src/aenet/tests/data/xsf-TiO2/` instead of synthetic labels.
- The repository includes a minimal benchmark script at
  `scripts/benchmark_force_loss_sparse_dense.py` for direct sparse-vs-dense
  `compute_force_loss()` timing on cached TiO2 force-training batches.
- A representative 10-structure TiO2 benchmark artifact was generated at
  `outputs/issue4_force_loss_sparse_dense_limit10.json`.
- Remaining force-path cleanup and deprecation work stays open under Issue 6
  and is not a blocker for the closed Issue 4 performance fix.

## Notes

- This issue is intentionally limited to the runtime force-training path.
  It does not require compatibility with Fortran-backed training-set files.
- If needed, the dense implementation may remain available temporarily as a
  correctness reference during rollout.
- The HDF5 derivative-caching work should reuse the derivative representation
  introduced here rather than defining a second incompatible format.
- Follow-up path simplification and deprecation work is tracked separately in
  Issue 6 so the sparse-force performance fix can be validated independently.

# Issue 5: [Performance] Add HDF5-backed precomputed descriptor derivatives for force training

**Priority**: Medium
**Status**: Completed (Closed; schema, trainer integration, validation coverage, and workflow documentation landed through Issues 7, 8, and 9)
**Created**: 2026-03-24

## Problem

Even after neighbor and triplet caching, the current PyTorch backend still
recomputes descriptor derivatives during force training. For repeated training
runs on fixed datasets, this leaves substantial performance on the table.

The remaining work is conceptually coherent, but it spans three separable
deliverables:

- HDF5 schema and persistence for the sparse local-derivative payload
- lazy loading and training-path integration
- regression testing, workflow validation, and user guidance

Treating those as separate child issues reduces rollout risk and makes it
easier to review on-disk format decisions before the trainer depends on them.

## Proposed Solution

Keep Issue 5 as the umbrella tracking issue and land the work through these
child issues:

- Issue 7: HDF5 derivative schema, metadata/versioning, and persistence
- Issue 8: HDF5 derivative loading and trainer/loss integration
- Issue 9: regression tests, equivalence validation, and documentation

The overall goal remains unchanged:

- PyTorch-native HDF5 storage for precomputed derivative blocks
- reuse with the PyTorch trainer only
- fixed-geometry datasets where descriptor derivatives are valid across epochs
- no requirement for compatibility with Fortran-generated training data

## Development Plan

The detailed phased plan is recorded in
`dev-notes/TORCH_FORCE_TRAINING_OPTIMIZATION_PLAN.md`.

Issue 5 corresponds to Workstream B in that note and depends on the derivative
representation introduced by Issue 4. The work should now proceed through
Issues 7, 8, and 9 rather than as one monolithic implementation.

## Acceptance Criteria

- [x] Issue 7 is completed: the HDF5 derivative schema, metadata contract,
      and persistence path are implemented and documented
- [x] Issue 8 is completed: persisted derivative blocks can be loaded lazily
      and used by the PyTorch training path
- [x] Issue 9 is completed: regression tests, equivalence validation, and
      documentation cover the new workflow
- [x] force training using precomputed derivatives matches the on-the-fly
      sparse path to numerical tolerance
- [x] the HDF5 path remains lazy-loading and does not require all derivative
      data to be loaded into memory at once

## Notes

- This issue is medium priority and should follow the sparse runtime
  contraction work, not precede it.
- The goal is repeated-training efficiency on fixed datasets, not a general
  cache for mutable structures or MD trajectories.
- The HDF5 schema should be designed around the PyTorch-native sparse
  contraction path, not around legacy ASCII/binary training-set compatibility.
- Persisted raw descriptor features remain out of scope for this umbrella and
  are tracked separately in Issue 10.

# Issue 6: [Refactor] Simplify PyTorch force-training paths after sparse rollout

**Priority**: Medium
**Status**: Completed (Closed; force training now defaults to graph/triplet sparse contraction with a single dense debug path)
**Created**: 2026-03-24

## Problem

After the sparse local force-contraction rollout from Issue 4, the force
training stack still contains multiple execution paths:

- graph/triplet sparse contraction (intended production path)
- graph/triplet dense contraction (validation/debug reference)
- neighbor-info dense contraction
- full recompute dense contraction

This is still manageable, but it increases maintenance cost and makes it too
easy for new force-training behavior to diverge across code paths.

## Proposed Solution

Treat the graph/triplet sparse path as the single normal production path for
force training and reduce the remaining force-path surface area.

The target end state is:

- one production runtime path: cached graph/triplet local derivatives with
  sparse contraction
- one explicit dense reference path reserved for regression testing and
  debugging
- no new parallel force-contraction variants
- clear deprecation or removal of redundant dense branches once equivalent
  behavior is proven

## Implementation Tasks

- audit the remaining force-training branches and document which callers still
  rely on each one
- decide whether neighbor-info force training should auto-build CSR/triplets or
  be deprecated in favor of the graph/triplet path
- collapse graph/triplet force training onto the sparse implementation as the
  only production path
- retain exactly one dense reference path behind explicit debug/testing control
- remove or deprecate redundant dense branches only after regression coverage
  is sufficient
- update tests and docs to reflect the simplified force-path model

## Development Plan

This cleanup should follow Issue 4 validation with the real TiO2 force-labeled
fixtures and should be coordinated with
`dev-notes/TORCH_FORCE_TRAINING_OPTIMIZATION_PLAN.md`.

## Acceptance Criteria

- [x] graph/triplet sparse contraction is the only normal production path for
      force training
- [x] exactly one dense reference path remains for debugging/regression checks,
      or the repository explicitly documents why an additional fallback remains
- [x] redundant dense branches are removed or deprecated without breaking
      supported force-training workflows
- [x] tests clearly cover the supported production path and the retained
      dense-reference path
- [x] docs and developer notes describe the simplified path model clearly

## Completion Notes

- Force-supervised `StructureDataset` and `HDF5StructureDataset` samples now
  always expose CSR graph/triplet payloads and compute force-view features via
  the graph-based descriptor path, even when `cache_force_triplets=False`.
- The training collate and training loop now require graph-backed force batches
  and no longer depend on `neighbor_info` on the normal training path.
- `compute_force_loss()` now treats the sparse graph/triplet contraction path
  as the default when graph data is provided; the dense graph path remains only
  behind `use_dense_path=True` for regression/debug use.
- Force-training tests now cover the default auto-upgraded graph path, and the
  TiO2 smoke test no longer requires an explicit triplet-cache opt-in.
- Docs and developer notes now describe `cache_force_triplets` as an optional
  cache/performance hint rather than a required switch for sparse force
  training.
- Single-structure force prediction fallback behavior remains intentionally out
  of scope for this issue; the simplified-path guarantee applies to training
  workflows.

## Notes

- This issue is a follow-up to Issue 4, not a blocker for accepting the sparse
  runtime force-contraction implementation.
- Existing force-labeled periodic TiO2 fixtures in
  `src/aenet/tests/data/xsf-TiO2/` should be used during validation before
  removing fallback paths.
- Any behavior-changing deprecations should be staged carefully so the cleanup
  remains lower risk than the original performance fix.

# Issue 7: [Performance] Define and persist the HDF5 schema for sparse local derivative caches

**Priority**: Medium
**Status**: Completed (Closed; versioned HDF5 schema, compatibility metadata, persistence helpers, and schema hardening tests landed)
**Created**: 2026-03-24

## Problem

Issue 4 established the canonical sparse local-derivative representation in
memory, but there is not yet a stable on-disk format for storing that payload
inside `HDF5StructureDataset` files. Without a documented schema and metadata
contract, later loader and trainer work would be forced to depend on an
implicit or unstable format.

## Proposed Solution

Add a versioned HDF5 schema for force-supervised local derivative payloads that
is explicitly aligned with the runtime representation from Issue 4.

The first slice should cover:

- schema versioning and descriptor-compatibility metadata
- storage layout for radial and angular local derivative blocks
- per-entry index metadata for force-supervised structures
- a preprocessing/build path that computes and persists derivative blocks
- developer and API documentation for the schema

## Implementation Tasks

- define the HDF5 node layout for radial and angular derivative payloads
- record schema version, storage dtype, and descriptor compatibility metadata
- add a build-time path to compute and persist derivatives for force-labeled
  structures
- add read helpers sufficient to validate round-tripping and metadata checks
- document the schema in code and Sphinx developer documentation

## Acceptance Criteria

- [x] HDF5 files can optionally persist sparse local derivative payloads for
      force-labeled structures
- [x] the on-disk schema is versioned and documented clearly
- [x] descriptor metadata detects incompatible cache reuse attempts clearly
- [x] round-trip tests validate the persisted payload structure and contents
- [x] the persistence layer does not yet require trainer integration to be
      considered complete

## Completion Notes

- `HDF5StructureDataset.build_database(..., persist_force_derivatives=True)`
  now writes a versioned `/force_derivatives` schema aligned with the sparse
  local derivative payload from Issue 4.
- The schema records descriptor-compatibility metadata, storage dtype, feature
  counts, and per-entry row indexing for cached force-labeled structures.
- Explicit inspection/load helpers now exist for the persisted cache so the
  format can be validated without changing the default training path.
- Developer and user documentation now describe the schema and the new
  build-time option.
- Regression coverage includes single-species round-tripping, incompatible
  descriptor rejection, multi-species/typespin payloads, float32 storage, and
  entries without force labels.

## Notes

- This issue is deliberately limited to the data format and persistence path.
- Training-path use of the persisted payload is tracked separately in Issue 8.
- Validation breadth beyond round-trip and metadata checks is tracked in
  Issue 9.

# Issue 8: [Performance] Load persisted HDF5 derivative caches in the PyTorch force-training path

**Priority**: Medium
**Status**: Completed (Closed; HDF5 samples expose persisted derivative payloads lazily, the trainer batches them, and compute_force_loss prefers them with graph fallback preserved)
**Created**: 2026-03-24

## Problem

After Issue 7 lands, HDF5 files may contain persisted derivative blocks, but
the dataset, collate, and loss code will still ignore them and continue to
recompute descriptor derivatives on the fly.

## Proposed Solution

Extend the HDF5 dataset and force-training pipeline to lazily load persisted
local derivative payloads and prefer them over on-the-fly derivative
recomputation when they are available and compatible.

## Implementation Tasks

- extend `HDF5StructureDataset` to expose persisted derivative payloads lazily
- update `_collate_fn()` to batch persisted local derivative blocks
- update `compute_force_loss()` to accept and prefer precomputed derivatives
- preserve the current on-the-fly sparse graph path as the fallback/reference
- ensure the new path coexists with existing feature and graph caching

## Acceptance Criteria

- [x] force-supervised HDF5 samples can expose persisted local derivative data
- [x] the trainer/loss path uses persisted derivatives when available
- [x] the existing sparse graph path remains available as a fallback/reference
- [x] the HDF5 integration remains lazy-loading and does not materialize the
      entire derivative cache at once

## Completion Notes

- `HDF5StructureDataset.__getitem__()` now attaches a `local_derivatives`
  payload lazily for force-supervised samples when a compatible persisted cache
  row is available.
- `_collate_fn()` now batches those sparse local derivative blocks with the
  same force-view atom ordering used for positions, forces, and graphs.
- `compute_force_loss()` now accepts precomputed force-view features and sparse
  local derivative payloads and prefers them over on-the-fly sparse derivative
  recomputation.
- The existing graph/triplet sparse path remains available and is still used as
  the fallback/reference path when persisted derivatives are not present.
- Regression coverage now includes HDF5 sample exposure, collate batching,
  force-loss equivalence for precomputed derivatives, and end-to-end HDF5
  training-path usage.

## Notes

- This issue depends on the schema and metadata contract from Issue 7.
- It should not redefine the derivative representation introduced by Issue 4.

# Issue 9: [Validation] Validate HDF5 derivative-cache equivalence and document the workflow

**Priority**: Medium
**Status**: Completed (Closed; trainer-level equivalence tests, lazy-load regression checks, lifecycle cleanup, and workflow docs landed)
**Created**: 2026-03-24

## Problem

Even with schema and integration work in place, the precomputed-derivative path
is not acceptable without explicit regression coverage, numerical-equivalence
checks, and user-facing guidance for when the additional storage cost is worth
it.

## Proposed Solution

Add the test and documentation layer needed to validate and explain the new
workflow.

## Implementation Tasks

- add HDF5 round-trip and metadata-mismatch tests beyond the schema smoke tests
- add training-equivalence tests versus the on-the-fly sparse path
- add lazy-loading/non-regression checks for existing HDF5 workflows
- update Sphinx docs and notebook guidance with storage/runtime tradeoffs

## Acceptance Criteria

- [x] training with persisted derivatives matches the on-the-fly sparse path to
      numerical tolerance
- [x] tests cover successful cache reuse and incompatible-cache failures
- [x] existing feature-only HDF5 workflows remain covered
- [x] documentation explains recommended use cases and tradeoffs clearly

## Completion Notes

- Trainer-level regression coverage now compares persisted-derivative HDF5
  training against the on-the-fly sparse path with matched splits and model
  initialization.
- HDF5-specific regressions now cover lazy per-entry derivative loading,
  initial random force-subset selection, and explicit handle lifecycle cleanup
  via ``close()`` and context-manager support.
- The trainer now treats prebuilt datasets as the source of truth for
  force-sampling and cache behavior, rejecting only conflicting non-default
  config overrides.
- The notebook and Sphinx documentation now distinguish clearly between
  persisted derivative caches and in-memory runtime caches.

## Notes

- This issue depends on Issues 7 and 8.
- It closes the validation/documentation gap for the umbrella Issue 5.

# Issue 10: [Performance] Umbrella: Persisted HDF5 storage for raw descriptor features

**Priority**: Medium
**Status**: Closed
**Created**: 2026-03-25

## Problem

The current HDF5 cache work persists sparse local derivative blocks for
force-supervised samples, but it does not persist the descriptor features
themselves. Repeated fixed-geometry training runs therefore still recompute
force-view and energy-view features even when derivative reuse is enabled.

This issue now serves as an umbrella for the follow-up work needed to decide
whether persisted features are worthwhile, define a canonical HDF5 cache
schema that accommodates files with and without persisted features, and
simplify redundant runtime paths where possible.

## Proposed Solution

Track the following subissues in sequence:

- Issue 14: design/evaluation pass for canonical persisted-feature strategy
  and potential removal of redundant runtime paths
- Issue 15: define a unified versioned HDF5 cache schema with optional
  persisted-feature sections
- Issue 16: load persisted features through ``HDF5StructureDataset`` and
  define runtime precedence with trainer-owned caches
- Issue 17: validate persisted-feature round trips, compatibility rules,
  and dtype-conversion behavior
- Issue 18: refactor duplicated dataset materialization paths after the
  design pass identifies a canonical representation
- Issue 19: document the persisted-feature workflow and the unified HDF5
  cache schema

The design/evaluation pass should answer:

- whether persisted features are worth the additional file size and schema
  complexity
- whether one raw feature representation can become canonical across
  energy-view and force-view paths
- how persisted features interact with dtype conversion, descriptor
  compatibility, and normalization
- how persisted features coexist with the current in-memory
  ``cache_features=True`` runtime cache
- which runtime code paths become redundant if persisted features are
  adopted under a unified schema

## Notes

- This issue is a follow-up to the completed HDF5 derivative-cache work and is
  intentionally out of scope for Issue 5.
- The current direction is to prefer one canonical versioned HDF5 cache schema
  with optional payload sections rather than maintaining separate unrelated
  schemas for derivatives and persisted features.

# Issue 14: [Design] Evaluate canonical persisted-feature storage and path simplification

**Priority**: Medium
**Status**: Completed (Closed; canonical raw `(N, F)` feature payload validated, repeated-run benchmark completed, and follow-up schema/runtime guidance recorded)
**Created**: 2026-03-25

## Problem

The current training stack computes raw descriptor features through multiple
partially overlapping paths:

- energy-view features via ``forward_from_positions()`` or ``forward()`` with
  cached neighbor info
- force-view features via ``forward_with_graph()``
- explicit energy-only precomputation through ``CachedStructureDataset``
- HDF5 derivative persistence that currently stores local derivatives but not
  the feature tensor returned alongside them

Before adding persisted-feature storage, we need to know whether one raw
feature representation can serve as the canonical internal payload and whether
some of these paths can be reduced or removed.

## Proposed Solution

Run a design/evaluation pass that answers:

- whether energy-view and force-view raw feature tensors are numerically
  equivalent for the supported descriptor set
- whether one feature-computation path can become canonical
- whether persisted features provide enough repeated-run speedup to justify
  added file size and schema complexity
- whether ``CachedStructureDataset`` or duplicated materialization branches can
  be simplified after the canonical representation is chosen

The expected output is a short design note with a go/no-go decision on
persisted features and explicit recommendations for follow-up issues.

## Notes

- This issue should not introduce new on-disk schema or runtime behavior yet.
- The goal is to reduce ambiguity before implementation, not to pre-commit to
  a specific storage layout.

## Completion Notes

- The design note now lives in
  `dev-notes/TORCH_PERSISTED_FEATURE_EVALUATION.md`.
- Regression coverage now validates that `forward_from_positions()`,
  `forward_with_graph()`, and
  `compute_features_and_local_derivatives_with_graph()` produce the same raw
  feature tensor for representative single-species, multi-species, and
  periodic TiO2 cases.
- The evaluation concluded that persisted raw features are worth adding:
  persisted derivatives already save substantial force-path time, and
  persisting raw features removes another significant chunk of the remaining
  runtime while adding a payload that is tiny relative to persisted sparse
  derivatives.
- The recommended canonical persisted payload is the raw, unnormalized
  descriptor `(N, F)` tensor, with runtime normalization and load-time dtype
  casting behavior preserved.

# Issue 11: [API] Support optional descriptor storage and recovery for HDF5 training datasets

**Priority**: Medium
**Status**: Completed (see notes)
**Created**: 2026-03-25

## Problem

`HDF5StructureDataset` currently requires the caller to provide a live
descriptor object even when the HDF5 file already contains a persisted
force-derivative cache that is specific to one descriptor configuration.

The file already records a descriptor-compatibility signature for cache
validation, but it does not store enough information to reconstruct the
descriptor object directly. This forces users to recreate the descriptor in
Python before loading the dataset, even though the descriptor configuration is
tiny relative to the training data and derivative payloads.

## Proposed Solution

Add an optional descriptor-storage path for HDF5 datasets and a corresponding
recovery API at load time.

The first design should cover:

- a versioned on-disk descriptor manifest stored alongside HDF5 training data
- descriptor recovery for supported descriptor classes without requiring the
  user to reconstruct them manually
- clear behavior when a caller provides both a stored descriptor and an
  explicit descriptor object
- compatibility checks between the recovered descriptor and any persisted
  derivative cache
- explicit opt-in behavior, enabled automatically when
  `persist_force_derivatives=True` or manually by the user

## Notes

- This is an API and usability issue, not just a performance issue.
- The stored descriptor manifest should be explicit enough to recreate the
  descriptor object safely rather than only validating a compatibility hash.
- This should remain separate from the follow-up work in Issue 10 unless the
  two designs naturally share a common HDF5 metadata layer.

## Completion Notes

- `HDF5StructureDataset.build_database()` now supports
  `persist_descriptor=True` and enables descriptor-manifest persistence
  automatically when `persist_force_derivatives=True`.
- HDF5 load mode can now recover supported descriptor objects from a versioned
  `/descriptor_manifest` group when the caller passes `descriptor=None`.
- Explicit descriptors are validated against the persisted manifest at load
  time, with clear errors for incompatible settings.
- The implementation reuses a shared descriptor serialization/recovery helper
  so model export and HDF5 recovery stay aligned.

# Issue 12: [API] Move runtime training-policy options back into TorchTrainingConfig

**Priority**: High
**Status**: Completed (Closed; runtime force-sampling and cache policy now lives in `TorchTrainingConfig`, with trainer-owned split-local wrappers applying that policy to passive datasets)
**Created**: 2026-03-25

## Problem

The current API lets dataset objects own several options that do not describe
dataset contents and do not belong to the persisted HDF5 file itself, such as:

- `force_fraction`
- `force_sampling`
- `cache_features`
- `cache_force_neighbors`
- `cache_force_triplets`
- `min_force_structures_per_epoch`

This is especially awkward for `HDF5StructureDataset`, where a single HDF5
file may be built once and reused for many training runs with different force
sampling and cache policies. These options are runtime training policy, not
file schema or dataset identity.

The recent cleanup made prebuilt datasets the source of truth for these
settings to reduce ambiguity in the short term, but that solidified a model
that is likely wrong at the API level.

## Proposed Solution

Refactor the PyTorch training API so runtime training-policy controls live in
`TorchTrainingConfig`, while dataset objects become passive data sources.

The intended split is:

- dataset/file owns stored structures, persisted caches, descriptor/file
  compatibility metadata, and data-access behavior
- training config owns force supervision fraction/sampling and in-memory
  runtime cache policy

Likely implementation directions:

- remove training-policy knobs from `HDF5StructureDataset.__init__()`
- decide whether the same removal should also apply to `StructureDataset`, or
  whether trainer-side dataset wrapping/views should be introduced
- have `TorchANNPotential.train()` apply runtime policy consistently for both
  in-memory and HDF5-backed datasets
- preserve support for repeated training over the same dataset object with
  different configs
- update docs and notebooks to reflect the new ownership model

## Notes

- This should be tackled next; it is the natural follow-up to the recent HDF5
  API cleanup.
- The refactor should avoid reintroducing ambiguous double ownership between
  dataset objects and `TorchTrainingConfig`.
- Issue 11 (descriptor storage/recovery) is related, but separate: Issue 12 is
  about training-policy ownership, not descriptor metadata.

## Completion Notes

- `StructureDataset` and `HDF5StructureDataset` no longer expose runtime
  force-sampling and runtime-cache policy through their public constructors.
- `TorchTrainingConfig` is now the single public owner for
  `force_fraction`, `force_sampling`,
  `force_min_structures_per_epoch`, `cache_features`,
  `cache_force_neighbors`, `cache_force_triplets`, and
  `cache_persist_dir`.
- `TorchANNPotential.train()` now applies those controls through internal
  trainer-owned wrappers created per train/test split, so prebuilt datasets
  can be reused across runs with different configs without leaking split-local
  sampling state.
- Docs-backed examples and the torch-training notebook were updated to reflect
  the new ownership model.
- Follow-up storage-policy questions about how much force-related data should
  be persisted at HDF5 build time are tracked separately in Issue 13.

# Issue 15: [Performance] Define a unified HDF5 cache schema with optional persisted-feature sections

**Priority**: Medium
**Status**: Completed (Closed; schema v2 now writes a unified `/torch_cache` container with optional raw-feature and force-derivative sections, while legacy v1 derivative-only files remain readable)
**Created**: 2026-03-25

## Problem

The current HDF5 derivative-cache schema is versioned and documented, but it
only covers persisted local derivatives. If persisted raw features are added
without a coherent schema plan, the HDF5 format will likely split into
parallel feature and derivative layouts with unclear compatibility and loading
rules.

## Proposed Solution

Define one canonical versioned HDF5 cache schema for torch-training datasets
that can accommodate files with and without persisted features.

The schema design should include:

- shared cache-level metadata such as schema version, payload format,
  descriptor-compatibility signature, and storage dtype
- optional payload sections for persisted energy-view features and any future
  persisted derivative or force-view payloads
- explicit metadata describing which payload sections are present
- compatibility rules for reopening files that omit optional sections
- a migration story from the current derivative-only layout, likely through a
  new schema revision rather than a silent in-place extension

## Notes

- The goal is one canonical schema with optional payload sections, not one
  mandatory payload layout.
- Files without persisted features should remain valid under the same schema as
  long as the metadata clearly declares which payloads are present.

## Completion Notes

- `HDF5StructureDataset.build_database()` now accepts
  `persist_features=True` and writes schema v2 whenever persisted cache
  payloads are requested.
- The new on-disk root is `/torch_cache`, with shared descriptor-compatibility
  metadata plus optional `/torch_cache/features` and
  `/torch_cache/force_derivatives` sections.
- `HDF5StructureDataset` now exposes persisted raw features through
  `has_persisted_features()`, `get_persisted_feature_cache_info()`, and
  `load_persisted_features(idx)`.
- Existing derivative helpers continue to work for both schema v2 cache files
  and legacy schema v1 derivative-only files stored under
  `/force_derivatives`.

# Issue 16: [API] Load persisted features through HDF5StructureDataset with clear runtime precedence

**Priority**: Medium
**Status**: Completed (Closed; HDF5 runtime materialization now prefers persisted raw features with explicit precedence and supports derivative-backed force batches without graph rebuilds)
**Created**: 2026-03-25

## Problem

Even after a unified persisted-feature schema exists, the runtime dataset and
trainer paths need explicit precedence rules. Today ``HDF5StructureDataset``
always computes features during sample materialization, while
``cache_features=True`` remains a trainer-owned in-memory cache policy.

## Proposed Solution

Extend ``HDF5StructureDataset`` so it can load persisted raw features when
they are present and descriptor-compatible, while preserving existing fallback
behavior when they are absent.

The implementation should define:

- when persisted features are preferred over on-the-fly computation
- how persisted features interact with trainer-owned ``cache_features=True``
- whether the first implementation loads only energy-view features or also
  covers force-view payloads
- how load-time dtype casting is applied while keeping normalization behavior
  unchanged

## Notes

- Persisted features should remain raw descriptor outputs; feature
  normalization should continue to be a runtime training concern.
- This issue depends on the design decisions from Issue 14 and the schema
  definition from Issue 15.

## Completion Notes

- `HDF5StructureDataset.materialize_sample()` now uses explicit runtime
  precedence for energy-view features:
  trainer-owned `cache_features=True` cache, then compatible persisted HDF5
  raw features, then on-the-fly recomputation.
- Force-view materialization now reuses persisted raw `(N, F)` features when
  they are present and descriptor-compatible.
- When both persisted raw features and persisted local derivatives are
  available for a force-supervised entry, `HDF5StructureDataset` now serves
  the force sample without rebuilding graph/triplet payloads.
- Trainer collation now accepts derivative-backed force batches without
  requiring graph payloads, matching the existing force-loss runtime that
  already supports `features + local_derivatives`.
- Targeted HDF5 dataset, force-training, and docs-backed tests were updated
  to cover the new precedence and force-path behavior.

# Issue 17: [Validation] Validate persisted-feature round trips, compatibility, and dtype behavior

**Priority**: Medium
**Status**: Completed (Closed; persisted-feature validation now covers representative round trips, cross-dtype loads, clear incompatibility failures, and trainer/runtime-cache value equivalence)
**Created**: 2026-03-25

## Problem

Persisted raw features add new compatibility and correctness risks beyond the
existing derivative-cache validation:

- raw feature payloads must round-trip exactly or within the expected dtype
  tolerance
- descriptor mismatches must fail clearly
- load-time dtype conversion must preserve the intended semantics
- runtime cache and persisted-cache interactions must not silently change the
  feature values seen by training

## Proposed Solution

Add focused tests and validation coverage for persisted-feature behavior,
including:

- round-trip equality against directly computed raw features
- rejection of incompatible descriptor settings
- reopening a file with a different descriptor dtype and validating the cast
- interaction with trainer-owned runtime caches and fallback paths when
  persisted features are absent

## Notes

- This issue should extend the existing HDF5 dataset and training tests rather
  than creating a disconnected benchmark-only harness.

## Completion Notes

- `src/aenet/torch_training/tests/test_hdf5_dataset.py` now validates
  persisted raw-feature round trips against direct descriptor recomputation for
  representative single-species, multi-species/typespin, and periodic TiO2
  structures.
- Cross-dtype reopen coverage now checks both `float64 -> float32` and
  `float32 -> float64` persisted-feature loads against direct features under
  the active descriptor dtype.
- Dataset-level fallback coverage now locks in the behavior when
  `/torch_cache/features` is absent for both energy-view and force-view
  materialization, and public sample materialization now has an explicit test
  for clear failure on incompatible persisted-feature descriptors.
- `src/aenet/torch_training/tests/test_trainer_smoke.py` now verifies that the
  trainer-owned `cache_features=True` path observes identical feature values
  whether HDF5 samples come from persisted features or on-the-fly
  recomputation.
- Targeted `pytest` and `ruff check` runs passed without requiring any runtime
  code changes, which indicates the Issue 16 implementation already preserved
  the intended persisted-feature semantics.

# Issue 18: [Refactor] Consolidate redundant feature-materialization paths after persisted-feature design

**Priority**: Medium
**Status**: Completed (Closed; dataset materialization now shares one internal helper layer across in-memory, HDF5, and eager cached energy-view paths, with parity/regression coverage locking in the canonical `(N, F)` behavior)
**Created**: 2026-03-25

## Problem

The current codebase contains duplicated feature-materialization logic across
``StructureDataset`` and ``HDF5StructureDataset``, plus an additional
``CachedStructureDataset`` path for energy-only precomputed features. That
duplication makes it harder to reason about canonical runtime behavior and
increases the cost of adding persisted-feature support cleanly.

## Proposed Solution

After Issue 14 identifies the canonical raw feature representation, refactor
the dataset stack to reduce redundant paths.

Likely directions include:

- extracting shared sample-materialization helpers between in-memory and HDF5
  datasets
- collapsing duplicated energy-view and force-view feature branches where the
  design pass shows they are equivalent
- reassessing whether ``CachedStructureDataset`` remains necessary once HDF5
  persisted features and trainer-owned caches cover the same workflow

## Notes

- This issue should follow the design/evaluation pass; it should not guess at
  simplifications before the canonical representation is decided.

## Completion Notes

- The shared internal helper module now lives in
  `src/aenet/torch_training/_materialization.py` and owns canonical structure
  filtering, tensor preparation, energy-view feature loading, force-view graph
  materialization, and final sample-dict assembly.
- `StructureDataset.materialize_sample()` and
  `HDF5StructureDataset.materialize_sample()` now route through the same
  helper layer, which removes duplicated energy-view and force-view
  materialization branches while preserving the persisted-feature and
  persisted-derivative runtime precedence established in Issue 16.
- `CachedStructureDataset` was kept as a public energy-only eager cache, but
  it now precomputes the same canonical raw `(N, F)` feature payload through
  the shared energy-view helper instead of maintaining an independent feature
  construction path.
- The HDF5 energy-view neighbor-cache miss path no longer computes features
  twice before storing neighbor data; a targeted regression test now locks in
  the single-pass behavior.
- New tests now validate parity between in-memory and HDF5 sample
  materialization when no persisted payloads are present, plus parity between
  `CachedStructureDataset` and canonical energy-view materialization.
- Targeted `pytest` and `ruff check` runs passed for the affected dataset,
  trainer-smoke, docs-backed, and prediction-semantic test slices.

# Issue 19: [Docs] Document unified persisted-feature storage and runtime cache semantics

**Priority**: Medium
**Status**: Completed (Closed; user-facing, developer-facing, and API-facing docs now describe the unified persisted-feature schema and cache precedence, with docs-backed tests covering the workflow)
**Created**: 2026-03-25

## Problem

Persisted features introduce another axis of caching and storage behavior on
top of existing derivative persistence and trainer-owned runtime caches. The
current docs already explain derivative caching, but they do not describe a
unified HDF5 cache schema or the distinction between persisted raw features
and in-memory runtime feature caches.

## Proposed Solution

Update the user and developer documentation to describe:

- the unified versioned HDF5 cache schema and its optional payload sections
- when persisted features are stored and when they are loaded
- the distinction between persisted features, persisted derivatives, and
  trainer-owned runtime caches such as ``cache_features=True``
- any simplifications to the dataset/runtime path introduced by Issue 18

## Notes

- The schema documentation should stay aligned with the actual HDF5 metadata
  and payload layout rather than describing an aspirational format.

## Completion Notes

- `docs/source/usage/torch_datasets.rst` now documents the unified
  `/torch_cache` workflow, including the distinction between persisted raw
  features, persisted force derivatives, `cache_features=True`, and
  `CachedStructureDataset`.
- `docs/source/usage/torch_training.rst` now clarifies that
  `cache_features=True` is a runtime in-memory cache layer and points readers
  to the HDF5 cache-precedence workflow for persisted reuse across sessions.
- `docs/source/dev/torch_force_hdf5_cache.rst` now explicitly positions the
  page as the schema/metadata reference behind the user-facing workflow docs.
- `src/aenet/torch_training/hdf5_dataset.py` docstrings now describe the
  unified `/torch_cache` schema, persisted descriptor manifests, and the
  implemented runtime precedence instead of the older derivative-only or
  "future integration" language.
- `src/aenet/torch_training/tests/test_docs_torch_datasets.py` now exercises
  the docs-backed HDF5 workflow with `persist_features=True` and
  `persist_force_derivatives=True`, including runtime-cache precedence and the
  graph-free force-path behavior when both persisted payloads are present.

# Issue 21: [API] Rename cache controls for clearer PyTorch training semantics

**Priority**: Medium
**Status**: Completed (Closed; `cache_force_neighbors` was renamed to `cache_neighbors` throughout the PyTorch training API, docs, notebook examples, and tests)
**Created**: 2026-03-26

## Problem

The current runtime-cache naming in `TorchTrainingConfig` no longer matches
what the implementation actually does.

- `cache_force_neighbors` is misleading because it also affects energy-view
  feature materialization and mixed energy/force runs.
- `cache_features` is not purely "energy-only" in practice: in mixed runs it
  caches energy-view features for samples that are not selected for force
  supervision in the current epoch window.
- `cache_force_triplets` still appears force-specific in the current
  implementation, which makes the naming asymmetry even more confusing.

This creates avoidable user confusion and makes the docs harder to read,
especially now that runtime caches, HDF5 persisted caches, and HDF5
`in_memory_cache_size` are all documented separately.

## Proposed Solution

Perform a narrow API cleanup pass focused on names and descriptions, without
changing the underlying runtime-cache ownership model introduced in Issue 12.

Likely implementation directions:

- rename `cache_force_neighbors` to `cache_neighbors`
- keep `cache_force_triplets` unchanged for now unless a stronger rename case
  emerges during review
- decide whether `cache_features` should remain named as-is but be documented
  more explicitly as an energy-view feature cache
- update validation, docstrings, user docs, notebook examples, and migration
  notes to use the new terminology consistently
- provide a compatibility path for the old name if one is deemed necessary

## Notes

- This issue should stay narrow and avoid expanding into runtime-cache
  redesign; that belongs in Issue 22.
- This issue should not reopen the dataset-vs-config ownership boundary
  settled in Issue 12.
- The main goal is clearer user-facing semantics and lower surprise, not a
  larger cache API regrouping.

## Completion Notes

- `TorchTrainingConfig` now exposes `cache_neighbors` instead of
  `cache_force_neighbors`, and the trainer-owned runtime policy wrappers plus
  dataset materialization helpers use the new name consistently.
- The rename was applied without a compatibility alias, matching the decision
  to keep the change narrow and avoid carrying old terminology forward.
- User-facing docs in `docs/source/usage/torch_training.rst` and
  `docs/source/usage/torch_datasets.rst` now use `cache_neighbors`
  consistently, while keeping `cache_force_triplets` unchanged.
- `notebooks/example-05-torch-training.ipynb` now uses
  `cache_neighbors=True` in the force-training example.
- Targeted torch-training, HDF5 dataset, force-training, and docs-backed
  `pytest` slices passed after the rename, and a Sphinx HTML build completed
  successfully.

# Issue 22: [Performance] Redesign trainer-owned runtime caches for large datasets

**Priority**: Medium
**Status**: Completed (Closed; trainer-owned runtime caches now use bounded per-cache limits by default, warmup is opt-in, and worker-mode semantics are documented explicitly)
**Created**: 2026-03-26

## Problem

The trainer-owned runtime caches currently work, but their behavior is not a
good fit for large datasets or worker-based loading.

- the trainer-owned runtime caches are currently unbounded
- cache warmup currently walks the full enabled split eagerly
- trainer-owned runtime caches are reset when `DataLoader` workers spawn, so
  eager warmup is less useful than it appears when `num_workers > 0`
- the current behavior is directionally acceptable for small in-memory runs,
  but it is not a clear large-dataset strategy

This is now the main remaining runtime-cache design problem after the
ownership cleanup in Issue 12 and the HDF5 persisted-cache precedence work in
Issues 16 and 19.

## Proposed Solution

Redesign the trainer-owned runtime-cache layer so it scales more predictably
for larger datasets and worker-based training.

Likely implementation directions:

- introduce explicit per-cache size limits rather than unbounded maps
- decide whether cache warmup should become opt-in rather than implicit
- document clearly what happens to trainer-owned caches when
  `num_workers > 0`
- decide whether worker-aware caching needs a different design from the
  current main-process-owned caches
- preserve the current HDF5-side bounded `in_memory_cache_size` design rather
  than merging these layers

## Notes

- This issue is about trainer-owned runtime caches only.
- HDF5 persisted-feature and persisted-derivative reuse semantics should stay
  unchanged unless a concrete correctness problem is discovered.
- Issue 21 should land first so the cache terminology is stable before any
  larger redesign.

## Completion Notes

- `TorchTrainingConfig` now exposes explicit per-cache entry limits for
  trainer-owned runtime caches:
  `cache_feature_max_entries`, `cache_neighbor_max_entries`, and
  `cache_force_triplet_max_entries`. These are bounded by default but can be
  set to `None` for an explicit unbounded cache or `0` to suppress storage.
- Trainer-owned runtime caches in `_TrainingPolicyDataset` now use bounded
  LRU-style caches instead of unbounded dictionaries, while preserving the
  existing HDF5-side `in_memory_cache_size` behavior and persisted cache
  precedence.
- Runtime cache warmup is now explicit via `cache_warmup=True` instead of
  always running implicitly. Warmup also stops early once all enabled bounded
  caches have filled, rather than eagerly walking the full split in every
  bounded-cache case.
- Warmup is skipped automatically when `num_workers > 0`, with a user-facing
  warning that trainer-owned runtime caches are worker-local after DataLoader
  worker spawn.
- The trainer now initializes force-sampling state before any optional warmup,
  so single-process warmup reflects the actual epoch-0 runtime policy.
- User docs in `docs/source/usage/torch_training.rst` and
  `docs/source/usage/torch_datasets.rst` now describe the new cache-limit and
  warmup controls and explain more clearly that trainer-owned caches are
  separate from HDF5 `in_memory_cache_size` and persisted HDF5 cache payloads.
- Targeted config, trainer smoke, and docs-backed torch-training pytest
  slices passed after the redesign. A narrower `ruff check --select F,E9`
  pass also completed successfully on the modified Python files. Full-file
  style lint on `trainer.py` / `config.py` remains blocked by pre-existing
  repository lint backlog outside this issue's scope.

# Issue 23: [Performance] Add parallel build support to `HDF5StructureDataset.build_database()`

**Priority**: Medium
**Status**: Completed (Closed; build-time worker parallelism, deterministic ordered writes, atomic rebuild cleanup, and HDF5-build docs landed)
**Created**: 2026-03-26

## Problem

`HDF5StructureDataset.build_database()` is currently a serial preprocessing
path over input files plus any requested persisted-cache computation.

- parser work is currently done in a single loop
- optional persisted feature generation is currently serialized
- optional persisted force-derivative generation is currently serialized
- for very large datasets, this can become a major preprocessing bottleneck

Current docs also use "multiprocessing" language that can be misread as
build-time parallelization support, even though the implemented
multiprocessing support today applies only to read-time worker access.

## Proposed Solution

Add explicit build-time parallelization support to
`HDF5StructureDataset.build_database()`, with a design that preserves
deterministic HDF5 output ordering and clear failure handling.

Likely implementation directions:

- separate worker-side parsing and optional featurization from coordinated
  ordered writes into the HDF5 file
- keep the single-writer HDF5 constraint explicit in the design
- define how progress reporting and partial-failure cleanup should behave
- document clearly that build-time parallelization is distinct from
  read-time `DataLoader` worker support
- add representative large-dataset and persisted-cache build tests

## Notes

- This issue should not change runtime sample materialization precedence.
- Any API for build parallelism should remain separate from training-time
  `num_workers`.
- Issue 24 should update the docs in parallel so users do not confuse the two
  kinds of parallelism.

## Completion Notes

- `HDF5StructureDataset.build_database()` now accepts `build_workers`,
  parallelizing parser execution and optional persisted-cache preparation while
  preserving deterministic ordered HDF5 writes through a single writer.
- The build path now stages output in a temporary file and replaces the target
  database only after a successful flush/close, so failed rebuilds do not
  clobber an existing HDF5 dataset.
- `src/aenet/torch_training/hdf5_dataset.py` now factors build-time payload
  preparation away from write-time HDF5 appends so persisted feature and
  force-derivative payloads can be prepared concurrently without changing
  runtime sample materialization precedence.
- `src/aenet/torch_training/tests/test_hdf5_dataset.py` now covers
  deterministic multiframe ordering, serial-vs-parallel persisted-cache
  equivalence, and failure cleanup that preserves an existing database.
- `docs/source/usage/torch_datasets.rst` now documents `build_workers` as a
  build-time concern distinct from training-time `num_workers`, and the
  docs-backed HDF5 workflow tests were updated to match the post-build lazy
  reopen behavior.
- Verification completed with targeted HDF5/docs-backed `pytest` slices, full
  `ruff check` on the touched Python files, and a successful Sphinx dummy
  build.

# Issue 20: [API] Clean up PyTorch training API semantics and documentation

**Priority**: Medium
**Status**: Completed (Closed; the umbrella cleanup landed through Issues 21-27, and future mixed-memory work was split into Issue 28)
**Created**: 2026-03-26

## Problem

This umbrella issue tracked a cluster of PyTorch training API and
documentation inconsistencies identified during the 2026-03-26 review
session.

The review started from runtime-cache naming, but the real scope was broader:

- runtime cache naming and semantics
- runtime cache layering versus persisted HDF5 cache reuse
- large-dataset runtime-cache behavior
- missing HDF5 build-time parallelism
- unclear documentation for devices, workers, and execution stages
- counter-intuitive resumed-training `iterations` semantics
- missing `save_energies=True` support for lazy HDF5-backed datasets

The implementation was mostly functional already, but too many semantics were
surprising, implicit, or under-documented for a stable user-facing training
API.

## Proposed Solution

Use the umbrella issue as a handoff and tracking point for a focused cleanup
pass that prioritized clarity and minimal user surprise over large redesign.

That cleanup was ultimately split into narrower follow-up issues:

- Issue 21: cache-control naming cleanup and targeted docs refresh
- Issue 22: trainer-owned runtime-cache redesign for large datasets
- Issue 23: parallel HDF5 build support in
  `HDF5StructureDataset.build_database()`
- Issue 24: clarify the execution model for `device`,
  `descriptor.device`, `num_workers`, and `memory_mode`
- Issue 25: make resumed-training `iterations` semantics explicit
- Issue 26: support `save_energies=True` for HDF5-backed datasets
- Issue 27: decide the future of `memory_mode='mixed'`

## Notes

- The umbrella issue intentionally grouped related cleanup work so the project
  could land targeted fixes without attempting a single large redesign.
- The only remaining forward-looking item after the cleanup pass was real
  mixed-memory support, which is now tracked as separate feature work in
  Issue 28 rather than as unfinished umbrella debt.

## Completion Notes

- Runtime cache naming and documentation were cleaned up, including the
  `cache_force_neighbors` to `cache_neighbors` rename and clearer user-facing
  cache-layer guidance.
- Trainer-owned runtime caches now use bounded per-cache limits by default,
  warmup is optional, and worker-local cache behavior is documented more
  explicitly.
- `HDF5StructureDataset.build_database()` now supports build-time worker
  parallelism with deterministic ordered writes and atomic rebuild cleanup.
- The training docs and maintained notebook now explain the current execution
  model explicitly, including the roles of `descriptor.device`,
  `config.device`, and `num_workers`.
- Resumed training now treats `iterations` as the number of additional epochs
  to run in that call.
- `save_energies=True` now works with lazy HDF5-backed datasets through a
  generalized dataset identifier interface.
- `memory_mode='mixed'` no longer silently follows the `'gpu'` path; it now
  raises `NotImplementedError`, and future real mixed-memory implementation
  work has been split into Issue 28.
- With Issues 21-27 closed and Issue 28 now standing as a separate future
  feature, this umbrella tracker no longer needs to remain open.

# Issue 24: [Docs] Clarify the PyTorch training execution model for devices and workers

**Priority**: Medium
**Status**: Completed (Closed; user docs and maintained notebook now explain the current device/worker execution model explicitly)
**Created**: 2026-03-26

## Problem

The current PyTorch training docs did not explain the execution model clearly
enough for combined GPU training and worker-based data loading.

- combined GPU training plus worker-side data preparation was already
  possible
- in practice, the implementation is best described as worker-side data
  loading and featurization feeding a model-training loop on the selected
  device
- the interaction between `device`, `descriptor.device`, `num_workers`, and
  `memory_mode` was not described clearly enough
- `memory_mode='mixed'` was accepted but was not a distinct execution mode in
  practice

This left users to infer too much from examples and could lead to incorrect
mental models about where work was actually happening.

## Proposed Solution

Produce a focused docs cleanup pass that explains the current execution model
as implemented today.

Likely documentation directions:

- distinguish model compute on the selected device from worker-side data
  loading and featurization
- explain how `num_workers > 0` interacts with lazy HDF5 loading and
  trainer-owned runtime caches
- explain the current role of `descriptor.device`
- document prominently that `memory_mode='mixed'` is not currently distinct
  from `'gpu'`
- update notebook examples and usage guidance to reflect the current
  implementation rather than an aspirational pipeline

## Notes

- This first pass was documentation-only.
- The status of `memory_mode='mixed'` was left for a later API decision in
  Issue 27; future real mixed-memory implementation work now lives in
  Issue 28.
- This issue intentionally stayed aligned with Issue 23 so build-time
  `build_workers` and training-time `num_workers` are documented as separate
  concepts.

## Completion Notes

- `docs/source/usage/torch_training.rst` now contains an explicit execution
  model section that distinguishes worker-side sample preparation from model
  compute on `config.device`, explains the role of `descriptor.device`, and
  calls out the current HDF5 worker-handle/cache behavior.
- The same page documented the then-current state of `memory_mode='mixed'` as
  an accepted compatibility placeholder rather than a distinct split-device
  execution mode. That API decision was later revisited and closed in
  Issue 27.
- The maintained notebook `notebooks/example-05-torch-training.ipynb` now
  mirrors that guidance so the notebook-first training workflow does not imply
  an aspirational mixed pipeline.
- `src/aenet/torch_training/tests/test_config.py` gained a targeted check that
  the compatibility-only `mixed` setting remains accepted.
- Verification completed with targeted docs-backed/config `pytest` slices,
  notebook JSON validation, `ruff check` on the touched test file, and a full
  Sphinx HTML build:
  `python -m sphinx -b html docs/source docs/build/html`.

# Issue 25: [API] Make resumed-training `iterations` semantics explicit

**Priority**: Medium
**Status**: Completed (Closed; resumed training now treats `iterations` as per-call additional epochs)
**Created**: 2026-03-26

## Problem

During resumed training, `iterations` previously behaved like a total target
epoch count rather than "epochs to run in this call."

- training only continued when `iterations > completed_epochs`
- users had to know the previous completed epoch count before choosing the
  next call
- resume behavior from `best_model.pt` and other non-numbered checkpoint
  names relied on weak filename inference rather than persisted checkpoint
  metadata

This made restart workflows more awkward than they needed to be and left too
much room for incorrect assumptions.

## Proposed Solution

Adopt the more user-facing interpretation directly: `iterations` should mean
the number of epochs to run in the current `train()` call, including resumed
runs.

Implementation directions completed here:

- resumed runs now compute the end epoch as `start_epoch + iterations`
- checkpoint loading now works even when the resumed call does not configure a
  new `checkpoint_dir`
- checkpoint epoch detection prefers persisted checkpoint metadata and only
  falls back to filename parsing
- docs, maintained notebook examples, and targeted tests now describe and
  assert the per-call semantics explicitly

## Notes

- This was implemented as a breaking API change because the code is still new
  and not yet widely distributed.
- The change stays focused on resume semantics; it does not redesign the
  broader checkpoint file format.

## Completion Notes

- `TorchTrainingConfig.iterations` and `TorchANNPotential.train()` now define
  `iterations` as the number of epochs to run in the current call, even when
  `resume_from=...` is used.
- Resumed training now loads checkpoints regardless of whether the resumed
  call also enables checkpoint saving via `checkpoint_dir`.
- Resume epoch detection now uses persisted checkpoint metadata first, which
  makes `best_model.pt` resumes behave correctly.
- Best-model checkpoints are now saved after the current epoch metrics/history
  are recorded, so saved history length matches the saved checkpoint epoch.
- `docs/source/usage/torch_training.rst` and
  `notebooks/example-05-torch-training.ipynb` now state that resumed
  `iterations` are additional epochs in that call.
- Verification completed with targeted `pytest` slices for checkpoint resume,
  progress bars, and docs-backed training examples; notebook JSON validation;
  and a full Sphinx HTML build. A broader `ruff check` over the touched Python
  modules still reports pre-existing legacy typing/docstring violations
  outside the scope of this issue.

# Issue 26: [API] Support `save_energies=True` for HDF5-backed datasets

**Priority**: Medium
**Status**: Completed (Closed; `save_energies=True` now supports lazy HDF5-backed datasets)
**Created**: 2026-03-26

## Problem

`save_energies=True` previously relied on training datasets exposing a
`.structures` attribute.

- `HDF5StructureDataset` is intentionally lazy and does not expose
  `.structures`
- the energy-output helper therefore skipped HDF5-backed datasets even though
  dataset-backed prediction itself already worked
- HDF5 metadata already persisted `path`, `frame`, and `name`, so the missing
  support was mainly an output-helper API gap rather than a schema problem

This made the energy-output helper less generally usable than the rest of
the training API.

## Proposed Solution

Generalize the energy-output helper path so it can work with lazy datasets,
including `HDF5StructureDataset`, while preserving the existing
`energies.*` file format.

Implementation directions completed here:

- stop relying exclusively on a `.structures` attribute
- support a more general dataset interface based on `get_structure()` plus
  optional `get_structure_identifier()`
- for HDF5-backed datasets, prefer persisted identifiers and append
  `#frame=N` so multi-frame sources remain distinguishable
- add tests covering HDF5-backed `save_energies` outputs and identifier
  precedence

## Notes

- This did not require an HDF5 schema change.
- The implementation stayed focused on lazy dataset support and did not
  redesign the `energies.*` file format.

## Completion Notes

- `TorchANNPotential.save_energies` helpers now extract structures from lazy
  datasets via `get_structure()` when available, including through
  trainer-owned `Subset`/policy wrappers.
- `StructureDataset`, `CachedStructureDataset`, and
  `HDF5StructureDataset` now expose a stable dataset identifier hook for
  energy-output paths.
- HDF5-backed energy outputs now reconstruct identifiers from persisted
  metadata in this order: source path, persisted name, then a synthetic
  `structure_XXXXXX` fallback, with an explicit `#frame=N` suffix in all
  cases so multi-frame inputs are unambiguous.
- `src/aenet/torch_training/tests/test_trainer_smoke.py` now covers HDF5
  train/test split outputs and the path/name/fallback identifier precedence.
- `docs/source/usage/torch_training.rst` now documents the HDF5
  `save_energies` identifier behavior explicitly.
- Verification completed with the full trainer smoke test file, targeted
  `ruff check --select F,E9` on the touched Python files, and a Sphinx dummy
  build. A broader repo-style `ruff check` still reports pre-existing legacy
  typing/docstring findings in `trainer.py` outside the scope of this issue.

# Issue 27: [API] Decide the future of `memory_mode='mixed'`

**Priority**: Medium
**Status**: Completed (Closed; `memory_mode='mixed'` now fails fast with `NotImplementedError`, and future real mixed-memory work moved to Issue 28)
**Created**: 2026-03-26

## Problem

`memory_mode='mixed'` was accepted by the config and trainer even though it
did not behave as a distinct execution mode in practice.

- users could reasonably infer a real mixed CPU/GPU execution path
- the implementation instead followed the `'gpu'` path
- documentation alone was not a sufficient long-term contract for an exposed
  API mode with no actual implementation

This left the project carrying under-specified public API behavior.

## Proposed Solution

Make an explicit project decision about the status of `memory_mode='mixed'`.

The implemented decision for this issue was:

- stop accepting `'mixed'` as a silent compatibility alias
- raise `NotImplementedError` when users request it
- reserve the name for a future real mixed-memory implementation
- spin the actual feature work into a separate follow-up issue

## Notes

- Issue 24 landed first so the then-current behavior was documented before the
  API decision changed.
- This issue intentionally stayed focused on the product/API contract rather
  than implementing the real mixed-memory mode itself.
- The future implementation work is now tracked separately in Issue 28.

## Completion Notes

- `TorchTrainingConfig` now raises `NotImplementedError` when
  `memory_mode='mixed'` is requested, while still rejecting other invalid
  strings with `ValueError`.
- `TorchANNPotential.train()` now includes a defensive `NotImplementedError`
  guard as well, so mutated configs cannot silently route `'mixed'` through
  the existing `'gpu'` path.
- `docs/source/usage/torch_training.rst` now states that `'mixed'` is a
  reserved future mode rather than a supported compatibility alias.
- `src/aenet/torch_training/tests/test_config.py` now verifies that
  requesting `'mixed'` fails fast.
- `ISSUES.md` now closes this decision issue and opens a separate Issue 28 for
  the future real mixed-memory implementation aimed at large force-training
  datasets.

# Issue 29: [API] Make HDF5 dataset building source-oriented instead of path-oriented

**Priority**: Medium
**Status**: Closed
**Created**: 2026-03-28

## Problem

`HDF5StructureDataset` currently exposes a build API centered on
`file_paths` plus a user-supplied `parser`. That works for ordinary
filesystem-backed structure collections, but the real contract is already
more general: the builder consumes a sequence of source records and turns
each record into one or more torch-training `Structure` objects.

This mismatch creates avoidable API friction for non-filesystem inputs such
as:

- archive members inside `.tar`, `.tar.gz`, or `.tar.bz2` files
- sequential or streamed structure sources
- manifest-driven structure collections
- future object-store-backed or database-backed sources

The path-oriented naming also leaks into the persisted HDF5 metadata, which
still stores a `path` field even when the originating source is not actually
a filesystem path.

As a result, legitimate workflows beyond plain files currently require
user-side workarounds that repurpose `file_paths` as generic source
identifiers, encode frame information into synthetic path strings, and
implement custom parser plumbing in notebooks or scripts.

## Proposed Solution

Refactor the HDF5 build API toward an explicit source-oriented model.

Recommended direction:

- replace the public `file_paths` argument with `sources`
- remove the separate top-level `parser` argument from the primary public API
- if `sources` contains strings or path-like objects, wrap them internally in
  a built-in file-source adapter based on
  `AtomicStructure.from_file(...).to_TorchStructure()`
- add first-class source adapters for standard non-file inputs, especially
  sequential archive-backed sources
- let source adapters advertise capabilities such as:
  - single-pass vs multi-pass traversal
  - random-access support
  - build-worker / parallel-build compatibility
- let source adapters own source-specific parsing and source-identity logic
- rename persisted metadata away from `path` toward structured source fields
  such as:
  - `source_id`
  - `frame_idx`
  - optionally `source_kind` and/or `display_name`

Breaking changes are acceptable for this internal-use codebase, so this
issue intentionally recommends a direct API cleanup rather than a deprecated
alias transition.

## Development Plan

The recommended rollout is:

1. Replace `file_paths` with `sources` across the public HDF5 dataset API
2. Implement a built-in file-source adapter for ordinary path-like sources
3. Define a minimal source-adapter protocol for HDF5 dataset building
4. Implement one built-in non-file adapter first, preferably an
   archive-backed source for XSF members
5. Update persisted metadata and internal naming from path-oriented to
   source-oriented terminology, including explicit frame metadata
6. Ensure any user-facing path-like labels needed by downstream outputs are
   synthesized from structured source metadata rather than stored as the
   canonical internal representation
7. Update tests, docs, and notebooks in one pass to reflect the new API

The detailed implementation and handoff plan is recorded in:

- `dev-notes/HDF5_SOURCE_ORIENTED_API_PLAN.md`

## Notes

- The key design goal is to make the core API source-oriented, not merely to
  rename one argument.
- Archive semantics should remain explicit via adapters or dedicated helper
  constructors rather than being inferred implicitly from strings.
- The adapter should own parsing behavior. Path-like inputs are just one
  built-in adapter case; non-file workflows should not need a separate
  parser callback.
- Sequential archive sources should support single-pass builds without
  forcing all structure contents into memory first.
- Archive-backed sources should be able to declare when they are not suitable
  for random-access parsing or threaded build workers.
- Multi-frame source handling should use structured metadata such as
  `source_id` plus `frame_idx` rather than encoding frame information into
  synthetic path strings.
- `TorchTrainingConfig.save_energies` still needs a Fortran-compatible output
  format that includes merged path/source and frame information. That
  formatting should be synthesized at file-generation time from the structured
  source metadata, without making the merged string the canonical stored
  representation inside the dataset.
- Current implementation status:
  - Phase 1 completed: source abstractions and the built-in file-source
    adapter landed
  - Phase 2 completed for the core dataset API: `HDF5StructureDataset` now
    consumes `sources` and no longer exposes the public top-level `parser`
  - Phase 3 completed: persisted HDF5 metadata now stores structured source
    fields such as `source_id`, `frame_idx`, `source_kind`, and
    `display_name`
  - Phase 4 completed: `save_energies` continues to emit the
    Fortran-compatible merged identifier format, synthesized at output time
    from structured source metadata
  - Phase 5 completed: archive-backed XSF sources are now supported through
    a built-in tar adapter, with conservative capability flags for
    compressed tar archives such as `*.tar.bz2`
  - Phase 6 completed: downstream tests were updated to cover the
    source-oriented API and archive-backed workflows
  - Phase 7 completed: user-facing docs and
    `notebooks/example-05-torch-training.ipynb` now reflect the
    source-oriented HDF5 API
- Convenience constructors such as `from_xsf_files(...)` or
  `from_tar_xsf_archive(...)` may still be useful for discoverability, but
  they should be thin wrappers around the same source-oriented core API.

# Issue 30: [Bug] Close HDF5 worker handles explicitly during multiprocess torch training

**Priority**: Medium
**Status**: Completed
**Created**: 2026-03-28

## Problem

`HDF5StructureDataset` lazily opened worker-local PyTables handles during
multiprocess training, but those handles were not closed deterministically
when DataLoader workers exited. This was especially visible when random force
resampling disabled persistent training workers and caused worker restarts
between epochs, producing noisy PyTables `UnclosedFileWarning` messages.

## Resolution

Implemented on 2026-03-28.

- Added trainer-side helpers to discover reachable
  `HDF5StructureDataset` roots through wrapper layers such as
  `_TrainingPolicyDataset` and `torch.utils.data.Subset`.
- Registered a worker-local `atexit` cleanup hook from the trainer's
  `worker_init_fn` so worker-owned HDF5 handles are closed explicitly on
  shutdown while preserving lazy-open behavior.
- Wired the cleanup hook into both training and evaluation DataLoaders when
  `num_workers > 0` without changing the public training API.
- Added regression tests covering wrapper traversal, cleanup registration,
  DataLoader wiring, and a real HDF5 force-training smoke path with worker
  restarts.
- Updated the torch-training docs to clarify that HDF5 handles are worker
  local and explicitly closed when workers exit.

## Validation

- `pytest src/aenet/torch_training/tests/test_trainer_smoke.py -k 'hdf5_root or worker_cleanup or persistent_train_workers'`
- `pytest src/aenet/torch_training/tests/test_force_training.py -k 'worker_restarts_smoke or random_sampling_initializes_force_selection'`
- `ruff check --select F,E9,I001 src/aenet/torch_training/trainer.py src/aenet/torch_training/tests/test_trainer_smoke.py src/aenet/torch_training/tests/test_force_training.py`

# Issue 31: [Bug] Align `max_energy` dataset filtering with referenced cohesive / formation-energy semantics

**Priority**: Medium
**Status**: Completed
**Created**: 2026-03-28

## Problem

The PyTorch training backend already used referenced cohesive or formation
energy per atom for non-uniform structure sampling when `atomic_energies`
was provided, with an intentional all-zero atomic-reference fallback when it
was not. However, `max_energy` filtering in the in-memory dataset classes
still used raw total energy per atom, so filtering could disagree with both
the effective training targets and `sampling_policy="energy_weighted"`.

The ambiguity was larger for HDF5 workflows: `TorchTrainingConfig.max_energy`
is a runtime setting, while HDF5 datasets are usually prepared ahead of time
and then reused across runs. The HDF5 path therefore needed an explicit
build-time design instead of silently inheriting runtime behavior.

## Resolution

Implemented on 2026-03-28.

- Added a shared referenced-energy-per-atom helper and used it for
  structure-list-backed `max_energy` filtering in both `StructureDataset`
  and `CachedStructureDataset`.
- Preserved the intentional zero-reference fallback when
  `atomic_energies` is omitted, so externally referenced user labels are
  still filtered as provided.
- Updated trainer-side dataset construction from raw `structures=...` so
  `TorchTrainingConfig.max_energy` and `config.atomic_energies` now use the
  same referenced-energy semantics.
- Preserved those semantics through `train_test_split(...)` by propagating
  dataset-level `atomic_energies`.
- Added a trainer warning when `TorchTrainingConfig.max_energy` is set but
  the run uses a prebuilt external dataset, because that runtime option
  cannot retroactively filter already-constructed datasets.
- Made the HDF5 policy explicit with build-time
  `HDF5StructureDataset.build_database(max_referenced_energy_per_atom=..., atomic_energies=...)`
  controls and persisted metadata recording whether explicit atomic
  references or the zero-reference fallback was used during filtering.
- Updated the torch-training and torch-datasets documentation to make the
  runtime-vs-build-time distinction explicit.

## Validation

- `/Users/aurban/.local/bin/micromamba run -n aenet-torch pytest -q src/aenet/torch_training/tests/test_predict_energy_semantics.py src/aenet/torch_training/tests/test_trainer_smoke.py src/aenet/torch_training/tests/test_hdf5_dataset.py`
  (`85 passed`)
- `/Users/aurban/.local/bin/micromamba run -n aenet-torch pytest -q src/aenet/torch_training/tests/test_docs_torch_datasets.py src/aenet/torch_training/tests/test_docs_torch_training.py`
  (`8 passed`)
- `/Users/aurban/.local/bin/micromamba run -n aenet-torch ruff check --select F,E9 src/aenet/torch_training/_materialization.py src/aenet/torch_training/dataset.py src/aenet/torch_training/trainer.py src/aenet/torch_training/hdf5_dataset.py src/aenet/torch_training/config.py src/aenet/torch_training/tests/test_predict_energy_semantics.py src/aenet/torch_training/tests/test_trainer_smoke.py src/aenet/torch_training/tests/test_hdf5_dataset.py`
  (`All checks passed!`)
