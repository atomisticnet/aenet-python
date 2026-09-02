# Issue 38: Implement force-informed local Taylor sampling for ANN training

**Type:** Feature
**Priority:** Medium
**Status:** Open
**Created:** 2026-08-30

## Problem

Accurate force prediction is essential for molecular dynamics and geometry
optimization, but direct force supervision is often the computational and
memory bottleneck in ANN-potential training. A reference structure provides
one total-energy label and `3N` force components, yet the current direct-force
training path must materialize descriptor derivatives and graph/triplet data,
contract predicted energy gradients into forces, and backpropagate a loss that
depends on those gradients.

The package needs a supported alternative that uses reference force
information without putting a force error directly in the training loss. For a
reference structure with positions `R`, energy `E(R)`, and forces
`F(R) = -dE/dR`, small local displacements `delta_R` provide approximate
energy labels through the first-order Taylor expansion

```text
E(R + delta_R) ~= E(R) - sum_i dot(delta_R_i, F_i(R)).
```

The displaced structures can then be trained as ordinary energy-labeled
structures. This preserves the standard energy-only ANN training path and
avoids force-loss evaluation during optimization. It does not eliminate the
need for force-bearing reference calculations, nor does it make the derived
energies exact: the approximation is useful only for sufficiently local
displacements, and excessive displacement or excessive replication can add
noise and degrade the energy fit.

There is currently no end-to-end API that validates force-bearing reference
structures, generates statistically controlled local displacements, assigns
Taylor-expanded energy labels, preserves parent/child provenance, prevents
split leakage, trains an ANN through the energy-only path, and evaluates the
resulting force accuracy on untouched reference structures.

## Current evidence

The reference method is described by Cooper et al., *Efficient training of ANN
potentials by including atomic forces via Taylor expansion and application to
water and a transition-metal oxide*, npj Computational Materials 6, 54 (2020),
https://doi.org/10.1038/s41524-020-0323-8.

The paper considers two first-order sampling strategies:

- displace one atom in a positive or negative Cartesian direction and derive
  the new energy from the corresponding force component; and
- displace all atoms with small random vectors, remove the net center-of-mass
  translation, and constrain each atomic displacement by a maximum magnitude.

Issue 38 will use the paper for the Taylor-labeling method but will implement
the local statistical sampling workflow with the repository's existing random
and D-optimal displacement transformations. The signed single-atom Cartesian
strategy from the paper is not part of the initial implementation.

The reported results establish useful design constraints rather than universal
defaults:

- the optimal displacement is material dependent and must be selected using
  validation data;
- water examples favored displacements around `0.008-0.01` Angstrom, whereas
  the oxide example used a larger value around `0.03` Angstrom;
- increasing the number of derived structures initially improved energy and
  force errors, but the benefit plateaued and sufficiently aggressive
  augmentation increased energy error;
- random all-atom displacement was a robust compromise for larger structures;
  and
- the Taylor method recovered part, but not all, of the force-accuracy gain of
  direct force training at substantially lower training cost in the reported
  examples.

The repository already contains reusable components for most of the surrounding
workflow:

- `aenet.geometry.AtomicStructure` and
  `aenet.torch_training.Structure` store coordinates, species, total energy,
  cell information, and optional atomic forces, with conversion between the
  two representations.
- The XSF reader and writer preserve total energies and force components, and
  the tracked structures under `notebooks/xsf-TiO2/` provide a practical
  force-bearing Ti--O dataset for a maintained example.
- `RandomDisplacementTransformation` already supports seeded generation,
  optional translation removal, and orthonormal or independent random
  patterns. Its magnitude is normalized using an RMS convention over all
  Cartesian components, and its translation removal subtracts the arithmetic
  mean displacement rather than a mass-weighted center of mass.
- `DOptimalDisplacementTransformation` already initializes an ensemble through
  `RandomDisplacementTransformation`, then uses the existing SciPy optimizer to
  maximize a regularized log-determinant diversity criterion. It supports
  seeded generation, fixed output count, RMS control, translation removal, and
  zero ensemble-mean enforcement.
- Both transformations are public from `aenet.geometry.transformations` and
  return displaced `AtomicStructure` copies. Taylor augmentation must use their
  returned coordinates, replace stale copied labels, and must not implement a
  separate displacement algorithm in the training layer.
- `StructureDataset`, `CachedStructureDataset`, and
  `HDF5StructureDataset` already support energy-only training. The HDF5 source
  abstraction permits one logical `SourceRecord` to load multiple structures,
  preserves deterministic source order, and can persist reusable raw
  descriptor features without persisting force derivatives.
- HDF5 entries already expose `source_id`, `frame_idx`, `source_kind`,
  `display_name`, and `name`. These fields can identify a reference parent and
  its derived children without relying only on output order.
- `TorchANNPotential.train(...)` with `TorchTrainingConfig(force_weight=0.0)`
  follows the energy-only optimization path, while
  `TorchANNPotential.predict(..., eval_forces=True)` can evaluate forces after
  training on an untouched force-bearing validation set.
- The existing generic dataset splitter operates on individual entries. If it
  is applied after augmentation, children of one reference structure can be
  split between training and validation, causing severe information leakage.
  Parent structures must therefore be partitioned before augmentation, or a
  group-aware split must be used.
- The repository already contains committee and ensemble training/inference
  support. Those components are relevant to the planned uncertainty-
  quantification milestone in Issue 39, but they are not required for this
  issue's single-model sampling and training workflow.

The active issues have the following boundaries with this work:

- Issue 13 concerns build-time persistence of force-derivative payloads. Taylor
  sampling should not require those payloads; derived samples should be usable
  with `persist_features=True` and `persist_force_derivatives=False`.
- Issue 28 concerns mixed-memory direct force training. It is not a prerequisite
  because this issue intentionally trains the derived dataset through the
  existing energy-only path.
- Issue 36 concerns general calculation-ready structure-library generation.
  This issue may reuse its low-level displacement concepts, but it is narrowly
  responsible for force-derived training labels and must not grow into a
  general structure-generation or electronic-structure workflow.
- Issue 37 concerns representative and random down-selection from a feature
  matrix. It is not a prerequisite. The D-optimal strategy in this issue
  optimizes displacement-space diversity before Taylor labeling; it is not
  descriptor-based representative down-selection of an existing dataset.

## Impact

A supported Taylor-sampling workflow would:

- convert force information already available from reference calculations into
  additional local energy constraints;
- allow ANN optimization to use the existing energy-only loss and feature
  caches instead of the direct force-loss and descriptor-derivative path;
- reduce training time and peak memory for force-informed training when the
  additional energy samples are cheaper than direct gradient supervision;
- make displacement magnitude, augmentation multiple, random state, and label
  provenance explicit and reproducible;
- prevent accidental training/validation leakage between a reference structure
  and its local children; and
- provide a measured basis for choosing Taylor sampling over energy-only or
  direct-force training for a particular dataset.

## Proposed approach

Make `aenet.geometry.sampling` the authoritative backend-neutral location for
Taylor labeling, reference validation, transformation orchestration,
provenance, reproducibility, and parent-aware splitting. The sampling module
is intentionally broader than structure-only or representation-row selection.
Keep conversion for `torch_training.Structure`, source collections, HDF5
persistence, and trainer-facing behavior in thin
`aenet.torch_training.taylor_sampling` adapters. Preserve the already
published `aenet.torch_training` imports as compatibility paths without
maintaining a second implementation. Keep this work separate from the general
structure-library workflow in Issue 36.

The public API should separate augmentation policy from ANN optimization
policy. A provisional in-memory workflow is:

```python
from aenet.geometry.transformations import (
    RandomDisplacementTransformation,
)
from aenet.geometry.sampling import (
    TaylorExpansionConfig,
    TaylorReference,
    iter_taylor_structures,
    split_reference_structures,
)

# Wrap one-frame AtomicStructure parents with stable identities, then split
# exact references before generating any children.
references = [
    TaylorReference(parent_id, structure)
    for parent_id, structure in exact_parents
]
train_references, validation_references = split_reference_structures(...)

taylor_config = TaylorExpansionConfig(
    transformation=RandomDisplacementTransformation(
        rms=0.01,
        max_structures=24,
        random_state=42,
        orthonormalize=False,
        remove_translations=True,
    ),
    include_reference=True,
)

train_structures = list(
    iter_taylor_structures(train_references, config=taylor_config)
)

from aenet.torch_training import (
    StructureDataset,
    TorchANNPotential,
    TorchTrainingConfig,
    generate_taylor_samples as generate_torch_taylor_samples,
)

torch_train_structures = generate_torch_taylor_samples(
    torch_train_parents,
    taylor_config,
    parent_ids=torch_train_parent_ids,
).structures
training_config = TorchTrainingConfig(
    force_weight=0.0,
    sampling_policy="uniform",
    ...,
)

potential = TorchANNPotential(arch, descriptor)
potential.train(
    train_dataset=StructureDataset(torch_train_structures, descriptor),
    test_dataset=StructureDataset(torch_validation_parents, descriptor),
    config=training_config,
)
```

The final dataset/training portion uses the compatibility adapters from
`aenet.torch_training`; the neutral example above deliberately remains
importable without PyTorch.

The same orchestration must accept `DOptimalDisplacementTransformation` for
D-optimal Taylor sampling. Accepting a transformation object keeps displacement
generation and its parameters in `aenet.geometry.transformations`; the Taylor
layer owns only reference-label validation, energy construction, provenance,
and dataset integration.

The exact names can change during API review, but the implementation should
cover the following stages.

### 1. Define the reference-data and result contracts

- Accept explicitly identified, single-frame `AtomicStructure` objects as the
  canonical neutral record. The PyTorch adapter accepts
  `aenet.torch_training.Structure` and supported conversion inputs while
  requiring unique names or explicit stable parent IDs.
- Require one finite total energy and a finite force array of shape `(N, 3)`
  for every parent selected for augmentation. Reject missing or malformed
  labels before generating any children.
- Require positions with shape `(N, 3)`, matching species and force counts,
  and a valid cell/PBC combination under the existing `Structure` rules.
- Treat positions as Angstrom, forces as eV/Angstrom, and energies as eV,
  consistent with existing structure I/O and training conventions. Document
  that units cannot be inferred automatically from arbitrary in-memory
  arrays.
- Leave each input parent unchanged. Every generated child must own independent
  coordinate storage, preserve species/cell/PBC, carry the Taylor energy, and
  set `forces=None` so stale parent forces cannot be interpreted as labels at
  the displaced geometry.
- Give the original reference and every child stable identifiers that encode
  or accompany the parent identity, child index, sampling strategy, and exact
  versus approximate label origin. Return or retain a compact generation
  record containing the configuration and accepted/skipped counts.

### 2. Implement and test the Taylor-label primitive

Add a small NumPy-level function for

```text
delta_E = -sum_i dot(delta_R_i, F_i)
E_child = E_parent + delta_E.
```

The function should:

- validate shapes, finite values, and floating-point conversion;
- calculate the label from the actual post-processed displacement applied to
  the child, not the initially drawn vector;
- preserve a documented floating-point dtype policy and avoid silent integer
  truncation; and
- make the force/sign convention explicit in the API documentation and unit
  tests.

### 3. Reuse random and D-optimal displacement transformations

Support two local sampling strategies independently of ANN featurization. Both
must use the public implementations from `aenet.geometry.transformations`.

1. **Random displacement sampling**
   - Use `RandomDisplacementTransformation` rather than reimplementing random
     coordinate generation in the Taylor layer.
   - Use `orthonormalize=False` for the maintained independent-random baseline;
     document the existing orthonormal mode if the public Taylor API permits it.
   - Map the requested augmentation count to `max_structures`, displacement
     magnitude to `rms`, and reproducibility to `random_state`.
   - Use the transformation's existing arithmetic-mean translation removal
     when `remove_translations=True` and document that this is not mass-weighted.
   - Detect or clearly document degenerate cases, such as removing all
     translational displacement from a one-atom structure.

2. **D-optimal displacement sampling**
   - Use `DOptimalDisplacementTransformation`, which initializes candidates
     through `RandomDisplacementTransformation` and optimizes their regularized
     log-determinant displacement-space diversity.
   - Map the requested augmentation count to `n_structures` and expose or
     document `rms`, `random_state`, `remove_translations`,
     `enforce_zero_mean`, `max_iter`, `tol`, and `logdet_regularization`.
   - Preserve the transformation's existing fallback to the projected random
     ensemble when optimization does not improve the D-optimal objective.
   - Validate the existing `n_structures >= 2` contract and any degenerate
     structure/constraint combination before yielding partial Taylor output.
   - Treat transformation optimization time as part of augmentation cost, not
     ANN training time.

For both strategies, calculate Taylor labels from the actual child-minus-parent
coordinates returned by the transformation after RMS scaling, translation
removal, zero-mean projection, and any D-optimal optimization. Every child must
have its copied parent energy and forces replaced so no stale label survives.

RMS displacement should be required rather than assigned a material-independent
scientific default. Values from the paper may be used as starting points, but
their maximum-per-atom convention is not identical to the transformations'
RMS-over-Cartesian-components convention and must not be presented as directly
equivalent.

### 4. Define dataset size, ordering, and reproducibility

- Treat `RandomDisplacementTransformation.max_structures` and
  `DOptimalDisplacementTransformation.n_structures` as the requested number of
  approximate children per exact parent. If a higher-level
  `samples_per_reference` option remains, it must map to those native parameters
  without creating a second sampling implementation.
- With `include_reference=True`, each parent contributes one exact record plus
  the number of children actually yielded by its transformation. Record any
  shortfall, including the orthonormal random mode's dimensionality limit.
- Preserve deterministic parent order and deterministic child order. Repeated
  generation with the same ordered parents and transformation configuration
  must reproduce coordinates, labels, identifiers, and output order under the
  documented NumPy/SciPy version scope.
- Define how transformation instances and their `random_state` generators are
  created per parent so fresh, equivalently initialized runs are stable and
  auditable. Document whether a caller-owned generator advances and whether
  inserting an unrelated parent changes later random streams.
- Detect and report exact duplicate displacements within a parent. General
  near-duplicate or descriptor-based selection belongs to Issues 36/37 and is
  not required here.
- Handle near-zero force parents explicitly. They may be retained as exact
  energy records, but derived labels with no meaningful first-order change
  should be skipped or warned about according to a documented tolerance.

### 5. Prevent parent/child split leakage

Make the safe split order part of the public workflow:

1. load and validate exact reference parents;
2. split parents into train, validation, and optional final test groups;
3. augment only the training parents;
4. keep validation and test parents exact and unaugmented; and
5. fit normalization statistics only from the augmented training split.

Provide either a parent-aware split helper or a result object that exposes
parent groups clearly enough that callers cannot accidentally use the generic
entry-level splitter. For HDF5 data, grouping should use persisted
`source_id`/parent metadata rather than filename parsing. Unit tests must prove
that no parent identifier occurs in more than one split.

### 6. Integrate with in-memory and HDF5 training paths

- Allow the augmented structures to feed `StructureDataset` and
  `CachedStructureDataset` without trainer changes.
- Add a source-collection adapter or equivalent streaming bridge so
  `HDF5StructureDataset` can build from force-bearing parent sources and emit
  one exact entry plus deterministic Taylor children for each parent.
- Preserve parent identity and label origin through HDF5 metadata and stable
  structure names. Reopening the database must reproduce the same mapping.
- Support `persist_features=True` for the exact and derived energy-view
  structures. Do not compute or persist local force derivatives for Taylor
  children, and document that `persist_force_derivatives=True` is unnecessary
  for the intended training path.
- Define filtering order: reference validation and any parent-level energy or
  force filters must run before augmentation so a rejected parent cannot leave
  accepted orphan children.
- Keep generation streaming at parent granularity so the full augmented
  dataset need not be materialized in RAM before an HDF5 build.

### 7. Train through the energy-only ANN path

The maintained workflow must train a neural network, not stop after structure
generation:

- use `TorchANNPotential` with `force_weight=0.0` and the existing energy loss;
- keep `sampling_policy="uniform"` in the reference workflow so the requested
  exact-to-approximate data multiple has clear weighting semantics;
- retain each exact parent once when `include_reference=True` and give each
  derived energy the same sample weight in the initial implementation;
- reuse `CachedStructureDataset` or persisted HDF5 features where appropriate;
- verify that the direct `compute_force_loss` path, force graph/triplet
  materialization, and persisted derivative caches are not reached during
  Taylor-only optimization; and
- preserve normal checkpoint, resume, normalization, atomic-reference-energy,
  and model-export behavior from ordinary energy-only training.

Because composition is unchanged by a local displacement, the Taylor energy
correction is compatible with total, cohesive, or formation-energy targets:
the fixed per-species atomic reference contribution cancels between a parent
and its children. This invariant should be documented and tested.

### 8. Evaluate accuracy and the actual bottleneck reduction

Add a controlled comparison using identical parent splits, architecture,
normalization, optimizer settings, epoch budgets, and random seeds:

1. exact-energy-only training on the parent structures;
2. random-displacement Taylor energy-only training;
3. D-optimal-displacement Taylor energy-only training; and
4. direct energy-plus-force training using the existing implementation on a
   small enough subset for the comparison to be practical.

Evaluate all models on the same untouched exact validation/test parents. Use
`predict(..., eval_forces=True)` and report at least:

- energy RMSE or MAE per atom;
- force-component RMSE and force-vector MAE;
- training wall time per epoch and total wall time;
- peak host and, when applicable, accelerator memory; and
- augmentation/build time and stored dataset size, separately from ANN
  optimization time.

Benchmark several RMS displacement magnitudes and augmentation multiples. Use
matched RMS values and child counts when comparing random and D-optimal
sampling, and report the D-optimal optimizer settings, convergence/fallback
behavior, and additional generation time. Select parameters only from
validation results, reserve the final test split for one unbiased comparison,
and report when Taylor noise begins to worsen energy accuracy. The acceptance
target should be a demonstrated force-accuracy improvement over
exact-energy-only training with lower training cost or memory than direct force
supervision for at least one maintained workflow. Do not promise the paper's
numerical speedup or error reduction for this codebase without measuring it.

### 9. Document an executable end-to-end example

Add a tracked notebook, provisionally
`notebooks/example-11-taylor-force-sampling.ipynb`, using a manageable subset
of the force-bearing TiO2 structures under `notebooks/xsf-TiO2/`. It should:

1. load and validate exact energies and forces;
2. split exact parents before augmentation;
3. visualize or summarize the applied displacement and Taylor energy-change
   distributions;
4. generate seeded random and D-optimal Taylor-augmented training sets through
   the existing transformations;
5. train a single ANN with `force_weight=0.0`;
6. train the exact-energy-only baseline with matched settings;
7. optionally run the small direct-force baseline where execution time permits;
8. evaluate energy and force errors on untouched parents; and
9. report timing, memory where available, all seeds, descriptor settings,
   architecture, RMS displacement, augmentation multiple, and D-optimal
   optimizer settings.

Add aligned Sphinx API and usage documentation and a focused profiling script
or extend the existing training profiler. The notebook and documentation must
run from a clean checkout without private data or hidden state.

## Acceptance criteria

- A documented public API generates local Taylor-expanded training records
  from force-bearing reference structures using
  `E_child = E_parent - sum_i delta_R_i dot F_i`.
- The authoritative public API is available from `aenet.geometry.sampling`
  without PyTorch; existing `aenet.torch_training` imports remain tested
  compatibility adapters for PyTorch structures and datasets.
- All child coordinates are generated by `RandomDisplacementTransformation` or
  `DOptimalDisplacementTransformation` from
  `aenet.geometry.transformations`; the Taylor layer contains no independent
  displacement algorithm.
- Missing energies, missing or malformed forces, non-finite labels, invalid
  displacement parameters, and inconsistent atom counts fail with clear
  errors before partial generation.
- Generated children preserve species, atom order, cell, and PBC, own
  independent coordinate arrays, carry finite approximate energies, and do not
  retain parent force labels.
- The random strategy uses the existing independent-random mode for the
  maintained baseline, is reproducible with a fixed `random_state`, removes the
  documented arithmetic-mean translation mode, and achieves the configured RMS
  displacement.
- The D-optimal strategy produces the configured `n_structures`, is
  reproducible under the documented NumPy/SciPy scope, satisfies its RMS,
  translation, and zero-ensemble-mean constraints, and preserves the existing
  random-ensemble fallback when optimization does not improve log-determinant
  diversity.
- The energy correction is calculated from the displacement actually applied
  to each child; analytic tests verify the sign and demonstrate the expected
  second-order truncation error for a smooth test potential.
- Native transformation output counts, any higher-level
  `samples_per_reference` mapping, original-structure inclusion, zero-force
  handling, output ordering, identifier construction, duplicate handling, and
  random-state behavior are documented and tested.
- Exact parents are partitioned before augmentation, or a tested group-aware
  split provides equivalent guarantees; no parent or child family crosses the
  train/validation/test boundary.
- The in-memory output works with `StructureDataset` and
  `CachedStructureDataset`.
- An HDF5 build can stream exact parents and Taylor children, preserve their
  parent/label-origin mapping across reopen, and reuse persisted raw features
  without requiring persisted force derivatives.
- A Taylor-augmented `TorchANNPotential` training run uses
  `force_weight=0.0`; a regression test proves that direct force-loss and
  descriptor-derivative materialization are not invoked during optimization.
- Atomic reference energies, feature/energy normalization, checkpoint/resume,
  and model export retain the documented energy-only behavior.
- Unit tests cover the Taylor primitive, random and D-optimal transformation
  integration, reproducibility, RMS and projection constraints, immutability,
  label clearing, periodic structures, near-zero forces, validation failures,
  stable identifiers, and parent-aware splitting.
- Integration tests cover in-memory augmentation, HDF5 build/reopen, persisted
  energy features, one small energy-only ANN training run on derived labels,
  and force prediction on untouched labeled structures.
- A maintained benchmark compares exact-energy-only, random-Taylor,
  D-optimal-Taylor, and feasible direct-force training with matched splits and
  settings, reporting energy/force error, time, memory, augmentation cost,
  optimizer/fallback status, and dataset size.
- At least one maintained benchmark demonstrates improved held-out force
  accuracy over exact-energy-only training while using less training time or
  peak memory than direct force training.
- Sphinx API/usage documentation and an executable tracked notebook show the
  complete reference-loading, parent-splitting, augmentation, single-ANN
  training, and held-out evaluation workflow.
- The implementation does not require Issue 13, Issue 28, Issue 36, Issue 37,
  or Issue 39 to be completed first.
- No committee, ensemble uncertainty, uncertainty calibration, or active-
  learning selection is implemented as part of this issue.

## Out of scope for the initial implementation

- Ensemble or committee training, uncertainty quantification, uncertainty-
  driven selection, or calibration; these belong to Issue 39.
- Active-learning loops or automated reference-calculation acquisition.
- Running DFT or another electronic-structure method to obtain missing exact
  energies or forces.
- Second- or higher-order Taylor expansions requiring Hessians.
- Signed Cartesian single-atom, normal-mode, or descriptor-selected Taylor
  displacements.
- Automatic selection of a universally optimal displacement or augmentation
  multiple.
- Species-specific displacement limits, selective substructure refinement, or
  force-magnitude-targeted atom selection.
- Per-sample loss weights or a new optimizer objective for approximate labels.
- A general structure-library manifest, geometry search, or command-line
  interface.
- Replacing or removing the existing direct force-training implementation.

## Notes

- Taylor sampling is force-informed energy augmentation, not direct force
  training. The reference forces are used once to construct labels; the ANN
  optimization loss remains energy-only.
- The approach is deliberately approximate. Both energy and force validation
  are required, and a smaller displacement is not automatically better if the
  induced energy changes become negligible relative to numerical noise.
- The original exact reference structure should normally remain in the
  training set so approximate children do not replace the anchor energy.
- A child must never inherit the parent's exact energy or forces unchanged.
  The current coordinate transformations should be audited for this label-
  invalidation invariant wherever they are reused.
- Development should begin on a dedicated Issue 38 branch and be divided into
  local issues covering the label primitive, existing-transformation
  integration, provenance/splitting, HDF5 integration, trainer integration,
  tests, benchmark, and documentation.
