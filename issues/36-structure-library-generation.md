# Issue 36: Provide an end-to-end structure-library generation workflow

**Type:** Feature  
**Priority:** Medium  
**Status:** Open  
**Created:** 2026-08-28

## Problem

The structure-transformation framework provides useful low-level operations for
perturbing a reference structure, but it does not yet provide a cohesive
workflow for turning an ideal or ground-state structure into a reproducible,
validated library of candidate structures for reference calculations and MLIP
training.

A user can currently instantiate transformations, iterate over their outputs,
and write files manually. The user must still design the sampling branches,
control their combined size, decide whether to include the reference structure,
validate generated geometries, remove duplicates, assign stable names, and
record enough provenance to reproduce the library. These responsibilities are
easy to implement inconsistently and become especially important when the
library will be submitted to expensive electronic-structure calculations.

## Current evidence

The public API under `aenet.geometry.transformations` currently includes:

- the `TransformationABC` protocol and sequential `TransformationChain`;
- deterministic atomic displacement via
  `AtomDisplacementTransformation`;
- stochastic atomic sampling via `RandomDisplacementTransformation` and
  `DOptimalDisplacementTransformation`;
- cell scaling and basis changes via `CellVolumeTransformation` and
  `CellTransformationMatrix`; and
- isovolumetric, uniaxial, shear, orthorhombic, and monoclinic strain paths.

All transformations lazily yield `AtomicStructure` instances from one input
structure. `TransformationChain` implements sequential, depth-first
composition, so output counts multiply across steps. There is no corresponding
high-level abstraction for taking the union of independent sampling branches,
which is a common structure-library workflow.

The documentation already includes an API reference and basic and advanced
transformation guides. The basic examples are covered by executable tests, and
the current transformation test suite and Sphinx doctests pass. The advanced
guide demonstrates manual iteration, chaining, output limiting, and file
writing, but it does not define a durable library-generation workflow or cover
validation, deduplication, provenance, or manifests.

An exploratory sampling workflow suggests two complementary categories that an
end-to-end API should represent explicitly:

- **cell-space sampling**, in which independent branches apply volume,
  isovolumetric, uniaxial, shear, orthorhombic, or monoclinic deformations to
  the same reference structure; and
- **local coordinate sampling**, in which random, orthonormal random, or
  D-optimal displacement patterns perturb the atoms around a fixed reference
  cell.

Those branches can be compared after featurization using per-atom descriptors
or global moment fingerprints. Standardized descriptor vectors and diversity
statistics can help assess whether two sampling strategies provide redundant
or complementary configuration-space coverage. This analysis is useful for
choosing a recipe, but it currently requires one-off orchestration and does not
itself produce a validated, reproducible structure library. The core workflow
should therefore retain transformation and branch identity in its manifest so
that optional downstream descriptor-space analysis can group and compare the
generated samples without relying on naming conventions.

Two related consistency questions should be resolved as part of the work:

- `RandomDisplacementTransformation` and
  `DOptimalDisplacementTransformation` calculate RMS over all Cartesian
  components, while the advanced guide defines a per-atom vector RMS. These
  conventions differ by a factor of `sqrt(3)`.
- `CellTransformationMatrix` should be audited for the same stale energy/force
  label invalidation guarantees documented and implemented by the other cell
  transformations.

## Impact

Without a supported end-to-end workflow:

- users must write substantial one-off orchestration code around otherwise
  reusable transformations;
- sequential composition can be confused with combining independent sampling
  families, causing accidental combinatorial growth;
- malformed, unphysical, or duplicate geometries may consume expensive
  reference-calculation resources;
- generated datasets may lack sufficient provenance for reproduction or
  auditing; and
- examples can demonstrate transformations without showing how to produce a
  practical calculation-ready library.

## Proposed approach

Design a higher-level structure-library generation API around the existing
iterator-based transformations. Decide whether this belongs in a new module
such as `aenet.geometry.sampling` or alongside the transformation framework
before fixing the public interface.

The design should define:

1. **Sampling recipes and composition**
   - Accept one or more reference structures and named sampling branches.
   - Distinguish explicitly between sequential transformation chains and the
     union of independent branches.
   - Allow inclusion of each unmodified reference structure.
   - Provide predictable per-branch and total output limits without requiring
     full materialization in memory.

2. **Reproducibility and provenance**
   - Assign stable sample identifiers and deterministic output names.
   - Record source identity, transformation path, parameters, branch name,
     random seed or stream information, and relevant package versions.
   - Define random-number behavior across transformations, branches, repeated
     generation, and resumed runs.
   - Prefer a serializable recipe representation if it can be introduced
     without prematurely committing to a large configuration framework.

3. **Validation and selection**
   - Support built-in validation of finite coordinates, periodic-cell
     validity, composition and atom count, and configurable minimum
     interatomic distances.
   - Allow user-defined validators or filters.
   - Define duplicate and near-duplicate handling, with a low-cost geometric
     baseline and optional descriptor-based selection kept separate from the
     core when it requires heavier dependencies.
   - Preserve rejection reasons in the generation record.

4. **Output and manifests**
   - Stream accepted structures to a chosen supported format and output
     directory.
   - Write a machine-readable manifest containing sample provenance,
     validation outcomes, output paths, and a summary of requested, accepted,
     rejected, and duplicate structures.
   - Define collision, overwrite, partial-failure, and resume behavior so an
     existing library is not silently corrupted.
   - Keep generation independent of PyTorch wherever practical; descriptor-
     based analysis or selection may be an optional extension.

5. **Documentation and examples**
   - Add an end-to-end guide beginning with an ideal reference structure and
     ending with a calculation-ready structure directory and manifest.
   - Explain parameter choices, physical validity checks, sequential versus
     independent composition, output-size estimates, and reproducibility.
   - Include a maintained notebook under `notebooks/` that loads one of the
     tracked structures in `notebooks/xsf-TiO2/`, constructs independent
     cell-space and local-coordinate sampling branches, writes the accepted
     structures, and uses manifest metadata to summarize coverage by branch.
   - Keep featurization, visualization, and descriptor-diversity analysis as an
     optional follow-on example so the core workflow remains lightweight.

A first implementation should prioritize a small composable Python API and a
tested manifest format. A command-line interface can be evaluated separately
after the Python semantics are stable rather than being required up front.

## Acceptance criteria

- A documented public Python API can generate a structure library from at
  least one reference `AtomicStructure` and multiple named sampling branches.
- Independent branch union and sequential transformation composition have
  distinct, tested semantics and predictable output counts.
- The workflow can include the original reference structure and enforce
  configurable per-branch and total limits while streaming.
- Seeded stochastic recipes reproduce the same structures, sample identifiers,
  ordering, and manifest content under the documented reproducibility scope.
- Built-in validators cover finite coordinates, valid periodic cells,
  composition/atom-count preservation, and minimum interatomic distance;
  rejected samples retain machine-readable reasons.
- Exact duplicates are handled deterministically, and the supported scope and
  limitations of near-duplicate detection are documented.
- Accepted structures are written with stable names and accompanied by a
  machine-readable provenance manifest and generation summary.
- Existing-output, partial-failure, and resume or restart behavior is explicit,
  tested, and avoids silent overwrites.
- RMS displacement terminology and implementation use one clearly documented
  convention, with compatibility implications addressed.
- Cell basis transformations follow documented energy/force label validity
  rules.
- Unit and integration tests cover composition, reproducibility, validation,
  deduplication, output/manifest behavior, and failure cleanup.
- User and API documentation include an executable end-to-end example that
  starts from an ideal structure and produces a calculation-ready library.
- A tracked notebook under `notebooks/` demonstrates the complete workflow
  using a tracked input such as `notebooks/xsf-TiO2/structure-001.xsf`. It runs
  from a clean checkout without private inputs or hidden execution state,
  records fixed seeds where randomness is used, and verifies or clearly
  summarizes the generated files and manifest.
- The implementation does not introduce a required PyTorch dependency.

## Out of scope for the initial implementation

- Running or scheduling electronic-structure calculations.
- Computing reference energies and forces for generated structures.
- Active-learning loops driven by model uncertainty.
- A comprehensive structure-search or phase-discovery framework.
- Requiring descriptor-space selection, PyTorch, or scikit-learn in the core
  generation path.

## Notes

- The existing transformation classes should remain useful independently; the
  new workflow should orchestrate them rather than duplicate their geometry
  logic.
- Descriptor diversity metrics and D-optimal displacements are useful optional
  selection tools, but geometric validity and provenance are prerequisites for
  a dependable baseline workflow.
- A useful optional analysis should compare random, orthonormal-random, and
  D-optimal displacement ensembles at equal sample counts and displacement
  magnitudes. Any comparison must use consistent descriptor standardization
  and report its seed and featurizer settings.
- Implementation should begin on a dedicated issue branch and be divided into
  local issues once the API and manifest design have been agreed.
