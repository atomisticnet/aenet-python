# Issue 37: Implement representative and random structure sampling

**Type:** Feature
**Priority:** Medium
**Status:** Open
**Created:** 2026-08-28

## Problem

Users assembling reference or training datasets often need to select a fixed-
size subset from a larger pool while retaining broad coverage of the sampled
configuration space. The package can produce global moment fingerprints from
local atomic fingerprints, but it does not provide a supported API for using
those representations to sample a representative subset of structures.

The intended methodology is to cluster global structure representations with
k-means, using the requested subset size as the number of clusters, and select
the observed structure nearest to each cluster centroid. Users currently have
to implement this orchestration themselves, including input validation,
reproducibility controls, and conversion from centroids back to unique source
indices.

Users also need a simple reproducible random subset as a baseline for
comparison and for workflows that do not require feature-space coverage. The
package does not currently provide a companion function with the same
index-based sampling semantics.

## Current evidence

`TrnSet` structures expose
`global_moment_fingerprint(...)`, which reduces per-atom fingerprints to a
fixed-length NumPy representation. The existing public APIs support the
following complete route to the representation matrix required by this issue:

1. Load structure paths from a tracked source collection, such as
   `notebooks/xsf-TiO2/`.
2. Configure `aenet.torch_featurize.ChebyshevDescriptor` for the dataset's
   species. A concrete Ti--O example can use `species=["Ti", "O"]`,
   `rad_order=10`, `rad_cutoff=6.0`, `ang_order=3`, `ang_cutoff=3.5`,
   `min_cutoff=0.5`, `device="cpu"`, and `dtype=torch.float64`.
3. Build an `aenet.torch_training.dataset.HDF5StructureDataset` from those
   paths with the descriptor and persist the atomic features. Reference atomic
   energies may be derived with `aenet.reference_energies.ReferenceEnergies`
   when required by the dataset-building workflow.
4. Reopen the HDF5 file with `aenet.trainset.TrnSet`, iterate over its
   structures, and call `global_moment_fingerprint(...)` for each entry. A
   concrete initial configuration is `outer_moment=1`, `inner_moment=1`,
   `weighted=True`, and `append_weighted=True`.
5. Stack the resulting one-dimensional fingerprints in source order to form a
   two-dimensional NumPy feature matrix with one row per structure.
6. Fit `sklearn.preprocessing.StandardScaler` to that matrix and pass the
   transformed matrix to `representative_subset(...)`. Pass the same matrix to
   `random_subset(...)` when constructing an equal-size random baseline.
7. Apply the returned indices to the ordered source paths or dataset only when
   the selected structures are needed; the structures do not need to be kept
   in memory during sampling.

The selection functions should consume a general numeric representation
matrix. They must not require the concrete descriptor, global-moment settings,
HDF5 storage, or PyTorch backend used by this example.

Issue 36 separately proposes an end-to-end structure-library generation
workflow. It identifies descriptor-based selection as an optional extension to
that workflow. This issue defines the reusable down-selection operation itself
and should remain usable independently of structure generation.

## Impact

A supported sampling method would:

- provide a concise and reproducible route from global representations to a
  fixed-size subset that covers the represented feature space;
- avoid one-off implementations that accidentally select the same source
  structure for more than one centroid or lose the mapping to the input data;
- work with in-memory structures, `TrnSet`, HDF5 datasets, and other
  caller-managed collections;
- make descriptor-space selection available to the future structure-library
  workflow without adding scikit-learn or PyTorch to its core requirements;
- provide a reproducible random baseline without requiring feature computation
  or scikit-learn.

## Proposed approach

Add the representative-selection operation to the provisional
`aenet.geometry.sampling` umbrella described by Issue 36. This namespace can
connect the related workflows of generating candidate geometries and sampling
representatives from an existing, featurized candidate set. The API should
remain centered on a numeric feature matrix:

```python
from aenet.geometry.sampling import representative_subset

subset_indices = representative_subset(
    standardized_features,
    num_structures=100,
    random_state=42,
)
selected_paths = [structure_paths[i] for i in subset_indices]
```

Provide a companion random sampler in the same module:

```python
from aenet.geometry.sampling import random_subset

random_indices = random_subset(
    standardized_features,
    num_structures=100,
    random_state=42,
)
random_paths = [structure_paths[i] for i in random_indices]
```

`random_subset` should sample uniformly without replacement and return unique
indices into the supplied representations. It only needs their row count and
should not inspect the representation values. Using the same representation-
first calling convention for both functions keeps the API consistent while
remaining independent of structure and dataset types. It should use NumPy's
random-number facilities and must not require scikit-learn or PyTorch.

Despite its location in the geometry package, `representative_subset` should
not generate, copy, or return geometry objects. It samples rows in the supplied
feature matrix and returns indices into the original row order. The caller is
responsible for applying those indices to its structures or dataset. This
distinction must be explicit in the function name, docstring, return type, and
examples so it is not confused with Issue 36's geometry-generation operations.

Both functions should accept representations only. Keeping structure
storage separate from sampling makes the functions independent of a
particular dataset or structure class and allows callers to apply the
returned indices to on-disk, lazy, or otherwise externally managed
datasets.

The first implementation should:

1. Accept a two-dimensional, finite, real-valued array-like object with one row
   per structure and a positive integer `num_structures`.
2. Fit scikit-learn k-means with `n_clusters=num_structures` and documented
   reproducibility controls, including `random_state` and an explicit or
   documented `n_init` policy.
3. For each cluster, choose the assigned member closest to that cluster's
   centroid. Restricting the search to cluster membership guarantees one
   selected source row per non-empty cluster and therefore unique indices.
4. Return indices into the original row order with a documented integer
   container type and deterministic ordering rule, such as cluster-label order
   or sorted source order.
5. Import scikit-learn only when the sampling function is called and raise a
   clear `ImportError` with an installation hint when it is unavailable. Add a
   named project extra, provisionally `aenet[sampling]`, rather than making
   scikit-learn a required dependency.
6. Implement `random_subset` over the rows of the supplied representations,
   using only the population length, with the same subset-size validation,
   documented reproducibility guarantees, deterministic result-ordering
   policy, and index-returning semantics as `representative_subset`, where
   applicable.

The design should explicitly resolve the following points before
implementation:

- whether feature scaling should remain caller-controlled, as proposed for the
  initial API, or be exposed through an explicit option. K-means uses Euclidean
  distance, so scaling changes the selected structures and must not happen
  invisibly;
- the public behavior when `num_structures` equals the number of samples;
- whether both functions return sampled indices in draw/cluster order or sort
  them into source order, and how that choice supports consistent downstream
  behavior;
- which random-state inputs are supported, such as integer seeds and NumPy
  generators, and whether either function mutates caller-provided generator
  state;
- validation and error messages for zero samples, non-finite values,
  inconsistent row lengths, non-integer counts, and requests larger than the
  input pool;
- behavior when duplicate feature rows or degenerate clusters prevent the
  fitted model from producing the requested number of distinct populated
  clusters; and
- whether optional diagnostics such as cluster labels, centroid distances, or
  inertia warrant a separate result object without complicating the common
  index-only use case.

The sampling API should not compute atomic descriptors or standardize features
itself in the initial implementation. Featurization, preprocessing, and
sampling have different data-source and policy concerns. Documentation should
show how to connect `global_moment_fingerprint(...)`, an explicit
scikit-learn `StandardScaler`, and `representative_subset(...)`.

Add a tracked example notebook,
`notebooks/example-09-sampled-structures-downselection.ipynb`, that runs from a
clean checkout using the tracked NaCl archive under
`notebooks/data/NaCl-sampled-structures/`. The notebook may use the PyTorch
featurization framework; it does not need to exercise the compiled Fortran
backend. It should demonstrate the complete workflow:

1. load and featurize the structures;
2. calculate one global moment representation per structure;
3. standardize the feature matrix, explaining why standardization matters for
   Euclidean k-means;
4. request a reproducible representative subset;
5. request an equal-size reproducible random subset as a baseline;
6. apply both sets of returned indices to the source structures; and
7. summarize or visualize how representative and random sampling cover the
   feature space relative to the full dataset.

The notebook should use fixed seeds, expose all relevant descriptor and
scaling settings, state the optional extras needed to run it, avoid private
inputs and hidden execution state, and be covered by an appropriate execution
check.

## Acceptance criteria

- A documented public `representative_subset` function selects exactly
  `num_structures` unique input rows by fitting k-means and choosing the member
  nearest each populated centroid.
- A documented public `random_subset` function selects exactly
  `num_structures` unique indices uniformly without replacement from a
  supplied collection of representations.
- The primary API accepts a two-dimensional feature matrix and returns indices
  that preserve an unambiguous mapping to the caller's input ordering.
- Neither function accepts `(structure, representation)` pairs or requires
  structures to be materialized in memory; both operate on representations and
  return indices for the caller to apply separately.
- The API documentation makes clear that `representative_subset` returns
  source-row indices and neither generates nor returns atomic geometries,
  despite residing under `aenet.geometry.sampling`.
- The API documentation makes clear that `random_subset` also returns source
  indices rather than structure or geometry objects.
- The sampling functions do not require `AtomicStructure`, `TrnSet`, HDF5,
  pandas, or PyTorch inputs.
- Repeated calls with the same features and documented reproducibility
  arguments return the same indices under the supported scikit-learn version
  scope.
- Repeated `random_subset` calls with the same representation count, subset
  size, and random-state argument return the same indices under the documented
  NumPy version scope.
- The feature-scaling policy, distance metric, tie-breaking behavior, result
  ordering, and degenerate-cluster behavior are documented and tested.
- Invalid population or subset sizes, malformed feature matrices, and
  non-finite values fail with clear exceptions.
- scikit-learn is an optional dependency exposed through a documented project
  extra; importing core `aenet` functionality continues to work without it.
- Calling `representative_subset` without scikit-learn raises a clear
  `ImportError` that includes the installation command.
- `random_subset` remains available and functional when scikit-learn and
  PyTorch are unavailable.
- Unit tests cover representative clusters, uniform random sampling without
  replacement, uniqueness, reproducibility, boundary sizes, ties or duplicate
  rows, validation failures, and missing optional dependency behavior.
- A tracked notebook under `notebooks/` runs from a clean checkout with tracked
  inputs and demonstrates featurization, global moment representations,
  explicit feature standardization, reproducible representative and random
  sampling, comparison of their feature-space coverage, and application of the
  returned indices to the source structures.
- The notebook uses the PyTorch featurization framework with documented
  optional dependencies; no equivalent Fortran-backend notebook is required.
- The notebook explains the effect of feature scaling on Euclidean k-means,
  records all descriptor, scaler, and random-seed settings, and is covered by
  an appropriate execution check.
- Sphinx API and user documentation explain the same end-to-end workflow and
  link to or align with the maintained notebook.
- The implementation does not introduce a required PyTorch dependency.

## Out of scope for the initial implementation

- Computing atomic or global descriptors inside `representative_subset`.
- Implicitly standardizing or otherwise transforming caller-provided features.
- Automatically reading or writing structure datasets.
- Active learning or model-uncertainty-based selection.
- Selecting an optimal subset size for the user.
- Requiring integration with the Issue 36 structure-library workflow.
- Alternative clustering or coverage algorithms beyond k-means medoid-like
  representative selection.

## Notes

- The returned structures are observed samples nearest to centroids, not the
  synthetic centroid vectors themselves; this distinction should be explicit
  in the documentation.
- Prefer the terms **representative sampling**, **representative subset**, or
  **feature-space coverage** for the public API and its guarantees. “Diversity”
  may describe the motivation informally, but the method should not be
  described as solving a formal global maximum-diversity objective.
- Implementation should begin on a dedicated issue branch and be divided into
  local issues after the API decisions above have been agreed.
