Representative Structure Sampling
=================================

Representative sampling selects source structures from an existing candidate
pool by operating on a numeric representation matrix.  The sampling functions
return row indices into the caller's original ordering.  They do not generate
geometries, load structure files, compute descriptors, or return
``AtomicStructure`` objects.

Install the optional sampling dependency before using k-means representative
selection:

.. code-block:: bash

   pip install "aenet[sampling]"

Random baseline sampling only requires NumPy.

Sampling From Feature Matrices
------------------------------

Each row of the representation matrix must describe one source structure:

.. code-block:: python

   import numpy as np

   from aenet.geometry.sampling import (
       random_subset,
       representative_subset,
   )

   features = np.array([
       [0.0, 0.1, 1.2],
       [0.2, 0.0, 1.1],
       [8.0, 3.5, 0.2],
       [8.2, 3.4, 0.1],
   ])

   representative_indices = representative_subset(
       features,
       num_structures=2,
       random_state=42,
   )
   random_indices = random_subset(
       features,
       num_structures=2,
       random_state=42,
   )

   selected_features = features[representative_indices]

Use the returned indices with the source collection that produced the rows:

.. code-block:: python

   selected_paths = [structure_paths[i] for i in representative_indices]

The returned index arrays are sorted into ascending source order.  This keeps
file lists, dataset slices, and downstream manifests aligned with the original
candidate ordering.

Why Scaling Is Explicit
-----------------------

``representative_subset`` uses k-means clustering, which is based on Euclidean
distance.  Feature scaling therefore changes which structures are selected.
The function does not standardize or otherwise transform features internally.
Callers should make preprocessing choices explicit:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   scaled_features = StandardScaler().fit_transform(features)
   representative_indices = representative_subset(
       scaled_features,
       num_structures=100,
       random_state=42,
   )

This avoids hidden double-scaling when features have already been transformed
and allows domain-specific weighting when some descriptor components should
matter more than others.

Connecting Global Moment Fingerprints
-------------------------------------

The sampling API is independent of how the representations are produced.  A
typical workflow is:

1. Build or open a training-set-like source with one entry per structure.
2. Compute one global moment fingerprint per structure.
3. Stack those fingerprints into a two-dimensional matrix.
4. Standardize the matrix explicitly.
5. Sample representative and random subsets.
6. Apply the returned indices to the original structure paths or dataset rows.

For example:

.. code-block:: python

   import numpy as np
   from sklearn.preprocessing import StandardScaler

   from aenet.geometry.sampling import (
       random_subset,
       representative_subset,
   )

   fingerprints = [
       structure.global_moment_fingerprint(
           outer_moment=1,
           inner_moment=1,
           weighted=True,
           append_weighted=True,
       )
       for structure in trainset
   ]
   features = np.vstack(fingerprints)
   scaled_features = StandardScaler().fit_transform(features)

   representative_indices = representative_subset(
       scaled_features,
       num_structures=100,
       random_state=42,
   )
   random_indices = random_subset(
       scaled_features,
       num_structures=100,
       random_state=42,
   )

   representative_paths = [
       structure_paths[i] for i in representative_indices
   ]
   random_paths = [structure_paths[i] for i in random_indices]

``representative_subset`` chooses the observed row nearest each populated
k-means centroid.  It returns actual source indices, not centroid vectors or
synthetic structures.

Boundary And Degenerate Cases
-----------------------------

If ``num_structures`` equals the number of input rows, both samplers return all
source indices.  No down-selection is needed in that case.

If k-means cannot produce the requested number of populated clusters, for
example because many representation rows are identical, representative
sampling raises ``ValueError``.  Reduce ``num_structures`` or provide less
degenerate representations.

Maintained NaCl Example
-----------------------

``notebooks/example-09-sampled-structures-downselection.ipynb`` demonstrates the
complete workflow on UMA-generated NaCl snapshots.  Its tracked data bundle is
stored under ``notebooks/data/NaCl-sampled-structures/`` and contains:

- one XZ-compressed archive with 20,000 XSF structures, split evenly among
  550 K, 700 K, 850 K, and 1000 K;
- the starting VASP structure and the scripts used for UMA molecular dynamics,
  trajectory conversion, and Chebyshev cutoff analysis; and
- the cutoff-analysis report and plots supporting ``rad_cutoff=4.8`` Angstrom
  and ``ang_cutoff=3.75`` Angstrom.

The notebook executes on a deterministic 100-structure slice with 25 snapshots
per temperature by default, so it remains useful on a workstation and in CI.
Set ``NUM_CANDIDATES=None`` and
``NUM_SELECTED=2000`` in its configuration cell to reproduce the full
20,000-to-2,000 down-selection.  A precomputed ``.npz`` feature file from an
HPC run can be assigned to ``FEATURE_FILE`` to skip local featurization while
retaining the same scaling, sampling, PCA, and t-SNE analysis.
