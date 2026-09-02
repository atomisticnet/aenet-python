Force-Informed Taylor Sampling
==============================

Taylor sampling converts forces on an exact reference configuration into
approximate energies at nearby configurations,

.. math::

   E(R + \Delta R) \approx E(R) - \sum_i \Delta R_i \cdot F_i(R).

The authoritative API is :mod:`aenet.geometry.sampling`. It uses NumPy and
:class:`~aenet.geometry.AtomicStructure` and does not require PyTorch. Positions
and displacements are in Angstrom, energies in eV, and forces in eV/Angstrom.
The package cannot infer or convert units supplied in arbitrary arrays.

Neutral workflow
----------------

Give every exact, single-frame reference a stable identity. Identity controls
provenance and the independent random stream; it must not be derived from the
parent's current collection position.

.. doctest::

   >>> import numpy as np
   >>> from aenet.geometry import AtomicStructure
   >>> from aenet.geometry.sampling import (
   ...     TaylorExpansionConfig,
   ...     TaylorReference,
   ...     generate_taylor_samples,
   ... )
   >>> from aenet.geometry.transformations import RandomDisplacementTransformation
   >>> parent = AtomicStructure(
   ...     coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
   ...     types=["H", "H"],
   ...     energy=-1.0,
   ...     forces=[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]],
   ... )
   >>> reference = TaylorReference("calculation-0042", parent)
   >>> config = TaylorExpansionConfig(
   ...     RandomDisplacementTransformation(
   ...         rms=0.01,
   ...         max_structures=2,
   ...         random_state=7,
   ...         remove_translations=True,
   ...     )
   ... )
   >>> result = generate_taylor_samples([reference], config)
   >>> (result.n_exact, result.n_derived, result.n_skipped)
   (1, 2, 0)
   >>> [record.label_origin for record in result.records]
   ['exact', 'taylor', 'taylor']
   >>> len(result.records[1].structure.forces[0])
   0

Inputs are not mutated, and every output owns its coordinate storage. Exact
records retain exact force labels; approximate children deliberately have no
force label. Labels use the actual transformed coordinates after RMS scaling
and constraint projection. ``displacement_rms`` is the root mean square over
all ``3N`` Cartesian components, while ``maximum_displacement`` is the largest
per-atom vector norm.

Random-state and transformation policies
----------------------------------------

Both :class:`~aenet.geometry.transformations.RandomDisplacementTransformation`
and :class:`~aenet.geometry.transformations.DOptimalDisplacementTransformation`
are supported. A configuration copies its transformation prototype. Generation
does not advance the caller's generator. Per-parent streams depend on the
prototype state, optional namespace, and stable parent ID, so inserting or
reordering unrelated parents does not change an existing family. Exact
reproduction is scoped to compatible NumPy, SciPy, and transformation versions.

The maintained random baseline uses ``orthonormalize=False``. Orthonormal mode
can return fewer children than requested when the constrained displacement
space is too small; ``unavailable_children`` records that shortfall.
``remove_translations=True`` subtracts the arithmetic mean displacement, not a
mass-weighted center of mass. It therefore leaves no internal displacement for
a one-atom reference.

``zero_force_policy`` and ``duplicate_policy`` accept ``"skip"``, ``"keep"``,
or ``"error"``. Their tolerances apply to the full force-array norm and
Cartesian displacement comparison, respectively. The result reports skipped
causes separately.

Split before augmentation
-------------------------

Always split exact parents first, then augment only the training references.
This prevents a parent and its nearly identical children from crossing into
validation or test data.

.. code-block:: python

   from aenet.geometry.sampling import split_reference_structures

   train_refs, validation_refs, test_refs = split_reference_structures(
       references,
       validation_fraction=0.1,
       test_fraction=0.1,
       seed=19,
   )
   augmented_train = generate_taylor_samples(train_refs, config).structures

Keep validation and test references exact and untouched when evaluating both
energy and force accuracy. First-order labels are approximate: displacement
size and replication count must be selected using validation data.

PyTorch compatibility adapter
-----------------------------

Existing imports from :mod:`aenet.torch_training` remain supported. They are
thin adapters returning :class:`~aenet.torch_training.Structure` objects. Bare
PyTorch structures require unique non-empty ``name`` values or explicit
``parent_ids``:

.. code-block:: python

   from aenet.torch_training import generate_taylor_samples

   result = generate_taylor_samples(
       torch_parents,
       config,
       parent_ids=["run-17/frame-0", "run-17/frame-1"],
   )

These compatibility imports require the optional PyTorch dependencies. Use the
geometry namespace for backend-neutral workflows.

HDF5 parent filtering and provenance
------------------------------------

Wrap sources with
:class:`~aenet.torch_training.taylor_sampling.TaylorSourceCollection` before
building an :class:`~aenet.torch_training.HDF5StructureDataset`. Dataset
``max_energy`` and ``max_forces`` filters are evaluated on each exact parent
before augmentation. A rejected parent contributes no entries; an accepted
parent keeps its complete family even if an approximate child independently
crosses the energy threshold.

.. code-block:: python

   from aenet.torch_training import HDF5StructureDataset, TaylorSourceCollection

   sources = TaylorSourceCollection(parent_sources, config)
   dataset = HDF5StructureDataset(
       descriptor,
       "taylor-training.h5",
       sources=sources,
       mode="build",
       max_energy=1.0,
       max_forces=20.0,
       atomic_energies=atomic_energies,
   )
   dataset.build_database(
       persist_features=True,
       persist_force_derivatives=False,
   )
   metadata = dataset.get_entry_metadata(0)
   print(metadata["taylor_parent_id"], metadata["taylor_label_origin"])

The versioned metadata stores ``taylor_parent_id``, ``taylor_child_index``,
``taylor_strategy``, ``taylor_label_origin``, ``taylor_generation_id``, and
``source_frame_idx`` without parsing structure names. Older and non-Taylor
databases return ``None`` for Taylor fields.

Energy-only training
--------------------

Taylor children are ordinary energy-labelled structures. Train them with
``force_weight=0.0`` and normally ``sampling_policy="uniform"``. Persisted raw
features are supported; force derivatives are unnecessary for children and are
not generated because they have no force labels. Exact parents may still be
used afterward for force evaluation with ``predict(..., eval_forces=True)``.

Atomic reference energies are compatible with Taylor corrections because a
local displacement does not change composition: the same per-species reference
contribution is subtracted from a parent and all of its children.
