Unified HDF5 Torch Cache Schema
===============================

This page documents the versioned on-disk cache schema used by
``HDF5StructureDataset.build_database(..., persist_features=...,``
``persist_force_derivatives=...)``.

The user-facing training and dataset guides describe when to enable these
cache sections and how they interact with ``cache_features=True`` at runtime.
This page focuses on the on-disk schema, metadata contract, and compatibility
rules behind that workflow.

Scope
-----

Schema version 2 introduces a unified ``/torch_cache`` container for optional
persisted payload sections:

- raw unnormalized descriptor features
- sparse local derivative payloads for force-labeled structures

New cache-writing builds use schema version 2 whenever either optional
payload is requested. Legacy derivative-only schema version 1 files stored
under ``/force_derivatives`` remain readable.

Compatibility Contract
----------------------

Persisted cache compatibility is keyed to the descriptor settings that change
the raw geometry-dependent payloads:

- descriptor class
- species order
- radial order and cutoff
- angular order and cutoff
- minimum cutoff
- whether multi-species/typespin weighting is active

Storage dtype is recorded as metadata, but it is not part of the compatibility
signature. A cache may therefore be written in one floating-point dtype and
loaded through another compatible descriptor dtype, with values cast on load.

Schema Version 2 Layout
-----------------------

The root group is ``/torch_cache``.

Root attributes:

- ``schema_version``: integer schema version, currently ``2``
- ``cache_format``: format identifier string,
  ``"aenet.torch_training.cache.v2"``
- ``descriptor_compat_json``: canonical JSON serialization of the
  compatibility-relevant descriptor settings
- ``descriptor_compat_sha256``: SHA-256 hash of that JSON payload
- ``storage_dtype``: floating-point dtype used for stored arrays
- ``contains_features``: whether the ``/torch_cache/features`` section exists
- ``contains_force_derivatives``: whether the
  ``/torch_cache/force_derivatives`` section exists

Feature Section
---------------

Feature payloads live under ``/torch_cache/features``.

Nodes:

- ``/torch_cache/features/index``
- ``/torch_cache/features/values``

Index columns:

- ``entry_idx``: dataset entry index in ``/entries/structures``
- ``cache_row``: row number used by ``values``
- ``n_atoms``: atom count for the structure
- ``n_features``: raw feature width ``F``

Payload semantics:

- one flattened raw ``(N, F)`` tensor per cached entry in ``values``
- features are stored pre-normalization
- load-time helpers reshape back to ``(N, F)`` and cast to the active
  descriptor dtype

Force-Derivative Section
------------------------

Derivative payloads live under ``/torch_cache/force_derivatives``.

Section attributes:

- ``schema_version``: derivative payload schema version, currently ``1``
- ``payload_format``: format identifier string,
  ``"aenet.torch_training.local_derivatives.v1"``
- ``descriptor_compat_json``
- ``descriptor_compat_sha256``
- ``storage_dtype``
- ``n_radial_features``
- ``n_angular_features``
- ``multi``
- ``contains_features``: currently ``False`` within the derivative subsection
- ``contains_positions``: currently ``False``

Index table:

- ``/torch_cache/force_derivatives/index``
- one row per cached force-labeled structure
- columns:
  - ``entry_idx``
  - ``cache_row``
  - ``n_atoms``
  - ``n_radial_edges``
  - ``n_angular_triplets``

Radial payload nodes:

- ``/torch_cache/force_derivatives/radial/center_idx``
- ``/torch_cache/force_derivatives/radial/neighbor_idx``
- ``/torch_cache/force_derivatives/radial/dG_drij``
- ``/torch_cache/force_derivatives/radial/neighbor_typespin``

Angular payload nodes:

- ``/torch_cache/force_derivatives/angular/center_idx``
- ``/torch_cache/force_derivatives/angular/neighbor_j_idx``
- ``/torch_cache/force_derivatives/angular/neighbor_k_idx``
- ``/torch_cache/force_derivatives/angular/grads_i``
- ``/torch_cache/force_derivatives/angular/grads_j``
- ``/torch_cache/force_derivatives/angular/grads_k``
- ``/torch_cache/force_derivatives/angular/triplet_typespin``

The logical tensor shapes are unchanged from the original derivative cache
design. The v2 schema only relocates the derivative section under the shared
cache root.

Loading Semantics
-----------------

The persistence layer exposes the cache through explicit dataset helpers:

- ``has_persisted_features()``
- ``get_persisted_feature_cache_info()``
- ``load_persisted_features(idx)``
- ``has_persisted_force_derivatives()``
- ``get_force_derivative_cache_info()``
- ``load_persisted_force_derivatives(idx)``

Runtime sample materialization now uses the persisted cache lazily when the
payload is present and descriptor-compatible:

- energy-view materialization checks the trainer-owned runtime
  ``cache_features=True`` cache first, then falls back to persisted HDF5
  features, then finally recomputes features on demand
- force-view materialization reuses persisted raw features when available
- when both persisted raw features and persisted local derivatives are
  available for a force-supervised entry, ``HDF5StructureDataset`` can serve
  the force sample without rebuilding graph/triplet payloads

This keeps feature normalization as a runtime training concern and preserves
on-the-fly fallback behavior when a persisted section is absent.

Legacy Version 1 Compatibility
------------------------------

Legacy derivative-only files with a root ``/force_derivatives`` group remain
supported for read access.

Version 1 characteristics:

- derivative-only layout
- ``schema_version = 1``
- no unified ``/torch_cache`` root
- no persisted raw feature section

New builds do not write schema version 1. They standardize on schema version
2 whenever persisted cache payloads are requested.

Related Descriptor Manifest
---------------------------

When ``persist_descriptor=True`` is requested explicitly, or implicitly via
``persist_features=True`` or ``persist_force_derivatives=True``, the HDF5 file
also stores a versioned descriptor manifest under ``/descriptor_manifest``.

That manifest remains distinct from the cache payload schema and exists only
to reconstruct supported descriptor objects safely when a dataset is reopened.
