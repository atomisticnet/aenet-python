"""
HDF5-backed Dataset and generic dataset utilities for torch training.

This module provides:
- HDF5StructureDataset: a database-backed lazy-loading PyTorch Dataset
  that stores serialized (pickled) torch Structure objects in an HDF5
  (PyTables) file with per-entry metadata. It supports building the database
  from source collections that load one or more structures per logical input,
  and efficient read-only access during training with multiprocessing.

- train_test_split_dataset: a generic dataset splitter that returns PyTorch
  Subset instances for training and test sets.

Design goals
------------
- Scale to very large datasets (10M+ structures) with minimal RAM usage.
- Avoid re-parsing raw structure files repeatedly by serializing Structures
  into a compressed HDF5 file (VLArray of uint8 per entry).
- Preserve training-time behavior and sample format identical to
  StructureDataset, enabling drop-in replacement in the trainer.
- Multiprocessing safe: each worker opens its own read-only HDF5 handle.

Notes
-----
- Uses PyTables (tables) which is already a project dependency.
- Serialization uses Python pickle; compression is handled by HDF5 filters.
- For extremely large datasets, consider building the database once, then
  distributing/read-only mounting on compute nodes.
- The database can optionally persist raw feature payloads and sparse local
  derivative payloads under a versioned ``/torch_cache`` schema.
- Legacy derivative-only ``/force_derivatives`` files remain readable.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import tables  # PyTables
import torch
from torch.utils.data import Dataset, Subset

from ._materialization import (
    build_force_graph_triplets,
    build_sample_dict,
    extract_runtime_caches,
    forward_force_features_with_graph,
    load_energy_view_features,
    materialize_force_view,
    prepare_structure_tensors,
    referenced_energy_per_atom,
)
from .config import Structure  # Torch Structure dataclass
from .descriptor_manifest import (
    DESCRIPTOR_MANIFEST_FORMAT,
    DESCRIPTOR_MANIFEST_SCHEMA_VERSION,
    descriptor_compat_signature_from_object,
    descriptor_from_manifest,
    descriptor_manifest_from_object,
    descriptor_matches_manifest,
)
from .sources import (
    SourceCollection,
    SourceRecord,
    _ChunkedSourceCollection,
    coerce_source_collection,
)

__all__ = [
    "HDF5StructureDataset",
    "train_test_split_dataset",
]


_build_force_graph_triplets = build_force_graph_triplets
_forward_force_features_with_graph = forward_force_features_with_graph
_SOURCE_ID_MAX_BYTES = 2048
_SOURCE_KIND_MAX_BYTES = 64
_DISPLAY_NAME_MAX_BYTES = 2048
_STRUCTURE_NAME_MAX_BYTES = 1024


def _descriptor_cache_signature(descriptor) -> dict:
    """
    Serialize the descriptor settings relevant to persisted cache reuse.

    Runtime placement details such as device and tensor dtype are excluded.
    Those affect storage and load-time casting, but not the underlying
    geometry-dependent derivative values.
    """
    return descriptor_compat_signature_from_object(descriptor)


def _descriptor_derivative_cache_metadata(descriptor) -> dict:
    """Build HDF5 metadata for persisted feature/derivative caches."""
    compat = _descriptor_cache_signature(descriptor)
    compat_json = json.dumps(compat, sort_keys=True, separators=(",", ":"))
    return {
        "compat": compat,
        "compat_json": compat_json,
        "compat_sha256": hashlib.sha256(
            compat_json.encode("utf-8")
        ).hexdigest(),
        "storage_dtype": str(getattr(descriptor, "dtype", "")).replace(
            "torch.", ""
        ),
        "n_radial_features": int(getattr(descriptor, "rad_order", 0)) + 1,
        "n_angular_features": int(getattr(descriptor, "ang_order", 0)) + 1,
        "multi": bool(getattr(descriptor, "multi", False)),
    }


def _tables_float_atom(dtype: torch.dtype) -> tables.Atom:
    """Return the PyTables atom matching the requested torch float dtype."""
    if dtype == torch.float32:
        return tables.Float32Atom()
    if dtype == torch.float64:
        return tables.Float64Atom()
    raise TypeError(
        "Persisted force-derivative caches require float32 or float64 "
        f"descriptor dtypes, got {dtype!r}."
    )


def _tensor_to_numpy_1d(
    value: torch.Tensor | None,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """Convert an optional tensor to a flattened NumPy array."""
    if value is None:
        return np.empty(0, dtype=dtype)
    tensor = value.detach().cpu().contiguous().view(-1)
    return tensor.numpy().astype(dtype, copy=False)


def _structure_energy_or_nan(struct: Structure) -> float:
    """Return the structure energy or ``nan`` if it is unavailable."""
    try:
        return float(struct.energy)
    except Exception:
        return float("nan")


def _should_keep_structure_for_build(
    struct: Structure,
    *,
    max_referenced_energy_per_atom: float | None,
    atomic_energies: dict[str, float] | None,
) -> bool:
    """Return whether a structure passes HDF5 build-time energy filtering."""
    if max_referenced_energy_per_atom is None:
        return True
    return referenced_energy_per_atom(
        struct,
        atomic_energies=atomic_energies,
    ) <= float(max_referenced_energy_per_atom)


def _write_energy_filter_metadata(
    h5: tables.File,
    *,
    max_referenced_energy_per_atom: float | None,
    atomic_energies: dict[str, float] | None,
) -> None:
    """Persist HDF5 build-time energy-filter semantics for auditability."""
    attrs = h5.root._v_attrs
    attrs.energy_filter_semantics = "referenced_energy_per_atom"
    attrs.energy_filter_enabled = bool(
        max_referenced_energy_per_atom is not None
    )
    attrs.energy_filter_reference_mode = (
        "explicit_atomic_references"
        if atomic_energies is not None
        else "zero_reference_fallback"
    )
    attrs.energy_filter_max_referenced_energy_per_atom = (
        float(max_referenced_energy_per_atom)
        if max_referenced_energy_per_atom is not None
        else float("nan")
    )
    attrs.energy_filter_atomic_energies_json = (
        json.dumps(atomic_energies, sort_keys=True, separators=(",", ":"))
        if atomic_energies is not None
        else ""
    )


def _decode_meta_text(value: object) -> str:
    """Normalize persisted HDF5 string metadata to a plain Python string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    return str(value).rstrip("\x00")


def _validated_hdf5_text(
    *,
    field_name: str,
    value: object,
    max_bytes: int,
) -> str:
    """
    Validate text destined for a fixed-width HDF5 string column.

    PyTables silently truncates overlong strings written to ``StringCol``.
    The source-oriented metadata path uses these values for stable output
    identifiers, so silent truncation would corrupt source identity.
    """
    text = "" if value is None else str(value)
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(
            f"HDF5 metadata field {field_name!r} exceeds the maximum stored "
            f"length of {max_bytes} bytes."
        )
    return text


@dataclass
class _PersistedForceDerivativePayload:
    """Worker-safe NumPy payload for one persisted derivative entry."""

    radial_center: np.ndarray
    radial_neighbor: np.ndarray
    radial_grads: np.ndarray
    radial_typespin: np.ndarray
    angular_center: np.ndarray
    angular_neighbor_j: np.ndarray
    angular_neighbor_k: np.ndarray
    angular_grads_i: np.ndarray
    angular_grads_j: np.ndarray
    angular_grads_k: np.ndarray
    angular_typespin: np.ndarray


@dataclass
class _BuildEntryPayload:
    """Worker-safe payload for one HDF5 entry."""

    source_id: str
    frame_idx: int
    source_kind: str
    display_name: str
    structure_bytes: bytes
    has_forces: bool
    n_atoms: int
    energy: float
    name: str
    feature_values: np.ndarray | None = None
    feature_n_features: int | None = None
    force_derivatives: _PersistedForceDerivativePayload | None = None


def _build_force_derivative_payload(
    local_derivatives: dict[str, dict[str, torch.Tensor | None]],
    *,
    descriptor_dtype: torch.dtype,
) -> _PersistedForceDerivativePayload:
    """Convert derivative tensors into a worker-safe NumPy payload."""
    float_dtype = (
        np.float32 if descriptor_dtype == torch.float32 else np.float64
    )
    radial = local_derivatives["radial"]
    angular = local_derivatives["angular"]
    return _PersistedForceDerivativePayload(
        radial_center=_tensor_to_numpy_1d(
            radial["center_idx"],
            dtype=np.int64,
        ),
        radial_neighbor=_tensor_to_numpy_1d(
            radial["neighbor_idx"],
            dtype=np.int64,
        ),
        radial_grads=_tensor_to_numpy_1d(
            radial["dG_drij"],
            dtype=float_dtype,
        ),
        radial_typespin=_tensor_to_numpy_1d(
            radial["neighbor_typespin"],
            dtype=float_dtype,
        ),
        angular_center=_tensor_to_numpy_1d(
            angular["center_idx"],
            dtype=np.int64,
        ),
        angular_neighbor_j=_tensor_to_numpy_1d(
            angular["neighbor_j_idx"],
            dtype=np.int64,
        ),
        angular_neighbor_k=_tensor_to_numpy_1d(
            angular["neighbor_k_idx"],
            dtype=np.int64,
        ),
        angular_grads_i=_tensor_to_numpy_1d(
            angular["grads_i"],
            dtype=float_dtype,
        ),
        angular_grads_j=_tensor_to_numpy_1d(
            angular["grads_j"],
            dtype=float_dtype,
        ),
        angular_grads_k=_tensor_to_numpy_1d(
            angular["grads_k"],
            dtype=float_dtype,
        ),
        angular_typespin=_tensor_to_numpy_1d(
            angular["triplet_typespin"],
            dtype=float_dtype,
        ),
    )


def _prepare_build_payloads_for_source_record(
    record: SourceRecord,
    *,
    descriptor,
    persist_features: bool,
    persist_force_derivatives: bool,
    max_referenced_energy_per_atom: float | None,
    atomic_energies: dict[str, float] | None,
) -> list[_BuildEntryPayload]:
    """
    Load one source record and prepare worker-safe payloads for ordered writes.

    This helper is intentionally top-level so concurrent build workers can
    execute it without capturing dataset instance state.
    """
    source_id = str(record.source_id)
    try:
        structs = record.load_structures()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load build source {source_id!r}."
        ) from exc

    payloads: list[_BuildEntryPayload] = []
    for frame_idx, struct in enumerate(structs):
        if not _should_keep_structure_for_build(
            struct,
            max_referenced_energy_per_atom=max_referenced_energy_per_atom,
            atomic_energies=atomic_energies,
        ):
            continue
        try:
            structure_bytes = pickle.dumps(
                struct,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            features = None
            local_derivatives = None

            if persist_features or persist_force_derivatives:
                positions = torch.from_numpy(struct.positions).to(
                    descriptor.dtype
                )
                cell = (
                    torch.from_numpy(struct.cell).to(descriptor.dtype)
                    if struct.cell is not None
                    else None
                )
                pbc = (
                    torch.from_numpy(struct.pbc)
                    if struct.pbc is not None
                    else None
                )

                if persist_force_derivatives and struct.has_forces():
                    species_indices = torch.tensor(
                        [descriptor.species_to_idx[s] for s in struct.species],
                        dtype=torch.long,
                    )
                    graph_trip = _build_force_graph_triplets(
                        descriptor=descriptor,
                        positions=positions,
                        cell=cell,
                        pbc=pbc,
                    )
                    features, local_derivatives = (
                        descriptor.compute_features_and_local_derivatives_with_graph(
                            positions=positions,
                            species_indices=species_indices,
                            graph=graph_trip["graph"],
                            triplets=graph_trip["triplets"],
                            center_indices=None,
                        )
                    )

                feature_values = None
                feature_n_features = None
                if persist_features:
                    if features is None:
                        features = descriptor.forward_from_positions(
                            positions,
                            struct.species,
                            cell,
                            pbc,
                        )
                    feature_values = _tensor_to_numpy_1d(
                        features,
                        dtype=np.dtype(
                            str(descriptor.dtype).replace("torch.", "")
                        ),
                    )
                    feature_n_features = int(features.shape[-1])

                derivative_payload = None
                if local_derivatives is not None:
                    derivative_payload = _build_force_derivative_payload(
                        local_derivatives,
                        descriptor_dtype=descriptor.dtype,
                    )
            else:
                feature_values = None
                feature_n_features = None
                derivative_payload = None
        except Exception as exc:
            raise RuntimeError(
                f"Failed to prepare persisted build payloads for {source_id!r} "
                f"(frame {frame_idx})."
            ) from exc

        payloads.append(
            _BuildEntryPayload(
                source_id=source_id,
                frame_idx=int(frame_idx),
                source_kind=str(record.source_kind or ""),
                display_name=str(record.display_name or ""),
                structure_bytes=structure_bytes,
                has_forces=bool(struct.has_forces()),
                n_atoms=int(struct.n_atoms),
                energy=_structure_energy_or_nan(struct),
                name=str(struct.name) if struct.name is not None else "",
                feature_values=feature_values,
                feature_n_features=feature_n_features,
                force_derivatives=derivative_payload,
            )
        )

    return payloads


def _parallel_build_chunk_size(build_workers: int) -> int:
    """
    Return the logical chunk size for streamed parallel HDF5 builds.

    Matching the chunk size to the worker count keeps the implementation
    simple and avoids buffering a large number of loaded source payloads
    in memory for compressed archive builds.
    """
    return max(1, int(build_workers))


@dataclass
class _FiltersConfig:
    """Compression filters configuration for HDF5 storage."""

    compression: str = "zlib"  # 'zlib', 'blosc', etc.
    compression_level: int = 5

    def to_tables_filters(self) -> tables.Filters:
        """Convert to PyTables Filters."""
        return tables.Filters(
            complevel=int(self.compression_level),
            complib=str(self.compression),
        )


class HDF5StructureDataset(Dataset):
    """
    HDF5-backed PyTorch Dataset that stores serialized torch Structures.

    This dataset is intended for very large datasets where keeping raw
    Structures in memory is infeasible. It builds (once) an HDF5 database
    file containing:
      - A VLArray 'entries/structures': pickled Structure per entry
      - A Table 'entries/meta': metadata per entry
          columns: source_id(str), frame_idx(int), source_kind(str),
                   display_name(str), has_forces(bool), n_atoms(int32),
                   energy(float64), name(str)
      - Optionally, a versioned '/torch_cache' container holding persisted
        raw feature payloads and/or sparse local derivative payloads plus
        descriptor-compatibility metadata
      - Optionally, a versioned '/descriptor_manifest' section for automatic
        descriptor recovery in later load-mode sessions

    New cache-writing builds use the unified ``/torch_cache`` schema. Legacy
    derivative-only ``/force_derivatives`` files remain readable for
    compatibility.

    At training time, items are read from the HDF5 file, unpickled into
    Structure objects, and featurized on-the-fly using the provided
    descriptor. The sample dict mirrors StructureDataset.__getitem__ and
    adds an optional ``local_derivatives`` entry for force-supervised HDF5
    samples when a compatible persisted cache is available. This allows the
    trainer to prefer persisted payloads lazily without loading the full cache
    into memory. Energy-view feature loading uses the runtime
    ``cache_features=True`` cache first, then compatible persisted HDF5 raw
    features, then on-the-fly featurization.

    Parameters
    ----------
    descriptor : object, optional
        Descriptor instance for featurization
        (dtype/device/species_index map). When ``mode='load'`` and the HDF5
        file contains a persisted descriptor manifest, this may be ``None``
        and the descriptor will be recovered automatically.
    database_file : str | os.PathLike
        Path-like location of the HDF5 database file. Will be created on
        build.
    sources : Sequence[str | os.PathLike] | SourceCollection, optional
        Build inputs used for database construction. Ordinary path-like input
        sequences are wrapped in the built-in file-source adapter. Custom
        source collections may load one or more structures per logical input.
        Required if you plan to call build_database().
    mode : str, optional
        One of {'auto', 'build', 'load'} controlling initialization behavior:
        - 'auto': if database_file exists, load; otherwise, expect that user
                  will call build_database() before reading.
        - 'build': do nothing on init; user must call build_database() to
                   create/overwrite the database.
        - 'load': open in read-only mode immediately; error if missing.
        Default: 'auto'
    seed : int, optional
        Reserved for deterministic helper utilities. Dataset contents and
        runtime training policy are unaffected by this value. Default: None
    in_memory_cache_size : int, optional
        Simple LRU cache size (entries) for unpickled Structures within a
        process/worker. Default: 2048
    compression : str, optional
        HDF5 compression library (e.g., 'zlib', 'blosc'). Default: 'zlib'
    compression_level : int, optional
        Compression level (0-9). Default: 5

    Notes on multiprocessing
    ------------------------
    - The HDF5 file handle is not pickled; on worker fork/deserialize,
      each worker lazily opens its own read-only handle on first use.
    - Source records are only used during ``build_database()``.
    - Build-time parallelism uses worker threads for parsing and optional
      persisted-cache preparation, but all HDF5 writes remain serialized in
      the main process to preserve deterministic output ordering.

    Lifecycle
    ---------
    ``build_database()`` leaves the same dataset instance ready for read-only
    use. Call ``close()`` for deterministic cleanup, or use the dataset as a
    context manager.
    """

    # --- HDF5 node paths
    _GROUP_ENTRIES = "/entries"
    _NODE_STRUCTURES = "/entries/structures"
    _NODE_META = "/entries/meta"
    _GROUP_DESCRIPTOR_MANIFEST = "/descriptor_manifest"
    _GROUP_TORCH_CACHE = "/torch_cache"
    _GROUP_CACHE_FEATURES = "/torch_cache/features"
    _GROUP_CACHE_FORCE_DERIVATIVES = "/torch_cache/force_derivatives"
    _NODE_CACHE_FEATURE_INDEX = "/torch_cache/features/index"
    _NODE_CACHE_FEATURE_VALUES = "/torch_cache/features/values"
    _NODE_CACHE_FORCE_DERIVATIVE_INDEX = "/torch_cache/force_derivatives/index"
    _NODE_CACHE_FORCE_DERIVATIVE_RADIAL_CENTER = (
        "/torch_cache/force_derivatives/radial/center_idx"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_RADIAL_NEIGHBOR = (
        "/torch_cache/force_derivatives/radial/neighbor_idx"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_RADIAL_GRADS = (
        "/torch_cache/force_derivatives/radial/dG_drij"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_RADIAL_TYPESPIN = (
        "/torch_cache/force_derivatives/radial/neighbor_typespin"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_CENTER = (
        "/torch_cache/force_derivatives/angular/center_idx"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_NEIGHBOR_J = (
        "/torch_cache/force_derivatives/angular/neighbor_j_idx"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_NEIGHBOR_K = (
        "/torch_cache/force_derivatives/angular/neighbor_k_idx"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_GRADS_I = (
        "/torch_cache/force_derivatives/angular/grads_i"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_GRADS_J = (
        "/torch_cache/force_derivatives/angular/grads_j"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_GRADS_K = (
        "/torch_cache/force_derivatives/angular/grads_k"
    )
    _NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_TYPESPIN = (
        "/torch_cache/force_derivatives/angular/triplet_typespin"
    )

    _GROUP_LEGACY_FORCE_DERIVATIVES = "/force_derivatives"
    _NODE_LEGACY_FORCE_DERIVATIVE_INDEX = "/force_derivatives/index"
    _NODE_LEGACY_FORCE_DERIVATIVE_RADIAL_CENTER = (
        "/force_derivatives/radial/center_idx"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_RADIAL_NEIGHBOR = (
        "/force_derivatives/radial/neighbor_idx"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_RADIAL_GRADS = (
        "/force_derivatives/radial/dG_drij"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_RADIAL_TYPESPIN = (
        "/force_derivatives/radial/neighbor_typespin"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_CENTER = (
        "/force_derivatives/angular/center_idx"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_NEIGHBOR_J = (
        "/force_derivatives/angular/neighbor_j_idx"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_NEIGHBOR_K = (
        "/force_derivatives/angular/neighbor_k_idx"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_GRADS_I = (
        "/force_derivatives/angular/grads_i"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_GRADS_J = (
        "/force_derivatives/angular/grads_j"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_GRADS_K = (
        "/force_derivatives/angular/grads_k"
    )
    _NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_TYPESPIN = (
        "/force_derivatives/angular/triplet_typespin"
    )

    _TORCH_CACHE_SCHEMA_VERSION = 2
    _TORCH_CACHE_FORMAT = "aenet.torch_training.cache.v2"
    _FORCE_DERIVATIVE_SCHEMA_VERSION = 1
    _FORCE_DERIVATIVE_PAYLOAD_FORMAT = (
        "aenet.torch_training.local_derivatives.v1"
    )
    _DESCRIPTOR_MANIFEST_SCHEMA_VERSION = (
        DESCRIPTOR_MANIFEST_SCHEMA_VERSION
    )
    _DESCRIPTOR_MANIFEST_FORMAT = DESCRIPTOR_MANIFEST_FORMAT
    # --- metadata table schema
    class _MetaRow(tables.IsDescription):
        source_id = tables.StringCol(_SOURCE_ID_MAX_BYTES)
        frame_idx = tables.Int64Col()
        source_kind = tables.StringCol(_SOURCE_KIND_MAX_BYTES)
        display_name = tables.StringCol(_DISPLAY_NAME_MAX_BYTES)
        has_forces = tables.BoolCol()
        n_atoms = tables.Int32Col()
        energy = tables.Float64Col()
        name = tables.StringCol(_STRUCTURE_NAME_MAX_BYTES)

    class _ForceDerivativeIndexRow(tables.IsDescription):
        entry_idx = tables.Int64Col()
        cache_row = tables.Int64Col()
        n_atoms = tables.Int32Col()
        n_radial_edges = tables.Int32Col()
        n_angular_triplets = tables.Int32Col()

    class _FeatureIndexRow(tables.IsDescription):
        entry_idx = tables.Int64Col()
        cache_row = tables.Int64Col()
        n_atoms = tables.Int32Col()
        n_features = tables.Int32Col()

    def __init__(
        self,
        descriptor,
        database_file: str | os.PathLike,
        sources: Sequence[str | os.PathLike] | SourceCollection | None = None,
        mode: str = "auto",
        *,
        seed: int | None = None,
        in_memory_cache_size: int = 2048,
        compression: str = "zlib",
        compression_level: int = 5,
    ):
        # Descriptor and featurization flags
        self.descriptor = descriptor

        # Random init
        self.seed = seed

        # Build/read configuration
        self._db_path = os.fspath(database_file)
        self._sources = (
            coerce_source_collection(sources)
            if sources is not None
            else None
        )
        self._filters = _FiltersConfig(
            compression=compression,
            compression_level=int(compression_level),
        )

        # Runtime state
        self._h5: tables.File | None = None  # lazily opened per process
        self._n_entries: int | None = None
        self._force_indices_all: list[int] | None = None
        self._descriptor_manifest_info: dict | None = None
        self._torch_cache_info: dict | None = None
        self._feature_rows_by_entry: dict[int, int] = {}
        self._feature_cache_info: dict | None = None
        self._force_derivative_rows_by_entry: dict[int, int] = {}
        self._force_derivative_cache_info: dict | None = None
        self._force_derivative_node_paths: dict[str, str] | None = None

        # Simple per-process LRU cache for unpickled Structures
        self._cache_capacity = max(0, int(in_memory_cache_size))
        self._cache: dict = _LRU(maxlen=self._cache_capacity)

        # Init per mode
        if mode not in ("auto", "build", "load"):
            raise ValueError(
                f"Invalid mode '{mode}' (must be 'auto'|'build'|'load')")
        self._mode = mode

        if mode == "load":
            if not os.path.exists(self._db_path):
                raise FileNotFoundError(
                    f"HDF5 database not found: {self._db_path}")
            self._open_readonly()  # set _n_entries and force indices
        elif mode == "auto":
            if os.path.exists(self._db_path):
                self._open_readonly()

    # -------------------- Build/Load helpers --------------------

    def build_database(
        self,
        show_progress: bool = True,
        *,
        build_workers: int = 0,
        max_referenced_energy_per_atom: float | None = None,
        atomic_energies: dict[str, float] | None = None,
        persist_descriptor: bool = False,
        persist_features: bool = False,
        persist_force_derivatives: bool = False,
    ) -> None:
        """
        Build (or overwrite) the HDF5 database from the configured sources.

        This will:
          - Create '/entries/structures' VLArray of uint8 (pickled Structure)
          - Create '/entries/meta' Table with per-entry metadata
          - Optionally create '/torch_cache' and persist raw features and/or
            sparse local derivative payloads
          - Populate force index list for efficient selection

        Parameters
        ----------
        show_progress : bool
            If True and tqdm is available, show a progress bar.
        build_workers : int
            Number of worker threads used for source-record loading and
            optional persisted-cache preparation. ``0`` and ``1`` keep the
            existing serial build path. Values greater than ``1`` preserve
            deterministic HDF5 entry ordering while keeping all HDF5 writes
            in the parent process. This setting is separate from training-time
            ``num_workers``. Default: ``0``
        max_referenced_energy_per_atom : float, optional
            Exclude structures whose referenced cohesive or formation energy
            per atom exceeds this build-time threshold. Unlike
            ``TorchTrainingConfig.max_energy``, this applies only while the
            HDF5 file is being constructed. Default: ``None``
        atomic_energies : dict[str, float], optional
            Atomic reference energies used when interpreting
            ``max_referenced_energy_per_atom``. When omitted, build-time
            filtering falls back to all-zero references so externally
            referenced labels are filtered as provided. Default: ``None``
        persist_descriptor : bool
            If True, persist a versioned descriptor manifest that can recover
            supported descriptor objects when the HDF5 dataset is reopened
            later. This is enabled automatically when
            ``persist_features=True`` or
            ``persist_force_derivatives=True``.
        persist_features : bool
            If True, persist raw unnormalized ``(N, F)`` descriptor features
            for each structure in the versioned ``/torch_cache`` schema.
            Later HDF5-backed sample materialization reuses these payloads
            lazily when they are descriptor-compatible.
        persist_force_derivatives : bool
            If True, compute and persist sparse local derivative payloads for
            force-labeled structures using the versioned ``/torch_cache``
            schema. Force-supervised samples loaded from that database will
            expose the persisted payload lazily through ``__getitem__`` when
            it is present and descriptor-compatible.
        """
        source_collection = self._sources
        if source_collection is None:
            raise ValueError("sources must be provided to build_database()")
        build_workers = int(build_workers)
        if build_workers < 0:
            raise ValueError("build_workers must be >= 0")
        if build_workers > 1 and not source_collection.capabilities.supports_parallel_build:
            raise ValueError(
                "The configured sources do not support build_workers > 1."
            )
        if atomic_energies is not None:
            atomic_energies = {
                str(key): float(value)
                for key, value in atomic_energies.items()
            }

        persist_descriptor = bool(
            persist_descriptor or persist_features or persist_force_derivatives
        )
        if (persist_features or persist_force_derivatives) and self.descriptor is None:
            raise RuntimeError(
                "Persisting HDF5 cache payloads requires a descriptor "
                "instance."
            )
        if persist_descriptor and self.descriptor is None:
            raise RuntimeError(
                "Persisting a descriptor manifest requires a descriptor "
                "instance."
            )

        # Ensure directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Close any open handles before writing
        self._close_handle()

        tmp_handle = tempfile.NamedTemporaryFile(
            prefix=f"{Path(self._db_path).name}.",
            suffix=".tmp",
            dir=str(Path(self._db_path).parent),
            delete=False,
        )
        tmp_path = tmp_handle.name
        tmp_handle.close()

        # Open HDF5 file for writing
        h5 = tables.open_file(
            tmp_path,
            mode="w",
            filters=self._filters.to_tables_filters(),
        )
        pbar = None
        build_succeeded = False
        try:
            if persist_force_derivatives and not hasattr(
                self.descriptor,
                "compute_features_and_local_derivatives_with_graph",
            ):
                raise RuntimeError(
                    "Persisting force-derivative caches requires descriptor "
                    "support for "
                    "'compute_features_and_local_derivatives_with_graph()'."
                )

            # Create groups and nodes
            entries_group = h5.create_group(
                "/", "entries", "Serialized entries")
            vl_struct = h5.create_vlarray(
                entries_group, "structures",
                atom=tables.UInt8Atom(), title="Pickled Structures"
            )
            meta_table = h5.create_table(
                entries_group, "meta", description=self._MetaRow)
            _write_energy_filter_metadata(
                h5,
                max_referenced_energy_per_atom=max_referenced_energy_per_atom,
                atomic_energies=atomic_energies,
            )
            if persist_descriptor:
                self._create_descriptor_manifest_storage(h5)
            if persist_features or persist_force_derivatives:
                self._create_torch_cache_storage(
                    h5,
                    persist_features=persist_features,
                    persist_force_derivatives=persist_force_derivatives,
                )

            total_records = None
            try:
                total_records = len(source_collection)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                total_records = None

            # Optional progress bar
            try:
                from tqdm import tqdm as _tqdm  # type: ignore
                pbar = (
                    _tqdm(
                        total=total_records,
                        desc="Building HDF5",
                        ncols=80,
                    )
                    if show_progress
                    else None
                )
            except Exception:
                pbar = None

            worker = partial(
                _prepare_build_payloads_for_source_record,
                descriptor=self.descriptor,
                persist_features=persist_features,
                persist_force_derivatives=persist_force_derivatives,
                max_referenced_energy_per_atom=max_referenced_energy_per_atom,
                atomic_energies=atomic_energies,
            )
            if build_workers <= 1:
                payload_batches = map(worker, source_collection.iter_records())
                for payload_batch in payload_batches:
                    self._append_build_payload_batch(
                        h5=h5,
                        vl_struct=vl_struct,
                        meta_table=meta_table,
                        payload_batch=payload_batch,
                    )
                    if pbar is not None:
                        pbar.update(1)
            elif isinstance(source_collection, _ChunkedSourceCollection):
                chunk_size = _parallel_build_chunk_size(build_workers)
                try:
                    with ThreadPoolExecutor(
                        max_workers=build_workers,
                    ) as executor:
                        for record_chunk in source_collection.iter_record_chunks(
                            chunk_size=chunk_size,
                        ):
                            for payload_batch in executor.map(
                                worker,
                                record_chunk,
                            ):
                                self._append_build_payload_batch(
                                    h5=h5,
                                    vl_struct=vl_struct,
                                    meta_table=meta_table,
                                    payload_batch=payload_batch,
                                )
                            if pbar is not None:
                                pbar.update(len(record_chunk))
                except Exception as exc:
                    if isinstance(exc, RuntimeError):
                        raise
                    raise RuntimeError(
                        "Parallel HDF5 build failed while streaming "
                        "source chunks."
                    ) from exc
            else:
                try:
                    with ThreadPoolExecutor(
                        max_workers=build_workers,
                    ) as executor:
                        for payload_batch in executor.map(
                            worker,
                            source_collection.iter_records(),
                        ):
                            self._append_build_payload_batch(
                                h5=h5,
                                vl_struct=vl_struct,
                                meta_table=meta_table,
                                payload_batch=payload_batch,
                            )
                            if pbar is not None:
                                pbar.update(1)
                except Exception as exc:
                    if isinstance(exc, RuntimeError):
                        raise
                    raise RuntimeError(
                        "Parallel HDF5 build failed while preparing "
                        "worker-side payloads."
                    ) from exc

            h5.flush()
            build_succeeded = True
        finally:
            if pbar is not None:
                pbar.close()
            h5.close()
            if not build_succeeded and os.path.exists(tmp_path):
                os.remove(tmp_path)

        try:
            os.replace(tmp_path, self._db_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # Clear any stale state; the dataset will reopen lazily on first use.
        self._close_handle()

    def _append_build_payload_batch(
        self,
        *,
        h5: tables.File,
        vl_struct,
        meta_table,
        payload_batch: list[_BuildEntryPayload],
    ) -> None:
        """Append one source record worth of payloads to the HDF5 file."""
        for payload in payload_batch:
            entry_idx = int(len(vl_struct))
            vl_struct.append(
                np.frombuffer(payload.structure_bytes, dtype=np.uint8)
            )

            row = meta_table.row
            row["source_id"] = _validated_hdf5_text(
                field_name="source_id",
                value=payload.source_id,
                max_bytes=_SOURCE_ID_MAX_BYTES,
            )
            row["frame_idx"] = int(payload.frame_idx)
            row["source_kind"] = _validated_hdf5_text(
                field_name="source_kind",
                value=payload.source_kind,
                max_bytes=_SOURCE_KIND_MAX_BYTES,
            )
            row["display_name"] = _validated_hdf5_text(
                field_name="display_name",
                value=payload.display_name,
                max_bytes=_DISPLAY_NAME_MAX_BYTES,
            )
            row["has_forces"] = bool(payload.has_forces)
            row["n_atoms"] = int(payload.n_atoms)
            row["energy"] = float(payload.energy)
            row["name"] = _validated_hdf5_text(
                field_name="name",
                value=payload.name,
                max_bytes=_STRUCTURE_NAME_MAX_BYTES,
            )
            row.append()

            if payload.feature_values is not None:
                if payload.feature_n_features is None:
                    raise RuntimeError("Missing persisted feature shape metadata.")
                self._append_feature_cache_payload(
                    h5=h5,
                    entry_idx=entry_idx,
                    n_atoms=int(payload.n_atoms),
                    n_features=int(payload.feature_n_features),
                    feature_values=payload.feature_values,
                )

            if payload.force_derivatives is not None:
                self._append_force_derivative_cache_payload(
                    h5=h5,
                    entry_idx=entry_idx,
                    n_atoms=int(payload.n_atoms),
                    payload=payload.force_derivatives,
                    node_paths=self._v2_force_derivative_node_paths(),
                )

        meta_table.flush()

    def _create_descriptor_manifest_storage(self, h5: tables.File) -> None:
        """Create the versioned HDF5 node used for descriptor recovery."""
        manifest_group = h5.create_group(
            "/",
            "descriptor_manifest",
            "Persisted descriptor reconstruction manifest",
        )
        manifest = descriptor_manifest_from_object(self.descriptor)
        attrs = manifest_group._v_attrs
        attrs.schema_version = manifest["schema_version"]
        attrs.manifest_format = manifest["manifest_format"]
        attrs.descriptor_class = manifest["descriptor_class"]
        attrs.config_json = manifest["config_json"]
        attrs.config_sha256 = manifest["config_sha256"]

    def _create_torch_cache_storage(
        self,
        h5: tables.File,
        *,
        persist_features: bool,
        persist_force_derivatives: bool,
    ) -> None:
        """Create the unified versioned cache container and requested sections."""
        cache_group = h5.create_group(
            "/",
            "torch_cache",
            "Unified persisted torch-training cache payloads",
        )
        metadata = _descriptor_derivative_cache_metadata(self.descriptor)
        attrs = cache_group._v_attrs
        attrs.schema_version = self._TORCH_CACHE_SCHEMA_VERSION
        attrs.cache_format = self._TORCH_CACHE_FORMAT
        attrs.descriptor_compat_json = metadata["compat_json"]
        attrs.descriptor_compat_sha256 = metadata["compat_sha256"]
        attrs.storage_dtype = metadata["storage_dtype"]
        attrs.contains_features = bool(persist_features)
        attrs.contains_force_derivatives = bool(persist_force_derivatives)

        if persist_features:
            self._create_feature_cache_storage(h5)
        if persist_force_derivatives:
            self._create_v2_force_derivative_storage(h5)

    def _create_feature_cache_storage(self, h5: tables.File) -> None:
        """Create the feature section of the versioned torch cache."""
        feature_group = h5.create_group(
            self._GROUP_TORCH_CACHE,
            "features",
            "Persisted raw descriptor feature payloads",
        )
        h5.create_table(
            feature_group,
            "index",
            description=self._FeatureIndexRow,
            title="Per-entry raw feature cache index",
        )
        h5.create_vlarray(
            feature_group,
            "values",
            atom=_tables_float_atom(self.descriptor.dtype),
            title="Flattened raw feature tensors per cached entry",
        )

    def _create_v2_force_derivative_storage(self, h5: tables.File) -> None:
        """Create the derivative section of the versioned torch cache."""
        deriv_group = h5.create_group(
            self._GROUP_TORCH_CACHE,
            "force_derivatives",
            "Persisted sparse local derivative payloads",
        )
        self._create_force_derivative_storage_nodes(h5, deriv_group)

    def _create_legacy_force_derivative_storage(self, h5: tables.File) -> None:
        """Create the legacy v1 derivative-only nodes for compatibility tests."""
        deriv_group = h5.create_group(
            "/",
            "force_derivatives",
            "Persisted sparse local derivative payloads",
        )
        self._create_force_derivative_storage_nodes(h5, deriv_group)

    def _create_force_derivative_storage_nodes(
        self,
        h5: tables.File,
        deriv_group,
    ) -> None:
        """Create derivative payload nodes under the requested parent group."""
        radial_group = h5.create_group(
            deriv_group,
            "radial",
            "Radial sparse local derivative blocks",
        )
        angular_group = h5.create_group(
            deriv_group,
            "angular",
            "Angular sparse local derivative blocks",
        )

        metadata = _descriptor_derivative_cache_metadata(self.descriptor)
        attrs = deriv_group._v_attrs
        attrs.schema_version = self._FORCE_DERIVATIVE_SCHEMA_VERSION
        attrs.payload_format = self._FORCE_DERIVATIVE_PAYLOAD_FORMAT
        attrs.descriptor_compat_json = metadata["compat_json"]
        attrs.descriptor_compat_sha256 = metadata["compat_sha256"]
        attrs.storage_dtype = metadata["storage_dtype"]
        attrs.n_radial_features = metadata["n_radial_features"]
        attrs.n_angular_features = metadata["n_angular_features"]
        attrs.multi = metadata["multi"]
        attrs.contains_features = False
        attrs.contains_positions = False

        h5.create_table(
            deriv_group,
            "index",
            description=self._ForceDerivativeIndexRow,
            title="Per-entry sparse derivative cache index",
        )

        float_atom = _tables_float_atom(self.descriptor.dtype)
        int_atom = tables.Int64Atom()

        h5.create_vlarray(
            radial_group,
            "center_idx",
            atom=int_atom,
            title="Flattened radial center indices per cached entry",
        )
        h5.create_vlarray(
            radial_group,
            "neighbor_idx",
            atom=int_atom,
            title="Flattened radial neighbor indices per cached entry",
        )
        h5.create_vlarray(
            radial_group,
            "dG_drij",
            atom=float_atom,
            title="Flattened radial derivative blocks per cached entry",
        )
        h5.create_vlarray(
            radial_group,
            "neighbor_typespin",
            atom=float_atom,
            title="Flattened radial typespin coefficients per cached entry",
        )

        h5.create_vlarray(
            angular_group,
            "center_idx",
            atom=int_atom,
            title="Flattened angular center indices per cached entry",
        )
        h5.create_vlarray(
            angular_group,
            "neighbor_j_idx",
            atom=int_atom,
            title="Flattened angular j-neighbor indices per cached entry",
        )
        h5.create_vlarray(
            angular_group,
            "neighbor_k_idx",
            atom=int_atom,
            title="Flattened angular k-neighbor indices per cached entry",
        )
        h5.create_vlarray(
            angular_group,
            "grads_i",
            atom=float_atom,
            title="Flattened angular center-atom derivative blocks",
        )
        h5.create_vlarray(
            angular_group,
            "grads_j",
            atom=float_atom,
            title="Flattened angular j-neighbor derivative blocks",
        )
        h5.create_vlarray(
            angular_group,
            "grads_k",
            atom=float_atom,
            title="Flattened angular k-neighbor derivative blocks",
        )
        h5.create_vlarray(
            angular_group,
            "triplet_typespin",
            atom=float_atom,
            title="Flattened angular typespin coefficients per cached entry",
        )

    def _append_feature_cache_entry(
        self,
        *,
        h5: tables.File,
        entry_idx: int,
        n_atoms: int,
        features: torch.Tensor,
    ) -> None:
        """Append one persisted raw feature tensor to the unified cache."""
        self._append_feature_cache_payload(
            h5=h5,
            entry_idx=entry_idx,
            n_atoms=n_atoms,
            n_features=int(features.shape[-1]),
            feature_values=_tensor_to_numpy_1d(
                features,
                dtype=np.dtype(
                    str(self.descriptor.dtype).replace("torch.", "")
                ),
            ),
        )

    def _append_feature_cache_payload(
        self,
        *,
        h5: tables.File,
        entry_idx: int,
        n_atoms: int,
        n_features: int,
        feature_values: np.ndarray,
    ) -> None:
        """Append one persisted raw feature payload to the unified cache."""
        index_table = h5.get_node(self._NODE_CACHE_FEATURE_INDEX)
        values = h5.get_node(self._NODE_CACHE_FEATURE_VALUES)
        cache_row = int(index_table.nrows)
        values.append(feature_values)

        row = index_table.row
        row["entry_idx"] = int(entry_idx)
        row["cache_row"] = int(cache_row)
        row["n_atoms"] = int(n_atoms)
        row["n_features"] = int(n_features)
        row.append()
        index_table.flush()

    def _append_force_derivative_cache_entry(
        self,
        *,
        h5: tables.File,
        entry_idx: int,
        n_atoms: int,
        local_derivatives: dict[str, dict[str, torch.Tensor | None]],
        node_paths: dict[str, str],
    ) -> None:
        """Append one force-labeled structure to the derivative cache."""
        self._append_force_derivative_cache_payload(
            h5=h5,
            entry_idx=entry_idx,
            n_atoms=n_atoms,
            payload=_build_force_derivative_payload(
                local_derivatives,
                descriptor_dtype=self.descriptor.dtype,
            ),
            node_paths=node_paths,
        )

    def _append_force_derivative_cache_payload(
        self,
        *,
        h5: tables.File,
        entry_idx: int,
        n_atoms: int,
        payload: _PersistedForceDerivativePayload,
        node_paths: dict[str, str],
    ) -> None:
        """Append one worker-safe derivative payload to the cache."""
        index_table = h5.get_node(node_paths["index"])
        cache_row = int(index_table.nrows)
        h5.get_node(node_paths["radial_center"]).append(
            payload.radial_center
        )
        h5.get_node(node_paths["radial_neighbor"]).append(
            payload.radial_neighbor
        )
        h5.get_node(node_paths["radial_grads"]).append(
            payload.radial_grads
        )
        h5.get_node(node_paths["radial_typespin"]).append(
            payload.radial_typespin
        )

        h5.get_node(node_paths["angular_center"]).append(
            payload.angular_center
        )
        h5.get_node(node_paths["angular_neighbor_j"]).append(
            payload.angular_neighbor_j
        )
        h5.get_node(node_paths["angular_neighbor_k"]).append(
            payload.angular_neighbor_k
        )
        h5.get_node(node_paths["angular_grads_i"]).append(
            payload.angular_grads_i
        )
        h5.get_node(node_paths["angular_grads_j"]).append(
            payload.angular_grads_j
        )
        h5.get_node(node_paths["angular_grads_k"]).append(
            payload.angular_grads_k
        )
        h5.get_node(node_paths["angular_typespin"]).append(
            payload.angular_typespin
        )

        row = index_table.row
        row["entry_idx"] = int(entry_idx)
        row["cache_row"] = int(cache_row)
        row["n_atoms"] = int(n_atoms)
        row["n_radial_edges"] = int(payload.radial_center.size)
        row["n_angular_triplets"] = int(payload.angular_center.size)
        row.append()
        index_table.flush()

    @classmethod
    def _v2_force_derivative_node_paths(cls) -> dict[str, str]:
        """Return the derivative node layout for schema v2 cache files."""
        return {
            "index": cls._NODE_CACHE_FORCE_DERIVATIVE_INDEX,
            "radial_center": cls._NODE_CACHE_FORCE_DERIVATIVE_RADIAL_CENTER,
            "radial_neighbor": cls._NODE_CACHE_FORCE_DERIVATIVE_RADIAL_NEIGHBOR,
            "radial_grads": cls._NODE_CACHE_FORCE_DERIVATIVE_RADIAL_GRADS,
            "radial_typespin": cls._NODE_CACHE_FORCE_DERIVATIVE_RADIAL_TYPESPIN,
            "angular_center": cls._NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_CENTER,
            "angular_neighbor_j": cls._NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_NEIGHBOR_J,
            "angular_neighbor_k": cls._NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_NEIGHBOR_K,
            "angular_grads_i": cls._NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_GRADS_I,
            "angular_grads_j": cls._NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_GRADS_J,
            "angular_grads_k": cls._NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_GRADS_K,
            "angular_typespin": cls._NODE_CACHE_FORCE_DERIVATIVE_ANGULAR_TYPESPIN,
        }

    @classmethod
    def _legacy_force_derivative_node_paths(cls) -> dict[str, str]:
        """Return the legacy v1 derivative node layout."""
        return {
            "index": cls._NODE_LEGACY_FORCE_DERIVATIVE_INDEX,
            "radial_center": cls._NODE_LEGACY_FORCE_DERIVATIVE_RADIAL_CENTER,
            "radial_neighbor": cls._NODE_LEGACY_FORCE_DERIVATIVE_RADIAL_NEIGHBOR,
            "radial_grads": cls._NODE_LEGACY_FORCE_DERIVATIVE_RADIAL_GRADS,
            "radial_typespin": cls._NODE_LEGACY_FORCE_DERIVATIVE_RADIAL_TYPESPIN,
            "angular_center": cls._NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_CENTER,
            "angular_neighbor_j": cls._NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_NEIGHBOR_J,
            "angular_neighbor_k": cls._NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_NEIGHBOR_K,
            "angular_grads_i": cls._NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_GRADS_I,
            "angular_grads_j": cls._NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_GRADS_J,
            "angular_grads_k": cls._NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_GRADS_K,
            "angular_typespin": cls._NODE_LEGACY_FORCE_DERIVATIVE_ANGULAR_TYPESPIN,
        }

    def _open_readonly(self) -> None:
        """Open the HDF5 database in read-only mode and initialize metadata."""
        self._h5 = tables.open_file(self._db_path, mode="r")
        self._descriptor_manifest_info = None
        self._torch_cache_info = None
        self._feature_rows_by_entry = {}
        self._feature_cache_info = None
        self._force_derivative_rows_by_entry = {}
        self._force_derivative_cache_info = None
        self._force_derivative_node_paths = None
        # Determine entry count from VLArray length
        try:
            vl = self._h5.get_node(self._NODE_STRUCTURES)
        except tables.NoSuchNodeError as exc:
            raise RuntimeError("Invalid database structure: missing "
                               + f"{self._NODE_STRUCTURES}") from exc
        self._n_entries = int(len(vl))

        # Initialize force indices from metadata table
        self._force_indices_all = []
        meta = self._h5.get_node(self._NODE_META)
        for i, row in enumerate(meta):  # type: ignore[assignment]
            if bool(row["has_forces"]):
                self._force_indices_all.append(i)

        self._initialize_descriptor_manifest_state()
        self._reconcile_descriptor_manifest()
        self._initialize_persisted_cache_state()

    def _initialize_descriptor_manifest_state(self) -> None:
        """Read descriptor-manifest metadata, if present."""
        if self._h5 is None:
            return
        try:
            manifest_group = self._h5.get_node(self._GROUP_DESCRIPTOR_MANIFEST)
        except tables.NoSuchNodeError:
            return

        attrs = manifest_group._v_attrs
        config_json = str(attrs.config_json)
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid persisted descriptor manifest: malformed JSON "
                "configuration."
            ) from exc

        self._descriptor_manifest_info = {
            "schema_version": int(attrs.schema_version),
            "manifest_format": str(attrs.manifest_format),
            "descriptor_class": str(attrs.descriptor_class),
            "config_json": config_json,
            "config_sha256": str(attrs.config_sha256),
            "config": config,
        }

    def _validate_descriptor_manifest(self) -> dict:
        """Ensure the persisted descriptor manifest is supported."""
        if self._descriptor_manifest_info is None:
            raise RuntimeError(
                "This HDF5 dataset does not contain a persisted descriptor "
                "manifest."
            )

        info = self._descriptor_manifest_info
        if info["schema_version"] != self._DESCRIPTOR_MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported descriptor manifest schema version "
                f"{info['schema_version']}; expected "
                f"{self._DESCRIPTOR_MANIFEST_SCHEMA_VERSION}."
            )
        if info["manifest_format"] != self._DESCRIPTOR_MANIFEST_FORMAT:
            raise RuntimeError(
                "Unsupported descriptor manifest format "
                f"{info['manifest_format']!r}."
            )
        return info

    def _reconcile_descriptor_manifest(self) -> None:
        """Recover or validate the descriptor against any persisted manifest."""
        if self._descriptor_manifest_info is None:
            return

        manifest = self._validate_descriptor_manifest()
        if self.descriptor is None:
            self.descriptor = descriptor_from_manifest(manifest)
            return

        if not descriptor_matches_manifest(self.descriptor, manifest):
            raise RuntimeError(
                "Persisted descriptor manifest is incompatible with the "
                "explicit descriptor configuration. Provide a matching "
                "descriptor or omit descriptor=... to recover the stored "
                "descriptor."
            )

    def _initialize_persisted_cache_state(self) -> None:
        """Read schema v2 cache metadata or fall back to legacy v1 layout."""
        if self._h5 is None:
            return
        try:
            cache_group = self._h5.get_node(self._GROUP_TORCH_CACHE)
        except tables.NoSuchNodeError:
            self._initialize_legacy_force_derivative_cache_state()
            return

        attrs = cache_group._v_attrs
        self._torch_cache_info = {
            "schema_version": int(attrs.schema_version),
            "cache_format": str(attrs.cache_format),
            "descriptor_compat_json": str(attrs.descriptor_compat_json),
            "descriptor_compat_sha256": str(attrs.descriptor_compat_sha256),
            "storage_dtype": str(attrs.storage_dtype),
            "contains_features": bool(attrs.contains_features),
            "contains_force_derivatives": bool(attrs.contains_force_derivatives),
        }
        self._validate_torch_cache_schema()

        if bool(self._torch_cache_info["contains_features"]):
            self._initialize_feature_cache_state()
        if bool(self._torch_cache_info["contains_force_derivatives"]):
            self._initialize_v2_force_derivative_cache_state()

    def _validate_torch_cache_schema(self) -> dict:
        """Ensure the unified torch cache metadata is supported."""
        if self._torch_cache_info is None:
            raise RuntimeError("This HDF5 dataset does not contain a torch cache.")
        info = self._torch_cache_info
        if info["schema_version"] != self._TORCH_CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported torch cache schema version "
                f"{info['schema_version']}; expected "
                f"{self._TORCH_CACHE_SCHEMA_VERSION}."
            )
        if info["cache_format"] != self._TORCH_CACHE_FORMAT:
            raise RuntimeError(
                "Unsupported torch cache format "
                f"{info['cache_format']!r}."
            )
        return info

    def _initialize_feature_cache_state(self) -> None:
        """Read persisted feature metadata and the per-entry row map."""
        info = self._validate_torch_cache_schema()
        self._feature_cache_info = dict(info)

        index = self._h5.get_node(self._NODE_CACHE_FEATURE_INDEX)
        for row in index:  # type: ignore[assignment]
            self._feature_rows_by_entry[int(row["entry_idx"])] = int(
                row["cache_row"]
            )

    def _initialize_v2_force_derivative_cache_state(self) -> None:
        """Read schema v2 derivative metadata and the per-entry row map."""
        info = self._validate_torch_cache_schema()
        deriv_group = self._h5.get_node(self._GROUP_CACHE_FORCE_DERIVATIVES)
        attrs = deriv_group._v_attrs
        self._force_derivative_cache_info = {
            **info,
            "payload_format": str(attrs.payload_format),
            "n_radial_features": int(attrs.n_radial_features),
            "n_angular_features": int(attrs.n_angular_features),
            "multi": bool(attrs.multi),
        }
        self._force_derivative_node_paths = self._v2_force_derivative_node_paths()

        index = self._h5.get_node(self._NODE_CACHE_FORCE_DERIVATIVE_INDEX)
        for row in index:  # type: ignore[assignment]
            self._force_derivative_rows_by_entry[int(row["entry_idx"])] = int(
                row["cache_row"]
            )

    def _initialize_legacy_force_derivative_cache_state(self) -> None:
        """Read legacy v1 derivative-cache metadata and the per-entry row map."""
        if self._h5 is None:
            return
        try:
            deriv_group = self._h5.get_node(self._GROUP_LEGACY_FORCE_DERIVATIVES)
        except tables.NoSuchNodeError:
            return

        attrs = deriv_group._v_attrs
        self._force_derivative_cache_info = {
            "schema_version": int(attrs.schema_version),
            "payload_format": str(attrs.payload_format),
            "descriptor_compat_json": str(attrs.descriptor_compat_json),
            "descriptor_compat_sha256": str(attrs.descriptor_compat_sha256),
            "storage_dtype": str(attrs.storage_dtype),
            "n_radial_features": int(attrs.n_radial_features),
            "n_angular_features": int(attrs.n_angular_features),
            "multi": bool(attrs.multi),
            "contains_features": bool(attrs.contains_features),
            "contains_positions": bool(attrs.contains_positions),
        }
        self._force_derivative_node_paths = self._legacy_force_derivative_node_paths()

        index = self._h5.get_node(self._NODE_LEGACY_FORCE_DERIVATIVE_INDEX)
        for row in index:  # type: ignore[assignment]
            self._force_derivative_rows_by_entry[int(row["entry_idx"])] = int(
                row["cache_row"]
            )

    def _validate_cache_descriptor_compatibility(self, info: dict, action: str) -> dict:
        """Ensure the persisted cache payload is compatible with the descriptor."""
        self._require_descriptor(action)
        expected = _descriptor_derivative_cache_metadata(self.descriptor)
        if info["descriptor_compat_sha256"] != expected["compat_sha256"]:
            raise RuntimeError(
                f"{action} found an incompatible persisted cache. Rebuild the "
                "cache with matching species/order/cutoff settings."
            )
        return info

    def _validate_feature_cache_compatibility(self) -> dict:
        """Ensure the persisted feature cache is supported and compatible."""
        if self._feature_cache_info is None:
            raise RuntimeError("This HDF5 dataset does not contain persisted features.")
        info = self._validate_torch_cache_schema()
        return self._validate_cache_descriptor_compatibility(
            info,
            "Loading persisted features",
        )

    def _validate_force_derivative_cache_compatibility(self) -> dict:
        """Ensure the persisted derivative cache is supported and compatible."""
        if self._force_derivative_cache_info is None:
            raise RuntimeError(
                "This HDF5 dataset does not contain a force-derivative cache."
            )

        info = self._force_derivative_cache_info
        if int(info["schema_version"]) == self._TORCH_CACHE_SCHEMA_VERSION:
            self._validate_torch_cache_schema()
        elif int(info["schema_version"]) != self._FORCE_DERIVATIVE_SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported force-derivative cache schema version "
                f"{info['schema_version']}."
            )
        if info["payload_format"] != self._FORCE_DERIVATIVE_PAYLOAD_FORMAT:
            raise RuntimeError(
                "Unsupported force-derivative cache payload format "
                f"{info['payload_format']!r}."
            )

        return self._validate_cache_descriptor_compatibility(
            info,
            "Loading persisted force derivatives",
        )

    def has_persisted_descriptor(self) -> bool:
        """Return True when the HDF5 file contains a descriptor manifest."""
        if self._h5 is None and os.path.exists(self._db_path):
            self._open_readonly()
        return self._descriptor_manifest_info is not None

    def get_descriptor_manifest(self) -> dict | None:
        """Return metadata describing the persisted descriptor manifest."""
        if self._h5 is None and os.path.exists(self._db_path):
            self._open_readonly()
        if self._descriptor_manifest_info is None:
            return None
        info = dict(self._descriptor_manifest_info)
        info["config"] = dict(info["config"])
        return info

    def load_persisted_descriptor(self):
        """Recover and return the persisted descriptor object, if available."""
        if self._h5 is None and os.path.exists(self._db_path):
            self._open_readonly()
        manifest = self._validate_descriptor_manifest()
        return descriptor_from_manifest(manifest)

    def has_persisted_features(self) -> bool:
        """Return True when the HDF5 file contains persisted feature payloads."""
        if self._h5 is None and os.path.exists(self._db_path):
            self._open_readonly()
        return bool(self._feature_rows_by_entry)

    def get_persisted_feature_cache_info(self) -> dict | None:
        """Return metadata describing the persisted feature cache section."""
        if self._h5 is None and os.path.exists(self._db_path):
            self._open_readonly()
        if self._feature_cache_info is None:
            return None
        return dict(self._feature_cache_info)

    def load_persisted_features(self, idx: int) -> torch.Tensor | None:
        """
        Load persisted raw descriptor features for one dataset entry.

        The returned tensor is reshaped to ``(N, F)`` and cast to the active
        descriptor dtype after compatibility validation.
        """
        if self._h5 is None:
            self._open_readonly()
        self._validate_feature_cache_compatibility()
        cache_row = self._feature_rows_by_entry.get(int(idx))
        if cache_row is None:
            return None

        index_table = self._h5.get_node(self._NODE_CACHE_FEATURE_INDEX)
        row = index_table[cache_row]
        n_atoms = int(row["n_atoms"])
        n_features = int(row["n_features"])
        values = np.asarray(
            self._h5.get_node(self._NODE_CACHE_FEATURE_VALUES)[cache_row]
        ).reshape(n_atoms, n_features)
        return torch.from_numpy(values).to(self.descriptor.dtype)

    def _load_runtime_or_persisted_features(
        self,
        idx: int,
        *,
        feature_cache: dict | None,
        cache_features: bool,
    ) -> torch.Tensor | None:
        """
        Load raw features from the runtime cache or persisted HDF5 payload.

        Runtime cache entries take precedence when ``cache_features=True``.
        Otherwise, compatible persisted HDF5 features are loaded lazily.
        """
        if cache_features and feature_cache is not None and idx in feature_cache:
            return feature_cache[idx]

        if self._feature_cache_info is None:
            return None

        features = self.load_persisted_features(idx)
        if (
            features is not None
            and cache_features
            and feature_cache is not None
        ):
            feature_cache[idx] = features
        return features

    def has_persisted_force_derivatives(self) -> bool:
        """Return True when the HDF5 file contains persisted derivative payloads."""
        if self._h5 is None and os.path.exists(self._db_path):
            self._open_readonly()
        return bool(self._force_derivative_rows_by_entry)

    def get_force_derivative_cache_info(self) -> dict | None:
        """
        Return metadata describing the persisted derivative-cache schema.

        The returned metadata is descriptive only. Compatibility with the
        current descriptor is validated when derivative payloads are loaded.
        """
        if self._h5 is None and os.path.exists(self._db_path):
            self._open_readonly()
        if self._force_derivative_cache_info is None:
            return None
        return dict(self._force_derivative_cache_info)

    def load_persisted_force_derivatives(
        self,
        idx: int,
    ) -> dict[str, dict[str, torch.Tensor | None]] | None:
        """
        Load persisted local derivatives for one dataset entry, if available.

        This helper is used directly by HDF5 sample materialization when the
        runtime policy requests persisted local derivatives for compatible
        force-supervised samples.
        """
        if self._h5 is None:
            self._open_readonly()
        info = self._validate_force_derivative_cache_compatibility()
        if self._force_derivative_node_paths is None:
            raise RuntimeError("Missing force-derivative node layout metadata.")
        cache_row = self._force_derivative_rows_by_entry.get(int(idx))
        if cache_row is None:
            return None

        node_paths = self._force_derivative_node_paths
        index_table = self._h5.get_node(node_paths["index"])
        row = index_table[cache_row]
        n_radial_edges = int(row["n_radial_edges"])
        n_angular_triplets = int(row["n_angular_triplets"])
        n_radial_features = int(info["n_radial_features"])
        n_angular_features = int(info["n_angular_features"])

        radial_center = np.asarray(
            self._h5.get_node(node_paths["radial_center"])[
                cache_row
            ],
            dtype=np.int64,
        )
        radial_neighbor = np.asarray(
            self._h5.get_node(node_paths["radial_neighbor"])[
                cache_row
            ],
            dtype=np.int64,
        )
        radial_grads = np.asarray(
            self._h5.get_node(node_paths["radial_grads"])[
                cache_row
            ]
        ).reshape(n_radial_edges, n_radial_features, 3)
        radial_typespin = np.asarray(
            self._h5.get_node(node_paths["radial_typespin"])[
                cache_row
            ]
        )

        angular_center = np.asarray(
            self._h5.get_node(node_paths["angular_center"])[
                cache_row
            ],
            dtype=np.int64,
        )
        angular_j = np.asarray(
            self._h5.get_node(node_paths["angular_neighbor_j"])[
                cache_row
            ],
            dtype=np.int64,
        )
        angular_k = np.asarray(
            self._h5.get_node(node_paths["angular_neighbor_k"])[
                cache_row
            ],
            dtype=np.int64,
        )
        angular_grads_i = np.asarray(
            self._h5.get_node(node_paths["angular_grads_i"])[
                cache_row
            ]
        ).reshape(n_angular_triplets, n_angular_features, 3)
        angular_grads_j = np.asarray(
            self._h5.get_node(node_paths["angular_grads_j"])[
                cache_row
            ]
        ).reshape(n_angular_triplets, n_angular_features, 3)
        angular_grads_k = np.asarray(
            self._h5.get_node(node_paths["angular_grads_k"])[
                cache_row
            ]
        ).reshape(n_angular_triplets, n_angular_features, 3)
        angular_typespin = np.asarray(
            self._h5.get_node(node_paths["angular_typespin"])[
                cache_row
            ]
        )

        radial_block: dict[str, torch.Tensor | None] = {
            "center_idx": torch.from_numpy(radial_center).to(torch.int64),
            "neighbor_idx": torch.from_numpy(radial_neighbor).to(torch.int64),
            "dG_drij": torch.from_numpy(radial_grads).to(self.descriptor.dtype),
            "neighbor_typespin": (
                torch.from_numpy(radial_typespin).to(self.descriptor.dtype)
                if bool(info["multi"])
                else None
            ),
        }
        angular_block: dict[str, torch.Tensor | None] = {
            "center_idx": torch.from_numpy(angular_center).to(torch.int64),
            "neighbor_j_idx": torch.from_numpy(angular_j).to(torch.int64),
            "neighbor_k_idx": torch.from_numpy(angular_k).to(torch.int64),
            "grads_i": torch.from_numpy(angular_grads_i).to(
                self.descriptor.dtype
            ),
            "grads_j": torch.from_numpy(angular_grads_j).to(
                self.descriptor.dtype
            ),
            "grads_k": torch.from_numpy(angular_grads_k).to(
                self.descriptor.dtype
            ),
            "triplet_typespin": (
                torch.from_numpy(angular_typespin).to(self.descriptor.dtype)
                if bool(info["multi"])
                else None
            ),
        }
        return {"radial": radial_block, "angular": angular_block}

    def _close_handle(self) -> None:
        """Close any open HDF5 handle."""
        try:
            if self._h5 is not None:
                self._h5.close()
        finally:
            self._h5 = None
            self._descriptor_manifest_info = None
            self._torch_cache_info = None
            self._feature_rows_by_entry = {}
            self._feature_cache_info = None
            self._force_derivative_rows_by_entry = {}
            self._force_derivative_cache_info = None
            self._force_derivative_node_paths = None

    def _require_descriptor(self, action: str) -> None:
        """Ensure a descriptor is available for descriptor-dependent paths."""
        if self.descriptor is not None:
            return
        if self._h5 is None and os.path.exists(self._db_path):
            self._open_readonly()
        if self.descriptor is not None:
            return
        raise RuntimeError(
            f"{action} requires a descriptor. Provide descriptor=... or "
            "persist a descriptor manifest with "
            "build_database(persist_descriptor=True)."
        )

    def close(self) -> None:
        """Close any open HDF5 handle held by this dataset instance."""
        self._close_handle()

    def __getstate__(self) -> dict:
        """
        Return picklable dataset state for DataLoader workers.

        Non-picklable file handles are removed and each worker will reopen its
        own read-only HDF5 handle on demand.
        """
        state = self.__dict__.copy()
        state["_h5"] = None  # drop handle
        # LRU cache resets in new process
        state["_cache"] = _LRU(maxlen=self._cache_capacity)
        return state

    # -------------------- Dataset protocol --------------------

    def __len__(self) -> int:
        if self._n_entries is None:
            # Not opened yet; in 'auto' or 'build' mode without load
            if os.path.exists(self._db_path):
                self._open_readonly()
            else:
                return 0
        return int(self._n_entries or 0)

    def get_structure(self, idx: int) -> Structure:
        """Load and return a single Structure from the HDF5 store."""
        if self._h5 is None:
            self._open_readonly()

        struct = self._cache_get(idx)
        if struct is None:
            vl = self._h5.get_node(self._NODE_STRUCTURES)
            data = np.array(vl[idx], copy=False)
            struct = pickle.loads(data.tobytes())
            self._cache_put(idx, struct)
        return struct

    def get_entry_metadata(self, idx: int) -> dict[str, object]:
        """
        Return persisted metadata for one HDF5 entry.

        The metadata is read lazily from the ``/entries/meta`` table and is
        intended for output helpers that need stable source identifiers
        without changing the serialized ``Structure`` payload.
        """
        if self._h5 is None:
            self._open_readonly()

        meta = self._h5.get_node(self._NODE_META)
        row = meta[int(idx)]
        return {
            "source_id": _decode_meta_text(row["source_id"]),
            "frame_idx": int(row["frame_idx"]),
            "source_kind": _decode_meta_text(row["source_kind"]),
            "display_name": _decode_meta_text(row["display_name"]),
            "name": _decode_meta_text(row["name"]),
            "has_forces": bool(row["has_forces"]),
            "n_atoms": int(row["n_atoms"]),
            "energy": float(row["energy"]),
        }

    def get_structure_identifier(self, idx: int) -> str:
        """
        Return the stable energy-output identifier for one HDF5 entry.

        HDF5-backed outputs synthesize the Fortran-compatible merged
        identifier from structured source metadata and always append the
        persisted frame index so multi-frame sources remain distinguishable.
        """
        meta = self.get_entry_metadata(idx)
        frame = int(meta["frame_idx"])
        base = (
            str(meta["display_name"])
            or str(meta["source_id"])
            or str(meta["name"])
            or f"structure_{idx:06d}"
        )
        return f"{base}#frame={frame}"

    def get_force_indices(self) -> list[int]:
        """Return indices of entries that carry force labels."""
        if self._force_indices_all is None and os.path.exists(self._db_path):
            self._open_readonly()
        return list(self._force_indices_all or [])

    def materialize_sample(
        self,
        idx: int,
        *,
        use_forces: bool,
        cache_state=None,
        cache_features: bool = False,
        cache_neighbors: bool = False,
        cache_force_triplets: bool = False,
        load_local_derivatives: bool = False,
    ) -> dict:
        """
        Materialize one HDF5-backed sample under an explicit runtime policy.

        Parameters
        ----------
        idx : int
            Dataset index to materialize.
        use_forces : bool
            Whether this sample should expose the force-training path.
        cache_state : object, optional
            Runtime cache owner providing ``feature_cache``,
            ``neighbor_cache``, and ``graph_cache`` dictionaries.
        cache_features : bool, optional
            Whether to cache energy-view features in ``cache_state``. When
            enabled, the runtime cache takes precedence over persisted HDF5
            features for repeated energy-view accesses.
        cache_neighbors : bool, optional
            Whether to cache neighbor payloads in ``cache_state``.
        cache_force_triplets : bool, optional
            Whether to cache graph/triplet payloads in ``cache_state``.
        load_local_derivatives : bool, optional
            Whether to attach persisted local derivatives when available. When
            both persisted features and persisted derivatives exist for a
            force-supervised sample, the sample reuses those payloads directly
            and does not build graph/triplet data.
        """
        self._require_descriptor("Materializing HDF5 samples")
        struct = self.get_structure(idx)
        use_forces = bool(use_forces and struct.has_forces())

        feature_cache, neighbor_cache, graph_cache = extract_runtime_caches(
            cache_state
        )
        prepared = prepare_structure_tensors(struct, self.descriptor)

        graph = None
        triplets = None
        local_derivatives = None

        if use_forces:
            features, graph, triplets, local_derivatives = (
                materialize_force_view(
                    idx,
                    descriptor=self.descriptor,
                    positions=prepared.positions,
                    cell=prepared.cell,
                    pbc=prepared.pbc,
                    species_indices=prepared.species_indices,
                    graph_cache=graph_cache,
                    cache_force_triplets=cache_force_triplets,
                    load_persisted_features=(
                        lambda entry_idx: self._load_runtime_or_persisted_features(
                            entry_idx,
                            feature_cache=None,
                            cache_features=False,
                        )
                    ),
                    load_local_derivatives=(
                        self.load_persisted_force_derivatives
                        if (
                            load_local_derivatives
                            and self._force_derivative_cache_info is not None
                        )
                        else None
                    ),
                )
            )
        else:
            features = load_energy_view_features(
                idx,
                descriptor=self.descriptor,
                positions=prepared.positions,
                species=struct.species,
                cell=prepared.cell,
                pbc=prepared.pbc,
                feature_cache=feature_cache,
                cache_features=cache_features,
                neighbor_cache=neighbor_cache,
                cache_neighbors=cache_neighbors,
                load_persisted_features=(
                    lambda entry_idx: self._load_runtime_or_persisted_features(
                        entry_idx,
                        feature_cache=feature_cache,
                        cache_features=cache_features,
                    )
                ),
            )

        return build_sample_dict(
            struct=struct,
            idx=idx,
            prepared=prepared,
            features=features,
            use_forces=use_forces,
            graph=graph,
            triplets=triplets,
            local_derivatives=local_derivatives,
            fallback_name_prefix="entry_",
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Load a single entry (Structure) from the HDF5 database and featurize.

        Direct dataset access treats every force-labeled structure as fully
        force-supervised. Runtime training-policy selection and cache reuse are
        applied by the trainer-side wrapper, not by this passive data source.
        """
        struct = self.get_structure(idx)
        return self.materialize_sample(
            idx,
            use_forces=struct.has_forces(),
            load_local_derivatives=struct.has_forces(),
        )

    # -------------------- Simple LRU for unpickled Structures --------

    def _cache_get(self, idx: int) -> Structure | None:
        if self._cache_capacity <= 0:
            return None
        return self._cache.get(idx)  # type: ignore[return-value]

    def _cache_put(self, idx: int, struct: Structure) -> None:
        if self._cache_capacity <= 0:
            return
        self._cache[idx] = struct  # type: ignore[index]

    # -------------------- Context manager support --------------------

    def __enter__(self) -> HDF5StructureDataset:
        """Return the dataset instance for context-manager usage."""
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        """Close the HDF5 handle when leaving a context-manager block."""
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _LRU(dict):
    """
    Tiny LRU cache using dict+manual eviction for predictable behavior.

    This is not a full OrderedDict implementation; eviction happens only
    when inserting new keys beyond capacity. Access does not reorder entries.
    Good enough for a lightweight within-worker cache of moderate size.
    """

    def __init__(self, maxlen: int):
        super().__init__()
        self._maxlen = maxlen
        self._keys: list[int] = []

    def get(self, key: int, default=None):
        return super().get(key, default)

    def __setitem__(self, key: int, value):
        if key not in self:
            self._keys.append(key)
        super().__setitem__(key, value)
        self._evict_if_needed()

    def _evict_if_needed(self):
        while self._maxlen > 0 and len(self._keys) > self._maxlen:
            k = self._keys.pop(0)
            try:
                super().pop(k, None)
            except KeyError:
                pass


# -------------------- Generic dataset splitter --------------------


def train_test_split_dataset(
    dataset: Dataset,
    test_fraction: float = 0.1,
    seed: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Split any PyTorch Dataset into training and test Subsets.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split. It must implement __len__ and __getitem__.
    test_fraction : float, optional
        Fraction of entries to use for the test set in [0,1].
        Default: 0.1
    seed : int, optional
        Random seed for reproducibility. Default: None

    Returns
    -------
    train_subset : Dataset
        A Subset instance containing the training indices.
    test_subset : Dataset
        A Subset instance containing the test indices.
    """
    if seed is not None:
        random.seed(seed)

    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    n_test = int(n * float(test_fraction))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return Subset(dataset, train_idx), Subset(dataset, test_idx)
