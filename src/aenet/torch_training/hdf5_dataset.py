"""
HDF5-backed Dataset and generic dataset utilities for torch training.

This module provides:
- HDF5StructureDataset: a database-backed lazy-loading PyTorch Dataset
  that stores serialized (pickled) torch Structure objects in an HDF5
  (PyTables) file with per-entry metadata. It supports building the database
  from a list of raw structure files via a user-provided parser callable,
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
"""

from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import tables  # PyTables
import torch
from torch.utils.data import Dataset, Subset

from .config import Structure  # Torch Structure dataclass
from aenet.torch_featurize.graph import (
    build_csr_from_neighborlist,
    build_triplets_from_csr,
)

__all__ = [
    "HDF5StructureDataset",
    "train_test_split_dataset",
]


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
          columns: path(str), frame(int), has_forces(bool),
                   n_atoms(int32), energy(float64), name(str)

    At training time, items are read from the HDF5 file, unpickled into
    Structure objects, and featurized on-the-fly using the provided
    descriptor. The sample dict matches StructureDataset.__getitem__.

    Parameters
    ----------
    descriptor : ChebyshevDescriptor
        Descriptor instance for featurization (dtype/device/species_index map).
    database_file : str
        Path to HDF5 database file. Will be created on build.
    file_paths : Sequence[str], optional
        List of raw structure file paths used for building the database.
        Required if you plan to call build_database().
    parser : Callable[[str], Structure], optional
        Callable that, given a file path, returns a Structure (or a list of
        Structures). If a list is returned, all frames are added as separate
        entries. If None, a sensible default should be provided by user code.
        Note: this callable must be a top-level function if used by
        multiprocessing (i.e., avoid lambdas that can't be pickled).
    mode : str, optional
        One of {'auto', 'build', 'load'} controlling initialization behavior:
        - 'auto': if database_file exists, load; otherwise, expect that user
                  will call build_database() before reading.
        - 'build': do nothing on init; user must call build_database() to
                   create/overwrite the database.
        - 'load': open in read-only mode immediately; error if missing.
        Default: 'auto'
    force_fraction : float, optional
        Fraction of force-labeled structures to select per epoch/window.
        Default: 1.0 (use all force structures).
    force_sampling : str, optional
        'random' or 'fixed'. Fixed chooses once at init (based on metadata).
        Random allows per-epoch resampling via resample_force_structures().
        Default: 'random'
    min_force_structures_per_epoch : int, optional
        Minimum count of force-labeled structures to select regardless of
        force_fraction. Default: None (no minimum).
    cache_features : bool, optional
        Cache features for structures not selected for forces. Default: False
    cache_force_neighbors : bool, optional
        Cache per-structure neighbor_info for reuse. Only relevant for force
        training. Default: False
    cache_force_triplets : bool, optional
        Cache per-structure CSR graphs + triplets for vectorized paths.
        Only relevant for force training.
        Default: False
    cache_persist_dir : str, optional
        Placeholder for future on-disk persistence of caches. Default: None
    seed : int, optional
        Random seed for deterministic sampling. Default: None
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
    - The parser is not used during read mode; it's only used for building.
    """

    # --- HDF5 node paths
    _GROUP_ENTRIES = "/entries"
    _NODE_STRUCTURES = "/entries/structures"
    _NODE_META = "/entries/meta"

    # --- metadata table schema
    class _MetaRow(tables.IsDescription):
        path = tables.StringCol(1024)
        frame = tables.Int64Col()
        has_forces = tables.BoolCol()
        n_atoms = tables.Int32Col()
        energy = tables.Float64Col()
        name = tables.StringCol(256)

    def __init__(
        self,
        descriptor,
        database_file: str,
        file_paths: Optional[Sequence[str]] = None,
        parser: Optional[Callable[[str], Structure]] = None,
        mode: str = "auto",
        *,
        force_fraction: float = 1.0,
        force_sampling: str = "random",
        min_force_structures_per_epoch: Optional[int] = None,
        cache_features: bool = False,
        cache_force_neighbors: bool = False,
        cache_force_triplets: bool = False,
        cache_persist_dir: Optional[str] = None,
        seed: Optional[int] = None,
        in_memory_cache_size: int = 2048,
        compression: str = "zlib",
        compression_level: int = 5,
    ):
        # Descriptor and featurization flags
        self.descriptor = descriptor
        self.force_fraction = float(force_fraction)
        self.force_sampling = str(force_sampling)
        self.min_force_structures_per_epoch = min_force_structures_per_epoch
        self.cache_features = bool(cache_features)
        self.cache_force_neighbors = bool(cache_force_neighbors)
        self.cache_force_triplets = bool(cache_force_triplets)
        self.cache_persist_dir = cache_persist_dir

        # Random init
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build/read configuration
        self._db_path = str(database_file)
        self._file_paths = list(file_paths) if file_paths is not None else None
        self._parser = parser
        self._filters = _FiltersConfig(
            compression=compression,
            compression_level=int(compression_level),
        )

        # Runtime state
        self._h5: Optional[tables.File] = None  # lazily opened per process
        self._n_entries: Optional[int] = None
        self._force_indices_all: Optional[List[int]] = None
        self.selected_force_indices: Optional[List[int]] = None

        # Simple per-process LRU cache for unpickled Structures
        self._cache_capacity = max(0, int(in_memory_cache_size))
        self._cache: dict = _LRU(maxlen=self._cache_capacity)

        # Feature cache for energy-only path (mirrors StructureDataset)
        self._feature_cache: dict[int, torch.Tensor] = {}
        self._neighbor_cache: dict[int, dict] = {}
        self._graph_cache: dict[int, dict] = {}

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

    def build_database(self, show_progress: bool = True) -> None:
        """
        Build (or overwrite) the HDF5 database from file_paths using parser.

        This will:
          - Create '/entries/structures' VLArray of uint8 (pickled Structure)
          - Create '/entries/meta' Table with per-entry metadata
          - Populate force index list for efficient selection

        Parameters
        ----------
        show_progress : bool
            If True and tqdm is available, show a progress bar.
        """
        if self._file_paths is None:
            raise ValueError("file_paths must be provided to build_database()")
        if self._parser is None:
            raise ValueError("parser must be provided to build_database()")

        # Ensure directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Close any open handles before writing
        self._close_handle()

        # Open HDF5 file for writing
        h5 = tables.open_file(self._db_path, mode="w",
                              filters=self._filters.to_tables_filters())
        try:
            # Create groups and nodes
            entries_group = h5.create_group(
                "/", "entries", "Serialized entries")
            vl_struct = h5.create_vlarray(
                entries_group, "structures",
                atom=tables.UInt8Atom(), title="Pickled Structures"
            )
            meta_table = h5.create_table(
                entries_group, "meta", description=self._MetaRow)

            # Optional progress bar
            pbar = None
            try:
                from tqdm import tqdm as _tqdm  # type: ignore
                pbar = _tqdm(total=len(self._file_paths),
                             desc="Building HDF5",
                             ncols=80) if show_progress else None
            except Exception:
                pbar = None

            # Iterate files and serialize Structures
            for path in self._file_paths:
                structures = self._parser(path)
                # Support parser returning a single Structure or a list
                structs: List[Structure]
                if (isinstance(structures, list)
                        or isinstance(structures, tuple)):
                    structs = list(structures)  # type: ignore[assignment]
                    # If AtomicStructure.to_TorchStructure() returns
                    # list of frames, keep all frames.
                else:
                    structs = [structures]  # type: ignore[list-item]

                # Append all frames as entries
                for frame_idx, struct in enumerate(structs):
                    # Pickle Structure to bytes
                    data_bytes = pickle.dumps(
                        struct, protocol=pickle.HIGHEST_PROTOCOL)
                    vl_struct.append(np.frombuffer(data_bytes, dtype=np.uint8))

                    # Metadata row
                    row = meta_table.row
                    row["path"] = str(path)
                    row["frame"] = int(frame_idx)
                    row["has_forces"] = bool(struct.has_forces())
                    row["n_atoms"] = int(struct.n_atoms)
                    # Some structures may miss energy; be defensive
                    try:
                        row["energy"] = float(struct.energy)
                    except Exception:
                        row["energy"] = float("nan")
                    row["name"] = (str(struct.name)
                                   if struct.name is not None else "")
                    row.append()

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            meta_table.flush()
            h5.flush()
        finally:
            h5.close()

        # Re-open read-only for later training use
        self._open_readonly()

    def _open_readonly(self) -> None:
        """Open the HDF5 database in read-only mode and initialize metadata."""
        self._h5 = tables.open_file(self._db_path, mode="r")
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

        # For fixed sampling, select once
        if self.force_sampling == "fixed" and len(self._force_indices_all) > 0:
            n_force = int(len(self._force_indices_all) * self.force_fraction)
            if self.min_force_structures_per_epoch is not None:
                n_force = max(n_force, self.min_force_structures_per_epoch)
            n_force = min(n_force, len(self._force_indices_all))
            self.selected_force_indices = (
                random.sample(self._force_indices_all, n_force)
                if n_force > 0 else []
            )
        else:
            self.selected_force_indices = None

    def _close_handle(self) -> None:
        """Close any open HDF5 handle."""
        try:
            if self._h5 is not None:
                self._h5.close()
        finally:
            self._h5 = None

    def __getstate__(self) -> dict:
        """
        Ensure dataset can be pickled for DataLoader workers by removing
        non-picklable file handles. Workers will reopen read-only handles.
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

    def __getitem__(self, idx: int) -> dict:
        """
        Load a single entry (Structure) from the HDF5 database and featurize.

        Returns a dict identical to StructureDataset.__getitem__.
        """
        if self._h5 is None:
            self._open_readonly()

        # Load or retrieve Structure from LRU cache
        struct = self._cache_get(idx)
        if struct is None:
            # Read pickled bytes from VLArray and unpickle
            vl = self._h5.get_node(self._NODE_STRUCTURES)
            data = np.array(vl[idx], copy=False)
            struct = pickle.loads(data.tobytes())
            self._cache_put(idx, struct)

        # Convert numpy arrays to tensors using descriptor dtype
        positions = torch.from_numpy(
            struct.positions).to(self.descriptor.dtype)
        cell = (
            torch.from_numpy(struct.cell).to(self.descriptor.dtype)
            if struct.cell is not None
            else None
        )
        pbc = torch.from_numpy(struct.pbc) if struct.pbc is not None else None

        # Determine whether to use forces for this index
        use_forces = self._should_use_forces(
            idx, struct_has_forces=struct.has_forces())

        # Compute features and neighbor information paths
        # (mirror StructureDataset)
        if use_forces:
            # Force-supervised path requires neighbor info
            if self.cache_force_neighbors and (idx in self._neighbor_cache):
                neighbor_info = self._neighbor_cache[idx]
                nb_idx_list_t = [
                    torch.as_tensor(arr, dtype=torch.long)
                    for arr in neighbor_info["neighbor_lists"]
                ]
                nb_vec_list_t = [
                    torch.as_tensor(vec, dtype=self.descriptor.dtype)
                    for vec in neighbor_info["neighbor_vectors"]
                ]
                features = self.descriptor.forward(
                    positions, struct.species, nb_idx_list_t, nb_vec_list_t)
            else:
                (features, neighbor_info
                 ) = self.descriptor.featurize_with_neighbor_info(
                    positions, struct.species, cell, pbc
                )
                if self.cache_force_neighbors:
                    self._neighbor_cache[idx] = neighbor_info

            # Optional CSR/Triplet vectorization cache
            graph = None
            triplets = None
            if self.cache_force_triplets:
                if idx in self._graph_cache:
                    g_trip = self._graph_cache[idx]
                else:
                    max_cut = float(max(self.descriptor.rad_cutoff,
                                        self.descriptor.ang_cutoff))
                    csr = build_csr_from_neighborlist(
                        positions=positions,
                        cell=cell,
                        pbc=pbc,
                        nbl=self.descriptor.nbl,
                        min_cutoff=float(self.descriptor.min_cutoff),
                        max_cutoff=max_cut,
                        device=positions.device,
                        dtype=self.descriptor.dtype,
                    )
                    trip = build_triplets_from_csr(
                        csr=csr,
                        ang_cutoff=float(self.descriptor.ang_cutoff),
                        min_cutoff=float(self.descriptor.min_cutoff),
                    )
                    g_trip = {"graph": csr, "triplets": trip}
                    self._graph_cache[idx] = g_trip
                graph = g_trip["graph"]
                triplets = g_trip["triplets"]
        else:
            # Energy-only forward path
            neighbor_info = None
            if self.cache_features and (idx in self._feature_cache):
                features = self._feature_cache[idx]
            else:
                if (self.cache_force_neighbors
                        and (idx in self._neighbor_cache)):
                    neighbor_info_cached = self._neighbor_cache[idx]
                    nb_idx_list_t = [
                        torch.as_tensor(arr, dtype=torch.long)
                        for arr in neighbor_info_cached["neighbor_lists"]
                    ]
                    nb_vec_list_t = [
                        torch.as_tensor(vec, dtype=self.descriptor.dtype)
                        for vec in neighbor_info_cached["neighbor_vectors"]
                    ]
                    features = self.descriptor.forward(
                        positions, struct.species, nb_idx_list_t, nb_vec_list_t
                    )
                else:
                    features = self.descriptor.forward_from_positions(
                        positions, struct.species, cell, pbc)
                    if self.cache_force_neighbors:
                        # Build once for reuse later
                        (feats, neighbor_info_cached
                         ) = self.descriptor.featurize_with_neighbor_info(
                            positions, struct.species, cell, pbc
                        )
                        self._neighbor_cache[idx] = neighbor_info_cached
                        features = feats
                if self.cache_features:
                    self._feature_cache[idx] = features
            graph = None
            triplets = None

        # Species indices tensor
        species_indices = torch.tensor(
            [self.descriptor.species_to_idx[s]
             for s in struct.species], dtype=torch.long
        )

        # Forces, when supervised
        forces = None
        if use_forces and struct.forces is not None:
            forces = torch.from_numpy(struct.forces).to(self.descriptor.dtype)

        sample = {
            "features": features,
            "neighbor_info": neighbor_info,
            "graph": graph,
            "triplets": triplets,
            "positions": positions,
            "species": struct.species,
            "species_indices": species_indices,
            "cell": cell,
            "pbc": pbc,
            "energy": float(struct.energy),
            "forces": forces,
            "has_forces": struct.has_forces(),
            "use_forces": use_forces,
            "n_atoms": int(struct.n_atoms),
            "name": (struct.name if struct.name is not None
                     else f"entry_{idx}"),
        }
        return sample

    # -------------------- Force selection controls --------------------

    def resample_force_structures(self) -> None:
        """
        Resample force-supervised structure indices for 'random' sampling.

        Should be called (by the trainer) at the start of an epoch or at a
        configured epoch-window boundary when using random sampling.
        """
        if self.force_sampling != "random":
            return
        if not self._force_indices_all:
            # If not initialized yet, open and scan metadata
            if self._h5 is None:
                self._open_readonly()
        n_force_total = len(self._force_indices_all or [])
        if n_force_total == 0:
            self.selected_force_indices = []
            return
        n_force = int(n_force_total * self.force_fraction)
        if self.min_force_structures_per_epoch is not None:
            n_force = max(n_force, self.min_force_structures_per_epoch)
        n_force = min(n_force, n_force_total)
        self.selected_force_indices = random.sample(
            self._force_indices_all, n_force) if n_force > 0 else []

    def _should_use_forces(self, idx: int, struct_has_forces: bool) -> bool:
        """Determine whether a given index should use force supervision."""
        if not struct_has_forces:
            return False
        if self.force_fraction >= 1.0:
            return True
        if self.selected_force_indices is None:
            return False
        return idx in self.selected_force_indices

    # -------------------- Simple LRU for unpickled Structures --------

    def _cache_get(self, idx: int) -> Optional[Structure]:
        if self._cache_capacity <= 0:
            return None
        return self._cache.get(idx)  # type: ignore[return-value]

    def _cache_put(self, idx: int, struct: Structure) -> None:
        if self._cache_capacity <= 0:
            return
        self._cache[idx] = struct  # type: ignore[index]

    # -------------------- Context manager support --------------------

    def __del__(self):
        try:
            self._close_handle()
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
        self._keys: List[int] = []

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
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
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
