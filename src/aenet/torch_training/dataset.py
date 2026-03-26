"""
PyTorch Dataset for on-the-fly structure featurization.

This module provides Dataset classes that featurize atomic structures
on-demand during training, avoiding the need to pre-compute and store
large feature arrays.
"""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ._materialization import (
    build_force_graph_triplets,
    build_sample_dict,
    extract_runtime_caches,
    filter_structures,
    forward_force_features_with_graph,
    load_energy_view_features,
    prepare_structure_tensors,
)
from .config import Structure

# Progress bar (match aenet.mlip behavior)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

# Optional import: HDF5-backed dataset and generic splitter.
# Wrapped in try/except to avoid import errors during partial installs.
try:
    from .hdf5_dataset import (
        HDF5StructureDataset,
        train_test_split_dataset,
    )
except Exception:  # pragma: no cover - fallback when optional file missing
    HDF5StructureDataset = None  # type: ignore
    train_test_split_dataset = None  # type: ignore

__all__ = [
    'StructureDataset',
    'CachedStructureDataset',
    'HDF5StructureDataset',
    'train_test_split_dataset',
    'convert_to_structures',
    ]


_build_force_graph_triplets = build_force_graph_triplets
_forward_force_features_with_graph = forward_force_features_with_graph


def convert_to_structures(
    inputs: Union[list[Structure], list, list[os.PathLike]],
) -> list[Structure]:
    """
    Convert various input types to List[Structure] (torch Structure objects).

    This utility function provides a unified interface for converting different
    input formats into the torch Structure format required by Dataset classes.

    Parameters
    ----------
    inputs : Union[List[Structure], List, List[os.PathLike]]
        Input structures in one of three formats:
        - List[Structure]: torch Structure objects (returned as-is)
        - List[AtomicStructure]: aenet.geometry.AtomicStructure objects
          (converted via to_TorchStructure())
        - List[os.PathLike]: File paths to structure files (loaded with
          AtomicStructure.from_file() and converted)

    Returns
    -------
    List[Structure]
        List of torch Structure objects

    Examples
    --------
    >>> # From file paths
    >>> structures = convert_to_structures(['file1.xsf', 'file2.xsf'])

    >>> # From AtomicStructure objects
    >>> from aenet.geometry import AtomicStructure
    >>> atomic_structs = [AtomicStructure.from_file('file.xsf')]
    >>> structures = convert_to_structures(atomic_structs)

    >>> # From torch Structures (no conversion needed)
    >>> structures = convert_to_structures(torch_structures)
    """
    if not inputs or len(inputs) == 0:
        return []

    # Import here to avoid circular dependencies
    from aenet.geometry import AtomicStructure
    from aenet.io.structure import read

    first = inputs[0]

    # Check type of first element to determine conversion strategy
    if isinstance(first, (str, os.PathLike)):
        # File paths - load and convert
        torch_structs: list[Structure] = []
        for path in inputs:
            atomic = read(path)
            converted = atomic.to_TorchStructure()
            path_str = os.fspath(path)
            # to_TorchStructure() may return list or single Structure
            if isinstance(converted, (list, tuple)):
                for struct in converted:
                    if getattr(struct, "name", None) in (None, ""):
                        struct.name = path_str
                torch_structs.extend(converted)
            else:
                if getattr(converted, "name", None) in (None, ""):
                    converted.name = path_str
                torch_structs.append(converted)
        return torch_structs

    elif isinstance(first, AtomicStructure):
        # AtomicStructure objects - convert
        torch_structs: list[Structure] = []
        for atomic in inputs:
            converted = atomic.to_TorchStructure()
            if isinstance(converted, (list, tuple)):
                torch_structs.extend(converted)
            else:
                torch_structs.append(converted)
        return torch_structs

    else:
        # Already torch Structure objects
        return list(inputs)


class StructureDataset(Dataset):
    """
    PyTorch Dataset that featurizes structures on-demand.

    This dataset stores raw atomic structures (positions, species, energies,
    forces) and computes features dynamically during training. This approach
    is more memory-efficient than pre-computing all features, and enables
    efficient force training using semi-analytical gradients.

    Parameters
    ----------
    structures : Union[List[Structure], List[AtomicStructure],
                     List[os.PathLike]]
        Structures to include in the dataset. Accepts:
        - List[Structure]: torch Structure objects (used directly)
        - List[AtomicStructure]: AtomicStructure objects
          (converted automatically)
        - List[os.PathLike]: File paths to structure files
          (loaded and converted)
    descriptor : ChebyshevDescriptor
        Descriptor instance for featurization
    max_energy : float, optional
        Exclude structures with energy per atom above this threshold.
        Default: None (no filtering)
    max_forces : float, optional
        Exclude structures with max force component above this threshold.
        Default: None (no filtering)
    seed : int, optional
        Reserved for deterministic helper utilities. Dataset contents and
        runtime training policy are unaffected by this value. Default: None

    Attributes
    ----------
    structures : List[Structure]
        Filtered list of structures in the dataset
    force_structures : List[int]
        Indices of structures with force data
    energy_only_structures : List[int]
        Indices of structures without force data
    """

    def __init__(
        self,
        structures: Union[list[Structure], list, list[os.PathLike]],
        descriptor,  # ChebyshevDescriptor - avoid circular import
        max_energy: Optional[float] = None,
        max_forces: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.descriptor = descriptor
        self.max_energy = max_energy
        self.max_forces = max_forces
        self.seed = seed

        # Convert input to torch Structure objects if needed
        structures = convert_to_structures(structures)

        # Filter structures based on energy and force thresholds
        self.structures = self._filter_structures(structures)

        # Separate structures with and without forces
        self.force_structures = []
        self.energy_only_structures = []

        for idx, struct in enumerate(self.structures):
            if struct.has_forces():
                self.force_structures.append(idx)
            else:
                self.energy_only_structures.append(idx)

    def _filter_structures(
        self,
        structures: list[Structure]
    ) -> list[Structure]:
        """
        Filter structures based on energy and force thresholds.

        Parameters
        ----------
        structures : List[Structure]
            Input structures

        Returns
        -------
        List[Structure]
            Filtered structures
        """
        return filter_structures(
            structures,
            max_energy=self.max_energy,
            max_forces=self.max_forces,
        )

    def get_structure(self, idx: int) -> Structure:
        """Return the filtered torch-training structure at ``idx``."""
        return self.structures[idx]

    def get_force_indices(self) -> list[int]:
        """Return indices of structures that carry force labels."""
        return list(self.force_structures)

    def __len__(self) -> int:
        """Return number of structures in dataset."""
        return len(self.structures)

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
        Materialize one dataset sample under an explicit runtime policy.

        Parameters
        ----------
        idx : int
            Structure index
        use_forces : bool
            Whether this sample should expose the force-training path.
        cache_state : object, optional
            Runtime cache owner providing ``feature_cache``,
            ``neighbor_cache``, and ``graph_cache`` dictionaries.
        cache_features : bool, optional
            Whether to cache energy-view features in ``cache_state``.
        cache_neighbors : bool, optional
            Whether to cache neighbor payloads in ``cache_state``.
        cache_force_triplets : bool, optional
            Whether to cache graph/triplet payloads in ``cache_state``.
        load_local_derivatives : bool, optional
            Accepted for API parity with HDF5-backed datasets. Ignored.

        Returns
        -------
        dict
            Sample dictionary in the trainer-compatible format.
        """
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
            graph_trip = (
                graph_cache.get(idx)
                if cache_force_triplets and graph_cache is not None
                else None
            )
            if graph_trip is None:
                graph_trip = _build_force_graph_triplets(
                    descriptor=self.descriptor,
                    positions=prepared.positions,
                    cell=prepared.cell,
                    pbc=prepared.pbc,
                )
                if cache_force_triplets and graph_cache is not None:
                    graph_cache[idx] = graph_trip
            graph = graph_trip["graph"]
            triplets = graph_trip["triplets"]
            features = _forward_force_features_with_graph(
                descriptor=self.descriptor,
                positions=prepared.positions,
                species_indices=prepared.species_indices,
                graph=graph,
                triplets=triplets,
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
            fallback_name_prefix="struct_",
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single structure with on-the-fly featurization.

        Direct dataset access treats every force-labeled structure as fully
        force-supervised. Runtime training-policy selection and cache reuse are
        applied by the trainer-side wrapper, not by this passive data source.
        """
        struct = self.get_structure(idx)
        return self.materialize_sample(
            idx,
            use_forces=struct.has_forces(),
        )

    def get_statistics(self) -> dict:
        """
        Get dataset statistics.

        Returns
        -------
        dict
            Dictionary with dataset statistics:
            - 'n_total': total number of structures
            - 'n_force': number of structures with forces
            - 'n_energy_only': number of energy-only structures
            - 'total_atoms': total number of atoms across all structures
            - 'avg_atoms': average atoms per structure
            - 'species': set of all species in dataset
        """
        total_atoms = sum(s.n_atoms for s in self.structures)
        species_set = set()
        for s in self.structures:
            species_set.update(s.species)

        return {
            'n_total': len(self.structures),
            'n_force': len(self.force_structures),
            'n_energy_only': len(self.energy_only_structures),
            'n_force_selected': len(self.force_structures),
            'total_atoms': total_atoms,
            'avg_atoms': (total_atoms / len(self.structures)
                          if self.structures else 0),
            'species': sorted(species_set),
        }

    def warmup_caches(self, show_progress: bool = True):
        """
        No-op retained for compatibility with older call sites.

        Parameters
        ----------
        show_progress : bool, optional
            Unused. Trainer-side wrappers own runtime caches.
        """
        return None


class CachedStructureDataset(Dataset):
    """
    Dataset that caches per-structure features for energy-only training.

    This avoids per-epoch featurization cost by computing features once
    at initialization. Force-related fields are disabled (use_forces=False).

    Parameters
    ----------
    structures : Union[List[Structure], List[AtomicStructure],
                     List[os.PathLike]]
        Structures to include in the dataset. Accepts:
        - List[Structure]: torch Structure objects (used directly)
        - List[AtomicStructure]: AtomicStructure objects
          (converted automatically)
        - List[os.PathLike]: File paths to structure files
          (loaded and converted)
    descriptor : ChebyshevDescriptor
        Descriptor instance for featurization
    max_energy : float, optional
        Exclude structures with energy per atom above this threshold.
        Default: None (no filtering)
    max_forces : float, optional
        Exclude structures with max force component above this threshold.
        Default: None (no filtering)
    seed : int, optional
        Random seed for reproducibility. Default: None
    """

    def __init__(
        self,
        structures: Union[list[Structure], list, list[os.PathLike]],
        descriptor,  # ChebyshevDescriptor
        max_energy: Optional[float] = None,
        max_forces: Optional[float] = None,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ):
        self.descriptor = descriptor
        self.max_energy = max_energy
        self.max_forces = max_forces
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Convert input to torch Structure objects if needed
        structures = convert_to_structures(structures)

        # Filter structures using the same logic as StructureDataset
        self.structures = filter_structures(
            structures,
            max_energy=self.max_energy,
            max_forces=self.max_forces,
        )

        # Build cached samples with progress feedback
        self._cached: list[dict] = []

        # Wrap structure iteration with progress bar
        structure_iter = self.structures
        if show_progress and tqdm is not None:
            structure_iter = tqdm(
                self.structures,
                desc="Caching features",
                ncols=80,
                leave=False
            )

        with torch.no_grad():
            for i, struct in enumerate(structure_iter):
                prepared = prepare_structure_tensors(struct, descriptor)
                features = load_energy_view_features(
                    i,
                    descriptor=descriptor,
                    positions=prepared.positions,
                    species=struct.species,
                    cell=prepared.cell,
                    pbc=prepared.pbc,
                    feature_cache=None,
                    cache_features=False,
                    neighbor_cache=None,
                    cache_neighbors=False,
                )
                sample = {
                    'features': features,
                    'species_indices': prepared.species_indices,
                    'n_atoms': int(struct.n_atoms),
                    'energy': float(struct.energy),
                    'species': struct.species,
                    # Force-related fields disabled for cached energy-only path
                    'positions': None,
                    'forces': None,
                    'neighbor_info': None,
                    'use_forces': False,
                    'cell': prepared.cell,
                    'pbc': prepared.pbc,
                    'name': (struct.name if struct.name is not None
                             else f"struct_{i}"),
                }
                self._cached.append(sample)

    def __len__(self) -> int:
        return len(self._cached)

    def __getitem__(self, idx: int) -> dict:
        return self._cached[idx]


def train_test_split(dataset: StructureDataset,
                     test_fraction: float = 0.1,
                     seed: Optional[int] = None,
                     ) -> tuple[StructureDataset, StructureDataset]:
    """
    Split dataset into training and test sets.

    Parameters
    ----------
    dataset : StructureDataset
        Dataset to split
    test_fraction : float, optional
        Fraction of data for test set (0-1). Default: 0.1
    seed : int, optional
        Random seed for reproducible split. Default: None

    Returns
    -------
    train_dataset : StructureDataset
        Training dataset
    test_dataset : StructureDataset
        Test dataset
    """
    if seed is not None:
        random.seed(seed)

    # Shuffle indices
    indices = list(range(len(dataset.structures)))
    random.shuffle(indices)

    # Split
    n_test = int(len(indices) * test_fraction)
    test_indices = set(indices[:n_test])
    train_indices = set(indices[n_test:])

    # Create new structure lists
    train_structures = [
        dataset.structures[i] for i in sorted(train_indices)
    ]
    test_structures = [
        dataset.structures[i] for i in sorted(test_indices)
    ]

    # Create new datasets with the same passive filtering parameters.
    train_dataset = StructureDataset(
        structures=train_structures,
        descriptor=dataset.descriptor,
        max_energy=dataset.max_energy,
        max_forces=dataset.max_forces,
        seed=seed,
    )

    test_dataset = StructureDataset(
        structures=test_structures,
        descriptor=dataset.descriptor,
        max_energy=dataset.max_energy,
        max_forces=dataset.max_forces,
        seed=seed,
    )

    return train_dataset, test_dataset
