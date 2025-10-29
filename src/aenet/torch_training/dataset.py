"""
PyTorch Dataset for on-the-fly structure featurization.

This module provides Dataset classes that featurize atomic structures
on-demand during training, avoiding the need to pre-compute and store
large feature arrays.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Structure

__all__ = ['StructureDataset', 'CachedStructureDataset']


class StructureDataset(Dataset):
    """
    PyTorch Dataset that featurizes structures on-demand.

    This dataset stores raw atomic structures (positions, species, energies,
    forces) and computes features dynamically during training. This approach
    is more memory-efficient than pre-computing all features, and enables
    efficient force training using semi-analytical gradients.

    Parameters
    ----------
    structures : List[Structure]
        List of atomic structures to include in the dataset
    descriptor : ChebyshevDescriptor
        Descriptor instance for featurization
    force_fraction : float, optional
        Fraction of structures with forces to use for force training (0-1).
        Default: 1.0 (use all structures with forces)
    force_sampling : str, optional
        Force sampling strategy:
        - 'random': Randomly sample force structures each epoch
        - 'fixed': Use a fixed subset of force structures
        Default: 'random'
    max_energy : float, optional
        Exclude structures with energy per atom above this threshold.
        Default: None (no filtering)
    max_forces : float, optional
        Exclude structures with max force component above this threshold.
        Default: None (no filtering)
    seed : int, optional
        Random seed for reproducible force sampling. Default: None

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
        structures: List[Structure],
        descriptor,  # ChebyshevDescriptor - avoid circular import
        force_fraction: float = 1.0,
        force_sampling: str = 'random',
        max_energy: Optional[float] = None,
        max_forces: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.descriptor = descriptor
        self.force_fraction = force_fraction
        self.force_sampling = force_sampling
        self.max_energy = max_energy
        self.max_forces = max_forces
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

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

        # For fixed sampling, select force structures once
        if self.force_sampling == 'fixed' and len(self.force_structures) > 0:
            n_force = int(len(self.force_structures) * self.force_fraction)
            self.selected_force_indices = random.sample(
                self.force_structures, n_force
            )
        else:
            self.selected_force_indices = None

    def _filter_structures(
        self,
        structures: List[Structure]
    ) -> List[Structure]:
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
        filtered = []

        for struct in structures:
            # Check energy threshold
            if self.max_energy is not None:
                energy_per_atom = struct.energy / struct.n_atoms
                if energy_per_atom > self.max_energy:
                    continue

            # Check force threshold
            if self.max_forces is not None and struct.has_forces():
                max_force = np.abs(struct.forces).max()
                if max_force > self.max_forces:
                    continue

            filtered.append(struct)

        return filtered

    def resample_force_structures(self):
        """
        Resample force structures for random sampling mode.

        Should be called at the beginning of each epoch when using
        force_sampling='random'.
        """
        if self.force_sampling == 'random' and len(self.force_structures) > 0:
            n_force = int(len(self.force_structures) * self.force_fraction)
            self.selected_force_indices = random.sample(
                self.force_structures, n_force
            )

    def should_use_forces(self, idx: int) -> bool:
        """
        Determine if forces should be used for a given structure.

        Parameters
        ----------
        idx : int
            Structure index

        Returns
        -------
        bool
            True if forces should be used for this structure
        """
        if not self.structures[idx].has_forces():
            return False

        if self.force_fraction >= 1.0:
            return True

        if self.selected_force_indices is None:
            return False

        return idx in self.selected_force_indices

    def __len__(self) -> int:
        """Return number of structures in dataset."""
        return len(self.structures)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single structure with on-the-fly featurization.

        Parameters
        ----------
        idx : int
            Structure index

        Returns
        -------
        dict
            Dictionary containing:
            - 'features': (N, F) feature tensor
            - 'neighbor_info': neighbor data for gradient computation
            - 'positions': (N, 3) positions tensor
            - 'species': List[str] species names
            - 'species_indices': (N,) species index tensor
            - 'cell': (3, 3) cell tensor or None
            - 'pbc': (3,) pbc tensor or None
            - 'energy': float total energy
            - 'forces': (N, 3) forces tensor or None
            - 'use_forces': bool whether to use forces for this structure
            - 'n_atoms': int number of atoms
            - 'name': str structure name
        """
        struct = self.structures[idx]

        # Convert to torch tensors
        positions = torch.from_numpy(struct.positions).to(torch.float64)
        cell = (
            torch.from_numpy(struct.cell).to(torch.float64)
            if struct.cell is not None else None
        )
        pbc = (
            torch.from_numpy(struct.pbc)
            if struct.pbc is not None else None
        )

        # Compute features and neighbor information
        features, neighbor_info = self.descriptor.featurize_with_neighbor_info(
            positions, struct.species, cell, pbc
        )

        # Convert species to indices
        species_indices = torch.tensor(
            [self.descriptor.species_to_idx[s] for s in struct.species],
            dtype=torch.long
        )

        # Prepare forces
        forces = None
        use_forces = self.should_use_forces(idx)
        if use_forces and struct.forces is not None:
            forces = torch.from_numpy(struct.forces).to(torch.float64)

        return {
            'features': features,
            'neighbor_info': neighbor_info,
            'positions': positions,
            'species': struct.species,
            'species_indices': species_indices,
            'cell': cell,
            'pbc': pbc,
            'energy': struct.energy,
            'forces': forces,
            'use_forces': use_forces,
            'n_atoms': struct.n_atoms,
            'name': (struct.name if struct.name is not None
                     else f"struct_{idx}"),
        }

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
            - 'n_force_selected': number of force structures being used
            - 'total_atoms': total number of atoms across all structures
            - 'avg_atoms': average atoms per structure
            - 'species': set of all species in dataset
        """
        total_atoms = sum(s.n_atoms for s in self.structures)
        species_set = set()
        for s in self.structures:
            species_set.update(s.species)

        n_force_selected = (
            len(self.selected_force_indices)
            if self.selected_force_indices is not None
            else len(self.force_structures)
        )

        return {
            'n_total': len(self.structures),
            'n_force': len(self.force_structures),
            'n_energy_only': len(self.energy_only_structures),
            'n_force_selected': n_force_selected,
            'total_atoms': total_atoms,
            'avg_atoms': (total_atoms / len(self.structures)
                          if self.structures else 0),
            'species': sorted(species_set),
        }


class CachedStructureDataset(Dataset):
    """
    Dataset that caches per-structure features for energy-only training.

    This avoids per-epoch featurization cost by computing features once
    at initialization. Force-related fields are disabled (use_forces=False).
    """
    def __init__(
        self,
        structures: List[Structure],
        descriptor,  # ChebyshevDescriptor
        max_energy: Optional[float] = None,
        max_forces: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.descriptor = descriptor
        self.max_energy = max_energy
        self.max_forces = max_forces
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Filter structures using the same logic as StructureDataset
        self.structures = self._filter_structures(structures)

        # Build cached samples
        self._cached: List[dict] = []
        with torch.no_grad():
            for i, struct in enumerate(self.structures):
                positions = torch.as_tensor(
                    struct.positions, dtype=descriptor.dtype)
                cell = (
                    torch.as_tensor(struct.cell, dtype=descriptor.dtype)
                    if struct.cell is not None else None
                )
                pbc = (
                    torch.as_tensor(struct.pbc)
                    if struct.pbc is not None else None
                )
                # Energy-only: features are sufficient;
                # use forward_from_positions
                features = descriptor.forward_from_positions(
                    positions, struct.species, cell, pbc)
                species_indices = torch.tensor(
                    [descriptor.species_to_idx[s] for s in struct.species],
                    dtype=torch.long
                )
                sample = {
                    'features': features,
                    'species_indices': species_indices,
                    'n_atoms': struct.n_atoms,
                    'energy': float(struct.energy),
                    'species': struct.species,
                    # Force-related fields disabled for cached energy-only path
                    'positions': None,
                    'forces': None,
                    'neighbor_info': None,
                    'use_forces': False,
                    'cell': cell,
                    'pbc': pbc,
                    'name': (struct.name if struct.name is not None
                             else f"struct_{i}"),
                }
                self._cached.append(sample)

    def _filter_structures(
        self,
        structures: List[Structure]
    ) -> List[Structure]:
        filtered: List[Structure] = []
        for struct in structures:
            # Energy threshold
            if self.max_energy is not None:
                energy_per_atom = struct.energy / struct.n_atoms
                if energy_per_atom > self.max_energy:
                    continue
            # Force threshold
            if self.max_forces is not None and struct.has_forces():
                max_force = np.abs(struct.forces).max()
                if max_force > self.max_forces:
                    continue
            filtered.append(struct)
        return filtered

    def __len__(self) -> int:
        return len(self._cached)

    def __getitem__(self, idx: int) -> dict:
        return self._cached[idx]


def train_test_split(dataset: StructureDataset,
                     test_fraction: float = 0.1,
                     seed: Optional[int] = None,
                     ) -> Tuple[StructureDataset, StructureDataset]:
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

    # Create new datasets with same parameters
    train_dataset = StructureDataset(
        structures=train_structures,
        descriptor=dataset.descriptor,
        force_fraction=dataset.force_fraction,
        force_sampling=dataset.force_sampling,
        max_energy=dataset.max_energy,
        max_forces=dataset.max_forces,
        seed=seed,
    )

    test_dataset = StructureDataset(
        structures=test_structures,
        descriptor=dataset.descriptor,
        force_fraction=dataset.force_fraction,
        force_sampling=dataset.force_sampling,
        max_energy=dataset.max_energy,
        max_forces=dataset.max_forces,
        seed=seed,
    )

    return train_dataset, test_dataset
