"""
Complete featurization pipeline using Chebyshev descriptors.

This module implements the AUC (Artrith-Urban-Ceder) descriptor with the
typespin architecture from aenet's Fortran code.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np

from .neighborlist import TorchNeighborList
from .chebyshev import RadialBasis, AngularBasis


class ChebyshevDescriptor(nn.Module):
    """
    Complete featurization pipeline using Chebyshev descriptors.

    Implements the typespin architecture from aenet's Fortran code:
    - Single radial and angular basis functions
    - Two feature sets: unweighted + typespin-weighted
    - Typespin coefficients centered around zero

    This exactly matches the behavior of aenet's Fortran implementation.
    """

    def __init__(self,
                 species: List[str],
                 rad_order: int,
                 rad_cutoff: float,
                 ang_order: int,
                 ang_cutoff: float,
                 min_cutoff: float = 0.55,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float64):
        """
        Initialize Chebyshev descriptor.

        Args:
            species: List of atomic species (e.g., ['O', 'H'])
            rad_order: Maximum radial Chebyshev order
            rad_cutoff: Radial cutoff radius (Angstroms)
            ang_order: Maximum angular Chebyshev order
            ang_cutoff: Angular cutoff radius (Angstroms)
            min_cutoff: Minimum distance cutoff (Angstroms)
            device: 'cpu' or 'cuda'
            dtype: torch.float64 for double precision
        """
        super().__init__()

        self.species = species
        self.n_species = len(species)
        self.rad_order = rad_order
        self.rad_cutoff = rad_cutoff
        self.ang_order = ang_order
        self.ang_cutoff = ang_cutoff
        self.min_cutoff = min_cutoff
        self.device = device
        self.dtype = dtype

        # Multi-species flag
        self.multi = (self.n_species > 1)

        # Create species to index mapping
        self.species_to_idx = {s: i for i, s in enumerate(species)}

        # Compute typespin coefficients (centered around zero)
        self.typespin = self._compute_typespin()

        # Neighbor list
        max_cutoff = max(rad_cutoff, ang_cutoff)
        self.nbl = TorchNeighborList(cutoff=max_cutoff, device=device, dtype=dtype)

        # SINGLE radial basis function (not per type pair!)
        self.rad_basis = RadialBasis(
            rad_order=rad_order,
            rad_cutoff=rad_cutoff,
            min_cutoff=min_cutoff,
            dtype=dtype
        )

        # SINGLE angular basis function (not per type triplet!)
        self.ang_basis = AngularBasis(
            ang_order=ang_order,
            ang_cutoff=ang_cutoff,
            min_cutoff=min_cutoff,
            dtype=dtype
        )

        # Calculate number of features
        self.n_features = self._calculate_n_features()

    def _compute_typespin(self) -> torch.Tensor:
        """
        Compute typespin coefficients matching Fortran implementation.

        For even number of species, zero is skipped:
        - 2 species: {-1, 1}
        - 4 species: {-2, -1, 1, 2}

        For odd number of species, zero is included:
        - 3 species: {-1, 0, 1}
        - 5 species: {-2, -1, 0, 1, 2}

        Returns:
            typespin: (n_species,) tensor of typespin values
        """
        typespin = torch.zeros(self.n_species, dtype=self.dtype, device=self.device)

        # Use int() to truncate towards zero like Fortran, not floor division
        s = int(-self.n_species / 2)
        for i in range(self.n_species):
            # Skip zero for even number of species
            if s == 0 and self.n_species % 2 == 0:
                s += 1
            typespin[i] = float(s)
            s += 1

        return typespin

    def _calculate_n_features(self) -> int:
        """
        Calculate number of features for each atom.

        For multi-species systems:
            - Radial: 2 * (rad_order + 1)
            - Angular: 2 * (ang_order + 1)

        For single-species systems:
            - Radial: (rad_order + 1)
            - Angular: (ang_order + 1)

        Returns:
            Number of features (same for all species)
        """
        n_rad = self.rad_order + 1
        n_ang = self.ang_order + 1

        if self.multi:
            # Two sets: unweighted + typespin-weighted
            n_features = 2 * (n_rad + n_ang)
        else:
            # Single set only
            n_features = n_rad + n_ang

        return n_features

    def get_n_features(self) -> int:
        """Get number of features (same for all species)."""
        return self.n_features

    def compute_radial_features(self,
                               positions: torch.Tensor,
                               species_indices: torch.Tensor,
                               cell: Optional[torch.Tensor] = None,
                               pbc: Optional[torch.Tensor] = None
                               ) -> torch.Tensor:
        """
        Compute radial features for all atoms using typespin architecture.

        For each atom i:
            - Set 1 (unweighted): sum over all neighbors j of G_rad(d_ij)
            - Set 2 (typespin): sum over all neighbors j of s_j * G_rad(d_ij)

        Args:
            positions: (N, 3) atomic positions
            species_indices: (N,) species index for each atom
            cell: (3, 3) lattice vectors (None for isolated systems)
            pbc: (3,) periodic boundary conditions

        Returns:
            radial_features: (N, n_rad_features) tensor
        """
        n_atoms = len(positions)
        n_rad = self.rad_order + 1

        # Feature sets: [unweighted, typespin-weighted]
        if self.multi:
            rad_features = torch.zeros(n_atoms, 2 * n_rad,
                                       dtype=self.dtype, device=self.device)
        else:
            rad_features = torch.zeros(n_atoms, n_rad,
                                       dtype=self.dtype, device=self.device)

        # Get all neighbors within radial cutoff
        neighbor_data = self.nbl.get_neighbors(positions, cell, pbc)
        edge_index = neighbor_data['edge_index']
        distances = neighbor_data['distances']

        # Filter by radial cutoff
        mask = (distances <= self.rad_cutoff) & (distances > self.min_cutoff)
        edge_index_rad = edge_index[:, mask]
        distances_rad = distances[mask]

        if len(distances_rad) == 0:
            return rad_features

        # Compute radial basis for all pairs
        G_rad = self.rad_basis(distances_rad)  # (n_pairs, n_rad)

        # Get center and neighbor indices
        center_indices = edge_index_rad[0]
        neighbor_indices = edge_index_rad[1]

        # Unweighted features: sum over neighbors
        for pair_idx in range(len(distances_rad)):
            center_idx = center_indices[pair_idx].item()
            rad_features[center_idx, :n_rad] += G_rad[pair_idx]

        # Typespin-weighted features (for multi-species only)
        if self.multi:
            neighbor_species = species_indices[neighbor_indices]
            neighbor_typespin = self.typespin[neighbor_species]  # (n_pairs,)

            # Multiply by typespin and sum
            G_rad_weighted = G_rad * neighbor_typespin.unsqueeze(-1)  # (n_pairs, n_rad)

            for pair_idx in range(len(distances_rad)):
                center_idx = center_indices[pair_idx].item()
                rad_features[center_idx, n_rad:2*n_rad] += G_rad_weighted[pair_idx]

        return rad_features

    def compute_angular_features(self,
                                positions: torch.Tensor,
                                species_indices: torch.Tensor,
                                cell: Optional[torch.Tensor] = None,
                                pbc: Optional[torch.Tensor] = None
                                ) -> torch.Tensor:
        """
        Compute angular features for all atoms using typespin architecture.

        For each atom i with neighbors j, k:
            - Set 1 (unweighted): sum over all triplets (i,j,k) of G_ang(d_ij, d_ik, cos_ijk)
            - Set 2 (typespin): sum over triplets of s_j * s_k * G_ang(d_ij, d_ik, cos_ijk)

        Args:
            positions: (N, 3) atomic positions
            species_indices: (N,) species index for each atom
            cell: (3, 3) lattice vectors
            pbc: (3,) periodic boundary conditions

        Returns:
            angular_features: (N, n_ang_features) tensor
        """
        n_atoms = len(positions)
        n_ang = self.ang_order + 1

        # Feature sets: [unweighted, typespin-weighted]
        if self.multi:
            ang_features = torch.zeros(n_atoms, 2 * n_ang,
                                       dtype=self.dtype, device=self.device)
        else:
            ang_features = torch.zeros(n_atoms, n_ang,
                                       dtype=self.dtype, device=self.device)

        # Get all neighbors within angular cutoff
        neighbor_data = self.nbl.get_neighbors(positions, cell, pbc)
        edge_index = neighbor_data['edge_index']
        distances = neighbor_data['distances']

        # Filter by angular cutoff
        mask = (distances <= self.ang_cutoff) & (distances > self.min_cutoff)
        edge_index_ang = edge_index[:, mask]
        distances_ang = distances[mask]

        if len(distances_ang) < 2:
            return ang_features

        # Compute displacement vectors
        i_indices = edge_index_ang[0]
        j_indices = edge_index_ang[1]

        if cell is not None:
            offsets = neighbor_data['offsets'][mask]
            # Convert offsets to correct dtype for matrix multiplication
            r_ij = (positions[j_indices] + offsets.to(self.dtype) @ cell) - positions[i_indices]
        else:
            r_ij = positions[j_indices] - positions[i_indices]

        # Normalize displacement vectors
        r_ij_norm = r_ij / distances_ang.unsqueeze(-1)

        # Process each central atom
        for center_idx in range(n_atoms):
            # Find neighbors of this atom
            neighbor_mask = i_indices == center_idx
            if neighbor_mask.sum() < 2:
                continue

            neighbor_pos = torch.where(neighbor_mask)[0]
            n_neighbors = len(neighbor_pos)

            # Get data for these neighbors
            neighbor_indices = j_indices[neighbor_pos]
            neighbor_distances = distances_ang[neighbor_pos]
            neighbor_vectors = r_ij_norm[neighbor_pos]
            neighbor_species = species_indices[neighbor_indices]

            # Generate all pairs (j, k) where j < k
            for j_idx in range(n_neighbors):
                for k_idx in range(j_idx + 1, n_neighbors):
                    d_ij = neighbor_distances[j_idx].unsqueeze(0)
                    d_ik = neighbor_distances[k_idx].unsqueeze(0)

                    # Compute cos(theta_ijk)
                    cos_theta = torch.dot(neighbor_vectors[j_idx],
                                         neighbor_vectors[k_idx]).clamp(-1.0, 1.0)
                    cos_theta = cos_theta.unsqueeze(0)

                    # Compute angular basis
                    G_ang = self.ang_basis(d_ij, d_ik, cos_theta).squeeze(0)  # (n_ang,)

                    # Unweighted features
                    ang_features[center_idx, :n_ang] += G_ang

                    # Typespin-weighted features (for multi-species only)
                    if self.multi:
                        s_j = self.typespin[neighbor_species[j_idx]]
                        s_k = self.typespin[neighbor_species[k_idx]]
                        ang_features[center_idx, n_ang:2*n_ang] += s_j * s_k * G_ang

        return ang_features

    def forward(self,
               positions: torch.Tensor,
               species: List[str],
               cell: Optional[torch.Tensor] = None,
               pbc: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """
        Featurize atomic structure.

        Args:
            positions: (N, 3) atomic positions
            species: List of N species names
            cell: (3, 3) lattice vectors (None for isolated)
            pbc: (3,) periodic boundary conditions

        Returns:
            features: (N, n_features) feature matrix
        """
        # Convert species to indices
        species_indices = torch.tensor(
            [self.species_to_idx[s] for s in species],
            dtype=torch.long, device=self.device
        )

        # Move inputs to device
        positions = positions.to(self.device).to(self.dtype)
        if cell is not None:
            cell = cell.to(self.device).to(self.dtype)

        # Compute radial features
        rad_features = self.compute_radial_features(
            positions, species_indices, cell, pbc)

        # Compute angular features
        ang_features = self.compute_angular_features(
            positions, species_indices, cell, pbc)

        # Concatenate in Fortran order: [radial_unweighted, angular_unweighted,
        #                                 radial_weighted, angular_weighted]
        # (or [radial, angular] for single-species)
        if self.multi:
            n_rad = self.rad_order + 1
            n_ang = self.ang_order + 1
            features = torch.cat([
                rad_features[:, :n_rad],      # radial unweighted
                ang_features[:, :n_ang],      # angular unweighted
                rad_features[:, n_rad:],      # radial weighted
                ang_features[:, n_ang:]       # angular weighted
            ], dim=1)
        else:
            # Single species: just concatenate radial and angular
            features = torch.cat([rad_features, ang_features], dim=1)

        return features

    def featurize_structure(self,
                           positions: np.ndarray,
                           species: List[str],
                           cell: Optional[np.ndarray] = None,
                           pbc: Optional[np.ndarray] = None
                           ) -> np.ndarray:
        """
        Numpy interface for featurization.

        Args:
            positions: (N, 3) numpy array
            species: List of species names
            cell: (3, 3) numpy array (optional)
            pbc: (3,) boolean numpy array (optional)

        Returns:
            features: (N, n_features) numpy array
        """
        # Convert to torch
        pos_torch = torch.from_numpy(positions).to(self.dtype)
        cell_torch = torch.from_numpy(cell).to(self.dtype) if cell is not None else None
        pbc_torch = torch.from_numpy(pbc) if pbc is not None else None

        # Featurize
        with torch.no_grad():
            features = self.forward(pos_torch, species, cell_torch, pbc_torch)

        # Convert back to numpy
        return features.cpu().numpy()


class BatchedFeaturizer(nn.Module):
    """
    Batched featurization for multiple structures.

    More efficient for training on datasets.
    """

    def __init__(self, featurizer: ChebyshevDescriptor):
        """
        Initialize batched featurizer.

        Args:
            featurizer: ChebyshevDescriptor instance to use
        """
        super().__init__()
        self.featurizer = featurizer

    def forward(self,
               batch_positions: List[torch.Tensor],
               batch_species: List[List[str]],
               batch_cells: Optional[List[torch.Tensor]] = None,
               batch_pbc: Optional[List[torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Featurize batch of structures.

        Args:
            batch_positions: List of (N_i, 3) position tensors
            batch_species: List of species lists
            batch_cells: List of (3, 3) cell tensors
            batch_pbc: List of (3,) pbc tensors

        Returns:
            features: (total_atoms, n_features) concatenated features
            batch_indices: (total_atoms,) batch index for each atom
        """
        all_features = []
        all_batch_indices = []

        for batch_idx, (pos, species) in enumerate(zip(batch_positions, batch_species)):
            cell = batch_cells[batch_idx] if batch_cells else None
            pbc = batch_pbc[batch_idx] if batch_pbc else None

            features = self.featurizer(pos, species, cell, pbc)
            all_features.append(features)
            all_batch_indices.append(
                torch.full((len(pos),), batch_idx, dtype=torch.long)
            )

        # Concatenate
        features = torch.cat(all_features, dim=0)
        batch_indices = torch.cat(all_batch_indices, dim=0)

        return features, batch_indices
