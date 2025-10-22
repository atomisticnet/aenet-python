"""
Complete featurization pipeline using Chebyshev descriptors.

This module implements the AUC (Artrith-Urban-Ceder) descriptor with the
typespin architecture from aenet's Fortran code.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_add

from .chebyshev import AngularBasis, RadialBasis
from .neighborlist import TorchNeighborList


class ChebyshevDescriptor(nn.Module):
    """
    Complete featurization pipeline using Chebyshev descriptors.

    Implements the typespin architecture from aenet's Fortran code:
    - Single radial and angular basis functions
    - Two feature sets: unweighted + typespin-weighted
    - Typespin coefficients centered around zero

    This exactly matches the behavior of aenet's Fortran implementation.
    """

    def __init__(
        self,
        species: List[str],
        rad_order: int,
        rad_cutoff: float,
        ang_order: int,
        ang_cutoff: float,
        min_cutoff: float = 0.55,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
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
        self.multi = self.n_species > 1

        # Create species to index mapping
        self.species_to_idx = {s: i for i, s in enumerate(species)}

        # Compute typespin coefficients (centered around zero)
        self.typespin = self._compute_typespin()

        # Neighbor list
        max_cutoff = max(rad_cutoff, ang_cutoff)
        self.nbl = TorchNeighborList(
            cutoff=max_cutoff, device=device, dtype=dtype
        )

        # SINGLE radial basis function (not per type pair!)
        self.rad_basis = RadialBasis(
            rad_order=rad_order,
            rad_cutoff=rad_cutoff,
            min_cutoff=min_cutoff,
            dtype=dtype,
        )

        # SINGLE angular basis function (not per type triplet!)
        self.ang_basis = AngularBasis(
            ang_order=ang_order,
            ang_cutoff=ang_cutoff,
            min_cutoff=min_cutoff,
            dtype=dtype,
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

        Returns
        -------
            typespin: (n_species,) tensor of typespin values
        """
        typespin = torch.zeros(
            self.n_species, dtype=self.dtype, device=self.device
        )

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

        Returns
        -------
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

    def compute_radial_features(
        self,
        positions: torch.Tensor,
        species_indices: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute radial features using typespin architecture.

        Uses scatter_add for efficient, gradient-preserving accumulation.

        For each atom i:
            - Set 1 (unweighted): sum over neighbors j of G_rad(d_ij)
            - Set 2 (typespin): sum over neighbors j of s_j * G_rad(d_ij)

        Args:
            positions: (N, 3) atomic positions
            species_indices: (N,) species index for each atom
            cell: (3, 3) lattice vectors (None for isolated systems)
            pbc: (3,) periodic boundary conditions

        Returns
        -------
            radial_features: (N, n_rad_features) tensor
        """
        n_atoms = len(positions)
        n_rad = self.rad_order + 1

        # Get all neighbors within radial cutoff
        # Positions are Cartesian, so use fractional=False for PBC
        neighbor_data = self.nbl.get_neighbors(
            positions, cell, pbc, fractional=False
        )
        edge_index = neighbor_data["edge_index"]
        distances = neighbor_data["distances"]

        # Filter by radial cutoff
        mask = (distances <= self.rad_cutoff) & (distances > self.min_cutoff)
        edge_index_rad = edge_index[:, mask]
        distances_rad = distances[mask]

        if len(distances_rad) == 0:
            # Return zeros if no neighbors
            if self.multi:
                return torch.zeros(
                    n_atoms, 2 * n_rad, dtype=self.dtype, device=self.device
                )
            else:
                return torch.zeros(
                    n_atoms, n_rad, dtype=self.dtype, device=self.device
                )

        # Compute radial basis for all pairs
        G_rad = self.rad_basis(distances_rad)  # (n_pairs, n_rad)

        # Get center and neighbor indices
        center_indices = edge_index_rad[0]
        neighbor_indices = edge_index_rad[1]

        # Unweighted features: scatter_add over neighbors
        rad_features_unweighted = scatter_add(
            G_rad, center_indices, dim=0, dim_size=n_atoms
        )

        if not self.multi:
            return rad_features_unweighted

        # Typespin-weighted features (for multi-species only)
        neighbor_species = species_indices[neighbor_indices]
        neighbor_typespin = self.typespin[neighbor_species]  # (n_pairs,)

        # Multiply by typespin
        G_rad_weighted = G_rad * neighbor_typespin.unsqueeze(-1)

        # Scatter_add weighted features
        rad_features_weighted = scatter_add(
            G_rad_weighted, center_indices, dim=0, dim_size=n_atoms
        )

        # Concatenate unweighted and weighted features
        rad_features = torch.cat(
            [rad_features_unweighted, rad_features_weighted], dim=1
        )

        return rad_features

    def compute_angular_features(
        self,
        positions: torch.Tensor,
        species_indices: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute angular features using typespin architecture.

        Uses vectorized triplet generation and scatter_add for
        efficient, gradient-preserving accumulation.

        For each atom i with neighbors j, k:
            - Set 1 (unweighted): sum over triplets of G_ang
            - Set 2 (typespin): sum over triplets of s_j * s_k * G_ang

        Args:
            positions: (N, 3) atomic positions
            species_indices: (N,) species index for each atom
            cell: (3, 3) lattice vectors
            pbc: (3,) periodic boundary conditions

        Returns
        -------
            angular_features: (N, n_ang_features) tensor
        """
        n_atoms = len(positions)
        n_ang = self.ang_order + 1

        # Get all neighbors within angular cutoff
        # Positions are Cartesian, so use fractional=False for PBC
        neighbor_data = self.nbl.get_neighbors(
            positions, cell, pbc, fractional=False
        )
        edge_index = neighbor_data["edge_index"]
        distances = neighbor_data["distances"]

        # Filter by angular cutoff
        mask = (distances <= self.ang_cutoff) & (distances > self.min_cutoff)
        edge_index_ang = edge_index[:, mask]
        distances_ang = distances[mask]

        if len(distances_ang) < 2:
            # Return zeros if insufficient neighbors
            if self.multi:
                return torch.zeros(
                    n_atoms, 2 * n_ang, dtype=self.dtype, device=self.device
                )
            else:
                return torch.zeros(
                    n_atoms, n_ang, dtype=self.dtype, device=self.device
                )

        # Compute displacement vectors
        i_indices = edge_index_ang[0]
        j_indices = edge_index_ang[1]

        if cell is not None:
            offsets = neighbor_data["offsets"][mask]
            r_ij = (
                positions[j_indices] + offsets.to(self.dtype) @ cell
            ) - positions[i_indices]
        else:
            r_ij = positions[j_indices] - positions[i_indices]

        # Normalize displacement vectors
        r_ij_norm = r_ij / distances_ang.unsqueeze(-1)

        # Generate all valid triplets (i, j, k) where j and k are
        # neighbors of i
        triplet_centers = []
        triplet_j_idx = []
        triplet_k_idx = []

        # Group edges by center atom
        for center_idx in range(n_atoms):
            neighbor_mask = i_indices == center_idx
            n_neighbors = neighbor_mask.sum().item()

            if n_neighbors < 2:
                continue

            # Get indices of edges for this center
            edge_positions = torch.where(neighbor_mask)[0]

            # Generate all pairs (j, k) for this center
            for idx_j in range(n_neighbors):
                for idx_k in range(idx_j + 1, n_neighbors):
                    triplet_centers.append(center_idx)
                    triplet_j_idx.append(edge_positions[idx_j].item())
                    triplet_k_idx.append(edge_positions[idx_k].item())

        if len(triplet_centers) == 0:
            # No valid triplets
            if self.multi:
                return torch.zeros(
                    n_atoms, 2 * n_ang, dtype=self.dtype, device=self.device
                )
            else:
                return torch.zeros(
                    n_atoms, n_ang, dtype=self.dtype, device=self.device
                )

        # Convert to tensors
        triplet_centers = torch.tensor(
            triplet_centers, dtype=torch.long, device=self.device
        )
        triplet_j_idx = torch.tensor(
            triplet_j_idx, dtype=torch.long, device=self.device
        )
        triplet_k_idx = torch.tensor(
            triplet_k_idx, dtype=torch.long, device=self.device
        )

        # Get distances and vectors for triplets
        d_ij = distances_ang[triplet_j_idx]
        d_ik = distances_ang[triplet_k_idx]
        vec_j = r_ij_norm[triplet_j_idx]
        vec_k = r_ij_norm[triplet_k_idx]

        # Compute cos(theta_ijk) for all triplets
        cos_theta = (vec_j * vec_k).sum(dim=1).clamp(-1.0, 1.0)

        # Compute angular basis for all triplets
        G_ang = self.ang_basis(d_ij, d_ik, cos_theta)  # (n_triplets, n_ang)

        # Scatter_add unweighted features
        ang_features_unweighted = scatter_add(
            G_ang, triplet_centers, dim=0, dim_size=n_atoms
        )

        if not self.multi:
            return ang_features_unweighted

        # Typespin-weighted features (for multi-species only)
        neighbor_j_species = species_indices[j_indices[triplet_j_idx]]
        neighbor_k_species = species_indices[j_indices[triplet_k_idx]]

        typespin_j = self.typespin[neighbor_j_species]
        typespin_k = self.typespin[neighbor_k_species]
        typespin_product = typespin_j * typespin_k

        # Multiply by typespin product
        G_ang_weighted = G_ang * typespin_product.unsqueeze(-1)

        # Scatter_add weighted features
        ang_features_weighted = scatter_add(
            G_ang_weighted, triplet_centers, dim=0, dim_size=n_atoms
        )

        # Concatenate unweighted and weighted features
        ang_features = torch.cat(
            [ang_features_unweighted, ang_features_weighted], dim=1
        )

        return ang_features

    def forward(
        self,
        positions: torch.Tensor,
        species: List[str],
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Featurize atomic structure.

        Args:
            positions: (N, 3) atomic positions
            species: List of N species names
            cell: (3, 3) lattice vectors (None for isolated)
            pbc: (3,) periodic boundary conditions

        Returns
        -------
            features: (N, n_features) feature matrix
        """
        # Convert species to indices
        species_indices = torch.tensor(
            [self.species_to_idx[s] for s in species],
            dtype=torch.long,
            device=self.device,
        )

        # Move inputs to device
        positions = positions.to(self.device).to(self.dtype)
        if cell is not None:
            cell = cell.to(self.device).to(self.dtype)

        # Compute radial features
        rad_features = self.compute_radial_features(
            positions, species_indices, cell, pbc
        )

        # Compute angular features
        ang_features = self.compute_angular_features(
            positions, species_indices, cell, pbc
        )

        # Concatenate in Fortran order: [radial_unweighted, angular_unweighted,
        #                                 radial_weighted, angular_weighted]
        # (or [radial, angular] for single-species)
        if self.multi:
            n_rad = self.rad_order + 1
            n_ang = self.ang_order + 1
            features = torch.cat(
                [
                    rad_features[:, :n_rad],  # radial unweighted
                    ang_features[:, :n_ang],  # angular unweighted
                    rad_features[:, n_rad:],  # radial weighted
                    ang_features[:, n_ang:],  # angular weighted
                ],
                dim=1,
            )
        else:
            # Single species: just concatenate radial and angular
            features = torch.cat([rad_features, ang_features], dim=1)

        return features

    def compute_feature_gradients(
        self,
        positions: torch.Tensor,
        species: List[str],
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute features and their gradients w.r.t. positions.

        Uses PyTorch autograd to compute ∂features/∂positions.

        Args:
            positions: (N, 3) atomic positions
            species: List of N species names
            cell: (3, 3) lattice vectors
            pbc: (3,) periodic boundary conditions

        Returns
        -------
            features: (N, F) feature tensor
            gradients: (N, F, N, 3) gradient tensor where
                      gradients[i, f, j, k] = ∂feature[i,f]/∂position[j,k]
        """
        # Enable gradient tracking
        positions_grad = positions.clone().detach().requires_grad_(True)

        # Compute features
        features = self.forward(positions_grad, species, cell, pbc)

        N, F = features.shape
        gradients = torch.zeros(
            N, F, N, 3, dtype=self.dtype, device=self.device
        )

        # Compute gradient for each feature of each atom
        for i in range(N):
            for f in range(F):
                # Create grad_outputs tensor
                grad_out = torch.zeros_like(features)
                grad_out[i, f] = 1.0

                # Compute gradient
                grad = torch.autograd.grad(
                    features,
                    positions_grad,
                    grad_outputs=grad_out,
                    retain_graph=True,
                    create_graph=False,
                )[0]

                gradients[i, f] = grad

        return features.detach(), gradients

    def compute_forces_from_energy(
        self,
        positions: torch.Tensor,
        species: List[str],
        energy_model: nn.Module,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute atomic forces from an energy model.

        Forces are computed as F = -∂E/∂r using autograd through
        the full featurization → energy prediction pipeline.

        Args:
            positions: (N, 3) atomic positions
            species: List of N species names
            energy_model: Neural network that predicts energy from features
            cell: (3, 3) lattice vectors
            pbc: (3,) periodic boundary conditions

        Returns
        -------
            energy: Scalar total energy
            forces: (N, 3) force on each atom
        """
        # Enable gradient tracking for positions
        positions_grad = positions.clone().detach().requires_grad_(True)

        # Compute features
        features = self.forward(positions_grad, species, cell, pbc)

        # Predict energy
        energy = energy_model(features).sum()

        # Compute forces via autograd: F = -∂E/∂r
        forces = -torch.autograd.grad(
            energy,
            positions_grad,
            create_graph=True,  # Enable higher-order derivatives
        )[0]

        return energy, forces

    def featurize_structure(
        self,
        positions: np.ndarray,
        species: List[str],
        cell: Optional[np.ndarray] = None,
        pbc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Numpy interface for featurization.

        Args:
            positions: (N, 3) numpy array
            species: List of species names
            cell: (3, 3) numpy array (optional)
            pbc: (3,) boolean numpy array (optional)

        Returns
        -------
            features: (N, n_features) numpy array
        """
        # Convert to torch
        pos_torch = torch.from_numpy(positions).to(self.dtype)
        cell_torch = (
            torch.from_numpy(cell).to(self.dtype) if cell is not None else None
        )
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

    def forward(
        self,
        batch_positions: List[torch.Tensor],
        batch_species: List[List[str]],
        batch_cells: Optional[List[torch.Tensor]] = None,
        batch_pbc: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Featurize batch of structures.

        Args:
            batch_positions: List of (N_i, 3) position tensors
            batch_species: List of species lists
            batch_cells: List of (3, 3) cell tensors
            batch_pbc: List of (3,) pbc tensors

        Returns
        -------
            features: (total_atoms, n_features) concatenated features
            batch_indices: (total_atoms,) batch index for each atom
        """
        all_features = []
        all_batch_indices = []

        for batch_idx, (pos, species) in enumerate(
            zip(batch_positions, batch_species)
        ):
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
