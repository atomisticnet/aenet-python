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
from .graph import center_ids_of_edge as _center_ids_of_edge


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
        neighbor_indices: List[torch.Tensor],
        neighbor_vectors: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute radial features from pre-computed neighbor information.

        This is the core implementation that accepts neighbor vectors directly
        from the local structural environment (LSE), enabling clean gradient
        computation without re-computing neighbors or applying PBC.

        For each atom i:
            - Set 1 (unweighted): sum over neighbors j of G_rad(d_ij)
            - Set 2 (typespin): sum over neighbors j of s_j * G_rad(d_ij)

        Args:
            positions: (N, 3) atomic positions (for autograd tracking)
            species_indices: (N,) species index for each atom
            neighbor_indices: List of (nnb_i,) tensors with neighbor
              atom indices
            neighbor_vectors: List of (nnb_i, 3) tensors with
              displacement vectors

        Returns
        -------
            radial_features: (N, n_rad_features) tensor
        """
        n_atoms = len(positions)
        n_rad = self.rad_order + 1

        # Build edge lists from neighbor information
        center_indices_list = []
        neighbor_indices_list = []
        distances_list = []

        for i, (nb_idx, nb_vec) in enumerate(zip(neighbor_indices,
                                                 neighbor_vectors)):
            if len(nb_idx) == 0:
                continue

            # Compute distances from vectors
            distances = torch.norm(nb_vec, dim=-1)

            # Filter by radial cutoff
            mask = (distances <= self.rad_cutoff
                    ) & (distances > self.min_cutoff)
            if mask.any():
                n_valid = mask.sum().item()
                center_indices_list.append(torch.full(
                    (n_valid,), i, dtype=torch.long, device=self.device))
                neighbor_indices_list.append(nb_idx[mask])
                distances_list.append(distances[mask])

        if len(distances_list) == 0:
            # Return zeros if no neighbors
            if self.multi:
                return torch.zeros(
                    n_atoms, 2 * n_rad, dtype=self.dtype, device=self.device
                )
            else:
                return torch.zeros(
                    n_atoms, n_rad, dtype=self.dtype, device=self.device
                )

        # Concatenate all edges
        center_indices = torch.cat(center_indices_list)
        neighbor_indices_flat = torch.cat(neighbor_indices_list)
        distances_rad = torch.cat(distances_list)

        # Compute radial basis for all pairs
        G_rad = self.rad_basis(distances_rad)  # (n_pairs, n_rad)

        # Unweighted features: scatter_add over neighbors
        rad_features_unweighted = scatter_add(
            G_rad, center_indices, dim=0, dim_size=n_atoms
        )

        if not self.multi:
            return rad_features_unweighted

        # Typespin-weighted features (for multi-species only)
        neighbor_species = species_indices[neighbor_indices_flat]
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
        neighbor_indices: List[torch.Tensor],
        neighbor_vectors: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute angular features from pre-computed neighbor information.

        This is the core implementation that accepts neighbor vectors directly
        from the local structural environment (LSE), enabling clean gradient
        computation without re-computing neighbors or applying PBC.

        For each atom i with neighbors j, k:
            - Set 1 (unweighted): sum over triplets of G_ang
            - Set 2 (typespin): sum over triplets of s_j * s_k * G_ang

        Args:
            positions: (N, 3) atomic positions (for autograd tracking)
            species_indices: (N,) species index for each atom
            neighbor_indices: List of (nnb_i,) tensors with neighbor
              atom indices
            neighbor_vectors: List of (nnb_i, 3) tensors with
              displacement vectors

        Returns
        -------
            angular_features: (N, n_ang_features) tensor
        """
        n_atoms = len(positions)
        n_ang = self.ang_order + 1

        # Generate all valid triplets (i, j, k) from neighbor information
        triplet_centers = []
        triplet_j_global = []
        triplet_k_global = []
        triplet_j_local = []
        triplet_k_local = []

        for i, (nb_idx, nb_vec) in enumerate(zip(neighbor_indices,
                                                 neighbor_vectors)):
            if len(nb_idx) < 2:
                continue

            # Compute distances from vectors
            distances = torch.norm(nb_vec, dim=-1)

            # Filter by angular cutoff
            mask = (distances <= self.ang_cutoff
                    ) & (distances > self.min_cutoff)
            valid_nb_idx = nb_idx[mask]

            n_valid = len(valid_nb_idx)
            if n_valid < 2:
                continue

            # Generate all pairs (j, k) for this center
            for j_local in range(n_valid):
                for k_local in range(j_local + 1, n_valid):
                    triplet_centers.append(i)
                    triplet_j_global.append(valid_nb_idx[j_local])
                    triplet_k_global.append(valid_nb_idx[k_local])
                    triplet_j_local.append(j_local)
                    triplet_k_local.append(k_local)

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
        triplet_j_global = torch.tensor(
            triplet_j_global, dtype=torch.long, device=self.device
        )
        triplet_k_global = torch.tensor(
            triplet_k_global, dtype=torch.long, device=self.device
        )

        # Collect distances and normalized vectors for all triplets
        d_ij_list = []
        d_ik_list = []
        vec_j_norm_list = []
        vec_k_norm_list = []

        for idx, center_i in enumerate(triplet_centers):
            center_i = center_i.item()
            j_local = triplet_j_local[idx]
            k_local = triplet_k_local[idx]

            nb_vec = neighbor_vectors[center_i]
            distances = torch.norm(nb_vec, dim=-1)

            # Filter by cutoff
            mask = (distances <= self.ang_cutoff
                    ) & (distances > self.min_cutoff)
            valid_vec = nb_vec[mask]
            valid_dist = distances[mask]

            d_ij_list.append(valid_dist[j_local])
            d_ik_list.append(valid_dist[k_local])
            vec_j_norm_list.append(valid_vec[j_local] / valid_dist[j_local])
            vec_k_norm_list.append(valid_vec[k_local] / valid_dist[k_local])

        d_ij = torch.stack(d_ij_list)
        d_ik = torch.stack(d_ik_list)
        vec_j_norm = torch.stack(vec_j_norm_list)
        vec_k_norm = torch.stack(vec_k_norm_list)

        # Compute cos(theta_ijk) for all triplets
        cos_theta = (vec_j_norm * vec_k_norm).sum(dim=1).clamp(-1.0, 1.0)

        # Compute angular basis for all triplets
        G_ang = self.ang_basis(d_ij, d_ik, cos_theta)  # (n_triplets, n_ang)

        # Scatter_add unweighted features
        ang_features_unweighted = scatter_add(
            G_ang, triplet_centers, dim=0, dim_size=n_atoms
        )

        if not self.multi:
            return ang_features_unweighted

        # Typespin-weighted features (for multi-species only)
        neighbor_j_species = species_indices[triplet_j_global]
        neighbor_k_species = species_indices[triplet_k_global]

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
        neighbor_indices: Optional[List[torch.Tensor]] = None,
        neighbor_vectors: Optional[List[torch.Tensor]] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Featurize atomic structure.

        This method supports two calling modes:

        1. With pre-computed neighbors (for gradient computation):
           forward(positions, species, neighbor_indices, neighbor_vectors)

        2. Legacy mode (automatic neighbor computation):
           forward(positions, species, cell=cell, pbc=pbc)

        Args:
            positions: (N, 3) atomic positions
            species: List of N species names
            neighbor_indices: List of (nnb_i,) tensors with neighbor
              atom indices (optional)
            neighbor_vectors: List of (nnb_i, 3) tensors with
              displacement vectors (optional)
            cell: (3, 3) lattice vectors (for legacy mode)
            pbc: (3,) periodic boundary conditions (for legacy mode)

        Returns
        -------
            features: (N, n_features) feature matrix
        """
        # Legacy mode: compute neighbors automatically
        if neighbor_indices is None or neighbor_vectors is None:
            return self.forward_from_positions(positions, species, cell, pbc)

        # New mode: use pre-computed neighbors for gradient computation
        # Convert species to indices
        species_indices = torch.tensor(
            [self.species_to_idx[s] for s in species],
            dtype=torch.long,
            device=self.device,
        )

        # Move inputs to device
        positions = positions.to(self.device).to(self.dtype)

        # Compute radial features
        rad_features = self.compute_radial_features(
            positions, species_indices, neighbor_indices, neighbor_vectors
        )

        # Compute angular features
        ang_features = self.compute_angular_features(
            positions, species_indices, neighbor_indices, neighbor_vectors
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

    def forward_from_positions(
        self,
        positions: torch.Tensor,
        species: List[str],
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience method that computes neighbors then calls forward().

        Use this when you don't need gradients and want automatic neighbor
        computation. For gradient computation, use forward() directly with
        pre-computed neighbor information.

        Args:
            positions: (N, 3) atomic positions
            species: List of N species names
            cell: (3, 3) lattice vectors (None for isolated)
            pbc: (3,) periodic boundary conditions

        Returns
        -------
            features: (N, n_features) feature matrix
        """
        # Move inputs to device
        positions = positions.to(self.device).to(self.dtype)
        if cell is not None:
            cell = cell.to(self.device).to(self.dtype)

        # Get neighbor data using the maximum cutoff
        neighbor_data = self.nbl.get_neighbors(
            positions, cell, pbc, fractional=False
        )

        edge_index = neighbor_data['edge_index']
        distances = neighbor_data['distances']
        offsets = neighbor_data['offsets']

        # Filter by minimum cutoff
        mask = distances > self.min_cutoff
        edge_index = edge_index[:, mask]
        distances = distances[mask]
        if offsets is not None:
            offsets = offsets[mask]

        # Compute displacement vectors
        i_indices = edge_index[0]
        j_indices = edge_index[1]

        if cell is not None and offsets is not None:
            r_ij = (
                positions[j_indices] + offsets.to(self.dtype) @ cell
            ) - positions[i_indices]
        else:
            r_ij = positions[j_indices] - positions[i_indices]

        # Organize per atom
        n_atoms = len(positions)
        neighbor_indices = []
        neighbor_vectors = []

        for atom_idx in range(n_atoms):
            neighbor_mask = i_indices == atom_idx
            atom_neighbors = j_indices[neighbor_mask]
            atom_vectors = r_ij[neighbor_mask]

            neighbor_indices.append(atom_neighbors)
            neighbor_vectors.append(atom_vectors)

        # Call core forward method
        return self.forward(positions, species,
                            neighbor_indices, neighbor_vectors)

    def _compute_radial_gradients(
        self,
        positions: torch.Tensor,
        species_indices: torch.Tensor,
        neighbor_indices: List[torch.Tensor],
        neighbor_vectors: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute gradients of radial features using a fully vectorized
        semi-analytical method.

        This method uses the analytical derivatives of the basis functions
        and the chain rule to compute gradients with respect to atomic
        positions in a single, vectorized pass.

        Args:
            positions: (N, 3) atomic positions
            species_indices: (N,) species index for each atom
            neighbor_indices: List of neighbor indices per atom
            neighbor_vectors: List of displacement vectors per atom

        Returns
        -------
            gradients: (N, n_rad_features, N, 3) gradient tensor
        """
        n_atoms = len(positions)
        n_rad = self.rad_order + 1

        # Build edge lists from neighbor information
        center_indices_list = []
        neighbor_indices_list = []
        vectors_list = []

        for i, (nb_idx, nb_vec) in enumerate(zip(neighbor_indices,
                                                 neighbor_vectors)):
            if len(nb_idx) == 0:
                continue

            distances = torch.norm(nb_vec, dim=-1)
            mask = (distances <= self.rad_cutoff
                    ) & (distances > self.min_cutoff)
            if mask.any():
                n_valid = mask.sum().item()
                center_indices_list.append(torch.full(
                    (n_valid,), i, dtype=torch.long, device=self.device))
                neighbor_indices_list.append(nb_idx[mask])
                vectors_list.append(nb_vec[mask])

        if len(vectors_list) == 0:
            # Return zeros if no neighbors
            n_rad_features = 2 * n_rad if self.multi else n_rad
            return torch.zeros(
                n_atoms, n_rad_features, n_atoms, 3,
                dtype=self.dtype, device=self.device
            )

        # Concatenate all edges
        center_indices = torch.cat(center_indices_list)
        neighbor_indices_flat = torch.cat(neighbor_indices_list)
        r_ij = torch.cat(vectors_list)

        # Compute distances and basis derivatives
        distances = torch.norm(r_ij, dim=-1)
        _, dG_rad_dr = self.rad_basis.forward_with_derivatives(distances)

        # Chain rule: dG/dr_vec = (dG/dr) * (dr/dr_vec)
        # dr/dr_vec is the normalized displacement vector
        dG_rad_drij = dG_rad_dr.unsqueeze(-1) * (
            r_ij / (distances.unsqueeze(-1) + 1e-10)).unsqueeze(1)

        # Initialize gradient tensor
        n_rad_features = 2 * n_rad if self.multi else n_rad
        gradients = torch.zeros(
            n_atoms, n_rad_features, n_atoms, 3,
            dtype=self.dtype, device=self.device
        )

        # Unweighted gradients
        # Contribution to central atom i: -dG/drij
        # Contribution to neighbor atom j: +dG/drij
        # We can do this with two scatter_add operations

        # Reshape for scattering: (n_pairs, n_rad, 3)
        grad_unweighted = dG_rad_drij

        # Scatter to central atoms (negative contribution)
        # This is tricky because scatter_add doesn't support (i, j) indexing
        # We will loop for now, but this can be optimized
        for pair_idx in range(len(center_indices)):
            i = center_indices[pair_idx].item()
            j = neighbor_indices_flat[pair_idx].item()

            gradients[i, :n_rad, i] -= grad_unweighted[pair_idx]
            gradients[i, :n_rad, j] += grad_unweighted[pair_idx]

        if not self.multi:
            return gradients

        # Typespin-weighted gradients
        neighbor_species = species_indices[neighbor_indices_flat]
        neighbor_typespin = self.typespin[neighbor_species]

        grad_weighted = (grad_unweighted
                         * neighbor_typespin.unsqueeze(-1).unsqueeze(-1))

        for pair_idx in range(len(center_indices)):
            i = center_indices[pair_idx].item()
            j = neighbor_indices_flat[pair_idx].item()

            gradients[i, n_rad:, i] -= grad_weighted[pair_idx]
            gradients[i, n_rad:, j] += grad_weighted[pair_idx]

        return gradients

    def _compute_angular_gradients(
        self,
        positions: torch.Tensor,
        species_indices: torch.Tensor,
        neighbor_indices: List[torch.Tensor],
        neighbor_vectors: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute gradients of angular features using a fully analytical,
        vectorized method.

        Avoids autograd on geometric quantities to
        improve performance and numerical stability (especially under PBC).

        Returns
        -------
            gradients: (N, n_ang_features, N, 3) gradient tensor
        """
        n_atoms = len(positions)
        n_ang = self.ang_order + 1
        eps_norm = 1e-12
        eps_dist = 1e-20

        # Build triplet lists (i, j, k) and corresponding displacement vectors
        triplet_i: list[int] = []
        triplet_j: list[int] = []
        triplet_k: list[int] = []
        r_ij_list: list[torch.Tensor] = []
        r_ik_list: list[torch.Tensor] = []

        for i, (nb_idx, nb_vec) in enumerate(zip(neighbor_indices,
                                                 neighbor_vectors)):
            if len(nb_idx) < 2:
                continue

            distances = torch.norm(nb_vec, dim=-1)
            mask = (distances <= self.ang_cutoff
                    ) & (distances > self.min_cutoff)

            if not mask.any():
                continue

            valid_nb_idx = nb_idx[mask]
            valid_vectors = nb_vec[mask]
            n_valid = len(valid_nb_idx)
            if n_valid < 2:
                continue

            # Generate all unique neighbor pairs (j, k) with j_local < k_local
            for j_local in range(n_valid):
                for k_local in range(j_local + 1, n_valid):
                    triplet_i.append(i)
                    triplet_j.append(int(valid_nb_idx[j_local].item()))
                    triplet_k.append(int(valid_nb_idx[k_local].item()))
                    r_ij_list.append(valid_vectors[j_local])
                    r_ik_list.append(valid_vectors[k_local])

        if len(triplet_i) == 0:
            n_ang_features = 2 * n_ang if self.multi else n_ang
            return torch.zeros(
                n_atoms, n_ang_features, n_atoms, 3,
                dtype=self.dtype, device=self.device
            )

        # Stack and prepare tensors
        triplet_i_t = torch.tensor(
            triplet_i, dtype=torch.long, device=self.device)
        triplet_j_t = torch.tensor(
            triplet_j, dtype=torch.long, device=self.device)
        triplet_k_t = torch.tensor(
            triplet_k, dtype=torch.long, device=self.device)
        r_ij = torch.stack(r_ij_list
                           ).to(self.dtype).to(self.device)  # (T, 3)
        r_ik = torch.stack(r_ik_list
                           ).to(self.dtype).to(self.device)  # (T, 3)

        # Distances with tiny epsilon to avoid zero-norm issues
        d_ij = torch.sqrt((r_ij * r_ij).sum(dim=-1) + eps_dist)  # (T,)
        d_ik = torch.sqrt((r_ik * r_ik).sum(dim=-1) + eps_dist)  # (T,)

        # Unit vectors
        u_ij = r_ij / (d_ij.unsqueeze(-1) + eps_norm)  # (T, 3)
        u_ik = r_ik / (d_ik.unsqueeze(-1) + eps_norm)  # (T, 3)

        # Cosine of angle at i
        cos_theta = (u_ij * u_ik).sum(dim=-1).clamp(-1.0, 1.0)  # (T,)

        # Evaluate angular basis and partial derivatives
        # Shapes: (T, n_ang)
        _, dG_dcos, dG_drij, dG_drik = self.ang_basis.forward_with_derivatives(
            d_ij, d_ik, cos_theta
        )

        # Geometric derivatives of cos(theta) wrt positions r_j, r_k, r_i
        # dcos/dr_j = (1/d_ij) * (u_ik - cos_theta * u_ij)
        # dcos/dr_k = (1/d_ik) * (u_ij - cos_theta * u_ik)
        # dcos/dr_i = - (dcos/dr_j + dcos/dr_k)
        dcos_drj = (u_ik - cos_theta.unsqueeze(-1) * u_ij
                    ) / (d_ij.unsqueeze(-1) + eps_norm)  # (T,3)
        dcos_drk = (u_ij - cos_theta.unsqueeze(-1) * u_ik
                    ) / (d_ik.unsqueeze(-1) + eps_norm)  # (T,3)
        dcos_dri = -(dcos_drj + dcos_drk)  # (T,3)

        # Total gradients per triplet and per angular feature (vectorized)
        # Shapes for grads_*: (T, n_ang, 3)
        dG_dcos_e = dG_dcos.unsqueeze(-1)        # (T, n_ang, 1)
        dG_drij_e = dG_drij.unsqueeze(-1)        # (T, n_ang, 1)
        dG_drik_e = dG_drik.unsqueeze(-1)        # (T, n_ang, 1)
        u_ij_e = u_ij.unsqueeze(1)               # (T, 1, 3)
        u_ik_e = u_ik.unsqueeze(1)               # (T, 1, 3)
        dcos_drj_e = dcos_drj.unsqueeze(1)       # (T, 1, 3)
        dcos_drk_e = dcos_drk.unsqueeze(1)       # (T, 1, 3)
        dcos_dri_e = dcos_dri.unsqueeze(1)       # (T, 1, 3)

        # Chain rule combinations
        grads_j = dG_dcos_e * dcos_drj_e + dG_drij_e * u_ij_e
        grads_k = dG_dcos_e * dcos_drk_e + dG_drik_e * u_ik_e
        grads_i = (dG_dcos_e * dcos_dri_e
                   - dG_drij_e * u_ij_e - dG_drik_e * u_ik_e)

        # Initialize output gradient tensor
        n_ang_features = 2 * n_ang if self.multi else n_ang
        gradients = torch.zeros(
            n_atoms, n_ang_features, n_atoms, 3,
            dtype=self.dtype, device=self.device
        )

        # Prepare flattened center-target indices for efficient scatter
        # idx = center * n_atoms + target
        flat_size = n_atoms * n_atoms
        flat_idx_cc = triplet_i_t * n_atoms + triplet_i_t  # (T,)
        flat_idx_cj = triplet_i_t * n_atoms + triplet_j_t  # (T,)
        flat_idx_ck = triplet_i_t * n_atoms + triplet_k_t  # (T,)

        # Accumulate unweighted gradients feature-by-feature
        for ang_idx in range(n_ang):
            # (T,3) slices
            gi = grads_i[:, ang_idx, :]
            gj = grads_j[:, ang_idx, :]
            gk = grads_k[:, ang_idx, :]

            # Accumulate into flattened (center, target) axis
            accum_flat = torch.zeros(
                flat_size, 3, dtype=self.dtype, device=self.device)
            accum_flat.index_add_(0, flat_idx_cc, gi)
            accum_flat.index_add_(0, flat_idx_cj, gj)
            accum_flat.index_add_(0, flat_idx_ck, gk)

            # Reshape back to (n_atoms, n_atoms, 3) and assign
            gradients[:, ang_idx, :, :] = accum_flat.view(n_atoms, n_atoms, 3)

        # Typespin-weighted angular gradients (multi-species only)
        if self.multi:
            species_j = species_indices[triplet_j_t]
            species_k = species_indices[triplet_k_t]
            typespin_prod = (
                self.typespin[species_j] * self.typespin[species_k]
                ).unsqueeze(-1)  # (T,1)

            for ang_idx in range(n_ang):
                gi_w = grads_i[:, ang_idx, :] * typespin_prod
                gj_w = grads_j[:, ang_idx, :] * typespin_prod
                gk_w = grads_k[:, ang_idx, :] * typespin_prod

                accum_flat_w = torch.zeros(
                    flat_size, 3, dtype=self.dtype, device=self.device)
                accum_flat_w.index_add_(0, flat_idx_cc, gi_w)
                accum_flat_w.index_add_(0, flat_idx_cj, gj_w)
                accum_flat_w.index_add_(0, flat_idx_ck, gk_w)

                gradients[:, n_ang + ang_idx, :, :
                          ] = accum_flat_w.view(n_atoms, n_atoms, 3)

        return gradients

    def compute_feature_gradients(
        self,
        positions: torch.Tensor,
        species: List[str],
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute features and their gradients w.r.t. positions efficiently.

        Uses a semi-analytical, vectorized approach that is much faster than
        the naive feature-by-feature autograd loop.

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
        # Get neighbor information
        positions_device = positions.to(self.device).to(self.dtype)
        if cell is not None:
            cell = cell.to(self.device).to(self.dtype)

        neighbor_data = self.nbl.get_neighbors(
            positions_device, cell, pbc, fractional=False
        )

        edge_index = neighbor_data['edge_index']
        distances = neighbor_data['distances']
        offsets = neighbor_data['offsets']

        mask = distances > self.min_cutoff
        edge_index = edge_index[:, mask]
        if offsets is not None:
            offsets = offsets[mask]

        i_indices = edge_index[0]
        j_indices = edge_index[1]

        if cell is not None and offsets is not None:
            r_ij = (
                positions_device[j_indices] + offsets.to(self.dtype) @ cell
            ) - positions_device[i_indices]
        else:
            r_ij = positions_device[j_indices] - positions_device[i_indices]

        n_atoms = len(positions)
        neighbor_indices = []
        neighbor_vectors = []
        for atom_idx in range(n_atoms):
            neighbor_mask = i_indices == atom_idx
            neighbor_indices.append(j_indices[neighbor_mask])
            neighbor_vectors.append(r_ij[neighbor_mask])

        # Compute features
        features = self.forward(
            positions, species, neighbor_indices, neighbor_vectors
        )

        # Convert species to indices for gradient computation
        species_indices = torch.tensor(
            [self.species_to_idx[s] for s in species],
            dtype=torch.long,
            device=self.device,
        )

        # Compute radial and angular gradients
        rad_grads = self._compute_radial_gradients(
            positions, species_indices, neighbor_indices, neighbor_vectors
        )
        ang_grads = self._compute_angular_gradients(
            positions, species_indices, neighbor_indices, neighbor_vectors
        )

        # Combine gradients in the correct feature order
        n_rad = self.rad_order + 1
        n_ang = self.ang_order + 1

        gradients = torch.zeros(
            n_atoms, self.n_features, n_atoms, 3,
            dtype=self.dtype, device=self.device
        )

        if self.multi:
            # Order: [rad_unweighted, ang_unweighted, rad_weighted,
            # ang_weighted]
            gradients[:, :n_rad] = rad_grads[:, :n_rad]
            gradients[:, n_rad:n_rad + n_ang] = ang_grads[:, :n_ang]
            gradients[:, n_rad + n_ang:2 * n_rad + n_ang
                      ] = rad_grads[:, n_rad:]
            gradients[:, 2 * n_rad + n_ang:] = ang_grads[:, n_ang:]
        else:
            gradients[:, :n_rad] = rad_grads
            gradients[:, n_rad:] = ang_grads

        return features, gradients

    def forward_with_graph(
        self,
        positions: torch.Tensor,
        species_indices: torch.Tensor,
        graph,
        triplets=None,
        center_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Vectorized forward using CSR neighbor graph and optional triplets.

        Args:
            positions: (N,3) positions, used only for dtype/device alignment
            species_indices: (N,) long tensor of species indices
            graph: NeighborGraph dict with keys
                   center_ptr[int32 N+1], nbr_idx[int32 E],
                   r_ij[float E,3], d_ij[float E]
            triplets: Optional TripletIndex dict with keys
                      tri_i/j/k[int32 T], tri_j_local/tri_k_local[int32 T]
            center_indices: Optional (M,) indices of centers to include
                            (not used in forward accumulation here)

        Returns
        -------
            features: (N, F)
        """
        device = self.device
        dtype = self.dtype
        positions = positions.to(device=device, dtype=dtype)
        species_indices = species_indices.to(device=device)

        N = positions.shape[0]
        n_rad = self.rad_order + 1
        n_ang = self.ang_order + 1

        # Edge-level arrays
        nbr_idx = graph["nbr_idx"].to(device=device, dtype=torch.int64)
        d_ij = graph["d_ij"].to(device=device, dtype=dtype)
        # Centers per edge from CSR
        center_of_edge = _center_ids_of_edge(graph).to(
            device=device, dtype=torch.int64)

        # Radial basis on all edges
        G_rad = self.rad_basis(d_ij)  # (E, n_rad)

        # Scatter to centers (unweighted radial)
        rad_unw = scatter_add(G_rad, center_of_edge, dim=0, dim_size=N)

        if self.multi:
            # Typespin-weighted radial
            neigh_types = species_indices[nbr_idx]
            tspin_j = self.typespin[neigh_types]  # (E,)
            G_rad_w = G_rad * tspin_j.unsqueeze(-1)
            rad_w = scatter_add(G_rad_w, center_of_edge, dim=0, dim_size=N)
        else:
            rad_w = None

        # Angular features if triplets provided
        if triplets is not None and n_ang > 0:
            center_ptr = graph["center_ptr"].to(
                device=device, dtype=torch.int64)
            r_edges = graph["r_ij"].to(device=device, dtype=dtype)

            tri_i = triplets["tri_i"].to(device=device, dtype=torch.int64)
            tri_j = triplets["tri_j"].to(device=device, dtype=torch.int64)
            tri_k = triplets["tri_k"].to(device=device, dtype=torch.int64)
            tri_j_local = triplets["tri_j_local"].to(
                device=device, dtype=torch.int64)
            tri_k_local = triplets["tri_k_local"].to(
                device=device, dtype=torch.int64)

            # Edge indices within r_ij for (i,j) and (i,k)
            start_i = center_ptr[tri_i]  # (T,)
            edge_j_idx = start_i + tri_j_local  # (T,)
            edge_k_idx = start_i + tri_k_local  # (T,)

            r_ij_vec = r_edges[edge_j_idx]  # (T,3)
            r_ik_vec = r_edges[edge_k_idx]  # (T,3)

            eps = 1e-20
            d_ij_t = torch.sqrt((r_ij_vec * r_ij_vec).sum(dim=-1) + eps)
            d_ik_t = torch.sqrt((r_ik_vec * r_ik_vec).sum(dim=-1) + eps)
            u_ij = r_ij_vec / (d_ij_t.unsqueeze(-1) + 1e-12)
            u_ik = r_ik_vec / (d_ik_t.unsqueeze(-1) + 1e-12)

            cos_theta = (u_ij * u_ik).sum(dim=-1).clamp(-1.0, 1.0)

            # Angular basis over all triplets
            G_ang = self.ang_basis(d_ij_t, d_ik_t, cos_theta)  # (T, n_ang)

            ang_unw = scatter_add(G_ang, tri_i, dim=0, dim_size=N)

            if self.multi:
                tspin_prod = (self.typespin[species_indices[tri_j]]
                              * self.typespin[species_indices[tri_k]])
                G_ang_w = G_ang * tspin_prod.unsqueeze(-1)
                ang_w = scatter_add(G_ang_w, tri_i, dim=0, dim_size=N)
            else:
                ang_w = None
        else:
            ang_unw = torch.zeros(N, n_ang, dtype=dtype, device=device)
            ang_w = torch.zeros_like(ang_unw) if self.multi else None

        # Assemble features in Fortran order used elsewhere
        if self.multi:
            features = torch.cat([rad_unw[:, :n_rad],
                                  ang_unw[:, :n_ang],
                                  rad_w[:, :n_rad],
                                  ang_w[:, :n_ang]], dim=1)
        else:
            features = torch.cat([rad_unw[:, :n_rad],
                                  ang_unw[:, :n_ang]], dim=1)
        return features

    def compute_feature_gradients_with_graph(
        self,
        positions: torch.Tensor,
        species_indices: torch.Tensor,
        graph,
        triplets,
        center_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized features and gradients using CSR neighbors and triplets.

        Returns
        -------
            features: (N, F)
            gradients: (N, F, N, 3)
        """
        device = self.device
        dtype = self.dtype
        positions = positions.to(device=device, dtype=dtype)
        species_indices = species_indices.to(device=device)

        N = positions.shape[0]
        n_rad = self.rad_order + 1
        n_ang = self.ang_order + 1

        # Forward pass via graph
        features = self.forward_with_graph(
            positions=positions,
            species_indices=species_indices,
            graph=graph,
            triplets=triplets,
            center_indices=center_indices,
        )

        # Prepare output gradients
        n_features = self.get_n_features()
        gradients = torch.zeros(N, n_features, N, 3,
                                dtype=dtype, device=device)

        # Edge-level data for radial gradients
        nbr_idx = graph["nbr_idx"].to(device=device, dtype=torch.int64)
        r_edges = graph["r_ij"].to(device=device, dtype=dtype)
        d_edges = torch.norm(r_edges, dim=-1).clamp_min(1e-20)
        u_edges = r_edges / (d_edges.unsqueeze(-1) + 1e-12)
        center_of_edge = _center_ids_of_edge(graph).to(
            device=device, dtype=torch.int64)

        # dG/dr for radial basis
        _, dG_dr = self.rad_basis.forward_with_derivatives(
            d_edges)  # (E, n_rad)
        dG_drij = dG_dr.unsqueeze(-1) * u_edges.unsqueeze(1)  # (E, n_rad, 3)

        # Accumulate unweighted radial gradients
        flat_size = N * N
        idx_cc = (center_of_edge * N + center_of_edge)  # (E,)
        idx_cj = (center_of_edge * N + nbr_idx)         # (E,)

        for k in range(n_rad):
            accum = torch.zeros(flat_size, 3, dtype=dtype, device=device)
            gk = dG_drij[:, k, :]  # (E,3)
            accum.index_add_(0, idx_cc, -gk)
            accum.index_add_(0, idx_cj, gk)
            gradients[:, k, :, :] = accum.view(N, N, 3)

        if self.multi:
            # Weighted radial with typespin(j)
            # (E,1,1) broadcast over (n_rad,3)
            tspin_j = self.typespin[species_indices[nbr_idx]].view(-1, 1, 1)
            dG_w = dG_drij * tspin_j  # (E, n_rad, 3)
            for k in range(n_rad):
                accum = torch.zeros(flat_size, 3, dtype=dtype, device=device)
                gk = dG_w[:, k, :]
                accum.index_add_(0, idx_cc, -gk)
                accum.index_add_(0, idx_cj, gk)
                gradients[:, n_rad + n_ang + k, :, :] = accum.view(N, N, 3)

        # Angular gradients via triplets
        if triplets is not None and n_ang > 0:
            center_ptr = graph["center_ptr"].to(
                device=device, dtype=torch.int64)

            tri_i = triplets["tri_i"].to(device=device, dtype=torch.int64)
            tri_j = triplets["tri_j"].to(device=device, dtype=torch.int64)
            tri_k = triplets["tri_k"].to(device=device, dtype=torch.int64)
            tri_j_local = triplets["tri_j_local"].to(
                device=device, dtype=torch.int64)
            tri_k_local = triplets["tri_k_local"].to(
                device=device, dtype=torch.int64)

            start_i = center_ptr[tri_i]
            edge_j_idx = start_i + tri_j_local
            edge_k_idx = start_i + tri_k_local

            r_ij = r_edges[edge_j_idx]
            r_ik = r_edges[edge_k_idx]

            eps = 1e-20
            d_ij = torch.sqrt((r_ij * r_ij).sum(dim=-1) + eps)
            d_ik = torch.sqrt((r_ik * r_ik).sum(dim=-1) + eps)
            u_ij = r_ij / (d_ij.unsqueeze(-1) + 1e-12)
            u_ik = r_ik / (d_ik.unsqueeze(-1) + 1e-12)

            # Angular basis derivatives
            (_, dG_dcos, dG_drij, dG_drik
             ) = self.ang_basis.forward_with_derivatives(
                d_ij, d_ik, (u_ij * u_ik).sum(dim=-1).clamp(-1.0, 1.0)
            )  # (T, n_ang) each

            # Geometry derivatives of cos(theta)
            cos_theta = (u_ij * u_ik).sum(dim=-1)
            dcos_drj = (u_ik - cos_theta.unsqueeze(-1) * u_ij
                        ) / (d_ij.unsqueeze(-1) + 1e-12)
            dcos_drk = (u_ij - cos_theta.unsqueeze(-1) * u_ik
                        ) / (d_ik.unsqueeze(-1) + 1e-12)
            dcos_dri = -(dcos_drj + dcos_drk)

            dG_dcos_e = dG_dcos.unsqueeze(-1)   # (T, n_ang, 1)
            dG_drij_e = dG_drij.unsqueeze(-1)   # (T, n_ang, 1)
            dG_drik_e = dG_drik.unsqueeze(-1)   # (T, n_ang, 1)
            u_ij_e = u_ij.unsqueeze(1)          # (T, 1, 3)
            u_ik_e = u_ik.unsqueeze(1)          # (T, 1, 3)
            dcos_drj_e = dcos_drj.unsqueeze(1)  # (T, 1, 3)
            dcos_drk_e = dcos_drk.unsqueeze(1)  # (T, 1, 3)
            dcos_dri_e = dcos_dri.unsqueeze(1)  # (T, 1, 3)

            grads_j = (dG_dcos_e * dcos_drj_e
                       + dG_drij_e * u_ij_e)  # (T,n_ang,3)
            grads_k = (dG_dcos_e * dcos_drk_e
                       + dG_drik_e * u_ik_e)  # (T,n_ang,3)
            grads_i = (dG_dcos_e * dcos_dri_e
                       - dG_drij_e * u_ij_e
                       - dG_drik_e * u_ik_e)

            flat_size = N * N
            idx_cc = tri_i * N + tri_i
            idx_cj = tri_i * N + tri_j
            idx_ck = tri_i * N + tri_k

            # Unweighted
            for a in range(n_ang):
                accum = torch.zeros(flat_size, 3, dtype=dtype, device=device)
                gi = grads_i[:, a, :]
                gj = grads_j[:, a, :]
                gk = grads_k[:, a, :]
                accum.index_add_(0, idx_cc, gi)
                accum.index_add_(0, idx_cj, gj)
                accum.index_add_(0, idx_ck, gk)
                gradients[:, n_rad + a, :, :] = accum.view(N, N, 3)

            if self.multi:
                tsp = (self.typespin[species_indices[tri_j]] *
                       self.typespin[species_indices[tri_k]]).unsqueeze(-1)
                for a in range(n_ang):
                    accum = torch.zeros(flat_size, 3,
                                        dtype=dtype, device=device)
                    gi = grads_i[:, a, :] * tsp
                    gj = grads_j[:, a, :] * tsp
                    gk = grads_k[:, a, :] * tsp
                    accum.index_add_(0, idx_cc, gi)
                    accum.index_add_(0, idx_cj, gj)
                    accum.index_add_(0, idx_ck, gk)
                    gradients[:, 2 * n_rad + n_ang + a, :, :
                              ] = accum.view(N, N, 3)

        return features, gradients

    def compute_feature_gradients_from_neighbor_info(
        self,
        positions: torch.Tensor,
        species: List[str],
        neighbor_indices: List[torch.Tensor],
        neighbor_vectors: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute features and their gradients using precomputed neighbor info.

        This avoids recomputing neighbors and displacement vectors by reusing
        neighbor_indices and neighbor_vectors produced earlier (e.g., by
        featurize_with_neighbor_info).

        Args:
            positions: (N, 3) atomic positions
            species: List of N species names
            neighbor_indices: List of (nnb_i,) tensors with neighbor indices
            neighbor_vectors: List of (nnb_i, 3) tensors with
                displacement vectors

        Returns
        -------
            features: (N, F) feature tensor
            gradients: (N, F, N, 3) gradient tensor where
                       gradients[i, f, j, k] = ∂feature[i,f]/∂position[j,k]
        """
        # Move positions to device/dtype
        positions_device = positions.to(self.device).to(self.dtype)

        # Compute features using provided neighbor data
        features = self.forward(
            positions_device, species, neighbor_indices, neighbor_vectors
        )

        # Species indices
        species_indices = torch.tensor(
            [self.species_to_idx[s] for s in species],
            dtype=torch.long,
            device=self.device,
        )

        # Compute radial and angular gradients using provided neighbor data
        rad_grads = self._compute_radial_gradients(
            positions_device, species_indices,
            neighbor_indices, neighbor_vectors
        )
        ang_grads = self._compute_angular_gradients(
            positions_device, species_indices,
            neighbor_indices, neighbor_vectors
        )

        # Combine in correct feature order
        n_rad = self.rad_order + 1
        n_ang = self.ang_order + 1

        n_atoms = positions_device.shape[0]
        gradients = torch.zeros(
            n_atoms, self.n_features, n_atoms, 3,
            dtype=self.dtype, device=self.device
        )

        if self.multi:
            # [rad_unweighted, ang_unweighted, rad_weighted, ang_weighted]
            gradients[:, :n_rad] = rad_grads[:, :n_rad]
            gradients[:, n_rad:n_rad + n_ang] = ang_grads[:, :n_ang]
            gradients[:, n_rad + n_ang:2 * n_rad + n_ang
                      ] = rad_grads[:, n_rad:]
            gradients[:, 2 * n_rad + n_ang:] = ang_grads[:, n_ang:]
        else:
            gradients[:, :n_rad] = rad_grads
            gradients[:, n_rad:] = ang_grads

        return features, gradients

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

        # Compute features using convenience wrapper
        features = self.forward_from_positions(
            positions_grad, species, cell, pbc)

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

        # Featurize using convenience wrapper
        with torch.no_grad():
            features = self.forward_from_positions(
                pos_torch, species, cell_torch, pbc_torch)

        # Convert back to numpy
        return features.cpu().numpy()

    def featurize_with_neighbor_info(
        self,
        positions: torch.Tensor,
        species: List[str],
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Featurize structure and extract neighbor information for
        force training.

        This method computes atomic features and extracts neighbor lists and
        displacement vectors needed for computing feature derivatives during
        force training. The neighbor information can be saved to HDF5 for
        later use with on-demand derivative computation.

        Args:
            positions: (N, 3) atomic positions (Cartesian coordinates)
            species: List of N species names
            cell: (3, 3) lattice vectors as rows (None for isolated systems)
            pbc: (3,) periodic boundary conditions (default: all True if
              cell provided)

        Returns
        -------
            features: (N, n_features) feature tensor
            neighbor_info: Dictionary containing:
                - 'neighbor_counts': (N,) number of neighbors per atom
                - 'neighbor_lists': List of N arrays, each containing
                      neighbor indices
                - 'neighbor_vectors': List of N arrays, each (nnb, 3)
                      displacement vectors
                - 'max_neighbors': int, maximum number of neighbors
                      across all atoms

        Example
        -------
            >>> descriptor = ChebyshevDescriptor(['O', 'H'], 10, 4.0, 3, 1.5)
            >>> positions = torch.tensor([[0.0, 0.0, 0.0],
            ...                           [0.0, 0.0, 1.0],
            ...                           [0.0, 1.0, 0.0]])
            >>> features, neighbor_info = \
            ...         descriptor.featurize_with_neighbor_info(
            ...     positions, ['O', 'H', 'H']
            ... )
            >>> print(neighbor_info['neighbor_counts'])  # tensor([2, 1, 1])
            >>> print(neighbor_info['max_neighbors'])    # 2

        Notes
        -----
        The neighbor cutoff used is the maximum of rad_cutoff and ang_cutoff
        to ensure all neighbors relevant for both radial and angular features
        are included.

        The displacement vectors are computed as r_j - r_i for neighbor j
        of atom i, and include periodic image offsets when applicable.
        """
        # Move inputs to device
        positions = positions.to(self.device).to(self.dtype)
        if cell is not None:
            cell = cell.to(self.device).to(self.dtype)

        # Get neighbor data using the maximum cutoff
        # Use fractional=False since positions are assumed Cartesian
        neighbor_data = self.nbl.get_neighbors(
            positions, cell, pbc, fractional=False
        )

        edge_index = neighbor_data['edge_index']
        distances = neighbor_data['distances']
        offsets = neighbor_data['offsets']  # None for isolated systems

        # Filter by minimum cutoff (remove too-close neighbors)
        mask = distances > self.min_cutoff
        edge_index = edge_index[:, mask]
        distances = distances[mask]
        if offsets is not None:
            offsets = offsets[mask]

        # Compute displacement vectors for all neighbor pairs
        i_indices = edge_index[0]
        j_indices = edge_index[1]

        if cell is not None and offsets is not None:
            # Periodic system: include cell offsets
            r_ij = (
                positions[j_indices] + offsets.to(self.dtype) @ cell
            ) - positions[i_indices]
        else:
            # Isolated system
            r_ij = positions[j_indices] - positions[i_indices]

        # Organize neighbor information per atom
        n_atoms = len(positions)
        neighbor_counts = torch.zeros(
            n_atoms, dtype=torch.long, device=self.device)
        neighbor_indices_list = []
        neighbor_vectors_list = []

        for atom_idx in range(n_atoms):
            # Find all neighbors of this atom
            neighbor_mask = i_indices == atom_idx
            atom_neighbors = j_indices[neighbor_mask]
            atom_vectors = r_ij[neighbor_mask]

            neighbor_counts[atom_idx] = len(atom_neighbors)
            neighbor_indices_list.append(atom_neighbors)
            neighbor_vectors_list.append(atom_vectors)

        # Compute features using core forward method with neighbor data
        features = self.forward(
            positions, species, neighbor_indices_list, neighbor_vectors_list)

        # Package neighbor info for output (convert to numpy)
        neighbor_lists_np = [nb.cpu().numpy() for nb in neighbor_indices_list]
        neighbor_vectors_np = [vec.cpu().numpy()
                               for vec in neighbor_vectors_list]

        max_neighbors = neighbor_counts.max().item() if n_atoms > 0 else 0

        neighbor_info = {
            'neighbor_counts': neighbor_counts.cpu().numpy(),
            'neighbor_lists': neighbor_lists_np,
            'neighbor_vectors': neighbor_vectors_np,
            'max_neighbors': max_neighbors,
        }

        return features, neighbor_info


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

            # Use convenience wrapper
            features = self.featurizer.forward_from_positions(
                pos, species, cell, pbc)
            all_features.append(features)
            all_batch_indices.append(
                torch.full((len(pos),), batch_idx, dtype=torch.long)
            )

        # Concatenate
        features = torch.cat(all_features, dim=0)
        batch_indices = torch.cat(all_batch_indices, dim=0)

        return features, batch_indices
