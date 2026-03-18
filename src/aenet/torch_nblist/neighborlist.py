"""
PyTorch-based neighbor list for atomic structures.

Supports:
- Periodic boundary conditions (PBC) with arbitrary cell shapes
- Isolated systems (molecules)
- GPU acceleration
- Double precision
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


try:
    from torch_cluster import radius, radius_graph

    TORCH_CLUSTER_AVAILABLE = True
except ImportError:
    TORCH_CLUSTER_AVAILABLE = False
    warnings.warn(
        "torch_cluster not available. Install with: "
        "pip install torch-cluster -f "
        "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
    )


class TorchNeighborList:
    """
    PyTorch-based neighbor list for atomic structures.

    Supports:
    - Periodic boundary conditions (PBC)
    - Isolated systems
    - GPU acceleration
    - Double precision

    Example:
        >>> nbl = TorchNeighborList(cutoff=4.0, device='cpu')
        >>> positions = torch.randn(10, 3, dtype=torch.float64)
        >>> result = nbl.get_neighbors(positions)
        >>> edge_index = result['edge_index']  # (2, num_edges)
        >>> distances = result['distances']    # (num_edges,)

    Notes:
        - Default PBC backend is 'ghost' which constructs a support set of
          ghost images near periodic boundaries and performs a single
          bipartite radius search (support -> central).
        - Offsets returned under PBC are integer lattice offsets defined
          relative to wrapped fractional coordinates (positions wrapped
          to [0,1) before offsetting). Downstream code should reconstruct
          displacements as:
          r_ij = ((frac[j] + offsets) @ cell) - (frac[i] @ cell)
          where frac = remainder(positions @ inv(cell), 1.0).
        - Fully differentiable and GPU-compatible. Truncation is handled by
          auto-growing max_num_neighbors up to auto_max_neighbors with
          warnings.
    """

    def __init__(
        self,
        cutoff: float,
        atom_types: Optional[torch.Tensor] = None,
        cutoff_dict: Optional[Dict[Tuple[int, int], float]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        max_num_neighbors: int = 256,
        truncation_handling: str = "auto",
        auto_multiplier: int = 2,
        auto_max_neighbors: int = 65536,
        pbc_backend: str = "ghost",
    ):
        """
        Initialize neighbor list.

        Args:
            cutoff: Maximum interaction cutoff radius in Angstroms
            atom_types: (N,) tensor of atom types (e.g., atomic numbers)
            cutoff_dict: Dict mapping (type_i, type_j) tuples to cutoff
                distances. Keys should be sorted tuples: (min, max)
            device: 'cpu' or 'cuda'
            dtype: torch.float32 or torch.float64 (recommended: float64)
            max_num_neighbors: Maximum number of neighbors per atom to consider
                (default: 256). Increase if you encounter systems with very
                dense neighbor environments.

        Raises
        ------
            ValueError: If cutoff_dict contains types not in atom_types
            ValueError: If cutoff_dict values exceed maximum cutoff
        """
        if not TORCH_CLUSTER_AVAILABLE:
            raise ImportError(
                "torch_cluster is required but not installed. "
                "Install with: pip install torch-cluster"
            )

        self.cutoff = cutoff
        self.atom_types = atom_types
        self.cutoff_dict = cutoff_dict
        self.device = device
        self.dtype = dtype
        self.max_num_neighbors = max_num_neighbors
        self.truncation_handling = truncation_handling
        self.auto_multiplier = int(auto_multiplier)
        self.auto_max_neighbors = int(auto_max_neighbors)
        self.pbc_backend = pbc_backend
        # Deprecation notice for legacy backend
        if self.pbc_backend == "legacy":
            warnings.warn(
                "TorchNeighborList: legacy PBC backend is deprecated and will"
                "be removed in a future release. Use pbc_backend='ghost' "
                "(default).",
                DeprecationWarning,
            )

        # Validate cutoff_dict if both types and dict are provided
        if cutoff_dict is not None and atom_types is not None:
            self._validate_cutoff_dict(cutoff_dict, atom_types)

        # Cache for efficiency
        self._cached_result = None
        self._cache_key = None

    def _radius_with_inclusive(self) -> float:
        """
        Return radius slightly larger than cutoff to include pairs at exactly
        the cutoff distance (matching legacy <= cutoff behavior).
        Use a conservative epsilon to account for accumulated FP error from
        replication, transforms, and torch_cluster distance computation.
        """
        eps = 1e-6 if self.dtype == torch.float64 else 1e-5
        return float(self.cutoff + eps)

    def _to_tensor(
        self,
        array: Union[np.ndarray, torch.Tensor],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Convert numpy array or torch tensor to appropriate tensor.

        Args:
            array: Input array (numpy or torch)
            dtype: Target dtype (uses self.dtype if not specified)

        Returns
        -------
            torch.Tensor on self.device with appropriate dtype
        """
        if dtype is None:
            dtype = self.dtype

        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
        else:
            tensor = array

        return tensor.to(self.device).to(dtype)

    @classmethod
    def from_AtomicStructure(
        cls,
        structure,
        cutoff: float,
        frame: int = -1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        max_num_neighbors: int = 256,
    ):
        """
        Factory method: create neighbor list from AtomicStructure.

        Args:
            structure: Instance of aenet.geometry.AtomicStructure
            cutoff: Maximum interaction cutoff radius in Angstroms
            frame: Frame index to use (default: -1 for last frame)
            device: 'cpu' or 'cuda'
            dtype: torch.float32 or torch.float64 (recommended: float64)
            max_num_neighbors: Maximum number of neighbors per atom

        Returns
        -------
            TorchNeighborList instance configured for the structure

        Example:
            >>> from aenet.geometry import AtomicStructure
            >>> from aenet.torch_featurize import TorchNeighborList
            >>> structure = AtomicStructure(coords, types, avec=avec)
            >>> nbl = TorchNeighborList.from_AtomicStructure(
            ...     structure, cutoff=4.0
            ... )
        """
        return cls(
            cutoff=cutoff,
            device=device,
            dtype=dtype,
            max_num_neighbors=max_num_neighbors,
        )

    def get_neighbors(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
        fractional: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Unified interface for neighbor finding.

        Args:
            positions: (N, 3) atom positions

                - For isolated systems: Always Cartesian coordinates
                  in Angstroms
                - For periodic systems: Fractional [0,1) or Cartesian,
                  see fractional arg

            cell: (3, 3) lattice vectors as rows (None for isolated systems)
            pbc: (3,) boolean tensor for PBC in each direction
                 (default: [True, True, True] if cell is provided)
            fractional: For periodic systems only. If True, positions
              are fractional coordinates [0, 1). If False, positions are
              Cartesian (Angstroms).

        Returns
        -------
            Dictionary containing:
            - 'edge_index': (2, num_edges) neighbor pairs [source, target]
            - 'distances': (num_edges,) pairwise distances in Angstroms
            - 'offsets': (num_edges, 3) cell offsets (None for isolated
                 systems)
            - 'num_neighbors': (N,) number of neighbors per atom
        """
        if cell is None:
            # Isolated system
            edge_index, distances = self.get_neighbors_isolated(positions)
            num_neighbors = self._count_neighbors(
                edge_index, positions.shape[0]
            )
            return {
                "edge_index": edge_index,
                "distances": distances,
                "offsets": None,
                "num_neighbors": num_neighbors,
            }
        else:
            # Periodic system
            edge_index, distances, offsets = self.get_neighbors_pbc(
                positions, cell, pbc, fractional
            )
            num_neighbors = self._count_neighbors(
                edge_index, positions.shape[0]
            )
            return {
                "edge_index": edge_index,
                "distances": distances,
                "offsets": offsets,
                "num_neighbors": num_neighbors,
            }

    def get_neighbors_isolated(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find neighbors for isolated system (no PBC).

        Args:
            positions: (N, 3) atom positions in Angstroms (Cartesian)

        Returns
        -------
            edge_index: (2, num_edges) neighbor pairs [source, target]
            distances: (num_edges,) pairwise distances in Angstroms
        """
        positions = positions.to(self.device).to(self.dtype)

        # Handle single atom case
        if positions.shape[0] <= 1:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            distances = torch.empty(0, dtype=self.dtype, device=self.device)
            return edge_index, distances

        # Use radius_graph from torch_cluster with truncation handling
        # Start with a higher baseline in isolated mode to reduce retries
        cur_max_nn = int(max(self.max_num_neighbors, 2048))
        while True:
            edge_index = radius_graph(
                positions,
                r=self._radius_with_inclusive(),
                max_num_neighbors=cur_max_nn,
                flow="source_to_target",
                loop=False,  # Don't include self-loops
            )

            # Compute distances
            if edge_index.shape[1] > 0:
                row, col = edge_index
                diff = positions[row] - positions[col]
                distances = torch.norm(diff, dim=1)
            else:
                distances = torch.empty(
                    0, dtype=self.dtype, device=self.device)

            # Check truncation: any source node with degree == cur_max_nn
            num_neighbors = self._count_neighbors(
                edge_index, positions.shape[0])
            truncated = (num_neighbors.numel() > 0
                         and num_neighbors.max().item() >= cur_max_nn)

            if truncated:
                if (self.truncation_handling == "auto"
                        and cur_max_nn < self.auto_max_neighbors):
                    new_max = min(cur_max_nn * self.auto_multiplier,
                                  self.auto_max_neighbors)
                    warnings.warn(
                        f"TorchNeighborList: max_num_neighbors={cur_max_nn} "
                        "hit in isolated mode; "
                        f"retrying with max_num_neighbors={new_max}.",
                        RuntimeWarning,
                    )
                    cur_max_nn = int(new_max)
                    continue
                elif self.truncation_handling == "error":
                    raise RuntimeError(
                        "TorchNeighborList: neighbor list truncated at "
                        f"max_num_neighbors={cur_max_nn} "
                        f"(isolated). Increase max_num_neighbors or "
                        "reduce cutoff."
                    )
                elif self.truncation_handling == "warn":
                    warnings.warn(
                        f"TorchNeighborList: neighbor list may be "
                        "truncated at "
                        f"max_num_neighbors={cur_max_nn} (isolated).",
                        RuntimeWarning
                    )
            # Update stored value if auto grew
            self.max_num_neighbors = max(self.max_num_neighbors, cur_max_nn)
            return edge_index, distances

    def get_neighbors_pbc(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: Optional[torch.Tensor] = None,
        fractional: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find neighbors with periodic boundary conditions.

        Args:
            positions: (N, 3) atom positions
            cell: (3, 3) lattice vectors as rows
            pbc: (3,) boolean tensor for PBC in each direction
                 (default: [True, True, True])
            fractional: If True, positions are fractional [0,1).
                       If False, positions are Cartesian (Angstroms).

        Returns
        -------
            edge_index: (2, num_edges) neighbor pairs [source, target]
            distances: (num_edges,) pairwise distances in Angstroms
            offsets: (num_edges, 3) cell offset vectors for each edge
        """
        backend = getattr(self, "pbc_backend", "legacy")
        if backend == "ghost":
            return self._get_neighbors_pbc_cell_list(
                positions, cell, pbc, fractional
            )
        elif backend == "legacy":
            return self._get_neighbors_pbc_old(
                positions, cell, pbc, fractional
            )
        else:
            raise ValueError(f"Unknown pbc_backend: {backend}")

    def _get_neighbors_pbc_cell_list(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: Optional[torch.Tensor] = None,
        fractional: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized PBC neighbor finding using ghost atoms.

        Instead of iterating over cells in Python (slow), we replicate atoms
        near the boundaries to create "ghost" atoms, then run a single
        radius_graph on the extended system.
        """
        if pbc is None:
            pbc = torch.tensor(
                [True, True, True], dtype=torch.bool, device=self.device
            )
        else:
            pbc = pbc.to(self.device)

        positions = positions.to(self.device).to(self.dtype)
        cell = cell.to(self.device).to(self.dtype)

        # Ensure we have fractional coordinates
        if not fractional:
            try:
                cell_inv = torch.linalg.inv(cell)
                positions_frac = positions @ cell_inv
            except RuntimeError:
                positions_frac = torch.zeros_like(positions)
        else:
            positions_frac = positions

        # Wrap fractional coordinates to [0, 1)
        positions_frac = torch.remainder(positions_frac, 1.0)

        # Calculate perpendicular widths of the cell to determine cutoff
        # in fractional coords
        # Volume = det(cell)
        # Area_i = |a_j x a_k|
        # Width_i = Volume / Area_i
        volume = torch.abs(torch.det(cell)).clamp_min(1e-12)
        cross_0 = torch.norm(torch.linalg.cross(cell[1], cell[2]))
        cross_1 = torch.norm(torch.linalg.cross(cell[2], cell[0]))
        cross_2 = torch.norm(torch.linalg.cross(cell[0], cell[1]))

        widths = volume / torch.stack([cross_0, cross_1, cross_2]
                                      ).clamp_min(1e-12)
        cutoff_frac = (self.cutoff / widths).to(self.device) * 1.01
        # Number of periodic image layers needed along each lattice vector
        search_layers = torch.ceil(self.cutoff / widths).to(torch.long)

        # Start with original atoms
        # We track: (positions, indices, offsets)
        # indices: original atom index
        # offsets: (3,) integer offset of the image

        # Current set of atoms (starts with central cell)
        current_pos = positions_frac
        current_indices = torch.arange(len(positions), device=self.device)
        current_offsets = torch.zeros((len(positions), 3),
                                      dtype=torch.long,
                                      device=self.device)

        # Iteratively replicate along each periodic dimension
        for i in range(3):
            if not pbc[i]:
                continue

            # Determine how many image layers are required along axis i
            s_i = int(search_layers[i].item())
            if s_i <= 0:
                continue

            # For each layer k, replicate a snapshot of the current set
            # to avoid re-replicating newly added ghosts within the same k
            for k in range(1, s_i + 1):
                base_pos_k = current_pos
                base_idx_k = current_indices
                base_off_k = current_offsets

                # Lower boundary: replicate to +k (image +k)
                mask_lower = base_pos_k[:, i] < cutoff_frac[i]
                if mask_lower.any():
                    ghosts_pos_lower = base_pos_k[mask_lower].clone()
                    ghosts_pos_lower[:, i] = ghosts_pos_lower[:, i] + float(k)
                    ghosts_indices_lower = base_idx_k[mask_lower]
                    ghosts_offsets_lower = base_off_k[mask_lower].clone()
                    ghosts_offsets_lower[:, i] = ghosts_offsets_lower[:, i] + k

                    current_pos = torch.cat([current_pos, ghosts_pos_lower],
                                            dim=0)
                    current_indices = torch.cat([current_indices,
                                                 ghosts_indices_lower], dim=0)
                    current_offsets = torch.cat([current_offsets,
                                                 ghosts_offsets_lower], dim=0)

                # Upper boundary: replicate to -k (image -k)
                mask_upper = base_pos_k[:, i] > (1.0 - cutoff_frac[i])
                if mask_upper.any():
                    ghosts_pos_upper = base_pos_k[mask_upper].clone()
                    ghosts_pos_upper[:, i] = ghosts_pos_upper[:, i] - float(k)
                    ghosts_indices_upper = base_idx_k[mask_upper]
                    ghosts_offsets_upper = base_off_k[mask_upper].clone()
                    ghosts_offsets_upper[:, i] = ghosts_offsets_upper[:, i] - k

                    current_pos = torch.cat([current_pos, ghosts_pos_upper],
                                            dim=0)
                    current_indices = torch.cat([current_indices,
                                                 ghosts_indices_upper], dim=0)
                    current_offsets = torch.cat([current_offsets,
                                                 ghosts_offsets_upper], dim=0)

        # Deduplicate ghosts by (original_index, offsets) to avoid duplicates
        # that can arise from different replication paths.
        if current_pos.shape[0] > 0:
            keys = torch.stack(
                [
                    current_indices.to(torch.long),
                    current_offsets[:, 0],
                    current_offsets[:, 1],
                    current_offsets[:, 2],
                ],
                dim=1,
            )
            # Build first-occurrence mask using a Python set
            # (small int tensors)
            seen = set()
            keep_list = []
            for r in keys.cpu().tolist():
                t = tuple(r)
                if t in seen:
                    keep_list.append(False)
                else:
                    seen.add(t)
                    keep_list.append(True)
            keep_mask = torch.tensor(keep_list, device=self.device,
                                     dtype=torch.bool)
            current_pos = current_pos[keep_mask]
            current_indices = current_indices[keep_mask]
            current_offsets = current_offsets[keep_mask]

        # Convert all fractional positions to Cartesian
        all_cart_pos = current_pos @ cell

        # Run radius_graph on the full set
        # We want neighbors for the original atoms (first N atoms)
        # But radius_graph(x) computes all pairs.
        # We can filter later, or use bipartite radius(x, y).
        # x = all atoms (potential neighbors)
        # y = original atoms (central cell)

        n_orig = len(positions)
        central_cart_pos = all_cart_pos[:n_orig]

        # Use radius(x, y) -> neighbors of y in x
        # x = support (all), y = query (central)
        cur_max_nn = int(max(self.max_num_neighbors, 4096))
        while True:
            row, col = radius(
                x=all_cart_pos,
                y=central_cart_pos,
                r=self._radius_with_inclusive(),
                max_num_neighbors=cur_max_nn,
                num_workers=1,
            )
            # Check truncation on the query (central) nodes
            truncated = False
            if row.numel() > 0:
                deg = torch.bincount(row, minlength=n_orig)
                truncated = deg.numel() > 0 and deg.max().item() >= cur_max_nn

            if truncated:
                if (self.truncation_handling == "auto"
                        and cur_max_nn < self.auto_max_neighbors):
                    new_max = min(cur_max_nn * self.auto_multiplier,
                                  self.auto_max_neighbors)
                    warnings.warn(
                        f"TorchNeighborList: max_num_neighbors={cur_max_nn} "
                        "hit under PBC (bipartite radius); "
                        f"retrying with max_num_neighbors={new_max}.",
                        RuntimeWarning,
                    )
                    cur_max_nn = int(new_max)
                    continue
                elif self.truncation_handling == "error":
                    raise RuntimeError(
                        "TorchNeighborList: neighbor list truncated at "
                        f"max_num_neighbors={cur_max_nn} (PBC bipartite). "
                        "Increase max_num_neighbors or reduce cutoff."
                    )
                elif self.truncation_handling == "warn":
                    warnings.warn(
                        f"TorchNeighborList: neighbor list may be truncated "
                        f"at max_num_neighbors={cur_max_nn} (PBC bipartite).",
                        RuntimeWarning
                    )
            break
        # Update stored value if auto grew
        self.max_num_neighbors = max(self.max_num_neighbors, cur_max_nn)

        # row: indices in central_cart_pos (0 to N-1) -> source
        # col: indices in all_cart_pos -> target

        # Filter out self-loops (where index is same AND offset is zero)
        # target indices (col) map to original indices via current_indices[col]
        target_indices = current_indices[col]
        target_offsets = current_offsets[col]

        # Self loop condition: source_idx == target_idx AND target_offset == 0
        # row is already source_idx (since we queried with
        # central atoms 0..N-1)
        mask_self = (row == target_indices) & (torch.all(target_offsets == 0,
                                                         dim=1))
        mask_valid = ~mask_self

        # Apply mask
        row = row[mask_valid]
        col = col[mask_valid]
        target_indices = target_indices[mask_valid]
        target_offsets = target_offsets[mask_valid]

        # Compute distances
        diff = central_cart_pos[row] - all_cart_pos[col]
        distances = torch.norm(diff, dim=1)

        # Construct edge_index
        edge_index = torch.stack([row, target_indices], dim=0)

        return edge_index, distances, target_offsets

    def _get_neighbors_pbc_old(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: Optional[torch.Tensor] = None,
        fractional: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Legacy implementation: Find neighbors with periodic boundary
        conditions.  (Kept for validation purposes)
        """
        if pbc is None:
            pbc = torch.tensor(
                [True, True, True], dtype=torch.bool, device=self.device
            )
        else:
            pbc = pbc.to(self.device)

        positions = positions.to(self.device).to(self.dtype)
        cell = cell.to(self.device).to(self.dtype)

        # Convert to Cartesian if needed
        if fractional:
            cart_positions = positions @ cell
        else:
            # Use original Cartesian positions without wrapping to keep
            # offsets consistent with the absolute coordinates returned.
            cart_positions = positions

        # Determine search range in each direction
        search_cells = self._determine_search_cells(cell, pbc)

        # Create replicated positions for periodic images
        all_positions, all_offsets = self._create_periodic_images(
            cart_positions, cell, search_cells, pbc
        )

        # Find neighbors using radius_graph with truncation handling
        # Start with a higher baseline under PBC to avoid premature truncation
        cur_max_nn = int(max(self.max_num_neighbors, 4096))
        while True:
            edge_index_all = radius_graph(
                all_positions,
                r=self._radius_with_inclusive(),
                max_num_neighbors=cur_max_nn,
                flow="source_to_target",
                loop=False,
            )

            # Filter and compute distances for central cell sources
            edge_index, distances, cell_offsets = self._process_pbc_edges(
                edge_index_all, all_positions, all_offsets, positions.shape[0]
            )

            # Check truncation on central unit cell sources.
            # We must detect truncation against the degrees computed on the
            # full replicated graph (edge_index_all), not the filtered graph.
            # Build per-source degree for all_positions nodes:
            degrees_all = torch.zeros(
                all_positions.shape[0], dtype=torch.long, device=self.device
            )
            if edge_index_all.shape[1] > 0:
                src_nodes = edge_index_all[0]
                unique_src, counts_src = torch.unique(
                    src_nodes, return_counts=True)
                degrees_all[unique_src] = counts_src
            # Identify the replicated source nodes that are in the central cell
            central_source_mask = torch.all(all_offsets == 0, dim=1)
            if torch.any(central_source_mask):
                degrees_central = degrees_all[central_source_mask]
                truncated = (degrees_central.numel() > 0
                             and degrees_central.max().item() >= cur_max_nn)
            else:
                truncated = False

            if truncated:
                if (self.truncation_handling == "auto"
                        and cur_max_nn < self.auto_max_neighbors):
                    new_max = min(cur_max_nn * self.auto_multiplier,
                                  self.auto_max_neighbors)
                    warnings.warn(
                        f"TorchNeighborList: max_num_neighbors={cur_max_nn} "
                        "hit under PBC; "
                        f"retrying with max_num_neighbors={new_max}.",
                        RuntimeWarning,
                    )
                    cur_max_nn = int(new_max)
                    continue
                elif self.truncation_handling == "error":
                    raise RuntimeError(
                        "TorchNeighborList: neighbor list truncated at "
                        f"max_num_neighbors={cur_max_nn} "
                        f"(PBC). Increase max_num_neighbors or reduce cutoff."
                    )
                elif self.truncation_handling == "warn":
                    warnings.warn(
                        f"TorchNeighborList: neighbor list may be "
                        "truncated at "
                        f"max_num_neighbors={cur_max_nn} (PBC).",
                        RuntimeWarning
                    )

            # Update stored value if auto grew
            self.max_num_neighbors = max(self.max_num_neighbors, cur_max_nn)
            return edge_index, distances, cell_offsets

    def _determine_search_cells(
        self, cell: torch.Tensor, pbc: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine how many periodic images to check in each direction.

        Args:
            cell: (3, 3) lattice vectors as rows
            pbc: (3,) boolean tensor for PBC

        Returns
        -------
            search_cells: (3,) integer number of cells to search in each
                          direction
        """
        # Distances between opposite faces of the unit cell:
        # d_i = Volume / Area(face_i), where face_i is
        # opposite lattice vector i.
        # This works for arbitrary (skewed) cells.
        volume = torch.abs(torch.det(cell)).clamp_min(1e-12)

        a = cell[0]
        b = cell[1]
        c = cell[2]

        area0 = torch.norm(torch.linalg.cross(b, c))  # face normal to a
        area1 = torch.norm(torch.linalg.cross(c, a))  # face normal to b
        area2 = torch.norm(torch.linalg.cross(a, b))  # face normal to c

        areas = torch.stack([area0, area1, area2]).clamp_min(1e-12)
        face_distances = volume / areas  # perpendicular distance between faces

        # Number of cells needed to cover the cutoff along each direction
        # Add +1 safety margin to ensure full coverage (corner/cross effects)
        search_cells = torch.ceil(self.cutoff / face_distances
                                  ).to(torch.long) + 1

        # Zero out non-periodic directions
        search_cells = torch.where(
            pbc, search_cells, torch.zeros_like(search_cells)
        )

        return search_cells

    def _create_periodic_images(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        search_cells: torch.Tensor,
        pbc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create periodic images of atoms.

        Args:
            positions: (N, 3) Cartesian positions
            cell: (3, 3) lattice vectors
            search_cells: (3,) number of cells to search in each direction
            pbc: (3,) boolean for PBC

        Returns
        -------
            all_positions: (num_images * N, 3) Cartesian positions including
               periodic images
            offsets: (num_images * N, 3) cell offset for each position
        """
        n_atoms = positions.shape[0]

        # Generate offset vectors
        ranges = []
        for s, p in zip(search_cells, pbc):
            if p:
                ranges.append(
                    torch.arange(
                        -s, s + 1, device=self.device, dtype=torch.long
                    )
                )
            else:
                ranges.append(
                    torch.tensor([0], device=self.device, dtype=torch.long)
                )

        # Create meshgrid of offsets
        offset_grid = torch.stack(
            torch.meshgrid(*ranges, indexing="ij"), dim=-1
        ).reshape(-1, 3)

        # Replicate positions for each offset
        replicated_positions = positions.unsqueeze(0).expand(
            offset_grid.shape[0], -1, -1
        )

        # Compute offset vectors in Cartesian coordinates
        offset_vectors = (
            offset_grid.to(self.dtype).unsqueeze(1) @ cell
        ).expand(-1, n_atoms, -1)

        # Apply offsets
        all_positions = (replicated_positions + offset_vectors).reshape(-1, 3)
        all_offsets = (
            offset_grid.unsqueeze(1).expand(-1, n_atoms, -1).reshape(-1, 3)
        )

        return all_positions, all_offsets

    def _process_pbc_edges(
        self,
        edge_index: torch.Tensor,
        all_positions: torch.Tensor,
        all_offsets: torch.Tensor,
        n_atoms: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process edges from periodic images and compute distances.

        Args:
            edge_index: (2, num_edges) edge indices in replicated system
            all_positions: (num_images * N, 3) all positions including images
            all_offsets: (num_images * N, 3) cell offsets for each position
            n_atoms: number of atoms in unit cell

        Returns
        -------
            edge_index: (2, num_edges) filtered edge indices (in unit cell)
            distances: (num_edges,) distances
            cell_offsets: (num_edges, 3) cell offsets for each edge
        """
        if edge_index.shape[1] == 0:
            # No edges found
            distances = torch.empty(0, dtype=self.dtype, device=self.device)
            cell_offsets = torch.empty(
                (0, 3), dtype=torch.long, device=self.device
            )
            return edge_index, distances, cell_offsets

        row, col = edge_index

        # Map replicated indices back to unit cell indices
        unit_cell_row = row % n_atoms
        unit_cell_col = col % n_atoms

        # Get cell offsets
        offset_row = all_offsets[row]
        offset_col = all_offsets[col]
        cell_offsets = offset_col - offset_row

        # Compute distances
        diff = all_positions[row] - all_positions[col]
        distances = torch.norm(diff, dim=1)

        # Filter: only keep edges where source is in central unit cell
        # (to avoid double counting)
        central_cell_mask = torch.all(offset_row == 0, dim=1)

        # Also remove self-interactions in the central cell
        self_interaction_mask = (unit_cell_row == unit_cell_col) & (
            torch.all(cell_offsets == 0, dim=1)
        )
        valid_mask = central_cell_mask & (~self_interaction_mask)

        # Apply filters
        edge_index = torch.stack(
            [unit_cell_row[valid_mask], unit_cell_col[valid_mask]]
        )
        distances = distances[valid_mask]
        cell_offsets = cell_offsets[valid_mask]

        return edge_index, distances, cell_offsets

    def _count_neighbors(
        self, edge_index: torch.Tensor, n_atoms: int
    ) -> torch.Tensor:
        """
        Count number of neighbors for each atom.

        Args:
            edge_index: (2, num_edges) edge indices
            n_atoms: total number of atoms

        Returns
        -------
            num_neighbors: (n_atoms,) number of neighbors per atom
        """
        if edge_index.shape[1] == 0:
            return torch.zeros(n_atoms, dtype=torch.long, device=self.device)

        num_neighbors = torch.zeros(
            n_atoms, dtype=torch.long, device=self.device
        )
        unique, counts = torch.unique(edge_index[0], return_counts=True)
        num_neighbors[unique] = counts

        return num_neighbors

    def _validate_cutoff_dict(
        self,
        cutoff_dict: Dict[Tuple[int, int], float],
        atom_types: torch.Tensor,
    ) -> None:
        """
        Validate that cutoff_dict is consistent with atom_types.

        Args:
            cutoff_dict: Dictionary of pair cutoffs
            atom_types: Tensor of atom types

        Raises
        ------
            ValueError: If cutoff_dict keys contain undefined types
            ValueError: If any cutoff exceeds self.cutoff
        """
        unique_types = set(atom_types.cpu().numpy().tolist())

        for (type_i, type_j), cutoff_val in cutoff_dict.items():
            # Check types are defined
            if type_i not in unique_types:
                raise ValueError(
                    f"Type {type_i} in cutoff_dict not found in "
                    f"atom_types. Available types: {sorted(unique_types)}"
                )
            if type_j not in unique_types:
                raise ValueError(
                    f"Type {type_j} in cutoff_dict not found in "
                    f"atom_types. Available types: {sorted(unique_types)}"
                )

            # Check cutoff doesn't exceed maximum
            if cutoff_val > self.cutoff:
                raise ValueError(
                    f"Cutoff {cutoff_val} for pair ({type_i}, {type_j}) "
                    f"exceeds maximum cutoff {self.cutoff}"
                )

    def get_neighbors_of_atom(
        self,
        atom_idx: int,
        positions: Union[np.ndarray, torch.Tensor],
        cell: Optional[Union[np.ndarray, torch.Tensor]] = None,
        pbc: Optional[torch.Tensor] = None,
        fractional: bool = True,
        atom_types: Optional[torch.Tensor] = None,
        cutoff_dict: Optional[Dict[Tuple[int, int], float]] = None,
        return_coordinates: bool = False,
        full_star: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get neighbors of a specific atom.

        Args:
            atom_idx: Index of the atom to query
            positions: (N, 3) atom positions (numpy array or torch tensor)
            cell: (3, 3) lattice vectors (None for isolated systems,
                  numpy array or torch tensor)
            pbc: (3,) PBC flags
            fractional: For periodic systems only. If True, positions
                are fractional coordinates [0, 1). If False, positions are
                Cartesian (Angstroms).
                Default: True for backward compatibility.
            atom_types: Override stored atom_types (optional)
            cutoff_dict: Override stored cutoff_dict (optional)
            return_coordinates: If True, also return actual neighbor
                coordinates with PBC offsets applied (default: False)
            full_star: If True, return all neighbors in both directions
                (where atom_idx is source and target). This is useful for
                extracting complete neighbor clusters for periodic systems.
                Default: False (returns half-star for efficiency).

        Returns
        -------
            Dictionary containing:
            - 'indices': (num_neighbors,) neighbor atom indices
            - 'distances': (num_neighbors,) distances to neighbors
            - 'offsets': (num_neighbors, 3) cell offsets (or None)
            - 'coordinates': (num_neighbors, 3) neighbor coordinates
                (only if return_coordinates=True)

        Note:
            If both atom_types and cutoff_dict are provided (either
            stored or as arguments), neighbors are filtered by
            type-specific cutoffs.
        """
        # Convert inputs to tensors
        positions = self._to_tensor(positions)
        if cell is not None:
            cell = self._to_tensor(cell)
        # Use stored values or overrides
        types = atom_types if atom_types is not None else self.atom_types
        cutoffs = cutoff_dict if cutoff_dict is not None else self.cutoff_dict

        # Validate if both are provided
        if cutoffs is not None and types is not None:
            self._validate_cutoff_dict(cutoffs, types)

        # Get or compute full neighbor list (with caching)
        result = self._get_or_compute_neighbors(
            positions, cell, pbc, fractional)

        # Extract neighbors of specific atom
        edge_index = result["edge_index"]

        if full_star:
            # Include both directions: where atom_idx is source AND target
            # Source edges (atom_idx -> neighbors)
            mask_source = edge_index[0] == atom_idx
            neighbor_indices_source = edge_index[1][mask_source]
            distances_source = result["distances"][mask_source]
            offsets_source = (
                result["offsets"][mask_source]
                if result["offsets"] is not None else None
            )

            # Target edges (neighbors -> atom_idx)
            # For these, we need to flip the direction
            mask_target = edge_index[1] == atom_idx
            neighbor_indices_target = edge_index[0][mask_target]
            distances_target = result["distances"][mask_target]
            offsets_target = (
                -result["offsets"][mask_target]  # Flip offset direction
                if result["offsets"] is not None else None
            )

            # Combine both sets of neighbors
            neighbor_indices = torch.cat(
                [neighbor_indices_source, neighbor_indices_target])
            distances = torch.cat([distances_source, distances_target])
            if offsets_source is not None and offsets_target is not None:
                offsets = torch.cat([offsets_source, offsets_target])

                # Remove duplicates based on (neighbor_index, offset) pairs
                # Create unique keys by combining index and offset
                unique_mask = []
                seen = set()

                for i in range(len(neighbor_indices)):
                    idx = neighbor_indices[i].item()
                    off = tuple(offsets[i].cpu().numpy())
                    key = (idx, off)
                    if key not in seen:
                        seen.add(key)
                        unique_mask.append(True)
                    else:
                        unique_mask.append(False)

                unique_mask = torch.tensor(unique_mask, device=self.device)
                neighbor_indices = neighbor_indices[unique_mask]
                distances = distances[unique_mask]
                offsets = offsets[unique_mask]
            else:
                offsets = None
        else:
            # Half-star: only edges where atom_idx is the source
            mask = edge_index[0] == atom_idx
            neighbor_indices = edge_index[1][mask]
            distances = result["distances"][mask]
            offsets = (
                result["offsets"][mask]
                if result["offsets"] is not None else None
            )

        # Apply type-specific cutoff filtering if applicable
        if types is not None and cutoffs is not None:
            filter_mask = self._filter_by_type_cutoff(
                atom_idx, neighbor_indices, distances, types, cutoffs
            )
            neighbor_indices = neighbor_indices[filter_mask]
            distances = distances[filter_mask]
            if offsets is not None:
                offsets = offsets[filter_mask]

        # Compute coordinates if requested
        coordinates = None
        if return_coordinates:
            coordinates = positions[neighbor_indices]
            if offsets is not None and cell is not None:
                # Apply PBC offsets: convert to float and apply cell matrix
                coordinates = coordinates + (offsets.to(self.dtype) @ cell)

        result_dict = {
            "indices": neighbor_indices,
            "distances": distances,
            "offsets": offsets,
        }

        if return_coordinates:
            result_dict["coordinates"] = coordinates

        return result_dict

    def get_neighbors_by_atom(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
        atom_types: Optional[torch.Tensor] = None,
        cutoff_dict: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> List[Dict[str, Optional[torch.Tensor]]]:
        """
        Get neighbors for all atoms in structured format.

        Args:
            positions: (N, 3) atom positions
            cell: (3, 3) lattice vectors (None for isolated systems)
            pbc: (3,) PBC flags
            atom_types: Override stored atom_types (optional)
            cutoff_dict: Override stored cutoff_dict (optional)

        Returns
        -------
            List of length N_atoms, where each element is a dict:
            - 'indices': neighbor indices
            - 'distances': distances
            - 'offsets': cell offsets
        """
        n_atoms = positions.shape[0]
        result = []

        for i in range(n_atoms):
            neighbors = self.get_neighbors_of_atom(
                i, positions, cell, pbc, atom_types, cutoff_dict
            )
            result.append(neighbors)

        return result

    def _filter_by_type_cutoff(
        self,
        source_idx: int,
        neighbor_indices: torch.Tensor,
        distances: torch.Tensor,
        atom_types: torch.Tensor,
        cutoff_dict: Dict[Tuple[int, int], float],
    ) -> torch.Tensor:
        """
        Filter neighbors based on type-dependent cutoffs.

        Args:
            source_idx: Index of source atom
            neighbor_indices: Indices of neighbor atoms
            distances: Distances to neighbors
            atom_types: Atom types tensor
            cutoff_dict: Dictionary of pair cutoffs

        Returns
        -------
            Boolean mask for neighbors within type-specific cutoffs
        """
        source_type = atom_types[source_idx].item()
        mask = torch.zeros(
            len(neighbor_indices), dtype=torch.bool, device=self.device
        )

        for i, (neigh_idx, dist) in enumerate(
            zip(neighbor_indices, distances)
        ):
            neigh_type = atom_types[neigh_idx].item()
            # Use sorted tuple as key
            pair = tuple(sorted([source_type, neigh_type]))
            pair_cutoff = cutoff_dict.get(pair, self.cutoff)
            mask[i] = dist <= pair_cutoff

        return mask

    def _get_or_compute_neighbors(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        pbc: Optional[torch.Tensor],
        fractional: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get cached neighbor list or compute new one.

        Args:
            positions: Atom positions
            cell: Lattice vectors
            pbc: PBC flags
            fractional: Whether positions are fractional or Cartesian

        Returns
        -------
            Neighbor list result dictionary
        """
        # Compute cache key
        cache_key = self._compute_cache_key(positions, cell, pbc, fractional)

        # Return cached result if available
        if cache_key == self._cache_key and self._cached_result is not None:
            return self._cached_result

        # Compute new result
        result = self.get_neighbors(positions, cell, pbc, fractional)

        # Update cache
        self._cache_key = cache_key
        self._cached_result = result

        return result

    def _compute_cache_key(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        pbc: Optional[torch.Tensor],
        fractional: bool = True,
    ) -> Tuple:
        """
        Compute cache key for positions/cell/pbc combination.

        Args:
            positions: Atom positions
            cell: Lattice vectors
            pbc: PBC flags
            fractional: Whether positions are fractional or Cartesian

        Returns
        -------
            Tuple that can be used as cache key
        """
        # Use hash of tensor data
        pos_hash = hash(positions.cpu().numpy().tobytes())
        cell_hash = (
            hash(cell.cpu().numpy().tobytes()) if cell is not None else None
        )
        pbc_hash = (
            hash(pbc.cpu().numpy().tobytes()) if pbc is not None else None
        )

        return (pos_hash, cell_hash, pbc_hash, fractional)

    def __repr__(self) -> str:
        return (
            f"TorchNeighborList(cutoff={self.cutoff}, "
            f"device='{self.device}', dtype={self.dtype}, "
            f"pbc_backend='{getattr(self, 'pbc_backend', 'ghost')}')"
        )
