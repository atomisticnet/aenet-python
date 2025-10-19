"""
PyTorch-based neighbor list for atomic structures.

Supports:
- Periodic boundary conditions (PBC) with arbitrary cell shapes
- Isolated systems (molecules)
- GPU acceleration
- Double precision
"""

import torch
from typing import Optional, Tuple, Dict, List
import warnings

try:
    from torch_cluster import radius_graph
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
    """

    def __init__(
        self,
        cutoff: float,
        atom_types: Optional[torch.Tensor] = None,
        cutoff_dict: Optional[Dict[Tuple[int, int], float]] = None,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64
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

        Raises:
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

        # Validate cutoff_dict if both types and dict are provided
        if cutoff_dict is not None and atom_types is not None:
            self._validate_cutoff_dict(cutoff_dict, atom_types)

        # Cache for efficiency
        self._cached_result = None
        self._cache_key = None

    def get_neighbors(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Unified interface for neighbor finding.

        Args:
            positions: (N, 3) atom positions
                - For isolated systems: Cartesian coordinates in Angstroms
                - For periodic systems: Fractional coordinates [0, 1)
            cell: (3, 3) lattice vectors as rows (None for isolated systems)
            pbc: (3,) boolean tensor for PBC in each direction
                 (default: [True, True, True] if cell is provided)

        Returns:
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
                edge_index, positions.shape[0])
            return {
                'edge_index': edge_index,
                'distances': distances,
                'offsets': None,
                'num_neighbors': num_neighbors
            }
        else:
            # Periodic system
            edge_index, distances, offsets = self.get_neighbors_pbc(
                positions, cell, pbc
            )
            num_neighbors = self._count_neighbors(
                edge_index, positions.shape[0])
            return {
                'edge_index': edge_index,
                'distances': distances,
                'offsets': offsets,
                'num_neighbors': num_neighbors
            }

    def get_neighbors_isolated(
        self,
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find neighbors for isolated system (no PBC).

        Args:
            positions: (N, 3) atom positions in Angstroms (Cartesian)

        Returns:
            edge_index: (2, num_edges) neighbor pairs [source, target]
            distances: (num_edges,) pairwise distances in Angstroms
        """
        positions = positions.to(self.device).to(self.dtype)

        # Handle single atom case
        if positions.shape[0] <= 1:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device)
            distances = torch.empty(0, dtype=self.dtype, device=self.device)
            return edge_index, distances

        # Use radius_graph from torch_cluster
        edge_index = radius_graph(
            positions,
            r=self.cutoff,
            max_num_neighbors=256,
            flow='source_to_target',
            loop=False  # Don't include self-loops
        )

        # Compute distances
        if edge_index.shape[1] > 0:
            row, col = edge_index
            diff = positions[row] - positions[col]
            distances = torch.norm(diff, dim=1)
        else:
            distances = torch.empty(0, dtype=self.dtype, device=self.device)

        return edge_index, distances

    def get_neighbors_pbc(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find neighbors with periodic boundary conditions.

        Args:
            positions: (N, 3) fractional coordinates [0, 1)
            cell: (3, 3) lattice vectors as rows
            pbc: (3,) boolean tensor for PBC in each direction
                 (default: [True, True, True])

        Returns:
            edge_index: (2, num_edges) neighbor pairs [source, target]
            distances: (num_edges,) pairwise distances in Angstroms
            offsets: (num_edges, 3) cell offset vectors for each edge
        """
        if pbc is None:
            pbc = torch.tensor(
                [True, True, True], dtype=torch.bool, device=self.device)
        else:
            pbc = pbc.to(self.device)

        positions = positions.to(self.device).to(self.dtype)
        cell = cell.to(self.device).to(self.dtype)

        # Convert fractional to Cartesian
        cart_positions = positions @ cell

        # Determine search range in each direction
        search_cells = self._determine_search_cells(cell, pbc)

        # Create replicated positions for periodic images
        all_positions, all_offsets = self._create_periodic_images(
            cart_positions, cell, search_cells, pbc
        )

        # Find neighbors using radius_graph
        edge_index = radius_graph(
            all_positions,
            r=self.cutoff,
            max_num_neighbors=256,
            flow='source_to_target',
            loop=False
        )

        # Filter and compute distances
        edge_index, distances, cell_offsets = self._process_pbc_edges(
            edge_index, all_positions, all_offsets, positions.shape[0]
        )

        return edge_index, distances, cell_offsets

    def _determine_search_cells(
        self,
        cell: torch.Tensor,
        pbc: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine how many periodic images to check in each direction.

        Args:
            cell: (3, 3) lattice vectors as rows
            pbc: (3,) boolean tensor for PBC

        Returns:
            search_cells: (3,) integer number of cells to search in each
                          direction
        """
        # Compute reciprocal lattice vectors
        volume = torch.abs(torch.det(cell))
        reciprocal = torch.linalg.inv(cell.T)

        # Distance from origin to cell face (perpendicular distance)
        face_distances = volume / torch.norm(reciprocal, dim=1)

        # Number of cells needed to cover cutoff
        search_cells = torch.ceil(self.cutoff / face_distances).long()
        search_cells = torch.where(pbc, search_cells, torch.zeros_like(
            search_cells))

        return search_cells

    def _create_periodic_images(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        search_cells: torch.Tensor,
        pbc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create periodic images of atoms.

        Args:
            positions: (N, 3) Cartesian positions
            cell: (3, 3) lattice vectors
            search_cells: (3,) number of cells to search in each direction
            pbc: (3,) boolean for PBC

        Returns:
            all_positions: (num_images * N, 3) Cartesian positions including
               periodic images
            offsets: (num_images * N, 3) cell offset for each position
        """
        n_atoms = positions.shape[0]

        # Generate offset vectors
        ranges = []
        for s, p in zip(search_cells, pbc):
            if p:
                ranges.append(torch.arange(
                    -s, s + 1, device=self.device, dtype=torch.long))
            else:
                ranges.append(torch.tensor(
                    [0], device=self.device, dtype=torch.long))

        # Create meshgrid of offsets
        offset_grid = torch.stack(
            torch.meshgrid(*ranges, indexing='ij'), dim=-1
        ).reshape(-1, 3)

        # Replicate positions for each offset
        replicated_positions = positions.unsqueeze(0).expand(
            offset_grid.shape[0], -1, -1)

        # Compute offset vectors in Cartesian coordinates
        offset_vectors = (offset_grid.to(self.dtype).unsqueeze(1) @ cell
                          ).expand(-1, n_atoms, -1)

        # Apply offsets
        all_positions = (replicated_positions + offset_vectors).reshape(-1, 3)
        all_offsets = offset_grid.unsqueeze(1).expand(-1, n_atoms, -1
                                                      ).reshape(-1, 3)

        return all_positions, all_offsets

    def _process_pbc_edges(
        self,
        edge_index: torch.Tensor,
        all_positions: torch.Tensor,
        all_offsets: torch.Tensor,
        n_atoms: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process edges from periodic images and compute distances.

        Args:
            edge_index: (2, num_edges) edge indices in replicated system
            all_positions: (num_images * N, 3) all positions including images
            all_offsets: (num_images * N, 3) cell offsets for each position
            n_atoms: number of atoms in unit cell

        Returns:
            edge_index: (2, num_edges) filtered edge indices (in unit cell)
            distances: (num_edges,) distances
            cell_offsets: (num_edges, 3) cell offsets for each edge
        """
        if edge_index.shape[1] == 0:
            # No edges found
            distances = torch.empty(0, dtype=self.dtype, device=self.device)
            cell_offsets = torch.empty(
                (0, 3), dtype=torch.long, device=self.device)
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
            torch.all(cell_offsets == 0, dim=1))
        valid_mask = central_cell_mask & (~self_interaction_mask)

        # Apply filters
        edge_index = torch.stack(
            [unit_cell_row[valid_mask], unit_cell_col[valid_mask]])
        distances = distances[valid_mask]
        cell_offsets = cell_offsets[valid_mask]

        return edge_index, distances, cell_offsets

    def _count_neighbors(self, edge_index: torch.Tensor, n_atoms: int
                         ) -> torch.Tensor:
        """
        Count number of neighbors for each atom.

        Args:
            edge_index: (2, num_edges) edge indices
            n_atoms: total number of atoms

        Returns:
            num_neighbors: (n_atoms,) number of neighbors per atom
        """
        if edge_index.shape[1] == 0:
            return torch.zeros(n_atoms, dtype=torch.long, device=self.device)

        num_neighbors = torch.zeros(
            n_atoms, dtype=torch.long, device=self.device)
        unique, counts = torch.unique(edge_index[0], return_counts=True)
        num_neighbors[unique] = counts

        return num_neighbors

    def _validate_cutoff_dict(
        self,
        cutoff_dict: Dict[Tuple[int, int], float],
        atom_types: torch.Tensor
    ) -> None:
        """
        Validate that cutoff_dict is consistent with atom_types.

        Args:
            cutoff_dict: Dictionary of pair cutoffs
            atom_types: Tensor of atom types

        Raises:
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
        positions: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
        atom_types: Optional[torch.Tensor] = None,
        cutoff_dict: Optional[Dict[Tuple[int, int], float]] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get neighbors of a specific atom.

        Args:
            atom_idx: Index of the atom to query
            positions: (N, 3) atom positions
            cell: (3, 3) lattice vectors (None for isolated systems)
            pbc: (3,) PBC flags
            atom_types: Override stored atom_types (optional)
            cutoff_dict: Override stored cutoff_dict (optional)

        Returns:
            Dictionary containing:
            - 'indices': (num_neighbors,) neighbor atom indices
            - 'distances': (num_neighbors,) distances to neighbors
            - 'offsets': (num_neighbors, 3) cell offsets (or None)

        Note:
            If both atom_types and cutoff_dict are provided (either
            stored or as arguments), neighbors are filtered by
            type-specific cutoffs.
        """
        # Use stored values or overrides
        types = atom_types if atom_types is not None else self.atom_types
        cutoffs = (cutoff_dict if cutoff_dict is not None
                   else self.cutoff_dict)

        # Validate if both are provided
        if cutoffs is not None and types is not None:
            self._validate_cutoff_dict(cutoffs, types)

        # Get or compute full neighbor list (with caching)
        result = self._get_or_compute_neighbors(positions, cell, pbc)

        # Extract neighbors of specific atom
        edge_index = result['edge_index']
        mask = edge_index[0] == atom_idx

        neighbor_indices = edge_index[1][mask]
        distances = result['distances'][mask]
        offsets = (result['offsets'][mask]
                   if result['offsets'] is not None else None)

        # Apply type-specific cutoff filtering if applicable
        if types is not None and cutoffs is not None:
            filter_mask = self._filter_by_type_cutoff(
                atom_idx, neighbor_indices, distances, types, cutoffs
            )
            neighbor_indices = neighbor_indices[filter_mask]
            distances = distances[filter_mask]
            if offsets is not None:
                offsets = offsets[filter_mask]

        return {
            'indices': neighbor_indices,
            'distances': distances,
            'offsets': offsets
        }

    def get_neighbors_by_atom(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
        atom_types: Optional[torch.Tensor] = None,
        cutoff_dict: Optional[Dict[Tuple[int, int], float]] = None
    ) -> List[Dict[str, Optional[torch.Tensor]]]:
        """
        Get neighbors for all atoms in structured format.

        Args:
            positions: (N, 3) atom positions
            cell: (3, 3) lattice vectors (None for isolated systems)
            pbc: (3,) PBC flags
            atom_types: Override stored atom_types (optional)
            cutoff_dict: Override stored cutoff_dict (optional)

        Returns:
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
        cutoff_dict: Dict[Tuple[int, int], float]
    ) -> torch.Tensor:
        """
        Filter neighbors based on type-dependent cutoffs.

        Args:
            source_idx: Index of source atom
            neighbor_indices: Indices of neighbor atoms
            distances: Distances to neighbors
            atom_types: Atom types tensor
            cutoff_dict: Dictionary of pair cutoffs

        Returns:
            Boolean mask for neighbors within type-specific cutoffs
        """
        source_type = atom_types[source_idx].item()
        mask = torch.zeros(len(neighbor_indices), dtype=torch.bool,
                           device=self.device)

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
        pbc: Optional[torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get cached neighbor list or compute new one.

        Args:
            positions: Atom positions
            cell: Lattice vectors
            pbc: PBC flags

        Returns:
            Neighbor list result dictionary
        """
        # Compute cache key
        cache_key = self._compute_cache_key(positions, cell, pbc)

        # Return cached result if available
        if cache_key == self._cache_key and self._cached_result is not None:
            return self._cached_result

        # Compute new result
        result = self.get_neighbors(positions, cell, pbc)

        # Update cache
        self._cache_key = cache_key
        self._cached_result = result

        return result

    def _compute_cache_key(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        pbc: Optional[torch.Tensor]
    ) -> Tuple:
        """
        Compute cache key for positions/cell/pbc combination.

        Args:
            positions: Atom positions
            cell: Lattice vectors
            pbc: PBC flags

        Returns:
            Tuple that can be used as cache key
        """
        # Use hash of tensor data
        pos_hash = hash(positions.cpu().numpy().tobytes())
        cell_hash = hash(cell.cpu().numpy().tobytes()
                         ) if cell is not None else None
        pbc_hash = hash(pbc.cpu().numpy().tobytes()
                        ) if pbc is not None else None

        return (pos_hash, cell_hash, pbc_hash)

    def __repr__(self) -> str:
        return (
            f"TorchNeighborList(cutoff={self.cutoff}, "
            f"device='{self.device}', dtype={self.dtype})"
        )
