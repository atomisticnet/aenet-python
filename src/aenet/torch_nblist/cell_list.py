"""
Cell list utilities for TorchNeighborList.

Provides a lightweight cell-list structure that partitions fractional
coordinates into a 3D grid for efficient periodic neighbor searches.
"""

from __future__ import annotations

from typing import Generator, List, Optional, Tuple

import torch


class CellList:
    """
    Helper structure to organize atoms into a regular grid of sub-cells
    within the unit cell. Used to limit the search space for periodic
    neighbor finding.

    Parameters
    ----------
    positions_frac : torch.Tensor
        (N, 3) fractional coordinates in [0, 1).
    pbc : torch.Tensor
        (3,) boolean tensor indicating periodic directions.
    cell : torch.Tensor
        (3, 3) lattice vectors (rows).
    cutoff : float
        Interaction cutoff radius in Angstroms.
    device : str
        Torch device.
    dtype : torch.dtype
        Torch floating dtype.

    Notes
    -----
    - Non-periodic directions always use a single cell.
    - Periodic directions use at least one cell; more are created when the
      cell dimension is significantly larger than the cutoff.
    """

    def __init__(
        self,
        positions_frac: torch.Tensor,
        pbc: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.pbc = pbc.to(torch.bool).to(self.device)
        self.cell = cell.to(self.device).to(self.dtype)
        self.cutoff = float(cutoff)

        # Fractional coordinates wrapped to [0, 1)
        self.positions_frac = torch.remainder(
            positions_frac.to(self.device).to(self.dtype), 1.0
        )

        self.grid_size = self._compute_grid_size()
        self.grid_size_float = self.grid_size.to(self.dtype)
        self.total_cells = int(torch.prod(self.grid_size).item())

        self.strides = torch.tensor(
            [
                self.grid_size[1] * self.grid_size[2],
                self.grid_size[2],
                1,
            ],
            dtype=torch.long,
            device=self.device,
        )

        self.atom_cells = self._assign_atoms_to_cells()
        self.cell_atoms = self._group_atoms_by_cell()
        self.cell_coords = self._compute_cell_coordinates()

    # ------------------------------------------------------------------
    # Grid construction helpers
    # ------------------------------------------------------------------
    def _compute_grid_size(self) -> torch.Tensor:
        """
        Determine the number of sub-cells along each lattice vector.

        Returns
        -------
        torch.Tensor
            (3,) tensor of integers.
        """
        lengths = torch.norm(self.cell, dim=1).clamp_min(1e-8)
        # Heuristic: aim for roughly cutoff-sized sub-cells.
        grid = torch.ceil(lengths / max(self.cutoff, 1e-8)).to(torch.long)
        grid = torch.clamp(grid, min=1)

        # Ensure at least one cell in non-periodic directions.
        grid = torch.where(self.pbc, grid, torch.ones_like(grid))

        return grid

    def _assign_atoms_to_cells(self) -> torch.Tensor:
        """
        Map each atom to a cell index (i, j, k).

        Returns
        -------
        torch.Tensor
            (N, 3) tensor of integer cell coordinates.
        """
        scaled = self.positions_frac * self.grid_size_float
        cell_indices = torch.floor(scaled + 1e-8).to(torch.long)
        cell_indices = torch.minimum(cell_indices, self.grid_size - 1)
        return cell_indices

    def _group_atoms_by_cell(self) -> List[List[int]]:
        """
        Group atom indices by cell.

        Returns
        -------
        list[list[int]]
            List where entry i contains the indices of atoms in cell i.
        """
        cell_atoms: List[List[int]] = [[] for _ in range(self.total_cells)]
        flat_indices = self.flatten_indices(self.atom_cells).tolist()
        for atom_idx, cell_id in enumerate(flat_indices):
            cell_atoms[cell_id].append(atom_idx)
        return cell_atoms

    def _compute_cell_coordinates(self) -> torch.Tensor:
        """
        Generate integer coordinates for each cell.

        Returns
        -------
        torch.Tensor
            (total_cells, 3) tensor of integer coordinates.
        """
        ranges = [
            torch.arange(gs, device=self.device, dtype=torch.long)
            for gs in self.grid_size
        ]
        coords = torch.stack(
            torch.meshgrid(*ranges, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        return coords

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def iter_cells(self) -> Generator[Tuple[int, torch.Tensor, List[int]],
                                      None, None]:
        """
        Iterate over cells that contain at least one atom.

        Yields
        ------
        tuple
            (cell_id, cell_coord, atom_indices)
        """
        for cell_id, atoms in enumerate(self.cell_atoms):
            if atoms:
                yield cell_id, self.cell_coords[cell_id], atoms

    def generate_neighbor_deltas(
        self, search_cells: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate relative cell offsets to consider for neighbor cells.

        Parameters
        ----------
        search_cells : torch.Tensor
            (3,) tensor indicating how many unit cells must be searched
            along each lattice vector.

        Returns
        -------
        torch.Tensor
            (M, 3) tensor of integer offsets (excluding [0, 0, 0]).
        """
        ranges: List[torch.Tensor] = []
        for axis in range(3):
            s = int(search_cells[axis].item())
            if self.pbc[axis]:
                rng = torch.arange(-s, s + 1, device=self.device,
                                   dtype=torch.long)
            else:
                # Non-periodic directions only consider the current cell.
                rng = torch.tensor([0], device=self.device, dtype=torch.long)
            ranges.append(rng)

        deltas = torch.stack(
            torch.meshgrid(*ranges, indexing="ij"), dim=-1
        ).reshape(-1, 3)

        if deltas.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.long, device=self.device)

        # Remove zero offset (central cell handled separately).
        non_zero = torch.any(deltas != 0, dim=1)
        return deltas[non_zero]

    def resolve_neighbor(
        self, cell_coord: torch.Tensor, delta: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Resolve a neighboring cell relative to a base cell plus offset.

        Parameters
        ----------
        cell_coord : torch.Tensor
            (3,) integer coordinates of the base cell.
        delta : torch.Tensor
            (3,) integer offsets.

        Returns
        -------
        tuple or None
            (wrapped_coord, offset_cells, atom_indices) if valid,
            otherwise None.
        """
        coord = cell_coord + delta
        wrap_coord = coord.clone()
        offset_cells = torch.zeros(3, dtype=torch.long, device=self.device)

        for axis in range(3):
            gs = int(self.grid_size[axis].item())
            if not bool(self.pbc[axis]):
                if wrap_coord[axis] < 0 or wrap_coord[axis] >= gs:
                    return None
                # No translation along non-periodic axes.
                continue

            # Compute wrap using floor division for negative values.
            offset_axis = torch.floor_divide(wrap_coord[axis], gs)
            wrap_coord[axis] = wrap_coord[axis] - offset_axis * gs
            offset_cells[axis] = offset_axis

        neighbor_cell_id = self.flatten_coord(wrap_coord)
        neighbor_atoms = self.cell_atoms[neighbor_cell_id]
        return wrap_coord, offset_cells, neighbor_atoms

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def flatten_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert (N, 3) cell coordinates to flat indices.

        Parameters
        ----------
        indices : torch.Tensor
            (N, 3) integer coordinates.

        Returns
        -------
        torch.Tensor
            (N,) flat indices.
        """
        return torch.sum(indices * self.strides, dim=1)

    def flatten_coord(self, coord: torch.Tensor) -> int:
        """
        Convert a single 3D coordinate to a flat index.

        Parameters
        ----------
        coord : torch.Tensor
            (3,) integer coordinate.

        Returns
        -------
        int
            Flat index.
        """
        return int(torch.dot(coord.to(self.device), self.strides).item())
