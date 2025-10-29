"""
Graph builders and lightweight structures for CSR neighbors and triplets.

This module provides:
- NeighborGraph: CSR representation of pairwise neighbors per center atom
- TripletIndex: Flat representation of (i,j,k) angular triplets with local
  indices into each center's CSR row for fast gathers of r_ij/r_ik.

Builders:
- build_csr_from_neighborlist(...)
- build_triplets_from_csr(...)

Notes
-----
- Indices are stored as int32 for compactness; cast to torch.long at use-sites
  that require indexing tensors.
- Displacement vectors (r_ij) and distances (d_ij) use the descriptor dtype.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import torch


class NeighborGraph(TypedDict):
    center_ptr: torch.Tensor  # int32, shape [N+1]
    nbr_idx: torch.Tensor     # int32, shape [E]
    r_ij: torch.Tensor        # float(dtype), shape [E,3]
    d_ij: torch.Tensor        # float(dtype), shape [E]


class TripletIndex(TypedDict):
    tri_i: torch.Tensor         # int32, shape [T]  (center indices)
    tri_j: torch.Tensor         # int32, shape [T]  (global neighbor j)
    tri_k: torch.Tensor         # int32, shape [T]  (global neighbor k)
    tri_j_local: torch.Tensor   # int32, shape [T]  (local index in CSR row)
    tri_k_local: torch.Tensor   # int32, shape [T]  (local index in CSR row)


def _compute_r_ij(
    positions: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    cell: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute displacement vectors r_ij with optional periodic offsets.
    """
    pos = positions.to(dtype)
    if cell is not None and offsets is not None:
        r_ij = (pos[j_idx] + offsets.to(dtype) @ cell.to(dtype)) - pos[i_idx]
    else:
        r_ij = pos[j_idx] - pos[i_idx]
    return r_ij


def build_csr_from_neighborlist(
    positions: torch.Tensor,
    cell: Optional[torch.Tensor],
    pbc: Optional[torch.Tensor],
    nbl,  # TorchNeighborList
    min_cutoff: float,
    max_cutoff: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> NeighborGraph:
    """
    Build CSR neighbor graph from TorchNeighborList results.

    Parameters
    ----------
    positions : (N,3) tensor
    cell : (3,3) tensor or None
    pbc : (3,) bool tensor or None
    nbl : TorchNeighborList
        Must be configured with cutoff >= max_cutoff
    min_cutoff : float
        Exclude neighbors with distance <= min_cutoff
    max_cutoff : float
        Include neighbors with distance <= max_cutoff (radial/ang cutoff)
    device : torch.device, optional
        Target device; defaults to positions.device
    dtype : torch.dtype, optional
        Floating dtype for r_ij/d_ij; defaults to positions.dtype

    Returns
    -------
    NeighborGraph
    """
    if device is None:
        device = positions.device
    if dtype is None:
        dtype = positions.dtype

    pos_dev = positions.to(device)
    cell_dev = cell.to(device) if cell is not None else None
    pbc_dev = pbc.to(device) if pbc is not None else None

    # Neighbor search using the (pre)configured maximum cutoff in nbl
    nb = nbl.get_neighbors(pos_dev.to(dtype), cell_dev,
                           pbc_dev, fractional=False)
    edge_index = nb["edge_index"]  # (2,E_raw)
    distances = nb["distances"]    # (E_raw,)
    offsets = nb["offsets"]        # (E_raw,3) or None

    if edge_index.numel() == 0:
        N = int(positions.shape[0])
        return {
            "center_ptr": torch.zeros(N + 1, dtype=torch.int32, device=device),
            "nbr_idx": torch.empty(0, dtype=torch.int32, device=device),
            "r_ij": torch.empty(0, 3, dtype=dtype, device=device),
            "d_ij": torch.empty(0, dtype=dtype, device=device),
        }

    # Distance filter: (min_cutoff, max_cutoff]
    mask = (distances > float(min_cutoff)) & (distances <= float(max_cutoff))
    if mask.sum().item() == 0:
        N = int(positions.shape[0])
        return {
            "center_ptr": torch.zeros(N + 1, dtype=torch.int32, device=device),
            "nbr_idx": torch.empty(0, dtype=torch.int32, device=device),
            "r_ij": torch.empty(0, 3, dtype=dtype, device=device),
            "d_ij": torch.empty(0, dtype=dtype, device=device),
        }

    i_idx = edge_index[0, mask].to(torch.long)
    j_idx = edge_index[1, mask].to(torch.long)
    d_sel = distances[mask].to(dtype)

    # Compute displacement vectors (respecting PBC offsets when present)
    offs_sel = offsets[mask] if offsets is not None else None
    r_sel = _compute_r_ij(pos_dev, i_idx, j_idx, cell_dev, offs_sel, dtype)

    # Sort edges by center atom i for CSR contiguity (stable sort)
    perm = torch.argsort(i_idx, stable=True)
    i_sorted = i_idx[perm]
    j_sorted = j_idx[perm]
    d_sorted = d_sel[perm]
    r_sorted = r_sel[perm]

    # Build CSR pointers
    N = int(positions.shape[0])
    deg = torch.bincount(i_sorted, minlength=N).to(torch.int64)  # counts per i
    center_ptr = torch.zeros(N + 1, dtype=torch.int64, device=device)
    center_ptr[1:] = torch.cumsum(deg, dim=0)

    # Cast to compact dtypes where appropriate
    center_ptr_i32 = center_ptr.to(torch.int32)
    nbr_idx_i32 = j_sorted.to(torch.int32)

    return {
        "center_ptr": center_ptr_i32,
        "nbr_idx": nbr_idx_i32,
        "r_ij": r_sorted.to(dtype),
        "d_ij": d_sorted.to(dtype),
    }


def build_triplets_from_csr(
    csr: NeighborGraph,
    ang_cutoff: float,
    min_cutoff: float,
) -> TripletIndex:
    """
    Build flat triplet arrays (i,j,k) and local CSR indices per center row.

    Parameters
    ----------
    csr : NeighborGraph
        CSR neighbor graph with center_ptr, nbr_idx, r_ij, d_ij
    ang_cutoff : float
        Angular cutoff (include neighbors with d_ij <= ang_cutoff)
    min_cutoff : float
        Exclude neighbors with d_ij <= min_cutoff

    Returns
    -------
    TripletIndex
    """
    center_ptr = csr["center_ptr"].to(torch.int64)  # for indexing math
    nbr_idx = csr["nbr_idx"].to(torch.int64)
    d_ij = csr["d_ij"]

    N = int(center_ptr.numel() - 1)
    tri_i_list = []
    tri_j_list = []
    tri_k_list = []
    tri_j_local_list = []
    tri_k_local_list = []

    # Loop per center row to enumerate local combinations efficiently
    for i in range(N):
        start = int(center_ptr[i].item())
        end = int(center_ptr[i + 1].item())
        if end <= start:
            continue
        # Per-row distances and neighbor globals
        d_row = d_ij[start:end]
        j_row = nbr_idx[start:end]

        # Filter by angular cutoff
        mask = (d_row <= float(ang_cutoff)) & (d_row > float(min_cutoff))
        if not torch.any(mask):
            continue
        # Local indices that survive mask
        local_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if local_idx.numel() < 2:
            continue

        # All unique pairs j_local < k_local
        pairs = torch.combinations(local_idx, r=2)  # (T_row, 2)
        if pairs.numel() == 0:
            continue

        t = pairs.shape[0]
        tri_i_list.append(torch.full((t,), i, dtype=torch.int64))
        tri_j_local_list.append(pairs[:, 0].to(torch.int64))
        tri_k_local_list.append(pairs[:, 1].to(torch.int64))

        # Map to global neighbor indices
        tri_j_list.append(j_row[pairs[:, 0]].to(torch.int64))
        tri_k_list.append(j_row[pairs[:, 1]].to(torch.int64))

    if len(tri_i_list) == 0:
        # Return empty tensors
        empty_i32 = torch.empty(0, dtype=torch.int32)
        return {
            "tri_i": empty_i32,
            "tri_j": empty_i32,
            "tri_k": empty_i32,
            "tri_j_local": empty_i32,
            "tri_k_local": empty_i32,
        }

    tri_i = torch.cat(tri_i_list)
    tri_j = torch.cat(tri_j_list)
    tri_k = torch.cat(tri_k_list)
    tri_j_local = torch.cat(tri_j_local_list)
    tri_k_local = torch.cat(tri_k_local_list)

    # Cast to compact dtype (int32) for storage
    return {
        "tri_i": tri_i.to(torch.int32),
        "tri_j": tri_j.to(torch.int32),
        "tri_k": tri_k.to(torch.int32),
        "tri_j_local": tri_j_local.to(torch.int32),
        "tri_k_local": tri_k_local.to(torch.int32),
    }


def center_ids_of_edge(csr: NeighborGraph) -> torch.Tensor:
    """
    Derive the center index for each edge from CSR center_ptr.

    Returns
    -------
    center_of_edge : torch.Tensor
        int64 tensor of shape [E] giving the center i for each edge.
    """
    center_ptr = csr["center_ptr"].to(torch.int64)
    N = int(center_ptr.numel() - 1)
    deg = (center_ptr[1:] - center_ptr[:-1]).to(torch.int64)
    centers = torch.arange(N, dtype=torch.int64, device=center_ptr.device)
    return torch.repeat_interleave(centers, deg)
