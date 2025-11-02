"""
Inference datasets and collate utilities for energy-only predictions.
"""

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from aenet.torch_featurize.graph import (
    build_csr_from_neighborlist,
    build_triplets_from_csr,
)

from ..config import Structure


class EnergyInferenceDataset(Dataset):
    """
    Dataset that featurizes structures for energy-only inference.

    Each item provides:
      - 'features': (N,F) feature tensor (CPU)
      - 'species_indices': (N,) long tensor (CPU)
      - 'n_atoms': int
      - 'species': list[str] (for cohesive conversion)
    """

    def __init__(self, structures: List[Structure], descriptor):
        self.structures = structures
        self.descriptor = descriptor

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        st = self.structures[idx]

        # Always compute features on CPU here to avoid GPU usage from workers
        positions = torch.from_numpy(st.positions)
        cell = (
            torch.from_numpy(st.cell)
            if getattr(st, "cell", None) is not None
            else None
        )
        pbc = (
            torch.from_numpy(st.pbc)
            if getattr(st, "pbc", None) is not None
            else None
        )

        # Energy-only path: no neighbor-info needed
        feats = self.descriptor.forward_from_positions(
            positions, st.species, cell, pbc
        )
        species_idx = torch.tensor(
            [self.descriptor.species_to_idx[s] for s in st.species],
            dtype=torch.long,
        )
        return {
            "features": feats.cpu(),
            "species_indices": species_idx.cpu(),
            "n_atoms": int(len(st.species)),
            "species": st.species,
        }


class ForceInferenceDataset(Dataset):
    """
    Dataset that prepares both energy-view features and a force-view
    neighbor graph (CSR + triplets) per structure for batched force
    inference.

    For each structure, returns:
      - 'features': (N,F) energy-view features
      - 'species_indices': (N,) long
      - 'n_atoms': int
      - 'positions': (N,3) tensor (descriptor dtype)
      - 'species': list[str]
      - 'graph': dict CSR (center_ptr, nbr_idx, r_ij, d_ij)
      - 'triplets': dict TripletIndex (optional)
    """

    def __init__(self, structures: List[Structure], descriptor):
        self.structures = structures
        self.descriptor = descriptor

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        st = self.structures[idx]
        # Tensors (CPU here; moved to device in predictor)
        positions = torch.from_numpy(st.positions).to(self.descriptor.dtype)
        cell = (
            torch.from_numpy(st.cell).to(self.descriptor.dtype)
            if getattr(st, "cell", None) is not None
            else None
        )
        pbc = (
            torch.from_numpy(st.pbc)
            if getattr(st, "pbc", None) is not None
            else None
        )

        # Energy-view features (no neighbor-info needed)
        feats = self.descriptor.forward_from_positions(
            positions, st.species, cell, pbc
        )
        species_idx = torch.tensor(
            [self.descriptor.species_to_idx[s] for s in st.species],
            dtype=torch.long,
        )

        # Build per-structure CSR + triplets for full force view
        max_cut = float(
            max(self.descriptor.rad_cutoff, self.descriptor.ang_cutoff)
        )
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

        return {
            "features": feats.cpu(),
            "species_indices": species_idx.cpu(),
            "n_atoms": int(len(st.species)),
            "positions": positions.cpu(),
            "species": st.species,
            "graph": csr,
            "triplets": trip,
        }


def force_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate list of force-inference samples into a batched energy + force view.

    Returns:
      - features: (N_total,F)
      - species_indices: (N_total,)
      - n_atoms: (B,)
      - positions_f: (N_total,3)
      - species_indices_f: (N_total,)
      - species_f: list[str] length N_total
      - graph_f: dict CSR for batched force view
      - triplets_f: dict TripletIndex for batched force view
    """
    # Energy view
    features_list: List[torch.Tensor] = []
    species_idx_list: List[torch.Tensor] = []
    n_atoms_list: List[int] = []

    # Force view parts
    positions_f_list: List[torch.Tensor] = []
    species_f_names: List[str] = []
    species_idx_f_list: List[torch.Tensor] = []

    deg_parts: List[torch.Tensor] = []
    nbr_idx_parts: List[torch.Tensor] = []
    r_ij_parts: List[torch.Tensor] = []
    d_ij_parts: List[torch.Tensor] = []
    tri_i_parts: List[torch.Tensor] = []
    tri_j_parts: List[torch.Tensor] = []
    tri_k_parts: List[torch.Tensor] = []
    tri_j_local_parts: List[torch.Tensor] = []
    tri_k_local_parts: List[torch.Tensor] = []

    base = 0
    for s in batch:
        N = int(s["n_atoms"])
        features_list.append(s["features"])
        species_idx_list.append(s["species_indices"])
        n_atoms_list.append(N)

        pos = s["positions"]
        positions_f_list.append(pos)
        species_idx_f_list.append(s["species_indices"])
        species_f_names.extend(s["species"])

        # CSR/Triplets parts (ensure tensors)
        g = s.get("graph", None)
        if g is not None:
            cp = torch.as_tensor(g["center_ptr"])
            deg_parts.append((cp[1:] - cp[:-1]).to(torch.int64))
            nbr_idx_parts.append(
                torch.as_tensor(g["nbr_idx"]).to(torch.int64) + base
            )
            r_ij_parts.append(torch.as_tensor(g["r_ij"]))
            d_ij_parts.append(torch.as_tensor(g["d_ij"]))
        t = s.get("triplets", None)
        if t is not None:
            tri_i_parts.append(
                torch.as_tensor(t["tri_i"]).to(torch.int64) + base)
            tri_j_parts.append(
                torch.as_tensor(t["tri_j"]).to(torch.int64) + base)
            tri_k_parts.append(
                torch.as_tensor(t["tri_k"]).to(torch.int64) + base)
            tri_j_local_parts.append(
                torch.as_tensor(t["tri_j_local"]).to(torch.int64))
            tri_k_local_parts.append(
                torch.as_tensor(t["tri_k_local"]).to(torch.int64))

        base += N

    # Energy-view tensors
    features = (torch.cat(features_list, dim=0)
                if features_list else torch.empty(0, 0))
    species_indices = (
        torch.cat(species_idx_list, dim=0)
        if species_idx_list
        else torch.empty(0, dtype=torch.long)
    )
    n_atoms = torch.tensor(n_atoms_list, dtype=torch.long)

    # Force view tensors
    positions_f = (torch.cat(positions_f_list, dim=0)
                   if positions_f_list else None)
    species_indices_f = (
        torch.cat(species_idx_f_list, dim=0) if species_idx_f_list else None
    )

    # Batched CSR/Triplets
    graph_f = None
    triplets_f = None
    if len(deg_parts) > 0:
        total_centers = (int(positions_f.shape[0])
                         if positions_f is not None else 0)
        deg_cat = (torch.cat(deg_parts) if len(deg_parts) > 0
                   else torch.empty(0, dtype=torch.int64))
        center_ptr = torch.zeros(total_centers + 1, dtype=torch.int64)
        if deg_cat.numel() == total_centers:
            center_ptr[1:] = torch.cumsum(deg_cat, dim=0)
        if len(nbr_idx_parts) > 0:
            nbr_idx_b = torch.cat(nbr_idx_parts).to(torch.int32)
            r_ij_b = torch.cat(r_ij_parts)
            d_ij_b = torch.cat(d_ij_parts)
            graph_f = {
                "center_ptr": center_ptr.to(torch.int32),
                "nbr_idx": nbr_idx_b,
                "r_ij": r_ij_b,
                "d_ij": d_ij_b,
            }
        if len(tri_i_parts) > 0:
            triplets_f = {
                "tri_i": torch.cat(tri_i_parts).to(torch.int32),
                "tri_j": torch.cat(tri_j_parts).to(torch.int32),
                "tri_k": torch.cat(tri_k_parts).to(torch.int32),
                "tri_j_local": torch.cat(tri_j_local_parts).to(torch.int32),
                "tri_k_local": torch.cat(tri_k_local_parts).to(torch.int32),
            }

    return {
        # Energy view
        "features": features,
        "species_indices": species_indices,
        "n_atoms": n_atoms,
        # Force view
        "positions_f": positions_f,
        "species_indices_f": species_indices_f,
        "species_f": species_f_names,
        "graph_f": graph_f,
        "triplets_f": triplets_f,
    }


def energy_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate list of energy-only samples into a single batch.

    Returns dict with:
      - features: (N_total,F) float tensor (CPU)
      - species_indices: (N_total,) long tensor (CPU)
      - n_atoms: (B,) long tensor (CPU)
      - species_lists: List[List[str]]
        per-structure species (for cohesive conv)
    """
    features_list: List[torch.Tensor] = []
    species_idx_list: List[torch.Tensor] = []
    n_atoms_list: List[int] = []
    species_lists: List[List[str]] = []

    for s in batch:
        features_list.append(s["features"])
        species_idx_list.append(s["species_indices"])
        n_atoms_list.append(int(s["n_atoms"]))
        species_lists.append(s["species"])

    features = (torch.cat(features_list, dim=0)
                if features_list else torch.empty(0, 0))
    species_indices = (
        torch.cat(species_idx_list, dim=0)
        if species_idx_list
        else torch.empty(0, dtype=torch.long)
    )
    n_atoms = torch.tensor(n_atoms_list, dtype=torch.long)
    return {
        "features": features,
        "species_indices": species_indices,
        "n_atoms": n_atoms,
        "species_lists": species_lists,
    }
