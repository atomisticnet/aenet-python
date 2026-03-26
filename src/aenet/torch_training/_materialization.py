"""Internal helpers shared by torch-training dataset materialization paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from aenet.torch_featurize.graph import (
    build_csr_from_neighborlist,
    build_triplets_from_csr,
)

from .config import Structure


@dataclass(frozen=True)
class _PreparedStructureTensors:
    """Tensor payload derived from one ``Structure`` for descriptor use."""

    positions: torch.Tensor
    cell: torch.Tensor | None
    pbc: torch.Tensor | None
    species_indices: torch.Tensor
    forces: torch.Tensor | None


def filter_structures(
    structures: list[Structure],
    *,
    max_energy: float | None,
    max_forces: float | None,
) -> list[Structure]:
    """Apply shared structure-level filtering for torch-training datasets."""
    filtered: list[Structure] = []
    for struct in structures:
        if max_energy is not None:
            energy_per_atom = struct.energy / struct.n_atoms
            if energy_per_atom > max_energy:
                continue

        if max_forces is not None and struct.has_forces():
            max_force = np.abs(struct.forces).max()
            if max_force > max_forces:
                continue

        filtered.append(struct)
    return filtered


def extract_runtime_caches(
    cache_state,
) -> tuple[dict | None, dict | None, dict | None]:
    """Return trainer-owned runtime caches if a cache owner is provided."""
    feature_cache = (
        getattr(cache_state, "feature_cache", None)
        if cache_state is not None
        else None
    )
    neighbor_cache = (
        getattr(cache_state, "neighbor_cache", None)
        if cache_state is not None
        else None
    )
    graph_cache = (
        getattr(cache_state, "graph_cache", None)
        if cache_state is not None
        else None
    )
    return feature_cache, neighbor_cache, graph_cache


def prepare_structure_tensors(
    struct: Structure,
    descriptor,
) -> _PreparedStructureTensors:
    """Convert one ``Structure`` into descriptor-aligned tensors."""
    device = getattr(descriptor, "device", None)

    tensor_kwargs = {"dtype": descriptor.dtype}
    if device is not None:
        tensor_kwargs["device"] = device

    positions = torch.from_numpy(struct.positions).to(**tensor_kwargs)
    cell = (
        torch.from_numpy(struct.cell).to(**tensor_kwargs)
        if struct.cell is not None
        else None
    )
    pbc = (
        torch.from_numpy(struct.pbc).to(device=device)
        if struct.pbc is not None and device is not None
        else (
            torch.from_numpy(struct.pbc)
            if struct.pbc is not None
            else None
        )
    )
    species_indices = torch.tensor(
        [descriptor.species_to_idx[s] for s in struct.species],
        dtype=torch.long,
    )
    forces = (
        torch.from_numpy(struct.forces).to(**tensor_kwargs)
        if struct.forces is not None
        else None
    )
    return _PreparedStructureTensors(
        positions=positions,
        cell=cell,
        pbc=pbc,
        species_indices=species_indices,
        forces=forces,
    )


def build_force_graph_triplets(
    descriptor,
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
) -> dict:
    """Build the graph/triplet payload used by the sparse force path."""
    max_cut = float(max(descriptor.rad_cutoff, descriptor.ang_cutoff))
    graph = build_csr_from_neighborlist(
        positions=positions,
        cell=cell,
        pbc=pbc,
        nbl=descriptor.nbl,
        min_cutoff=float(descriptor.min_cutoff),
        max_cutoff=max_cut,
        device=positions.device,
        dtype=descriptor.dtype,
    )
    triplets = build_triplets_from_csr(
        csr=graph,
        ang_cutoff=float(descriptor.ang_cutoff),
        min_cutoff=float(descriptor.min_cutoff),
    )
    return {"graph": graph, "triplets": triplets}


def forward_force_features_with_graph(
    descriptor,
    positions: torch.Tensor,
    species_indices: torch.Tensor,
    graph: dict,
    triplets: dict | None,
) -> torch.Tensor:
    """Compute force-training features from the graph-based descriptor path."""
    if not hasattr(descriptor, "forward_with_graph"):
        raise RuntimeError(
            "Force training now requires graph-based descriptor support via "
            "'forward_with_graph()'."
        )
    return descriptor.forward_with_graph(
        positions=positions,
        species_indices=species_indices.to(device=positions.device),
        graph=graph,
        triplets=triplets,
    )


def load_energy_view_features(
    idx: int,
    *,
    descriptor,
    positions: torch.Tensor,
    species: list[str],
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    feature_cache: dict | None,
    cache_features: bool,
    neighbor_cache: dict | None,
    cache_neighbors: bool,
    load_persisted_features: Callable[[int], torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """
    Materialize canonical raw energy-view features for one dataset entry.

    Runtime feature cache entries take precedence. Persisted raw features, when
    available, are used before on-the-fly recomputation.
    """
    if cache_features and feature_cache is not None and idx in feature_cache:
        return feature_cache[idx]

    if load_persisted_features is not None:
        features = load_persisted_features(idx)
        if features is not None:
            if cache_features and feature_cache is not None:
                feature_cache[idx] = features
            return features

    if cache_neighbors and neighbor_cache is not None:
        neighbor_info_cached = neighbor_cache.get(idx)
        if neighbor_info_cached is not None:
            nb_idx_list_t = [
                torch.as_tensor(arr, dtype=torch.long).to(positions.device)
                for arr in neighbor_info_cached["neighbor_lists"]
            ]
            nb_vec_list_t = [
                torch.as_tensor(vec, dtype=descriptor.dtype).to(
                    positions.device
                )
                for vec in neighbor_info_cached["neighbor_vectors"]
            ]
            features = descriptor.forward(
                positions,
                species,
                nb_idx_list_t,
                nb_vec_list_t,
            )
        else:
            features, neighbor_info_cached = (
                descriptor.featurize_with_neighbor_info(
                    positions,
                    species,
                    cell,
                    pbc,
                )
            )
            neighbor_cache[idx] = neighbor_info_cached
    else:
        features = descriptor.forward_from_positions(
            positions,
            species,
            cell,
            pbc,
        )

    if cache_features and feature_cache is not None:
        feature_cache[idx] = features
    return features


def materialize_force_view(
    idx: int,
    *,
    descriptor,
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    species_indices: torch.Tensor,
    graph_cache: dict | None,
    cache_force_triplets: bool,
    load_persisted_features: Callable[[int], torch.Tensor | None] | None = None,
    load_local_derivatives: Callable[[int], dict | None] | None = None,
) -> tuple[torch.Tensor, dict | None, dict | None, dict | None]:
    """
    Materialize canonical force-view payloads for one dataset entry.

    Persisted raw features and persisted local derivatives are loaded first
    when callbacks are provided. Graph/triplet payloads are only built when
    needed for missing force-view inputs.
    """
    features = (
        load_persisted_features(idx)
        if load_persisted_features is not None
        else None
    )
    local_derivatives = (
        load_local_derivatives(idx)
        if load_local_derivatives is not None
        else None
    )

    graph = None
    triplets = None
    if features is None or local_derivatives is None:
        graph_trip = (
            graph_cache.get(idx)
            if cache_force_triplets and graph_cache is not None
            else None
        )
        if graph_trip is None:
            graph_trip = build_force_graph_triplets(
                descriptor=descriptor,
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
            if cache_force_triplets and graph_cache is not None:
                graph_cache[idx] = graph_trip
        graph = graph_trip["graph"]
        triplets = graph_trip["triplets"]
        if features is None:
            features = forward_force_features_with_graph(
                descriptor=descriptor,
                positions=positions,
                species_indices=species_indices,
                graph=graph,
                triplets=triplets,
            )

    if features is None:
        raise RuntimeError(
            "Force-view materialization failed to produce raw features."
        )
    return features, graph, triplets, local_derivatives


def build_sample_dict(
    *,
    struct: Structure,
    idx: int,
    prepared: _PreparedStructureTensors,
    features: torch.Tensor,
    use_forces: bool,
    graph: dict | None,
    triplets: dict | None,
    local_derivatives: dict | None,
    fallback_name_prefix: str,
) -> dict:
    """Build the common trainer-facing sample dictionary."""
    return {
        "features": features,
        "neighbor_info": None,
        "graph": graph,
        "triplets": triplets,
        "local_derivatives": local_derivatives,
        "positions": prepared.positions,
        "species": struct.species,
        "species_indices": prepared.species_indices,
        "cell": prepared.cell,
        "pbc": prepared.pbc,
        "energy": float(struct.energy),
        "forces": prepared.forces if use_forces else None,
        "has_forces": struct.has_forces(),
        "use_forces": use_forces,
        "n_atoms": int(struct.n_atoms),
        "name": (
            struct.name
            if struct.name is not None
            else f"{fallback_name_prefix}{idx}"
        ),
    }
