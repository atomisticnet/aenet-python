"""
Loss functions for energy and force training.

This module implements loss computation for MLIP training, including
energy RMSE and force RMSE using semi-analytical gradients. It is designed
to work with an energy-model adapter that provides per-atom energies from
flat features and per-atom species indices.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "compute_energy_loss",
    "compute_force_loss",
    "compute_combined_loss",
]


def compute_energy_loss(
    features: torch.Tensor,
    energy_ref: torch.Tensor,
    n_atoms: torch.Tensor,
    network: nn.Module,
    species_indices: torch.Tensor,
    E_shift: float = 0.0,
    E_scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute energy loss (RMSE) for a batch of structures.

    Parameters
    ----------
    features : torch.Tensor
        (N_total, F) features for all atoms in batch
    energy_ref : torch.Tensor
        (B,) reference energies for each structure (unnormalized)
    n_atoms : torch.Tensor
        (B,) number of atoms in each structure
    network : nn.Module
        Neural network (e.g., EnergyModelAdapter over NetAtom) that supports
        network(features, species_indices) -> (N_total,) per-atom energies
    species_indices : torch.Tensor
        (N_total,) species index for each atom
    E_shift : float, optional
        Energy shift used during normalization
        (E_norm = (E - E_shift)/E_scaling).
        Default: 0.0
    E_scaling : float, optional
        Energy scaling used during normalization. Default: 1.0

    Returns
    -------
    loss : torch.Tensor
        Energy RMSE loss (scalar)
    energy_pred_denorm : torch.Tensor
        (B,) predicted total energies (denormalized)
    """
    # Predict per-atom energies (normalized space)
    E_atomic = network(features, species_indices)  # (N_total,)

    # Sum atomic energies to get per-structure energies (normalized)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(n_atoms), device=features.device), n_atoms.long()
    )
    energy_pred_norm = torch.zeros(
        len(n_atoms), dtype=features.dtype, device=features.device
    )
    energy_pred_norm.scatter_add_(0, batch_idx, E_atomic.squeeze())

    # Denormalize: E_total = E_pred/E_scaling + E_shift*N_atoms
    energy_pred_denorm = energy_pred_norm / E_scaling + E_shift * n_atoms

    # RMSE per atom (as used in aenet-PyTorch)
    diff_per_atom = (energy_pred_denorm - energy_ref) / n_atoms
    loss = torch.sqrt(torch.mean(diff_per_atom**2))

    return loss, energy_pred_denorm


def _contract_forces(
    grad_E_wrt_features: torch.Tensor,
    grad_features: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Contract forces F = - dE/dr = - sum_{i,f} (dE/dG_if) * (dG_if/dr).

    Parameters
    ----------
    grad_E_wrt_features : torch.Tensor
        (N, F) gradients of total energy wrt features
    grad_features : torch.Tensor
        (N, F, N, 3) gradients of features wrt positions
    chunk_size : int, optional
        If provided, compute contraction in chunks along i-dimension to reduce
        peak memory usage.

    Returns
    -------
    forces_pred : torch.Tensor
        (N, 3) predicted forces (in normalized energy units)
    """
    N = grad_E_wrt_features.shape[0]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= N:
        # Single contraction
        return -torch.einsum("if,ifjk->jk", grad_E_wrt_features, grad_features)

    # Chunked contraction
    forces_pred = torch.zeros(
        N, 3, dtype=grad_features.dtype, device=grad_features.device
    )
    for i0 in range(0, N, chunk_size):
        i1 = min(N, i0 + chunk_size)
        forces_pred += -torch.einsum(
            "if,ifjk->jk",
            grad_E_wrt_features[i0:i1],
            grad_features[i0:i1],
        )
    return forces_pred


def _feature_derivative_scales(
    feature_std: Optional[torch.Tensor],
    *,
    descriptor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """
    Build per-block derivative scaling factors for normalized features.

    If features are normalized as ``G_norm = (G - mean) / std``, the
    descriptor derivatives must be scaled by ``1/std`` before contraction.
    The returned tensors are shaped for direct broadcasting over local
    derivative blocks.
    """
    n_rad = descriptor.rad_order + 1
    n_ang = descriptor.ang_order + 1

    one_rad = torch.ones(1, n_rad, 1, dtype=dtype, device=device)
    one_ang = torch.ones(1, n_ang, 1, dtype=dtype, device=device)

    if feature_std is None:
        return {
            "radial": one_rad,
            "angular": one_ang,
            "radial_weighted": one_rad,
            "angular_weighted": one_ang,
        }

    eps = 1e-12
    fs = torch.clamp(
        feature_std.to(device=device, dtype=dtype),
        min=float(eps),
    )

    if descriptor.multi:
        return {
            "radial": (1.0 / fs[:n_rad]).view(1, n_rad, 1),
            "angular": (1.0 / fs[n_rad:n_rad + n_ang]).view(1, n_ang, 1),
            "radial_weighted": (
                1.0 / fs[n_rad + n_ang:2 * n_rad + n_ang]
            ).view(1, n_rad, 1),
            "angular_weighted": (
                1.0 / fs[2 * n_rad + n_ang:]
            ).view(1, n_ang, 1),
        }

    return {
        "radial": (1.0 / fs[:n_rad]).view(1, n_rad, 1),
        "angular": (1.0 / fs[n_rad:]).view(1, n_ang, 1),
        "radial_weighted": one_rad,
        "angular_weighted": one_ang,
    }


def _contract_forces_sparse(
    grad_E_wrt_features: torch.Tensor,
    local_derivatives: dict[str, dict[str, Optional[torch.Tensor]]],
    *,
    descriptor,
    feature_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Contract forces from sparse local derivative blocks.

    Parameters
    ----------
    grad_E_wrt_features : torch.Tensor
        ``(N, F)`` gradients of the total energy with respect to features.
    local_derivatives : dict
        Sparse local derivative representation produced by the descriptor.
    descriptor : ChebyshevDescriptor
        Descriptor instance providing feature layout metadata.
    feature_std : torch.Tensor, optional
        Feature standard deviations used for normalization. When provided,
        local derivative blocks are scaled by ``1/std`` to match the
        normalized feature space.

    Returns
    -------
    torch.Tensor
        Predicted forces with shape ``(N, 3)`` in normalized energy units.
    """
    device = grad_E_wrt_features.device
    dtype = grad_E_wrt_features.dtype
    n_atoms = grad_E_wrt_features.shape[0]
    n_rad = descriptor.rad_order + 1
    n_ang = descriptor.ang_order + 1
    scales = _feature_derivative_scales(
        feature_std,
        descriptor=descriptor,
        device=device,
        dtype=dtype,
    )

    forces_pred = torch.zeros(n_atoms, 3, dtype=dtype, device=device)

    radial = local_derivatives["radial"]
    center_idx = radial["center_idx"]
    if center_idx.numel() > 0:
        neighbor_idx = radial["neighbor_idx"]
        dG_drij_raw = radial["dG_drij"].to(device=device, dtype=dtype)
        dG_drij = dG_drij_raw * scales["radial"]

        coeff_rad = grad_E_wrt_features.index_select(0, center_idx)[:, :n_rad]
        contrib = torch.einsum("ef,efk->ek", coeff_rad, dG_drij)
        forces_pred.index_add_(0, center_idx, contrib)
        forces_pred.index_add_(0, neighbor_idx, -contrib)

        if descriptor.multi:
            coeff_rad_w = grad_E_wrt_features.index_select(
                0, center_idx
            )[:, n_rad + n_ang:2 * n_rad + n_ang]
            tspin_j = radial["neighbor_typespin"].to(
                device=device, dtype=dtype
            ).view(-1, 1, 1)
            dG_drij_w = dG_drij_raw * tspin_j * scales["radial_weighted"]
            contrib_w = torch.einsum("ef,efk->ek", coeff_rad_w, dG_drij_w)
            forces_pred.index_add_(0, center_idx, contrib_w)
            forces_pred.index_add_(0, neighbor_idx, -contrib_w)

    angular = local_derivatives["angular"]
    tri_i = angular["center_idx"]
    if tri_i.numel() > 0 and n_ang > 0:
        tri_j = angular["neighbor_j_idx"]
        tri_k = angular["neighbor_k_idx"]
        grads_i = angular["grads_i"].to(device=device, dtype=dtype)
        grads_j = angular["grads_j"].to(device=device, dtype=dtype)
        grads_k = angular["grads_k"].to(device=device, dtype=dtype)

        coeff_ang = grad_E_wrt_features.index_select(0, tri_i)[
            :, n_rad:n_rad + n_ang
        ]
        grads_i_unw = grads_i * scales["angular"]
        grads_j_unw = grads_j * scales["angular"]
        grads_k_unw = grads_k * scales["angular"]
        contrib_i = -torch.einsum("tf,tfk->tk", coeff_ang, grads_i_unw)
        contrib_j = -torch.einsum("tf,tfk->tk", coeff_ang, grads_j_unw)
        contrib_k = -torch.einsum("tf,tfk->tk", coeff_ang, grads_k_unw)
        forces_pred.index_add_(0, tri_i, contrib_i)
        forces_pred.index_add_(0, tri_j, contrib_j)
        forces_pred.index_add_(0, tri_k, contrib_k)

        if descriptor.multi:
            coeff_ang_w = grad_E_wrt_features.index_select(0, tri_i)[
                :, 2 * n_rad + n_ang:
            ]
            tsp = angular["triplet_typespin"].to(
                device=device, dtype=dtype
            ).view(-1, 1, 1)
            grads_i_w = grads_i * tsp * scales["angular_weighted"]
            grads_j_w = grads_j * tsp * scales["angular_weighted"]
            grads_k_w = grads_k * tsp * scales["angular_weighted"]
            contrib_i_w = -torch.einsum("tf,tfk->tk", coeff_ang_w, grads_i_w)
            contrib_j_w = -torch.einsum("tf,tfk->tk", coeff_ang_w, grads_j_w)
            contrib_k_w = -torch.einsum("tf,tfk->tk", coeff_ang_w, grads_k_w)
            forces_pred.index_add_(0, tri_i, contrib_i_w)
            forces_pred.index_add_(0, tri_j, contrib_j_w)
            forces_pred.index_add_(0, tri_k, contrib_k_w)

    return forces_pred


def _local_derivatives_to_device(
    local_derivatives: dict[str, dict[str, Optional[torch.Tensor]]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, Optional[torch.Tensor]]]:
    """Move a sparse local-derivative payload onto the requested device."""
    index_keys = {
        "center_idx",
        "neighbor_idx",
        "neighbor_j_idx",
        "neighbor_k_idx",
    }
    moved: dict[str, dict[str, Optional[torch.Tensor]]] = {}
    for block_name, block in local_derivatives.items():
        moved_block: dict[str, Optional[torch.Tensor]] = {}
        for key, value in block.items():
            if value is None:
                moved_block[key] = None
            elif key in index_keys:
                moved_block[key] = value.to(device=device, dtype=torch.int64)
            else:
                moved_block[key] = value.to(device=device, dtype=dtype)
        moved[block_name] = moved_block
    return moved


def compute_force_loss(
    positions: torch.Tensor,
    species: list,
    forces_ref: torch.Tensor,
    descriptor,  # ChebyshevDescriptor
    network: nn.Module,
    species_indices: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
    E_scaling: float = 1.0,
    neighbor_info: Optional[dict] = None,
    chunk_size: Optional[int] = None,
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    local_derivatives: Optional[
        dict[str, dict[str, Optional[torch.Tensor]]]
    ] = None,
    graph: Optional[dict] = None,
    triplets: Optional[dict] = None,
    center_indices: Optional[torch.Tensor] = None,
    use_dense_path: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute force loss (RMSE) using semi-analytical gradients.

    This function uses the efficient gradient computation from torch_featurize
    instead of autograd through the featurization process.

    Parameters
    ----------
    positions : torch.Tensor
        (N, 3) atomic positions
    species : list
        N species names
    forces_ref : torch.Tensor
        (N, 3) reference forces
    descriptor : ChebyshevDescriptor
        Descriptor for computing features and gradients
    network : nn.Module
        Neural network (adapter) that supports per-atom energy prediction
    species_indices : torch.Tensor
        (N,) species indices
    cell : torch.Tensor, optional
        (3, 3) cell vectors. Default: None
    pbc : torch.Tensor, optional
        (3,) periodic boundary conditions. Default: None
    E_scaling : float, optional
        Energy scaling for normalization. Default: 1.0
    neighbor_info : dict, optional
        Precomputed neighbor information with keys:
          - 'neighbor_lists': list of length N with (nnb_i,) arrays
          - 'neighbor_vectors': list of length N with (nnb_i, 3) arrays
        If provided, avoids recomputing neighbors on legacy non-graph paths.
    features : torch.Tensor, optional
        Precomputed force-view features with shape ``(N, F)``. When provided,
        the network forward pass reuses these features instead of recomputing
        them inside ``compute_force_loss()``.
    local_derivatives : dict, optional
        Precomputed sparse local derivative payload aligned with ``features``.
        When provided, this payload is preferred over on-the-fly sparse
        derivative recomputation.
    graph : dict, optional
        Batched CSR neighbor graph for the current force view with keys:
          - 'center_ptr': (Nf+1,) int32 CSR row pointers
          - 'nbr_idx': (E,) int32 neighbor indices (offset to force view)
          - 'r_ij': (E, 3) dtype displacement vectors
          - 'd_ij': (E,) dtype distances
        If provided, enables the graph-based force-training path.
    triplets : dict, optional
        Batched TripletIndex with keys:
          - 'tri_i','tri_j','tri_k': (T,) int32
          - 'tri_j_local','tri_k_local': (T,) int32 local CSR indices
        Optional; if omitted, only radial paths are used.
    center_indices : torch.Tensor, optional
        (M,) indices of centers to include (filtering) in the batched graph
        coordinate space. If None, all centers in the force view are used.
    chunk_size : int, optional
        If set, contract forces in chunks along i-dimension to reduce
        peak memory usage.
    use_dense_path : bool, optional
        Force the legacy dense ``(N, F, N, 3)`` contraction path even when
        sparse local derivative blocks are available. Intended only for
        regression testing and validation.

    Returns
    -------
    loss : torch.Tensor
        Force RMSE loss
    forces_pred : torch.Tensor
        (N, 3) predicted forces (denormalized)
    """
    positions = positions.clone().detach()
    grad_features = None
    sparse_local_derivatives = None
    if local_derivatives is not None:
        sparse_local_derivatives = _local_derivatives_to_device(
            local_derivatives,
            device=positions.device,
            dtype=positions.dtype,
        )

    # Compute features and derivative information.
    if features is not None:
        features = features.to(device=positions.device, dtype=positions.dtype)
    elif graph is not None:
        graph_dev = {
            k: (v.to(positions.device) if isinstance(v, torch.Tensor) else v)
            for k, v in graph.items()
        }
        triplets_dev = (
            {
                k: (v.to(positions.device)
                    if isinstance(v, torch.Tensor) else v)
                for k, v in triplets.items()
            }
            if triplets is not None else None
        )
        center_indices_dev = (
            center_indices.to(device=positions.device)
            if center_indices is not None else None
        )
        if use_dense_path:
            if not hasattr(descriptor, "compute_feature_gradients_with_graph"):
                raise RuntimeError(
                    "Dense graph reference path requested, but the "
                    "descriptor does not implement "
                    "'compute_feature_gradients_with_graph()'."
                )
            positions = positions.requires_grad_(True)
            (features, grad_features
             ) = descriptor.compute_feature_gradients_with_graph(
                positions=positions,
                species_indices=species_indices.to(device=positions.device),
                graph=graph_dev,
                triplets=triplets_dev,
                center_indices=center_indices_dev,
            )
        else:
            if sparse_local_derivatives is not None:
                if not hasattr(descriptor, "forward_with_graph"):
                    raise RuntimeError(
                        "Graph-based force training now requires graph-based "
                        "descriptor support via 'forward_with_graph()'."
                    )
                features = descriptor.forward_with_graph(
                    positions=positions,
                    species_indices=species_indices.to(device=positions.device),
                    graph=graph_dev,
                    triplets=triplets_dev,
                )
            else:
                if not hasattr(
                    descriptor,
                    "compute_features_and_local_derivatives_with_graph",
                ):
                    raise RuntimeError(
                        "Graph-based force training now requires sparse local "
                        "derivative support. Use 'use_dense_path=True' only "
                        "for regression or debugging."
                    )
                (features, sparse_local_derivatives
                 ) = descriptor.compute_features_and_local_derivatives_with_graph(
                    positions=positions,
                    species_indices=species_indices.to(
                        device=positions.device
                    ),
                    graph=graph_dev,
                    triplets=triplets_dev,
                    center_indices=center_indices_dev,
                )
    elif neighbor_info is not None and hasattr(
        descriptor, "compute_feature_gradients_from_neighbor_info"
    ):
        positions = positions.requires_grad_(True)
        # Convert neighbor_info lists to tensors on
        # same device/dtype as positions
        nb_idx_list_t: list[torch.Tensor] = []
        nb_vec_list_t: list[torch.Tensor] = []
        for nb_idx, nb_vec in zip(
            neighbor_info["neighbor_lists"], neighbor_info["neighbor_vectors"]
        ):
            nb_idx_list_t.append(
                torch.as_tensor(nb_idx, dtype=torch.long,
                                device=positions.device)
            )
            nb_vec_list_t.append(
                torch.as_tensor(nb_vec, dtype=positions.dtype,
                                device=positions.device)
            )
        (features, grad_features
         ) = descriptor.compute_feature_gradients_from_neighbor_info(
            positions=positions,
            species=species,
            neighbor_indices=nb_idx_list_t,
            neighbor_vectors=nb_vec_list_t,
        )
    else:
        positions = positions.requires_grad_(True)
        # Fallback: recompute neighbor information (slower)
        features, grad_features = descriptor.compute_feature_gradients(
            positions, species, cell, pbc
        )

    # Optional feature normalization: G_norm = (G - mean) / std
    if feature_mean is not None and feature_std is not None:
        eps = 1e-12
        fm = feature_mean.to(
            device=features.device, dtype=features.dtype
        )
        fs = feature_std.to(
            device=features.device, dtype=features.dtype
        )
        fs_safe = torch.clamp(fs, min=float(eps))
        features = (features - fm) / fs_safe
        # Adjust gradients: dG_norm/dx = (1/std) * dG/dx
        if grad_features is not None:
            fs_view = fs_safe.view(1, -1, 1, 1)
            grad_features = grad_features / fs_view

    features = features.requires_grad_(True)

    # Network forward pass: features -> per-atom energies
    E_atomic = network(features, species_indices)  # (N,)
    E_total = E_atomic.sum()

    # Compute dE_total/dG (N, F)
    grad_E_wrt_features = torch.autograd.grad(
        E_total,
        features,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Contract to forces: F = - dE/dr
    if sparse_local_derivatives is not None and not use_dense_path:
        forces_pred = _contract_forces_sparse(
            grad_E_wrt_features=grad_E_wrt_features,
            local_derivatives=sparse_local_derivatives,
            descriptor=descriptor,
            feature_std=feature_std,
        )
    else:
        if grad_features is None:
            raise RuntimeError(
                "Force loss requires either precomputed sparse local "
                "derivatives or dense feature gradients."
            )
        forces_pred = _contract_forces(
            grad_E_wrt_features=grad_E_wrt_features,
            grad_features=grad_features,
            chunk_size=chunk_size,
        )

    # Denormalize forces if energies were normalized by E_scaling
    forces_pred = forces_pred / E_scaling

    # RMSE over all components
    diff = forces_pred - forces_ref
    loss = torch.sqrt(torch.mean(diff**2))

    return loss, forces_pred


def compute_combined_loss(
    # Energy terms
    features: torch.Tensor,
    energy_ref: torch.Tensor,
    n_atoms: torch.Tensor,
    # Force terms (optional)
    positions: Optional[torch.Tensor],
    species: Optional[list],
    forces_ref: Optional[torch.Tensor],
    # Common terms
    network: nn.Module,
    species_indices: torch.Tensor,
    descriptor=None,  # ChebyshevDescriptor
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
    # Parameters
    alpha: float = 0.0,
    E_shift: float = 0.0,
    E_scaling: float = 1.0,
    use_forces: bool = False,
    # Optional acceleration / memory
    neighbor_info: Optional[dict] = None,
    chunk_size: Optional[int] = None,
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
    """
    Compute combined energy and force loss.

    L = (1 - alpha) * RMSE_energy + alpha * RMSE_force

    Parameters
    ----------
    features : torch.Tensor
        (N_total, F) features for all atoms
    energy_ref : torch.Tensor
        (B,) reference energies
    n_atoms : torch.Tensor
        (B,) number of atoms per structure
    positions : torch.Tensor, optional
        (N_total, 3) positions for force computation
    species : list, optional
        N_total species names for force computation
    forces_ref : torch.Tensor, optional
        (N_total, 3) reference forces
    network : nn.Module
        Neural network (adapter) providing per-atom energies
    species_indices : torch.Tensor
        (N_total,) species indices
    descriptor : ChebyshevDescriptor, optional
        Descriptor for force computation
    cell : torch.Tensor, optional
        (3, 3) cell vectors
    pbc : torch.Tensor, optional
        (3,) periodic boundary conditions
    alpha : float, optional
        Force weight (0 = energy only, 1 = force only). Default: 0.0
    E_shift : float, optional
        Energy shift for normalization. Default: 0.0
    E_scaling : float, optional
        Energy scaling for normalization. Default: 1.0
    use_forces : bool, optional
        Whether to compute force loss. Default: False
    neighbor_info : dict, optional
        Precomputed neighbor info to avoid recomputation
    chunk_size : int, optional
        Chunk size for force contraction

    Returns
    -------
    loss : torch.Tensor
        Combined loss
    metrics : dict
        {
          'loss': combined loss,
          'energy_loss': energy RMSE,
          'force_loss': force RMSE or None,
          'energy_pred': predicted energies (B,),
          'forces_pred': predicted forces (N,3) or None,
        }
    """
    # Energy term
    energy_loss, energy_pred = compute_energy_loss(
        features=features,
        energy_ref=energy_ref,
        n_atoms=n_atoms,
        network=network,
        species_indices=species_indices,
        E_shift=E_shift,
        E_scaling=E_scaling,
    )

    # Force term (optional)
    force_loss = None
    forces_pred = None
    if use_forces and alpha > 0.0:
        if positions is None or species is None or forces_ref is None:
            raise ValueError(
                "positions, species, and forces_ref must be provided when "
                "use_forces=True and alpha>0."
            )
        if descriptor is None:
            raise ValueError(
                "descriptor must be provided when computing force loss."
            )

        force_loss, forces_pred = compute_force_loss(
            positions=positions,
            species=species,
            forces_ref=forces_ref,
            descriptor=descriptor,
            network=network,
            species_indices=species_indices,
            cell=cell,
            pbc=pbc,
            E_scaling=E_scaling,
            neighbor_info=neighbor_info,
            chunk_size=chunk_size,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    # Combine
    if force_loss is None:
        combined = (1.0 - alpha) * energy_loss
    else:
        combined = (1.0 - alpha) * energy_loss + alpha * force_loss

    metrics = {
        "loss": combined,
        "energy_loss": energy_loss,
        "force_loss": force_loss,
        "energy_pred": energy_pred,
        "forces_pred": forces_pred,
    }
    return combined, metrics
