"""
Loss functions for energy and force training.

This module implements loss computation for MLIP training, including
energy RMSE and force RMSE using semi-analytical gradients. It is designed
to work with an energy-model adapter that provides per-atom energies from
flat features and per-atom species indices.
"""

from typing import Dict, Optional, Tuple

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
    neighbor_info: Optional[Dict] = None,
    chunk_size: Optional[int] = None,
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
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
        If provided, avoids recomputing neighbors.
    chunk_size : int, optional
        If set, contract forces in chunks along i-dimension to reduce
        peak memory usage.

    Returns
    -------
    loss : torch.Tensor
        Force RMSE loss
    forces_pred : torch.Tensor
        (N, 3) predicted forces (denormalized)
    """
    # Ensure positions require grad if we fallback to autograd anywhere
    positions = positions.clone().detach().requires_grad_(True)

    # Compute features and their gradients w.r.t. positions
    if neighbor_info is not None and hasattr(
        descriptor, "compute_feature_gradients_from_neighbor_info"
    ):
        # Convert neighbor_info lists to tensors on same device/dtype as positions
        nb_idx_list_t: list[torch.Tensor] = []
        nb_vec_list_t: list[torch.Tensor] = []
        for nb_idx, nb_vec in zip(
            neighbor_info["neighbor_lists"], neighbor_info["neighbor_vectors"]
        ):
            nb_idx_list_t.append(
                torch.as_tensor(nb_idx, dtype=torch.long, device=positions.device)
            )
            nb_vec_list_t.append(
                torch.as_tensor(nb_vec, dtype=positions.dtype, device=positions.device)
            )

        features, grad_features = descriptor.compute_feature_gradients_from_neighbor_info(
            positions=positions,
            species=species,
            neighbor_indices=nb_idx_list_t,
            neighbor_vectors=nb_vec_list_t,
        )
    else:
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
    neighbor_info: Optional[Dict] = None,
    chunk_size: Optional[int] = None,
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
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
