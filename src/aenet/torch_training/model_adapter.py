"""
Model adapter to present a per-atom energy interface over NetAtom.

aenet-PyTorch's NetAtom groups descriptors by species and typically returns
per-structure energies. For our training losses we need a per-atom energy
API: forward_atomic(features, species_indices) -> (N,).

This adapter wraps a NetAtom instance and exposes a simple callable that
accepts a flat (N, F) feature tensor and a (N,) species index tensor and
returns per-atom energies in the original atom order.
"""

import torch
import torch.nn as nn


class EnergyModelAdapter(nn.Module):
    """
    Adapter over NetAtom to compute per-atom energies given flat features.

    Parameters
    ----------
    net : nn.Module
        aenet-PyTorch NetAtom instance
    n_species : int
        Number of species in the system
    """

    def __init__(self, net: nn.Module, n_species: int):
        super().__init__()
        self.net = net
        self.n_species = n_species
        # Expose for convenience
        self.device = getattr(net, "device", "cpu")

    def forward_atomic(
        self, features: torch.Tensor, species_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-atom energies.

        Parameters
        ----------
        features : torch.Tensor
            (N, F) feature tensor (dtype should match network precision)
        species_indices : torch.Tensor
            (N,) long tensor with species index for each atom, consistent
            with the NetAtom species ordering

        Returns
        -------
        E_atomic : torch.Tensor
            (N,) per-atom energies in the original atom order
        """
        assert features.dim() == 2, "features must be (N, F)"
        assert species_indices.dim() == 1, "species_indices must be (N,)"

        N = features.shape[0]
        device = features.device
        dtype = features.dtype

        # Prepare output
        E_atomic_full = torch.zeros(N, 1, device=device, dtype=dtype)

        # For each species, select its atoms and apply the species-specific MLP
        # NetAtom.functions[iesp] is an nn.Sequential mapping (F_in) -> (1)
        for iesp in range(self.n_species):
            mask = (species_indices == iesp)
            if not torch.any(mask):
                continue

            x = features[mask]  # (n_i, F)
            # Sanity check on input size (optional)
            # expected_in = self.net.functions[iesp][0].in_features
            # if x.shape[1] != expected_in:
            #     raise RuntimeError(
            #         f"Feature size {x.shape[1]} != expected {expected_in} "
            #         f"for species {iesp}"
            #     )

            e_i = self.net.functions[iesp](x)  # (n_i, 1)
            E_atomic_full[mask] = e_i

        return E_atomic_full.squeeze(1)

    def forward(
        self, features: torch.Tensor, species_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Alias to forward_atomic for convenience.

        Allows adapter(features, idx).
        """
        return self.forward_atomic(features, species_indices)
