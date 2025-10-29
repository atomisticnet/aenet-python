"""
Feature and energy normalization for PyTorch training.

Handles computation and application of normalization statistics.
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader


class NormalizationManager:
    """
    Manages feature and energy normalization for training.

    Handles:
    - Computing feature statistics (mean/std) from training data
    - Computing energy normalization (shift/scaling) from training data
    - Applying normalization during training and inference
    - Storing and retrieving normalization parameters

    Parameters
    ----------
    normalize_features : bool
        Whether to normalize features.
    normalize_energy : bool
        Whether to normalize energy targets.
    dtype : torch.dtype
        Data type for statistics.
    device : torch.device
        Device for statistics tensors.
    """

    def __init__(
        self,
        normalize_features: bool = True,
        normalize_energy: bool = True,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ):
        self._normalize_features = normalize_features
        self._normalize_energy = normalize_energy
        self.dtype = dtype
        self.device = device

        # Feature normalization statistics
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

        # Energy normalization statistics
        self.E_shift: float = 0.0
        self.E_scaling: float = 1.0

    @property
    def normalize_features(self) -> bool:
        """Whether feature normalization is enabled."""
        return self._normalize_features

    @property
    def normalize_energy(self) -> bool:
        """Whether energy normalization is enabled."""
        return self._normalize_energy

    def has_feature_stats(self) -> bool:
        """Check if feature statistics have been computed or set."""
        return self.feature_mean is not None and self.feature_std is not None

    def set_feature_stats(self, mean, std):
        """
        Set feature normalization statistics from provided values.

        Parameters
        ----------
        mean : array-like
            Feature means.
        std : array-like
            Feature standard deviations.
        """
        self.feature_mean = torch.as_tensor(
            mean, dtype=self.dtype, device=self.device
        ).view(-1)
        self.feature_std = torch.as_tensor(
            std, dtype=self.dtype, device=self.device
        ).view(-1)

    def set_energy_stats(self, shift: float, scaling: float):
        """
        Set energy normalization parameters.

        Parameters
        ----------
        shift : float
            Energy shift (per-atom midpoint).
        scaling : float
            Energy scaling factor.
        """
        self.E_shift = float(shift)
        self.E_scaling = float(scaling)

    def compute_feature_stats(
        self,
        dataloader: DataLoader,
        n_features: int,
        provided_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Compute or load feature normalization statistics.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for training data (no shuffle needed).
        n_features : int
            Number of feature dimensions.
        provided_stats : dict, optional
            Pre-computed statistics with 'mean' and 'std'/'cov' keys.
            If provided, these will be used instead of computing from data.
        """
        if not self._normalize_features:
            return

        # Use provided stats if available
        if provided_stats is not None:
            mean_np = provided_stats.get("mean", None)
            std_np = provided_stats.get("std", None)
            if std_np is None:
                std_np = provided_stats.get("cov", None)

            if mean_np is not None and std_np is not None:
                self.feature_mean = torch.as_tensor(
                    mean_np, dtype=self.dtype, device=self.device
                ).view(-1)
                self.feature_std = torch.as_tensor(
                    std_np, dtype=self.dtype, device=self.device
                ).view(-1)
                return

        # Compute from training data
        sum_f = torch.zeros(n_features, dtype=self.dtype, device="cpu")
        sumsq_f = torch.zeros(n_features, dtype=self.dtype, device="cpu")
        total_atoms = 0

        with torch.no_grad():
            for batch in dataloader:
                feats = batch["features"]
                feats = feats.to(dtype=self.dtype, device="cpu")
                sum_f += feats.sum(dim=0).cpu()
                sumsq_f += (feats * feats).sum(dim=0).cpu()
                total_atoms += int(feats.shape[0])

        if total_atoms > 0:
            mean = sum_f / float(total_atoms)
            var = torch.clamp(
                sumsq_f / float(total_atoms) - mean * mean, min=0.0
            )
            std = torch.sqrt(var + torch.as_tensor(1e-12, dtype=var.dtype))
            self.feature_mean = mean.to(device=self.device)
            self.feature_std = std.to(device=self.device)

    def compute_energy_stats(
        self,
        dataloader: DataLoader,
        E_atomic_by_index: Optional[torch.Tensor] = None,
        energy_target: str = "cohesive",
        provided_shift: Optional[float] = None,
        provided_scaling: Optional[float] = None,
    ):
        """
        Compute energy normalization statistics.

        Computes per-atom energy min/max/avg and derives shift/scaling
        to normalize to [-1, 1] range (matching aenet convention).

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for training data.
        E_atomic_by_index : torch.Tensor, optional
            Per-species atomic energies, indexed by species_indices.
            Used to convert total energies to cohesive.
        energy_target : str
            Either 'cohesive' or 'total'. If 'cohesive', atomic energies
            are subtracted.
        provided_shift : float, optional
            Override computed E_shift.
        provided_scaling : float, optional
            Override computed E_scaling.
        """
        if not self.normalize_energy:
            return

        # Use provided values if available
        if provided_shift is not None:
            self.E_shift = float(provided_shift)
        if provided_scaling is not None:
            self.E_scaling = float(provided_scaling)

        # If both provided, no need to compute
        if provided_shift is not None and provided_scaling is not None:
            return

        # Compute from data
        e_min = None
        e_max = None
        e_sum = 0.0
        n_struct = 0

        with torch.no_grad():
            for batch in dataloader:
                n_atoms_b = batch["n_atoms"].to(self.device)
                energy_ref_b = batch["energy_ref"].to(
                    self.device, dtype=self.dtype
                )
                species_indices_b = batch["species_indices"].to(self.device)

                # Convert to target space (cohesive if requested)
                energy_target_b = energy_ref_b
                if (
                    energy_target == "cohesive"
                    and E_atomic_by_index is not None
                ):
                    per_atom_Ea_b = E_atomic_by_index[species_indices_b]
                    batch_idx_b = torch.repeat_interleave(
                        torch.arange(len(n_atoms_b), device=self.device),
                        n_atoms_b.long(),
                    )
                    Ea_sum_b = torch.zeros(
                        len(n_atoms_b),
                        dtype=energy_ref_b.dtype,
                        device=self.device,
                    )
                    Ea_sum_b.scatter_add_(0, batch_idx_b, per_atom_Ea_b)
                    energy_target_b = energy_ref_b - Ea_sum_b

                # Per-atom energies
                e_pa = energy_target_b / n_atoms_b

                # Update stats
                batch_min = float(torch.min(e_pa).item())
                batch_max = float(torch.max(e_pa).item())
                e_min = batch_min if e_min is None else min(e_min, batch_min)
                e_max = batch_max if e_max is None else max(e_max, batch_max)
                e_sum += float(torch.sum(e_pa).item())
                n_struct += int(len(n_atoms_b))

        if e_min is not None and e_max is not None and e_max > e_min:
            # Normalize to [-1, 1] range
            if provided_scaling is None:
                self.E_scaling = float(2.0 / (e_max - e_min))
            if provided_shift is None:
                self.E_shift = float(0.5 * (e_max + e_min))
        else:
            # Degenerate case: disable energy normalization
            self.E_scaling = 1.0
            self.E_shift = 0.0

    def apply_feature_normalization(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply feature normalization.

        Parameters
        ----------
        features : torch.Tensor
            Raw features, shape (N, F).

        Returns
        -------
        torch.Tensor
            Normalized features if normalization is enabled,
            otherwise returns input unchanged.
        """
        if (
            not self.normalize_features
            or self.feature_mean is None
            or self.feature_std is None
        ):
            return features

        mean = self.feature_mean.to(
            device=features.device, dtype=features.dtype
        )
        std = torch.clamp(
            self.feature_std.to(device=features.device, dtype=features.dtype),
            min=1e-12,
        )
        return (features - mean) / std

    def apply_energy_normalization(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Apply energy normalization (model output space).

        Parameters
        ----------
        energy : torch.Tensor
            Raw per-atom energies summed over structure.

        Returns
        -------
        torch.Tensor
            Normalized energy.
        """
        if not self.normalize_energy:
            return energy
        return energy  # Already in normalized space during training

    def denormalize_energy(
        self, energy_norm: torch.Tensor, n_atoms: int
    ) -> float:
        """
        Convert normalized model output to physical energy.

        Parameters
        ----------
        energy_norm : torch.Tensor
            Normalized energy from model (sum of per-atom energies).
        n_atoms : int
            Number of atoms in structure.

        Returns
        -------
        float
            Physical energy.
        """
        if not self.normalize_energy:
            return float(energy_norm.detach().cpu())

        return float(
            (energy_norm / self.E_scaling + self.E_shift * n_atoms)
            .detach()
            .cpu()
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Get normalization state for serialization.

        Returns
        -------
        dict
            Dictionary containing normalization parameters.
        """
        state = {
            "normalize_features": self.normalize_features,
            "normalize_energy": self.normalize_energy,
            "E_shift": self.E_shift,
            "E_scaling": self.E_scaling,
        }

        if self.feature_mean is not None:
            state["feature_mean"] = self.feature_mean.cpu().numpy()
        if self.feature_std is not None:
            state["feature_std"] = self.feature_std.cpu().numpy()

        return state

    def set_state(self, state: Dict[str, Any]):
        """
        Restore normalization state from dictionary.

        Parameters
        ----------
        state : dict
            Dictionary containing normalization parameters.
        """
        self._normalize_features = state.get("normalize_features", True)
        self._normalize_energy = state.get("normalize_energy", True)
        self.E_shift = float(state.get("E_shift", 0.0))
        self.E_scaling = float(state.get("E_scaling", 1.0))

        if "feature_mean" in state:
            self.feature_mean = torch.as_tensor(
                state["feature_mean"], dtype=self.dtype, device=self.device
            )
        if "feature_std" in state:
            self.feature_std = torch.as_tensor(
                state["feature_std"], dtype=self.dtype, device=self.device
            )
