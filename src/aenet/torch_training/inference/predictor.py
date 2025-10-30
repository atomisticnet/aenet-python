"""
Prediction and inference for trained PyTorch models.

Handles energy and force prediction on new structures.
"""

from typing import Dict, List, Optional, Tuple

import torch

from ..config import Structure
from ..loss import compute_force_loss
from ..training.normalization import NormalizationManager


class Predictor:
    """
    Handles prediction for trained models.

    Parameters
    ----------
    model : nn.Module
        Trained model (EnergyModelAdapter).
    descriptor : ChebyshevDescriptor
        Descriptor for featurization.
    normalizer : NormalizationManager
        Normalization manager with trained statistics.
    energy_target : str
        Either 'cohesive' or 'total'. Determines how to interpret
        model outputs.
    E_atomic : dict, optional
        Atomic reference energies per species (for cohesive conversion).
    device : torch.device
        Device for computation.
    dtype : torch.dtype
        Data type for tensors.
    """

    def __init__(
        self,
        model,
        descriptor,
        normalizer: NormalizationManager,
        energy_target: str = "cohesive",
        E_atomic: Optional[Dict[str, float]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.model = model
        self.descriptor = descriptor
        self.normalizer = normalizer
        self.energy_target = energy_target
        self.E_atomic = E_atomic
        self.device = device
        self.dtype = dtype

        # Build E_atomic_by_index if available
        self.E_atomic_by_index: Optional[torch.Tensor] = None
        if E_atomic is not None:
            e_list = []
            for s in descriptor.species:
                e_list.append(float(E_atomic.get(s, 0.0)))
            self.E_atomic_by_index = torch.tensor(
                e_list, dtype=dtype, device=device
            )

        self._warn_missing_E_atomic_once: bool = False

    def predict(
        self, structures: List[Structure], predict_forces: bool = False
    ) -> Tuple[List[float], Optional[List[torch.Tensor]]]:
        """
        Predict energies (and optionally forces) for structures.

        Parameters
        ----------
        structures : list of Structure
            Structures to predict on.
        predict_forces : bool
            Whether to also predict forces.

        Returns
        -------
        energies : list of float
            Predicted energies (total energies if energy_target='cohesive',
            otherwise model outputs directly).
        forces : list of Tensor or None
            Predicted forces (N_i, 3) per structure if requested,
            otherwise None.
        """
        energies: List[float] = []
        forces_out: List[torch.Tensor] = []

        for st in structures:
            # Build tensors
            positions = torch.from_numpy(st.positions).to(self.device)
            if self.dtype == torch.float64:
                positions = positions.double()
            else:
                positions = positions.float()

            # Featurize and get neighbor info
            features, nb_info = (
                self.descriptor.featurize_with_neighbor_info(
                    positions, st.species, None, None
                )
            )
            species_indices = torch.tensor(
                [self.descriptor.species_to_idx[s] for s in st.species],
                dtype=torch.long,
                device=self.device,
            )
            if self.dtype == torch.float64:
                features = features.double()
            else:
                features = features.float()

            # Feature normalization
            features = self.normalizer.apply_feature_normalization(features)

            # Predict per-atom energies (normalized model output)
            E_atomic = self.model(features, species_indices)
            E_pred_norm = E_atomic.sum()

            # Denormalize model output
            E_pred = self.normalizer.denormalize_energy(
                E_pred_norm, len(st.species)
            )

            # Convert to total energy if training target was cohesive
            if self.energy_target == "cohesive":
                # Sum of atomic reference energies for the structure
                if self.E_atomic is not None:
                    E_atoms_sum = sum(
                        self.E_atomic.get(s, 0.0) for s in st.species
                    )
                elif self.E_atomic_by_index is not None:
                    E_atoms_sum = float(
                        self.E_atomic_by_index[species_indices]
                        .sum()
                        .detach()
                        .cpu()
                    )
                else:
                    if not self._warn_missing_E_atomic_once:
                        print(
                            "[WARN] energy_target='cohesive' but atomic "
                            "energies are unavailable; returning "
                            "cohesive energies."
                        )
                        self._warn_missing_E_atomic_once = True
                    E_atoms_sum = 0.0
                E_total = E_pred + E_atoms_sum
            else:
                # Model was trained on total energies
                E_total = E_pred

            energies.append(E_total)

            if predict_forces:
                # Use semi-analytical gradient path to predict forces
                neighbor_info = {
                    "neighbor_lists": nb_info["neighbor_lists"],
                    "neighbor_vectors": nb_info["neighbor_vectors"],
                }
                # Dummy zeros for forces_ref; we only want predictions
                forces_ref = torch.zeros_like(positions)
                # Enable gradients for force prediction (autograd for dE/dR)
                with torch.enable_grad():
                    _, forces_pred = compute_force_loss(
                        positions=positions.clone(),
                        species=st.species,
                        forces_ref=forces_ref,
                        descriptor=self.descriptor,
                        network=self.model,
                        species_indices=species_indices,
                        cell=None,
                        pbc=None,
                        E_scaling=float(self.normalizer.E_scaling),
                        neighbor_info=neighbor_info,
                        chunk_size=None,
                        feature_mean=(
                            self.normalizer.feature_mean
                            if self.normalizer.normalize_features
                            else None
                        ),
                        feature_std=(
                            self.normalizer.feature_std
                            if self.normalizer.normalize_features
                            else None
                        ),
                    )
                forces_out.append(forces_pred.detach().cpu())

        return energies, (forces_out if predict_forces else None)

    def cohesive_energy(
        self,
        structure: Structure,
        atomic_energies: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute cohesive energy from a structure with total energy.

        Parameters
        ----------
        structure : Structure
            Structure containing total energy and species list.
        atomic_energies : dict, optional
            Per-species atomic reference energies. If None, uses
            predictor's stored E_atomic.

        Returns
        -------
        float
            Cohesive energy (total - sum of atomic reference energies).

        Raises
        ------
        ValueError
            If no atomic energies are available.
        """
        if atomic_energies is None:
            atomic_energies = self.E_atomic
        if atomic_energies is None:
            raise ValueError(
                "Atomic energies not available. Provide atomic_energies or "
                "use a predictor with E_atomic set."
            )

        E_atoms = sum(
            float(atomic_energies.get(el, 0.0)) for el in structure.species
        )
        return float(structure.energy) - E_atoms
