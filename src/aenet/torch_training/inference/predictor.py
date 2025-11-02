"""
Prediction and inference for trained PyTorch models.

Handles energy and force prediction on new structures.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from .datasets import (
    EnergyInferenceDataset,
    energy_collate,
    ForceInferenceDataset,
    force_collate,
)

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
        self,
        structures: List[Structure],
        eval_forces: bool = False,
        return_atom_energies: bool = False,
        track_timing: bool = False,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: Optional[bool] = None,
    ) -> Tuple[
        List[float],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[Dict[str, List[float]]],
    ]:
        """
        Predict energies (and optionally forces) for structures.

        Parameters
        ----------
        structures : list of Structure
            Structures to predict on.
        eval_forces : bool
            Whether to also predict forces.
        return_atom_energies : bool
            Whether to return per-atom energies.
        track_timing : bool
            Whether to track and return timing information.

        Returns
        -------
        energies : list of float
            Predicted energies (total energies if energy_target='cohesive',
            otherwise model outputs directly).
        forces : list of Tensor or None
            Predicted forces (N_i, 3) per structure if requested,
            otherwise None.
        atom_energies : list of Tensor or None
            Per-atom energies (N_i,) per structure if requested,
            otherwise None.
        timing : dict or None
            Timing information per structure if requested, otherwise None.
            Keys: 'featurization', 'energy_eval', 'force_eval', 'total'
        """
        import time as time_module

        energies: List[float] = []
        forces_out: List[torch.Tensor] = []
        atom_energies_out: List[torch.Tensor] = []

        # Initialize timing dict if requested
        timing_dict: Optional[Dict[str, List[float]]] = None
        if track_timing:
            timing_dict = {
                'featurization': [],
                'energy_eval': [],
                'force_eval': [],
                'total': []
            }

        # DataLoader-powered energy-only path
        if (not eval_forces) and (batch_size is not None):
            # Defaults for dataloader knobs
            nw = int(num_workers) if num_workers is not None else 0
            pf = int(prefetch_factor) if prefetch_factor is not None else 2
            pw = (bool(persistent_workers)
                  if persistent_workers is not None else True)

            ds = EnergyInferenceDataset(structures, self.descriptor)
            dl_kwargs: Dict[str, Any] = dict(num_workers=nw)
            if nw > 0:
                dl_kwargs.update(prefetch_factor=pf, persistent_workers=pw)
            loader = DataLoader(
                ds,
                batch_size=int(batch_size),
                shuffle=False,
                collate_fn=energy_collate,
                **dl_kwargs,
            )

            import time as _time
            for batch in loader:
                t_b0 = _time.time() if track_timing else 0.0

                feats = batch["features"].to(
                    device=self.device, dtype=self.dtype)
                species_idx = batch["species_indices"].to(device=self.device)
                n_atoms_b = batch["n_atoms"].to(device=self.device)
                species_lists = batch["species_lists"]

                # Feature normalization (on device)
                feats = self.normalizer.apply_feature_normalization(feats)

                t_energy_start = _time.time() if track_timing else 0.0
                E_atomic_b = self.model(feats, species_idx)
                # Sum per-structure in normalized space
                batch_idx = torch.repeat_interleave(
                    torch.arange(len(n_atoms_b), device=self.device),
                    n_atoms_b.long(),
                )
                energy_pred_norm_b = torch.zeros(
                    len(n_atoms_b), dtype=feats.dtype, device=self.device
                )
                energy_pred_norm_b.scatter_add_(
                    0, batch_idx, E_atomic_b.squeeze())
                t_energy_end = _time.time() if track_timing else 0.0

                # Denormalize and convert to total energy if needed
                offsets = [0]
                for n in n_atoms_b.tolist():
                    offsets.append(offsets[-1] + int(n))
                for j in range(len(n_atoms_b)):
                    E_pred = self.normalizer.denormalize_energy(
                        energy_pred_norm_b[j], int(n_atoms_b[j].item())
                    )
                    if self.energy_target == "cohesive":
                        if self.E_atomic is not None:
                            E_atoms_sum = sum(
                                self.E_atomic.get(s, 0.0)
                                for s in species_lists[j]
                            )
                        elif self.E_atomic_by_index is not None:
                            sl = slice(offsets[j], offsets[j + 1])
                            E_atoms_sum = float(
                                self.E_atomic_by_index[species_idx[sl]]
                                .sum()
                                .detach()
                                .cpu()
                            )
                        else:
                            if not self._warn_missing_E_atomic_once:
                                print(
                                    "[WARN] energy_target='cohesive' but "
                                    "atomic energies are unavailable; "
                                    "returning cohesive energies."
                                )
                                self._warn_missing_E_atomic_once = True
                            E_atoms_sum = 0.0
                        E_total = E_pred + E_atoms_sum
                    else:
                        E_total = E_pred
                    energies.append(E_total)

                    # Per-atom energies if requested
                    if return_atom_energies:
                        sl = slice(offsets[j], offsets[j + 1])
                        E_atomic_denorm = (
                            E_atomic_b[sl] * self.normalizer.E_scaling)
                        atom_energies_out.append(
                            E_atomic_denorm.detach().cpu())

                if track_timing and timing_dict is not None:
                    t_b1 = _time.time()
                    per = (t_b1 - t_b0) / float(len(n_atoms_b))
                    for _ in range(len(n_atoms_b)):
                        timing_dict["featurization"].append(0.0)
                        timing_dict["energy_eval"].append(
                            t_energy_end - t_energy_start)
                        timing_dict["force_eval"].append(0.0)
                        timing_dict["total"].append(per)

            return (
                energies,
                None,
                atom_energies_out if return_atom_energies else None,
                timing_dict,
            )

        # DataLoader-powered batched forces path
        if eval_forces and (batch_size is not None):
            nw = int(num_workers) if num_workers is not None else 0
            pf = int(prefetch_factor) if prefetch_factor is not None else 2
            pw = (bool(persistent_workers)
                  if persistent_workers is not None else True)

            ds_f = ForceInferenceDataset(structures, self.descriptor)
            dl_kwargs_f: Dict[str, Any] = dict(num_workers=nw)
            if nw > 0:
                dl_kwargs_f.update(prefetch_factor=pf, persistent_workers=pw)
            loader_f = DataLoader(
                ds_f,
                batch_size=int(batch_size),
                shuffle=False,
                collate_fn=force_collate,
                **dl_kwargs_f,
            )

            import time as _time
            for batch in loader_f:
                t_b0 = _time.time() if track_timing else 0.0

                # Energy-view forward (batched over concatenated atoms)
                feats = batch["features"].to(
                    device=self.device, dtype=self.dtype)
                species_idx = batch["species_indices"].to(device=self.device)
                n_atoms_b = batch["n_atoms"].to(device=self.device)
                feats = self.normalizer.apply_feature_normalization(feats)

                t_energy_start = _time.time() if track_timing else 0.0
                E_atomic_b = self.model(feats, species_idx)
                batch_idx = torch.repeat_interleave(
                    torch.arange(len(n_atoms_b), device=self.device),
                    n_atoms_b.long(),
                )
                energy_pred_norm_b = torch.zeros(
                    len(n_atoms_b), dtype=feats.dtype, device=self.device
                )
                energy_pred_norm_b.scatter_add_(
                    0, batch_idx, E_atomic_b.squeeze())
                t_energy_end = _time.time() if track_timing else 0.0

                # Prepare offsets for per-structure slicing
                offsets = [0]
                for n in n_atoms_b.tolist():
                    offsets.append(offsets[-1] + int(n))

                # Cohesive conversion and per-atom energies if requested
                species_f = batch["species_f"]  # list[str] length N_total
                for j in range(len(n_atoms_b)):
                    E_pred = self.normalizer.denormalize_energy(
                        energy_pred_norm_b[j], int(n_atoms_b[j].item())
                    )
                    if self.energy_target == "cohesive":
                        if self.E_atomic is not None:
                            sl = slice(offsets[j], offsets[j + 1])
                            E_atoms_sum = sum(
                                self.E_atomic.get(s, 0.0)
                                for s in species_f[sl]
                            )
                        elif self.E_atomic_by_index is not None:
                            sl = slice(offsets[j], offsets[j + 1])
                            E_atoms_sum = float(
                                self.E_atomic_by_index[species_idx[sl]]
                                .sum()
                                .detach()
                                .cpu()
                            )
                        else:
                            if not self._warn_missing_E_atomic_once:
                                print(
                                    "[WARN] energy_target='cohesive' but "
                                    "atomic energies are unavailable; "
                                    "returning cohesive energies."
                                )
                                self._warn_missing_E_atomic_once = True
                            E_atoms_sum = 0.0
                        E_total = E_pred + E_atoms_sum
                    else:
                        E_total = E_pred
                    energies.append(E_total)

                    if return_atom_energies:
                        sl = slice(offsets[j], offsets[j + 1])
                        E_atomic_denorm = (
                            E_atomic_b[sl] * self.normalizer.E_scaling)
                        atom_energies_out.append(
                            E_atomic_denorm.detach().cpu())

                # Batched force prediction using CSR/Triplets graph
                positions_f = batch["positions_f"].to(
                    device=self.device, dtype=self.dtype
                )
                species_idx_f = batch["species_indices_f"].to(self.device)
                graph_f = batch["graph_f"]
                triplets_f = batch["triplets_f"]

                # Move graph/triplets to device
                graph_dev = None
                if graph_f is not None:
                    graph_dev = {
                        "center_ptr": graph_f["center_ptr"].to(self.device),
                        "nbr_idx": graph_f["nbr_idx"].to(self.device),
                        "r_ij": graph_f["r_ij"].to(
                            self.device, dtype=self.dtype),
                        "d_ij": graph_f["d_ij"].to(
                            self.device, dtype=self.dtype),
                    }
                triplets_dev = None
                if triplets_f is not None:
                    triplets_dev = {
                        "tri_i": triplets_f["tri_i"].to(self.device),
                        "tri_j": triplets_f["tri_j"].to(self.device),
                        "tri_k": triplets_f["tri_k"].to(self.device),
                        "tri_j_local": triplets_f["tri_j_local"].to(
                            self.device),
                        "tri_k_local": triplets_f["tri_k_local"].to(
                            self.device),
                    }

                t_force_start = _time.time() if track_timing else 0.0
                forces_ref = torch.zeros_like(positions_f)
                with torch.enable_grad():
                    _, forces_pred = compute_force_loss(
                        positions=positions_f.clone(),
                        species=species_f,
                        forces_ref=forces_ref,
                        descriptor=self.descriptor,
                        network=self.model,
                        species_indices=species_idx_f,
                        cell=None,
                        pbc=None,
                        E_scaling=float(self.normalizer.E_scaling),
                        neighbor_info=None,
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
                        graph=graph_dev,
                        triplets=triplets_dev,
                    )
                t_force_end = _time.time() if track_timing else 0.0

                # Split forces back per-structure using offsets
                for j in range(len(n_atoms_b)):
                    sl = slice(offsets[j], offsets[j + 1])
                    forces_out.append(forces_pred[sl].detach().cpu())

                if track_timing and timing_dict is not None:
                    t_b1 = _time.time()
                    per = (t_b1 - t_b0) / float(len(n_atoms_b))
                    for _ in range(len(n_atoms_b)):
                        timing_dict["featurization"].append(0.0)
                        timing_dict["energy_eval"].append(
                            t_energy_end - t_energy_start)
                        timing_dict["force_eval"].append(
                            t_force_end - t_force_start)
                        timing_dict["total"].append(per)

            return (
                energies,
                forces_out,
                atom_energies_out if return_atom_energies else None,
                timing_dict,
            )

        for st in structures:
            t_start = time_module.time() if track_timing else 0.0

            # Build tensors
            positions = torch.from_numpy(st.positions).to(self.device)
            if self.dtype == torch.float64:
                positions = positions.double()
            else:
                positions = positions.float()

            # Prepare PBC tensors (if available) and featurize
            cell_torch = None
            pbc_torch = None
            if getattr(st, "cell", None) is not None:
                cell_torch = torch.as_tensor(
                    st.cell, dtype=self.dtype, device=self.device
                )
            if getattr(st, "pbc", None) is not None:
                pbc_torch = torch.as_tensor(
                    st.pbc, dtype=torch.bool, device=self.device
                )

            t_feat_start = time_module.time() if track_timing else 0.0
            features, nb_info = self.descriptor.featurize_with_neighbor_info(
                positions, st.species, cell_torch, pbc_torch
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
            t_feat_end = time_module.time() if track_timing else 0.0

            # Predict per-atom energies (normalized model output)
            t_energy_start = time_module.time() if track_timing else 0.0
            E_atomic = self.model(features, species_indices)
            E_pred_norm = E_atomic.sum()

            # Denormalize model output
            E_pred = self.normalizer.denormalize_energy(
                E_pred_norm, len(st.species)
            )
            t_energy_end = time_module.time() if track_timing else 0.0

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

            # Store per-atom energies if requested
            if return_atom_energies:
                # Denormalize per-atom energies
                E_atomic_denorm = E_atomic * self.normalizer.E_scaling
                atom_energies_out.append(E_atomic_denorm.detach().cpu())

            t_force_start = 0.0
            t_force_end = 0.0
            if eval_forces:
                t_force_start = time_module.time() if track_timing else 0.0
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
                        cell=cell_torch,
                        pbc=pbc_torch,
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
                t_force_end = time_module.time() if track_timing else 0.0

            # Record timing for this structure
            if track_timing and timing_dict is not None:
                t_end = time_module.time()
                timing_dict['featurization'].append(t_feat_end - t_feat_start)
                timing_dict['energy_eval'].append(
                    t_energy_end - t_energy_start)
                timing_dict['force_eval'].append(
                    t_force_end - t_force_start if eval_forces else 0.0)
                timing_dict['total'].append(t_end - t_start)

        return (
            energies,
            forces_out if eval_forces else None,
            atom_energies_out if return_atom_energies else None,
            timing_dict,
        )

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
