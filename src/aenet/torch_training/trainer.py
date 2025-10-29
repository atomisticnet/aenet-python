"""
Trainer for PyTorch-based MLIP using on-the-fly featurization.

Implements TorchANNPotential which mirrors aenet.mlip.ANNPotential-style API
for training and prediction, using:
- ChebyshevDescriptor (on-the-fly features + neighbor_info)
- EnergyModelAdapter over aenet-PyTorch NetAtom
- Modular components for building, training, and inference

Notes
-----
- Default dtype is float64 for scientific reproducibility
- Devices: 'cpu' or 'cuda' (auto if config.device is None)
- Memory modes: 'cpu' and 'gpu' behave as expected; 'mixed' currently
  behaves like 'gpu' (streaming/CPU-GPU mixing can be added later)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Progress bar (match aenet.mlip behavior)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

from .config import TorchTrainingConfig, Structure, TrainingMethod
from .dataset import StructureDataset, train_test_split
from .model_adapter import EnergyModelAdapter

# Refactored modules
from .builders import NetworkBuilder, OptimizerBuilder
from .training import (
    CheckpointManager,
    MetricsTracker,
    NormalizationManager,
    TrainingLoop,
)
from .training import training_loop as training_loop_mod
from .inference import Predictor


def _resolve_device(config: TorchTrainingConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _iter_progress(iterable, enable: bool, desc: str):
    """
    Wrap an iterable with tqdm progress bar if enabled and available.
    """
    if enable and tqdm is not None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None
        return tqdm(iterable, total=total, desc=desc, ncols=80, leave=False)
    return iterable


def _collate_fn(batch: List[dict]) -> Dict[str, Any]:
    """
    Collate a batch of samples from StructureDataset.

    Returns a dict with:
      - energy view (all structures):
          features: (N,F), species_indices: (N,), n_atoms: (B,),
          energy_ref: (B,)
      - force view (only selected structures):
          positions_f, species_f (list[str]), species_indices_f: (Nf,),
          forces_ref_f: (Nf,3), neighbor_info_f: dict (lists length Nf)
      - misc: batch_sizes and masks for debugging
    """
    # Energy view aggregation (all structures)
    features_list: List[torch.Tensor] = []
    species_idx_list: List[torch.Tensor] = []
    n_atoms_list: List[int] = []
    energy_ref_list: List[float] = []

    # For optional force view
    positions_f_list: List[torch.Tensor] = []
    species_f_names: List[str] = []
    species_idx_f_list: List[torch.Tensor] = []
    forces_f_list: List[torch.Tensor] = []
    nb_lists_f: List[Any] = []         # list of numpy arrays (indices)
    nb_vectors_f: List[Any] = []       # list of numpy arrays (vectors)

    # We also need neighbor_info for the full batch of force-selected atoms
    # We will offset neighbor indices when concatenating.
    base_atom_idx_force = 0

    for sample in batch:
        N = int(sample["n_atoms"])
        features_list.append(sample["features"])
        species_idx_list.append(sample["species_indices"])
        n_atoms_list.append(N)
        energy_ref_list.append(float(sample["energy"]))

    # Build energy-view tensors
    features = (torch.cat(features_list, dim=0)
                if features_list else torch.empty(0, 0))
    species_indices = (
        torch.cat(species_idx_list, dim=0)
        if species_idx_list
        else torch.empty(0, dtype=torch.long)
    )
    n_atoms = torch.tensor(n_atoms_list, dtype=torch.long)
    energy_ref = torch.tensor(energy_ref_list, dtype=features.dtype)

    # Build force-view if any sample is selected for forces
    for sample in batch:
        if not bool(sample["use_forces"]) or sample["forces"] is None:
            continue
        pos = sample["positions"]
        frc = sample["forces"]
        species_idx = sample["species_indices"]
        species_names = sample["species"]
        nb_info = sample["neighbor_info"]  # numpy arrays

        positions_f_list.append(pos)
        forces_f_list.append(frc)
        species_idx_f_list.append(species_idx)
        species_f_names.extend(species_names)

        # Offset neighbor indices by base_atom_idx_force
        for arr in nb_info["neighbor_lists"]:
            if arr is None or len(arr) == 0:
                nb_lists_f.append(arr)
            else:
                nb_lists_f.append(arr + base_atom_idx_force)
        for vec in nb_info["neighbor_vectors"]:
            nb_vectors_f.append(vec)

        base_atom_idx_force += int(pos.shape[0])

    force_view_present = len(positions_f_list) > 0
    positions_f = (torch.cat(positions_f_list, dim=0)
                   if force_view_present else None)
    species_indices_f = (
        torch.cat(species_idx_f_list, dim=0) if force_view_present else None
    )
    forces_ref_f = (torch.cat(forces_f_list, dim=0)
                    if force_view_present else None)
    neighbor_info_f = (
        {"neighbor_lists": nb_lists_f, "neighbor_vectors": nb_vectors_f}
        if force_view_present
        else None
    )

    return {
        # Energy view
        "features": features,
        "species_indices": species_indices,
        "n_atoms": n_atoms,
        "energy_ref": energy_ref,
        # Force view
        "positions_f": positions_f,
        "species_f": species_f_names if force_view_present else None,
        "species_indices_f": species_indices_f,
        "forces_ref_f": forces_ref_f,
        "neighbor_info_f": neighbor_info_f,
    }


def _descriptor_config(desc) -> Dict[str, Any]:
    """Serialize descriptor configuration to a plain dict."""
    try:
        n_features = int(desc.get_n_features())
    except Exception:
        n_features = None  # type: ignore[assignment]
    return {
        "species": list(getattr(desc, "species", [])),
        "rad_order": int(getattr(desc, "rad_order", 0)),
        "rad_cutoff": float(getattr(desc, "rad_cutoff", 0.0)),
        "ang_order": int(getattr(desc, "ang_order", 0)),
        "ang_cutoff": float(getattr(desc, "ang_cutoff", 0.0)),
        "min_cutoff": float(getattr(desc, "min_cutoff", 0.0)),
        "dtype": str(getattr(desc, "dtype", "")).replace("torch.", ""),
        "device": str(getattr(desc, "device", "")),
        "n_features": n_features if n_features is not None else 0,
    }


class TorchANNPotential:
    """
    PyTorch MLIP trainer using on-the-fly featurization and
    aenet-PyTorch NetAtom.

    Parameters
    ----------
    arch : Dict[str, List[Tuple[int, str]]]
        Architecture per species: list of (hidden_nodes, activation)
        Output layer is implicit linear(1)
    descriptor : ChebyshevDescriptor
        Descriptor instance used for featurization
    """

    def __init__(self, arch: Dict[str, List[Tuple[int, str]]], descriptor):
        self.arch = arch
        self.descriptor = descriptor
        self.device = torch.device(str(descriptor.device))
        self.dtype = descriptor.dtype

        # Build underlying network and adapter using NetworkBuilder
        builder = NetworkBuilder(descriptor=descriptor, device=self.device, dtype=self.dtype)
        net = builder.build_network(arch=self.arch)
        self.net = net
        self.model = EnergyModelAdapter(
            net=net, n_species=len(descriptor.species))
        # Ensure adapter on same device/dtype
        if self.dtype == torch.float64:
            self.model = self.model.double()
        else:
            self.model = self.model.float()
        self.model.to(self.device)

        # Training state - use MetricsTracker
        self._metrics = MetricsTracker(track_detailed_timing=True)
        self.best_val: Optional[float] = None

        # Normalization - will be initialized during training
        self._normalizer: Optional[NormalizationManager] = None

        # Atomic reference energies
        self._E_atomic: Optional[Dict[str, float]] = None

        # Energy target (cohesive vs total) - set during training
        self._energy_target: str = "cohesive"

    @property
    def history(self) -> Dict[str, List[float]]:
        """Access training history through metrics tracker."""
        return self._metrics.get_history()

    @history.setter
    def history(self, value: Dict[str, List[float]]):
        """Set training history (for loading from checkpoint)."""
        self._metrics.set_history(value)

    def train(
        self,
        structures: List[Structure],
        config: Optional[TorchTrainingConfig] = None,
        checkpoint_dir: Optional[str] = "checkpoints",
        checkpoint_interval: int = 1,
        max_checkpoints: Optional[int] = None,
        resume_from: Optional[str] = None,
        save_best: bool = True,
        use_scheduler: bool = False,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
    ) -> Dict[str, List[float]]:
        """
        Train the MLIP and return training history.

        Notes
        -----
        - Energy RMSE is computed per-atom (consistent with aenet-PyTorch)
        - Force RMSE is overall RMSE across all components in batch
        """
        if config is None:
            config = TorchTrainingConfig()

        # Resolve device/dtype and memory mode
        device = _resolve_device(config)
        memory_mode = config.memory_mode
        if memory_mode not in ("cpu", "gpu", "mixed"):
            raise ValueError(f"Invalid memory_mode '{memory_mode}'")

        # Ensure model on device/dtype
        self.model.to(device)
        if self.dtype == torch.float64:
            self.model = self.model.double()
        else:
            self.model = self.model.float()

        # Cohesive energy configuration
        energy_target = getattr(config, "energy_target", "cohesive")
        self._energy_target = energy_target  # Store for later use in predict
        E_atomic_by_index = None
        if energy_target == "cohesive":
            E_atomic_cfg = getattr(config, "E_atomic", None)
            if E_atomic_cfg:
                e_list: List[float] = []
                missing: List[str] = []
                for s in self.descriptor.species:
                    if s in E_atomic_cfg:
                        e_list.append(float(E_atomic_cfg[s]))
                    else:
                        e_list.append(0.0)
                        missing.append(s)
                if missing:
                    print(f"[WARN] E_atomic missing for species {missing}; "
                          + "treating as 0.0 for cohesive conversion.")
                E_atomic_by_index = torch.tensor(
                    e_list, dtype=self.dtype, device=device)
                # Persist dictionary form for later prediction/export
                try:
                    self._E_atomic = {str(k): float(v)
                                      for k, v in E_atomic_cfg.items()}
                except Exception:
                    self._E_atomic = dict(E_atomic_cfg)
            else:
                print("[WARN] energy_target='cohesive' but config.E_atomic "
                      "not provided; training will use provided energies "
                      "unchanged.")

        # Dataset and split
        if bool(getattr(config, "cached_features", False)
                ) and float(config.alpha) == 0.0:
            # Energy-only cached-features path
            structures_all = structures
            if config.testpercent > 0:
                test_fraction = config.testpercent / 100.0
                import random as _rand
                idx = list(range(len(structures_all)))
                _rand.shuffle(idx)
                n_test = int(len(idx) * test_fraction)
                test_idx = set(idx[:n_test])
                train_idx = set(idx[n_test:])
                train_structs = [structures_all[i] for i in sorted(train_idx)]
                test_structs = [structures_all[i] for i in sorted(test_idx)]
            else:
                train_structs = structures_all
                test_structs = []
            from .dataset import CachedStructureDataset
            train_ds = CachedStructureDataset(
                structures=train_structs,
                descriptor=self.descriptor,
                max_energy=config.max_energy,
                max_forces=config.max_forces,
                seed=None,
            )
            test_ds = (
                CachedStructureDataset(
                    structures=test_structs,
                    descriptor=self.descriptor,
                    max_energy=config.max_energy,
                    max_forces=config.max_forces,
                    seed=None,
                )
                if (config.testpercent > 0 and len(test_structs) > 0)
                else None
            )
        else:
            full_ds = StructureDataset(
                structures=structures,
                descriptor=self.descriptor,
                force_fraction=config.force_fraction,
                force_sampling=config.force_sampling,
                max_energy=config.max_energy,
                max_forces=config.max_forces,
                seed=None,
            )
            if config.testpercent > 0:
                test_fraction = config.testpercent / 100.0
                train_ds, test_ds = train_test_split(
                    full_ds, test_fraction=test_fraction)
            else:
                train_ds, test_ds = full_ds, None

        # DataLoaders
        batch_size = OptimizerBuilder.get_batch_size(config.method)
        dl_kwargs: Dict[str, Any] = {}
        nw = int(getattr(config, "num_workers", 0))
        if nw > 0:
            dl_kwargs.update(
                dict(
                    num_workers=nw,
                    prefetch_factor=int(
                        getattr(config, "prefetch_factor", 2)),
                    persistent_workers=bool(
                        getattr(config, "persistent_workers", True)),
                )
            )
        else:
            dl_kwargs.update(dict(num_workers=0))

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            **dl_kwargs,
        )
        test_loader = (
            DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_fn,
                **dl_kwargs,
            )
            if test_ds is not None
            else None
        )

        # Initialize normalization manager
        normalize_features = bool(getattr(config, "normalize_features", True))
        normalize_energy = bool(getattr(config, "normalize_energy", True))

        self._normalizer = NormalizationManager(
            normalize_features=normalize_features,
            normalize_energy=normalize_energy,
            dtype=self.dtype,
            device=device,
        )

        # Stats DataLoader (no shuffle)
        stats_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=False, collate_fn=_collate_fn
        )

        # Compute normalization statistics
        try:
            n_features = int(self.descriptor.get_n_features())
        except Exception:
            n_features = None  # type: ignore[assignment]

        # Feature stats
        if normalize_features and n_features is not None:
            provided_stats = getattr(config, "feature_stats", None)
            if provided_stats:
                mean_np = provided_stats.get("mean", None)
                std_np = provided_stats.get("std", None) or provided_stats.get("cov", None)
                if mean_np is not None and std_np is not None:
                    self._normalizer.set_feature_stats(mean_np, std_np)

            if not self._normalizer.has_feature_stats():
                self._normalizer.compute_feature_stats(stats_loader, n_features)

        # Energy normalization stats
        if normalize_energy:
            # Check for provided overrides
            provided_shift = getattr(config, "E_shift", None)
            provided_scaling = getattr(config, "E_scaling", None)

            if provided_shift is not None and provided_scaling is not None:
                self._normalizer.set_energy_stats(
                    shift=float(provided_shift),
                    scaling=float(provided_scaling)
                )
            else:
                # Compute from data
                self._normalizer.compute_energy_stats(
                    stats_loader, E_atomic_by_index
                )

        # Optimizer and scheduler using OptimizerBuilder
        opt_builder = OptimizerBuilder(self.model)
        optimizer = opt_builder.build_optimizer(config.method)
        scheduler = opt_builder.build_scheduler(
            optimizer,
            use_scheduler=bool(use_scheduler) and (test_loader is not None),
            scheduler_patience=int(scheduler_patience),
            scheduler_factor=float(scheduler_factor),
            scheduler_min_lr=float(scheduler_min_lr),
        )

        # Checkpoint manager
        ckpt_manager = None
        if checkpoint_dir is not None:
            ckpt_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_to_keep=max_checkpoints,
                save_best=save_best,
            )

        start_epoch = 0
        if resume_from is not None and ckpt_manager is not None:
            payload = ckpt_manager.load_checkpoint(
                resume_from, self.model, optimizer, device
            )
            if payload:
                # Restore training state
                if "history" in payload:
                    self._metrics.set_history(payload["history"])
                if "best_val_loss" in payload:
                    self.best_val = payload["best_val_loss"]
                if "normalization" in payload:
                    self._normalizer.set_state(payload["normalization"])
                # Infer start epoch
                start_epoch = ckpt_manager.infer_start_epoch(resume_from)

        n_epochs = int(config.iterations)
        alpha = float(config.alpha)

        # Training loop instance
        training_loop = TrainingLoop(
            model=self.model,
            descriptor=self.descriptor,
            normalizer=self._normalizer,
            device=device,
            dtype=self.dtype,
        )

        # Progress bar configuration
        show_progress = bool(getattr(config, "show_progress", True))
        show_batch = bool(getattr(config, "show_batch_progress", False)
                          ) and show_progress

        # Ensure inner batch progress bars use trainer.tqdm (for tests)
        if show_batch:
            try:
                training_loop_mod.tqdm = tqdm
            except Exception:
                pass

        pbar = (
            tqdm(total=(n_epochs - start_epoch), desc="Training", ncols=80)
            if show_progress and tqdm is not None
            else None
        )

        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            # Force sampling each epoch
            if hasattr(train_ds, "resample_force_structures"
                       ) and config.force_sampling == "random":
                train_ds.resample_force_structures()

            # Train one epoch
            train_energy_rmse, train_force_rmse, train_timing = training_loop.run_epoch(
                loader=train_loader,
                optimizer=optimizer,
                alpha=alpha,
                energy_target=energy_target,
                E_atomic_by_index=E_atomic_by_index,
                train=True,
                show_batch_progress=show_batch,
            )

            # Validation
            val_energy_rmse = float("nan")
            val_force_rmse = float("nan")
            val_timing = {}
            if test_loader is not None:
                val_energy_rmse, val_force_rmse, val_timing = training_loop.run_epoch(
                    loader=test_loader,
                    optimizer=None,
                    alpha=alpha,
                    energy_target=energy_target,
                    E_atomic_by_index=E_atomic_by_index,
                    train=False,
                    show_batch_progress=False,
                )

                # Compute validation loss for monitoring
                import math
                val_loss_monitor = (
                    (1 - alpha) * val_energy_rmse
                    + (alpha * val_force_rmse
                       if not math.isnan(val_force_rmse) else 0.0)
                )

                # Best model tracking
                if save_best and ckpt_manager is not None:
                    if ckpt_manager.should_save_best(val_loss_monitor):
                        self.best_val = float(ckpt_manager.best_val_loss) if ckpt_manager.best_val_loss is not None else float(val_loss_monitor)
                        ckpt_manager.save_best_model(
                            model=self.model,
                            optimizer=optimizer,
                            epoch=epoch,
                            history=self._metrics.get_history(),
                            architecture=self.arch,
                            descriptor_config=_descriptor_config(self.descriptor),
                        )

                # Scheduler
                if scheduler is not None:
                    scheduler.step(val_loss_monitor)

            # Update metrics
            epoch_time = time.time() - t0
            self._metrics.update(
                train_energy_rmse=float(train_energy_rmse),
                train_force_rmse=float(train_force_rmse),
                test_energy_rmse=float(val_energy_rmse),
                test_force_rmse=float(val_force_rmse),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                epoch_time=epoch_time,
                forward_time=training_loop.last_forward_time,
                backward_time=training_loop.last_backward_time,
                train_timing=train_timing,
                val_timing=val_timing,
            )

            # Update progress bar
            if pbar is not None:
                import math
                lr_val = float(optimizer.param_groups[0]["lr"])
                pf: Dict[str, Any] = {
                    "trE": f"{train_energy_rmse:.4g}",
                    "lr": f"{lr_val:.2e}",
                    "fwd": f"{training_loop.last_forward_time:.2f}s",
                    "bwd": f"{training_loop.last_backward_time:.2f}s",
                }
                if not math.isnan(train_force_rmse):
                    pf["trF"] = f"{train_force_rmse:.4g}"
                if not math.isnan(val_energy_rmse):
                    pf["vaE"] = f"{val_energy_rmse:.4g}"
                if not math.isnan(val_force_rmse):
                    pf["vaF"] = f"{val_force_rmse:.4g}"
                pbar.set_postfix(pf)
                pbar.update(1)

            # Checkpointing
            if ckpt_manager is not None and checkpoint_interval > 0:
                if ((epoch + 1) % int(checkpoint_interval)) == 0:
                    ckpt_manager.save_checkpoint(
                        model=self.model,
                        optimizer=optimizer,
                        epoch=epoch,
                        history=self._metrics.get_history(),
                        architecture=self.arch,
                        descriptor_config=_descriptor_config(self.descriptor),
                    )

        if pbar is not None:
            pbar.close()

        return self._metrics.get_history()

    @torch.no_grad()
    def predict(
        self,
        structures: List[Structure],
        predict_forces: bool = False,
    ) -> Tuple[List[float], Optional[List[torch.Tensor]]]:
        """
        Predict energies (and optionally forces) for a list of structures.

        Returns
        -------
        energies : List[float]
        forces : Optional[List[Tensor]]  # (N_i, 3) per-structure if requested
        """
        # Create predictor instance
        energy_target = getattr(self, "_energy_target", "cohesive")
        if not hasattr(self, "_energy_target"):
            energy_target = "cohesive"  # default for backward compatibility

        predictor = Predictor(
            model=self.model,
            descriptor=self.descriptor,
            normalizer=self._normalizer,
            energy_target=energy_target,
            E_atomic=self._E_atomic,
            device=self.device,
            dtype=self.dtype,
        )

        return predictor.predict(structures, predict_forces=predict_forces)

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
        atomic_energies : dict[str, float], optional
            Per-species atomic reference energies. If None, uses trainer's
            stored E_atomic from training.

        Returns
        -------
        float
            Cohesive energy (total - sum of atomic reference energies).

        Raises
        ------
        ValueError
            If no atomic energies are available.
        """
        predictor = Predictor(
            model=self.model,
            descriptor=self.descriptor,
            normalizer=self._normalizer,
            energy_target="total",
            E_atomic=atomic_energies if atomic_energies is not None else self._E_atomic,
            device=self.device,
            dtype=self.dtype,
        )

        return predictor.cohesive_energy(structure)
