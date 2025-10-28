"""
Trainer for PyTorch-based MLIP using on-the-fly featurization.

Implements TorchANNPotential which mirrors aenet.mlip.ANNPotential-style API
for training and prediction, using:
- ChebyshevDescriptor (on-the-fly features + neighbor_info)
- EnergyModelAdapter over aenet-PyTorch NetAtom
- Loss functions from src/aenet/torch_training/loss.py

Notes
-----
- Default dtype is float64 for scientific reproducibility
- Devices: 'cpu' or 'cuda' (auto if config.device is None)
- Memory modes: 'cpu' and 'gpu' behave as expected; 'mixed' currently
  behaves like 'gpu' (streaming/CPU-GPU mixing can be added later)
"""

from __future__ import annotations

import time
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TorchTrainingConfig, Structure, TrainingMethod, Adam, SGD
from .dataset import StructureDataset, train_test_split
from .model_adapter import EnergyModelAdapter
from .loss import compute_energy_loss, compute_force_loss


def _resolve_device(config: TorchTrainingConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _method_batch_size(method: TrainingMethod) -> int:
    if hasattr(method, "batchsize"):
        return int(getattr(method, "batchsize"))
    if hasattr(method, "batch_size"):
        return int(getattr(method, "batch_size"))
    return 32


def _import_netatom() -> Optional[type]:
    """
    Dynamically import aenet-PyTorch NetAtom from external/aenet-pytorch.

    Returns
    -------
    NetAtom class or None if not found.
    """
    try:
        # trainer.py -> torch_training -> aenet -> src -> project root
        root = Path(__file__).resolve().parents[3]
        net_path = root / "external" / "aenet-pytorch" / "src" / "network.py"
        if not net_path.exists():
            return None
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "aenet_pytorch.network", str(net_path)
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[assignment]
        if hasattr(module, "NetAtom"):
            return getattr(module, "NetAtom")
        return None
    except Exception:
        return None


def _validate_arch(
    arch: Dict[str, List[Tuple[int, str]]],
    species_order: List[str],
) -> Tuple[List[List[int]], List[List[str]]]:
    """
    Validate architecture and produce per-species hidden sizes and activations.

    Parameters
    ----------
    arch : dict
        {species_symbol: [(nodes, activation), ...]} (output layer is implicit)
    species_order : list[str]
        Species ordering to align with descriptor/species_indices

    Returns
    -------
    hidden_sizes : list[list[int]]
    activations : list[list[str]]

    Raises
    ------
    ValueError on unsupported activation or missing species.
    """
    supported = {"linear", "tanh", "sigmoid"}
    hidden_sizes: List[List[int]] = []
    activations: List[List[str]] = []

    for s in species_order:
        if s not in arch:
            raise ValueError(f"Species '{s}' missing in architecture.")
        layers = arch[s]
        hs: List[int] = []
        acts: List[str] = []
        for nodes, act in layers:
            act_l = act.lower()
            if act_l not in supported:
                raise ValueError(
                    f"Unsupported activation '{act}' for species '{s}'. "
                    f"Supported: {sorted(supported)}"
                )
            hs.append(int(nodes))
            acts.append(act_l)
        if len(hs) == 0:
            raise ValueError(f"Architecture for species '{s}' must be non-empty.")
        hidden_sizes.append(hs)
        activations.append(acts)
    return hidden_sizes, activations


def _build_fallback_per_species_mlps(
    n_features: int,
    species: List[str],
    hidden_sizes: List[List[int]],
    activations: List[List[str]],
) -> nn.ModuleList:
    """
    Fallback builder that mimics NetAtom.functions[iesp] layout.

    Returns
    -------
    nn.ModuleList of per-species nn.Sequential models mapping (F) -> (1)
    """
    act_map: Dict[str, nn.Module] = {
        "linear": nn.Identity(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    seqs: List[nn.Sequential] = []
    for i, _sp in enumerate(species):
        hs = hidden_sizes[i]
        acts = activations[i]
        layers: List[Tuple[str, nn.Module]] = []
        # First linear + act
        layers.append((f"Linear_Sp{i+1}_F1", nn.Linear(n_features, hs[0])))
        layers.append((f"Active_Sp{i+1}_F1", act_map[acts[0]]))
        # Hidden stacks
        for j in range(1, len(hs)):
            layers.append(
                (f"Linear_Sp{i+1}_F{j+1}", nn.Linear(hs[j - 1], hs[j]))
            )
            layers.append((f"Active_Sp{i+1}_F{j+1}", act_map[acts[j]]))
        # Output layer
        layers.append((f"Linear_Sp{i+1}_F{len(hs)+1}", nn.Linear(hs[-1], 1)))
        seqs.append(nn.Sequential(dict(layers)))  # type: ignore[arg-type]
    return nn.ModuleList(seqs)


def _build_network_from_arch(
    arch: Dict[str, List[Tuple[int, str]]],
    descriptor,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """
    Build NetAtom (preferred) or fallback per-species MLPs.

    Returns
    -------
    net : nn.Module
        - If NetAtom available: an instance with attributes:
            .functions: ModuleList of per-species Sequential MLPs
            .device: device string
        - Else: a small wrapper exposing .functions and .device
    """
    species = list(descriptor.species)
    n_features = int(descriptor.get_n_features())
    hidden_sizes, activations = _validate_arch(arch, species)
    NetAtom = _import_netatom()

    if NetAtom is not None:
        # NetAtom expects lists per species
        input_size = [n_features for _ in species]
        alpha = 1.0
        net = NetAtom(
            input_size=input_size,
            hidden_size=hidden_sizes,
            species=species,
            activations=activations,
            alpha=alpha,
            device=str(device),
        )
        # Ensure dtype/device
        if dtype == torch.float64:
            net = net.double()
        else:
            net = net.float()
        # NetAtom stores device string
        net.device = str(device)
        net.to(device)
        return net

    # Fallback: simple wrapper with .functions and .device
    class _FallbackNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.functions = _build_fallback_per_species_mlps(
                n_features=n_features,
                species=species,
                hidden_sizes=hidden_sizes,
                activations=activations,
            )
            self.species = species  # for debugging
            self.device = str(device)

        def forward(self, *args, **kwargs):  # not used; adapter calls .functions
            raise RuntimeError("Use EnergyModelAdapter to call per-atom energies.")

    net = _FallbackNet()
    if dtype == torch.float64:
        net = net.double()
    else:
        net = net.float()
    net.to(device)
    return net


def _optimizer_from_config(
    params, method: TrainingMethod
) -> torch.optim.Optimizer:
    if isinstance(method, Adam):
        return torch.optim.Adam(
            params,
            lr=float(method.mu),
            betas=(float(method.beta1), float(method.beta2)),
            eps=float(method.epsilon),
            weight_decay=float(method.weight_decay),
        )
    if isinstance(method, SGD):
        return torch.optim.SGD(
            params,
            lr=float(method.lr),
            momentum=float(method.momentum),
            weight_decay=float(method.weight_decay),
        )
    # Default to Adam with conservative params if unknown
    return torch.optim.Adam(params, lr=1e-3)


def _maybe_scheduler(
    optimizer: torch.optim.Optimizer,
    use_scheduler: bool,
    scheduler_patience: int,
    scheduler_factor: float,
    scheduler_min_lr: float,
):
    if not use_scheduler:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
        verbose=False,
    )


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _rotate_checkpoints(ckpt_dir: Path, max_to_keep: int):
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if len(ckpts) <= max_to_keep:
        return
    to_remove = ckpts[: len(ckpts) - max_to_keep]
    for p in to_remove:
        try:
            p.unlink()
        except Exception:
            pass


def _serialize_config(config: TorchTrainingConfig) -> Dict[str, Any]:
    # Dataclass to dict, but nested classes may not be dataclasses
    d = asdict(config)
    if isinstance(config.method, TrainingMethod):
        # Replace method with a simple dict
        d["method"] = {
            "name": getattr(config.method, "method_name", "unknown"),
            **{k: v for k, v in config.method.__dict__.items()},
        }
    return d


def _descriptor_config(desc) -> Dict[str, Any]:
    return {
        "species": list(desc.species),
        "rad_order": int(desc.rad_order),
        "rad_cutoff": float(desc.rad_cutoff),
        "ang_order": int(desc.ang_order),
        "ang_cutoff": float(desc.ang_cutoff),
        "min_cutoff": float(desc.min_cutoff),
        "dtype": str(desc.dtype).replace("torch.", ""),
        "device": str(desc.device),
        "n_features": int(desc.get_n_features()),
    }


def _collate_fn(batch: List[dict]) -> Dict[str, Any]:
    """
    Collate a batch of samples from StructureDataset.

    Returns a dict with:
      - energy view (all structures):
          features: (N,F), species_indices: (N,), n_atoms: (B,), energy_ref: (B,)
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
    features = torch.cat(features_list, dim=0) if features_list else torch.empty(0, 0)
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
    positions_f = torch.cat(positions_f_list, dim=0) if force_view_present else None
    species_indices_f = (
        torch.cat(species_idx_f_list, dim=0) if force_view_present else None
    )
    forces_ref_f = torch.cat(forces_f_list, dim=0) if force_view_present else None
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


class TorchANNPotential:
    """
    PyTorch MLIP trainer using on-the-fly featurization and aenet-PyTorch NetAtom.

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

        # Build underlying network and adapter
        net = _build_network_from_arch(
            arch=self.arch, descriptor=self.descriptor, device=self.device, dtype=self.dtype
        )
        self.net = net
        self.model = EnergyModelAdapter(net=net, n_species=len(descriptor.species))
        # Ensure adapter on same device/dtype
        if self.dtype == torch.float64:
            self.model = self.model.double()
        else:
            self.model = self.model.float()
        self.model.to(self.device)

        # Training state
        self.history: Dict[str, List[float]] = {
            "train_energy_rmse": [],
            "test_energy_rmse": [],
            "train_force_rmse": [],
            "test_force_rmse": [],
            "learning_rates": [],
            "epoch_times": [],
            "epoch_forward_time": [],
            "epoch_backward_time": [],
        }
        self.best_val: Optional[float] = None

        # Normalization state
        self._normalize_features: bool = False
        self._normalize_energy: bool = False
        self._feature_mean: Optional[torch.Tensor] = None  # (F,)
        self._feature_std: Optional[torch.Tensor] = None   # (F,)
        self._E_shift: float = 0.0  # per-atom shift
        self._E_scaling: float = 1.0

    def train(
        self,
        structures: List[Structure],
        config: Optional[TorchTrainingConfig] = None,
        # Phase 3 trainer options:
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
        # For now, treat 'mixed' like 'gpu'
        memory_mode = config.memory_mode
        if memory_mode not in ("cpu", "gpu", "mixed"):
            raise ValueError(f"Invalid memory_mode '{memory_mode}'")

        # Ensure model on device/dtype
        self.model.to(device)
        if self.dtype == torch.float64:
            self.model = self.model.double()
        else:
            self.model = self.model.float()

        # Cohesive energy configuration and timing flag
        self._energy_target = getattr(config, "energy_target", "cohesive")
        self._timing = bool(getattr(config, "timing", False))
        self._E_atomic_by_index = None
        if self._energy_target == "cohesive":
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
                    print(f"[WARN] E_atomic missing for species {missing}; treating as 0.0 for cohesive conversion.")
                self._E_atomic_by_index = torch.tensor(e_list, dtype=self.dtype, device=device)
            else:
                print("[WARN] energy_target='cohesive' but config.E_atomic not provided; training will use provided energies unchanged.")

        # Dataset and split
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
            train_ds, test_ds = train_test_split(full_ds, test_fraction=test_fraction)
        else:
            train_ds, test_ds = full_ds, None

        # DataLoaders
        batch_size = _method_batch_size(config.method)  # type: ignore[arg-type]
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn
        )
        test_loader = (
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
            if test_ds is not None
            else None
        )

        # Normalization configuration
        self._normalize_features = bool(getattr(config, "normalize_features", True))
        self._normalize_energy = bool(getattr(config, "normalize_energy", True))

        # Compute/assign normalization statistics from training split only
        # Feature stats: mean/std per feature dimension
        # Energy stats: per-atom E_min/E_max/E_avg and E_shift/E_scaling
        try:
            F = int(self.descriptor.get_n_features())
        except Exception:
            F = None  # type: ignore[assignment]

        # Stats DataLoader (no shuffle)
        stats_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn
        )

        # Feature stats
        self._feature_mean = None
        self._feature_std = None
        if self._normalize_features and F is not None:
            if getattr(config, "feature_stats", None):
                # Use provided stats (expects numpy arrays)
                fs = getattr(config, "feature_stats")
                mean_np = fs.get("mean", None)
                std_np = fs.get("std", None) or fs.get("cov", None)
                if mean_np is not None and std_np is not None:
                    self._feature_mean = torch.as_tensor(mean_np, dtype=self.dtype, device=device).view(-1)
                    self._feature_std = torch.as_tensor(std_np, dtype=self.dtype, device=device).view(-1)
            if self._feature_mean is None or self._feature_std is None:
                # Compute from training split
                sum_f = torch.zeros(F, dtype=self.dtype, device="cpu")
                sumsq_f = torch.zeros(F, dtype=self.dtype, device="cpu")
                total_atoms_stats = 0
                with torch.no_grad():
                    for b in stats_loader:
                        feats = b["features"]
                        feats = feats.to(dtype=self.dtype, device="cpu")
                        sum_f += feats.sum(dim=0).cpu()
                        sumsq_f += (feats * feats).sum(dim=0).cpu()
                        total_atoms_stats += int(feats.shape[0])
                if total_atoms_stats > 0:
                    mean = sum_f / float(total_atoms_stats)
                    var = torch.clamp(sumsq_f / float(total_atoms_stats) - mean * mean, min=0.0)
                    std = torch.sqrt(var + torch.as_tensor(1e-12, dtype=var.dtype))
                    self._feature_mean = mean.to(device=device)
                    self._feature_std = std.to(device=device)

        # Energy normalization stats
        # Prefer overrides if provided
        if getattr(config, "E_shift", None) is not None:
            self._E_shift = float(getattr(config, "E_shift"))
        if getattr(config, "E_scaling", None) is not None:
            self._E_scaling = float(getattr(config, "E_scaling"))

        if self._normalize_energy and (getattr(config, "E_shift", None) is None or getattr(config, "E_scaling", None) is None):
            # Compute per-atom energy min/max/avg from chosen target space
            e_min = None
            e_max = None
            e_sum = 0.0
            n_struct = 0
            with torch.no_grad():
                for b in stats_loader:
                    n_atoms_b = b["n_atoms"].to(device)
                    energy_ref_b = b["energy_ref"].to(device, dtype=self.dtype)
                    species_indices_b = b["species_indices"].to(device)
                    # Convert totals to cohesive if requested and E_atomic available
                    energy_target_b = energy_ref_b
                    if getattr(self, "_energy_target", "cohesive") == "cohesive" and getattr(self, "_E_atomic_by_index", None) is not None:
                        per_atom_Ea_b = self._E_atomic_by_index[species_indices_b]
                        batch_idx_b = torch.repeat_interleave(
                            torch.arange(len(n_atoms_b), device=device), n_atoms_b.long()
                        )
                        Ea_sum_b = torch.zeros(len(n_atoms_b), dtype=energy_ref_b.dtype, device=device)
                        Ea_sum_b.scatter_add_(0, batch_idx_b, per_atom_Ea_b)
                        energy_target_b = energy_ref_b - Ea_sum_b
                    # Per-atom energies for structures
                    e_pa = energy_target_b / n_atoms_b
                    # Update stats
                    e_min = float(torch.min(e_pa).item()) if e_min is None else min(e_min, float(torch.min(e_pa).item()))
                    e_max = float(torch.max(e_pa).item()) if e_max is None else max(e_max, float(torch.max(e_pa).item()))
                    e_sum += float(torch.sum(e_pa).item())
                    n_struct += int(len(n_atoms_b))
            if e_min is not None and e_max is not None and e_max > e_min:
                # Match aenet: normalize to [-1, 1]
                self._E_scaling = float(2.0 / (e_max - e_min))
                self._E_shift = float(0.5 * (e_max + e_min))
            else:
                # Degenerate case: disable energy normalization
                self._E_scaling = 1.0
                self._E_shift = 0.0

        # Optimizer and optional scheduler
        optimizer = _optimizer_from_config(self.model.parameters(), config.method)  # type: ignore[arg-type]
        scheduler = _maybe_scheduler(
            optimizer,
            use_scheduler=bool(use_scheduler) and (test_loader is not None),
            scheduler_patience=int(scheduler_patience),
            scheduler_factor=float(scheduler_factor),
            scheduler_min_lr=float(scheduler_min_lr),
        )

        # Checkpoint directory
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if ckpt_dir is not None:
            _ensure_dir(ckpt_dir)

        start_epoch = 0
        if resume_from is not None:
            self._load_checkpoint(resume_from, optimizer)

            # Attempt to infer start_epoch from filename
            try:
                name = Path(resume_from).name
                if name.startswith("checkpoint_epoch_") and name.endswith(".pt"):
                    start_epoch = int(name[len("checkpoint_epoch_") : -3]) + 1
            except Exception:
                start_epoch = 0

        n_epochs = int(config.iterations)
        alpha = float(config.alpha)  # alias of force_weight

        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            # Force sampling (random) each epoch
            if hasattr(train_ds, "resample_force_structures") and config.force_sampling == "random":
                train_ds.resample_force_structures()

            # Train one epoch
            train_energy_rmse, train_force_rmse = self._run_epoch(
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                alpha=alpha,
                train=True,
            )

            # Validation
            if test_loader is not None:
                val_energy_rmse, val_force_rmse = self._run_epoch(
                    loader=test_loader,
                    optimizer=None,
                    device=device,
                    alpha=alpha,
                    train=False,
                )
                val_loss_monitor = (
                    (1 - alpha) * val_energy_rmse + (alpha * val_force_rmse if not math.isnan(val_force_rmse) else 0.0)
                )
                # Best model tracking
                if save_best:
                    improved = (self.best_val is None) or (val_loss_monitor < float(self.best_val))
                    if improved:
                        self.best_val = float(val_loss_monitor)
                        if ckpt_dir is not None:
                            best_path = ckpt_dir / "best_model.pt"
                            self._save_checkpoint(best_path, epoch, optimizer)
                # Scheduler
                if scheduler is not None:
                    scheduler.step(val_loss_monitor)
            else:
                val_energy_rmse, val_force_rmse = float("nan"), float("nan")

            # History/logging
            self.history["train_energy_rmse"].append(float(train_energy_rmse))
            self.history["train_force_rmse"].append(float(train_force_rmse))
            self.history["test_energy_rmse"].append(float(val_energy_rmse))
            self.history["test_force_rmse"].append(float(val_force_rmse))
            # Learning rate (first param group)
            self.history["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))
            # Timing
            self.history["epoch_times"].append(time.time() - t0)
            self.history["epoch_forward_time"].append(float(getattr(self, "_last_forward_time", 0.0)))
            self.history["epoch_backward_time"].append(float(getattr(self, "_last_backward_time", 0.0)))

            # Checkpointing
            if ckpt_dir is not None and checkpoint_interval > 0:
                if ((epoch + 1) % int(checkpoint_interval)) == 0:
                    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
                    self._save_checkpoint(ckpt_path, epoch, optimizer)
                    if max_checkpoints is not None and max_checkpoints > 0:
                        _rotate_checkpoints(ckpt_dir, max_to_keep=int(max_checkpoints))

        return self.history

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device,
        alpha: float,
        train: bool,
    ) -> Tuple[float, float]:
        """
        Run one epoch over loader and return (energy_rmse, force_rmse).

        Notes
        -----
        - We don't wrap validation with torch.no_grad() because force loss
          uses autograd to compute dE/dG via network; no backward/optimizer
          is called for validation.
        """
        energy_losses: List[float] = []
        force_losses: List[float] = []
        forward_time_total: float = 0.0
        backward_time_total: float = 0.0
        for batch in loader:
            # Energy view tensors
            features = batch["features"].to(device)
            species_indices = batch["species_indices"].to(device)
            n_atoms = batch["n_atoms"].to(device)
            # energy_ref shape aligns with features dtype
            energy_ref = batch["energy_ref"].to(device)

            # Ensure dtype consistency
            if self.dtype == torch.float64:
                features = features.double()
                energy_ref = energy_ref.double()
            else:
                features = features.float()
                energy_ref = energy_ref.float()

            # Convert targets to cohesive energies if configured and E_atomic provided
            if getattr(self, "_energy_target", "cohesive") == "cohesive" and getattr(self, "_E_atomic_by_index", None) is not None:
                per_atom_Ea = self._E_atomic_by_index[species_indices]
                batch_idx = torch.repeat_interleave(
                    torch.arange(len(n_atoms), device=device), n_atoms.long()
                )
                Ea_sum = torch.zeros(len(n_atoms), dtype=energy_ref.dtype, device=device)
                Ea_sum.scatter_add_(0, batch_idx, per_atom_Ea)
                energy_ref = energy_ref - Ea_sum

            t_forward_start = time.perf_counter()

            # Feature normalization (if enabled)
            if self._normalize_features and self._feature_mean is not None and self._feature_std is not None:
                fm = self._feature_mean.to(device=device, dtype=features.dtype)
                fs = torch.clamp(self._feature_std.to(device=device, dtype=features.dtype), min=1e-12)
                features = (features - fm) / fs

            # Energy loss
            E_shift = self._E_shift if self._normalize_energy else 0.0
            E_scaling = self._E_scaling if self._normalize_energy else 1.0
            energy_loss_t, _ = compute_energy_loss(
                features=features,
                energy_ref=energy_ref,
                n_atoms=n_atoms,
                network=self.model,
                species_indices=species_indices,
                E_shift=float(E_shift),
                E_scaling=float(E_scaling),
            )

            # Optional force loss
            force_loss_t: Optional[torch.Tensor] = None
            if alpha > 0.0 and batch["positions_f"] is not None:
                positions_f = batch["positions_f"].to(device)
                forces_ref_f = batch["forces_ref_f"].to(device)
                species_indices_f = batch["species_indices_f"].to(device)
                species_f = batch["species_f"]  # list[str]
                neighbor_info_f = batch["neighbor_info_f"]  # dict of lists

                # dtype
                if self.dtype == torch.float64:
                    positions_f = positions_f.double()
                    forces_ref_f = forces_ref_f.double()
                else:
                    positions_f = positions_f.float()
                    forces_ref_f = forces_ref_f.float()

                force_loss_t, _ = compute_force_loss(
                    positions=positions_f,
                    species=species_f,
                    forces_ref=forces_ref_f,
                    descriptor=self.descriptor,
                    network=self.model,
                    species_indices=species_indices_f,
                    cell=None,
                    pbc=None,
                    E_scaling=float(E_scaling),
                    neighbor_info=neighbor_info_f,
                    chunk_size=None,
                    feature_mean=(self._feature_mean if self._normalize_features else None),
                    feature_std=(self._feature_std if self._normalize_features else None),
                )

            # Combine
            if force_loss_t is None:
                combined = (1.0 - alpha) * energy_loss_t
            else:
                combined = (1.0 - alpha) * energy_loss_t + alpha * force_loss_t

            t_forward_end = time.perf_counter()
            forward_time_total += (t_forward_end - t_forward_start)

            if train and optimizer is not None:
                t_backward_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                combined.backward()
                optimizer.step()
                t_backward_end = time.perf_counter()
                backward_time_total += (t_backward_end - t_backward_start)

            energy_losses.append(float(energy_loss_t.detach().cpu()))
            if force_loss_t is not None:
                force_losses.append(float(force_loss_t.detach().cpu()))

        energy_rmse = float(sum(energy_losses) / max(1, len(energy_losses)))
        force_rmse = float(sum(force_losses) / max(1, len(force_losses))) if force_losses else float("nan")
        # record timing for this epoch
        self._last_forward_time = forward_time_total
        self._last_backward_time = backward_time_total
        return energy_rmse, force_rmse

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
    ):
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": self.history,
            "architecture": self.arch,
            "descriptor_config": _descriptor_config(self.descriptor),
            "best_val_loss": float(self.best_val) if self.best_val is not None else None,
        }
        try:
            torch.save(payload, str(path))
        except Exception as e:
            # Best-effort; do not crash training
            print(f"[WARN] Failed to save checkpoint at {path}: {e}")

    def _load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer):
        try:
            payload = torch.load(path, map_location=self.device)
            self.model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            if "history" in payload and isinstance(payload["history"], dict):
                self.history = payload["history"]
            self.best_val = payload.get("best_val_loss", None)
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint '{path}': {e}")

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
        energies: List[float] = []
        forces_out: List[torch.Tensor] = []

        for st in structures:
            # Build tensors
            positions = torch.from_numpy(st.positions).to(self.device)
            if self.dtype == torch.float64:
                positions = positions.double()
            else:
                positions = positions.float()

            # Featurize and neighbor info
            features, nb_info = self.descriptor.featurize_with_neighbor_info(
                positions, st.species, None, None
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

            # Feature normalization (if enabled)
            if self._feature_mean is not None and self._feature_std is not None and self._normalize_features:
                fm = self._feature_mean.to(device=self.device, dtype=features.dtype)
                fs = torch.clamp(self._feature_std.to(device=self.device, dtype=features.dtype), min=1e-12)
                features = (features - fm) / fs

            # Energies (normalized model output)
            E_atomic = self.model(features, species_indices)
            E_pred_norm = E_atomic.sum()
            # Denormalize to cohesive energy if enabled
            if self._normalize_energy:
                E_coh = float((E_pred_norm / self._E_scaling + self._E_shift * len(st.species)).detach().cpu())
            else:
                E_coh = float(E_pred_norm.detach().cpu())
            energies.append(E_coh)

            if predict_forces:
                # Use semi-analytical gradient path to predict forces
                # Prepare neighbor_info to pass through compute_force_loss
                neighbor_info = {
                    "neighbor_lists": nb_info["neighbor_lists"],
                    "neighbor_vectors": nb_info["neighbor_vectors"],
                }
                # Dummy zeros for forces_ref; we only want predictions
                forces_ref = torch.zeros_like(positions)
                loss_f, forces_pred = compute_force_loss(
                    positions=positions.clone(),
                    species=st.species,
                    forces_ref=forces_ref,
                    descriptor=self.descriptor,
                    network=self.model,
                    species_indices=species_indices,
                    cell=None,
                    pbc=None,
                    E_scaling=float(self._E_scaling),
                    neighbor_info=neighbor_info,
                    chunk_size=None,
                    feature_mean=(self._feature_mean if self._normalize_features else None),
                    feature_std=(self._feature_std if self._normalize_features else None),
                )
                forces_out.append(forces_pred.detach().cpu())

        return energies, (forces_out if predict_forces else None)
