"""
Model export and import utilities for PyTorch-based MLIP training.

This module provides:
- save_model: persist a trained TorchANNPotential with rich metadata
- load_model: reconstruct a TorchANNPotential from a saved file
- export_history: write training history to JSON and optional CSV
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import csv
import json
import torch

from .trainer import TorchANNPotential
from .config import TorchTrainingConfig
from aenet.torch_featurize import ChebyshevDescriptor

Payload = Dict[str, Any]
PathLike = Union[str, Path]

_SCHEMA_VERSION = "1.0"


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower().replace("torch.", "")
    if s == "float64" or s == "double":
        return torch.float64
    if s == "float32" or s == "float":
        return torch.float32
    # Default to float64 for scientific reproducibility
    return torch.float64


def _descriptor_to_config(desc: ChebyshevDescriptor) -> Dict[str, Any]:
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


def _descriptor_from_config(cfg: Dict[str, Any]) -> ChebyshevDescriptor:
    dtype = _dtype_from_str(str(cfg.get("dtype", "float64")))
    device = str(cfg.get("device", "cpu"))
    return ChebyshevDescriptor(
        species=list(cfg["species"]),
        rad_order=int(cfg["rad_order"]),
        rad_cutoff=float(cfg["rad_cutoff"]),
        ang_order=int(cfg["ang_order"]),
        ang_cutoff=float(cfg["ang_cutoff"]),
        min_cutoff=float(cfg.get("min_cutoff", 0.55)),
        device=device,
        dtype=dtype,
    )


def _serialize_training_config(
        cfg: Optional[TorchTrainingConfig]) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    try:
        d = asdict(cfg)
    except Exception:
        # Fallback: best-effort serialization
        d = {k: v for k, v in cfg.__dict__.items()}
    # Normalize method serialization for readability
    method = getattr(cfg, "method", None)
    if method is not None:
        d["method"] = {
            "name": getattr(method, "method_name", "unknown"),
            **{k: v for k, v in method.__dict__.items()},
        }
    return d


def save_model(
    trainer: TorchANNPotential,
    path: PathLike,
    optimizer: Optional[torch.optim.Optimizer] = None,
    training_config: Optional[TorchTrainingConfig] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a trained TorchANNPotential and rich metadata to a file.

    Parameters
    ----------
    trainer : TorchANNPotential
        Trained model wrapper.
    path : str | Path
        Destination file (.pt / .pth).
    optimizer : torch.optim.Optimizer, optional
        If provided, include optimizer state dict.
    training_config : TorchTrainingConfig, optional
        Persisted training configuration.
    extra_metadata : dict, optional
        Any additional JSON-serializable metadata to include.
    """
    out_path = Path(path)
    _ensure_parent(out_path)

    descriptor_cfg = _descriptor_to_config(trainer.descriptor)
    model_state = trainer.model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None

    # Normalization metadata (serialize tensors to lists)
    # Prefer NormalizationManager state; fall back to legacy attrs if present.
    norm = getattr(trainer, "_normalizer", None)
    normalize_features = bool(
        getattr(norm, "normalize_features",
                getattr(trainer, "_normalize_features", False)))
    normalize_energy = bool(
        getattr(norm, "normalize_energy",
                getattr(trainer, "_normalize_energy", False)))
    E_shift = float(getattr(norm, "E_shift",
                            getattr(trainer, "_E_shift", 0.0)))
    E_scaling = float(getattr(norm, "E_scaling",
                              getattr(trainer, "_E_scaling", 1.0)))
    fm = None
    fs = None
    try:
        if (norm is not None
                and getattr(norm, "feature_mean", None) is not None):
            fm = norm.feature_mean.detach().cpu().tolist()
        elif getattr(trainer, "_feature_mean", None) is not None:
            fm = getattr(trainer, "_feature_mean").detach().cpu().tolist()
    except Exception:
        fm = None
    try:
        if norm is not None and getattr(norm, "feature_std", None) is not None:
            fs = norm.feature_std.detach().cpu().tolist()
        elif getattr(trainer, "_feature_std", None) is not None:
            fs = getattr(trainer, "_feature_std").detach().cpu().tolist()
    except Exception:
        fs = None

    # Include feature min/max if available
    fmin = None
    fmax = None
    try:
        if norm is not None and getattr(norm, "feature_min", None) is not None:
            fmin = norm.feature_min.detach().cpu().tolist()
    except Exception:
        fmin = None
    try:
        if norm is not None and getattr(norm, "feature_max", None) is not None:
            fmax = norm.feature_max.detach().cpu().tolist()
    except Exception:
        fmax = None

    norm_meta: Dict[str, Any] = {
        "normalize_features": normalize_features,
        "normalize_energy": normalize_energy,
        "E_shift": E_shift,
        "E_scaling": E_scaling,
        "feature_mean": fm,
        "feature_std": fs,
        "feature_min": fmin,
        "feature_max": fmax,
    }

    payload: Payload = {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "device_hint": str(trainer.device),
        "architecture": trainer.arch,
        "descriptor_config": descriptor_cfg,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "training_history": (dict(trainer.history)
                             if isinstance(trainer.history, dict) else {}),
        "best_val_loss": (float(trainer.best_val)
                          if trainer.best_val is not None else None),
        "training_config": _serialize_training_config(training_config),
        "normalization": norm_meta,
        # Persist energy target and atomic reference
        # energies for prediction semantics
        "energy_target": getattr(trainer, "_energy_target", "cohesive"),
        "E_atomic": getattr(trainer, "_E_atomic", None),
        "extra_metadata": extra_metadata or {},
    }

    try:
        torch.save(payload, str(out_path))
    except Exception as e:
        raise RuntimeError(f"Failed to save model to '{out_path}': {e}") from e


def load_model(path: PathLike) -> Tuple[TorchANNPotential, Dict[str, Any]]:
    """
    Load a TorchANNPotential and metadata from a saved file.

    Parameters
    ----------
    path : str | Path
        Source file produced by save_model().

    Returns
    -------
    trainer : TorchANNPotential
        Reconstructed trainer with weights loaded (on CPU by default).
    metadata : dict
        Additional metadata from the payload
        (descriptor_config, architecture, etc.).
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"No such file: '{in_path}'")

    try:
        payload: Payload = torch.load(
            str(in_path), map_location="cpu", weights_only=False
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from '{in_path}': {e}") from e

    # Build descriptor and trainer
    descriptor_cfg = payload["descriptor_config"]
    descriptor = _descriptor_from_config(descriptor_cfg)

    arch = payload["architecture"]
    trainer = TorchANNPotential(arch=arch, descriptor=descriptor)

    # Load weights into adapter model
    state_dict = payload["model_state_dict"]
    trainer.model.load_state_dict(state_dict)

    # Restore history/best if available
    if "training_history" in payload and isinstance(
            payload["training_history"], dict):
        trainer.history = payload["training_history"]
    trainer.best_val = payload.get("best_val_loss", None)

    # Restore normalization metadata if present
    norm_meta = payload.get("normalization", {}) or {}
    if norm_meta:
        try:
            trainer._normalizer.set_state({
                "normalize_features": norm_meta.get(
                    "normalize_features", True),
                "normalize_energy": norm_meta.get("normalize_energy", True),
                "E_shift": norm_meta.get("E_shift", 0.0),
                "E_scaling": norm_meta.get("E_scaling", 1.0),
                "feature_mean": norm_meta.get("feature_mean"),
                "feature_std": norm_meta.get("feature_std"),
                "feature_min": norm_meta.get("feature_min"),
                "feature_max": norm_meta.get("feature_max"),
            })
        except Exception:
            # Best-effort; prediction still works without
            # normalization restoration
            pass

    # Restore energy target and atomic reference energies if present
    try:
        trainer._energy_target = payload.get("energy_target", "cohesive")
    except Exception:
        # Keep default if not present
        pass
    trainer._E_atomic = payload.get("E_atomic", None)

    # Compose metadata (excluding large tensors)
    meta_keys = [
        "schema_version",
        "timestamp",
        "torch_version",
        "device_hint",
        "architecture",
        "descriptor_config",
        "optimizer_state_dict",
        "training_history",
        "best_val_loss",
        "training_config",
        "extra_metadata",
        "normalization",
        "energy_target",
        "E_atomic",
    ]
    metadata = {k: payload.get(k) for k in meta_keys if k in payload}

    return trainer, metadata


def export_history(
        history: Dict[str, List[float]],
        json_path: PathLike,
        csv_path: Optional[PathLike] = None,
        ) -> None:
    """
    Export training history to JSON and optionally CSV.

    Parameters
    ----------
    history : dict[str, list[float]]
        Training metrics history as produced by trainer.train().
    json_path : str | Path
        Output path for JSON file.
    csv_path : str | Path, optional
        Output path for CSV file. If provided, a CSV will be written.
    """
    # JSON
    json_p = Path(json_path)
    _ensure_parent(json_p)
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # CSV
    if csv_path is not None:
        csv_p = Path(csv_path)
        _ensure_parent(csv_p)

        # Determine max epochs across tracked metrics
        max_len = 0
        for v in history.values():
            if isinstance(v, list):
                max_len = max(max_len, len(v))

        # Column order
        columns = [
            "epoch",
            "train_energy_rmse",
            "test_energy_rmse",
            "train_force_rmse",
            "test_force_rmse",
            "learning_rates",
            "epoch_times",
        ]

        # Write rows
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i in range(max_len):
                row = [i]
                for col in columns[1:]:
                    seq = history.get(col, [])
                    val = (seq[i] if isinstance(seq, list)
                           and i < len(seq) else float("nan"))
                    row.append(val)
                writer.writerow(row)
