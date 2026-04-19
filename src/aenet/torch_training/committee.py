"""Committee training orchestration for the PyTorch backend."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .._optional import is_sphinx_build
from ..io.predict import PredictOut
from ..io.train import TrainOut

try:
    from torch.utils.data import Dataset, Subset
except Exception:  # pragma: no cover - exercised via docs-only fallback test
    if not is_sphinx_build():
        raise

    class Dataset:
        """Minimal torch.utils.data.Dataset stub for Sphinx builds."""

    class Subset(Dataset):
        """Minimal torch.utils.data.Subset stub for Sphinx builds."""

        def __init__(self, dataset=None, indices=None):
            self.dataset = dataset
            self.indices = indices

from ..mlip.ensemble import AenetEnsembleResult, normalize_ensemble_members
from .config import Structure, TorchTrainingConfig
from .dataset import (
    CachedStructureDataset,
    HDF5StructureDataset,
    StructureDataset,
    convert_to_structures,
    train_test_split,
    train_test_split_dataset,
)
from .descriptor_manifest import descriptor_config_from_object
from .model_export import export_history
from .trainer import TorchANNPotential, _resolve_device

_COMMITTEE_SCHEMA_VERSION = "1.0"

__all__ = [
    "TorchCommitteeConfig",
    "TorchCommitteeMemberResult",
    "TorchCommitteeTrainResult",
    "TorchCommitteePredictResult",
    "TorchCommitteePotential",
]


@dataclass
class TorchCommitteeConfig:
    """
    Configuration for committee training orchestration.

    Parameters
    ----------
    num_members : int
        Number of independently trained committee members.
    base_seed : int, optional
        Base run-level seed for deterministic member-seed derivation.
        Member ``i`` receives ``base_seed + i`` unless ``member_seeds`` is
        provided explicitly.
    member_seeds : list[int], optional
        Explicit per-member run seeds. When provided, the list length must
        equal ``num_members`` and overrides ``base_seed``.
    max_parallel : int, optional
        Maximum number of committee members to train concurrently.
        Default: ``1``.
    devices : list[str], optional
        Explicit device assignment pool. Member devices are assigned
        round-robin across this list. When omitted, members inherit
        ``TorchTrainingConfig.device``.
    output_dir : str or Path, optional
        Output directory for committee artifacts. Defaults to
        ``committee_run`` in the current working directory.
    """

    num_members: int
    base_seed: int | None = None
    member_seeds: list[int] | None = None
    max_parallel: int = 1
    devices: list[str] | None = None
    output_dir: os.PathLike | str | None = None

    def __post_init__(self) -> None:
        """Validate committee orchestration parameters."""
        if int(self.num_members) < 1:
            raise ValueError("num_members must be >= 1")
        self.num_members = int(self.num_members)

        if self.base_seed is not None and not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an integer or None")

        if self.member_seeds is not None:
            normalized = [int(seed) for seed in self.member_seeds]
            if len(normalized) != self.num_members:
                raise ValueError(
                    "member_seeds must have length equal to num_members"
                )
            self.member_seeds = normalized

        if int(self.max_parallel) < 1:
            raise ValueError("max_parallel must be >= 1")
        self.max_parallel = int(self.max_parallel)

        if self.devices is not None:
            if len(self.devices) == 0:
                raise ValueError("devices must not be empty when provided")
            self.devices = [str(device) for device in self.devices]


@dataclass
class TorchCommitteeMemberResult:
    """Structured result for one committee member run."""

    member_index: int
    seed: int | None
    split_seed: int | None
    device: str
    member_dir: Path
    model_path: Path | None
    history_json_path: Path | None
    history_csv_path: Path | None
    summary_path: Path | None
    checkpoint_dir: Path | None
    status: str
    metrics: dict[str, float | None]
    error: str | None = None

    @property
    def stats(self) -> dict[str, float]:
        """Return TrainOut-style final metrics for this committee member."""
        return _normalize_member_stats(self.metrics)

    @property
    def trainout(self) -> TrainOut | None:
        """
        Return this member's TrainOut result, loaded from history.json.

        The top-level committee metadata stores compact summaries only.
        Member histories are persisted separately, so this property rebuilds
        the familiar single-network TrainOut view lazily when it is available.
        """
        if self.status != "completed" or self.history_json_path is None:
            return None
        if not self.history_json_path.exists():
            return None
        with self.history_json_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        return TrainOut.from_torch_history(history)

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata view."""
        return {
            "member_index": self.member_index,
            "seed": self.seed,
            "split_seed": self.split_seed,
            "device": self.device,
            "member_dir": str(self.member_dir),
            "model_path": (
                None if self.model_path is None else str(self.model_path)
            ),
            "history_json_path": (
                None
                if self.history_json_path is None
                else str(self.history_json_path)
            ),
            "history_csv_path": (
                None
                if self.history_csv_path is None
                else str(self.history_csv_path)
            ),
            "summary_path": (
                None if self.summary_path is None else str(self.summary_path)
            ),
            "checkpoint_dir": (
                None
                if self.checkpoint_dir is None
                else str(self.checkpoint_dir)
            ),
            "status": self.status,
            "metrics": dict(self.metrics),
            "error": self.error,
        }


@dataclass
class TorchCommitteeTrainResult:
    """Top-level result for one committee training run."""

    output_dir: Path
    metadata_path: Path
    members: list[TorchCommitteeMemberResult]
    execution_mode: str

    @property
    def failed_members(self) -> list[TorchCommitteeMemberResult]:
        """Return any members that failed during training."""
        return [
            member for member in self.members if member.status != "completed"
        ]

    @property
    def completed_members(self) -> list[TorchCommitteeMemberResult]:
        """Return successfully trained committee members."""
        return [
            member for member in self.members if member.status == "completed"
        ]

    def to_dataframe(self):
        """Return one row per member with status, paths, and final metrics."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for member in self.members:
            row: dict[str, Any] = {
                "member": member.member_index,
                "seed": member.seed,
                "split_seed": member.split_seed,
                "device": member.device,
                "status": member.status,
                "model_path": member.model_path,
                "history_json_path": member.history_json_path,
                "history_csv_path": member.history_csv_path,
                "summary_path": member.summary_path,
                "checkpoint_dir": member.checkpoint_dir,
                "error": member.error,
            }
            row.update(member.stats)
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def trainouts(self) -> list[TrainOut]:
        """Return TrainOut objects for completed members with histories."""
        return [
            trainout for member in self.completed_members
            if (trainout := member.trainout) is not None
        ]

    @property
    def stats(self) -> dict[str, dict[str, float | int]]:
        """Return aggregate statistics over completed committee members."""
        values_by_metric: dict[str, list[float]] = {}
        for member in self.completed_members:
            for key, value in member.stats.items():
                numeric = _finite_metric_value(value)
                if numeric is None:
                    continue
                values_by_metric.setdefault(key, []).append(numeric)

        stats: dict[str, dict[str, float | int]] = {}
        for key, values in values_by_metric.items():
            array = np.asarray(values, dtype=np.float64)
            stats[key] = {
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=0)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "n": int(len(values)),
            }
        return stats

    def __str__(self) -> str:
        """Return a TrainOut-like committee statistics summary."""
        lines = ["Training statistics:"]
        aggregate_stats = self.stats
        printed: set[str] = set()

        for key in _COMMITTEE_STATS_PRINT_ORDER:
            if key not in aggregate_stats:
                continue
            entry = aggregate_stats[key]
            lines.append(
                f"  {key}: {_format_mean_std(entry['mean'], entry['std'])}"
            )
            printed.add(key)

        for key in sorted(aggregate_stats):
            if key in printed:
                continue
            entry = aggregate_stats[key]
            lines.append(
                f"  {key}: {_format_mean_std(entry['mean'], entry['std'])}"
            )

        lines.append(f"  completed_members: {len(self.completed_members)}")
        lines.append(f"  failed_members: {len(self.failed_members)}")
        lines.append(f"  execution_mode: {self.execution_mode}")
        lines.append(f"  output_dir: {self.output_dir}")
        return "\n".join(lines) + "\n"


@dataclass
class TorchCommitteePredictResult:
    """List-like committee prediction result with per-member PredictOuts."""

    predictions: list[AenetEnsembleResult]
    member_outputs: list[PredictOut]
    indices: list[int]
    source_indices: list[int | None]
    identifiers: list[str | None]

    def __len__(self) -> int:
        """Return the number of predicted structures."""
        return len(self.predictions)

    def __iter__(self):
        """Iterate over aggregated per-structure ensemble predictions."""
        return iter(self.predictions)

    def __getitem__(self, index):
        """Return one aggregated prediction or a slice of predictions."""
        return self.predictions[index]

    @property
    def energy(self) -> np.ndarray:
        """Reported aggregate energies for all structures."""
        return np.array([prediction.energy for prediction in self.predictions])

    @property
    def energy_mean(self) -> np.ndarray:
        """Mean committee energies for all structures."""
        return np.array([
            prediction.energy_mean for prediction in self.predictions
        ])

    @property
    def energy_std(self) -> np.ndarray:
        """Committee energy standard deviations for all structures."""
        return np.array([
            prediction.energy_std for prediction in self.predictions
        ])

    @property
    def max_force_uncertainty(self) -> np.ndarray:
        """Maximum per-atom force uncertainty for each structure."""
        return np.array([
            np.nan
            if prediction.force_uncertainty is None
            else float(np.max(prediction.force_uncertainty))
            for prediction in self.predictions
        ])

    @property
    def mean_force_uncertainty(self) -> np.ndarray:
        """Mean per-atom force uncertainty for each structure."""
        return np.array([
            np.nan
            if prediction.force_uncertainty is None
            else float(np.mean(prediction.force_uncertainty))
            for prediction in self.predictions
        ])

    def to_dataframe(self):
        """Return one row per predicted structure with uncertainty metrics."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for row_index, prediction in enumerate(self.predictions):
            force_uncertainty = prediction.force_uncertainty
            rows.append(
                {
                    "index": self.indices[row_index],
                    "source_index": self.source_indices[row_index],
                    "identifier": self.identifiers[row_index],
                    "energy": prediction.energy,
                    "energy_mean": prediction.energy_mean,
                    "energy_std": prediction.energy_std,
                    "member_energies": prediction.member_energies,
                    "max_force_uncertainty": (
                        None
                        if force_uncertainty is None
                        else float(np.max(force_uncertainty))
                    ),
                    "mean_force_uncertainty": (
                        None
                        if force_uncertainty is None
                        else float(np.mean(force_uncertainty))
                    ),
                    "num_members": prediction.num_members,
                    "aggregation": prediction.aggregation,
                    "reference_member": prediction.reference_member,
                }
            )
        return pd.DataFrame(rows)

    def sort_by(
        self,
        column: str = "energy_std",
        *,
        ascending: bool = False,
    ):
        """Return ``to_dataframe()`` sorted by one uncertainty column."""
        return self.to_dataframe().sort_values(
            column,
            ascending=ascending,
        )

    def top_uncertain(
        self,
        n: int = 10,
        *,
        metric: str = "energy_std",
    ):
        """Return the top ``n`` most uncertain structures as a DataFrame."""
        return self.sort_by(metric, ascending=False).head(int(n))

    def __str__(self) -> str:
        """Return a compact prediction-result summary."""
        return (
            f"TorchCommitteePredictResult("
            f"num_structures={len(self)}, "
            f"num_members={len(self.member_outputs)})"
        )


@dataclass
class _MemberTask:
    """Concrete member task planned by the parent process."""

    member_index: int
    seed: int | None
    split_seed: int | None
    device: str
    config: TorchTrainingConfig
    member_dir: Path
    model_path: Path
    history_json_path: Path
    history_csv_path: Path
    summary_path: Path
    checkpoint_dir: Path | None
    train_dataset: Dataset
    test_dataset: Dataset | None


def _completed_model_paths(
    members: Sequence[TorchCommitteeMemberResult],
) -> list[Path]:
    """Return the ordered model paths for completed committee members."""
    paths: list[Path] = []
    for member in members:
        if member.status != "completed":
            continue
        if member.model_path is None:
            raise ValueError(
                "Committee metadata references a completed member without a "
                "model path."
            )
        paths.append(Path(member.model_path))
    if not paths:
        raise ValueError("No completed committee members were found.")
    return paths


def _serialize_training_config(config: TorchTrainingConfig) -> dict[str, Any]:
    """Return a JSON-friendly view of a training config."""
    payload = asdict(config)
    method = getattr(config, "method", None)
    if method is not None:
        payload["method"] = {
            "name": getattr(method, "method_name", "unknown"),
            **{
                key: value for key, value in method.__dict__.items()
                if not key.startswith("_")
            },
        }
    if payload.get("checkpoint_dir") is not None:
        payload["checkpoint_dir"] = str(payload["checkpoint_dir"])
    return _jsonify(payload)


def _jsonify(value: Any) -> Any:
    """Convert nested values into JSON-serializable Python types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _dataset_kind(dataset: Dataset | None) -> str | None:
    """Return a compact identifier for the resolved dataset type."""
    if dataset is None:
        return None
    if isinstance(dataset, Subset):
        return f"Subset[{_dataset_kind(dataset.dataset)}]"
    return dataset.__class__.__name__


def _is_builtin_parallel_dataset(dataset: Dataset | None) -> bool:
    """Return whether the dataset is supported for parallel committee runs."""
    if dataset is None:
        return True
    if isinstance(dataset, Subset):
        return _is_builtin_parallel_dataset(dataset.dataset)
    builtin_types = [StructureDataset, CachedStructureDataset]
    if HDF5StructureDataset is not None:
        builtin_types.append(HDF5StructureDataset)
    return isinstance(dataset, tuple(builtin_types))


def _resolve_member_seeds(
    committee_config: TorchCommitteeConfig,
    training_config: TorchTrainingConfig,
) -> list[int | None]:
    """Plan the run-level seed for each committee member."""
    if committee_config.member_seeds is not None:
        return list(committee_config.member_seeds)

    base_seed = committee_config.base_seed
    if base_seed is None:
        base_seed = getattr(training_config, "seed", None)
    if base_seed is None:
        return [None] * committee_config.num_members
    return [int(base_seed) + index for index in range(committee_config.num_members)]


def _resolve_member_devices(
    committee_config: TorchCommitteeConfig,
    training_config: TorchTrainingConfig,
) -> list[str]:
    """Plan the device assignment for each committee member."""
    if committee_config.devices is None:
        inherited = str(_resolve_device(training_config))
        if committee_config.max_parallel > 1 and inherited != "cpu":
            raise ValueError(
                "Parallel committee training on a non-CPU device requires "
                "an explicit devices=[...] list."
            )
        return [inherited] * committee_config.num_members

    return [
        committee_config.devices[index % len(committee_config.devices)]
        for index in range(committee_config.num_members)
    ]


def _final_metric(
    history: dict[str, list[float]],
    key: str,
) -> float | None:
    """Return the last recorded value for a metric key."""
    values = history.get(key, [])
    if not values:
        return None
    return _finite_metric_value(values[-1])


_HISTORY_TO_TRAINOUT_METRICS: tuple[tuple[str, str], ...] = (
    ("final_MAE_train", "train_energy_mae"),
    ("final_RMSE_train", "train_energy_rmse"),
    ("final_MAE_test", "test_energy_mae"),
    ("final_RMSE_test", "test_energy_rmse"),
    ("final_RMSE_force_train", "train_force_rmse"),
    ("final_RMSE_force_test", "test_force_rmse"),
)

_LEGACY_MEMBER_METRIC_ALIASES: dict[str, str] = {
    "train_energy_rmse": "final_RMSE_train",
    "test_energy_rmse": "final_RMSE_test",
    "train_force_rmse": "final_RMSE_force_train",
    "test_force_rmse": "final_RMSE_force_test",
}

_COMMITTEE_STATS_PRINT_ORDER: tuple[str, ...] = (
    "final_MAE_train",
    "final_RMSE_train",
    "final_MAE_test",
    "final_RMSE_test",
    "min_RMSE_test",
    "epoch_min_RMSE_test",
    "final_RMSE_force_train",
    "final_RMSE_force_test",
    "min_RMSE_force_test",
    "epoch_min_RMSE_force_test",
    "best_val_loss",
    "epochs_recorded",
)


def _finite_metric_value(value: Any) -> float | None:
    """Return a finite float metric value, or None for unavailable values."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _format_mean_std(mean: float | int, std: float | int) -> str:
    """Format an aggregate committee metric as mean plus/minus std."""
    return f"{float(mean):.6g} \u00b1 {float(std):.6g}"


def _min_metric(history: dict[str, list[float]], key: str) -> float | None:
    """Return the minimum finite recorded value for a metric key."""
    values = [
        numeric for value in history.get(key, [])
        if (numeric := _finite_metric_value(value)) is not None
    ]
    if not values:
        return None
    return float(min(values))


def _epoch_min_metric(history: dict[str, list[float]], key: str) -> float | None:
    """Return the 1-based epoch of the minimum finite metric value."""
    best_epoch: int | None = None
    best_value: float | None = None
    for epoch, value in enumerate(history.get(key, []), start=1):
        numeric = _finite_metric_value(value)
        if numeric is None:
            continue
        if best_value is None or numeric < best_value:
            best_value = numeric
            best_epoch = epoch
    return None if best_epoch is None else float(best_epoch)


def _normalize_member_stats(
    metrics: dict[str, float | None],
) -> dict[str, float]:
    """Return finite per-member metrics with TrainOut-compatible names."""
    stats: dict[str, float] = {}
    for key, value in metrics.items():
        if (
            key.startswith("final_")
            or key.startswith("min_")
            or key.startswith("epoch_min_")
            or key in {"best_val_loss", "epochs_recorded"}
        ):
            numeric = _finite_metric_value(value)
            if numeric is not None:
                stats[key] = numeric

    for legacy_key, trainout_key in _LEGACY_MEMBER_METRIC_ALIASES.items():
        if trainout_key in stats:
            continue
        numeric = _finite_metric_value(metrics.get(legacy_key))
        if numeric is not None:
            stats[trainout_key] = numeric
    return stats


def _history_summary(history: dict[str, list[float]]) -> dict[str, float | None]:
    """Build the compact per-member metric summary written to metadata."""
    metrics = {
        trainout_key: _final_metric(history, history_key)
        for trainout_key, history_key in _HISTORY_TO_TRAINOUT_METRICS
    }
    metrics.update(
        {
            "min_RMSE_test": _min_metric(history, "test_energy_rmse"),
            "epoch_min_RMSE_test": _epoch_min_metric(
                history,
                "test_energy_rmse",
            ),
            "min_RMSE_force_test": _min_metric(history, "test_force_rmse"),
            "epoch_min_RMSE_force_test": _epoch_min_metric(
                history,
                "test_force_rmse",
            ),
        }
    )
    metrics.update({
        "train_energy_rmse": metrics["final_RMSE_train"],
        "test_energy_rmse": metrics["final_RMSE_test"],
        "train_force_rmse": metrics["final_RMSE_force_train"],
        "test_force_rmse": metrics["final_RMSE_force_test"],
        "best_val_loss": metrics["min_RMSE_test"],
        "epochs_recorded": (
            None
            if len(history.get("train_energy_rmse", [])) == 0
            else float(len(history["train_energy_rmse"]))
        ),
    })
    return metrics


@contextmanager
def _working_directory(path: Path):
    """Temporarily change the process working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_committee_datasets(
    *,
    structures: StructureInput | None,
    dataset: Dataset | None,
    train_dataset: Dataset | None,
    test_dataset: Dataset | None,
    config: TorchTrainingConfig,
    descriptor: Any,
) -> tuple[Dataset, Dataset | None]:
    """Resolve committee inputs into a fixed train/test dataset pair."""
    split_seed = getattr(config, "split_seed", None)

    if train_dataset is not None:
        return train_dataset, test_dataset

    if dataset is not None:
        base_ds = dataset
        if config.testpercent > 0 and test_dataset is None:
            test_fraction = config.testpercent / 100.0
            if train_test_split_dataset is not None:
                return train_test_split_dataset(
                    base_ds,
                    test_fraction=test_fraction,
                    seed=split_seed,
                )

            indices = list(range(len(base_ds)))
            rng = random.Random(split_seed)
            rng.shuffle(indices)
            n_test = int(len(indices) * test_fraction)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
            return Subset(base_ds, train_idx), Subset(base_ds, test_idx)
        return base_ds, test_dataset

    if structures is None:
        raise ValueError(
            "Provide either 'structures' or 'dataset'/'train_dataset'"
        )

    converted = convert_to_structures(structures)
    converted = [
        TorchANNPotential._structure_with_identifier(structure, idx)
        for idx, structure in enumerate(converted)
    ]

    if bool(getattr(config, "cache_features", False)) and float(config.alpha) == 0.0:
        all_structures = converted
        if config.testpercent > 0:
            test_fraction = config.testpercent / 100.0
            indices = list(range(len(all_structures)))
            rng = random.Random(split_seed)
            rng.shuffle(indices)
            n_test = int(len(indices) * test_fraction)
            test_idx = set(indices[:n_test])
            train_idx = set(indices[n_test:])
            train_structures = [all_structures[i] for i in sorted(train_idx)]
            test_structures = [all_structures[i] for i in sorted(test_idx)]
        else:
            train_structures = all_structures
            test_structures = []

        show_progress = bool(getattr(config, "show_progress", True))
        train_ds = CachedStructureDataset(
            structures=train_structures,
            descriptor=descriptor,
            max_energy=config.max_energy,
            max_forces=config.max_forces,
            atomic_energies=config.atomic_energies,
            seed=split_seed,
            show_progress=show_progress,
        )
        test_ds = (
            CachedStructureDataset(
                structures=test_structures,
                descriptor=descriptor,
                max_energy=config.max_energy,
                max_forces=config.max_forces,
                atomic_energies=config.atomic_energies,
                seed=split_seed,
                show_progress=show_progress,
            )
            if config.testpercent > 0 and len(test_structures) > 0
            else None
        )
        return train_ds, test_ds

    full_ds = StructureDataset(
        structures=converted,
        descriptor=descriptor,
        max_energy=config.max_energy,
        max_forces=config.max_forces,
        atomic_energies=config.atomic_energies,
        seed=split_seed,
    )
    if config.testpercent > 0:
        return train_test_split(
            full_ds,
            test_fraction=config.testpercent / 100.0,
            seed=split_seed,
        )
    return full_ds, None


def _member_width(num_members: int) -> int:
    """Return the zero-padding width for member directory names."""
    return max(3, len(str(max(0, int(num_members) - 1))))


def _member_dir(output_dir: Path, member_index: int, width: int) -> Path:
    """Return the stable directory path for one committee member."""
    return output_dir / f"member_{member_index:0{width}d}"


def _make_member_task(
    *,
    member_index: int,
    seed: int | None,
    device: str,
    split_seed: int | None,
    base_config: TorchTrainingConfig,
    train_dataset: Dataset,
    test_dataset: Dataset | None,
    output_dir: Path,
    width: int,
) -> _MemberTask:
    """Create the concrete member task handed to sequential or parallel execution."""
    member_dir = _member_dir(output_dir, member_index, width)
    member_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir: Path | None = None
    if (
        base_config.checkpoint_dir is not None
        or bool(base_config.save_best)
        or int(base_config.checkpoint_interval) > 0
    ):
        checkpoint_dir = member_dir / "checkpoints"

    member_config = replace(
        base_config,
        seed=seed,
        device=device,
        checkpoint_dir=(
            None if checkpoint_dir is None else str(checkpoint_dir)
        ),
    )

    return _MemberTask(
        member_index=member_index,
        seed=seed,
        split_seed=split_seed,
        device=device,
        config=member_config,
        member_dir=member_dir,
        model_path=member_dir / "model.pt",
        history_json_path=member_dir / "history.json",
        history_csv_path=member_dir / "history.csv",
        summary_path=member_dir / "summary.json",
        checkpoint_dir=checkpoint_dir,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )


def _run_member_task(
    *,
    arch: dict[str, list[tuple[int, str]]],
    descriptor: Any,
    task: _MemberTask,
) -> dict[str, Any]:
    """Train one committee member and persist its artifacts."""
    task.member_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = task.checkpoint_dir
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        with _working_directory(task.member_dir):
            trainer = TorchANNPotential(arch=arch, descriptor=descriptor)
            trainer.train(
                train_dataset=task.train_dataset,
                test_dataset=task.test_dataset,
                config=task.config,
            )
            trainer.save(
                task.model_path,
                extra_metadata={
                    "committee_member_index": task.member_index,
                    "committee_seed": task.seed,
                    "committee_split_seed": task.split_seed,
                },
            )
            export_history(
                trainer.history,
                task.history_json_path,
                task.history_csv_path,
            )

            metrics = _history_summary(trainer.history)
            metrics["best_val_loss"] = _finite_metric_value(trainer.best_val)
            summary_payload = {
                "member_index": task.member_index,
                "seed": task.seed,
                "split_seed": task.split_seed,
                "device": task.device,
                "status": "completed",
                "metrics": metrics,
                "model_path": str(task.model_path),
                "history_json_path": str(task.history_json_path),
                "history_csv_path": str(task.history_csv_path),
                "checkpoint_dir": (
                    None
                    if checkpoint_dir is None
                    else str(checkpoint_dir)
                ),
            }
            with task.summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary_payload, handle, indent=2)

        return {
            "member_index": task.member_index,
            "seed": task.seed,
            "split_seed": task.split_seed,
            "device": task.device,
            "member_dir": str(task.member_dir),
            "model_path": str(task.model_path),
            "history_json_path": str(task.history_json_path),
            "history_csv_path": str(task.history_csv_path),
            "summary_path": str(task.summary_path),
            "checkpoint_dir": (
                None if checkpoint_dir is None else str(checkpoint_dir)
            ),
            "status": "completed",
            "metrics": metrics,
            "error": None,
        }
    except Exception as exc:
        failure = {
            "member_index": task.member_index,
            "seed": task.seed,
            "split_seed": task.split_seed,
            "device": task.device,
            "member_dir": str(task.member_dir),
            "model_path": None,
            "history_json_path": None,
            "history_csv_path": None,
            "summary_path": None,
            "checkpoint_dir": (
                None if checkpoint_dir is None else str(checkpoint_dir)
            ),
            "status": "failed",
            "metrics": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
        if task.summary_path.parent.exists():
            with task.summary_path.open("w", encoding="utf-8") as handle:
                json.dump(failure, handle, indent=2)
            failure["summary_path"] = str(task.summary_path)
        return failure


def _task_result_from_payload(payload: dict[str, Any]) -> TorchCommitteeMemberResult:
    """Convert a worker payload into the public member-result dataclass."""
    return TorchCommitteeMemberResult(
        member_index=int(payload["member_index"]),
        seed=payload.get("seed"),
        split_seed=payload.get("split_seed"),
        device=str(payload["device"]),
        member_dir=Path(payload["member_dir"]),
        model_path=(
            None
            if payload.get("model_path") is None
            else Path(payload["model_path"])
        ),
        history_json_path=(
            None
            if payload.get("history_json_path") is None
            else Path(payload["history_json_path"])
        ),
        history_csv_path=(
            None
            if payload.get("history_csv_path") is None
            else Path(payload["history_csv_path"])
        ),
        summary_path=(
            None
            if payload.get("summary_path") is None
            else Path(payload["summary_path"])
        ),
        checkpoint_dir=(
            None
            if payload.get("checkpoint_dir") is None
            else Path(payload["checkpoint_dir"])
        ),
        status=str(payload["status"]),
        metrics=dict(payload.get("metrics", {})),
        error=payload.get("error"),
    )


StructureInput = list[Structure] | list[Any] | list[os.PathLike]


def _flatten_dataset_indices(dataset: Dataset) -> tuple[Dataset, list[int] | None]:
    """Resolve nested Subset wrappers to a root dataset and root indices."""
    current: Dataset = dataset
    index_map: list[int] | None = None

    while isinstance(current, Subset):
        current_indices = [int(index) for index in current.indices]
        if index_map is None:
            index_map = current_indices
        else:
            index_map = [current_indices[index] for index in index_map]
        current = current.dataset

    return current, index_map


def _dataset_identifier(
    dataset: Dataset,
    index: int,
) -> str | None:
    """Return the preferred identifier for one dataset entry, if available."""
    getter = getattr(dataset, "get_structure_identifier", None)
    if callable(getter):
        identifier = getter(index)
        if identifier not in (None, ""):
            return str(identifier)

    get_structure = getattr(dataset, "get_structure", None)
    if callable(get_structure):
        structure = get_structure(index)
        name = getattr(structure, "name", None)
        if name not in (None, ""):
            return str(name)

    structures = getattr(dataset, "structures", None)
    if structures is not None:
        name = getattr(structures[index], "name", None)
        if name not in (None, ""):
            return str(name)

    return None


def _prediction_metadata_from_dataset(
    dataset: Dataset,
) -> tuple[list[int], list[int | None], list[str | None]]:
    """Return local indices, root/source indices, and identifiers."""
    root_dataset, root_indices = _flatten_dataset_indices(dataset)
    indices = list(range(len(dataset)))
    source_indices = (
        [int(index) for index in root_indices]
        if root_indices is not None
        else list(indices)
    )
    identifiers = [
        _dataset_identifier(root_dataset, source_index)
        for source_index in source_indices
    ]
    return indices, source_indices, identifiers


def _prediction_metadata_from_structures(
    structures: StructureInput,
    first_output: PredictOut,
) -> tuple[list[int], list[int | None], list[str | None]]:
    """Return local indices and best-effort identifiers for raw inputs."""
    indices = list(range(first_output.num_structures))
    source_indices = [None] * len(indices)

    if first_output.structure_paths is not None:
        identifiers = [str(path) for path in first_output.structure_paths]
        return indices, source_indices, identifiers

    identifiers: list[str | None] = []
    for item in structures:
        name = getattr(item, "name", None)
        if name in (None, ""):
            identifiers.append(None)
        else:
            identifiers.append(str(name))

    if len(identifiers) != len(indices):
        identifiers = [None] * len(indices)
    return indices, source_indices, identifiers


def _committee_predict_result_from_member_outputs(
    member_outputs: list[PredictOut],
    *,
    indices: list[int],
    source_indices: list[int | None],
    identifiers: list[str | None],
    eval_forces: bool,
    aggregation: str,
    reference_member: int,
) -> TorchCommitteePredictResult:
    """Aggregate member PredictOut objects into a committee result."""
    if not member_outputs:
        return TorchCommitteePredictResult(
            predictions=[],
            member_outputs=[],
            indices=[],
            source_indices=[],
            identifiers=[],
        )

    num_structures = len(member_outputs[0].total_energy)
    for output in member_outputs[1:]:
        if len(output.total_energy) != num_structures:
            raise RuntimeError(
                "Committee members returned different numbers of "
                "predictions."
            )

    if not (
        len(indices) == len(source_indices) == len(identifiers) == num_structures
    ):
        raise RuntimeError(
            "Committee prediction metadata length does not match predictions."
        )

    predictions: list[AenetEnsembleResult] = []
    for structure_index in range(num_structures):
        member_energies = [
            float(output.total_energy[structure_index])
            for output in member_outputs
        ]
        member_forces = None
        if eval_forces:
            member_forces = [
                np.asarray(output.forces[structure_index], dtype=np.float64)
                for output in member_outputs
            ]
        predictions.append(
            AenetEnsembleResult.from_member_predictions(
                member_energies=member_energies,
                member_forces=member_forces,
                aggregation=aggregation,
                reference_member=reference_member,
            )
        )

    return TorchCommitteePredictResult(
        predictions=predictions,
        member_outputs=member_outputs,
        indices=indices,
        source_indices=source_indices,
        identifiers=identifiers,
    )


def _resolve_metadata_path(path: os.PathLike | str) -> Path:
    """Resolve a committee metadata file path from a file or directory input."""
    resolved = Path(path).resolve()
    if resolved.is_dir():
        resolved = resolved / "committee_metadata.json"
    return resolved


def _load_train_result_from_metadata(
    path: os.PathLike | str,
) -> TorchCommitteeTrainResult:
    """Load the saved top-level committee result metadata."""
    metadata_path = _resolve_metadata_path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No committee metadata file found at '{metadata_path}'."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    members = [
        _task_result_from_payload(member_payload)
        for member_payload in payload.get("members", [])
    ]
    return TorchCommitteeTrainResult(
        output_dir=metadata_path.parent,
        metadata_path=metadata_path,
        members=members,
        execution_mode=str(payload.get("execution_mode", "unknown")),
    )


def _assert_compatible_loaded_members(
    members: Sequence[TorchANNPotential],
    model_paths: Sequence[Path],
) -> None:
    """Validate that loaded member models can form one committee."""
    if not members:
        raise ValueError("At least one committee member model is required.")

    reference_arch = members[0].arch
    reference_descriptor = descriptor_config_from_object(members[0].descriptor)
    reference_atomic_energies = getattr(members[0], "_atomic_energies", None)

    for index, member in enumerate(members[1:], start=1):
        if member.arch != reference_arch:
            raise ValueError(
                "All committee member models must share the same architecture. "
                f"Model 0 ('{model_paths[0]}') and model {index} "
                f"('{model_paths[index]}') differ."
            )
        descriptor_cfg = descriptor_config_from_object(member.descriptor)
        if descriptor_cfg != reference_descriptor:
            raise ValueError(
                "All committee member models must share the same descriptor "
                f"configuration. Model {index} ('{model_paths[index]}') "
                "is incompatible with model 0."
            )
        if getattr(member, "_atomic_energies", None) != reference_atomic_energies:
            raise ValueError(
                "All committee member models must share the same "
                "atomic-reference energy mapping."
            )


def _member_manifest_from_ascii_paths(
    paths: Sequence[os.PathLike | str],
    *,
    prefix: str,
) -> dict[str, str]:
    """Convert one member's ASCII-exported files into an ensemble manifest."""
    manifest: dict[str, str] = {}
    expected_prefix = f"{prefix}."
    expected_suffix = ".nn.ascii"

    for path in paths:
        resolved = Path(path).resolve()
        name = resolved.name
        if not name.startswith(expected_prefix) or not name.endswith(expected_suffix):
            raise ValueError(
                "Unexpected ASCII export filename. Expected files like "
                f"'{prefix}.SPECIES.nn.ascii', got '{name}'."
            )
        species = name[len(expected_prefix) : -len(expected_suffix)]
        if len(species) == 0:
            raise ValueError(
                f"Could not infer species from exported ASCII file '{name}'."
            )
        manifest[species] = str(resolved)
    return manifest


class TorchCommitteePotential:
    """
    Committee-training orchestration built on top of ``TorchANNPotential``.

    Parameters
    ----------
    arch : dict[str, list[tuple[int, str]]]
        Shared per-species network architecture for all members.
    descriptor : Any
        Shared descriptor instance for all members.
    """

    def __init__(self, arch: dict[str, list[tuple[int, str]]], descriptor: Any):
        self.arch = arch
        self.descriptor = descriptor
        self.last_result: TorchCommitteeTrainResult | None = None
        self._loaded_members: list[TorchANNPotential] = []
        self._loaded_member_paths: list[Path] = []

    @classmethod
    def from_files(
        cls,
        model_paths: Sequence[os.PathLike | str],
        device: str | None = None,
    ) -> TorchCommitteePotential:
        """
        Build a committee from one or more saved member model files.

        Parameters
        ----------
        model_paths : sequence[str | Path]
            Saved ``model.pt`` files produced by ``TorchANNPotential.save()``
            or by committee training.
        device : str, optional
            Device used when loading the saved member models.

        Returns
        -------
        TorchCommitteePotential
            Committee wrapper with the loaded members cached for inference
            and export.
        """
        resolved_paths = [Path(path).resolve() for path in model_paths]
        if not resolved_paths:
            raise ValueError("At least one model path must be provided.")

        loaded_members = [
            TorchANNPotential.from_file(path, device=device)
            for path in resolved_paths
        ]
        _assert_compatible_loaded_members(loaded_members, resolved_paths)

        committee = cls(
            arch=loaded_members[0].arch,
            descriptor=loaded_members[0].descriptor,
        )
        committee._loaded_members = loaded_members
        committee._loaded_member_paths = resolved_paths
        return committee

    @classmethod
    def from_directory(
        cls,
        path: os.PathLike | str,
        device: str | None = None,
    ) -> TorchCommitteePotential:
        """
        Build a committee from a saved committee output directory.

        Parameters
        ----------
        path : str | Path
            Committee output directory or the
            ``committee_metadata.json`` file inside it.
        device : str, optional
            Device used when loading the saved member models.

        Returns
        -------
        TorchCommitteePotential
            Committee wrapper with member models loaded from the saved
            committee run.
        """
        result = _load_train_result_from_metadata(path)
        committee = cls.from_files(
            _completed_model_paths(result.members),
            device=device,
        )
        committee.last_result = result
        return committee

    def load_members(
        self,
        device: str | None = None,
    ) -> list[TorchANNPotential]:
        """
        Load committee member models from the latest saved committee result.

        Parameters
        ----------
        device : str, optional
            Device used when loading the saved member models. When omitted,
            models are loaded on CPU.

        Returns
        -------
        list[TorchANNPotential]
            Loaded member trainers in committee order.
        """
        if self._loaded_members:
            return list(self._loaded_members)

        model_paths = self.member_model_paths()
        loaded_members = [
            TorchANNPotential.from_file(path, device=device)
            for path in model_paths
        ]
        _assert_compatible_loaded_members(loaded_members, model_paths)
        self._loaded_members = loaded_members
        self._loaded_member_paths = model_paths
        return list(self._loaded_members)

    def member_model_paths(self) -> list[Path]:
        """
        Return the saved model paths for the current committee members.

        Returns
        -------
        list[pathlib.Path]
            Saved model paths in member order.
        """
        if self._loaded_member_paths:
            return list(self._loaded_member_paths)
        if self.last_result is None:
            raise ValueError(
                "No committee member models are available. Call train(), "
                "from_directory(), or from_files() first."
            )
        paths = _completed_model_paths(self.last_result.members)
        self._loaded_member_paths = paths
        return list(self._loaded_member_paths)

    def predict(
        self,
        structures: StructureInput,
        eval_forces: bool = False,
        config: Any | None = None,
        aggregation: str = "mean",
        reference_member: int = 0,
    ) -> list[AenetEnsembleResult]:
        """
        Predict committee energies and uncertainties for one or more structures.

        Parameters
        ----------
        structures : list
            Same structure inputs accepted by
            :meth:`aenet.torch_training.TorchANNPotential.predict`.
        eval_forces : bool, optional
            If True, also aggregate atomic-force predictions.
        config : PredictionConfig, optional
            Prediction configuration passed through to each member.
        aggregation : {"mean", "reference"}, optional
            Aggregation mode for the reported energy and forces.
        reference_member : int, optional
            Reference member index used when
            ``aggregation='reference'``.

        Returns
        -------
        TorchCommitteePredictResult
            List-like committee result containing aggregate per-structure
            predictions and the per-member ``PredictOut`` objects.
        """
        members = self.load_members()
        member_outputs = [
            member.predict(
                structures=structures,
                eval_forces=eval_forces,
                config=config,
            )
            for member in members
        ]
        indices, source_indices, identifiers = _prediction_metadata_from_structures(
            structures,
            member_outputs[0],
        )
        return _committee_predict_result_from_member_outputs(
            member_outputs,
            indices=indices,
            source_indices=source_indices,
            identifiers=identifiers,
            eval_forces=eval_forces,
            aggregation=aggregation,
            reference_member=reference_member,
        )

    def predict_dataset(
        self,
        dataset: Dataset,
        eval_forces: bool = False,
        config: Any | None = None,
        aggregation: str = "mean",
        reference_member: int = 0,
    ) -> TorchCommitteePredictResult:
        """
        Predict committee energies and uncertainties for a dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset accepted by ``TorchANNPotential.predict_dataset``.
            ``Subset`` wrappers are supported and source indices are tracked
            back to the root dataset.
        eval_forces : bool, optional
            Dataset-backed force prediction is currently limited by
            ``TorchANNPotential.predict_dataset`` and is not implemented.
        config : PredictionConfig, optional
            Prediction configuration passed through to each member.
        aggregation : {"mean", "reference"}, optional
            Aggregation mode for the reported energy and forces.
        reference_member : int, optional
            Reference member index used when
            ``aggregation='reference'``.

        Returns
        -------
        TorchCommitteePredictResult
            List-like committee result containing aggregate per-structure
            predictions, per-member ``PredictOut`` objects, and dataset
            indices/identifiers.
        """
        members = self.load_members()
        member_outputs = [
            member.predict_dataset(
                dataset=dataset,
                eval_forces=eval_forces,
                config=config,
            )
            for member in members
        ]
        indices, source_indices, identifiers = _prediction_metadata_from_dataset(
            dataset
        )
        return _committee_predict_result_from_member_outputs(
            member_outputs,
            indices=indices,
            source_indices=source_indices,
            identifiers=identifiers,
            eval_forces=eval_forces,
            aggregation=aggregation,
            reference_member=reference_member,
        )

    def to_aenet_ascii(
        self,
        output_dir: os.PathLike | str,
        prefix: str = "potential",
        descriptor_stats: dict[str, Any] | None = None,
        structures: list[Structure] | None = None,
        compute_stats: bool = True,
    ) -> list[dict[str, str]]:
        """
        Export all committee members to aenet ``.nn.ascii`` files.

        Parameters
        ----------
        output_dir : str | Path
            Destination directory for the exported committee.
        prefix : str, optional
            Filename prefix passed through to each member export.
        descriptor_stats : dict, optional
            Pre-computed descriptor statistics shared across all members.
        structures : list[Structure], optional
            Structures used to compute exact descriptor statistics.
        compute_stats : bool, optional
            Whether descriptor statistics should be computed when they are
            not provided explicitly.

        Returns
        -------
        list[dict[str, str]]
            Per-member species-to-file mapping compatible with
            ``AenetEnsembleInterface`` and ``AenetEnsembleCalculator``.
        """
        members = self.load_members()
        output_root = Path(output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        width = _member_width(len(members))

        manifest: list[dict[str, str]] = []
        for index, member in enumerate(members):
            member_output_dir = _member_dir(output_root, index, width)
            ascii_paths = member.to_aenet_ascii(
                output_dir=member_output_dir,
                prefix=prefix,
                descriptor_stats=descriptor_stats,
                structures=structures,
                compute_stats=compute_stats,
            )
            manifest.append(
                _member_manifest_from_ascii_paths(
                    ascii_paths,
                    prefix=prefix,
                )
            )
        return normalize_ensemble_members(manifest)

    def _write_metadata(
        self,
        *,
        metadata_path: Path,
        committee_config: TorchCommitteeConfig,
        training_config: TorchTrainingConfig,
        members: list[TorchCommitteeMemberResult],
        execution_mode: str,
        train_dataset: Dataset,
        test_dataset: Dataset | None,
    ) -> None:
        """Persist the top-level committee metadata manifest."""
        payload = {
            "schema_version": _COMMITTEE_SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "execution_mode": execution_mode,
            "output_dir": str(metadata_path.parent),
            "committee_config": {
                **_jsonify(asdict(committee_config)),
                "output_dir": (
                    str(committee_config.output_dir)
                    if committee_config.output_dir is not None
                    else None
                ),
            },
            "training_config": _serialize_training_config(training_config),
            "architecture": self.arch,
            "descriptor_config": descriptor_config_from_object(self.descriptor),
            "resolved_train_dataset": _dataset_kind(train_dataset),
            "resolved_test_dataset": _dataset_kind(test_dataset),
            "members": [member.to_metadata() for member in members],
            "failures": [
                member.to_metadata()
                for member in members
                if member.status != "completed"
            ],
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def train(
        self,
        structures: StructureInput | None = None,
        dataset: Dataset | None = None,
        train_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        config: TorchTrainingConfig | None = None,
        committee_config: TorchCommitteeConfig | None = None,
    ) -> TorchCommitteeTrainResult:
        """
        Train multiple committee members with shared training settings.

        Parameters
        ----------
        structures, dataset, train_dataset, test_dataset
            Same top-level data inputs accepted by
            :meth:`aenet.torch_training.TorchANNPotential.train`.
        config : TorchTrainingConfig, optional
            Shared single-model training configuration used as the baseline
            for all committee members.
        committee_config : TorchCommitteeConfig, optional
            Committee orchestration parameters.

        Returns
        -------
        TorchCommitteeTrainResult
            Structured result containing per-member artifacts and the
            top-level metadata path.
        """
        if config is None:
            config = TorchTrainingConfig()
        if committee_config is None:
            committee_config = TorchCommitteeConfig(num_members=1)

        output_dir = Path(
            committee_config.output_dir
            if committee_config.output_dir is not None
            else "committee_run"
        ).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "committee_metadata.json"

        resolved_train_dataset, resolved_test_dataset = _resolve_committee_datasets(
            structures=structures,
            dataset=dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=config,
            descriptor=self.descriptor,
        )

        member_seeds = _resolve_member_seeds(committee_config, config)
        member_devices = _resolve_member_devices(committee_config, config)
        width = _member_width(committee_config.num_members)

        tasks = [
            _make_member_task(
                member_index=index,
                seed=member_seeds[index],
                device=member_devices[index],
                split_seed=getattr(config, "split_seed", None),
                base_config=config,
                train_dataset=resolved_train_dataset,
                test_dataset=resolved_test_dataset,
                output_dir=output_dir,
                width=width,
            )
            for index in range(committee_config.num_members)
        ]

        execution_mode = (
            "parallel"
            if committee_config.max_parallel > 1 and committee_config.num_members > 1
            else "sequential"
        )

        if execution_mode == "parallel":
            if not _is_builtin_parallel_dataset(resolved_train_dataset):
                raise ValueError(
                    "Parallel committee training only supports the built-in "
                    "StructureDataset, CachedStructureDataset, "
                    "HDF5StructureDataset, or Subset wrappers around them."
                )
            if not _is_builtin_parallel_dataset(resolved_test_dataset):
                raise ValueError(
                    "Parallel committee training only supports the built-in "
                    "test-dataset workflows documented by the PyTorch backend."
                )

        member_results: list[TorchCommitteeMemberResult] = []
        if execution_mode == "sequential":
            for task in tasks:
                payload = _run_member_task(
                    arch=self.arch,
                    descriptor=self.descriptor,
                    task=task,
                )
                member_results.append(_task_result_from_payload(payload))
        else:
            max_workers = min(committee_config.max_parallel, len(tasks))
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
            ) as executor:
                futures = [
                    executor.submit(
                        _run_member_task,
                        arch=self.arch,
                        descriptor=self.descriptor,
                        task=task,
                    )
                    for task in tasks
                ]
                payloads = [future.result() for future in futures]
            payloads.sort(key=lambda payload: int(payload["member_index"]))
            member_results = [
                _task_result_from_payload(payload) for payload in payloads
            ]

        self._write_metadata(
            metadata_path=metadata_path,
            committee_config=committee_config,
            training_config=config,
            members=member_results,
            execution_mode=execution_mode,
            train_dataset=resolved_train_dataset,
            test_dataset=resolved_test_dataset,
        )

        result = TorchCommitteeTrainResult(
            output_dir=output_dir,
            metadata_path=metadata_path,
            members=member_results,
            execution_mode=execution_mode,
        )
        self.last_result = result
        self._loaded_members = []
        self._loaded_member_paths = []

        failed_members = result.failed_members
        if failed_members:
            raise RuntimeError(
                "Committee training failed for member(s): "
                + ", ".join(
                    f"{member.member_index} ({member.error})"
                    for member in failed_members
                )
            )
        return result
