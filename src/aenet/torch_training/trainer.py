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
- Memory modes: 'cpu' and 'gpu' are supported; 'mixed' is reserved for a
  future real mixed-memory mode and currently raises NotImplementedError
"""

from __future__ import annotations

import atexit
import math
import os
import random
import time
import warnings
from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

# Progress bar (match aenet.mlip behavior)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

# Refactored modules
from ._materialization import referenced_energy_per_atom
from .builders import NetworkBuilder, OptimizerBuilder
from .config import Structure, TorchTrainingConfig
from .dataset import (
    HDF5StructureDataset,
    StructureDataset,
    train_test_split,
    train_test_split_dataset,
)
from .inference import Predictor
from .model_adapter import EnergyModelAdapter
from .training import (
    CheckpointManager,
    MetricsTracker,
    NormalizationManager,
    TrainingLoop,
)
from .training import training_loop as training_loop_mod


def _resolve_device(config: TorchTrainingConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_SMALL_VALIDATION_WARNING_THRESHOLD = 10


def _warn_on_small_validation_set(
    *,
    n_val: int,
    use_scheduler: bool,
    save_best: bool,
) -> None:
    """
    Warn when validation-driven controls are enabled on a tiny split.

    Parameters
    ----------
    n_val : int
        Number of validation structures.
    use_scheduler : bool
        Whether ReduceLROnPlateau monitoring is enabled for this run.
    save_best : bool
        Whether best-checkpoint selection is enabled for this run.
    """
    if n_val <= 0 or n_val >= _SMALL_VALIDATION_WARNING_THRESHOLD:
        return

    noun = "structure" if n_val == 1 else "structures"

    if use_scheduler:
        warnings.warn(
            "use_scheduler=True with a validation set of only "
            f"{n_val} {noun} can make ReduceLROnPlateau react to noisy "
            "metrics. Consider use_scheduler=False, a larger validation "
            "split, or an explicit train/test split.",
            UserWarning,
        )

    if save_best:
        warnings.warn(
            "save_best=True with a validation set of only "
            f"{n_val} {noun} can select a checkpoint from a noisy "
            "validation loss. Consider save_best=False, a larger "
            "validation split, or an explicit train/test split.",
            UserWarning,
        )


def _cohesive_energy_per_atom_for_sampling(
    structure: Structure,
    *,
    atomic_energies: Dict[str, float],
) -> float:
    """
    Return the referenced cohesive or formation energy per atom.

    Sampling policies should use the same per-atom energy semantics as
    training targets, namely total energy minus atomic reference energies,
    normalized by atom count.
    """
    try:
        return referenced_energy_per_atom(
            structure,
            atomic_energies=atomic_energies,
        )
    except Exception as exc:
        raise ValueError(
            "sampling_policy='energy_weighted' requires finite structure "
            "energies."
        ) from exc


def _compute_energy_sampling_weights(
    dataset: Dataset,
    *,
    atomic_energies: Dict[str, float],
) -> torch.Tensor:
    """
    Build deterministic positive structure weights from per-atom energies.

    Lower referenced cohesive or formation energies per atom receive larger
    weights. The scaling is normalized to the training-split energy range so
    the policy remains stable across small and large datasets.
    """
    if not hasattr(dataset, "get_structure"):
        raise TypeError(
            "sampling_policy='energy_weighted' requires a dataset exposing "
            "get_structure(idx)."
        )

    per_atom_energies = [
        _cohesive_energy_per_atom_for_sampling(
            dataset.get_structure(idx),  # type: ignore[attr-defined]
            atomic_energies=atomic_energies,
        )
        for idx in range(len(dataset))
    ]

    if not per_atom_energies:
        raise ValueError(
            "sampling_policy='energy_weighted' requires at least one "
            "training structure."
        )

    min_energy = min(per_atom_energies)
    deltas = [energy - min_energy for energy in per_atom_energies]
    max_delta = max(deltas)
    if max_delta <= 0.0:
        return torch.ones(len(per_atom_energies), dtype=torch.double)

    weights = [
        1.0 / (1.0 + (delta / max_delta))
        for delta in deltas
    ]
    return torch.tensor(weights, dtype=torch.double)


def _compute_error_sampling_weights(
    scores: torch.Tensor,
) -> torch.Tensor:
    """
    Convert non-negative structure scores into positive sampling weights.

    The resulting weights are proportional to the most recent structure
    scores and normalized to unit mean so epoch length remains unchanged
    while higher-loss structures are drawn more often.
    """
    if scores.numel() == 0:
        raise ValueError(
            "sampling_policy='error_weighted' requires at least one "
            "training structure."
        )

    weights = scores.detach().to(dtype=torch.double, device="cpu").clone()
    invalid = ~torch.isfinite(weights)
    if torch.any(invalid):
        weights[invalid] = 0.0
    weights.clamp_(min=0.0)

    if torch.count_nonzero(weights) == 0:
        return torch.ones_like(weights, dtype=torch.double)

    weights.clamp_(min=1e-12)
    mean_weight = float(weights.mean().item())
    if not math.isfinite(mean_weight) or mean_weight <= 0.0:
        return torch.ones_like(weights, dtype=torch.double)
    return weights / mean_weight


def _warn_if_max_energy_is_ignored_for_prebuilt_datasets(
    *,
    config: TorchTrainingConfig,
    dataset: Dataset | None,
    train_dataset: Dataset | None,
    test_dataset: Dataset | None,
) -> None:
    """Warn when ``config.max_energy`` cannot affect prebuilt datasets."""
    if getattr(config, "max_energy", None) is None:
        return
    if dataset is None and train_dataset is None and test_dataset is None:
        return
    warnings.warn(
        "TorchTrainingConfig.max_energy is ignored when using prebuilt "
        "dataset objects. Apply energy filtering when constructing the "
        "dataset instead.",
        UserWarning,
    )


class _ErrorWeightedSamplingState:
    """Mutable trainer-owned state for adaptive error-weighted sampling."""

    def __init__(self, num_structures: int) -> None:
        if num_structures <= 0:
            raise ValueError(
                "sampling_policy='error_weighted' requires at least one "
                "training structure."
            )
        self.scores = torch.ones(num_structures, dtype=torch.double)

    def current_weights(self) -> torch.Tensor:
        """Return the current normalized sampling weights."""
        return _compute_error_sampling_weights(self.scores)

    def update_from_epoch(
        self,
        structure_scores: dict[int, float],
    ) -> torch.Tensor:
        """Update observed scores and return normalized sampler weights."""
        for idx, score in structure_scores.items():
            if idx < 0 or idx >= len(self.scores):
                continue
            try:
                score_value = float(score)
            except Exception:
                continue
            if not math.isfinite(score_value) or score_value < 0.0:
                continue
            self.scores[idx] = score_value
        return self.current_weights()


def _build_training_sampler(
    dataset: Dataset,
    *,
    config: TorchTrainingConfig,
    atomic_energies: Dict[str, float],
) -> tuple[WeightedRandomSampler | None, _ErrorWeightedSamplingState | None]:
    """Return the training sampler implied by ``config.sampling_policy``."""
    sampling_policy = str(getattr(config, "sampling_policy", "uniform"))
    if sampling_policy == "uniform":
        return None, None
    if sampling_policy == "energy_weighted":
        weights = _compute_energy_sampling_weights(
            dataset,
            atomic_energies=atomic_energies,
        )
        return (
            WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            ),
            None,
        )
    if sampling_policy == "error_weighted":
        adaptive_state = _ErrorWeightedSamplingState(len(dataset))
        weights = adaptive_state.current_weights()
        return (
            WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            ),
            adaptive_state,
        )
    raise ValueError(
        "sampling_policy must be 'uniform', 'energy_weighted', or "
        "'error_weighted', "
        f"got '{sampling_policy}'"
    )


class _RuntimeCacheState:
    """Split-local runtime caches owned by the trainer policy wrapper."""

    def __init__(
        self,
        *,
        feature_max_entries: int | None,
        neighbor_max_entries: int | None,
        graph_max_entries: int | None,
    ) -> None:
        self.feature_cache = _BoundedCache(max_entries=feature_max_entries)
        self.neighbor_cache = _BoundedCache(max_entries=neighbor_max_entries)
        self.graph_cache = _BoundedCache(max_entries=graph_max_entries)


class _BoundedCache:
    """Small LRU cache with an optional maximum entry count."""

    def __init__(self, *, max_entries: int | None) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[int, Any] = OrderedDict()

    def __contains__(self, key: int) -> bool:
        return key in self._entries

    def __getitem__(self, key: int) -> Any:
        value = self._entries[key]
        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key: int, value: Any) -> None:
        if self.max_entries == 0:
            return
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        if self.max_entries is not None:
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: int, default: Any = None) -> Any:
        if key not in self._entries:
            return default
        return self[key]

    def is_enabled(self) -> bool:
        """Return whether this cache can retain any entries."""
        return self.max_entries is None or self.max_entries > 0

    def is_full(self) -> bool:
        """Return whether a bounded cache has reached capacity."""
        return (
            self.max_entries is not None
            and self.max_entries > 0
            and len(self._entries) >= self.max_entries
        )


def _flatten_subset_indices(dataset: Dataset) -> Tuple[Dataset, Optional[List[int]]]:
    """Resolve nested ``Subset`` wrappers to a root dataset plus root indices."""
    current: Dataset = dataset
    index_map: Optional[List[int]] = None

    while isinstance(current, Subset):
        current_indices = list(current.indices)
        if index_map is None:
            index_map = current_indices
        else:
            index_map = [current_indices[i] for i in index_map]
        current = current.dataset

    return current, index_map


class _TrainingPolicyDataset(Dataset):
    """Trainer-only dataset wrapper that owns runtime sampling and caches."""

    def __init__(
        self,
        dataset: Dataset,
        config: TorchTrainingConfig,
        *,
        split: str,
    ) -> None:
        self._wrapped_dataset = dataset
        self._root_dataset, self._indices = _flatten_subset_indices(dataset)
        self._config = config
        self._split = split
        self._cache_limits = {
            "feature_max_entries": getattr(
                config, "cache_feature_max_entries", None
            ),
            "neighbor_max_entries": getattr(
                config, "cache_neighbor_max_entries", None
            ),
            "graph_max_entries": getattr(
                config, "cache_force_triplet_max_entries", None
            ),
        }
        self._cache_state = _RuntimeCacheState(**self._cache_limits)

        cache_scope = str(getattr(config, "cache_scope", "all"))
        cache_enabled = (
            cache_scope == "all"
            or (cache_scope == "train" and split == "train")
            or (cache_scope == "val" and split == "val")
        )
        self.cache_features = bool(config.cache_features) and cache_enabled
        self.cache_neighbors = (
            bool(config.cache_neighbors) and cache_enabled
        )
        self.cache_force_triplets = (
            bool(config.cache_force_triplets) and cache_enabled
        )
        self.force_sampling = str(config.force_sampling)
        self.force_fraction = float(config.force_fraction)
        self.force_min_structures_per_epoch = getattr(
            config, "force_min_structures_per_epoch", None
        )
        self.selected_force_indices: Optional[List[int]] = None

        force_indices_all = list(self._root_dataset.get_force_indices())
        if self._indices is None:
            self._force_indices = force_indices_all
        else:
            force_index_set = set(force_indices_all)
            self._force_indices = [
                source_idx
                for source_idx in self._indices
                if source_idx in force_index_set
            ]

        if self.force_sampling == "fixed" and self.force_fraction < 1.0:
            self.selected_force_indices = self._sample_force_indices()

    def __getstate__(self) -> dict:
        """Reset split-local runtime caches when DataLoader workers spawn."""
        state = self.__dict__.copy()
        state["_cache_state"] = _RuntimeCacheState(**state["_cache_limits"])
        return state

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self._root_dataset)

    @property
    def structures(self) -> Optional[List[Structure]]:
        """Expose structure lists for output helpers when available."""
        base_structures = getattr(self._root_dataset, "structures", None)
        if base_structures is None:
            return None
        if self._indices is None:
            return list(base_structures)
        return [base_structures[i] for i in self._indices]

    def get_structure(self, idx: int) -> Structure:
        """Return the wrapped structure for one split-local dataset index."""
        return self._root_dataset.get_structure(self._source_index(idx))

    def get_structure_identifier(self, idx: int) -> Optional[str]:
        """Return the wrapped identifier for one split-local dataset index."""
        getter = getattr(self._root_dataset, "get_structure_identifier", None)
        if callable(getter):
            return getter(self._source_index(idx))
        return None

    def _source_index(self, idx: int) -> int:
        if self._indices is None:
            return idx
        return int(self._indices[idx])

    def _sample_force_indices(self) -> List[int]:
        """Sample source-dataset force indices for this split."""
        n_force_total = len(self._force_indices)
        if n_force_total == 0:
            return []

        n_force = int(n_force_total * self.force_fraction)
        if self.force_min_structures_per_epoch is not None:
            n_force = max(n_force, self.force_min_structures_per_epoch)
        n_force = min(n_force, n_force_total)
        if n_force <= 0:
            return []
        return random.sample(self._force_indices, n_force)

    def initialize_force_sampling(self) -> None:
        """Populate the initial random force subset before epoch 0."""
        if float(getattr(self._config, "force_weight", 0.0)) <= 0.0:
            self.selected_force_indices = []
            return
        if self.force_sampling == "random" and self.force_fraction < 1.0:
            self.selected_force_indices = self._sample_force_indices()

    def resample_force_structures(self) -> None:
        """Refresh the random force-supervision subset for this split."""
        if float(getattr(self._config, "force_weight", 0.0)) <= 0.0:
            self.selected_force_indices = []
            return
        if self.force_sampling != "random":
            return
        self.selected_force_indices = self._sample_force_indices()

    def should_use_forces(self, source_idx: int, struct_has_forces: bool) -> bool:
        """Return whether the wrapped sample should use force supervision."""
        if not struct_has_forces:
            return False
        if float(getattr(self._config, "force_weight", 0.0)) <= 0.0:
            return False
        if self.force_fraction >= 1.0:
            return True
        if self.selected_force_indices is None:
            return False
        return source_idx in self.selected_force_indices

    def warmup_caches(self, show_progress: bool = True) -> None:
        """
        Pre-populate split-local runtime caches for the current policy.

        When all enabled caches have bounded capacities, warmup stops once
        every enabled cache has reached its limit instead of walking the full
        split eagerly.
        """
        if not self.has_enabled_runtime_caches():
            return

        iterator = range(len(self))
        if show_progress and tqdm is not None:
            iterator = tqdm(
                iterator,
                desc=f"Warming {self._split} caches",
                ncols=80,
                leave=False,
            )

        for idx in iterator:
            _ = self[idx]
            if self._warmup_can_stop_early() and self._all_warmup_targets_full():
                break

    def has_enabled_runtime_caches(self) -> bool:
        """Return whether this split can retain any trainer-owned cache data."""
        return any(cache.is_enabled() for cache in self._enabled_caches())

    def _enabled_caches(self) -> list[_BoundedCache]:
        caches: list[_BoundedCache] = []
        if self.cache_features:
            caches.append(self._cache_state.feature_cache)
        if self.cache_neighbors:
            caches.append(self._cache_state.neighbor_cache)
        if self.cache_force_triplets:
            caches.append(self._cache_state.graph_cache)
        return [cache for cache in caches if cache.is_enabled()]

    def _warmup_can_stop_early(self) -> bool:
        caches = self._enabled_caches()
        return bool(caches) and all(
            cache.max_entries is not None and cache.max_entries > 0
            for cache in caches
        )

    def _all_warmup_targets_full(self) -> bool:
        caches = self._enabled_caches()
        return bool(caches) and all(cache.is_full() for cache in caches)

    def __getitem__(self, idx: int) -> dict:
        source_idx = self._source_index(idx)
        struct = self._root_dataset.get_structure(source_idx)
        use_forces = self.should_use_forces(source_idx, struct.has_forces())
        sample = self._root_dataset.materialize_sample(
            source_idx,
            use_forces=use_forces,
            cache_state=self._cache_state,
            cache_features=self.cache_features,
            cache_neighbors=self.cache_neighbors,
            cache_force_triplets=self.cache_force_triplets,
            load_local_derivatives=use_forces,
        )
        sample["sample_index"] = int(idx)
        sample["source_index"] = int(source_idx)
        return sample

    def materialize_uncached_sample(self, idx: int) -> dict:
        """
        Materialize one sample without touching trainer-owned runtime caches.

        This is used for pre-training stats collection so normalization does
        not implicitly prefill or reorder the runtime caches that back the
        actual training loop.
        """
        source_idx = self._source_index(idx)
        struct = self._root_dataset.get_structure(source_idx)
        use_forces = self.should_use_forces(source_idx, struct.has_forces())
        sample = self._root_dataset.materialize_sample(
            source_idx,
            use_forces=use_forces,
            cache_state=None,
            cache_features=False,
            cache_neighbors=False,
            cache_force_triplets=False,
            load_local_derivatives=use_forces,
        )
        sample["sample_index"] = int(idx)
        sample["source_index"] = int(source_idx)
        return sample


def _find_hdf5_root_datasets(dataset: Optional[Dataset]) -> list[HDF5StructureDataset]:
    """
    Return reachable HDF5-backed datasets through known wrapper layers.

    The trainer currently wraps datasets with ``_TrainingPolicyDataset`` and
    may also see ``Subset`` or other thin delegating wrappers. Worker cleanup
    needs to find the HDF5-backed root instance so worker-local handles can be
    closed deterministically on shutdown.
    """
    if dataset is None or HDF5StructureDataset is None:
        return []

    discovered: list[HDF5StructureDataset] = []
    pending: list[Dataset] = [dataset]
    visited: set[int] = set()

    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        if isinstance(current, HDF5StructureDataset):
            discovered.append(current)
            continue

        candidates: list[Any] = []
        if isinstance(current, Subset):
            candidates.append(current.dataset)
        candidates.extend(
            [
                getattr(current, "dataset", None),
                getattr(current, "_dataset", None),
                getattr(current, "_wrapped_dataset", None),
                getattr(current, "_root_dataset", None),
            ]
        )
        for candidate in candidates:
            if isinstance(candidate, Dataset):
                pending.append(candidate)

    return discovered


def _close_worker_hdf5_datasets(dataset: Optional[Dataset]) -> None:
    """Close any reachable HDF5-backed datasets for the current worker."""
    for hdf5_dataset in _find_hdf5_root_datasets(dataset):
        hdf5_dataset.close()


def _register_hdf5_worker_cleanup(worker_id: int) -> None:
    """
    Register worker-exit cleanup for HDF5-backed datasets.

    DataLoader workers lazily open worker-local HDF5 handles on first access.
    Registering an ``atexit`` cleanup closes those handles deterministically
    before the worker process exits, including when workers are restarted
    between epochs.
    """
    del worker_id  # worker_info already identifies the active worker.

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    if not _find_hdf5_root_datasets(dataset):
        return

    def _cleanup(*, dataset: Dataset = dataset) -> None:
        _close_worker_hdf5_datasets(dataset)

    atexit.register(_cleanup)


class _TrainingStatsDataset(Dataset):
    """Stats-only wrapper that bypasses trainer-owned runtime caches."""

    def __init__(self, dataset: _TrainingPolicyDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        return self._dataset.materialize_uncached_sample(idx)


def _should_wrap_training_policy_dataset(dataset: Optional[Dataset]) -> bool:
    """Return True when the dataset supports trainer-owned policy wrapping."""
    if dataset is None:
        return False
    root_dataset, _ = _flatten_subset_indices(dataset)
    return (
        hasattr(root_dataset, "materialize_sample")
        and hasattr(root_dataset, "get_force_indices")
        and hasattr(root_dataset, "get_structure")
    )


def _requires_training_worker_restart(
    dataset: Optional[Dataset],
    config: TorchTrainingConfig,
) -> bool:
    """
    Return True when training workers must restart to observe new sampling.

    Random force resampling mutates trainer-owned dataset state between epochs.
    Persistent DataLoader workers keep their own copy of that state, so they
    must be restarted whenever epoch-to-epoch resampling is enabled.
    """
    if not isinstance(dataset, _TrainingPolicyDataset):
        return False
    if int(getattr(config, "num_workers", 0)) <= 0:
        return False
    if float(getattr(config, "force_weight", 0.0)) <= 0.0:
        return False
    if dataset.force_sampling != "random":
        return False
    if dataset.force_fraction >= 1.0:
        return False
    return int(getattr(config, "force_resample_num_epochs", 0)) > 0


def _should_skip_runtime_cache_warmup(
    *,
    train_dataset: Optional[Dataset],
    val_dataset: Optional[Dataset],
    config: TorchTrainingConfig,
) -> bool:
    """Return True when configured warmup should be skipped for worker mode."""
    if not bool(getattr(config, "cache_warmup", False)):
        return False
    if int(getattr(config, "num_workers", 0)) <= 0:
        return False
    for dataset in (train_dataset, val_dataset):
        if (
            isinstance(dataset, _TrainingPolicyDataset)
            and dataset.has_enabled_runtime_caches()
        ):
            return True
    return False


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
          features_f: (Nf, F), unnormalized per-atom features for the
              force-supervised structures in force-view order
          positions_f: (Nf,3), species_f (list[str]),
          species_indices_f: (Nf,), forces_ref_f: (Nf,3),
          local_derivatives_f: Optional[dict] batched sparse local
              derivative payload aligned with positions_f
          graph_f: Optional[dict] batched CSR neighbor graph for the
              force view (keys: center_ptr[int32 Nf+1], nbr_idx[int32 E],
              r_ij[E,3], d_ij[E]),
          triplets_f: Optional[dict] batched TripletIndex for the force
              view (keys: tri_i/tri_j/tri_k[int32 T],
              tri_j_local/tri_k_local[int32 T])
      - bookkeeping:
          n_atoms_force_total: int, total atoms from structures that have
            force labels in this batch (regardless of selection)
          n_atoms_force_supervised: int, total atoms supervised for forces
            in this batch (sum over selected force structures)
    """
    # Energy view aggregation (all structures)
    features_list: List[torch.Tensor] = []
    species_idx_list: List[torch.Tensor] = []
    n_atoms_list: List[int] = []
    energy_ref_list: List[float] = []
    sample_index_list: List[int] = []

    # For optional force view
    features_f_list: List[torch.Tensor] = []
    positions_f_list: List[torch.Tensor] = []
    species_f_names: List[str] = []
    species_idx_f_list: List[torch.Tensor] = []
    forces_f_list: List[torch.Tensor] = []
    force_sample_index_list: List[int] = []
    force_n_atoms_list: List[int] = []
    radial_center_parts: List[torch.Tensor] = []
    radial_neighbor_parts: List[torch.Tensor] = []
    radial_grads_parts: List[torch.Tensor] = []
    radial_typespin_parts: List[torch.Tensor] = []
    angular_center_parts: List[torch.Tensor] = []
    angular_j_parts: List[torch.Tensor] = []
    angular_k_parts: List[torch.Tensor] = []
    angular_grads_i_parts: List[torch.Tensor] = []
    angular_grads_j_parts: List[torch.Tensor] = []
    angular_grads_k_parts: List[torch.Tensor] = []
    angular_typespin_parts: List[torch.Tensor] = []
    # Optional batched CSR/Triplets parts for force view
    deg_parts: List[torch.Tensor] = []  # degrees per center (concat later)
    nbr_idx_parts: List[torch.Tensor] = []
    r_ij_parts: List[torch.Tensor] = []
    d_ij_parts: List[torch.Tensor] = []
    tri_i_parts: List[torch.Tensor] = []
    tri_j_parts: List[torch.Tensor] = []
    tri_k_parts: List[torch.Tensor] = []
    tri_j_local_parts: List[torch.Tensor] = []
    tri_k_local_parts: List[torch.Tensor] = []

    base_atom_idx_force = 0
    local_derivatives_available = True
    graph_available_for_all = True
    n_force_samples = 0
    # Track total atoms with available force labels in this batch
    n_atoms_force_total: int = 0

    for sample in batch:
        N = int(sample["n_atoms"])
        features_list.append(sample["features"])
        species_idx_list.append(sample["species_indices"])
        n_atoms_list.append(N)
        energy_ref_list.append(float(sample["energy"]))
        sample_index_list.append(int(sample.get("sample_index", -1)))
        # Count atoms from structures that have force labels
        # (regardless of selection)
        if bool(sample.get("has_forces", False)):
            n_atoms_force_total += N

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
    sample_indices = torch.tensor(sample_index_list, dtype=torch.long)

    # Build force-view if any sample is selected for forces
    for sample in batch:
        if not bool(sample["use_forces"]) or sample["forces"] is None:
            continue
        pos = sample["positions"]
        frc = sample["forces"]
        feat = sample["features"]
        species_idx = sample["species_indices"]
        species_names = sample["species"]

        n_force_samples += 1
        features_f_list.append(feat)
        positions_f_list.append(pos)
        forces_f_list.append(frc)
        species_idx_f_list.append(species_idx)
        species_f_names.extend(species_names)
        force_sample_index_list.append(int(sample.get("sample_index", -1)))
        force_n_atoms_list.append(int(pos.shape[0]))

        local_derivatives = sample.get("local_derivatives", None)
        if local_derivatives is None:
            local_derivatives_available = False
        elif local_derivatives_available:
            radial = local_derivatives["radial"]
            radial_center_parts.append(
                torch.as_tensor(radial["center_idx"]).to(torch.int64)
                + base_atom_idx_force
            )
            radial_neighbor_parts.append(
                torch.as_tensor(radial["neighbor_idx"]).to(torch.int64)
                + base_atom_idx_force
            )
            radial_grads_parts.append(torch.as_tensor(radial["dG_drij"]))
            if radial["neighbor_typespin"] is not None:
                radial_typespin_parts.append(
                    torch.as_tensor(radial["neighbor_typespin"])
                )

            angular = local_derivatives["angular"]
            angular_center_parts.append(
                torch.as_tensor(angular["center_idx"]).to(torch.int64)
                + base_atom_idx_force
            )
            angular_j_parts.append(
                torch.as_tensor(angular["neighbor_j_idx"]).to(torch.int64)
                + base_atom_idx_force
            )
            angular_k_parts.append(
                torch.as_tensor(angular["neighbor_k_idx"]).to(torch.int64)
                + base_atom_idx_force
            )
            angular_grads_i_parts.append(torch.as_tensor(angular["grads_i"]))
            angular_grads_j_parts.append(torch.as_tensor(angular["grads_j"]))
            angular_grads_k_parts.append(torch.as_tensor(angular["grads_k"]))
            if angular["triplet_typespin"] is not None:
                angular_typespin_parts.append(
                    torch.as_tensor(angular["triplet_typespin"])
                )

        g = sample.get("graph", None)
        if g is None:
            graph_available_for_all = False
        else:
            cp = torch.as_tensor(g["center_ptr"])
            deg_parts.append((cp[1:] - cp[:-1]).to(torch.int64))
            nbr_idx_parts.append(torch.as_tensor(g["nbr_idx"]).to(
                torch.int64) + base_atom_idx_force)
            r_ij_parts.append(torch.as_tensor(g["r_ij"]))
            d_ij_parts.append(torch.as_tensor(g["d_ij"]))

            t = sample.get("triplets", None)
            if t is not None:
                tri_i_parts.append(torch.as_tensor(t["tri_i"]).to(
                    torch.int64) + base_atom_idx_force)
                tri_j_parts.append(torch.as_tensor(t["tri_j"]).to(
                    torch.int64) + base_atom_idx_force)
                tri_k_parts.append(torch.as_tensor(t["tri_k"]).to(
                    torch.int64) + base_atom_idx_force)
                tri_j_local_parts.append(
                    torch.as_tensor(t["tri_j_local"]).to(torch.int64))
                tri_k_local_parts.append(
                    torch.as_tensor(t["tri_k_local"]).to(torch.int64))

        base_atom_idx_force += int(pos.shape[0])

    force_view_present = len(positions_f_list) > 0
    features_f = (torch.cat(features_f_list, dim=0)
                  if force_view_present else None)
    positions_f = (torch.cat(positions_f_list, dim=0)
                   if force_view_present else None)
    species_indices_f = (
        torch.cat(species_idx_f_list, dim=0) if force_view_present else None
    )
    forces_ref_f = (torch.cat(forces_f_list, dim=0)
                    if force_view_present else None)
    local_derivatives_f = None
    if force_view_present and local_derivatives_available and n_force_samples > 0:
        radial_block = {
            "center_idx": torch.cat(radial_center_parts).to(torch.int64),
            "neighbor_idx": torch.cat(radial_neighbor_parts).to(torch.int64),
            "dG_drij": torch.cat(radial_grads_parts, dim=0),
            "neighbor_typespin": (
                torch.cat(radial_typespin_parts, dim=0)
                if len(radial_typespin_parts) > 0
                else None
            ),
        }
        angular_block = {
            "center_idx": torch.cat(angular_center_parts).to(torch.int64),
            "neighbor_j_idx": torch.cat(angular_j_parts).to(torch.int64),
            "neighbor_k_idx": torch.cat(angular_k_parts).to(torch.int64),
            "grads_i": torch.cat(angular_grads_i_parts, dim=0),
            "grads_j": torch.cat(angular_grads_j_parts, dim=0),
            "grads_k": torch.cat(angular_grads_k_parts, dim=0),
            "triplet_typespin": (
                torch.cat(angular_typespin_parts, dim=0)
                if len(angular_typespin_parts) > 0
                else None
            ),
        }
        local_derivatives_f = {
            "radial": radial_block,
            "angular": angular_block,
        }
    elif force_view_present and not graph_available_for_all:
        raise RuntimeError(
            "Force-supervised training batches must provide either "
            "precomputed local derivatives or graph payloads for every "
            "force sample."
        )

    # Build batched CSR/Triplets for force view if any parts were collected
    graph_f = None
    triplets_f = None
    if force_view_present and graph_available_for_all and len(deg_parts) > 0:
        total_centers = int(positions_f.shape[0])
        deg_cat = (torch.cat(deg_parts) if len(deg_parts) > 0
                   else torch.empty(0, dtype=torch.int64))
        center_ptr = torch.zeros(total_centers + 1, dtype=torch.int64)
        if deg_cat.numel() != total_centers:
            raise RuntimeError(
                "Malformed force-training graph payload: CSR degrees do not "
                "match the batched force view."
            )
        center_ptr[1:] = torch.cumsum(deg_cat, dim=0)
        nbr_idx_b = torch.cat(nbr_idx_parts).to(torch.int32)
        r_ij_b = torch.cat(r_ij_parts)
        d_ij_b = torch.cat(d_ij_parts)
        graph_f = {
            "center_ptr": center_ptr.to(torch.int32),
            "nbr_idx": nbr_idx_b,
            "r_ij": r_ij_b,
            "d_ij": d_ij_b,
        }
        if len(tri_i_parts) > 0:
            triplets_f = {
                "tri_i": torch.cat(tri_i_parts).to(torch.int32),
                "tri_j": torch.cat(tri_j_parts).to(torch.int32),
                "tri_k": torch.cat(tri_k_parts).to(torch.int32),
                "tri_j_local": torch.cat(
                    tri_j_local_parts).to(torch.int32),
                "tri_k_local": torch.cat(
                    tri_k_local_parts).to(torch.int32),
            }

    return {
        # Energy view
        "features": features,
        "species_indices": species_indices,
        "n_atoms": n_atoms,
        "energy_ref": energy_ref,
        "sample_indices": sample_indices,
        # Force view
        "features_f": features_f,
        "positions_f": positions_f,
        "species_f": species_f_names if force_view_present else None,
        "species_indices_f": species_indices_f,
        "forces_ref_f": forces_ref_f,
        "force_sample_indices": (
            torch.tensor(force_sample_index_list, dtype=torch.long)
            if force_view_present else None
        ),
        "force_sample_n_atoms": (
            torch.tensor(force_n_atoms_list, dtype=torch.long)
            if force_view_present else None
        ),
        "local_derivatives_f": local_derivatives_f,
        "graph_f": graph_f,
        "triplets_f": triplets_f,
        # Bookkeeping for unbiased scaling / diagnostics
        "n_atoms_force_total": n_atoms_force_total,
        "n_atoms_force_supervised": (int(positions_f.shape[0])
                                     if force_view_present else 0),
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

    # Supported PredictionConfig options for PyTorch API
    _supported_config_options = {
        'timing', 'print_atomic_energies', 'debug',
        'verbosity', 'batch_size',
        # DataLoader knobs for inference
        'num_workers', 'prefetch_factor', 'persistent_workers'
    }

    def __init__(self, arch: Dict[str, List[Tuple[int, str]]], descriptor):
        self.arch = arch
        self.descriptor = descriptor
        self.device = torch.device(str(descriptor.device))
        self.dtype = descriptor.dtype

        # Build underlying network and adapter using NetworkBuilder
        builder = NetworkBuilder(descriptor=descriptor,
                                 device=self.device,
                                 dtype=self.dtype)
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

        # Normalization - initialize with defaults
        self._normalizer = NormalizationManager(
            normalize_features=True,
            normalize_energy=True,
            dtype=self.dtype,
            device=self.device,
        )

        # Atomic reference energies (defaults to zeros if not provided)
        self._atomic_energies: Optional[Dict[str, float]] = None

        # Training state for checkpointing/saving
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._training_config: Optional[TorchTrainingConfig] = None

        # Metadata from loaded model files
        self._metadata: Optional[Dict[str, Any]] = None

    @property
    def history(self) -> Dict[str, List[float]]:
        """Access training history through metrics tracker."""
        return self._metrics.get_history()

    @history.setter
    def history(self, value: Dict[str, List[float]]):
        """Set training history (for loading from checkpoint)."""
        self._metrics.set_history(value)

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """
        Access model metadata from loaded files.

        Returns None if the model was not loaded from a file using
        `from_file()`. Provides access to rich metadata including:
        - schema_version: Format version
        - timestamp: When the model was saved
        - training_config: Training configuration used
        - descriptor_config: Descriptor configuration
        - normalization: Normalization parameters
        - atomic_energies: Reference atomic energies
        - training_history: Full training history
        - architecture: Network architecture
        - extra_metadata: Any user-provided metadata

        Returns
        -------
        dict or None
            Metadata dictionary if model was loaded from file, None otherwise.

        Examples
        --------
        >>> pot = TorchANNPotential.from_file('model.pt')
        >>> print(pot.metadata['timestamp'])
        >>> print(pot.metadata['training_config'])
        """
        return self._metadata

    def _structures_from_dataset(
        self,
        dataset: Optional[Dataset],
    ) -> Optional[List[Structure]]:
        """Best-effort extraction of torch-training structures from a dataset."""
        if dataset is None:
            return None
        if isinstance(dataset, Subset):
            base_structures = self._structures_from_dataset(dataset.dataset)
            if base_structures is None:
                return None
            return [base_structures[i] for i in dataset.indices]

        get_structure = getattr(dataset, "get_structure", None)
        if callable(get_structure):
            structures: List[Structure] = []
            for idx in range(len(dataset)):
                struct = get_structure(idx)
                identifier = self._dataset_structure_identifier(
                    dataset,
                    idx,
                    structure=struct,
                )
                structures.append(
                    self._structure_with_identifier(
                        struct,
                        idx,
                        identifier=identifier,
                    )
                )
            return structures

        structures = getattr(dataset, "structures", None)
        if structures is None:
            return None
        return [
            self._structure_with_identifier(
                struct,
                idx,
                identifier=self._dataset_structure_identifier(
                    dataset,
                    idx,
                    structure=struct,
                ),
            )
            for idx, struct in enumerate(structures)
        ]

    @staticmethod
    def _dataset_structure_identifier(
        dataset: Dataset,
        index: int,
        *,
        structure: Optional[Structure] = None,
    ) -> Optional[str]:
        """
        Return the preferred output identifier for one dataset entry.

        Datasets that persist structured source metadata, such as
        ``HDF5StructureDataset``, own the merged identifier synthesis via
        ``get_structure_identifier()``. The trainer stays agnostic to the
        underlying metadata fields and only falls back to ``Structure.name``
        when the dataset does not provide an identifier.
        """
        getter = getattr(dataset, "get_structure_identifier", None)
        if callable(getter):
            identifier = getter(index)
            if identifier not in (None, ""):
                return str(identifier)

        if structure is not None and getattr(structure, "name", None) not in (
            None,
            "",
        ):
            return str(structure.name)

        return None

    @staticmethod
    def _structure_with_identifier(
        structure: Structure,
        index: int,
        identifier: Optional[str] = None,
    ) -> Structure:
        """Return a structure carrying a stable identifier for outputs."""
        resolved = identifier
        if resolved in (None, "") and getattr(structure, "name", None) not in (
            None,
            "",
        ):
            resolved = str(structure.name)
        if resolved in (None, ""):
            resolved = f"structure_{index:06d}"

        if getattr(structure, "name", None) == resolved:
            return structure
        return replace(structure, name=resolved)

    def _write_energies_file(
        self,
        structures: List[Structure],
        outfile: Union[str, os.PathLike],
        predict_out: Optional[Any] = None,
    ) -> str:
        """Write Fortran-compatible energies.* output for a structure list."""
        if predict_out is None:
            predict_out = self.predict(structures, eval_forces=False)
        species_order = [str(sp) for sp in self.descriptor.species]
        species_headers = "".join(f" #{sp}" for sp in species_order)
        header = (
            "#  Ref(eV)        ANN(eV)      #atoms  Ref(eV/atom)"
            "   ANN(eV/atom) Ref-ANN(eV/atom)    Cost-Func"
            f"{species_headers}    Path-of-input-file"
        )

        out_path = Path(outfile)
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write(header + "\n")
            for idx, (struct, ann_energy) in enumerate(
                zip(structures, predict_out.total_energy)
            ):
                n_atoms = int(struct.n_atoms)
                ref_energy = float(struct.energy)
                ann_energy = float(ann_energy)
                ref_per_atom = ref_energy / float(n_atoms)
                ann_per_atom = ann_energy / float(n_atoms)
                diff_per_atom = ref_per_atom - ann_per_atom
                if self._normalizer.normalize_energy:
                    cost_func = diff_per_atom * float(self._normalizer.E_scaling)
                else:
                    cost_func = diff_per_atom

                species_counts = Counter(str(sp) for sp in struct.species)
                path_str = (
                    str(struct.name)
                    if getattr(struct, "name", None)
                    else f"structure_{idx:06d}"
                )

                fp.write(
                    " "
                    f"{ref_energy:14.6E} "
                    f"{ann_energy:14.6E} "
                    f"{n_atoms:5d} "
                    f"{ref_per_atom:14.6E} "
                    f"{ann_per_atom:14.6E} "
                    f"{diff_per_atom:14.6E} "
                    f"{cost_func:14.6E}"
                )
                for sp in species_order:
                    fp.write(f" {species_counts.get(sp, 0):4d}")
                fp.write(f" {path_str}\n")

        return str(out_path)

    def _save_energy_outputs(
        self,
        config: TorchTrainingConfig,
        train_ds: Optional[Dataset],
        test_ds: Optional[Dataset],
    ) -> Tuple[List[str], List[str]]:
        """Emit energies.train/test files when requested."""
        if not bool(getattr(config, "save_energies", False)):
            return [], []

        train_structures = self._structures_from_dataset(train_ds)
        test_structures = self._structures_from_dataset(test_ds)

        if train_structures is None:
            warnings.warn(
                "save_energies=True, but the training dataset does not expose "
                "raw structures. Energy files were not written.",
                UserWarning,
            )
            return [], []

        train_predict_out = None
        test_predict_out = None
        try:
            train_predict_out = self.predict_dataset(train_ds)
            if test_ds is not None:
                test_predict_out = self.predict_dataset(test_ds)
        except (NotImplementedError, ValueError, KeyError, TypeError):
            train_predict_out = None
            test_predict_out = None

        train_files = [os.path.basename(
            self._write_energies_file(
                train_structures,
                "energies.train.0",
                predict_out=train_predict_out,
            )
        )]
        test_files: List[str] = []
        if test_structures:
            test_files.append(os.path.basename(
                self._write_energies_file(
                    test_structures,
                    "energies.test.0",
                    predict_out=test_predict_out,
                )
            ))
        return train_files, test_files

    def train(
        self,
        structures: Optional[Union[List[Structure],
                                   List, List[os.PathLike]]] = None,
        dataset: Optional[Dataset] = None,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        config: Optional[TorchTrainingConfig] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Train the MLIP and return training results.

        Parameters
        ----------
        structures : List[Structure] or List[AtomicStructure]
                         or List[os.PathLike], optional
            Training structures in one of three formats:
            - List of torch Structure objects (direct use)
            - List of AtomicStructure objects (automatically converted)
            - List of file paths to structure files (loaded and converted)
            If provided and dataset/train_dataset are None, a StructureDataset
            is created.
        dataset : torch.utils.data.Dataset, optional
            Generic dataset providing samples compatible with the collate_fn.
            If provided and train_dataset/test_dataset are None, a train/test
            split will be created using config.testpercent.
        train_dataset : torch.utils.data.Dataset, optional
            Explicit training dataset. If provided, takes precedence over
            'dataset' and 'structures'.
        test_dataset : torch.utils.data.Dataset, optional
            Explicit test dataset to use alongside train_dataset.
        config : TorchTrainingConfig, optional
            Training configuration. ``config.iterations`` always means the
            number of epochs to run in this call. When ``resume_from`` is
            provided, the trainer loads the checkpoint state and then runs
            that many additional epochs.
        resume_from : str, optional
            Path to a training checkpoint created by the checkpoint manager.
            The checkpoint's saved epoch, optimizer state, history, and
            normalization state are restored before continuing training.

        Notes
        -----
        Priority of data sources:
          1) train_dataset/test_dataset (if provided)
          2) dataset (auto-split using config.testpercent)
          3) structures list (legacy path; may use CachedStructureDataset
             when energy-only with cached_features=True)

        - Energy RMSE is computed per-atom (consistent with aenet-PyTorch)
        - Force RMSE is overall RMSE across all components in batch
        """
        if config is None:
            config = TorchTrainingConfig()

        # Resolve device/dtype and memory mode
        device = _resolve_device(config)
        self.device = device  # sync trainer device with resolved device
        memory_mode = config.memory_mode
        if memory_mode not in ("cpu", "gpu", "mixed"):
            raise ValueError(f"Invalid memory_mode '{memory_mode}'")
        if memory_mode == "mixed":
            raise NotImplementedError(
                "memory_mode='mixed' is reserved for a future real "
                "mixed-memory execution mode and is not implemented yet. "
                "Use memory_mode='cpu' or memory_mode='gpu' for now."
            )

        # Ensure model on device/dtype
        self.model.to(device)
        if self.dtype == torch.float64:
            self.model = self.model.double()
        else:
            self.model = self.model.float()
        # Ensure final placement on resolved device
        self.model.to(device)

        # Atomic reference energy configuration
        # Default all species to 0.0, then update with user-provided values
        import warnings
        atomic_energies_dict: Dict[str, float] = {
            s: 0.0 for s in self.descriptor.species
        }

        atomic_energies_cfg = getattr(config, "atomic_energies", None)
        if atomic_energies_cfg:
            atomic_energies_dict.update(atomic_energies_cfg)
            # Store for later prediction/export
            try:
                self._atomic_energies = {
                    str(k): float(v) for k, v in atomic_energies_dict.items()
                }
            except Exception:
                self._atomic_energies = dict(atomic_energies_dict)
        else:
            warnings.warn(
                "No atomic_energies provided. Training on total energies "
                "(all atomic reference energies set to 0.0). For cohesive "
                "or explicit formation-energy training, provide atomic "
                "reference energies in config.atomic_energies. If your "
                "labels already include an external user-defined reference, "
                "this zero-reference fallback preserves those semantics.",
                UserWarning
            )
            self._atomic_energies = atomic_energies_dict

        # Convert to tensor indexed by species for efficient lookup
        e_list: List[float] = [atomic_energies_dict[s]
                               for s in self.descriptor.species]
        atomic_energies_by_index = torch.tensor(
            e_list, dtype=self.dtype, device=device
        )

        # Convert various input types to List[Structure] (torch Structure)
        if structures is not None and len(structures) > 0:
            from .dataset import convert_to_structures
            structures = convert_to_structures(structures)
            structures = [
                self._structure_with_identifier(structure, idx)
                for idx, structure in enumerate(structures)
            ]

        # Dataset and split
        # (priority: explicit train/test > dataset > structures)
        train_ds = None
        test_ds = None
        _warn_if_max_energy_is_ignored_for_prebuilt_datasets(
            config=config,
            dataset=dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

        if train_dataset is not None:
            train_ds = train_dataset
            test_ds = test_dataset
        elif dataset is not None:
            base_ds = dataset
            if config.testpercent > 0 and test_dataset is None:
                test_fraction = config.testpercent / 100.0
                if train_test_split_dataset is not None:
                    train_ds, test_ds = train_test_split_dataset(
                        base_ds, test_fraction=test_fraction
                    )
                else:
                    # Generic fallback split using Subset
                    import random as _rand
                    n = len(base_ds)
                    indices = list(range(n))
                    _rand.shuffle(indices)
                    n_test = int(n * test_fraction)
                    test_idx = indices[:n_test]
                    train_idx = indices[n_test:]
                    train_ds = Subset(base_ds, train_idx)
                    test_ds = Subset(base_ds, test_idx)
            else:
                train_ds = base_ds
                test_ds = test_dataset
        else:
            # Legacy path from list[Structure]
            if structures is None:
                raise ValueError(
                    "Provide either 'structures' or 'dataset'/'train_dataset'"
                )
            if bool(getattr(config, "cache_features", False)
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
                    train_structs = [structures_all[i]
                                     for i in sorted(train_idx)]
                    test_structs = [structures_all[i]
                                    for i in sorted(test_idx)]
                else:
                    train_structs = structures_all
                    test_structs = []
                from .dataset import CachedStructureDataset
                show_progress = bool(getattr(config, "show_progress", True))
                train_ds = CachedStructureDataset(
                    structures=train_structs,
                    descriptor=self.descriptor,
                    max_energy=config.max_energy,
                    max_forces=config.max_forces,
                    atomic_energies=config.atomic_energies,
                    seed=None,
                    show_progress=show_progress,
                )
                test_ds = (
                    CachedStructureDataset(
                        structures=test_structs,
                        descriptor=self.descriptor,
                        max_energy=config.max_energy,
                        max_forces=config.max_forces,
                        atomic_energies=config.atomic_energies,
                        seed=None,
                        show_progress=show_progress,
                    )
                    if (config.testpercent > 0 and len(test_structs) > 0)
                    else None
                )
            else:
                full_ds = StructureDataset(
                    structures=structures,
                    descriptor=self.descriptor,
                    max_energy=config.max_energy,
                    max_forces=config.max_forces,
                    atomic_energies=config.atomic_energies,
                    seed=None,
                )
                if config.testpercent > 0:
                    test_fraction = config.testpercent / 100.0
                    train_ds, test_ds = train_test_split(
                        full_ds, test_fraction=test_fraction
                    )
                else:
                    train_ds, test_ds = full_ds, None

        if _should_wrap_training_policy_dataset(train_ds):
            train_ds = _TrainingPolicyDataset(
                train_ds,
                config,
                split="train",
            )
        if _should_wrap_training_policy_dataset(test_ds):
            test_ds = _TrainingPolicyDataset(
                test_ds,
                config,
                split="val",
            )

        # Initial force structure selection for random sampling.
        if isinstance(train_ds, _TrainingPolicyDataset):
            train_ds.initialize_force_sampling()
        if isinstance(test_ds, _TrainingPolicyDataset):
            test_ds.initialize_force_sampling()

        # Optional trainer-owned runtime-cache warmup.
        show_progress = bool(getattr(config, "show_progress", True))
        if _should_skip_runtime_cache_warmup(
            train_dataset=train_ds,
            val_dataset=test_ds,
            config=config,
        ):
            warnings.warn(
                "Skipping trainer-owned runtime cache warmup because "
                "num_workers > 0 creates worker-local caches after "
                "DataLoader worker spawn.",
                UserWarning,
            )
        elif bool(getattr(config, "cache_warmup", False)):
            if isinstance(train_ds, _TrainingPolicyDataset):
                train_ds.warmup_caches(show_progress=show_progress)
            if isinstance(test_ds, _TrainingPolicyDataset):
                test_ds.warmup_caches(show_progress=show_progress)

        # DataLoaders
        batch_size = OptimizerBuilder.get_batch_size(config.method)
        train_dl_kwargs: Dict[str, Any] = {}
        eval_dl_kwargs: Dict[str, Any] = {}
        nw = int(getattr(config, "num_workers", 0))
        if nw > 0:
            prefetch_factor = int(getattr(config, "prefetch_factor", 2))
            eval_persistent_workers = bool(
                getattr(config, "persistent_workers", True)
            )
            train_persistent_workers = eval_persistent_workers
            if _requires_training_worker_restart(train_ds, config):
                if train_persistent_workers:
                    warnings.warn(
                        "Disabling persistent_workers for the training "
                        "DataLoader because random force resampling updates "
                        "dataset state between epochs and persistent worker "
                        "copies would otherwise keep a stale force subset.",
                        UserWarning,
                    )
                train_persistent_workers = False

            train_dl_kwargs.update(
                dict(
                    num_workers=nw,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=train_persistent_workers,
                    worker_init_fn=_register_hdf5_worker_cleanup,
                )
            )
            eval_dl_kwargs.update(
                dict(
                    num_workers=nw,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=eval_persistent_workers,
                    worker_init_fn=_register_hdf5_worker_cleanup,
                )
            )
        else:
            train_dl_kwargs.update(dict(num_workers=0))
            eval_dl_kwargs.update(dict(num_workers=0))

        train_sampler, adaptive_sampling_state = _build_training_sampler(
            train_ds,
            config=config,
            atomic_energies=atomic_energies_dict,
        )
        train_shuffle = train_sampler is None
        if train_sampler is not None:
            train_dl_kwargs["sampler"] = train_sampler

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            collate_fn=_collate_fn,
            **train_dl_kwargs,
        )
        test_loader = (
            DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_fn,
                **eval_dl_kwargs,
            )
            if test_ds is not None
            else None
        )

        n_val = int(len(test_ds)) if test_ds is not None else 0
        _warn_on_small_validation_set(
            n_val=n_val,
            use_scheduler=bool(config.use_scheduler) and (test_loader is not None),
            save_best=bool(config.save_best)
            and (config.checkpoint_dir is not None)
            and (test_loader is not None),
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

        # Stats DataLoader (no shuffle). Keep normalization/stat collection
        # independent from trainer-owned runtime cache population.
        stats_dataset = (
            _TrainingStatsDataset(train_ds)
            if isinstance(train_ds, _TrainingPolicyDataset)
            else train_ds
        )
        stats_loader = DataLoader(
            stats_dataset, batch_size=batch_size,
            shuffle=False, collate_fn=_collate_fn
        )

        # Compute normalization statistics
        try:
            n_features = int(self.descriptor.get_n_features())
        except Exception:
            n_features = None  # type: ignore[assignment]

        # Feature stats
        show_progress = bool(getattr(config, "show_progress", True))
        if normalize_features and n_features is not None:
            provided_stats = getattr(config, "feature_stats", None)
            if provided_stats:
                mean_np = provided_stats.get("mean", None)
                std_np = provided_stats.get(
                    "std", None) or provided_stats.get("cov", None)
                if mean_np is not None and std_np is not None:
                    self._normalizer.set_feature_stats(mean_np, std_np)

            if not self._normalizer.has_feature_stats():
                self._normalizer.compute_feature_stats(
                    stats_loader, n_features, show_progress=show_progress)

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
                    stats_loader, atomic_energies_by_index,
                    show_progress=show_progress
                )

        # Optimizer and scheduler using OptimizerBuilder
        opt_builder = OptimizerBuilder(self.model)
        optimizer = opt_builder.build_optimizer(config.method)
        scheduler = opt_builder.build_scheduler(
            optimizer,
            use_scheduler=bool(config.use_scheduler
                               ) and (test_loader is not None),
            scheduler_patience=int(config.scheduler_patience),
            scheduler_factor=float(config.scheduler_factor),
            scheduler_min_lr=float(config.scheduler_min_lr),
        )

        # Checkpoint manager
        ckpt_manager = None
        if config.checkpoint_dir is not None or resume_from is not None:
            ckpt_manager = CheckpointManager(
                checkpoint_dir=config.checkpoint_dir,
                max_to_keep=config.max_checkpoints,
                save_best=config.save_best,
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
                start_epoch = ckpt_manager.infer_start_epoch(
                    resume_from,
                    payload=payload,
                )

        n_epochs = int(config.iterations)
        end_epoch = start_epoch + n_epochs
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
            tqdm(total=n_epochs, desc="Training", ncols=80)
            if show_progress and tqdm is not None
            else None
        )
        adaptive_error_sampling = (
            str(getattr(config, "sampling_policy", "uniform"))
            == "error_weighted"
        )

        for epoch in range(start_epoch, end_epoch):
            t0 = time.time()
            save_best_checkpoint = False
            # Force sampling per epoch-window
            if isinstance(train_ds, _TrainingPolicyDataset):
                # Resample if force_resample_num_epochs > 0 and we're at
                # the right epoch modulo
                should_resample = (
                    train_ds.force_sampling == "random"
                    and train_ds.force_fraction < 1.0
                    and config.force_resample_num_epochs > 0
                    and ((epoch - start_epoch)
                         % config.force_resample_num_epochs == 0)
                )
                if should_resample:
                    train_ds.resample_force_structures()

            # Train one epoch
            (train_energy_rmse, train_energy_mae,
             train_force_rmse, train_timing, train_structure_scores
             ) = training_loop.run_epoch(
                loader=train_loader,
                optimizer=optimizer,
                alpha=alpha,
                atomic_energies_by_index=atomic_energies_by_index,
                train=True,
                show_batch_progress=show_batch,
                force_scale_unbiased=bool(
                    getattr(config, "force_scale_unbiased", False)),
                collect_structure_scores=adaptive_error_sampling,
            )
            if (
                adaptive_error_sampling
                and adaptive_sampling_state is not None
                and train_sampler is not None
                and train_structure_scores is not None
            ):
                train_sampler.weights = adaptive_sampling_state.update_from_epoch(
                    train_structure_scores
                )

            # Validation
            val_energy_rmse = float("nan")
            val_energy_mae = float("nan")
            val_force_rmse = float("nan")
            val_timing = {}
            if test_loader is not None:
                (val_energy_rmse, val_energy_mae, val_force_rmse, val_timing
                 , _
                 ) = training_loop.run_epoch(
                    loader=test_loader,
                    optimizer=None,
                    alpha=alpha,
                    atomic_energies_by_index=atomic_energies_by_index,
                    train=False,
                    show_batch_progress=False,
                    force_scale_unbiased=bool(
                        getattr(config, "force_scale_unbiased", False)),
                    collect_structure_scores=False,
                )

                # Compute validation loss for monitoring
                import math
                val_loss_monitor = (
                    (1 - alpha) * val_energy_rmse
                    + (alpha * val_force_rmse
                       if not math.isnan(val_force_rmse) else 0.0)
                )

                # Best model tracking
                if config.save_best and ckpt_manager is not None:
                    save_best_checkpoint = ckpt_manager.should_save_best(
                        val_loss_monitor
                    )
                    if save_best_checkpoint:
                        self.best_val = (
                            float(ckpt_manager.best_val_loss)
                            if ckpt_manager.best_val_loss is not None
                            else float(val_loss_monitor)
                        )

                # Scheduler
                if scheduler is not None:
                    scheduler.step(val_loss_monitor)

            # Update metrics
            epoch_time = time.time() - t0
            self._metrics.update(
                train_energy_rmse=float(train_energy_rmse),
                train_energy_mae=float(train_energy_mae),
                train_force_rmse=float(train_force_rmse),
                test_energy_rmse=float(val_energy_rmse),
                test_energy_mae=float(val_energy_mae),
                test_force_rmse=float(val_force_rmse),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                epoch_time=epoch_time,
                forward_time=training_loop.last_forward_time,
                backward_time=training_loop.last_backward_time,
                train_timing=train_timing,
                val_timing=val_timing,
            )

            if save_best_checkpoint and ckpt_manager is not None:
                ckpt_manager.save_best_model(
                    trainer=self,
                    optimizer=optimizer,
                    epoch=epoch,
                    training_config=config,
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
            if ckpt_manager is not None and config.checkpoint_interval > 0:
                if ((epoch + 1) % int(config.checkpoint_interval)) == 0:
                    ckpt_manager.save_checkpoint(
                        trainer=self,
                        optimizer=optimizer,
                        epoch=epoch,
                        training_config=config,
                    )

        if pbar is not None:
            pbar.close()

        # Store optimizer and config for later saving
        self._optimizer = optimizer
        self._training_config = config
        energies_train_files, energies_test_files = self._save_energy_outputs(
            config=config,
            train_ds=train_ds,
            test_ds=test_ds,
        )

        # Import TrainOut and return structured results
        from aenet.io.train import TrainOut
        return TrainOut.from_torch_history(
            history=self._metrics.get_history(),
            config=config,
            energies_train_files=energies_train_files,
            energies_test_files=energies_test_files,
        )

    @torch.no_grad()
    def predict(
        self,
        structures,
        eval_forces: bool = False,
        config: Optional[Any] = None,
    ):
        """
        Predict energies (and optionally forces) for structures.

        Parameters
        ----------
        structures : List[os.PathLike] or List[AtomicStructure] or
            List[Structure]
            Either file paths to structure files, AtomicStructure objects,
            or torch Structure objects.
        eval_forces : bool, optional
            If True, compute and return atomic forces. Default: False
        config : PredictionConfig, optional
            Prediction configuration. If None, uses defaults. Default: None

        Returns
        -------
        PredictOut
            Prediction results containing energies, forces, coordinates,
            and optional timing information.

        Examples
        --------
        >>> # Load from files
        >>> results = pot.predict(['struct1.xsf', 'struct2.xsf'],
        ...                       eval_forces=True)
        >>> print(results.total_energy)

        >>> # Use AtomicStructure objects
        >>> from aenet.geometry import AtomicStructure
        >>> strucs = [AtomicStructure.from_file('file.xsf')]
        >>> results = pot.predict(strucs, eval_forces=True,
        ...                       config=PredictionConfig(timing=True))
        """
        import os
        import warnings

        from aenet.io.predict import PredictOut
        from aenet.mlip import PredictionConfig as PC

        # Use default config if not provided
        if config is None:
            config = PC()

        # Warn about unsupported config options
        changed_options = config.user_changed()
        unsupported = (set(changed_options.keys()) -
                       self._supported_config_options)
        if unsupported:
            unsupported_list = ', '.join(sorted(unsupported))
            warnings.warn(
                f"The following PredictionConfig parameters are not "
                f"supported by the PyTorch API and will be ignored: "
                f"{unsupported_list}",
                UserWarning
            )

        # Convert structures to torch Structure objects
        from .dataset import convert_to_structures

        # Track if input was file paths for returning structure_paths
        input_paths: Optional[List[str]] = None
        if isinstance(structures[0], (str, os.PathLike)):
            input_paths = [str(p) for p in structures]

        torch_structures = convert_to_structures(structures)

        # Create Predictor and call with options
        predictor = Predictor(
            model=self.model,
            descriptor=self.descriptor,
            normalizer=self._normalizer,
            atomic_energies=self._atomic_energies,
            device=self.device,
            dtype=self.dtype,
        )

        energies, forces, atom_energies, timing = predictor.predict(
            structures=torch_structures,
            eval_forces=eval_forces,
            return_atom_energies=config.print_atomic_energies,
            track_timing=config.timing,
            batch_size=getattr(config, "batch_size", None),
            num_workers=getattr(config, "num_workers", None),
            prefetch_factor=getattr(config, "prefetch_factor", None),
            persistent_workers=getattr(config, "persistent_workers", None),
        )

        # Compute cohesive energies
        cohesive_energies: List[float] = []
        for i, s in enumerate(torch_structures):
            total_e = energies[i]
            if self._atomic_energies:
                atomic_sum = sum(self._atomic_energies.get(sp, 0.0)
                                 for sp in s.species)
                cohesive_energies.append(total_e - atomic_sum)
            else:
                cohesive_energies.append(total_e)

        # Build PredictOut object
        coords = [s.positions for s in torch_structures]
        atom_types = [s.species for s in torch_structures]

        # Convert forces to numpy if present
        forces_np = None
        if forces:
            forces_np = [f.numpy() for f in forces]

        # Convert atom_energies to numpy if present
        atom_energies_np = None
        if atom_energies:
            atom_energies_np = [ae.numpy() for ae in atom_energies]

        return PredictOut(
            coords=coords,
            forces=forces_np,
            atom_types=atom_types,
            atom_energies=atom_energies_np,
            cohesive_energy=cohesive_energies,
            total_energy=energies,
            structure_paths=input_paths,
            timing=timing,
        )

    def predict_dataset(
        self,
        dataset: Dataset,
        eval_forces: bool = False,
        config: Optional[Any] = None,
    ):
        """
        Predict energies from a PyTorch dataset.

        This torch-only convenience API is useful when the dataset already
        stores precomputed features, such as ``CachedStructureDataset``, so
        inference can reuse cached tensors instead of featurizing again.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset yielding samples with ``features``, ``species_indices``,
            ``n_atoms``, and ``species`` fields. ``Subset`` wrappers are
            supported.
        eval_forces : bool, optional
            Force prediction is not currently implemented for dataset-backed
            inference. Default: False.
        config : PredictionConfig, optional
            Prediction configuration. If None, uses defaults. Default: None

        Returns
        -------
        PredictOut
            Prediction results. This matches the structure-based ``predict()``
            return type.
        """
        import warnings

        from aenet.io.predict import PredictOut
        from aenet.mlip import PredictionConfig as PC

        if config is None:
            config = PC()

        changed_options = config.user_changed()
        unsupported = (set(changed_options.keys()) -
                       self._supported_config_options)
        if unsupported:
            unsupported_list = ', '.join(sorted(unsupported))
            warnings.warn(
                f"The following PredictionConfig parameters are not "
                f"supported by the PyTorch API and will be ignored: "
                f"{unsupported_list}",
                UserWarning
            )

        predictor = Predictor(
            model=self.model,
            descriptor=self.descriptor,
            normalizer=self._normalizer,
            atomic_energies=self._atomic_energies,
            device=self.device,
            dtype=self.dtype,
        )

        energies, _, atom_energies, timing, metadata = predictor.predict_dataset(
            dataset=dataset,
            eval_forces=eval_forces,
            return_atom_energies=config.print_atomic_energies,
            track_timing=config.timing,
            batch_size=getattr(config, "batch_size", None),
            num_workers=getattr(config, "num_workers", None),
            prefetch_factor=getattr(config, "prefetch_factor", None),
            persistent_workers=getattr(config, "persistent_workers", None),
        )

        structures = self._structures_from_dataset(dataset)
        coords = metadata["coords"]
        atom_types = metadata["atom_types"]
        names = metadata["names"]
        if structures is not None:
            coords = [s.positions for s in structures]
            atom_types = [s.species for s in structures]
            names = [str(s.name) if s.name is not None else None
                     for s in structures]

        if any(c is None for c in coords):
            raise ValueError(
                "predict_dataset() could not reconstruct coordinates for all "
                "dataset entries."
            )

        cohesive_energies: List[float] = []
        for i, species in enumerate(atom_types):
            total_e = energies[i]
            if self._atomic_energies:
                atomic_sum = sum(self._atomic_energies.get(sp, 0.0)
                                 for sp in species)
                cohesive_energies.append(total_e - atomic_sum)
            else:
                cohesive_energies.append(total_e)

        atom_energies_np = None
        if atom_energies:
            atom_energies_np = [ae.numpy() for ae in atom_energies]

        structure_paths = names if any(name is not None for name in names) else None

        return PredictOut(
            coords=coords,
            forces=None,
            atom_types=atom_types,
            atom_energies=atom_energies_np,
            cohesive_energy=cohesive_energies,
            total_energy=energies,
            structure_paths=structure_paths,
            timing=timing,
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
            atomic_energies=(atomic_energies
                             if atomic_energies is not None
                             else self._atomic_energies),
            device=self.device,
            dtype=self.dtype,
        )

        return predictor.cohesive_energy(structure)

    @classmethod
    def from_file(
        cls,
        path,
        device: Optional[str] = None,
    ) -> "TorchANNPotential":
        """
        Load a trained model from file and return a TorchANNPotential.

        Parameters
        ----------
        path : str | Path
            Path to saved model file produced by save_model().
        device : str, optional
            Device to move the model to ('cpu', 'cuda', etc.). By default
            leaves the model on CPU.

        Returns
        -------
        TorchANNPotential
            Loaded trainer instance with metadata accessible via the
            `metadata` property.
        """
        from .model_export import load_model

        trainer, metadata = load_model(path)
        # Store metadata for user access
        trainer._metadata = metadata

        if device is not None:
            trainer.device = torch.device(device)
            trainer.model.to(trainer.device)
            try:
                # Keep descriptor in sync for subsequent featurization
                trainer.descriptor.device = str(trainer.device)
            except Exception:
                pass
        return trainer

    def save(
        self,
        path,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save this trained model and metadata to a file.

        The optimizer state and training configuration are automatically
        included if the model has been trained. These are stored during
        training and ensure the model can be resumed.

        Parameters
        ----------
        path : str | Path
            Destination file (.pt/.pth).
        extra_metadata : dict, optional
            Any additional JSON-serializable metadata to include.
        """
        from .model_export import save_model

        save_model(
            trainer=self,
            path=path,
            optimizer=self._optimizer,
            training_config=self._training_config,
            extra_metadata=extra_metadata,
        )

    def to_aenet_ascii(
        self,
        output_dir,
        prefix: str = "potential",
        descriptor_stats: Optional[Dict[str, Any]] = None,
        structures: Optional[List[Structure]] = None,
        compute_stats: bool = True,
    ):
        """
        Export model to aenet .nn.ascii format (one file per species).

        Parameters
        ----------
        output_dir : str | Path
            Output directory to write files into.
        prefix : str, optional
            Filename prefix. Files are named '{prefix}.{SPECIES}.nn.ascii'.
        descriptor_stats : dict, optional
            Pre-computed stats with keys 'min','max','avg','cov'.
            If provided, avoids computing from structures/normalizer.
        structures : list[Structure], optional
            Structures used to compute exact descriptor statistics and
            training set metadata. Recommended for accurate min/max.
        compute_stats : bool, optional
            If True and descriptor_stats is not provided, compute stats
            from structures. If False, derive from normalizer.

        Returns
        -------
        list[pathlib.Path]
            Paths to written files.
        """
        from .ascii_export import export_to_ascii_impl

        return export_to_ascii_impl(
            trainer=self,
            output_dir=output_dir,
            prefix=prefix,
            descriptor_stats=descriptor_stats,
            structures=structures,
            compute_stats=compute_stats,
        )
