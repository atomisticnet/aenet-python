"""
Training configuration for PyTorch-based MLIP training.

This module provides configuration classes for PyTorch training that mirror
the API design of aenet.mlip while adding PyTorch-specific features.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np

__all__ = [
    'Structure',
    'TorchTrainingConfig',
    'TrainingMethod',
    'Adam',
    'SGD',
]


@dataclass
class Structure:
    """
    Atomic structure for training.

    Simple dataclass representing a single atomic structure with positions,
    species, energies, and optionally forces.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) array of Cartesian atomic coordinates
    species : list of str
        N species names (element symbols)
    energy : float
        Total energy of the structure
    forces : np.ndarray, optional
        (N, 3) array of atomic forces. None if forces not available.
    cell : np.ndarray, optional
        (3, 3) array of lattice vectors as rows. None for isolated systems.
    pbc : np.ndarray, optional
        (3,) boolean array of periodic boundary conditions. None for isolated.
    name : str, optional
        Structure identifier or filename
    """

    positions: np.ndarray
    species: list
    energy: float
    forces: Optional[np.ndarray] = None
    cell: Optional[np.ndarray] = None
    pbc: Optional[np.ndarray] = None
    name: Optional[str] = None

    def __post_init__(self):
        """Validate structure data."""
        # Validate positions shape
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (N, 3), got {self.positions.shape}"
            )

        # Validate species length
        if len(self.species) != len(self.positions):
            raise ValueError(
                f"species length ({len(self.species)}) must match "
                f"positions length ({len(self.positions)})"
            )

        # Validate forces if provided
        if self.forces is not None:
            if self.forces.shape != self.positions.shape:
                raise ValueError(
                    f"forces shape {self.forces.shape} must match "
                    f"positions shape {self.positions.shape}"
                )

        # Validate cell if provided
        if self.cell is not None:
            if self.cell.shape != (3, 3):
                raise ValueError(
                    f"cell must be (3, 3), got {self.cell.shape}"
                )

        # Validate pbc if provided
        if self.pbc is not None:
            if len(self.pbc) != 3:
                raise ValueError(
                    f"pbc must have length 3, got {len(self.pbc)}"
                )

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the structure."""
        return len(self.positions)

    def has_forces(self) -> bool:
        """Check if structure has force data."""
        return self.forces is not None

    def is_periodic(self) -> bool:
        """Check if structure is periodic."""
        return self.cell is not None


@dataclass
class TrainingMethod:
    """
    Base class for training method configurations.

    Each training method subclass encodes both the method name and its
    parameters with appropriate defaults.
    """

    @property
    def method_name(self) -> str:
        """Return the method identifier."""
        raise NotImplementedError

    def to_params_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to parameter dictionary.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


@dataclass
class Adam(TrainingMethod):
    """
    Adam optimizer.

    Parameters
    ----------
    mu : float, optional
        Learning rate. Default: 0.001
    batchsize : int, optional
        Number of structures per batch. Default: 32
    weight_decay : float, optional
        L2 regularization coefficient. Default: 0.0
    beta1 : float, optional
        Exponential decay rate for first moment estimates. Default: 0.9
    beta2 : float, optional
        Exponential decay rate for second moment estimates. Default: 0.999
    epsilon : float, optional
        Small constant for numerical stability. Default: 1e-8
    """

    mu: float = 0.001
    batchsize: int = 32
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    @property
    def method_name(self) -> str:
        """Return the optimizer method name."""
        return "adam"


@dataclass
class SGD(TrainingMethod):
    """
    Stochastic gradient descent optimizer.

    Parameters
    ----------
    lr : float, optional
        Learning rate. Default: 0.01
    batchsize : int, optional
        Number of structures per batch. Default: 32
    momentum : float, optional
        Momentum factor. Default: 0.0
    weight_decay : float, optional
        L2 regularization coefficient. Default: 0.0
    """

    lr: float = 0.01
    batchsize: int = 32
    momentum: float = 0.0
    weight_decay: float = 0.0

    @property
    def method_name(self) -> str:
        """Return the optimizer method name."""
        return "sgd"


@dataclass
class TorchTrainingConfig:
    """
    Configuration for PyTorch ANN potential training.

    Centralizes all training parameters with built-in validation.
    Mirrors the aenet.mlip.TrainingConfig API with PyTorch-specific additions.

    Parameters
    ----------
    iterations : int, optional
        Maximum number of training epochs. Default: 100
    method : TrainingMethod, optional
        Training method configuration (Adam or SGD).
        Default: None (will use Adam with defaults)
    testpercent : int, optional
        Percentage of data for validation set (0-100). Default: 10
    force_weight : float, optional
        Weight for force loss (alpha parameter). 0 = energy only, 1 = force
        only. Default: 0.0
    force_fraction : float, optional
        Fraction of structures to use for force training (0.0-1.0).
        Default: 1.0 (all structures with forces)
    force_sampling : str, optional
        Force sampling strategy: 'random' (resample each epoch) or 'fixed'
        (fixed subset). Default: 'random'
    force_resample_each_epoch : bool, optional
        When force_sampling='random', resample the subset at the beginning
        of each epoch. Default: True
    force_min_structures_per_epoch : int, optional
        Minimum number of force-labeled structures to include per epoch,
        regardless of force_fraction. Default: 1
    force_scale_unbiased : bool, optional
        If True, apply sqrt(1/f) scaling to the per-batch force RMSE, where
        f is the supervised fraction of atoms from force-labeled structures,
        to approximate a constant scale under sub-sampling. Default: False
    cached_features_for_force : bool, optional
        When alpha>0, cache features for structures that are not selected
        for force supervision in the current epoch-window. For those
        energy-only structures, reuse cached features and skip
        neighbor_info computation.  Default: False
    cache_neighbors : bool, optional
        Cache per-structure neighbor graphs (indices and displacement vectors)
        to avoid repeated neighbor searches across epochs when geometries are
        fixed. When True, neighbor graphs are computed once per structure and
        reused for both energy and force paths. Default: False
    cache_triplets : bool, optional
        Build and cache CSR neighbor graphs and precomputed angular triplet
        indices per structure to enable vectorized featurization and gradient
        paths (removes Python-level enumeration loops). Default: False
    cache_persist_dir : str, optional
        Optional root directory for persisted graph/triplet caches
        (planned follow-up). When provided, caches may be serialized to disk
        and reloaded across runs. Default: None
    cache_scope : {'train', 'val', 'all'}, optional
        Scope limiting which dataset split(s) should be cached/persisted,
        allowing memory/I-O control. Default: 'all'
    epochs_per_force_window : int, optional
        Resample the random subset of force-supervised structures every this
        many epochs (when force_sampling='random'). 1 = resample every epoch
        (default behavior). Values >1 amortize cached features across multiple
        epochs. Default: 1
    memory_mode : str, optional
        Memory management strategy: 'cpu', 'gpu', or 'mixed'.
        Default: 'gpu'
    max_energy : float, optional
        Exclude structures with energy above this threshold. Default: None
    max_forces : float, optional
        Exclude structures with max force above this threshold. Default: None
    save_energies : bool, optional
        Save predicted energies for train/test sets. Default: False
    save_forces : bool, optional
        Save predicted forces for train/test sets. Default: False
    timing : bool, optional
        Enable detailed timing output. Default: False
    device : str, optional
        Device to use ('cpu', 'cuda', or 'cuda:X'). Auto-detect if None.
        Default: None

    Raises
    ------
    ValueError
        If parameters are out of valid ranges.
    """

    iterations: int = 100
    method: Optional[TrainingMethod] = None
    testpercent: int = 10
    force_weight: float = 0.0
    force_fraction: float = 1.0
    force_sampling: Literal['random', 'fixed'] = 'random'
    # Force subsampling controls
    force_resample_each_epoch: bool = True
    force_min_structures_per_epoch: Optional[int] = 1
    force_scale_unbiased: bool = False
    # Mixed-run caching: cache features for non-force structures
    # in current window
    cached_features_for_force: bool = False
    # Cache per-structure neighbor graphs to avoid repeated neighbor
    # searches across epochs (indices and displacement vectors)
    cache_neighbors: bool = False
    # CSR + Triplet caching/vectorization (Issue 5 Phase 2 / Issue 7)
    cache_triplets: bool = False
    # Optional on-disk persistence root (Phase 4 follow-up may enable writing)
    cache_persist_dir: Optional[str] = None
    # Scope for caching/persistence
    cache_scope: Literal['train', 'val', 'all'] = 'all'
    # Resample random force subset every this many epochs (1 = each epoch)
    epochs_per_force_window: int = 1
    memory_mode: Literal['cpu', 'gpu', 'mixed'] = 'gpu'
    max_energy: Optional[float] = None
    max_forces: Optional[float] = None
    save_energies: bool = False
    save_forces: bool = False
    timing: bool = False
    device: Optional[str] = None
    # Default numeric precision control for training/inference
    precision: Literal['auto', 'float32', 'float64'] = 'auto'
    # Energy target space and atomic reference energies
    energy_target: Literal['cohesive', 'total'] = 'cohesive'
    E_atomic: Optional[Dict[str, float]] = None
    # Normalization controls (defaults match aenet-Fortran/PyTorch behavior)
    normalize_features: bool = True
    normalize_energy: bool = True
    # Cached features path (energy-only optimization)
    cached_features: bool = False
    # DataLoader parallelism (on-the-fly featurization)
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    # Optional overrides/stats (if not provided, computed from training split)
    # feature statistics format:
    # {'mean': np.ndarray[F], 'std': np.ndarray[F]}
    feature_stats: Optional[Dict[str, Any]] = None
    E_shift: Optional[float] = None  # per-atom shift
    E_scaling: Optional[float] = None  # energy scaling
    # Progress display
    show_progress: bool = True
    show_batch_progress: bool = False

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate testpercent
        if not 0 <= self.testpercent <= 100:
            raise ValueError(
                f"testpercent must be 0-100, got {self.testpercent}"
            )

        # Validate force_weight
        if not 0.0 <= self.force_weight <= 1.0:
            raise ValueError(
                f"force_weight must be 0.0-1.0, got {self.force_weight}"
            )

        # Validate force_fraction
        if not 0.0 <= self.force_fraction <= 1.0:
            raise ValueError(
                f"force_fraction must be 0.0-1.0, got {self.force_fraction}"
            )

        # Validate force_sampling
        if self.force_sampling not in ['random', 'fixed']:
            raise ValueError(
                f"force_sampling must be 'random' or 'fixed', "
                f"got '{self.force_sampling}'"
            )

        # Validate force_min_structures_per_epoch
        if (self.force_min_structures_per_epoch is not None
                and self.force_min_structures_per_epoch < 0):
            raise ValueError(
                f"force_min_structures_per_epoch must be >= 0 or None, "
                f"got {self.force_min_structures_per_epoch}"
            )

        # Validate memory_mode
        if self.memory_mode not in ['cpu', 'gpu', 'mixed']:
            raise ValueError(
                f"memory_mode must be 'cpu', 'gpu', or 'mixed', "
                f"got '{self.memory_mode}'"
            )

        # Validate cache_scope
        if self.cache_scope not in ['train', 'val', 'all']:
            raise ValueError(
                f"cache_scope must be 'train', 'val', or 'all', "
                f"got '{self.cache_scope}'"
            )

        # Validate epochs_per_force_window
        if int(self.epochs_per_force_window) < 1:
            raise ValueError(
                f"epochs_per_force_window must be >= 1, "
                f"got {self.epochs_per_force_window}"
            )

        # Validate energy_target
        if self.energy_target not in ['cohesive', 'total']:
            raise ValueError(
                f"energy_target must be 'cohesive' or 'total', "
                f"got '{self.energy_target}'"
            )

        # Validate precision
        if self.precision not in ['auto', 'float32', 'float64']:
            raise ValueError(
                f"precision must be 'auto', 'float32', or 'float64', "
                f"got '{self.precision}'"
            )

        # Validate iterations
        if self.iterations < 0:
            raise ValueError(
                f"iterations must be >= 0, got {self.iterations}"
            )
        # Validate DataLoader workers
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be >= 0, got {self.num_workers}"
            )
        if self.prefetch_factor < 1:
            raise ValueError(
                f"prefetch_factor must be >= 1, got {self.prefetch_factor}"
            )

        # Default method to Adam if not provided
        if self.method is None:
            self.method = Adam()

    @property
    def alpha(self) -> float:
        """Alias for force_weight (matches aenet-PyTorch naming)."""
        return self.force_weight

    @property
    def batch_size(self) -> int:
        """Get batch size from method configuration."""
        if hasattr(self.method, 'batchsize'):
            return self.method.batchsize
        elif hasattr(self.method, 'batch_size'):
            return self.method.batch_size
        else:
            return 32  # Default fallback
