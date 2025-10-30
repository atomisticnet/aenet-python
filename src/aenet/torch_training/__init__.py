"""
PyTorch-based training for machine-learning interatomic potentials.

This subpackage is optional. To use it, install PyTorch:
    pip install "aenet[torch]"

When PyTorch is not installed, importing `aenet.torch_training` succeeds,
but accessing symbols will raise a clear ImportError with installation
guidance.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

from .._optional import require_torch

__all__ = [
    "Structure",
    "TorchTrainingConfig",
    "TrainingMethod",
    "Adam",
    "SGD",
    # Datasets and helpers
    "StructureDataset",
    "train_test_split",
    "HDF5StructureDataset",
    "train_test_split_dataset",
    # Model and trainer
    "EnergyModelAdapter",
    "TorchANNPotential",
    # Model I/O
    "save_model",
    "load_model",
    "export_history",
    # Modular components
    "NetworkBuilder",
    "OptimizerBuilder",
    "CheckpointManager",
    "MetricsTracker",
    "NormalizationManager",
    "TrainingLoop",
    "Predictor",
]

# Map exported names to (relative_module, attr_name)
_NAME_TO_SPEC: Dict[str, Tuple[str, str]] = {
    # config
    "SGD": (".config", "SGD"),
    "Adam": (".config", "Adam"),
    "Structure": (".config", "Structure"),
    "TorchTrainingConfig": (".config", "TorchTrainingConfig"),
    "TrainingMethod": (".config", "TrainingMethod"),
    # dataset
    "StructureDataset": (".dataset", "StructureDataset"),
    "train_test_split": (".dataset", "train_test_split"),
    "HDF5StructureDataset": (".hdf5_dataset", "HDF5StructureDataset"),
    "train_test_split_dataset": (".dataset", "train_test_split_dataset"),
    # model adapter and trainer
    "EnergyModelAdapter": (".model_adapter", "EnergyModelAdapter"),
    "TorchANNPotential": (".trainer", "TorchANNPotential"),
    # model export
    "save_model": (".model_export", "save_model"),
    "load_model": (".model_export", "load_model"),
    "export_history": (".model_export", "export_history"),
    # builders
    "NetworkBuilder": (".builders.network_builder", "NetworkBuilder"),
    "OptimizerBuilder": (".builders.optimizer_builder", "OptimizerBuilder"),
    # training components
    "CheckpointManager": (".training.checkpoint_manager", "CheckpointManager"),
    "MetricsTracker": (".training.metrics", "MetricsTracker"),
    "NormalizationManager": (
        ".training.normalization", "NormalizationManager"),
    "TrainingLoop": (".training.training_loop", "TrainingLoop"),
    # inference
    "Predictor": (".inference.predictor", "Predictor"),
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name in _NAME_TO_SPEC:
        # Only core torch is required for training
        require_torch(feature=f"{name}")
        rel_mod, attr = _NAME_TO_SPEC[name]
        mod = import_module(rel_mod, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    return sorted(list(globals().keys()) + __all__)


__version__ = "0.1.0"
