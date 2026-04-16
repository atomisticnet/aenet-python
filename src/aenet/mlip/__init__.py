"""
Machine-learning interatomic potential (MLIP) interfaces.

This module provides interfaces to aenet neural network potentials:
- ANNPotential: Main class for training and prediction (subprocess-based)
- LibAenetInterface: High-level library interface for direct libaenet calls
- AenetEnsembleInterface: Committee-based libaenet inference
- AenetCalculator: ASE Calculator integration (optional)
- AenetEnsembleCalculator: ASE committee-based inference

"""

from importlib import import_module

from .. import config as cfg
from ..trainset import TrnSet

# Import main classes from potential module for backward compatibility
from .potential import (
    BFGS,
    EKF,
    LM,
    Activation,
    Adam,
    ANNArchitecture,
    ANNPotential,
    OnlineSD,
    PredictionConfig,
    TrainingConfig,
    TrainingMethod,
)

__all__ = [
    "ANNPotential",
    "PredictionConfig",
    "TrainingConfig",
    "TrainingMethod",
    "OnlineSD",
    "Adam",
    "EKF",
    "LM",
    "BFGS",
    "Activation",
    "ANNArchitecture",
    "LibAenetInterface",
    "AenetEnsembleInterface",
    "AenetCalculator",
    "AenetEnsembleCalculator",
    "TrnSet",
    "cfg",
]

_LAZY_EXPORTS = {
    "LibAenetInterface": (".interface", "LibAenetInterface"),
    "AenetEnsembleInterface": (".interface", "AenetEnsembleInterface"),
    "AenetCalculator": (".calculator", "AenetCalculator"),
    "AenetEnsembleCalculator": (
        ".calculator",
        "AenetEnsembleCalculator",
    ),
}


def __getattr__(name):
    """Lazily import optional libaenet-backed interfaces."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
