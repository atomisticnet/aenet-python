"""
Machine-learning interatomic potential (MLIP) interfaces.

This module provides interfaces to aenet neural network potentials:
- ANNPotential: Main class for training and prediction (subprocess-based)
- LibAenetInterface: High-level library interface for direct libaenet calls
- AenetCalculator: ASE Calculator integration (optional)

"""

# Import main classes from potential module for backward compatibility
from .potential import (
    ANNPotential,
    PredictionConfig,
    TrainingConfig,
    TrainingMethod,
    OnlineSD,
    Adam,
    EKF,
    LM,
    BFGS,
    Activation,
    ANNArchitecture,
)

# Import library interface and ASE calculator
from .interface import LibAenetInterface
from .calculator import AenetCalculator

# Legacy re-exports expected by tests and downstream callers
from ..trainset import TrnSet
from .. import config as cfg

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
    "AenetCalculator",
    "TrnSet",
    "cfg",
]
