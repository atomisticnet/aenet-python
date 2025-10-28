"""
PyTorch-based training for machine-learning interatomic potentials.

This module provides PyTorch training capabilities with on-the-fly
featurization, leveraging the semi-analytical gradients from torch_featurize.
"""

from .config import (
    SGD,
    Adam,
    Structure,
    TorchTrainingConfig,
    TrainingMethod,
)
from .dataset import StructureDataset, train_test_split
from .model_adapter import EnergyModelAdapter
from .trainer import TorchANNPotential
from .model_export import save_model, load_model, export_history

__all__ = [
    'Structure',
    'TorchTrainingConfig',
    'TrainingMethod',
    'Adam',
    'SGD',
    # Datasets and helpers
    'StructureDataset',
    'train_test_split',
    # Model and trainer
    'EnergyModelAdapter',
    'TorchANNPotential',
    # Model I/O
    'save_model',
    'load_model',
    'export_history',
]

__version__ = '0.1.0'
