"""
PyTorch-based neighbor list implementation for atomic structures.

This module provides efficient neighbor list computation with support for:
- Periodic boundary conditions (PBC)
- Isolated systems (molecules)
- GPU acceleration
- Full differentiability for gradients

Main class: TorchNeighborList
"""

from .neighborlist import TorchNeighborList

__all__ = ["TorchNeighborList"]
