"""
PyTorch-based featurization for aenet.

This module provides GPU-accelerated implementations of neighbor list
construction, Chebyshev polynomial evaluation, and atomic featurization.
"""

from .neighborlist import TorchNeighborList

__all__ = ['TorchNeighborList']
