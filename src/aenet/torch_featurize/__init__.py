"""
PyTorch-based featurization for aenet.

This module provides GPU-accelerated implementations of neighbor list
construction, Chebyshev polynomial evaluation, and atomic featurization.
"""

from .chebyshev import AngularBasis, ChebyshevPolynomials, RadialBasis
from .featurize import BatchedFeaturizer, ChebyshevDescriptor
from .hdf5_compat import (
    TorchAUCFeaturizer,
    featurize_and_write_hdf5,
    write_features_to_hdf5,
)
from .neighborlist import TorchNeighborList

__all__ = [
    "TorchNeighborList",
    "ChebyshevPolynomials",
    "RadialBasis",
    "AngularBasis",
    "ChebyshevDescriptor",
    "BatchedFeaturizer",
    "TorchAUCFeaturizer",
    "write_features_to_hdf5",
    "featurize_and_write_hdf5",
]
