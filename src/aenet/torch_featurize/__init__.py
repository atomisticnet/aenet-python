"""
PyTorch-based featurization for aenet.

This module provides GPU-accelerated implementations of neighbor list
construction, Chebyshev polynomial evaluation, and atomic featurization.
"""

from .neighborlist import TorchNeighborList
from .chebyshev import (
    ChebyshevPolynomials,
    RadialBasis,
    AngularBasis
)
from .featurize import (
    ChebyshevDescriptor,
    BatchedFeaturizer
)

__all__ = [
    'TorchNeighborList',
    'ChebyshevPolynomials',
    'RadialBasis',
    'AngularBasis',
    'ChebyshevDescriptor',
    'BatchedFeaturizer'
]
