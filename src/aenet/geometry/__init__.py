"""
Geometry module for atomic structure manipulation.

This module has been refactored into a package. The core classes
(AtomicStructure, interpolate) are still available at the same import path.

Utility functions have been moved to aenet.geometry.utils and should be
imported from there in new code.
"""

from .structure import AtomicStructure
from .interpolation import interpolate
from . import utils

__all__ = ['AtomicStructure', 'interpolate', 'utils']


def __getattr__(name):
    """
    Provide deprecated access to utility functions.

    This allows backward compatibility for code that imports utilities
    directly from aenet.geometry, while warning users to update their
    import paths.
    """
    if hasattr(utils, name):
        import warnings
        warnings.warn(
            f"Importing '{name}' from aenet.geometry is deprecated and "
            f"will be removed in a future version. "
            f"Use 'from aenet.geometry.utils import {name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return getattr(utils, name)
    raise AttributeError(f"module 'aenet.geometry' has no attribute '{name}'")
