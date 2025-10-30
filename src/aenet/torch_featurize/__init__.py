"""
PyTorch-based featurization for aenet.

This subpackage is optional. To use it, install the PyTorch extras:
    pip install "aenet[torch]"

When the PyTorch stack (torch, torch-scatter, torch-cluster) is not
installed, importing `aenet.torch_featurize` succeeds, but accessing
symbols will raise a clear ImportError with installation guidance.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

from .._optional import torch_stack_status

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

# Map exported names to (relative_module, attr_name)
_NAME_TO_SPEC: Dict[str, Tuple[str, str]] = {
    # neighbor list
    "TorchNeighborList": (".neighborlist", "TorchNeighborList"),
    # chebyshev bases
    "ChebyshevPolynomials": (".chebyshev", "ChebyshevPolynomials"),
    "RadialBasis": (".chebyshev", "RadialBasis"),
    "AngularBasis": (".chebyshev", "AngularBasis"),
    # featurization
    "ChebyshevDescriptor": (".featurize", "ChebyshevDescriptor"),
    "BatchedFeaturizer": (".featurize", "BatchedFeaturizer"),
    # HDF5 helpers
    "TorchAUCFeaturizer": (".hdf5_compat", "TorchAUCFeaturizer"),
    "write_features_to_hdf5": (".hdf5_compat", "write_features_to_hdf5"),
    "featurize_and_write_hdf5": (".hdf5_compat", "featurize_and_write_hdf5"),
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name in _NAME_TO_SPEC:
        ok, reason = torch_stack_status()
        if not ok:
            raise ImportError(
                f"{name} requires the PyTorch extras. {reason} "
                "Install with: pip install 'aenet[torch]'"
            )
        rel_mod, attr = _NAME_TO_SPEC[name]
        mod = import_module(rel_mod, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    return sorted(list(globals().keys()) + __all__)
