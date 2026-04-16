"""
PyTorch-based featurization for aenet.

This subpackage is optional. To use it, install core torch support:
    pip install "aenet[torch]"

PyG-backed featurization also requires `torch-scatter` and `torch-cluster`
from the matching wheel index:
    https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

When the PyTorch stack (torch, torch-scatter, torch-cluster) is not
installed, importing `aenet.torch_featurize` succeeds, but accessing
symbols will raise a clear ImportError with installation guidance.
"""

from importlib import import_module
from typing import Any

from .._optional import (
    TORCH_STACK_INSTALL_HINT,
    is_sphinx_build,
    torch_stack_status,
)

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
_NAME_TO_SPEC: dict[str, tuple[str, str]] = {
    # neighbor list
    "TorchNeighborList": ("..torch_nblist", "TorchNeighborList"),
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
        rel_mod, attr = _NAME_TO_SPEC[name]
        if not is_sphinx_build():
            ok, reason = torch_stack_status()
            if not ok:
                raise ImportError(
                    f"{name} requires the PyTorch extras. {reason} "
                    + TORCH_STACK_INSTALL_HINT
                )
        mod = import_module(rel_mod, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    return sorted(list(globals().keys()) + __all__)
