"""
Helpers for optional runtime dependencies (PyTorch stack).

This module centralizes checks and error messages for optional packages so
subpackages like `aenet.torch_featurize` and `aenet.torch_training` can be
imported safely even when the PyTorch stack is not installed.

Users can install core torch support with:
    pip install "aenet[torch]"

PyG-backed features additionally require `torch-scatter` and
`torch-cluster` from the matching wheel index at:
    https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
"""

from __future__ import annotations

import sys

PYG_WHEEL_INDEX = "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
TORCH_INSTALL_HINT = "pip install 'aenet[torch]'"
TORCH_STACK_INSTALL_HINT = (
    "Install core torch with: "
    f"{TORCH_INSTALL_HINT}. "
    "Then install torch-scatter and torch-cluster from the matching "
    f"PyG wheel index: {PYG_WHEEL_INDEX}"
)


def _safe_import(module: str) -> tuple[bool, Exception | None]:
    try:
        __import__(module)
        return True, None
    except Exception as e:  # pragma: no cover
        return False, e


def is_torch_available() -> bool:
    ok, _ = _safe_import("torch")
    return ok


def is_torch_scatter_available() -> bool:
    ok, _ = _safe_import("torch_scatter")
    return ok


def is_torch_cluster_available() -> bool:
    ok, _ = _safe_import("torch_cluster")
    return ok


def torch_status() -> tuple[bool, str]:
    """
    Returns (available, reason_if_unavailable).

    Only checks core PyTorch.
    """
    ok, err = _safe_import("torch")
    if ok:
        return True, ""
    reason = (f"PyTorch not installed ({err})"
              if err else "PyTorch not installed")
    return False, reason


def torch_stack_status() -> tuple[bool, str]:
    """
    Returns (available, reason_if_unavailable) for full featurization stack.

      - torch
      - torch-scatter
      - torch-cluster
    """
    missing = []
    ok, err = _safe_import("torch")
    if not ok:
        return (False, f"PyTorch not installed ({err})"
                if err else "PyTorch not installed")

    ok, err = _safe_import("torch_scatter")
    if not ok:
        missing.append("torch-scatter")
    ok, err = _safe_import("torch_cluster")
    if not ok:
        missing.append("torch-cluster")

    if missing:
        return (
            False,
            "Missing optional dependencies: "
            + ", ".join(missing)
            + ". "
            + TORCH_STACK_INSTALL_HINT
            + " (ensure torch/CPU/CUDA wheels match your system)",
        )
    return True, ""


def is_sphinx_build() -> bool:
    """
    Return whether execution is happening inside a Sphinx build.

    Sphinx autodoc/autosummary can provide mocked optional dependencies.
    Package-level lazy exports should defer to those mocks instead of raising
    an eager ImportError before autodoc gets a chance to import the target
    module.
    """
    return "sphinx" in sys.modules


def require_torch(feature: str = "this feature") -> None:
    ok, reason = torch_status()
    if not ok:
        raise ImportError(
            f"{feature} requires PyTorch. {reason}. "
            f"Install with: {TORCH_INSTALL_HINT}"
        )


def require_torch_stack(feature: str = "this feature") -> None:
    ok, reason = torch_stack_status()
    if not ok:
        raise ImportError(
            f"{feature} requires the PyTorch extras. {reason} "
            + TORCH_STACK_INSTALL_HINT
        )
