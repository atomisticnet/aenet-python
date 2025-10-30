"""
Helpers for optional runtime dependencies (PyTorch stack).

This module centralizes checks and error messages for optional packages so
subpackages like `aenet.torch_featurize` and `aenet.torch_training` can be
imported safely even when the PyTorch stack is not installed.

Users can install extras with:
    pip install "aenet[torch]"
"""

from __future__ import annotations

from typing import Tuple, Optional


def _safe_import(module: str) -> Tuple[bool, Optional[Exception]]:
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


def torch_status() -> Tuple[bool, str]:
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


def torch_stack_status() -> Tuple[bool, str]:
    """
    Returns (available, reason_if_unavailable) for full featurization stack:
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
            + ". Install with: pip install 'aenet[torch]' "
              "(ensure torch/CPU/CUDA wheels match your system)",
        )
    return True, ""


def require_torch(feature: str = "this feature") -> None:
    ok, reason = torch_status()
    if not ok:
        raise ImportError(
            f"{feature} requires PyTorch. {reason}. "
            "Install with: pip install 'aenet[torch]'"
        )


def require_torch_stack(feature: str = "this feature") -> None:
    ok, reason = torch_stack_status()
    if not ok:
        raise ImportError(
            f"{feature} requires the PyTorch extras. {reason} "
            "Install with: pip install 'aenet[torch]'"
        )
