"""Shared descriptor serialization and recovery helpers for torch training."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from aenet.torch_featurize import ChebyshevDescriptor

DescriptorConfig = dict[str, Any]
DescriptorManifest = dict[str, Any]

DESCRIPTOR_MANIFEST_SCHEMA_VERSION = 1
DESCRIPTOR_MANIFEST_FORMAT = "aenet.torch_training.descriptor_manifest.v1"
CHEBYSHEV_DESCRIPTOR_CLASS = (
    "aenet.torch_featurize.featurize.ChebyshevDescriptor"
)


def dtype_from_str(value: str) -> torch.dtype:
    """Convert a serialized dtype string to a torch dtype."""
    normalized = value.lower().replace("torch.", "")
    if normalized in {"float64", "double"}:
        return torch.float64
    if normalized in {"float32", "float"}:
        return torch.float32
    # Default to float64 for scientific reproducibility.
    return torch.float64


def descriptor_class_path(descriptor) -> str:
    """Return the fully-qualified class path for a descriptor object."""
    return f"{type(descriptor).__module__}.{type(descriptor).__name__}"


def descriptor_config_from_object(descriptor) -> DescriptorConfig:
    """Serialize a descriptor object to a plain constructor-style config."""
    try:
        n_features = int(descriptor.get_n_features())
    except Exception:
        n_features = 0

    return {
        "species": list(getattr(descriptor, "species", [])),
        "rad_order": int(getattr(descriptor, "rad_order", 0)),
        "rad_cutoff": float(getattr(descriptor, "rad_cutoff", 0.0)),
        "ang_order": int(getattr(descriptor, "ang_order", 0)),
        "ang_cutoff": float(getattr(descriptor, "ang_cutoff", 0.0)),
        "min_cutoff": float(getattr(descriptor, "min_cutoff", 0.55)),
        "dtype": str(getattr(descriptor, "dtype", "")).replace("torch.", ""),
        "device": str(getattr(descriptor, "device", "")),
        "n_features": n_features,
    }


def descriptor_compat_signature_from_config(
    config: DescriptorConfig,
    *,
    descriptor_class: str,
) -> dict[str, Any]:
    """
    Build the geometry-relevant compatibility signature for a descriptor.

    Runtime placement details such as dtype and device are intentionally
    excluded so persisted derivative caches can be loaded into a compatible
    descriptor with different runtime placement.
    """
    species = list(config.get("species", []))
    return {
        "descriptor_class": descriptor_class,
        "species": species,
        "rad_order": int(config.get("rad_order", 0)),
        "rad_cutoff": float(config.get("rad_cutoff", 0.0)),
        "ang_order": int(config.get("ang_order", 0)),
        "ang_cutoff": float(config.get("ang_cutoff", 0.0)),
        "min_cutoff": float(config.get("min_cutoff", 0.0)),
        "multi": bool(len(species) > 1),
    }


def descriptor_compat_signature_from_object(descriptor) -> dict[str, Any]:
    """Build the geometry-relevant compatibility signature for an object."""
    return descriptor_compat_signature_from_config(
        descriptor_config_from_object(descriptor),
        descriptor_class=descriptor_class_path(descriptor),
    )


def descriptor_manifest_from_object(descriptor) -> DescriptorManifest:
    """Serialize a descriptor object to a versioned manifest payload."""
    descriptor_class = descriptor_class_path(descriptor)
    config = descriptor_config_from_object(descriptor)
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": DESCRIPTOR_MANIFEST_SCHEMA_VERSION,
        "manifest_format": DESCRIPTOR_MANIFEST_FORMAT,
        "descriptor_class": descriptor_class,
        "config": config,
        "config_json": config_json,
        "config_sha256": hashlib.sha256(
            config_json.encode("utf-8")
        ).hexdigest(),
    }


def descriptor_from_config(
    config: DescriptorConfig,
    *,
    descriptor_class: str = CHEBYSHEV_DESCRIPTOR_CLASS,
):
    """Reconstruct a supported descriptor object from a serialized config."""
    if descriptor_class != CHEBYSHEV_DESCRIPTOR_CLASS:
        raise RuntimeError(
            "Unsupported persisted descriptor class "
            f"{descriptor_class!r}."
        )

    dtype = dtype_from_str(str(config.get("dtype", "float64")))
    device = str(config.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    return ChebyshevDescriptor(
        species=list(config["species"]),
        rad_order=int(config["rad_order"]),
        rad_cutoff=float(config["rad_cutoff"]),
        ang_order=int(config["ang_order"]),
        ang_cutoff=float(config["ang_cutoff"]),
        min_cutoff=float(config.get("min_cutoff", 0.55)),
        device=device,
        dtype=dtype,
    )


def descriptor_from_manifest(manifest: DescriptorManifest):
    """Reconstruct a supported descriptor object from a versioned manifest."""
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version != DESCRIPTOR_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported descriptor manifest schema version "
            f"{schema_version}; expected "
            f"{DESCRIPTOR_MANIFEST_SCHEMA_VERSION}."
        )

    manifest_format = str(manifest.get("manifest_format", ""))
    if manifest_format != DESCRIPTOR_MANIFEST_FORMAT:
        raise RuntimeError(
            "Unsupported descriptor manifest format "
            f"{manifest_format!r}."
        )

    return descriptor_from_config(
        manifest["config"],
        descriptor_class=str(manifest["descriptor_class"]),
    )


def descriptor_matches_manifest(
    descriptor,
    manifest: DescriptorManifest,
) -> bool:
    """Return True when a descriptor matches a persisted manifest."""
    expected = descriptor_compat_signature_from_config(
        manifest["config"],
        descriptor_class=str(manifest["descriptor_class"]),
    )
    actual = descriptor_compat_signature_from_object(descriptor)
    return actual == expected
