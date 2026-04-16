"""Regression tests for Sphinx-safe optional imports."""

from types import SimpleNamespace

import pytest

import aenet.mlip as mlip
import aenet.torch_featurize as torch_featurize
import aenet.torch_training as torch_training


def test_torch_training_lazy_exports_require_torch_outside_sphinx(
    monkeypatch,
):
    """Torch training exports should still fail fast outside Sphinx."""

    def _raise_require_torch(*, feature):
        raise ImportError(f"{feature} requires PyTorch")

    monkeypatch.setattr(
        torch_training,
        "is_sphinx_build",
        lambda: False,
    )
    monkeypatch.setattr(
        torch_training,
        "require_torch",
        _raise_require_torch,
    )

    with pytest.raises(ImportError, match="TorchCommitteeConfig requires"):
        getattr(torch_training, "TorchCommitteeConfig")


def test_torch_training_lazy_exports_defer_to_sphinx_imports(monkeypatch):
    """Torch training exports should defer to Sphinx-time mocked imports."""
    sentinel = object()

    monkeypatch.setattr(
        torch_training,
        "is_sphinx_build",
        lambda: True,
    )
    monkeypatch.setattr(
        torch_training,
        "require_torch",
        lambda *, feature: pytest.fail(
            f"require_torch called during Sphinx build for {feature}"
        ),
    )
    monkeypatch.setattr(
        torch_training,
        "import_module",
        lambda rel_mod, package: SimpleNamespace(
            TorchCommitteeConfig=sentinel
        ),
    )

    assert getattr(torch_training, "TorchCommitteeConfig") is sentinel


def test_torch_featurize_lazy_exports_require_stack_outside_sphinx(
    monkeypatch,
):
    """Torch featurize exports should still fail fast outside Sphinx."""
    monkeypatch.setattr(
        torch_featurize,
        "is_sphinx_build",
        lambda: False,
    )
    monkeypatch.setattr(
        torch_featurize,
        "torch_stack_status",
        lambda: (False, "PyTorch stack unavailable"),
    )

    with pytest.raises(ImportError, match="AngularBasis requires"):
        getattr(torch_featurize, "AngularBasis")


def test_torch_featurize_lazy_exports_defer_to_sphinx_imports(monkeypatch):
    """Torch featurize exports should defer to Sphinx-time mocked imports."""
    sentinel = object()

    monkeypatch.setattr(
        torch_featurize,
        "is_sphinx_build",
        lambda: True,
    )
    monkeypatch.setattr(
        torch_featurize,
        "torch_stack_status",
        lambda: pytest.fail("torch_stack_status called during Sphinx build"),
    )
    monkeypatch.setattr(
        torch_featurize,
        "import_module",
        lambda rel_mod, package: SimpleNamespace(AngularBasis=sentinel),
    )

    assert getattr(torch_featurize, "AngularBasis") is sentinel


def test_mlip_optional_interfaces_are_imported_lazily(monkeypatch):
    """MLIP should not import libaenet-backed interfaces eagerly."""
    sentinel = object()

    monkeypatch.setattr(
        mlip,
        "import_module",
        lambda rel_mod, package: SimpleNamespace(LibAenetInterface=sentinel),
    )

    assert getattr(mlip, "LibAenetInterface") is sentinel
