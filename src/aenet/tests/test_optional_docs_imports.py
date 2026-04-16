"""Regression tests for Sphinx-safe optional imports."""

import builtins
import importlib
from types import SimpleNamespace

import pytest

import aenet._optional as optional
import aenet.torch_featurize as torch_featurize
import aenet.torch_training as torch_training
import aenet.torch_training.committee as torch_committee


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


def test_committee_module_accepts_sphinx_torch_data_fallback(monkeypatch):
    """Committee docs imports should survive missing torch.utils.data."""
    real_import = builtins.__import__

    def _import_with_missing_torch_data(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        if name == "torch.utils.data":
            raise ImportError("blocked import: torch.utils.data")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(optional, "is_sphinx_build", lambda: True)
    monkeypatch.setattr(builtins, "__import__", _import_with_missing_torch_data)

    reloaded = importlib.reload(torch_committee)

    try:
        subset = reloaded.Subset(dataset="dataset", indices=[1, 2])
        assert subset.dataset == "dataset"
        assert subset.indices == [1, 2]
    finally:
        importlib.reload(torch_committee)
