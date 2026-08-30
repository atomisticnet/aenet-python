"""Tests for feature-matrix sampling utilities."""

import builtins

import numpy as np
import pytest

from aenet.geometry import random_subset as public_random_subset
from aenet.geometry import (
    representative_subset as public_representative_subset,
)
from aenet.geometry.sampling import random_subset, representative_subset


def clustered_features():
    """Return a tiny feature matrix with three obvious clusters."""
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.9],
            [-10.0, 10.0],
            [-9.8, 10.1],
        ]
    )


def test_random_subset_selects_unique_indices_reproducibly():
    """Random sampling should be unique and reproducible with a seed."""
    features = np.arange(30, dtype=float).reshape(10, 3)

    first = random_subset(features, num_structures=4, random_state=42)
    second = random_subset(features, num_structures=4, random_state=42)

    assert np.array_equal(first, second)
    assert first.shape == (4,)
    assert len(np.unique(first)) == 4
    assert np.all(first[:-1] <= first[1:])


def test_sampling_functions_are_exported_from_geometry():
    """The public geometry namespace should expose both sampling functions."""
    assert public_random_subset is random_subset
    assert public_representative_subset is representative_subset


def test_random_subset_accepts_legacy_random_state():
    """RandomState inputs should be supported for callers using legacy NumPy."""
    features = np.arange(20, dtype=float).reshape(10, 2)

    first = random_subset(
        features,
        num_structures=3,
        random_state=np.random.RandomState(7),
    )
    second = random_subset(
        features,
        num_structures=3,
        random_state=np.random.RandomState(7),
    )

    assert np.array_equal(first, second)


def test_full_size_subset_returns_all_indices():
    """Asking for all rows should not run a down-selection algorithm."""
    features = np.arange(12, dtype=float).reshape(4, 3)

    assert np.array_equal(random_subset(features, 4), np.array([0, 1, 2, 3]))
    assert np.array_equal(
        representative_subset(features, 4),
        np.array([0, 1, 2, 3]),
    )


@pytest.mark.parametrize(
    "bad_features, message",
    [
        ([1.0, 2.0, 3.0], "2D array"),
        (np.ones((2, 2, 2)), "2D array"),
        ([], "2D array"),
        ([[], []], "at least one feature"),
        ([[0.0], [np.nan]], "finite"),
        ([[0.0], [np.inf]], "finite"),
        ([[1.0], [1.0 + 2.0j]], "real-valued"),
        ([[1.0], ["not-a-number"]], "numeric 2D array-like"),
        ([[1.0], [1.0, 2.0]], "numeric 2D array-like"),
    ],
)
def test_sampling_rejects_invalid_representations(bad_features, message):
    """Malformed representation matrices should fail before sampling."""
    with pytest.raises(ValueError, match=message):
        random_subset(bad_features, num_structures=1)


@pytest.mark.parametrize(
    "num_structures, error_type, message",
    [
        (0, ValueError, "positive integer"),
        (-1, ValueError, "positive integer"),
        (1.5, TypeError, "positive integer"),
        (True, TypeError, "positive integer"),
        (4, ValueError, "less than or equal"),
    ],
)
def test_sampling_rejects_invalid_subset_sizes(
    num_structures,
    error_type,
    message,
):
    """Subset sizes should be positive integers within the population."""
    features = np.ones((3, 2))

    with pytest.raises(error_type, match=message):
        random_subset(features, num_structures=num_structures)


def test_representative_subset_selects_one_member_per_cluster():
    """Representative sampling should choose observed rows across clusters."""
    pytest.importorskip("sklearn")
    features = clustered_features()

    selected = representative_subset(
        features,
        num_structures=3,
        random_state=0,
    )

    assert selected.shape == (3,)
    assert len(np.unique(selected)) == 3
    assert np.all(selected[:-1] <= selected[1:])
    assert any(index in selected for index in (0, 1))
    assert any(index in selected for index in (2, 3))
    assert any(index in selected for index in (4, 5))


def test_representative_subset_is_reproducible():
    """Fixed k-means seeds should produce stable representatives."""
    pytest.importorskip("sklearn")
    features = clustered_features()

    first = representative_subset(features, 3, random_state=11)
    second = representative_subset(features, 3, random_state=11)

    assert np.array_equal(first, second)


def test_representative_subset_ties_choose_lowest_source_index():
    """Distance ties should resolve to the lowest source-row index."""
    pytest.importorskip("sklearn")
    features = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [10.0, 0.0],
            [10.2, 0.0],
        ]
    )

    selected = representative_subset(features, 2, random_state=0)

    assert 0 in selected
    assert len(np.unique(selected)) == 2


def test_representative_subset_rejects_degenerate_clusters():
    """Duplicate feature rows can make the requested cluster count impossible."""
    pytest.importorskip("sklearn")
    features = np.ones((4, 2))

    with pytest.raises(ValueError, match="fewer populated clusters"):
        representative_subset(features, 2, random_state=0)


def test_representative_subset_reports_missing_sklearn(monkeypatch):
    """The optional dependency error should tell users how to install it."""
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("blocked sklearn import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"pip install 'aenet\[sampling\]'"):
        representative_subset([[0.0], [1.0], [2.0]], 2, random_state=0)
