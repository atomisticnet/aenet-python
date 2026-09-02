"""Sampling utilities for structure selection and local augmentation.

The subset-selection functions operate on representation matrices and return
source-row indices. Taylor sampling instead generates force-informed local
configurations through the public geometry transformations.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from ._taylor_sampling import (
    TaylorExpansionConfig,
    TaylorReference,
    TaylorSampleRecord,
    TaylorSamplingResult,
    generate_taylor_samples,
    iter_taylor_records,
    iter_taylor_structures,
    split_reference_structures,
    taylor_energy,
)

__all__ = [
    "TaylorExpansionConfig",
    "TaylorReference",
    "TaylorSampleRecord",
    "TaylorSamplingResult",
    "generate_taylor_samples",
    "iter_taylor_records",
    "iter_taylor_structures",
    "random_subset",
    "representative_subset",
    "split_reference_structures",
    "taylor_energy",
]

SKLEARN_INSTALL_HINT = "pip install 'aenet[sampling]'"


def _validate_population(
    representations: Any,
    num_structures: int,
) -> np.ndarray:
    """Return an array with a validated structure-row population."""
    if not isinstance(num_structures, Integral) or isinstance(
        num_structures,
        bool,
    ):
        raise TypeError("num_structures must be a positive integer")
    if num_structures <= 0:
        raise ValueError("num_structures must be a positive integer")

    try:
        raw = np.asarray(representations)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "representations must be a numeric 2D array-like object"
        ) from exc
    if raw.ndim != 2:
        raise ValueError(
            "representations must be a 2D array with one row per structure"
        )
    if raw.shape[0] == 0:
        raise ValueError("representations must contain at least one row")
    if raw.shape[1] == 0:
        raise ValueError("representations must contain at least one feature")
    if num_structures > raw.shape[0]:
        raise ValueError(
            "num_structures must be less than or equal to the number of rows"
        )

    return raw


def _validate_representations(
    representations: Any,
    num_structures: int,
) -> np.ndarray:
    """Return a validated floating-point representation matrix."""
    raw = _validate_population(representations, num_structures)
    if np.iscomplexobj(raw):
        raise ValueError("representations must be real-valued")

    try:
        features = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "representations must be a numeric 2D array-like object"
        ) from exc

    if not np.all(np.isfinite(features)):
        raise ValueError("representations must contain only finite values")

    return features


def _sorted_all_indices(n_samples: int) -> np.ndarray:
    """Return all source indices in source order."""
    return np.arange(n_samples, dtype=np.intp)


def random_subset(
    representations: Any,
    num_structures: int,
    random_state: Any = None,
) -> np.ndarray:
    """Select a reproducible random subset of representation rows.

    Parameters
    ----------
    representations : array-like of shape (n_structures, n_features)
        Representation matrix with one row per source structure.  Only its
        two-dimensional shape and row count are inspected; representation
        values do not affect random sampling.
    num_structures : int
        Number of source rows to select.  Must be positive and no larger than
        the number of rows in ``representations``.
    random_state : None, int, numpy.random.Generator, or numpy.random.RandomState
        Random-state control.  Integer seeds create an independent NumPy
        generator.  Caller-provided generator/state objects are advanced.

    Returns
    -------
    numpy.ndarray
        One-dimensional integer array containing selected source-row indices in
        ascending source order.
    """
    population = _validate_population(representations, num_structures)
    n_samples = population.shape[0]
    if num_structures == n_samples:
        return _sorted_all_indices(n_samples)

    if isinstance(random_state, np.random.RandomState):
        selected = random_state.choice(
            n_samples,
            size=num_structures,
            replace=False,
        )
    else:
        rng = np.random.default_rng(random_state)
        selected = rng.choice(n_samples, size=num_structures, replace=False)

    return np.sort(np.asarray(selected, dtype=np.intp))


def representative_subset(
    representations: Any,
    num_structures: int,
    random_state: Any = None,
    n_init: int | str = 10,
) -> np.ndarray:
    """Select representative source rows by k-means centroid assignment.

    The function fits k-means to the supplied numeric representation matrix
    using ``num_structures`` clusters.  For each populated cluster, it returns
    the observed source row nearest to that cluster centroid.  It does not
    generate geometries, compute descriptors, scale features, or return
    centroid vectors.

    Parameters
    ----------
    representations : array-like of shape (n_structures, n_features)
        Numeric representation matrix with one row per source structure.
        Callers are responsible for any descriptor calculation or feature
        scaling before calling this function.
    num_structures : int
        Number of representative source rows to select.  Must be positive and
        no larger than the number of rows in ``representations``.
    random_state : None, int, or scikit-learn-compatible random state
        Random-state control passed to ``sklearn.cluster.KMeans``.
    n_init : int or "auto", default=10
        Number of k-means initializations passed to
        ``sklearn.cluster.KMeans``.  The integer default keeps behavior
        explicit across supported scikit-learn versions.

    Returns
    -------
    numpy.ndarray
        One-dimensional integer array containing selected source-row indices in
        ascending source order.

    Notes
    -----
    If multiple structures have exactly the same computed distance to a
    centroid, the structure with the lowest source index is selected.

    Raises
    ------
    ImportError
        If scikit-learn is unavailable.  Install the optional sampling
        dependency with ``pip install 'aenet[sampling]'``.
    ValueError
        If the input matrix is invalid or k-means produces fewer populated
        clusters than requested.
    """
    features = _validate_representations(representations, num_structures)
    n_samples = features.shape[0]
    if num_structures == n_samples:
        return _sorted_all_indices(n_samples)

    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            "representative_subset requires scikit-learn. "
            f"Install it with: {SKLEARN_INSTALL_HINT}"
        ) from exc

    model = KMeans(
        n_clusters=num_structures,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(features)
    centers = model.cluster_centers_

    selected: list[int] = []
    for cluster_id in range(num_structures):
        member_indices = np.flatnonzero(labels == cluster_id)
        if member_indices.size == 0:
            raise ValueError(
                "k-means produced fewer populated clusters than "
                "num_structures; reduce num_structures or provide less "
                "degenerate representations"
            )

        member_features = features[member_indices]
        distances = np.linalg.norm(
            member_features - centers[cluster_id],
            axis=1,
        )
        selected.append(int(member_indices[np.argmin(distances)]))

    unique_selected = np.unique(np.asarray(selected, dtype=np.intp))
    if unique_selected.size != num_structures:
        raise ValueError(
            "k-means representatives were not unique; provide less "
            "degenerate representations or reduce num_structures"
        )
    return np.sort(unique_selected)
