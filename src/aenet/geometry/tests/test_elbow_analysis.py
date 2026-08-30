from pathlib import Path

import numpy as np
import pytest

from aenet.geometry.elbow_analysis import (
    _x_ticks,
    elbow_inertias,
    plot_elbow_method,
)


def test_x_ticks_stay_sparse_for_large_k():
    ticks = _x_ticks(250)

    assert ticks[0] == 1
    assert ticks[-1] == 250
    assert len(ticks) <= 10
    assert np.all(np.diff(ticks) > 0)


def test_elbow_inertias_returns_one_value_per_k():
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [20.0, 20.0],
        ]
    )

    k_values, inertias = elbow_inertias(
        features,
        max_k=3,
        random_state=0,
        n_init=1,
    )

    assert np.array_equal(k_values, np.array([1, 2, 3]))
    assert inertias.shape == (3,)
    assert np.all(np.diff(inertias) < 0)


def test_plot_elbow_method_writes_png(tmp_path: Path):
    features = np.random.default_rng(0).normal(size=(40, 4))
    output_path = tmp_path / "elbow.png"

    returned_path = plot_elbow_method(
        features,
        max_k=20,
        random_state=0,
        n_init=1,
        output_path=output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_elbow_inertias_rejects_max_k_larger_than_samples():
    with pytest.raises(ValueError, match="number of samples"):
        elbow_inertias(np.ones((3, 2)), max_k=4)
