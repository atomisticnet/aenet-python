"""Docs-backed smoke tests for ``docs/source/usage/torch_featurization.rst``."""

import numpy as np
import pytest
import torch

torch_cluster = pytest.importorskip("torch_cluster")

from aenet.geometry import AtomicStructure  # noqa: E402
from aenet.torch_featurize import (  # noqa: E402
    BatchedFeaturizer,
    ChebyshevDescriptor,
    TorchAUCFeaturizer,
)


@pytest.fixture
def docs_water_structure():
    """Create the small water-like structure used by the docs examples."""
    return AtomicStructure(
        np.array(
            [
                [0.0, 0.0, 0.12],
                [0.0, 0.76, -0.47],
                [0.0, -0.76, -0.47],
            ]
        ),
        ["O", "H", "H"],
    )


@pytest.fixture
def docs_periodic_structure():
    """Create the small periodic structure used by the docs examples."""
    return AtomicStructure(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 2.0, 0.0],
            ]
        ),
        ["Cu", "Cu", "Au", "Au"],
        avec=np.array(
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ),
    )


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_high_level_and_periodic_examples(
    docs_water_structure,
    docs_periodic_structure,
):
    """The compact high-level docs examples should remain deterministic."""
    water_featurizer = TorchAUCFeaturizer(
        typenames=["O", "H"],
        rad_order=10,
        rad_cutoff=4.0,
        ang_order=3,
        ang_cutoff=1.5,
    )
    featurized = water_featurizer.featurize_structure(docs_water_structure)

    assert featurized.atom_features.shape == (3, 30)

    periodic_featurizer = TorchAUCFeaturizer(
        typenames=["Au", "Cu"],
        rad_order=8,
        rad_cutoff=3.5,
        ang_order=5,
        ang_cutoff=3.5,
    )
    periodic_features = periodic_featurizer.featurize_structure(
        docs_periodic_structure
    )

    assert periodic_features.atom_features.shape == (4, 30)


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_batched_featurizer_example():
    """The batch-processing docs example should keep its documented shape."""
    descriptor = ChebyshevDescriptor(
        species=["O", "H"],
        rad_order=10,
        rad_cutoff=4.0,
        ang_order=3,
        ang_cutoff=1.5,
    )
    batch_featurizer = BatchedFeaturizer(descriptor)

    batch_positions = [
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    ]
    batch_species = [
        ["O", "H", "H"],
        ["O", "H"],
    ]

    features, batch_indices = batch_featurizer(batch_positions, batch_species)

    assert features.shape == (5, 30)
    assert batch_indices.tolist() == [0, 0, 0, 1, 1]
