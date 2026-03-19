"""Docs-backed smoke tests for ``docs/source/dev/neighbor_lists.rst``."""

import numpy as np
import pytest
import torch

torch_cluster = pytest.importorskip("torch_cluster")

from aenet.geometry import AtomicStructure  # noqa: E402
from aenet.torch_featurize import TorchNeighborList  # noqa: E402


@pytest.fixture
def docs_water_structure():
    """Create the small water-like structure used by the docs examples."""
    return AtomicStructure(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
            ]
        ),
        ["O", "H", "H"],
    )


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_atomic_structure_and_direct_usage_examples(docs_water_structure):
    """The quick-start docs examples should remain deterministic."""
    neighbors = docs_water_structure.get_neighbors(i=0, cutoff=2.0)

    assert neighbors.natoms == 3
    assert neighbors.types.tolist() == ["O", "H", "H"]

    nbl = TorchNeighborList(cutoff=4.0, device="cpu")
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    result = nbl.get_neighbors_of_atom(0, positions)

    assert result["indices"].cpu().tolist() == [1, 2]
    np.testing.assert_allclose(
        result["distances"].cpu().numpy(),
        [1.5, 3.0],
    )
    assert result["offsets"] is None


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_periodic_and_return_coordinates_examples():
    """Periodic docs examples should make the Cartesian convention explicit."""
    cell = np.eye(3) * 5.0
    positions = np.array(
        [
            [0.5, 2.5, 2.5],
            [4.5, 2.5, 2.5],
        ]
    )
    nbl = TorchNeighborList(cutoff=2.0, device="cpu")

    result = nbl.get_neighbors_of_atom(
        0,
        positions,
        cell=cell,
        fractional=False,
    )
    assert result["indices"].cpu().tolist() == [1]
    assert result["offsets"].cpu().tolist() == [[-1, 0, 0]]

    result_with_coords = nbl.get_neighbors_of_atom(
        0,
        positions,
        cell=cell,
        fractional=False,
        return_coordinates=True,
    )
    np.testing.assert_allclose(
        result_with_coords["coordinates"].cpu().numpy(),
        [[-0.5, 2.5, 2.5]],
    )


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_type_dependent_and_per_atom_examples():
    """Advanced docs examples should reflect the supported APIs."""
    positions = np.array(
        [
            [0.00000, 0.00000, 0.11779],
            [0.00000, 0.75545, -0.47116],
            [0.00000, -0.75545, -0.47116],
        ]
    )
    atom_types = torch.tensor([8, 1, 1], dtype=torch.long)
    cutoff_dict = {
        (1, 1): 1.0,
        (1, 8): 2.5,
        (8, 8): 3.0,
    }
    nbl_typed = TorchNeighborList(
        cutoff=5.0,
        atom_types=atom_types,
        cutoff_dict=cutoff_dict,
        device="cpu",
    )

    oxygen_neighbors = nbl_typed.get_neighbors_of_atom(0, positions)
    hydrogen_neighbors = nbl_typed.get_neighbors_of_atom(1, positions)

    assert oxygen_neighbors["indices"].cpu().tolist() == [1, 2]
    assert hydrogen_neighbors["indices"].cpu().tolist() == [0]

    bulk_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    bulk_nbl = TorchNeighborList(cutoff=1.5, device="cpu")
    all_neighbors = bulk_nbl.get_neighbors_by_atom(bulk_positions)

    assert [len(atom_neighbors["indices"]) for atom_neighbors in all_neighbors] == [
        2,
        2,
        2,
    ]


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_factory_and_cache_examples(docs_water_structure):
    """Factory and cache snippets should match the current API behavior."""
    factory_nbl = TorchNeighborList.from_AtomicStructure(
        docs_water_structure,
        cutoff=2.0,
        device="cpu",
    )
    assert factory_nbl.cutoff == 2.0
    assert factory_nbl.max_num_neighbors == 256

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    cache_nbl = TorchNeighborList(cutoff=1.5, device="cpu")
    result1 = cache_nbl.get_neighbors_of_atom(0, positions)
    result2 = cache_nbl.get_neighbors_of_atom(1, positions)
    result3 = cache_nbl.get_neighbors_of_atom(2, positions)
    result4 = cache_nbl.get_neighbors_of_atom(0, positions + 0.1)

    assert [len(result["indices"]) for result in (result1, result2, result3)] == [
        2,
        2,
        2,
    ]
    assert len(result4["indices"]) == 2
