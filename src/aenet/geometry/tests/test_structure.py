"""Tests for AtomicStructure geometry mutation helpers."""

import numpy as np
import pytest

from aenet.exceptions import ArgumentError
from aenet.geometry import AtomicStructure


@pytest.fixture
def periodic_structure_with_labels():
    """Create a periodic structure with energy and force labels."""
    coords = np.array([
        [0.3, 0.4, 0.5],
        [1.2, 1.6, 2.1],
    ])
    types = ["Si", "O"]
    avec = np.array([
        [4.0, 0.0, 0.0],
        [0.8, 5.1, 0.0],
        [0.4, 1.2, 6.0],
    ])
    forces = np.array([
        [0.1, -0.2, 0.3],
        [-0.4, 0.5, -0.6],
    ])
    return AtomicStructure(
        coords,
        types,
        avec=avec,
        energy=-3.25,
        forces=forces,
    )


def test_update_cell_preserves_fractional_coordinates_by_default(
    periodic_structure_with_labels,
):
    """Cartesian coordinates should be rebuilt from preserved fractions."""
    structure = periodic_structure_with_labels.copy()
    frac_before = structure.cart2frac(structure.coords[-1]).copy()
    coords_before = structure.coords[-1].copy()
    new_avec = np.array([
        [4.4, 0.0, 0.0],
        [1.0, 4.9, 0.0],
        [0.7, 1.5, 5.8],
    ])

    structure.update_cell(new_avec)

    np.testing.assert_allclose(structure.avec[-1], new_avec)
    np.testing.assert_allclose(
        structure.cart2frac(structure.coords[-1]),
        frac_before,
    )
    np.testing.assert_allclose(structure.coords[-1], frac_before @ new_avec)
    assert not np.allclose(structure.coords[-1], coords_before)
    assert structure.energy[-1] is None
    assert structure.forces[-1] is None


def test_update_cell_can_preserve_cartesian_coordinates(
    periodic_structure_with_labels,
):
    """Cartesian coordinates should remain unchanged when requested."""
    structure = periodic_structure_with_labels.copy()
    coords_before = structure.coords[-1].copy()
    new_avec = np.array([
        [3.7, 0.0, 0.0],
        [0.5, 5.4, 0.0],
        [0.9, 1.8, 6.4],
    ])

    structure.update_cell(new_avec, preserve="cartesian")

    np.testing.assert_allclose(structure.avec[-1], new_avec)
    np.testing.assert_allclose(structure.coords[-1], coords_before)
    assert structure.energy[-1] is None
    assert structure.forces[-1] is None


def test_update_cell_can_keep_labels(periodic_structure_with_labels):
    """Energy and forces should be retained only when explicitly requested."""
    structure = periodic_structure_with_labels.copy()
    energy_before = structure.energy[-1]
    forces_before = structure.forces[-1].copy()
    new_avec = np.array([
        [4.2, 0.0, 0.0],
        [0.9, 5.0, 0.0],
        [0.6, 1.4, 5.9],
    ])

    structure.update_cell(
        new_avec,
        keep_energy=True,
        keep_forces=True,
    )

    assert structure.energy[-1] == energy_before
    np.testing.assert_allclose(structure.forces[-1], forces_before)


def test_update_cell_rejects_non_periodic_structure():
    """Updating the cell of an isolated structure should fail."""
    structure = AtomicStructure([[0.0, 0.0, 0.0]], ["He"])

    with pytest.raises(ArgumentError, match="non-periodic"):
        structure.update_cell(np.eye(3))


def test_update_cell_rejects_invalid_preserve_mode(
    periodic_structure_with_labels,
):
    """Only documented preserve modes should be accepted."""
    with pytest.raises(ArgumentError, match="Invalid preserve mode"):
        periodic_structure_with_labels.update_cell(
            np.eye(3),
            preserve="lattice",
        )


def test_update_cell_rejects_invalid_cell_shape(
    periodic_structure_with_labels,
):
    """The new cell must be a 3x3 matrix."""
    with pytest.raises(ArgumentError, match="shape"):
        periodic_structure_with_labels.update_cell(np.eye(2))
