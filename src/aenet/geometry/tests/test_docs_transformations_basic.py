"""Docs-backed smoke tests for ``docs/source/usage/transformations_basic.rst``."""

import itertools

import numpy as np
import pytest

from aenet.geometry import AtomicStructure
from aenet.geometry.transformations import (
    AtomDisplacementTransformation,
    CellVolumeTransformation,
    IsovolumetricStrainTransformation,
    ShearStrainTransformation,
)


@pytest.fixture
def docs_periodic_structure():
    """Create the small periodic structure used by the docs examples."""
    return AtomicStructure(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ["Si", "O"],
        avec=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_iterator_and_limiting_output_examples(docs_periodic_structure):
    """The docs iterator examples should produce deterministic counts."""
    transform = AtomDisplacementTransformation(displacement=0.05)

    all_structures = list(transform.apply_transformation(docs_periodic_structure))
    first_two = list(
        itertools.islice(
            transform.apply_transformation(docs_periodic_structure),
            2,
        )
    )
    limited = list(
        itertools.islice(
            transform.apply_transformation(docs_periodic_structure),
            100,
        )
    )

    assert len(all_structures) == 6
    assert len(first_two) == 2
    assert len(limited) == 6
    np.testing.assert_allclose(
        first_two[0].coords[-1][0],
        [0.05, 0.0, 0.0],
    )


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_cell_volume_example_matches_documented_values(docs_periodic_structure):
    """The documented volume example should reflect actual scaling behavior."""
    transform = CellVolumeTransformation(
        min_percent=-5.0,
        max_percent=5.0,
        steps=5,
    )

    original_volume = docs_periodic_structure.cellvolume()
    volumes = []
    percent_changes = []
    for structure in transform.apply_transformation(docs_periodic_structure):
        new_volume = structure.cellvolume()
        volumes.append(round(new_volume, 3))
        percent_changes.append(
            round(100 * (new_volume - original_volume) / original_volume, 2)
        )

    assert volumes == [54.872, 59.319, 64.0, 68.921, 74.088]
    assert percent_changes == [-14.26, -7.31, 0.0, 7.69, 15.76]


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_isovolumetric_and_shear_examples_preserve_volume(
    docs_periodic_structure,
):
    """The strain examples on the page should conserve cell volume."""
    original_volume = docs_periodic_structure.cellvolume()

    isovolumetric = IsovolumetricStrainTransformation(
        direction=1,
        len_min=0.9,
        len_max=1.1,
        steps=3,
    )
    volume_errors = [
        abs(structure.cellvolume() - original_volume)
        for structure in isovolumetric.apply_transformation(
            docs_periodic_structure
        )
    ]

    shear = ShearStrainTransformation(
        direction=1,
        shear_min=-0.1,
        shear_max=0.1,
        steps=3,
    )
    sheared_cells = [
        structure.avec[-1]
        for structure in shear.apply_transformation(docs_periodic_structure)
    ]

    assert all(np.isclose(error, 0.0) for error in volume_errors)
    assert [round(np.linalg.det(cell), 8) for cell in sheared_cells] == [
        64.0,
        64.0,
        64.0,
    ]
    np.testing.assert_allclose(
        sheared_cells[0][0],
        [4.0, -0.4, 0.0],
    )
    np.testing.assert_allclose(
        sheared_cells[-1][0],
        [4.0, 0.4, 0.0],
    )


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_save_all_structures_example_writes_numbered_xsf_files(
    docs_periodic_structure,
    tmp_path,
):
    """The documented save-all loop should emit one file per structure."""
    transform = AtomDisplacementTransformation(displacement=0.05)

    for i, structure in enumerate(
        transform.apply_transformation(docs_periodic_structure)
    ):
        structure.to_file(str(tmp_path / f"output_{i:04d}.xsf"))

    written_files = sorted(path.name for path in tmp_path.glob("output_*.xsf"))

    assert written_files == [
        "output_0000.xsf",
        "output_0001.xsf",
        "output_0002.xsf",
        "output_0003.xsf",
        "output_0004.xsf",
        "output_0005.xsf",
    ]
