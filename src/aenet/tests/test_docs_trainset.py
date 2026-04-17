"""Docs-backed smoke tests for ``docs/source/api/trainset.rst``."""

from pathlib import Path

import numpy as np
import pytest

from aenet.trainset import TrnSet

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_H5 = DATA_DIR / "sample.h5"
SAMPLE_ASCII = DATA_DIR / "sample.train.ascii"
FEATURES_WITH_NEIGHBORS = DATA_DIR / "features_with_neighbors.h5"


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_hdf5_trainset_inspection_example():
    """The compact HDF5 inspection example should stay runnable."""
    with TrnSet.from_file(SAMPLE_H5) as trnset:
        struct = trnset[0]

        assert trnset.schema == "trnset_hdf5"
        assert trnset.num_structures == 5
        assert trnset.atom_types == ["Ti", "O"]
        assert struct.num_atoms == 23
        assert struct.atom_features.shape == (23, 30)
        assert struct.coords.shape == (23, 3)
        assert struct.forces.shape == (23, 3)


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_hdf5_and_ascii_reader_examples():
    """The documented HDF5/ASCII comparison should reflect real fixtures."""
    with TrnSet.from_file(SAMPLE_H5) as trnset_h5, \
            TrnSet.from_file(SAMPLE_ASCII) as trnset_ascii:
        struct_h5 = trnset_h5.read_structure(0, read_coords=True, read_forces=True)
        struct_ascii = trnset_ascii.read_structure(
            0,
            read_coords=True,
            read_forces=True,
        )

        assert trnset_h5.num_structures == trnset_ascii.num_structures
        assert trnset_h5.atom_types == trnset_ascii.atom_types
        assert struct_h5.num_atoms == struct_ascii.num_atoms
        assert struct_h5.atom_features.shape == struct_ascii.atom_features.shape
        assert struct_h5.coords.shape == struct_ascii.coords.shape
        np.testing.assert_allclose(struct_h5.atom_features, struct_ascii.atom_features)
        np.testing.assert_allclose(struct_h5.coords, struct_ascii.coords)
        np.testing.assert_allclose(struct_h5.forces, struct_ascii.forces)


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_neighbor_info_guard_example():
    """The docs should show the safe access pattern for optional neighbor data."""
    with TrnSet.from_file(FEATURES_WITH_NEIGHBORS) as trnset:
        assert trnset.has_neighbor_info()

        struct = trnset.read_structure(0, read_coords=True, read_forces=True)

        assert not struct.has_neighbor_info
        assert struct.neighbor_info is None
