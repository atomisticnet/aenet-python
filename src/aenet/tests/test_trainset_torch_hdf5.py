"""Regression tests for torch-training HDF5 compatibility in ``TrnSet``."""

from pathlib import Path

import numpy as np
import pytest

from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import HDF5StructureDataset
from aenet.trainset import TrnSet

REPO_ROOT = Path(__file__).resolve().parents[3]
TIO2_XSF = REPO_ROOT / "notebooks" / "xsf-TiO2" / "structure-001.xsf"


@pytest.mark.cpu
def test_trnset_reads_torch_training_hdf5_with_persisted_features(tmp_path: Path):
    """TrnSet should expose persisted torch-training features as fingerprints."""
    db_path = tmp_path / "torch_training_features.h5"
    descriptor = ChebyshevDescriptor(
        species=["Ti", "O"],
        rad_order=10,
        rad_cutoff=6.0,
        ang_order=3,
        ang_cutoff=3.5,
    )
    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=[str(TIO2_XSF)],
        mode="build",
    )
    dataset.build_database(show_progress=False, persist_features=True)

    with TrnSet.from_file(db_path) as trnset:
        struct = trnset.read_structure(0, read_coords=True, read_forces=True)

        assert trnset.format == "hdf5"
        assert trnset.schema == "torch_training_hdf5"
        assert trnset.atom_types == ["Ti", "O"]
        assert trnset.num_structures == 1
        assert not trnset.has_neighbor_info()
        assert struct.num_atoms == 23
        assert struct.atom_features.shape == (23, descriptor.get_n_features())
        assert struct.coords.shape == (23, 3)
        assert struct.forces.shape == (23, 3)
        assert struct.has_cell
        assert struct.is_periodic

        persisted = dataset.load_persisted_features(0)
        assert persisted is not None
        np.testing.assert_allclose(
            struct.atom_features,
            persisted.cpu().numpy(),
        )


@pytest.mark.cpu
def test_trnset_warns_and_exposes_empty_features_for_featureless_torch_hdf5(
    tmp_path: Path,
):
    """Featureless torch-training HDF5 files should warn and expose empty fingerprints."""
    db_path = tmp_path / "torch_training_no_features.h5"
    dataset = HDF5StructureDataset(
        descriptor=None,
        database_file=str(db_path),
        sources=[str(TIO2_XSF)],
        mode="build",
    )
    dataset.build_database(show_progress=False)

    with pytest.warns(UserWarning, match="does not contain persisted features"):
        trnset = TrnSet.from_file(db_path)

    with trnset:
        struct = trnset.read_structure(0, read_coords=True, read_forces=True)

        assert trnset.schema == "torch_training_hdf5"
        assert trnset.atom_types == ["Ti", "O"]
        assert trnset.num_structures == 1
        assert not trnset.has_neighbor_info()
        assert struct.num_atoms == 23
        assert struct.atom_features.shape == (23, 0)
        assert all(len(atom["fingerprint"]) == 0 for atom in struct.atoms)
        assert struct.coords.shape == (23, 3)
        assert struct.forces.shape == (23, 3)
        assert struct.has_cell
        assert struct.is_periodic
