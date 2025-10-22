"""
Tests for HDF5 compatibility layer.

This module tests that the PyTorch-based featurizer can write HDF5 files
that are compatible with the TrnSet class and match the format of
Fortran-generated files.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from aenet.formats.xsf import XSFParser
from aenet.geometry import AtomicStructure
from aenet.torch_featurize import TorchAUCFeaturizer
from aenet.trainset import TrnSet


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def water_structure():
    """Create a simple water molecule structure."""
    positions = np.array([
        [0.000, 0.000, 0.118],  # O
        [0.000, 0.755, -0.471],  # H
        [0.000, -0.755, -0.471]  # H
    ])
    species = ['O', 'H', 'H']
    energy = -15.5
    forces = np.array([
        [0.0, 0.0, 0.1],
        [0.0, 0.1, -0.05],
        [0.0, -0.1, -0.05]
    ])

    return AtomicStructure(
        coords=positions,
        types=species,
        energy=energy,
        forces=forces
    )


@pytest.fixture
def periodic_structure():
    """Create a periodic crystal structure."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    species = ['Cu', 'Cu']
    avec = np.array([
        [3.6, 0.0, 0.0],
        [0.0, 3.6, 0.0],
        [0.0, 0.0, 3.6]
    ])
    energy = -8.0
    forces = np.zeros((2, 3))

    return AtomicStructure(
        coords=positions,
        types=species,
        avec=avec,
        energy=energy,
        forces=forces
    )


class TestTorchAUCFeaturizer:
    """Test TorchAUCFeaturizer class API."""

    def test_initialization(self):
        """Test featurizer initialization."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        assert featurizer.typenames == ['O', 'H']
        assert featurizer.ntypes == 2
        assert featurizer.rad_order == 10
        assert featurizer.rad_cutoff == 4.0
        assert featurizer.ang_order == 3
        assert featurizer.ang_cutoff == 1.5

    def test_featurize_structure(self, water_structure):
        """Test single structure featurization."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        result = featurizer.featurize_structure(water_structure)

        # Now returns FeaturizedAtomicStructure for API compatibility
        from aenet.trainset import FeaturizedAtomicStructure
        assert isinstance(result, FeaturizedAtomicStructure)
        assert hasattr(result, 'atom_features')
        assert hasattr(result, 'energy')

        # Check atom features shape
        assert result.atom_features.shape[0] == 3  # 3 atoms
        assert result.atom_features.shape[1] > 0  # non-zero features

    def test_featurize_structures(self, water_structure):
        """Test multiple structure featurization."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        structures = [water_structure, water_structure]
        results = featurizer.featurize_structures(structures)

        assert len(results) == 2
        # Now returns FeaturizedAtomicStructure objects
        from aenet.trainset import FeaturizedAtomicStructure
        assert all(isinstance(r, FeaturizedAtomicStructure) for r in results)

    def test_periodic_structure(self, periodic_structure):
        """Test featurization of periodic structure."""
        featurizer = TorchAUCFeaturizer(
            typenames=['Cu'],
            rad_order=8,
            rad_cutoff=3.5,
            ang_order=5,
            ang_cutoff=3.5
        )

        result = featurizer.featurize_structure(periodic_structure)

        # Now returns FeaturizedAtomicStructure
        from aenet.trainset import FeaturizedAtomicStructure
        assert isinstance(result, FeaturizedAtomicStructure)
        assert result.atom_features.shape[0] == 2  # 2 atoms


class TestHDF5Writing:
    """Test HDF5 file writing functionality."""

    def test_write_hdf5(self, temp_dir, water_structure):
        """Test writing features to HDF5 file."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Create temporary XSF file
        xsf_file = os.path.join(temp_dir, 'water.xsf')
        water_structure.path = xsf_file
        xsf_parser = XSFParser()
        xsf_parser.write(water_structure, xsf_file)

        # Run featurization
        hdf5_file = os.path.join(temp_dir, 'features.h5')
        featurizer.run_aenet_generate(
            xsf_files=[xsf_file],
            hdf5_filename=hdf5_file,
            atomic_energies={'O': -10.0, 'H': -2.5}
        )

        # Check file exists
        assert os.path.exists(hdf5_file)

    def test_hdf5_can_be_read_by_trnset(self, temp_dir, water_structure):
        """Test that TrnSet can read torch-generated HDF5 files."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Create XSF file
        xsf_file = os.path.join(temp_dir, 'water.xsf')
        water_structure.path = xsf_file
        xsf_parser = XSFParser()
        xsf_parser.write(water_structure, xsf_file)

        # Generate HDF5
        hdf5_file = os.path.join(temp_dir, 'features.h5')
        featurizer.run_aenet_generate(
            xsf_files=[xsf_file],
            hdf5_filename=hdf5_file,
            atomic_energies={'O': -10.0, 'H': -2.5}
        )

        # Read with TrnSet
        with TrnSet.from_file(hdf5_file) as ts:
            assert ts.num_structures == 1
            assert ts.num_types == 2
            assert ts.atom_types == ['O', 'H']
            assert ts.atomic_energy[0] == -10.0
            assert ts.atomic_energy[1] == -2.5

            # Read structure
            struc = ts.read_structure(0, read_coords=True, read_forces=True)
            assert struc.num_atoms == 3
            assert len(struc.atoms) == 3

            # Check feature dimensions
            n_features = featurizer.descriptor.get_n_features()
            for atom in struc.atoms:
                assert len(atom['fingerprint']) == n_features

    def test_multiple_structures(self, temp_dir, water_structure):
        """Test writing multiple structures to HDF5."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Create multiple XSF files
        xsf_files = []
        for i in range(3):
            xsf_file = os.path.join(temp_dir, f'water_{i}.xsf')
            perturbed = water_structure.copy()
            perturbed.coords[-1] += 0.1 * np.random.randn(
                *perturbed.coords[-1].shape)
            perturbed.path = xsf_file
            xsf_parser = XSFParser()
            xsf_parser.write(perturbed, xsf_file)
            xsf_files.append(xsf_file)

        # Generate HDF5
        hdf5_file = os.path.join(temp_dir, 'features.h5')
        featurizer.run_aenet_generate(
            xsf_files=xsf_files,
            hdf5_filename=hdf5_file,
            atomic_energies={'O': -10.0, 'H': -2.5}
        )

        # Read with TrnSet
        with TrnSet.from_file(hdf5_file) as ts:
            assert ts.num_structures == 3
            assert ts.num_atoms_tot == 9  # 3 atoms × 3 structures

            # Iterate over structures
            structures = list(ts)
            assert len(structures) == 3

    def test_glob_pattern(self, temp_dir, water_structure):
        """Test using glob pattern for XSF files."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Create multiple XSF files
        for i in range(3):
            xsf_file = os.path.join(temp_dir, f'water_{i}.xsf')
            xsf_parser = XSFParser()
            xsf_parser.write(water_structure, xsf_file)

        # Use glob pattern
        hdf5_file = os.path.join(temp_dir, 'features.h5')
        glob_pattern = os.path.join(temp_dir, 'water_*.xsf')
        featurizer.run_aenet_generate(
            xsf_files=glob_pattern,
            hdf5_filename=hdf5_file,
            atomic_energies={'O': -10.0, 'H': -2.5}
        )

        # Verify
        with TrnSet.from_file(hdf5_file) as ts:
            assert ts.num_structures == 3


class TestHDF5Metadata:
    """Test HDF5 metadata correctness."""

    def test_metadata_fields(self, temp_dir, water_structure):
        """Test that all metadata fields are correctly written."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Create XSF file
        xsf_file = os.path.join(temp_dir, 'water.xsf')
        xsf_parser = XSFParser()
        xsf_parser.write(water_structure, xsf_file)

        # Generate HDF5
        hdf5_file = os.path.join(temp_dir, 'features.h5')
        atomic_energies = {'O': -10.5, 'H': -2.7}
        featurizer.run_aenet_generate(
            xsf_files=[xsf_file],
            hdf5_filename=hdf5_file,
            atomic_energies=atomic_energies
        )

        # Read and check metadata
        with TrnSet.from_file(hdf5_file) as ts:
            assert ts.name == "PyTorch-generated training set"
            assert not ts.normalized
            assert ts.scale == 1.0
            assert ts.shift == 0.0
            assert ts.atom_types == ['O', 'H']
            assert np.allclose(ts.atomic_energy[0], -10.5)
            assert np.allclose(ts.atomic_energy[1], -2.7)
            assert ts.num_atoms_tot == 3
            assert ts.num_structures == 1
            # Energy statistics are per-atom normalized energies
            # E = (total_energy - atomic_contribution) / num_atoms
            # = (-15.5 - (-10.5 - 2*2.7)) / 3 = (-15.5 + 15.9) / 3 = 0.4 / 3
            expected_E = 0.4 / 3
            assert np.allclose(ts.E_min, expected_E)
            assert np.allclose(ts.E_max, expected_E)
            assert np.allclose(ts.E_av, expected_E)

    def test_energy_statistics(self, temp_dir, water_structure):
        """Test energy statistics calculation."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Create structures with different energies
        xsf_files = []
        energies = [-15.5, -14.0, -16.2]

        for i, energy in enumerate(energies):
            struc = water_structure.copy()
            struc.energy[-1] = energy
            xsf_file = os.path.join(temp_dir, f'water_{i}.xsf')
            xsf_parser = XSFParser()
            xsf_parser.write(struc, xsf_file)
            xsf_files.append(xsf_file)

        # Generate HDF5
        hdf5_file = os.path.join(temp_dir, 'features.h5')
        # Use default atomic energies of 0.0 for simplicity
        featurizer.run_aenet_generate(
            xsf_files=xsf_files,
            hdf5_filename=hdf5_file,
            atomic_energies={'O': 0.0, 'H': 0.0}
        )

        # Energy statistics are per-atom normalized
        # E = (total - atomic_contrib) / num_atoms = (total - 0) / 3
        normalized_energies = [e / 3 for e in energies]

        # Check statistics
        with TrnSet.from_file(hdf5_file) as ts:
            assert np.allclose(ts.E_min, min(normalized_energies))
            assert np.allclose(ts.E_max, max(normalized_energies))
            assert np.allclose(ts.E_av, np.mean(normalized_energies))


class TestFeatureConsistency:
    """Test that features match between direct computation and HDF5."""

    def test_feature_values_match(self, temp_dir, water_structure):
        """Test that features in HDF5 match direct computation."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Direct computation - now returns FeaturizedAtomicStructure
        direct_result = featurizer.featurize_structure(water_structure)
        direct_features = direct_result.atom_features

        # Via HDF5
        xsf_file = os.path.join(temp_dir, 'water.xsf')
        xsf_parser = XSFParser()
        xsf_parser.write(water_structure, xsf_file)

        hdf5_file = os.path.join(temp_dir, 'features.h5')
        featurizer.run_aenet_generate(
            xsf_files=[xsf_file],
            hdf5_filename=hdf5_file
        )

        # Read from HDF5
        with TrnSet.from_file(hdf5_file) as ts:
            struc = ts.read_structure(0)
            hdf5_features = np.array(
                [atom['fingerprint'] for atom in struc.atoms])

        # Compare
        assert np.allclose(
            direct_features, hdf5_features, rtol=1e-12, atol=1e-14)

    def test_coordinates_and_forces(self, temp_dir, water_structure):
        """Test that coordinates and forces are correctly stored."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        xsf_file = os.path.join(temp_dir, 'water.xsf')
        xsf_parser = XSFParser()
        xsf_parser.write(water_structure, xsf_file)

        hdf5_file = os.path.join(temp_dir, 'features.h5')
        featurizer.run_aenet_generate(
            xsf_files=[xsf_file],
            hdf5_filename=hdf5_file
        )

        # Read and compare
        with TrnSet.from_file(hdf5_file) as ts:
            struc = ts.read_structure(0, read_coords=True, read_forces=True)

            stored_coords = np.array([atom['coords'] for atom in struc.atoms])
            stored_forces = np.array([atom['forces'] for atom in struc.atoms])

            assert np.allclose(stored_coords, water_structure.coords[-1])
            assert np.allclose(stored_forces, water_structure.forces[-1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
