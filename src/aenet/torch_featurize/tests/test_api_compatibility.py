"""
Tests for API compatibility between TorchAUCFeaturizer and AenetAUCFeaturizer.

This module tests that TorchAUCFeaturizer provides the same API as
AenetAUCFeaturizer, ensuring drop-in replacement capability.
"""

import tempfile
import shutil
import os

import numpy as np
import pytest

from aenet.geometry import AtomicStructure
from aenet.featurize import AtomicFeaturizer
from aenet.trainset import FeaturizedAtomicStructure
from aenet.torch_featurize import TorchAUCFeaturizer


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
def temp_dir():
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


class TestAPICompatibility:
    """Test that TorchAUCFeaturizer matches AenetAUCFeaturizer API."""

    def test_inherits_from_atomic_featurizer(self):
        """Test that TorchAUCFeaturizer inherits from AtomicFeaturizer."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Should inherit from AtomicFeaturizer
        assert isinstance(featurizer, AtomicFeaturizer)

    def test_from_structure_classmethod(self, water_structure):
        """Test from_structure classmethod inherited from base."""
        # This should work since we inherit from AtomicFeaturizer
        featurizer = TorchAUCFeaturizer.from_structure(
            water_structure,
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Typenames extracted from structure in order of appearance
        assert set(featurizer.typenames) == {'H', 'O'}
        assert featurizer.ntypes == 2

    def test_featurize_structure_accepts_atomic_structure(
            self, water_structure):
        """Test that featurize_structure accepts AtomicStructure objects."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # This is the key API requirement from the user
        result = featurizer.featurize_structure(water_structure)

        # Should return FeaturizedAtomicStructure
        assert isinstance(result, FeaturizedAtomicStructure)
        assert hasattr(result, 'atom_features')
        assert hasattr(result, 'energy')

    def test_featurize_structures_batch_method(self, water_structure):
        """Test featurize_structures method for batch processing."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Create batch of structures
        structures = [water_structure, water_structure, water_structure]

        # This is the API requirement from the user
        results = featurizer.featurize_structures(structures)

        # Should return list of FeaturizedAtomicStructure objects
        assert len(results) == 3
        assert all(isinstance(r, FeaturizedAtomicStructure) for r in results)
        assert all(hasattr(r, 'atom_features') for r in results)

    def test_returns_featurized_atomic_structure(self, water_structure):
        """Test that output has the same type as AenetAUCFeaturizer."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        result = featurizer.featurize_structure(water_structure)

        # Should have all properties of FeaturizedAtomicStructure
        assert hasattr(result, 'atom_features')
        assert hasattr(result, 'energy')
        assert hasattr(result, 'atoms')
        assert hasattr(result, 'atom_types')
        assert hasattr(result, 'num_atoms')

        # Check dimensions
        assert result.num_atoms == 3
        assert result.atom_features.shape[0] == 3
        assert result.atom_features.shape[1] > 0

    def test_user_requested_api_pattern(self, temp_dir):
        """
        Test the exact API pattern requested by the user in the task.

        This is the most important test - it verifies the user's
        desired workflow.
        """
        # Create a simple XYZ file
        xyz_content = """3
Water molecule
O  0.000  0.000  0.118
H  0.000  0.755 -0.471
H  0.000 -0.755 -0.471
"""
        xyz_file = os.path.join(temp_dir, 'water.xyz')
        with open(xyz_file, 'w') as f:
            f.write(xyz_content)

        # This is the exact pattern from the user's request:
        # import aenet.io.structure
        # from aenet.torch_featurize import TorchAUCFeaturizer
        from aenet.io.structure import read as read_structure

        struc = read_structure(xyz_file)

        descriptor = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,      # Radial polynomial order
            rad_cutoff=4.0,    # Radial cutoff (Angstroms)
            ang_order=3,       # Angular polynomial order
            ang_cutoff=1.5     # Angular cutoff (Angstroms)
        )

        # The key API call
        featurized_structure = descriptor.featurize_structure(struc)

        # Verify it works as expected
        assert isinstance(featurized_structure, FeaturizedAtomicStructure)

        # For multi-species: 2 * (rad_order+1 + ang_order+1)
        # = 2 * (11 + 4) = 30
        expected_features = 2 * (10 + 1 + 3 + 1)
        assert featurized_structure.atom_features.shape == (
            3, expected_features)
        assert featurized_structure.num_atoms == 3

    def test_batch_featurization_api(self, water_structure):
        """Test batch featurization works as expected."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Create multiple structures
        structures = [water_structure for _ in range(5)]

        # Batch featurization
        results = featurizer.featurize_structures(structures)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, FeaturizedAtomicStructure)
            assert result.num_atoms == 3
            assert result.atom_features.shape[0] == 3

    def test_common_properties(self):
        """Test that common properties match AenetAUCFeaturizer."""
        typenames = ['Cu', 'O']
        featurizer = TorchAUCFeaturizer(
            typenames=typenames,
            rad_order=8,
            rad_cutoff=5.0,
            ang_order=4,
            ang_cutoff=3.5,
            min_cutoff=0.5
        )

        # Test properties that should match AenetAUCFeaturizer
        assert featurizer.typenames == typenames
        assert featurizer.ntypes == 2
        assert featurizer.rad_order == 8
        assert featurizer.rad_cutoff == 5.0
        assert featurizer.ang_order == 4
        assert featurizer.ang_cutoff == 3.5
        assert featurizer.min_cutoff == 0.5

    def test_periodic_structure_support(self):
        """Test that periodic structures are handled correctly."""
        # Create a periodic structure
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

        struc = AtomicStructure(
            coords=positions,
            types=species,
            avec=avec
        )

        featurizer = TorchAUCFeaturizer(
            typenames=['Cu'],
            rad_order=8,
            rad_cutoff=3.5,
            ang_order=5,
            ang_cutoff=3.5
        )

        result = featurizer.featurize_structure(struc)

        assert isinstance(result, FeaturizedAtomicStructure)
        assert result.num_atoms == 2
        assert result.atom_features.shape[0] == 2

    def test_initialization_parameters(self):
        """Test all initialization parameters work correctly."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5,
            min_cutoff=0.55,
            device='cpu'
        )

        assert featurizer.typenames == ['O', 'H']
        assert featurizer.rad_order == 10
        assert featurizer.rad_cutoff == 4.0
        assert featurizer.ang_order == 3
        assert featurizer.ang_cutoff == 1.5
        assert featurizer.min_cutoff == 0.55
        assert featurizer.device == 'cpu'

    def test_feature_dimensions(self, water_structure):
        """Test that feature dimensions match expected values."""
        featurizer = TorchAUCFeaturizer(
            typenames=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        result = featurizer.featurize_structure(water_structure)

        # For multi-species: 2 * (rad_order+1 + ang_order+1)
        # = 2 * (11 + 4) = 30
        expected_features = 2 * (10 + 1 + 3 + 1)
        assert result.atom_features.shape == (3, expected_features)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
