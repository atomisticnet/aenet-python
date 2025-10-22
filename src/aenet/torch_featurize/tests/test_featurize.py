"""
Unit tests for complete featurization pipeline.
"""

import pytest
import torch
import numpy as np

from aenet.torch_featurize.featurize import (
    ChebyshevDescriptor,
    BatchedFeaturizer
)


class TestChebyshevDescriptor:
    """Test complete featurization pipeline."""

    def test_water_molecule_shape(self):
        """Test featurization shape on water molecule."""
        positions = torch.tensor([
            [0.00000, 0.00000, 0.11779],   # O
            [0.00000, 0.75545, -0.47116],  # H
            [0.00000, -0.75545, -0.47116]  # H
        ], dtype=torch.float64)

        species = ['O', 'H', 'H']

        featurizer = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5,
            min_cutoff=0.55
        )

        features = featurizer(positions, species)

        # Validate shape
        assert features.shape == (3, 30), \
            f"Expected shape (3, 30), got {features.shape}"
        assert features.dtype == torch.float64

        # Validate no NaN or Inf
        assert torch.all(torch.isfinite(features)), \
            "Features contain NaN or Inf values"

    def test_feature_count(self):
        """Test feature count calculation."""
        # 2 species system
        fz2 = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )
        # Radial: 2 * (10+1) = 22
        # Angular: 2 * (3+1) = 8
        # Total: 30
        assert fz2.get_n_features() == 30

        # 3 species system
        fz3 = ChebyshevDescriptor(
            species=['Li', 'O', 'Ti'],
            rad_order=8,
            rad_cutoff=3.5,
            ang_order=2,
            ang_cutoff=1.5
        )
        # Radial: 2 * (8+1) = 18
        # Angular: 2 * (2+1) = 6
        # Total: 24
        assert fz3.get_n_features() == 24

    def test_typespin_values(self):
        """Test typespin coefficient computation."""
        # 2 species: should be {-1, 1}
        fz2 = ChebyshevDescriptor(['O', 'H'], 10, 4.0, 3, 1.5)
        expected_2 = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        assert torch.allclose(fz2.typespin, expected_2), \
            f"Expected {expected_2}, got {fz2.typespin}"

        # 3 species: should be {-1, 0, 1}
        fz3 = ChebyshevDescriptor(['Li', 'O', 'H'], 10, 4.0, 3, 1.5)
        expected_3 = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        assert torch.allclose(fz3.typespin, expected_3), \
            f"Expected {expected_3}, got {fz3.typespin}"

        # 4 species: should be {-2, -1, 1, 2}
        fz4 = ChebyshevDescriptor(
            ['Li', 'O', 'Ti', 'H'], 10, 4.0, 3, 1.5
        )
        expected_4 = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(fz4.typespin, expected_4), \
            f"Expected {expected_4}, got {fz4.typespin}"

        # 5 species: should be {-2, -1, 0, 1, 2}
        fz5 = ChebyshevDescriptor(
            ['A', 'B', 'C', 'D', 'E'], 10, 4.0, 3, 1.5
        )
        expected_5 = torch.tensor(
            [-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64
        )
        assert torch.allclose(fz5.typespin, expected_5), \
            f"Expected {expected_5}, got {fz5.typespin}"

    def test_single_species(self):
        """Test single-species system (no typespin weighting)."""
        positions = torch.randn(10, 3, dtype=torch.float64) * 5
        species = ['O'] * 10

        featurizer = ChebyshevDescriptor(
            species=['O'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        features = featurizer(positions, species)

        # Single species: (10+1) + (3+1) = 15 features
        assert features.shape == (10, 15), \
            f"Expected shape (10, 15), got {features.shape}"
        assert not featurizer.multi, \
            "Single species system should have multi=False"

    def test_numpy_interface(self):
        """Test numpy interface for featurization."""
        positions_np = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        species = ['O', 'H', 'H']

        featurizer = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        features_np = featurizer.featurize_structure(positions_np, species)

        assert isinstance(features_np, np.ndarray), \
            "Output should be numpy array"
        assert features_np.shape == (3, 30), \
            f"Expected shape (3, 30), got {features_np.shape}"
        assert features_np.dtype == np.float64, \
            f"Expected float64, got {features_np.dtype}"

    def test_radial_features_only(self):
        """Test with neighbors only in radial range."""
        # Two atoms far apart (only radial contributions)
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]  # 3 Å apart
        ], dtype=torch.float64)
        species = ['O', 'O']

        featurizer = ChebyshevDescriptor(
            species=['O'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5  # Angular cutoff too small
        )

        features = featurizer(positions, species)

        # Check that features are computed
        assert torch.all(torch.isfinite(features))
        # Radial features should be non-zero
        assert torch.any(features != 0.0), "Features should be non-zero"

    def test_angular_features(self):
        """Test angular features with close neighbors."""
        # Three atoms forming a triangle
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0]  # Equilateral triangle
        ], dtype=torch.float64)
        species = ['O', 'H', 'H']

        featurizer = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=2.0  # Large enough for all pairs
        )

        features = featurizer(positions, species)

        # Check that angular features are non-zero
        # Angular features are in the last 8 positions for 2-species system
        angular_start = 22  # After 22 radial features
        angular_features = features[:, angular_start:]
        assert torch.any(angular_features != 0.0), \
            "Angular features should be non-zero"

    def test_isolated_atom(self):
        """Test single atom with no neighbors."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        species = ['O']

        featurizer = ChebyshevDescriptor(
            species=['O'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        features = featurizer(positions, species)

        # Features should be all zeros (no neighbors)
        assert torch.allclose(features, torch.zeros_like(features)), \
            "Isolated atom should have zero features"


class TestBatchedFeaturizer:
    """Test batched featurization."""

    def test_batch_processing(self):
        """Test batched featurization of multiple structures."""
        featurizer = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        batched_fz = BatchedFeaturizer(featurizer)

        # Create two water molecules
        pos1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float64)

        pos2 = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0]
        ], dtype=torch.float64)

        batch_positions = [pos1, pos2]
        batch_species = [['O', 'H', 'H'], ['O', 'H']]

        features, batch_indices = batched_fz(batch_positions, batch_species)

        # Check shapes
        total_atoms = 3 + 2
        assert features.shape == (total_atoms, 30), \
            f"Expected shape ({total_atoms}, 30), got {features.shape}"
        assert batch_indices.shape == (total_atoms,), \
            f"Expected shape ({total_atoms},), got {batch_indices.shape}"

        # Check batch indices
        expected_indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        assert torch.all(batch_indices == expected_indices), \
            f"Expected indices {expected_indices}, got {batch_indices}"

    def test_variable_sizes(self):
        """Test batching with variable structure sizes."""
        featurizer = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=5,
            rad_cutoff=3.0,
            ang_order=2,
            ang_cutoff=1.5
        )

        batched_fz = BatchedFeaturizer(featurizer)

        # Different sized structures
        structures = [
            torch.randn(2, 3, dtype=torch.float64),
            torch.randn(5, 3, dtype=torch.float64),
            torch.randn(3, 3, dtype=torch.float64)
        ]

        species_lists = [
            ['O', 'H'],
            ['O', 'H', 'H', 'O', 'H'],
            ['O', 'H', 'H']
        ]

        features, batch_indices = batched_fz(structures, species_lists)

        # Check total size
        total_atoms = 2 + 5 + 3
        assert features.shape[0] == total_atoms, \
            f"Expected {total_atoms} atoms, got {features.shape[0]}"


class TestFeatureValues:
    """Test actual feature value computation."""

    def test_symmetry(self):
        """Test that symmetric structures give symmetric features."""
        featurizer = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=5,
            rad_cutoff=4.0,
            ang_order=2,
            ang_cutoff=1.5
        )

        # Symmetric water molecule
        positions = torch.tensor([
            [0.0, 0.0, 0.0],      # O at origin
            [1.0, 0.0, 0.0],      # H on x-axis
            [-1.0, 0.0, 0.0]      # H on -x-axis
        ], dtype=torch.float64)
        species = ['O', 'H', 'H']

        features = featurizer(positions, species)

        # Two H atoms should have identical features
        assert torch.allclose(features[1], features[2], atol=1e-10), \
            "Symmetric atoms should have identical features"

    def test_feature_scaling(self):
        """Test that features scale appropriately with distance."""
        featurizer = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Two configurations with different distances
        pos1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=torch.float64)

        pos2 = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ], dtype=torch.float64)

        species = ['O', 'H']

        feat1 = featurizer(pos1, species)
        feat2 = featurizer(pos2, species)

        # Features should be different
        assert not torch.allclose(feat1, feat2), \
            "Features should differ for different distances"


class TestRotationalInvariance:
    """Test rotational invariance of features (from Fortran test_sfbasis)."""

    @staticmethod
    def generate_fcc_neighbors(Rc, alat, Xi, avec):
        """
        Generate FCC periodic neighbors within cutoff radius.

        Port of get_coo from fortran/test_sfbasis.f90

        Args:
            Rc: Cutoff radius
            alat: Lattice parameter
            Xi: Central atom position (fractional coordinates)
            avec: (3,3) lattice vectors

        Returns:
            X: (n_neighbors, 3) neighbor positions
            n_neighbors: Number of neighbors found
        """
        Rc2 = Rc * Rc
        neighbors = []

        # Convert Xi to cartesian
        Xi_cart = (Xi[0] * avec[0] + Xi[1] * avec[1] + Xi[2] * avec[2])

        # Search over unit cell images
        for iz in range(-10, 11):
            for iy in range(-10, 11):
                for ix in range(-10, 11):
                    if ix == 0 and iy == 0 and iz == 0:
                        continue

                    # Compute translation vector
                    t = ix * avec[0] + iy * avec[1] + iz * avec[2]
                    d2 = np.sum(t * t)

                    if d2 < Rc2:
                        neighbors.append(Xi_cart + t)

        return np.array(neighbors) if neighbors else np.zeros((0, 3))

    @staticmethod
    def create_rotation_matrix_x(angle_deg):
        """Create rotation matrix around x-axis."""
        angle = np.deg2rad(angle_deg)
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c]
        ])


    def test_simple_rotation_invariance(self):
        """Test simpler case: water molecule under rotation."""
        featurizer = ChebyshevDescriptor(
            species=['O', 'H'],
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Original water molecule
        positions = np.array([
            [0.0, 0.0, 0.0],      # O
            [1.0, 0.0, 0.0],      # H
            [0.0, 1.0, 0.0]       # H
        ])
        species = ['O', 'H', 'H']

        # Compute features
        features1 = featurizer.featurize_structure(positions, species)

        # Rotate 90 degrees around z-axis
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        positions_rot = positions @ R.T

        # Compute features for rotated structure
        features2 = featurizer.featurize_structure(positions_rot, species)

        # Features should be identical
        assert np.allclose(features1, features2, atol=1e-10), \
            f"Simple rotation failed!\n" \
            f"Max diff: {np.max(np.abs(features1 - features2))}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
