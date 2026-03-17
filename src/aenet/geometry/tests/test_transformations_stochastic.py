"""
Tests for stochastic transformation classes (iterator-based API).

This module tests the RandomDisplacementTransformation and related
functionality, including reproducibility, orthogonality, and RMS
displacement validation.
"""

import itertools
import time

import numpy as np
import pytest

from aenet.geometry import AtomicStructure
from aenet.geometry.transformations import (
    AtomDisplacementTransformation,
    DOptimalDisplacementTransformation,
    RandomDisplacementTransformation,
    TransformationChain,
)


# Test fixtures
@pytest.fixture
def simple_structure():
    """Create a simple test structure with 3 atoms."""
    coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]
    types = ['H', 'H', 'O']
    avec = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    return AtomicStructure(coords, types, avec=avec)


@pytest.fixture
def single_atom_structure():
    """Create a structure with a single atom."""
    coords = [[0.5, 0.5, 0.5]]
    types = ['H']
    avec = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    return AtomicStructure(coords, types, avec=avec)


@pytest.fixture
def larger_structure():
    """Create a larger structure with 10 atoms."""
    rng = np.random.default_rng(123)
    coords = (rng.random((10, 3)) * 5.0).tolist()
    types = ['C', 'N'] * 5
    avec = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    return AtomicStructure(coords, types, avec=avec)


# Parameter validation tests
class TestRandomDisplacementTransformationParameters:
    """Test parameter validation in RandomDisplacementTransformation."""

    def test_negative_rms(self):
        """Test that negative RMS raises ValueError."""
        with pytest.raises(ValueError, match="RMS must be positive"):
            RandomDisplacementTransformation(rms=-0.1)

    def test_zero_rms(self):
        """Test that zero RMS raises ValueError."""
        with pytest.raises(ValueError, match="RMS must be positive"):
            RandomDisplacementTransformation(rms=0.0)

    def test_negative_max_structures(self):
        """Test that negative max_structures raises ValueError."""
        with pytest.raises(ValueError,
                           match="max_structures must be positive"):
            RandomDisplacementTransformation(max_structures=-5)

    def test_zero_max_structures(self):
        """Test that zero max_structures raises ValueError."""
        with pytest.raises(ValueError,
                           match="max_structures must be positive"):
            RandomDisplacementTransformation(max_structures=0)

    def test_valid_parameters(self):
        """Test that valid parameters are accepted."""
        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=10,
            random_state=42,
            orthonormalize=True,
            remove_translations=True
        )
        assert transform.rms == 0.1
        assert transform.max_structures == 10
        assert transform.orthonormalize is True
        assert transform.remove_translations is True


# Reproducibility tests
class TestRandomDisplacementTransformationReproducibility:
    """Test reproducibility with random seeds."""

    def test_same_seed_same_results(self, simple_structure):
        """Test that same seed produces identical results."""
        seed = 42

        # First run
        transform1 = RandomDisplacementTransformation(
            rms=0.1, max_structures=5, random_state=seed
        )
        structures1 = list(transform1.apply_transformation(simple_structure))

        # Second run with same seed
        transform2 = RandomDisplacementTransformation(
            rms=0.1, max_structures=5, random_state=seed
        )
        structures2 = list(transform2.apply_transformation(simple_structure))

        # Should produce identical results
        assert len(structures1) == len(structures2)
        for s1, s2 in zip(structures1, structures2):
            np.testing.assert_allclose(
                s1.coords[-1], s2.coords[-1], rtol=1e-14
            )

    def test_different_seeds_different_results(self, simple_structure):
        """Test that different seeds produce different results."""
        # First run
        transform1 = RandomDisplacementTransformation(
            rms=0.1, max_structures=5, random_state=42
        )
        structures1 = list(transform1.apply_transformation(simple_structure))

        # Second run with different seed
        transform2 = RandomDisplacementTransformation(
            rms=0.1, max_structures=5, random_state=123
        )
        structures2 = list(transform2.apply_transformation(simple_structure))

        # Should produce different results
        assert len(structures1) == len(structures2)
        # At least one structure should be different
        differences = []
        for s1, s2 in zip(structures1, structures2):
            diff = np.max(np.abs(s1.coords[-1] - s2.coords[-1]))
            differences.append(diff)

        # At least one should have significant difference
        assert max(differences) > 1e-6

    def test_generator_reproducibility(self, simple_structure):
        """Test reproducibility with np.random.Generator."""
        # Create generators with same seed
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        transform1 = RandomDisplacementTransformation(
            rms=0.1, max_structures=5, random_state=rng1
        )
        structures1 = list(transform1.apply_transformation(simple_structure))

        transform2 = RandomDisplacementTransformation(
            rms=0.1, max_structures=5, random_state=rng2
        )
        structures2 = list(transform2.apply_transformation(simple_structure))

        # Should produce identical results
        assert len(structures1) == len(structures2)
        for s1, s2 in zip(structures1, structures2):
            np.testing.assert_allclose(
                s1.coords[-1], s2.coords[-1], rtol=1e-14
            )


# Physics constraint tests
class TestRandomDisplacementTransformationPhysics:
    """Test physical constraints (RMS, orthogonality)."""

    def test_rms_displacement_accuracy(self, simple_structure):
        """Test that RMS displacement matches target."""
        target_rms = 0.15
        transform = RandomDisplacementTransformation(
            rms=target_rms,
            max_structures=5,
            random_state=42
        )
        structures = list(transform.apply_transformation(simple_structure))

        # Check RMS for each structure
        for s in structures:
            displacement = s.coords[-1] - simple_structure.coords[-1]
            rms = np.sqrt(np.mean(displacement ** 2))

            # Should match target within tolerance
            assert abs(rms - target_rms
                       ) < RandomDisplacementTransformation.RMS_TOLERANCE

    def test_orthogonality_no_translation_removal(self, simple_structure):
        """
        Test orthogonality of displacement vectors without
        translation removal.
        """
        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=5,
            random_state=42,
            orthonormalize=True,
            remove_translations=False
        )
        structures = list(transform.apply_transformation(simple_structure))

        # Extract displacement vectors (flattened)
        displacements = []
        for s in structures:
            disp = s.coords[-1] - simple_structure.coords[-1]
            displacements.append(disp.flatten())

        # Check orthogonality (dot products should be ~0)
        for i in range(len(displacements)):
            for j in range(i + 1, len(displacements)):
                dot_product = np.dot(displacements[i], displacements[j])
                assert abs(dot_product) < 1e-10, \
                    f"Non-orthogonal vectors: dot({i},{j}) = {dot_product}"

    def test_orthogonality_with_translation_removal(self, simple_structure):
        """Test orthogonality after removing translational modes."""
        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=5,
            random_state=42,
            orthonormalize=True,
            remove_translations=True
        )
        structures = list(transform.apply_transformation(simple_structure))

        # Extract displacement vectors (flattened)
        displacements = []
        for s in structures:
            disp = s.coords[-1] - simple_structure.coords[-1]
            displacements.append(disp.flatten())

        # Check orthogonality
        for i in range(len(displacements)):
            for j in range(i + 1, len(displacements)):
                dot_product = np.dot(displacements[i], displacements[j])
                assert abs(dot_product) < 1e-10

    def test_translation_mode_removal(self, simple_structure):
        """Test that translational modes are removed."""
        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=3,
            random_state=42,
            orthonormalize=True,
            remove_translations=True
        )
        structures = list(transform.apply_transformation(simple_structure))

        # Define translational mode vectors
        natoms = simple_structure.natoms
        t_x = np.zeros((natoms, 3))
        t_y = np.zeros((natoms, 3))
        t_z = np.zeros((natoms, 3))
        t_x[:, 0] = 1.0
        t_y[:, 1] = 1.0
        t_z[:, 2] = 1.0

        # Normalize them
        t_x = t_x / np.linalg.norm(t_x)
        t_y = t_y / np.linalg.norm(t_y)
        t_z = t_z / np.linalg.norm(t_z)

        # Check that displacement vectors have no translational component
        for s in structures:
            disp = s.coords[-1] - simple_structure.coords[-1]
            disp_flat = disp.flatten()

            # Dot products with translational modes should be ~0
            dot_x = np.dot(disp_flat, t_x.flatten())
            dot_y = np.dot(disp_flat, t_y.flatten())
            dot_z = np.dot(disp_flat, t_z.flatten())

            assert abs(dot_x) < 1e-9, f"X-translation component: {dot_x}"
            assert abs(dot_y) < 1e-9, f"Y-translation component: {dot_y}"
            assert abs(dot_z) < 1e-9, (
                f"Z-translation component: {dot_z}"
            )

    def test_number_of_structures_with_translations(self, simple_structure):
        """Test correct number of structures without translation removal."""
        natoms = simple_structure.natoms
        max_structures = 3 * natoms  # Should be able to generate 3N

        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=max_structures,
            random_state=42,
            orthonormalize=True,
            remove_translations=False
        )
        structures = list(transform.apply_transformation(simple_structure))

        # Should generate exactly max_structures
        assert len(structures) == max_structures

    def test_number_of_structures_without_translations(self, simple_structure):
        """Test correct number of structures with translation removal."""
        natoms = simple_structure.natoms
        max_structures = 3 * natoms - 3  # Should be able to generate 3N-3

        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=max_structures,
            random_state=42,
            orthonormalize=True,
            remove_translations=True
        )
        structures = list(transform.apply_transformation(simple_structure))

        # Should generate exactly max_structures
        assert len(structures) == max_structures


# Edge case tests
class TestRandomDisplacementTransformationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_atom_no_translation_removal(self, single_atom_structure):
        """Test single atom structure without translation removal."""
        # Single atom has 3 DOF (3N = 3)
        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=3,
            random_state=42,
            orthonormalize=True,
            remove_translations=False
        )
        structures = list(
            transform.apply_transformation(single_atom_structure))

        assert len(structures) == 3

    def test_single_atom_with_translation_removal(self, single_atom_structure):
        """Test single atom structure with translation removal."""
        # Single atom has 0 non-translational DOF (3N-3 = 0)
        # Should generate no structures
        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=3,
            random_state=42,
            orthonormalize=True,
            remove_translations=True
        )

        # Should warn about insufficient vectors
        with pytest.warns(RuntimeWarning, match="only .* orthonormal vectors"):
            structures = list(transform.apply_transformation(
                single_atom_structure
            ))

        # Should generate 0 structures
        assert len(structures) == 0

    def test_request_too_many_structures(self, simple_structure):
        """Test requesting more structures than available."""
        natoms = simple_structure.natoms
        max_available = 3 * natoms - 3  # 3N-3 with translation removal
        requested = max_available + 10

        transform = RandomDisplacementTransformation(
            rms=0.1,
            max_structures=requested,
            random_state=42,
            orthonormalize=True,
            remove_translations=True
        )

        # Should warn about limit
        with pytest.warns(RuntimeWarning, match="only .* orthonormal vectors"):
            structures = list(transform.apply_transformation(simple_structure))

        # Should generate only max_available structures
        assert len(structures) == max_available

    def test_very_small_rms(self, simple_structure):
        """Test with very small RMS displacement."""
        transform = RandomDisplacementTransformation(
            rms=1e-8,
            max_structures=3,
            random_state=42
        )
        structures = list(transform.apply_transformation(simple_structure))

        # Should still work
        assert len(structures) == 3

        # Verify small displacements
        for s in structures:
            disp = s.coords[-1] - simple_structure.coords[-1]
            max_disp = np.max(np.abs(disp))
            assert max_disp < 1e-6

    def test_very_large_rms(self, simple_structure):
        """Test with very large RMS displacement."""
        transform = RandomDisplacementTransformation(
            rms=10.0,
            max_structures=3,
            random_state=42
        )
        structures = list(transform.apply_transformation(simple_structure))

        # Should still work
        assert len(structures) == 3

        # Verify RMS is correct
        for s in structures:
            disp = s.coords[-1] - simple_structure.coords[-1]
            rms = np.sqrt(np.mean(disp ** 2))
            tolerance = RandomDisplacementTransformation.RMS_TOLERANCE
            assert abs(rms - 10.0) < tolerance


# Integration tests
class TestRandomDisplacementTransformationIntegration:
    """Test integration with other transformation features."""

    def test_chain_with_deterministic_transform(self, simple_structure):
        """Test chaining with a deterministic transformation."""
        chain = TransformationChain(
            [
                RandomDisplacementTransformation(
                    rms=0.1, max_structures=3, random_state=42
                ),
                AtomDisplacementTransformation(displacement=0.05)
            ]
        )

        # Consume chain lazily and count up to 30
        it = chain.apply_transformation(simple_structure)
        first_30 = list(itertools.islice(it, 30))
        # For 3 atoms: RAND(3) × DISP(9) = 27 total if fully consumed
        assert len(first_30) == 27

    def test_chain_laziness_and_islice(self, simple_structure):
        """Test chain laziness with itertools.islice."""
        chain = TransformationChain(
            [
                RandomDisplacementTransformation(
                    rms=0.1, max_structures=6, random_state=42
                )
            ]
        )
        it = chain.apply_transformation(simple_structure)
        first_5 = list(itertools.islice(it, 5))
        assert len(first_5) == 5


# Performance tests
class TestRandomDisplacementTransformationPerformance:
    """Test performance and memory usage."""

    def test_medium_system_performance(self, larger_structure):
        """Test performance with medium-sized system (10 atoms)."""
        transform = RandomDisplacementTransformation(
            rms=0.1, max_structures=20, random_state=42
        )

        start = time.time()
        structures = list(transform.apply_transformation(larger_structure))
        elapsed = time.time() - start

        # Should complete reasonably quickly (< 1 second)
        assert elapsed < 1.0
        assert len(structures) == 20

    def test_qr_decomposition_stability(self, larger_structure):
        """Test numerical stability of QR decomposition."""
        transform = RandomDisplacementTransformation(
            rms=0.1, max_structures=15, random_state=42
        )
        structures = list(transform.apply_transformation(larger_structure))

        # Extract displacements
        displacements = []
        for s in structures:
            disp = s.coords[-1] - larger_structure.coords[-1]
            displacements.append(disp.flatten())

        # Verify near-orthogonality (after RMS scaling, not unit norm)
        for i in range(len(displacements)):
            for j in range(i + 1, len(displacements)):
                dot_product = np.dot(displacements[i], displacements[j])
                assert abs(dot_product) < 1e-9


class TestDOptimalDisplacementTransformationParameters:
    """Test parameter validation in DOptimalDisplacementTransformation."""

    def test_invalid_rms(self):
        """Negative or zero RMS should raise ValueError."""
        with pytest.raises(ValueError, match="RMS must be positive"):
            DOptimalDisplacementTransformation(rms=-0.1)
        with pytest.raises(ValueError, match="RMS must be positive"):
            DOptimalDisplacementTransformation(rms=0.0)

    def test_invalid_n_structures(self):
        """n_structures must be at least 2."""
        with pytest.raises(
              ValueError, match="n_structures must be at least 2"):
            DOptimalDisplacementTransformation(n_structures=1)

    def test_invalid_optimization_parameters(self):
        """Invalid optimization parameters should raise ValueError."""
        with pytest.raises(ValueError, match="max_iter must be positive"):
            DOptimalDisplacementTransformation(max_iter=0)
        with pytest.raises(ValueError,
                           match="learning_rate must be positive"):
            DOptimalDisplacementTransformation(learning_rate=0.0)
        with pytest.raises(ValueError, match="tol must be positive"):
            DOptimalDisplacementTransformation(tol=0.0)
        with pytest.raises(ValueError,
                           match="logdet_regularization must be positive"):
            DOptimalDisplacementTransformation(logdet_regularization=0.0)

    def test_valid_parameters(self):
        """Valid parameters should be accepted and stored."""
        transform = DOptimalDisplacementTransformation(
            rms=0.1,
            n_structures=8,
            max_iter=50,
            learning_rate=0.2,
            tol=1e-4,
            logdet_regularization=1e-6,
            random_state=42,
            remove_translations=True,
            enforce_zero_mean=True,
            verbose=True,
        )
        assert transform.rms == 0.1
        assert transform.n_structures == 8
        assert transform.max_iter == 50
        assert transform.learning_rate == 0.2
        assert transform.tol == 1e-4
        assert transform.logdet_regularization == 1e-6
        assert transform.remove_translations is True
        assert transform.enforce_zero_mean is True
        assert transform.verbose is True


class TestDOptimalDisplacementTransformationConstraints:
    """Test that D-optimal displacements satisfy physical constraints."""

    def test_rms_and_com_and_zero_mean(self, simple_structure):
        """RMS, COM, and ensemble-mean constraints should hold."""
        target_rms = 0.15
        n_structures = 6
        transform = DOptimalDisplacementTransformation(
            rms=target_rms,
            n_structures=n_structures,
            max_iter=50,
            learning_rate=0.2,
            random_state=42,
            remove_translations=True,
            enforce_zero_mean=True,
        )

        structures = list(transform.apply_transformation(simple_structure))
        assert len(structures) == n_structures

        flat_displacements = []

        for s in structures:
            disp = s.coords[-1] - simple_structure.coords[-1]

            # RMS per structure should match target
            # Tolerance relaxed to 1e-3 Å to account for competing constraints
            # (zero-mean centering slightly perturbs RMS after scaling)
            rms = np.sqrt(np.mean(disp ** 2))
            assert abs(rms - target_rms) < 1e-3

            # Center-of-mass shift should be ~0
            com_shift = disp.mean(axis=0)
            assert np.linalg.norm(com_shift) < 1e-9

            flat_displacements.append(disp.flatten())

        # Ensemble mean displacement should be ~0 (up to numerical noise)
        X = np.vstack(flat_displacements)
        mean_vec = X.mean(axis=0)
        # Allow for floating-point roundoff after repeated projections
        assert np.linalg.norm(mean_vec) < 1e-6


class TestDOptimalDisplacementTransformationDiversity:
    """Compare diversity of D-optimal vs random displacements."""

    @staticmethod
    def _logdet_cov(X: np.ndarray, eps: float = 1e-6) -> float:
        """Helper: compute log det of regularized covariance of rows of X."""
        n_p, d = X.shape
        if n_p <= 1:
            return -np.inf
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / float(n_p - 1)
        Sigma_reg = Sigma + eps * np.eye(d)
        sign, logdet = np.linalg.slogdet(Sigma_reg)
        if sign <= 0:
            return -np.inf
        return logdet

    def test_doptimal_has_at_least_random_diversity(self, simple_structure):
        """D-optimal sampler should achieve >= random logdet Σ (for a seed)."""
        n_structures = 8
        rms = 0.1

        # Baseline: random displacements (non-orthonormalized)
        rand_transform = RandomDisplacementTransformation(
            rms=rms,
            max_structures=n_structures,
            random_state=42,
            orthonormalize=False,
            remove_translations=True,
        )
        rand_structures = list(
            rand_transform.apply_transformation(simple_structure)
        )

        # D-optimal sampler
        dopt_transform = DOptimalDisplacementTransformation(
            rms=rms,
            n_structures=n_structures,
            max_iter=50,
            learning_rate=0.2,
            random_state=42,
            remove_translations=True,
            enforce_zero_mean=True,
        )
        dopt_structures = list(
            dopt_transform.apply_transformation(simple_structure)
        )

        # Extract flattened displacements
        def _get_flat_disps(structures, ref):
            return np.vstack([
                (s.coords[-1] - ref.coords[-1]).flatten()
                for s in structures
            ])

        X_rand = _get_flat_disps(rand_structures, simple_structure)
        X_dopt = _get_flat_disps(dopt_structures, simple_structure)

        logdet_rand = self._logdet_cov(X_rand)
        logdet_dopt = self._logdet_cov(X_dopt)

        # Allow a small tolerance for stochastic differences but
        # expect D-optimal to be at least as good as random.
        assert logdet_dopt + 1e-8 >= logdet_rand


class TestDOptimalDisplacementTransformationEdgeCases:
    """Test edge cases and performance of D-optimal sampler."""

    def test_small_n_structures(self, simple_structure):
        """Sampler should work with very small n_structures (>=2)."""
        transform = DOptimalDisplacementTransformation(
            rms=0.1,
            n_structures=2,
            max_iter=30,
            random_state=123,
        )
        structures = list(transform.apply_transformation(simple_structure))
        assert len(structures) == 2

    def test_medium_system_performance(self, larger_structure):
        """Test performance with a medium-sized system."""
        transform = DOptimalDisplacementTransformation(
            rms=0.1,
            n_structures=10,
            max_iter=50,
            random_state=42,
        )

        start = time.time()
        structures = list(transform.apply_transformation(larger_structure))
        elapsed = time.time() - start

        assert elapsed < 2.0
        assert len(structures) == 10
