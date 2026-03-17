"""
Tests for deterministic transformation classes (iterator-based API).

This module tests the deterministic transformation implementations
including AtomDisplacementTransformation, CellVolumeTransformation,
CellTransformationMatrix, IsovolumetricStrainTransformation,
MonoclinicStrainTransformation, OrthorhombicStrainTransformation,
ShearStrainTransformation, and UniaxialStrainTransformation.
"""

import itertools

import numpy as np
import pytest

from aenet.geometry import AtomicStructure
from aenet.geometry.transformations import (
    AtomDisplacementTransformation,
    CellTransformationMatrix,
    CellVolumeTransformation,
    IsovolumetricStrainTransformation,
    MonoclinicStrainTransformation,
    OrthorhombicStrainTransformation,
    ShearStrainTransformation,
    UniaxialStrainTransformation,
)


# Fixtures
@pytest.fixture
def simple_structure():
    """Create a simple 2-atom structure for testing."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    types = ["H", "H"]
    avec = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    return AtomicStructure(coords, types, avec=avec)


@pytest.fixture
def water_structure():
    """Create a water molecule structure."""
    coords = [
        [0.0, 0.0, 0.0],      # O
        [0.757, 0.586, 0.0],  # H
        [-0.757, 0.586, 0.0]  # H
    ]
    types = ["O", "H", "H"]
    avec = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    return AtomicStructure(coords, types, avec=avec)


@pytest.fixture
def non_pbc_structure():
    """Create a non-periodic structure."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    types = ["C", "C"]
    return AtomicStructure(coords, types)


@pytest.fixture
def skewed_structure_with_labels():
    """Create a skewed periodic structure with labels for regression tests."""
    coords = np.array([
        [0.35, 0.55, 0.75],
        [1.60, 1.25, 2.40],
    ])
    types = ["Si", "O"]
    avec = np.array([
        [4.2, 0.0, 0.0],
        [0.7, 5.1, 0.0],
        [0.3, 1.1, 6.3],
    ])
    forces = np.array([
        [0.2, -0.1, 0.4],
        [-0.3, 0.6, -0.5],
    ])
    return AtomicStructure(
        coords,
        types,
        avec=avec,
        energy=-7.5,
        forces=forces,
    )


# Tests for AtomDisplacementTransformation
class TestAtomDisplacementTransformation:
    """Test suite for AtomDisplacementTransformation."""

    def test_initialization(self):
        """Test basic initialization."""
        transform = AtomDisplacementTransformation(displacement=0.05)
        assert transform.displacement == 0.05

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        transform = AtomDisplacementTransformation()
        assert transform.displacement == 0.1

    def test_invalid_displacement_negative(self):
        """Test that negative displacement raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            AtomDisplacementTransformation(displacement=-0.1)

    def test_invalid_displacement_zero(self):
        """Test that zero displacement raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            AtomDisplacementTransformation(displacement=0.0)

    def test_structure_count(self, simple_structure):
        """Test correct number of structures generated."""
        transform = AtomDisplacementTransformation(displacement=0.1)
        results = list(transform.apply_transformation(simple_structure))
        # 2 atoms × 3 directions = 6 structures
        assert len(results) == 6

    def test_structure_count_water(self, water_structure):
        """Test correct number of structures for water molecule."""
        transform = AtomDisplacementTransformation(displacement=0.1)
        results = list(transform.apply_transformation(water_structure))
        # 3 atoms × 3 directions = 9 structures
        assert len(results) == 9

    def test_displacement_magnitude(self, simple_structure):
        """Test that displacement magnitudes are correct."""
        displacement = 0.05
        transform = AtomDisplacementTransformation(displacement=displacement)
        results = list(transform.apply_transformation(simple_structure))

        original_coords = simple_structure.coords[-1]

        # Check first 3 structures (first atom displaced in x, y, z)
        for i in range(3):
            displaced_coords = results[i].coords[-1]
            diff = displaced_coords[0] - original_coords[0]
            magnitude = np.linalg.norm(diff)
            assert np.isclose(magnitude, displacement, atol=1e-10)

    def test_displacement_directions(self, simple_structure):
        """Test that displacements are in correct directions."""
        displacement = 0.1
        transform = AtomDisplacementTransformation(displacement=displacement)
        results = list(transform.apply_transformation(simple_structure))

        original_coords = simple_structure.coords[-1][0]

        # First atom: x direction
        diff_x = results[0].coords[-1][0] - original_coords
        assert np.allclose(diff_x, [displacement, 0.0, 0.0])

        # First atom: y direction
        diff_y = results[1].coords[-1][0] - original_coords
        assert np.allclose(diff_y, [0.0, displacement, 0.0])

        # First atom: z direction
        diff_z = results[2].coords[-1][0] - original_coords
        assert np.allclose(diff_z, [0.0, 0.0, displacement])

    def test_other_atoms_unchanged(self, simple_structure):
        """Test that non-displaced atoms remain unchanged."""
        transform = AtomDisplacementTransformation(displacement=0.1)
        results = list(transform.apply_transformation(simple_structure))

        original_coords = simple_structure.coords[-1]

        # When first atom is displaced, second atom should be unchanged
        for i in range(3):
            assert np.allclose(
                results[i].coords[-1][1],
                original_coords[1]
            )

        # When second atom is displaced, first atom should be unchanged
        for i in range(3, 6):
            assert np.allclose(
                results[i].coords[-1][0],
                original_coords[0]
            )

    def test_take_first_n_structures(self, simple_structure):
        """Test taking first N structures from iterator."""
        transform = AtomDisplacementTransformation(displacement=0.1)
        iterator = transform.apply_transformation(simple_structure)
        first_three = list(itertools.islice(iterator, 3))
        assert len(first_three) == 3

    def test_structure_properties_preserved(self, simple_structure):
        """Test that structure properties are preserved."""
        transform = AtomDisplacementTransformation(displacement=0.1)
        results = list(transform.apply_transformation(simple_structure))

        for result in results:
            assert result.natoms == simple_structure.natoms
            assert result.pbc == simple_structure.pbc
            assert np.allclose(result.avec[-1], simple_structure.avec[-1])
            assert all(result.types == simple_structure.types)

    def test_non_pbc_structure(self, non_pbc_structure):
        """Test transformation works with non-periodic structures."""
        transform = AtomDisplacementTransformation(displacement=0.1)
        results = list(transform.apply_transformation(non_pbc_structure))
        assert len(results) == 3 * non_pbc_structure.natoms
        for result in results:
            assert not result.pbc

    def test_single_atom_structure(self):
        """Test with single atom structure."""
        coords = [[0.0, 0.0, 0.0]]
        types = ["He"]
        avec = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
        structure = AtomicStructure(coords, types, avec=avec)

        transform = AtomDisplacementTransformation(displacement=0.1)
        results = list(transform.apply_transformation(structure))
        assert len(results) == 3  # 1 atom × 3 directions

    def test_large_displacement(self, simple_structure):
        """Test with large displacement value."""
        displacement = 2.0
        transform = AtomDisplacementTransformation(displacement=displacement)
        results = list(transform.apply_transformation(simple_structure))

        # Check displacement magnitude
        original_coords = simple_structure.coords[-1][0]
        displaced_coords = results[0].coords[-1][0]
        diff = displaced_coords - original_coords
        magnitude = np.linalg.norm(diff)
        assert np.isclose(magnitude, displacement)

    def test_small_displacement(self, simple_structure):
        """Test with very small displacement value."""
        displacement = 0.001
        transform = AtomDisplacementTransformation(displacement=displacement)
        results = list(transform.apply_transformation(simple_structure))

        # Check displacement magnitude
        original_coords = simple_structure.coords[-1][0]
        displaced_coords = results[0].coords[-1][0]
        diff = displaced_coords - original_coords
        magnitude = np.linalg.norm(diff)
        assert np.isclose(magnitude, displacement, atol=1e-10)

    def test_structure_independence(self, simple_structure):
        """Test that generated structures are independent copies."""
        transform = AtomDisplacementTransformation(displacement=0.1)
        results = list(transform.apply_transformation(simple_structure))

        # Store original coordinates from first structure
        original_first_coords = results[0].coords[-1].copy()

        # Modify the first structure
        results[0].coords[-1][0][0] += 10.0

        # Verify the modification took effect
        expected = original_first_coords[0][0] + 10.0
        assert results[0].coords[-1][0][0] == expected

        # Verify other structures were not affected
        for i in range(1, len(results)):
            assert (results[i].coords[-1][0][0] !=
                    results[0].coords[-1][0][0])


# Tests for CellVolumeTransformation
class TestCellVolumeTransformation:
    """Test suite for CellVolumeTransformation."""

    def test_initialization(self):
        """Test basic initialization."""
        transform = CellVolumeTransformation(
            min_percent=-5.0, max_percent=5.0, steps=5
        )
        assert transform.min_percent == -5.0
        assert transform.max_percent == 5.0
        assert transform.steps == 5

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        transform = CellVolumeTransformation()
        assert transform.min_percent == -5.0
        assert transform.max_percent == 5.0
        assert transform.steps == 5

    def test_invalid_steps(self):
        """Test that non-positive steps raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            CellVolumeTransformation(steps=0)

    def test_invalid_percent_range(self):
        """Test that min_percent >= max_percent raises error."""
        with pytest.raises(ValueError, match="must be less than"):
            CellVolumeTransformation(min_percent=5.0, max_percent=-5.0)

    def test_structure_count(self, simple_structure):
        """Test correct number of structures generated."""
        transform = CellVolumeTransformation(steps=7)
        results = list(transform.apply_transformation(simple_structure))
        assert len(results) == 7

    def test_volume_scaling(self, simple_structure):
        """Test that volumes are correctly scaled."""
        transform = CellVolumeTransformation(
            min_percent=-10.0, max_percent=10.0, steps=5
        )
        results = list(transform.apply_transformation(simple_structure))

        original_volume = simple_structure.cellvolume()
        expected_scales = [0.9, 0.95, 1.0, 1.05, 1.1]

        for i, scale in enumerate(expected_scales):
            expected_volume = original_volume * (scale ** 3)
            actual_volume = results[i].cellvolume()
            assert np.isclose(actual_volume, expected_volume, rtol=1e-10)

    def test_lattice_vector_scaling(self, simple_structure):
        """Test that lattice vectors are uniformly scaled."""
        transform = CellVolumeTransformation(
            min_percent=-5.0, max_percent=5.0, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        scales = [0.95,  1.0, 1.05]

        for i, scale in enumerate(scales):
            expected_avec = simple_structure.avec[-1] * scale
            assert np.allclose(results[i].avec[-1], expected_avec)

    def test_non_pbc_raises_error(self, non_pbc_structure):
        """Test that non-periodic structure raises error."""
        transform = CellVolumeTransformation()
        with pytest.raises(ValueError, match="requires periodic"):
            # Should raise at call time
            transform.apply_transformation(non_pbc_structure)

    def test_preserves_fractional_coordinates_and_clears_labels(
        self, skewed_structure_with_labels
    ):
        """Volume scaling should rebuild Cartesian coordinates from fractions."""
        transform = CellVolumeTransformation(
            min_percent=5.0,
            max_percent=10.0,
            steps=2,
        )
        frac_before = skewed_structure_with_labels.cart2frac(
            skewed_structure_with_labels.coords[-1]
        )

        result = next(iter(transform.apply_transformation(
            skewed_structure_with_labels
        )))

        np.testing.assert_allclose(
            result.cart2frac(result.coords[-1]),
            frac_before,
        )
        assert not np.allclose(
            result.coords[-1],
            skewed_structure_with_labels.coords[-1],
        )
        assert result.energy[-1] is None
        assert result.forces[-1] is None


# Tests for IsovolumetricStrainTransformation
class TestIsovolumetricStrainTransformation:
    """Test suite for IsovolumetricStrainTransformation."""

    def test_initialization(self):
        """Test basic initialization."""
        transform = IsovolumetricStrainTransformation(
            direction=1, len_min=0.9, len_max=1.1, steps=5
        )
        assert transform.direction == 1
        assert transform.len_min == 0.9
        assert transform.len_max == 1.1
        assert transform.steps == 5

    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="must be 1, 2, or 3"):
            IsovolumetricStrainTransformation(
                direction=4, len_min=0.9, len_max=1.1, steps=5
            )

    def test_invalid_steps(self):
        """Test that non-positive steps raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            IsovolumetricStrainTransformation(
                direction=1, len_min=0.9, len_max=1.1, steps=0
            )

    def test_invalid_scaling_factors(self):
        """Test that non-positive scaling factors raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            IsovolumetricStrainTransformation(
                direction=1, len_min=-0.1, len_max=1.1, steps=5
            )

    def test_volume_conservation(self, simple_structure):
        """Test that volume is conserved."""
        transform = IsovolumetricStrainTransformation(
            direction=1, len_min=0.8, len_max=1.2, steps=5
        )
        results = list(transform.apply_transformation(simple_structure))

        original_volume = simple_structure.cellvolume()

        for result in results:
            volume_error = abs(result.cellvolume() - original_volume)
            assert volume_error < 1e-5  # VOLUME_TOLERANCE

    def test_direction_1_scaling(self, simple_structure):
        """Test scaling in direction 1 (a-axis)."""
        transform = IsovolumetricStrainTransformation(
            direction=1, len_min=0.9, len_max=1.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        s_values = [0.9, 1.0, 1.1]

        for i, s in enumerate(s_values):
            s_orth = (1.0 / s) ** 0.5
            scaling = np.diag([s, s_orth, s_orth])
            expected_avec = simple_structure.avec[-1] @ scaling
            assert np.allclose(results[i].avec[-1], expected_avec)

    def test_direction_2_scaling(self, simple_structure):
        """Test scaling in direction 2 (b-axis)."""
        transform = IsovolumetricStrainTransformation(
            direction=2, len_min=0.9, len_max=1.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        s_values = [0.9, 1.0, 1.1]

        for i, s in enumerate(s_values):
            s_orth = (1.0 / s) ** 0.5
            scaling = np.diag([s_orth, s, s_orth])
            expected_avec = simple_structure.avec[-1] @ scaling
            assert np.allclose(results[i].avec[-1], expected_avec)

    def test_direction_3_scaling(self, simple_structure):
        """Test scaling in direction 3 (c-axis)."""
        transform = IsovolumetricStrainTransformation(
            direction=3, len_min=0.9, len_max=1.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        s_values = [0.9, 1.0, 1.1]

        for i, s in enumerate(s_values):
            s_orth = (1.0 / s) ** 0.5
            scaling = np.diag([s_orth, s_orth, s])
            expected_avec = simple_structure.avec[-1] @ scaling
            assert np.allclose(results[i].avec[-1], expected_avec)

    def test_non_pbc_raises_error(self, non_pbc_structure):
        """Test that non-periodic structure raises error."""
        transform = IsovolumetricStrainTransformation(
            direction=1, len_min=0.9, len_max=1.1, steps=5
        )
        with pytest.raises(ValueError, match="requires periodic"):
            # Should raise at call time
            transform.apply_transformation(non_pbc_structure)

    def test_preserves_fractional_coordinates_and_clears_labels(
        self, skewed_structure_with_labels
    ):
        """Isovolumetric strain should preserve fractions and drop labels."""
        transform = IsovolumetricStrainTransformation(
            direction=2,
            len_min=1.1,
            len_max=1.2,
            steps=2,
        )
        frac_before = skewed_structure_with_labels.cart2frac(
            skewed_structure_with_labels.coords[-1]
        )

        result = next(iter(transform.apply_transformation(
            skewed_structure_with_labels
        )))

        np.testing.assert_allclose(
            result.cart2frac(result.coords[-1]),
            frac_before,
        )
        assert not np.allclose(
            result.coords[-1],
            skewed_structure_with_labels.coords[-1],
        )
        assert result.energy[-1] is None
        assert result.forces[-1] is None


# Tests for ShearStrainTransformation
class TestShearStrainTransformation:
    """Test suite for ShearStrainTransformation."""

    def test_initialization(self):
        """Test basic initialization."""
        transform = ShearStrainTransformation(
            direction=1, shear_min=-0.1, shear_max=0.1, steps=5
        )
        assert transform.direction == 1
        assert transform.shear_min == -0.1
        assert transform.shear_max == 0.1
        assert transform.steps == 5

    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="must be 1, 2, or 3"):
            ShearStrainTransformation(
                direction=0, shear_min=-0.1, shear_max=0.1, steps=5
            )

    def test_invalid_steps(self):
        """Test that non-positive steps raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            ShearStrainTransformation(
                direction=1, shear_min=-0.1, shear_max=0.1, steps=-1
            )

    def test_invalid_shear_range(self):
        """Test that shear_min >= shear_max raises error."""
        with pytest.raises(ValueError, match="must be less than"):
            ShearStrainTransformation(
                direction=1, shear_min=0.1, shear_max=-0.1, steps=5
            )

    def test_determinant_preservation(self, simple_structure):
        """Test that determinant equals 1 (volume preservation)."""
        transform = ShearStrainTransformation(
            direction=1, shear_min=-0.2, shear_max=0.2, steps=5
        )
        results = list(transform.apply_transformation(simple_structure))

        for result in results:
            # Compute determinant of transformation
            det = np.linalg.det(
                result.avec[-1] @ np.linalg.inv(simple_structure.avec[-1])
            )
            assert np.isclose(det, 1.0, atol=1e-10)

    def test_direction_1_shear(self, simple_structure):
        """Test xy shear (direction 1)."""
        transform = ShearStrainTransformation(
            direction=1, shear_min=-0.1, shear_max=0.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        shear_values = [-0.1, 0.0, 0.1]

        for i, shear in enumerate(shear_values):
            shear_matrix = np.eye(3)
            shear_matrix[0, 1] = shear
            expected_avec = simple_structure.avec[-1] @ shear_matrix
            assert np.allclose(results[i].avec[-1], expected_avec)

    def test_direction_2_shear(self, simple_structure):
        """Test xz shear (direction 2)."""
        transform = ShearStrainTransformation(
            direction=2, shear_min=-0.1, shear_max=0.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        shear_values = [-0.1, 0.0, 0.1]

        for i, shear in enumerate(shear_values):
            shear_matrix = np.eye(3)
            shear_matrix[0, 2] = shear
            expected_avec = simple_structure.avec[-1] @ shear_matrix
            assert np.allclose(results[i].avec[-1], expected_avec)

    def test_direction_3_shear(self, simple_structure):
        """Test yz shear (direction 3)."""
        transform = ShearStrainTransformation(
            direction=3, shear_min=-0.1, shear_max=0.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        shear_values = [-0.1, 0.0, 0.1]

        for i, shear in enumerate(shear_values):
            shear_matrix = np.eye(3)
            shear_matrix[1, 2] = shear
            expected_avec = simple_structure.avec[-1] @ shear_matrix
            assert np.allclose(results[i].avec[-1], expected_avec)

    def test_non_pbc_raises_error(self, non_pbc_structure):
        """Test that non-periodic structure raises error."""
        transform = ShearStrainTransformation(
            direction=1, shear_min=-0.1, shear_max=0.1, steps=5
        )
        with pytest.raises(ValueError, match="requires periodic"):
            # Should raise at call time
            transform.apply_transformation(non_pbc_structure)

    def test_preserves_fractional_coordinates_and_clears_labels(
        self, skewed_structure_with_labels
    ):
        """Shear should preserve fractions and clear stale labels."""
        transform = ShearStrainTransformation(
            direction=3,
            shear_min=0.1,
            shear_max=0.2,
            steps=2,
        )
        frac_before = skewed_structure_with_labels.cart2frac(
            skewed_structure_with_labels.coords[-1]
        )

        result = next(iter(transform.apply_transformation(
            skewed_structure_with_labels
        )))

        np.testing.assert_allclose(
            result.cart2frac(result.coords[-1]),
            frac_before,
        )
        assert not np.allclose(
            result.coords[-1],
            skewed_structure_with_labels.coords[-1],
        )
        assert result.energy[-1] is None
        assert result.forces[-1] is None


# Tests for UniaxialStrainTransformation
class TestUniaxialStrainTransformation:
    """Test suite for UniaxialStrainTransformation."""

    def test_initialization(self):
        """Test basic initialization."""
        transform = UniaxialStrainTransformation(
            direction=1, len_min=0.9, len_max=1.1, steps=5
        )
        assert transform.direction == 1
        assert transform.len_min == 0.9
        assert transform.len_max == 1.1
        assert transform.steps == 5

    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="must be 1, 2, or 3"):
            UniaxialStrainTransformation(
                direction=0, len_min=0.9, len_max=1.1, steps=5
            )

    def test_structure_count(self, simple_structure):
        """Test correct number of structures generated."""
        transform = UniaxialStrainTransformation(
            direction=1, len_min=0.9, len_max=1.1, steps=7
        )
        results = list(transform.apply_transformation(simple_structure))
        assert len(results) == 7

    def test_direction_1_scaling(self, simple_structure):
        """Test that direction 1 scales only a-axis."""
        transform = UniaxialStrainTransformation(
            direction=1, len_min=0.9, len_max=1.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        s_values = [0.9, 1.0, 1.1]
        for i, s in enumerate(s_values):
            scaling = np.diag([s, 1.0, 1.0])
            expected_avec = simple_structure.avec[-1] @ scaling
            np.testing.assert_allclose(results[i].avec[-1], expected_avec)

    def test_direction_2_scaling(self, simple_structure):
        """Test that direction 2 scales only b-axis."""
        transform = UniaxialStrainTransformation(
            direction=2, len_min=0.9, len_max=1.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        s_values = [0.9, 1.0, 1.1]
        for i, s in enumerate(s_values):
            scaling = np.diag([1.0, s, 1.0])
            expected_avec = simple_structure.avec[-1] @ scaling
            np.testing.assert_allclose(results[i].avec[-1], expected_avec)

    def test_volume_changes(self, simple_structure):
        """Test that volume changes as expected."""
        transform = UniaxialStrainTransformation(
            direction=1, len_min=0.9, len_max=1.1, steps=3
        )
        results = list(transform.apply_transformation(simple_structure))

        original_volume = simple_structure.cellvolume()
        s_values = [0.9, 1.0, 1.1]

        for i, s in enumerate(s_values):
            expected_volume = original_volume * s
            actual_volume = results[i].cellvolume()
            assert np.isclose(actual_volume, expected_volume)

    def test_non_pbc_raises_error(self, non_pbc_structure):
        """Test that non-periodic structure raises error."""
        transform = UniaxialStrainTransformation(
            direction=1, len_min=0.9, len_max=1.1, steps=5
        )
        with pytest.raises(ValueError, match="requires periodic"):
            transform.apply_transformation(non_pbc_structure)

    def test_preserves_fractional_coordinates_and_clears_labels(
        self, skewed_structure_with_labels
    ):
        """Uniaxial strain should preserve fractions and clear stale labels."""
        transform = UniaxialStrainTransformation(
            direction=1,
            len_min=1.1,
            len_max=1.2,
            steps=2,
        )
        frac_before = skewed_structure_with_labels.cart2frac(
            skewed_structure_with_labels.coords[-1]
        )

        result = next(iter(transform.apply_transformation(
            skewed_structure_with_labels
        )))

        np.testing.assert_allclose(
            result.cart2frac(result.coords[-1]),
            frac_before,
        )
        assert not np.allclose(
            result.coords[-1],
            skewed_structure_with_labels.coords[-1],
        )
        assert result.energy[-1] is None
        assert result.forces[-1] is None


# Tests for OrthorhombicStrainTransformation
class TestOrthorhombicStrainTransformation:
    """Test suite for OrthorhombicStrainTransformation."""

    def test_initialization(self):
        """Test basic initialization."""
        transform = OrthorhombicStrainTransformation(
            direction=1, e_min=-0.02, e_max=0.02, steps=5
        )
        assert transform.direction == 1
        assert transform.e_min == -0.02
        assert transform.e_max == 0.02
        assert transform.steps == 5

    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="must be 1, 2, or 3"):
            OrthorhombicStrainTransformation(
                direction=4, e_min=-0.02, e_max=0.02, steps=5
            )

    def test_structure_count(self, simple_structure):
        """Test correct number of structures generated."""
        transform = OrthorhombicStrainTransformation(
            direction=1, e_min=-0.02, e_max=0.02, steps=5
        )
        results = list(transform.apply_transformation(simple_structure))
        assert len(results) == 5

    def test_volume_conservation(self, simple_structure):
        """Test that volume is approximately conserved."""
        transform = OrthorhombicStrainTransformation(
            direction=1, e_min=-0.05, e_max=0.05, steps=5
        )
        results = list(transform.apply_transformation(simple_structure))

        original_volume = simple_structure.cellvolume()

        for result in results:
            volume_error = abs(result.cellvolume() - original_volume)
            assert volume_error < 1e-10

    def test_non_pbc_raises_error(self, non_pbc_structure):
        """Test that non-periodic structure raises error."""
        transform = OrthorhombicStrainTransformation(
            direction=1, e_min=-0.02, e_max=0.02, steps=5
        )
        with pytest.raises(ValueError, match="requires periodic"):
            transform.apply_transformation(non_pbc_structure)

    def test_preserves_fractional_coordinates_and_clears_labels(
        self, skewed_structure_with_labels
    ):
        """Orthorhombic strain should preserve fractions and clear labels."""
        transform = OrthorhombicStrainTransformation(
            direction=2,
            e_min=0.03,
            e_max=0.05,
            steps=2,
        )
        frac_before = skewed_structure_with_labels.cart2frac(
            skewed_structure_with_labels.coords[-1]
        )

        result = next(iter(transform.apply_transformation(
            skewed_structure_with_labels
        )))

        np.testing.assert_allclose(
            result.cart2frac(result.coords[-1]),
            frac_before,
        )
        assert not np.allclose(
            result.coords[-1],
            skewed_structure_with_labels.coords[-1],
        )
        assert result.energy[-1] is None
        assert result.forces[-1] is None


# Tests for MonoclinicStrainTransformation
class TestMonoclinicStrainTransformation:
    """Test suite for MonoclinicStrainTransformation."""

    def test_initialization(self):
        """Test basic initialization."""
        transform = MonoclinicStrainTransformation(
            direction=1, gamma_min=-0.1, gamma_max=0.1, steps=5
        )
        assert transform.direction == 1
        assert transform.gamma_min == -0.1
        assert transform.gamma_max == 0.1
        assert transform.steps == 5

    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="must be 1, 2, or 3"):
            MonoclinicStrainTransformation(
                direction=0, gamma_min=-0.1, gamma_max=0.1, steps=5
            )

    def test_structure_count(self, simple_structure):
        """Test correct number of structures generated."""
        transform = MonoclinicStrainTransformation(
            direction=1, gamma_min=-0.1, gamma_max=0.1, steps=7
        )
        results = list(transform.apply_transformation(simple_structure))
        assert len(results) == 7

    def test_volume_conservation(self, simple_structure):
        """Test that volume is approximately conserved."""
        transform = MonoclinicStrainTransformation(
            direction=1, gamma_min=-0.1, gamma_max=0.1, steps=5
        )
        results = list(transform.apply_transformation(simple_structure))

        original_volume = simple_structure.cellvolume()

        for result in results:
            volume_error = abs(result.cellvolume() - original_volume)
            assert volume_error < 1e-10

    def test_non_pbc_raises_error(self, non_pbc_structure):
        """Test that non-periodic structure raises error."""
        transform = MonoclinicStrainTransformation(
            direction=1, gamma_min=-0.1, gamma_max=0.1, steps=5
        )
        with pytest.raises(ValueError, match="requires periodic"):
            transform.apply_transformation(non_pbc_structure)

    def test_preserves_fractional_coordinates_and_clears_labels(
        self, skewed_structure_with_labels
    ):
        """Monoclinic strain should preserve fractions and clear labels."""
        transform = MonoclinicStrainTransformation(
            direction=2,
            gamma_min=0.1,
            gamma_max=0.2,
            steps=2,
        )
        frac_before = skewed_structure_with_labels.cart2frac(
            skewed_structure_with_labels.coords[-1]
        )

        result = next(iter(transform.apply_transformation(
            skewed_structure_with_labels
        )))

        np.testing.assert_allclose(
            result.cart2frac(result.coords[-1]),
            frac_before,
        )
        assert not np.allclose(
            result.coords[-1],
            skewed_structure_with_labels.coords[-1],
        )
        assert result.energy[-1] is None
        assert result.forces[-1] is None


# Tests for CellTransformationMatrix
class TestCellTransformationMatrix:
    """Test suite for CellTransformationMatrix."""

    def test_initialization(self):
        """Test basic initialization."""
        T = np.eye(3) * 2
        transform = CellTransformationMatrix(T)
        np.testing.assert_array_equal(transform.T, T)

    def test_invalid_matrix_shape(self):
        """Test that invalid matrix shape raises error."""
        with pytest.raises(ValueError, match="must be shape"):
            CellTransformationMatrix(np.eye(2))

    def test_identity_transformation(self, simple_structure):
        """Test identity transformation returns equivalent structure."""
        T = np.eye(3)
        transform = CellTransformationMatrix(T)
        results = list(transform.apply_transformation(simple_structure))

        assert len(results) == 1
        result = results[0]

        assert result.natoms == simple_structure.natoms
        np.testing.assert_allclose(result.avec[-1], simple_structure.avec[-1])

    def test_supercell_2x2x2(self, simple_structure):
        """Test 2x2x2 supercell transformation."""
        T = np.diag([2, 2, 2])
        transform = CellTransformationMatrix(T)
        results = list(transform.apply_transformation(simple_structure))

        assert len(results) == 1
        result = results[0]

        assert result.natoms == simple_structure.natoms * 8

        original_volume = simple_structure.cellvolume()
        new_volume = result.cellvolume()
        assert np.isclose(new_volume, original_volume * 8)

    def test_non_pbc_raises_error(self, non_pbc_structure):
        """Test that non-periodic structure raises error."""
        T = np.eye(3) * 2
        transform = CellTransformationMatrix(T)
        with pytest.raises(ValueError, match="requires.*periodic"):
            transform.apply_transformation(non_pbc_structure)
