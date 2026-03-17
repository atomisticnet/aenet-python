"""
Unit tests for AenetCalculator (ASE interface).

Tests the ASE Calculator interface and verifies equivalence with
the direct LibAenetInterface.

"""

import os
import unittest

import numpy as np

from aenet.mlip import AenetCalculator, libaenet

# Check if ASE is available
try:
    import ase
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


@unittest.skipIf(not ASE_AVAILABLE, "ASE not available")
class TestAenetCalculator(unittest.TestCase):
    """Test cases for AenetCalculator (ASE neighbor list version)."""

    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        test_dir = os.path.dirname(__file__)
        cls.data_dir = os.path.join(test_dir, '../../tests/data')

        cls.potential_paths = {
            'Ti': os.path.join(cls.data_dir, 'Ti.nn'),
            'O': os.path.join(cls.data_dir, 'O.nn')
        }

        cls.has_potentials = all(
            os.path.exists(p) for p in cls.potential_paths.values()
        )

        cls.xsf_dir = os.path.join(cls.data_dir, 'xsf-TiO2')
        cls.has_structures = os.path.exists(cls.xsf_dir)

    def setUp(self):
        """Set up each test."""
        if not self.has_potentials:
            self.skipTest("Test potentials not available")

    def tearDown(self):
        """Clean up after each test."""
        libaenet.cleanup_sessions()

    def test_calculator_initialization(self):
        """Test AenetCalculator initialization."""
        calc = AenetCalculator(self.potential_paths)
        self.assertEqual(calc.potential_paths, self.potential_paths)
        self.assertIsNone(calc.potential_format)
        self.assertIsNone(calc._session)

    def test_energy_calculation_periodic(self):
        """Test energy calculation for periodic structure."""
        if not self.has_structures:
            self.skipTest("Test structures not available")

        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        atoms = ase.io.read(xsf_file)

        calc = AenetCalculator(self.potential_paths)
        atoms.calc = calc

        energy = atoms.get_potential_energy()

        self.assertIsInstance(energy, float)
        self.assertFalse(np.isnan(energy))
        self.assertFalse(np.isinf(energy))

    def test_forces_calculation_periodic(self):
        """Test force calculation for periodic structure."""
        if not self.has_structures:
            self.skipTest("Test structures not available")

        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        atoms = ase.io.read(xsf_file)

        calc = AenetCalculator(self.potential_paths)
        atoms.calc = calc

        forces = atoms.get_forces()

        self.assertIsInstance(forces, np.ndarray)
        self.assertEqual(forces.shape, (len(atoms), 3))
        self.assertFalse(np.any(np.isnan(forces)))
        self.assertFalse(np.any(np.isinf(forces)))

    def test_energy_recalculation_on_position_change(self):
        """Test that energy is recalculated when positions change."""
        if not self.has_structures:
            self.skipTest("Test structures not available")

        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        atoms = ase.io.read(xsf_file)

        calc = AenetCalculator(self.potential_paths)
        atoms.calc = calc

        # Get initial energy
        energy1 = atoms.get_potential_energy()

        # Modify positions
        original_pos = atoms.positions[0].copy()
        atoms.positions[0] += np.array([0.1, 0.0, 0.0])

        # Get energy after modification
        energy2 = atoms.get_potential_energy()

        # Energy should be different
        self.assertNotEqual(
            energy1, energy2,
            "Energy should change when positions change"
        )

        # Restore original positions
        atoms.positions[0] = original_pos

        # Get energy after restoration
        energy3 = atoms.get_potential_energy()

        # Should be close to original energy (within numerical precision)
        self.assertAlmostEqual(
            energy1, energy3, places=10,
            msg="Energy should return to original value when positions "
                "are restored"
        )

    def test_forces_recalculation_on_position_change(self):
        """Test that forces are recalculated when positions change."""
        if not self.has_structures:
            self.skipTest("Test structures not available")

        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        atoms = ase.io.read(xsf_file)

        calc = AenetCalculator(self.potential_paths)
        atoms.calc = calc

        # Get initial forces
        forces1 = atoms.get_forces()

        # Modify positions
        original_pos = atoms.positions[0].copy()
        atoms.positions[0] += np.array([0.1, 0.0, 0.0])

        # Get forces after modification
        forces2 = atoms.get_forces()

        # Forces should be different
        self.assertFalse(
            np.allclose(forces1, forces2),
            "Forces should change when positions change"
        )

        # Restore original positions
        atoms.positions[0] = original_pos

        # Get forces after restoration
        forces3 = atoms.get_forces()

        # Should be close to original forces (within numerical precision)
        np.testing.assert_array_almost_equal(
            forces1, forces3, decimal=10,
            err_msg="Forces should return to original values when "
                    "positions are restored"
        )

    def test_energy_caching(self):
        """Test that energy is cached when nothing changes."""
        if not self.has_structures:
            self.skipTest("Test structures not available")

        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        atoms = ase.io.read(xsf_file)

        calc = AenetCalculator(self.potential_paths)
        atoms.calc = calc

        # Calculate energy twice without changing anything
        energy1 = atoms.get_potential_energy()
        energy2 = atoms.get_potential_energy()

        # Should be exactly the same (not just close)
        self.assertEqual(
            energy1, energy2,
            "Cached energy should be returned when structure unchanged"
        )


@unittest.skipIf(not ASE_AVAILABLE, "ASE not available")
class TestAenetCalculatorErrors(unittest.TestCase):
    """Test error handling in AenetCalculator."""

    def tearDown(self):
        """Clean up after each test."""
        libaenet.cleanup_sessions()

    def test_missing_potential_file(self):
        """Test error for missing potential file."""
        potential_paths = {
            'Ti': 'nonexistent_Ti.nn',
            'O': 'nonexistent_O.nn'
        }

        calc = AenetCalculator(potential_paths)

        atoms = ase.Atoms(
            'TiO',
            positions=[[0, 0, 0], [2, 0, 0]],
            cell=[10, 10, 10],
            pbc=True
        )
        atoms.calc = calc

        # Session manager raises AenetError (I/O error) when file not found
        from aenet.mlip.libaenet import AenetError
        with self.assertRaises(AenetError):
            atoms.get_potential_energy()

    def test_unknown_atom_type(self):
        """Test error for unknown atom type in structure."""
        test_dir = os.path.dirname(__file__)
        data_dir = os.path.join(test_dir, '../../tests/data')

        potential_paths = {
            'Ti': os.path.join(data_dir, 'Ti.nn'),
            'O': os.path.join(data_dir, 'O.nn')
        }

        if not all(os.path.exists(p) for p in potential_paths.values()):
            self.skipTest("Test potentials not available")

        calc = AenetCalculator(potential_paths)

        atoms = ase.Atoms(
            'Si',
            positions=[[0, 0, 0]],
            cell=[10, 10, 10],
            pbc=True
        )
        atoms.calc = calc

        with self.assertRaises(ValueError):
            atoms.get_potential_energy()


if __name__ == '__main__':
    unittest.main()
