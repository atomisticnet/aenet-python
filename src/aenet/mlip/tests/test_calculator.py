"""
Unit tests for AenetCalculator (ASE interface).

Tests the ASE Calculator interface and verifies equivalence with
the direct LibAenetInterface.

"""

import os
import unittest
from unittest.mock import patch

import numpy as np

from aenet.mlip import AenetCalculator, AenetEnsembleCalculator, libaenet

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


@unittest.skipIf(not ASE_AVAILABLE, "ASE not available")
class TestAenetEnsembleCalculator(unittest.TestCase):
    """Test cases for AenetEnsembleCalculator."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test data paths."""
        test_dir = os.path.dirname(__file__)
        cls.data_dir = os.path.join(test_dir, '../../tests/data')
        cls.potential_paths = {
            'Ti': os.path.join(cls.data_dir, 'Ti.nn'),
            'O': os.path.join(cls.data_dir, 'O.nn')
        }
        cls.has_potentials = all(
            os.path.exists(path) for path in cls.potential_paths.values()
        )
        cls.xsf_dir = os.path.join(cls.data_dir, 'xsf-TiO2')
        cls.has_structures = os.path.exists(cls.xsf_dir)

    def setUp(self):
        """Set up each test."""
        if not self.has_potentials:
            self.skipTest("Test potentials not available")
        if not self.has_structures:
            self.skipTest("Test structures not available")

    def tearDown(self):
        """Clean up after each test."""
        libaenet.cleanup_sessions()

    def test_single_member_matches_aenet_calculator(self):
        """Single-member ensemble should match AenetCalculator."""
        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        atoms_single = ase.io.read(xsf_file)
        atoms_ensemble = ase.io.read(xsf_file)

        calc_single = AenetCalculator(self.potential_paths)
        calc_ensemble = AenetEnsembleCalculator([self.potential_paths])
        atoms_single.calc = calc_single
        atoms_ensemble.calc = calc_ensemble

        energy_single = atoms_single.get_potential_energy()
        forces_single = atoms_single.get_forces()
        energy_ensemble = atoms_ensemble.get_potential_energy()
        forces_ensemble = atoms_ensemble.get_forces()

        self.assertAlmostEqual(energy_ensemble, energy_single, places=12)
        self.assertAlmostEqual(
            calc_ensemble.results['energy_std'],
            0.0,
            places=12,
        )
        np.testing.assert_allclose(forces_ensemble, forces_single, atol=1e-12)
        np.testing.assert_allclose(
            calc_ensemble.results['force_uncertainty'],
            np.zeros(len(atoms_ensemble)),
            atol=1e-12,
        )

    def test_duplicate_members_have_zero_uncertainty(self):
        """Duplicate calculator members should produce zero spread."""
        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        atoms = ase.io.read(xsf_file)
        calc = AenetEnsembleCalculator(
            [self.potential_paths, self.potential_paths]
        )
        atoms.calc = calc

        _ = atoms.get_forces()

        self.assertAlmostEqual(calc.results['energy_std'], 0.0, places=12)
        np.testing.assert_allclose(
            calc.results['forces_std'],
            np.zeros_like(calc.results['forces_std']),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            calc.results['force_uncertainty'],
            np.zeros(len(atoms)),
            atol=1e-12,
        )

    def test_reference_aggregation_uses_reference_member(self):
        """Reference aggregation should expose the selected member output."""
        atoms = ase.Atoms(
            'Ti',
            positions=[[0.0, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        calc = AenetEnsembleCalculator(
            members=[{'Ti': 'member-0.nn'}, {'Ti': 'member-1.nn'}],
            aggregation='reference',
            reference_member=1,
        )

        force0 = np.array([[1.0, 2.0, 3.0]])
        force1 = np.array([[4.0, 5.0, 6.0]])
        with patch.object(
            calc,
            '_get_cutoff_radius',
            return_value=(0.0, 3.5),
        ), patch.object(
            calc,
            '_predict_member',
            side_effect=[(1.0, force0), (3.0, force1)],
        ):
            atoms.calc = calc
            reported_forces = atoms.get_forces()

        self.assertAlmostEqual(calc.results['energy'], 3.0, places=12)
        self.assertAlmostEqual(calc.results['energy_mean'], 2.0, places=12)
        self.assertAlmostEqual(calc.results['energy_std'], 1.0, places=12)
        np.testing.assert_allclose(reported_forces, force1, atol=1e-12)
        np.testing.assert_allclose(
            calc.results['forces_mean'],
            np.array([[2.5, 3.5, 4.5]]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            calc.results['force_uncertainty'],
            np.array([1.5]),
            atol=1e-12,
        )

    def test_invalid_reference_member_raises(self):
        """Invalid reference member indices should raise an error."""
        with self.assertRaises(ValueError):
            AenetEnsembleCalculator(
                members=[{'Ti': 'member-0.nn'}],
                aggregation='reference',
                reference_member=2,
            )


if __name__ == '__main__':
    unittest.main()
