"""
Unit tests for LibAenetInterface.

Tests the high-level library interface and compares results with
the subprocess-based ANNPotential.predict() method.

"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aenet.geometry import AtomicStructure
from aenet.mlip import (
    AenetEnsembleInterface,
    ANNPotential,
    LibAenetInterface,
    libaenet,
)


class TestLibAenetInterface(unittest.TestCase):
    """Test cases for LibAenetInterface."""

    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        # Get path to shared test data
        test_dir = os.path.dirname(__file__)
        cls.data_dir = os.path.join(test_dir, '../../tests/data')

        cls.potential_paths = {
            'Ti': os.path.join(cls.data_dir, 'Ti.nn'),
            'O': os.path.join(cls.data_dir, 'O.nn')
        }

        # Check if test data exists
        cls.has_potentials = all(
            os.path.exists(p) for p in cls.potential_paths.values()
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

    def test_initialization(self):
        """Test LibAenetInterface initialization."""
        interface = LibAenetInterface(self.potential_paths)
        self.assertEqual(interface.potential_paths, self.potential_paths)
        self.assertIsNone(interface.potential_format)
        self.assertIsNone(interface._session)

    def test_predict_energy_only(self):
        """Test energy prediction without forces."""
        from aenet.formats.xsf import XSFParser

        interface = LibAenetInterface(self.potential_paths)

        # Load a structure
        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        parser = XSFParser()
        structure = parser.read(xsf_file)

        # Predict energy
        energy = interface.predict(structure, forces=False)

        # Check that we got a float
        self.assertIsInstance(energy, float)
        self.assertFalse(np.isnan(energy))
        self.assertFalse(np.isinf(energy))

    def test_predict_with_forces(self):
        """Test energy and force prediction."""
        from aenet.formats.xsf import XSFParser

        interface = LibAenetInterface(self.potential_paths)

        # Load a structure
        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        parser = XSFParser()
        structure = parser.read(xsf_file)

        # Predict energy and forces
        energy, forces = interface.predict(structure, forces=True)

        # Check energy
        self.assertIsInstance(energy, float)
        self.assertFalse(np.isnan(energy))

        # Check forces
        self.assertIsInstance(forces, np.ndarray)
        self.assertEqual(forces.shape, (structure.natoms, 3))
        self.assertFalse(np.any(np.isnan(forces)))
        self.assertFalse(np.any(np.isinf(forces)))

    @unittest.skipIf(
        not os.path.exists(os.path.join(os.path.dirname(__file__),
                                        '../../tests/data/Ti.nn')),
        "Test data not available"
    )
    def test_compare_with_subprocess(self):
        """
        Critical test: Compare LibAenetInterface with ANNPotential.predict().

        Both should give the same results for the same structure.
        """
        from aenet.formats.xsf import XSFParser

        # Load structure
        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        parser = XSFParser()
        structure = parser.read(xsf_file)

        # Check if predict.x is available
        from aenet import config as cfg
        aenet_paths = cfg.read('aenet')
        if not os.path.exists(aenet_paths.get('predict_x_path', '')):
            self.skipTest("predict.x not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Build both interfaces inside the temp cwd so any transient
                # converted network files stay isolated.
                interface = LibAenetInterface(self.potential_paths)
                lib_energy, lib_forces = interface.predict(
                    structure, forces=True)

                # Build the subprocess-backed potential inside the temp cwd so
                # any transient converted network files stay isolated.
                potential = ANNPotential.from_files(
                    self.potential_paths,
                )
                results = potential.predict(
                    [xsf_file],
                    eval_forces=True,
                )
            finally:
                os.chdir(cwd)

        # Compare total energies (predict.x headline value)
        subprocess_energy = results.total_energy[0]
        subprocess_forces = results.forces[0]

        # Compare energies (should match within numerical precision)
        energy_diff = abs(lib_energy - subprocess_energy)
        self.assertLess(energy_diff, 1e-6,
                        f"Energy difference: {energy_diff}")

        # Compare forces (should match within numerical precision)
        max_force_diff = np.max(np.abs(lib_forces - subprocess_forces))
        self.assertLess(max_force_diff, 1e-6,
                        f"Max force difference: {max_force_diff}")

        print("\nComparison successful!")
        print(f"  Energy difference: {energy_diff:.2e} eV")
        print(f"  Max force difference: {max_force_diff:.2e} eV/Å")

    def test_multiple_structures(self):
        """Test prediction on multiple structures."""
        from aenet.formats.xsf import XSFParser

        interface = LibAenetInterface(self.potential_paths)
        parser = XSFParser()

        # Load multiple structures
        energies = []
        for i in range(1, 4):  # structures 001-003
            xsf_file = os.path.join(self.xsf_dir, f'structure-00{i}.xsf')
            if not os.path.exists(xsf_file):
                continue

            structure = parser.read(xsf_file)
            energy = interface.predict(structure, forces=False)
            energies.append(energy)

        # Check that we got results for all structures
        self.assertGreater(len(energies), 0)

        # Check that energies are reasonable
        for energy in energies:
            self.assertFalse(np.isnan(energy))
            self.assertFalse(np.isinf(energy))

    def test_cleanup(self):
        """Test that session cleanup happens properly."""
        # Create and use interface
        interface = LibAenetInterface(self.potential_paths)

        # Force initialization
        from aenet.formats.xsf import XSFParser
        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        parser = XSFParser()
        structure = parser.read(xsf_file)
        _ = interface.predict(structure, forces=False)

        # Check that session was acquired
        self.assertIsNotNone(interface._session)

        # Delete interface
        del interface
        libaenet.cleanup_sessions()

        # Session manager should have released the session
        self.assertFalse(libaenet._session_manager.is_active())


class TestLibAenetInterfaceErrors(unittest.TestCase):
    """Test error handling in LibAenetInterface."""

    def test_missing_potential_file(self):
        """Test error for missing potential file."""
        potential_paths = {
            'Ti': 'nonexistent_Ti.nn',
            'O': 'nonexistent_O.nn'
        }

        interface = LibAenetInterface(potential_paths)

        # Create a dummy structure
        structure = AtomicStructure(
            coords=np.array([[0, 0, 0], [2, 0, 0]]),
            types=['Ti', 'O'],
            avec=np.eye(3) * 10
        )

        # Session manager raises AenetError (I/O error) when file not found
        from aenet.mlip.libaenet import AenetError
        with self.assertRaises(AenetError):
            interface.predict(structure)

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

        lib = LibAenetInterface(potential_paths)

        # Create structure with atom type not in potentials
        structure = AtomicStructure(
            coords=np.array([[0, 0, 0]]),
            types=['Si'],  # Not in potentials
            avec=np.eye(3) * 10
        )

        # Should raise ValueError
        with self.assertRaises(ValueError):
            lib.predict(structure)


class TestAenetEnsembleInterface(unittest.TestCase):
    """Test cases for AenetEnsembleInterface."""

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

    def test_single_member_matches_lib_interface(self):
        """Single-member ensembles should match LibAenetInterface."""
        from aenet.formats.xsf import XSFParser

        parser = XSFParser()
        structure = parser.read(
            os.path.join(self.xsf_dir, 'structure-001.xsf')
        )

        single = LibAenetInterface(self.potential_paths)
        ensemble = AenetEnsembleInterface([self.potential_paths])

        single_energy, single_forces = single.predict(structure, forces=True)
        result = ensemble.predict(structure, forces=True)

        self.assertAlmostEqual(result.energy, single_energy, places=12)
        self.assertAlmostEqual(result.energy_mean, single_energy, places=12)
        self.assertAlmostEqual(result.energy_std, 0.0, places=12)
        np.testing.assert_allclose(result.forces, single_forces, atol=1e-12)
        np.testing.assert_allclose(
            result.force_uncertainty,
            np.zeros(structure.natoms),
            atol=1e-12,
        )

    def test_duplicate_members_have_zero_uncertainty(self):
        """Duplicate members should produce zero energy/force spread."""
        from aenet.formats.xsf import XSFParser

        parser = XSFParser()
        structure = parser.read(
            os.path.join(self.xsf_dir, 'structure-001.xsf')
        )

        ensemble = AenetEnsembleInterface(
            [self.potential_paths, self.potential_paths]
        )
        result = ensemble.predict(structure, forces=True)

        self.assertAlmostEqual(result.energy_std, 0.0, places=12)
        np.testing.assert_allclose(
            result.forces_std,
            np.zeros_like(result.forces_std),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.force_uncertainty,
            np.zeros(structure.natoms),
            atol=1e-12,
        )

    def test_reference_aggregation_uses_reference_member(self):
        """Reference aggregation should report the selected member output."""
        structure = AtomicStructure(
            coords=np.array([[0.0, 0.0, 0.0]]),
            types=['Ti'],
            avec=np.eye(3) * 10.0,
        )
        ensemble = AenetEnsembleInterface(
            members=[{'Ti': 'member-0.nn'}, {'Ti': 'member-1.nn'}],
            aggregation='reference',
            reference_member=1,
        )

        force0 = np.array([[1.0, 2.0, 3.0]])
        force1 = np.array([[4.0, 5.0, 6.0]])
        with patch.object(
            ensemble,
            '_get_cutoff_radius',
            return_value=(0.0, 3.5),
        ), patch.object(
            ensemble,
            '_predict_member',
            side_effect=[(1.0, force0), (3.0, force1)],
        ):
            result = ensemble.predict(structure, forces=True)

        self.assertAlmostEqual(result.energy, 3.0, places=12)
        self.assertAlmostEqual(result.energy_mean, 2.0, places=12)
        self.assertAlmostEqual(result.energy_std, 1.0, places=12)
        np.testing.assert_allclose(result.forces, force1, atol=1e-12)
        np.testing.assert_allclose(
            result.forces_mean,
            np.array([[2.5, 3.5, 4.5]]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.force_uncertainty,
            np.array([1.5]),
            atol=1e-12,
        )

    def test_invalid_reference_member_raises(self):
        """Invalid reference member indices should raise an error."""
        with self.assertRaises(ValueError):
            AenetEnsembleInterface(
                members=[{'Ti': 'member-0.nn'}],
                aggregation='reference',
                reference_member=2,
            )

    def test_inconsistent_member_atom_types_raise(self):
        """All members must define the same atom types."""
        with self.assertRaises(ValueError):
            AenetEnsembleInterface(
                members=[{'Ti': 'member-0.nn'}, {'O': 'member-1.nn'}]
            )


if __name__ == '__main__':
    unittest.main()
