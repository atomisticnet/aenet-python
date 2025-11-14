"""
Unit tests comparing forces from different prediction methods.

This test suite compares forces predicted by:
1. LibAenetInterface (Python neighbor list + libaenet)
2. AenetCalculator (ASE neighbor list + libaenet)
3. ANNPotential.predict() (command-line reference)

The goal is to ensure all three methods produce identical forces,
particularly for non-periodic (isolated) structures.
"""

import os
import unittest
import numpy as np

from aenet.mlip import (LibAenetInterface, AenetCalculator,
                        ANNPotential, libaenet)
from aenet.geometry import AtomicStructure
from aenet import config as cfg


# Check if ASE is available
try:
    import ase
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


class TestForceComparison(unittest.TestCase):
    """Test force predictions across different methods."""

    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        # Get path to shared test data
        test_dir = os.path.dirname(__file__)
        cls.data_dir = os.path.join(test_dir, '../../tests/data')

        cls.potential_paths = {
            'Ti': os.path.join(cls.data_dir, 'Ti.nn.ascii'),
            'O': os.path.join(cls.data_dir, 'O.nn.ascii')
        }

        # Check if test data exists
        cls.has_potentials = all(
            os.path.exists(p) for p in cls.potential_paths.values()
        )

        cls.xsf_dir = os.path.join(cls.data_dir, 'xsf-TiO2')
        cls.has_structures = os.path.exists(cls.xsf_dir)

        # Check for torch training test data (isolated cluster)
        cls.cluster_file = os.path.join(
            test_dir, '../../torch_training/tests/data/TiO2-cluster-12A.xsf'
        )
        cls.has_cluster = os.path.exists(cls.cluster_file)

        # Check if predict.x is available
        aenet_paths = cfg.read('aenet')
        cls.has_predict_x = os.path.exists(
            aenet_paths.get('predict_x_path', '')
        )

    def setUp(self):
        """Set up each test."""
        if not self.has_potentials:
            self.skipTest("Test potentials not available")

    def tearDown(self):
        """Clean up after each test."""
        libaenet.cleanup_sessions()

    def _compare_forces(self, lib_forces, calc_forces, subprocess_forces=None,
                        atol=1e-6, structure=None):
        """
        Helper to compare forces with detailed diagnostics.

        Parameters
        ----------
        lib_forces : np.ndarray
            Forces from LibAenetInterface
        calc_forces : np.ndarray
            Forces from AenetCalculator
        subprocess_forces : np.ndarray, optional
            Forces from ANNPotential.predict()
        atol : float
            Absolute tolerance for comparison
        structure : AtomicStructure, optional
            Structure for detailed output
        """

        print("\n" + "="*70)
        print("FORCE COMPARISON DIAGNOSTICS")
        print("="*70)

        # Compare LibAenetInterface vs AenetCalculator
        diff_lib_calc = lib_forces - calc_forces
        max_diff_lib_calc = np.max(np.abs(diff_lib_calc))
        rms_diff_lib_calc = np.sqrt(np.mean(diff_lib_calc**2))

        print("\nLibAenetInterface vs AenetCalculator:")
        print(f"  Max absolute difference: {max_diff_lib_calc:.2e} eV/Å")
        print(f"  RMS difference:          {rms_diff_lib_calc:.2e} eV/Å")

        # Find worst atoms
        atom_diff_lib_calc = np.linalg.norm(diff_lib_calc, axis=1)
        worst_atoms = np.argsort(atom_diff_lib_calc)[-5:][::-1]

        print("\n  Top 5 atoms with largest force differences:")
        for idx in worst_atoms:
            if structure:
                atom_type = structure.types[idx]
            else:
                atom_type = "?"
            print(f"    Atom {idx:3d} ({atom_type}): "
                  f"{atom_diff_lib_calc[idx]:.2e} eV/Å")
            print(f"      LibAenet: {lib_forces[idx]}")
            print(f"      ASECalc:  {calc_forces[idx]}")
            print(f"      Diff:     {diff_lib_calc[idx]}")

        # Compare with subprocess if available
        if subprocess_forces is not None:
            diff_lib_sub = lib_forces - subprocess_forces
            diff_calc_sub = calc_forces - subprocess_forces

            max_diff_lib_sub = np.max(np.abs(diff_lib_sub))
            rms_diff_lib_sub = np.sqrt(np.mean(diff_lib_sub**2))

            max_diff_calc_sub = np.max(np.abs(diff_calc_sub))
            rms_diff_calc_sub = np.sqrt(np.mean(diff_calc_sub**2))

            print("\nLibAenetInterface vs ANNPotential (subprocess):")
            print(f"  Max absolute difference: {max_diff_lib_sub:.2e} eV/Å")
            print(f"  RMS difference:          {rms_diff_lib_sub:.2e} eV/Å")

            print("\nAenetCalculator vs ANNPotential (subprocess):")
            print(f"  Max absolute difference: {max_diff_calc_sub:.2e} eV/Å")
            print(f"  RMS difference:          {rms_diff_calc_sub:.2e} eV/Å")

        print("="*70 + "\n")

        # Assertions
        self.assertLess(
            max_diff_lib_calc, atol,
            f"LibAenet vs ASECalc force difference {max_diff_lib_calc:.2e} "
            f"exceeds tolerance {atol:.2e}"
        )

        if subprocess_forces is not None:
            self.assertLess(
                max_diff_calc_sub, atol,
                "ASECalc vs subprocess force difference "
                f"{max_diff_calc_sub:.2e} "
                f"exceeds tolerance {atol:.2e}"
            )

    @unittest.skipIf(not ASE_AVAILABLE, "ASE not available")
    def test_periodic_structure_forces(self):
        """Test force agreement for periodic structures."""
        if not self.has_structures:
            self.skipTest("Test structures not available")

        from aenet.formats.xsf import XSFParser

        # Load a periodic structure
        xsf_file = os.path.join(self.xsf_dir, 'structure-001.xsf')
        parser = XSFParser()
        structure = parser.read(xsf_file)

        print(f"\n{'='*70}")
        print("Testing periodic structure: structure-001.xsf")
        print(f"  natoms: {structure.natoms}")
        print(f"  PBC: {structure.pbc}")
        print(f"{'='*70}")

        # Method 1: LibAenetInterface
        interface = LibAenetInterface(
            self.potential_paths, potential_format='ascii'
        )
        lib_energy, lib_forces = interface.predict(structure, forces=True)

        # Method 2: AenetCalculator (ASE)
        atoms = ase.io.read(xsf_file)
        calc = AenetCalculator(
            self.potential_paths, potential_format='ascii'
        )
        atoms.calc = calc
        calc_energy = atoms.get_potential_energy()
        calc_forces = atoms.get_forces()

        # Method 3: Subprocess (if available)
        subprocess_forces = None
        if self.has_predict_x:
            potential = ANNPotential.from_files(
                self.potential_paths, potential_format='ascii'
            )
            results = potential.predict([structure], eval_forces=True)
            subprocess_energy = results.total_energy[0]
            subprocess_forces = results.forces[0]

        # Compare forces
        self._compare_forces(
            lib_forces, calc_forces, subprocess_forces,
            atol=1e-6, structure=structure
        )

        # Also check energies
        energy_diff = abs(lib_energy - calc_energy)
        print("Energy comparison:")
        print(f"  LibAenet:  {lib_energy:.10f} eV")
        print(f"  ASECalc:   {calc_energy:.10f} eV")
        print(f"  Difference: {energy_diff:.2e} eV")
        if self.has_predict_x:
            print(f"  Subprocess: {subprocess_energy:.10f} eV")

        self.assertLess(energy_diff, 1e-6,
                        f"Energy difference {energy_diff:.2e} exceeds 1e-6")

    @unittest.skipIf(not ASE_AVAILABLE, "ASE not available")
    def test_isolated_cluster_forces(self):
        """
        Test force agreement for isolated (non-periodic) structures.

        This is the critical test that should reveal the neighbor list bug.
        """
        if not self.has_cluster:
            self.skipTest("Test cluster not available")

        from aenet.formats.xsf import XSFParser

        # Load the isolated cluster
        parser = XSFParser()
        structure = parser.read(self.cluster_file)

        print(f"\n{'='*70}")
        print("Testing isolated cluster: TiO2-cluster-12A.xsf")
        print(f"  natoms: {structure.natoms}")
        print(f"  PBC: {structure.pbc}")
        print(f"  Types: {set(structure.types)}")
        print(f"{'='*70}")

        # Method 1: LibAenetInterface (Python neighbor list)
        interface = LibAenetInterface(
            self.potential_paths, potential_format='ascii'
        )
        lib_energy, lib_forces = interface.predict(structure, forces=True)

        # Method 2: AenetCalculator (ASE neighbor list)
        atoms = ase.io.read(self.cluster_file)
        calc = AenetCalculator(
            self.potential_paths, potential_format='ascii'
        )
        atoms.calc = calc
        calc_energy = atoms.get_potential_energy()
        calc_forces = atoms.get_forces()

        # Method 3: Subprocess (if available)
        subprocess_forces = None
        if self.has_predict_x:
            potential = ANNPotential.from_files(
                self.potential_paths, potential_format='ascii'
            )
            results = potential.predict([self.cluster_file], eval_forces=True)
            subprocess_energy = results.total_energy[0]
            subprocess_forces = results.forces[0]

        # Compare forces with detailed diagnostics
        self._compare_forces(
            lib_forces, calc_forces, subprocess_forces,
            atol=1e-6, structure=structure
        )

        # Also check energies
        energy_diff = abs(lib_energy - calc_energy)
        print("\nEnergy comparison:")
        print(f"  LibAenet:  {lib_energy:.10f} eV")
        print(f"  ASECalc:   {calc_energy:.10f} eV")
        print(f"  Difference: {energy_diff:.2e} eV")
        if self.has_predict_x:
            print(f"  Subprocess: {subprocess_energy:.10f} eV")
            print("  Lib-Sub diff: "
                  f"{abs(lib_energy - subprocess_energy):.2e} eV")

        self.assertLess(energy_diff, 1e-6,
                        f"Energy difference {energy_diff:.2e} exceeds 1e-6")

    @unittest.skipIf(not ASE_AVAILABLE, "ASE not available")
    def test_small_isolated_molecule(self):
        """Test on a small isolated molecule for easier debugging."""
        # Create a simple 3-atom molecule (water-like geometry)
        coords = np.array([
            [0.0, 0.0, 0.0],     # Ti
            [2.0, 0.0, 0.0],     # O
            [0.0, 2.0, 0.0],     # O
        ])

        structure = AtomicStructure(
            coords=coords,
            types=['Ti', 'O', 'O'],
            avec=np.eye(3) * 10  # Large box
        )
        structure.pbc = False  # Explicitly non-periodic

        print(f"\n{'='*70}")
        print("Testing small 3-atom molecule")
        print(f"  natoms: {structure.natoms}")
        print(f"  PBC: {structure.pbc}")
        print(f"{'='*70}")

        # Method 1: LibAenetInterface
        interface = LibAenetInterface(
            self.potential_paths, potential_format='ascii'
        )
        lib_energy, lib_forces = interface.predict(structure, forces=True)

        # Method 2: AenetCalculator
        atoms = ase.Atoms(
            symbols=structure.types,
            positions=coords,
            cell=structure.avec[-1],
            pbc=False
        )
        calc = AenetCalculator(
            self.potential_paths, potential_format='ascii'
        )
        atoms.calc = calc
        calc_energy = atoms.get_potential_energy()
        calc_forces = atoms.get_forces()

        # Compare
        self._compare_forces(
            lib_forces, calc_forces, None,
            atol=1e-6, structure=structure
        )

        # Check energies
        energy_diff = abs(lib_energy - calc_energy)
        print("\nEnergy comparison:")
        print(f"  LibAenet: {lib_energy:.10f} eV")
        print(f"  ASECalc:  {calc_energy:.10f} eV")
        print(f"  Difference: {energy_diff:.2e} eV")

        self.assertLess(energy_diff, 1e-6)


if __name__ == '__main__':
    unittest.main()
