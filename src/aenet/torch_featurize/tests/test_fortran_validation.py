"""
Validation tests against Fortran reference outputs.

These tests compare our PyTorch implementation against features generated
by the original Fortran aenet code to ensure exact numerical agreement.
"""

import pytest
import numpy as np
import os

import aenet.io.structure
from aenet.torch_featurize import ChebyshevDescriptor


class TestFortranValidation:
    """Validate PyTorch implementation against Fortran reference data."""

    def test_water_validation(self):
        """
        Validate water molecule against Fortran output.

        Tests isolated system (no PBC) with 2 species (O, H).
        Reference generated with rad_order=10, ang_order=3.
        """
        # Paths relative to test file location
        import pathlib
        test_dir = pathlib.Path(__file__).parent
        structure_path = test_dir / 'data' / 'water.xyz'
        reference_path = test_dir / 'data' / 'water_ref.csv'

        # Check files exist
        assert os.path.exists(structure_path), \
            f"Structure file not found: {structure_path}"
        assert os.path.exists(reference_path), \
            f"Reference file not found: {reference_path}"

        # Load structure using existing aenet tools
        struc = aenet.io.structure.read(str(structure_path))

        # Load reference features
        ref_features = np.loadtxt(reference_path, delimiter=',')

        # Create our descriptor with same parameters
        descriptor = ChebyshevDescriptor(
            species=struc.typenames,  # ['O', 'H']
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Featurize
        positions = struc.coords[-1]  # Latest coordinates
        species_list = struc.types
        features = descriptor.featurize_structure(positions, species_list)

        # Validate shape
        assert features.shape == ref_features.shape, \
            f"Shape mismatch: {features.shape} vs {ref_features.shape}"

        # Validate values
        max_abs_diff = np.max(np.abs(features - ref_features))
        max_rel_diff = max_abs_diff / (np.max(np.abs(ref_features)) + 1e-20)

        # Detailed per-atom comparison
        for i, sp in enumerate(species_list):
            atom_diff = np.max(np.abs(features[i] - ref_features[i]))
            print(f"Atom {i} ({sp}): max diff = {atom_diff:.2e}")

        print(f"\nWater validation:")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")

        assert np.allclose(features, ref_features, atol=1e-10), \
            f"Water features don't match Fortran! Max diff: {max_abs_diff:.2e}"

        print("✓ Water validation passed!")

    def test_lmnto_validation(self):
        """
        Validate LMNTO periodic structure against Fortran output.

        Tests periodic system (PBC) with 5 species (Li, Mo, Ni, Ti, O).
        Reference generated with rad_order=10, ang_order=3.
        """
        # Paths relative to test file location
        import pathlib
        test_dir = pathlib.Path(__file__).parent
        structure_path = test_dir / 'data' / 'lmnto_structure001.xsf'
        reference_path = test_dir / 'data' / 'lmnto_ref.csv'

        # Check files exist
        assert os.path.exists(structure_path), \
            f"Structure file not found: {structure_path}"
        assert os.path.exists(reference_path), \
            f"Reference file not found: {reference_path}"

        # Load structure
        struc = aenet.io.structure.read(str(structure_path))

        # Load reference features
        ref_features = np.loadtxt(reference_path, delimiter=',')

        # Create descriptor with same parameters as dev.ipynb
        descriptor = ChebyshevDescriptor(
            species=['Li', 'Mo', 'Ni', 'Ti', 'O'],  # Note: Mo not Mn
            rad_order=10,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=1.5
        )

        # Featurize with periodic boundary conditions
        positions = struc.coords[-1]
        species_list = struc.types
        cell = struc.avec[-1] if struc.pbc else None
        pbc = np.array([True, True, True]) if struc.pbc else None

        features = descriptor.featurize_structure(
            positions, species_list, cell, pbc
        )

        # Validate shape
        assert features.shape == ref_features.shape, \
            f"Shape mismatch: {features.shape} vs {ref_features.shape}"

        # Validate values
        max_abs_diff = np.max(np.abs(features - ref_features))
        max_rel_diff = max_abs_diff / (np.max(np.abs(ref_features)) + 1e-20)

        # Sample per-atom comparison (first 5 atoms)
        for i in range(min(5, len(species_list))):
            atom_diff = np.max(np.abs(features[i] - ref_features[i]))
            print(f"Atom {i} ({species_list[i]}): max diff = {atom_diff:.2e}")

        print(f"\nLMNTO validation:")
        print(f"  Structure: {len(species_list)} atoms, periodic")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")

        assert np.allclose(features, ref_features, atol=1e-10), \
            f"LMNTO features don't match Fortran! Max diff: {max_abs_diff:.2e}"

        print("✓ LMNTO validation passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
