"""
Cross-validation test comparing PyTorch and Fortran inference results.

This test validates that TorchANNPotential predictions match those from
the Fortran ANNPotential when using ASCII-exported models.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from aenet import config
from aenet.mlip import ANNPotential, PredictionConfig


# Path to test data
TEST_DATA_DIR = Path(__file__).parent / "data"
TIO2_MODEL_PATH = TEST_DATA_DIR / "Ti-O.pt"
TIO2_CLUSTER_PATH = TEST_DATA_DIR / "TiO2-cluster-12A.xsf"
TIO2_CELL_PATH = TEST_DATA_DIR / "TiO2-cell.xsf"


# Check if Fortran tools are available
def fortran_tools_available():
    """Check if aenet predict executable is configured."""
    try:
        aenet_paths = config.read('aenet')
        predict_x_exists = os.path.exists(
            aenet_paths.get('predict_x_path', '')
        )
        return predict_x_exists
    except Exception:
        return False


# Skip tests if Fortran not available
pytestmark = pytest.mark.skipif(
    not fortran_tools_available(),
    reason="Fortran aenet-predict executable not available"
)


class TestPyTorchFortranInferenceEquivalence:
    """Test that PyTorch and Fortran inference produce equivalent results."""

    def test_tio2_cluster_inference_equivalence(self, tmp_path, monkeypatch):
        """
        Test isolated TiO2 cluster: PyTorch vs Fortran predictions.

        This test:
        1. Loads PyTorch model from Ti-O.pt
        2. Predicts energy and forces with PyTorch
        3. Exports model to ASCII format
        4. Loads ASCII model with Fortran ANNPotential
        5. Predicts with Fortran
        6. Compares results for numerical equivalence
        """
        # Import here to avoid errors when torch not available
        from aenet.torch_training.trainer import TorchANNPotential
        import aenet.io.structure

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Load structure
        struc = aenet.io.structure.read(str(TIO2_CLUSTER_PATH))
        torch_struc = struc.to_TorchStructure()[0]

        # Load PyTorch model
        torch_pot = TorchANNPotential.from_file(str(TIO2_MODEL_PATH))

        # Predict with PyTorch
        torch_results = torch_pot.predict([torch_struc], eval_forces=True)

        # Export to ASCII format
        ascii_dir = tmp_path / "ascii_models"
        ascii_dir.mkdir()
        torch_pot.to_aenet_ascii(str(ascii_dir), prefix="potential")

        # Verify ASCII files were created
        ti_ascii = ascii_dir / "potential.Ti.nn.ascii"
        o_ascii = ascii_dir / "potential.O.nn.ascii"
        assert ti_ascii.exists(), "Ti ASCII file not created"
        assert o_ascii.exists(), "O ASCII file not created"

        # Load Fortran ANNPotential
        fortran_pot = ANNPotential.from_files(
            {
                'Ti': str(ti_ascii),
                'O': str(o_ascii)
            },
            potential_format='ascii'
        )

        # Predict with Fortran
        fortran_workdir = tmp_path / "fortran_predict"
        fortran_workdir.mkdir()
        fortran_results = fortran_pot.predict(
            [str(TIO2_CLUSTER_PATH)],
            config=PredictionConfig(print_atomic_energies=True),
            eval_forces=True,
            workdir=str(fortran_workdir)
        )

        # Compare energies
        torch_energy = torch_results.total_energy[0]
        fortran_energy = fortran_results.total_energy[0]

        print(f"\nIsolated TiO2 cluster comparison:")
        print(f"  PyTorch energy:  {torch_energy:.8f} eV")
        print(f"  Fortran energy:  {fortran_energy:.8f} eV")
        print(f"  Difference:      {abs(torch_energy - fortran_energy):.2e} eV")

        np.testing.assert_allclose(
            torch_energy,
            fortran_energy,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Total energies differ between PyTorch and Fortran"
        )

        # Compare forces (shape: [n_atoms, 3])
        torch_forces = torch_results.forces[0]  # numpy array
        fortran_forces = fortran_results.forces[0]  # numpy array

        assert torch_forces.shape == fortran_forces.shape, \
            f"Force shapes differ: {torch_forces.shape} vs {fortran_forces.shape}"

        # Check forces per atom
        n_atoms = torch_forces.shape[0]
        for i in range(n_atoms):
            torch_f = torch_forces[i]
            fortran_f = fortran_forces[i]
            diff = np.linalg.norm(torch_f - fortran_f)
            print(f"  Atom {i} force diff: {diff:.2e} eV/Å")

        np.testing.assert_allclose(
            torch_forces,
            fortran_forces,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Forces differ between PyTorch and Fortran"
        )

        print("  ✓ PyTorch and Fortran predictions match!")

    def test_tio2_cell_inference_equivalence(self, tmp_path, monkeypatch):
        """
        Test periodic TiO2 cell: PyTorch vs Fortran predictions.
        """
        # Import here to avoid errors when torch not available
        from aenet.torch_training.trainer import TorchANNPotential
        import aenet.io.structure

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Load structure
        struc = aenet.io.structure.read(str(TIO2_CELL_PATH))
        torch_struc = struc.to_TorchStructure()[0]

        # Load PyTorch model
        torch_pot = TorchANNPotential.from_file(str(TIO2_MODEL_PATH))

        # Predict with PyTorch
        torch_results = torch_pot.predict([torch_struc], eval_forces=True)

        # Export to ASCII format
        ascii_dir = tmp_path / "ascii_models"
        ascii_dir.mkdir()
        torch_pot.to_aenet_ascii(str(ascii_dir), prefix="potential")

        # Verify ASCII files were created
        ti_ascii = ascii_dir / "potential.Ti.nn.ascii"
        o_ascii = ascii_dir / "potential.O.nn.ascii"
        assert ti_ascii.exists(), "Ti ASCII file not created"
        assert o_ascii.exists(), "O ASCII file not created"

        # Load Fortran ANNPotential
        fortran_pot = ANNPotential.from_files(
            {
                'Ti': str(ti_ascii),
                'O': str(o_ascii)
            },
            potential_format='ascii'
        )

        # Predict with Fortran
        fortran_workdir = tmp_path / "fortran_predict"
        fortran_workdir.mkdir()
        fortran_results = fortran_pot.predict(
            [str(TIO2_CELL_PATH)],
            config=PredictionConfig(print_atomic_energies=True),
            eval_forces=True,
            workdir=str(fortran_workdir)
        )

        # Compare energies
        torch_energy = torch_results.total_energy[0]
        fortran_energy = fortran_results.total_energy[0]

        print(f"\nPeriodic TiO2 cell comparison:")
        print(f"  PyTorch energy:  {torch_energy:.8f} eV")
        print(f"  Fortran energy:  {fortran_energy:.8f} eV")
        print(f"  Difference:      {abs(torch_energy - fortran_energy):.2e} eV")

        np.testing.assert_allclose(
            torch_energy,
            fortran_energy,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Total energies differ between PyTorch and Fortran"
        )

        # Compare forces (shape: [n_atoms, 3])
        torch_forces = torch_results.forces[0]  # numpy array
        fortran_forces = fortran_results.forces[0]  # numpy array

        assert torch_forces.shape == fortran_forces.shape, \
            f"Force shapes differ: {torch_forces.shape} vs {fortran_forces.shape}"

        # Check forces per atom
        n_atoms = torch_forces.shape[0]
        for i in range(n_atoms):
            torch_f = torch_forces[i]
            fortran_f = fortran_forces[i]
            diff = np.linalg.norm(torch_f - fortran_f)
            print(f"  Atom {i} force diff: {diff:.2e} eV/Å")

        np.testing.assert_allclose(
            torch_forces,
            fortran_forces,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Forces differ between PyTorch and Fortran"
        )

        print("  ✓ PyTorch and Fortran predictions match!")


    def test_fallback_network_fortran_equivalence(self, tmp_path, monkeypatch):
        """
        Test PyTorch-Fortran equivalence using existing trained model.

        This ensures the ASCII export (whether using preferred path with metadata
        or fallback path without) produces numerically equivalent results to
        PyTorch inference.
        """
        # Import here to avoid errors when torch not available
        from aenet.torch_training.trainer import TorchANNPotential
        import aenet.io.structure

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Load structure
        struc = aenet.io.structure.read(str(TIO2_CLUSTER_PATH))
        torch_struc = struc.to_TorchStructure()[0]

        # Load existing trained model (has normalization stats, training history)
        torch_pot = TorchANNPotential.from_file(str(TIO2_MODEL_PATH))

        # Predict with PyTorch
        torch_results = torch_pot.predict([torch_struc], eval_forces=True)

        # Export to ASCII without providing structures
        # This uses stored statistics from training
        ascii_dir = tmp_path / "ascii_models_fallback"
        ascii_dir.mkdir()
        torch_pot.to_aenet_ascii(str(ascii_dir), prefix="potential")

        # Verify ASCII files were created
        ti_ascii = ascii_dir / "potential.Ti.nn.ascii"
        o_ascii = ascii_dir / "potential.O.nn.ascii"
        assert ti_ascii.exists(), "Ti ASCII file not created"
        assert o_ascii.exists(), "O ASCII file not created"

        # Load Fortran ANNPotential
        fortran_pot = ANNPotential.from_files(
            {
                'Ti': str(ti_ascii),
                'O': str(o_ascii)
            },
            potential_format='ascii'
        )

        # Predict with Fortran
        fortran_workdir = tmp_path / "fortran_predict_fallback"
        fortran_workdir.mkdir()
        fortran_results = fortran_pot.predict(
            [str(TIO2_CLUSTER_PATH)],
            config=PredictionConfig(print_atomic_energies=True),
            eval_forces=True,
            workdir=str(fortran_workdir)
        )

        # Compare energies
        torch_energy = torch_results.total_energy[0]
        fortran_energy = fortran_results.total_energy[0]

        print(f"\nFallback network test:")
        print(f"  Network type: {type(torch_pot.net).__name__}")
        print(f"  Has metadata: hidden_size={hasattr(torch_pot.net, 'hidden_size')}, "
              f"active_names={hasattr(torch_pot.net, 'active_names')}")
        print(f"  PyTorch energy:  {torch_energy:.8f} eV")
        print(f"  Fortran energy:  {fortran_energy:.8f} eV")
        print(f"  Difference:      {abs(torch_energy - fortran_energy):.2e} eV")

        np.testing.assert_allclose(
            torch_energy,
            fortran_energy,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Total energies differ between PyTorch and Fortran (fallback network)"
        )

        # Compare forces
        torch_forces = torch_results.forces[0]
        fortran_forces = fortran_results.forces[0]

        assert torch_forces.shape == fortran_forces.shape, \
            f"Force shapes differ: {torch_forces.shape} vs {fortran_forces.shape}"

        # Check a few atoms
        n_atoms = min(5, torch_forces.shape[0])
        for i in range(n_atoms):
            torch_f = torch_forces[i]
            fortran_f = fortran_forces[i]
            diff = np.linalg.norm(torch_f - fortran_f)
            print(f"  Atom {i} force diff: {diff:.2e} eV/Å")

        np.testing.assert_allclose(
            torch_forces,
            fortran_forces,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Forces differ between PyTorch and Fortran (fallback network)"
        )

        print("  ✓ Fallback network produces equivalent results!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
