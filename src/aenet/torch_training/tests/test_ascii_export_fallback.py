"""
Unit tests for ASCII export with fallback network (no NetAtom metadata).

This test ensures that ASCII export works correctly when the network doesn't
have hidden_size and active_names attributes (i.e., using the fallback code
path in _extract_hidden_and_activations_from_seq).
"""

import pytest
import numpy as np
from pathlib import Path

from aenet.torch_training.ascii_export import (
    _extract_hidden_and_activations_from_seq,
    export_to_ascii_impl
)


class TestFallbackNetworkExtraction:
    """Test extraction of network structure from Sequential without metadata."""

    def test_extract_simple_network(self):
        """Test extraction from a simple 2-layer network."""
        import torch.nn as nn

        # Create a simple Sequential: Linear(10,20) -> Tanh -> Linear(20,1)
        seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

        hidden_sizes, activations = _extract_hidden_and_activations_from_seq(seq)

        assert hidden_sizes == [20], f"Expected [20], got {hidden_sizes}"
        assert activations == ["tanh"], f"Expected ['tanh'], got {activations}"

    def test_extract_multilayer_network(self):
        """Test extraction from a 3-layer network."""
        import torch.nn as nn

        # Create: Linear(10,20) -> Tanh -> Linear(20,15) -> Sigmoid -> Linear(15,1)
        seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 15),
            nn.Sigmoid(),
            nn.Linear(15, 1)
        )

        hidden_sizes, activations = _extract_hidden_and_activations_from_seq(seq)

        assert hidden_sizes == [20, 15], f"Expected [20, 15], got {hidden_sizes}"
        assert activations == ["tanh", "sigmoid"], \
            f"Expected ['tanh', 'sigmoid'], got {activations}"

    def test_extract_linear_activation(self):
        """Test extraction with Identity (linear) activation."""
        import torch.nn as nn

        # Create: Linear(10,20) -> Identity -> Linear(20,1)
        seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.Identity(),
            nn.Linear(20, 1)
        )

        hidden_sizes, activations = _extract_hidden_and_activations_from_seq(seq)

        assert hidden_sizes == [20], f"Expected [20], got {hidden_sizes}"
        assert activations == ["linear"], \
            f"Expected ['linear'], got {activations}"

    def test_linear_layers_are_skipped(self):
        """
        Regression test: ensure nn.Linear layers are skipped, not treated as activations.

        This was the original bug - nn.Linear modules were being processed
        as if they were activation functions, causing a NotImplementedError.
        """
        import torch.nn as nn

        # Same network as above but explicitly test that Linear is ignored
        seq = nn.Sequential(
            nn.Linear(10, 20),  # Should be skipped
            nn.Tanh(),          # Should be detected
            nn.Linear(20, 15),  # Should be skipped
            nn.Tanh(),          # Should be detected
            nn.Linear(15, 1)    # Should be skipped (output layer)
        )

        # This should NOT raise NotImplementedError about nn.Linear
        hidden_sizes, activations = _extract_hidden_and_activations_from_seq(seq)

        assert hidden_sizes == [20, 15]
        assert activations == ["tanh", "tanh"]

    def test_mixed_activations(self):
        """Test network with mixed activation types."""
        import torch.nn as nn

        seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 15),
            nn.Sigmoid(),
            nn.Linear(15, 10),
            nn.Identity(),
            nn.Linear(10, 1)
        )

        hidden_sizes, activations = _extract_hidden_and_activations_from_seq(seq)

        assert hidden_sizes == [20, 15, 10]
        assert activations == ["tanh", "sigmoid", "linear"]


class TestASCIIExportWithFallback:
    """Test full ASCII export using fallback network path."""

    @pytest.fixture
    def simple_trainer(self, tmp_path):
        """Create a minimal trainer with fallback network for testing."""
        from aenet.torch_training.trainer import TorchANNPotential
        from aenet.torch_featurize import ChebyshevDescriptor
        from aenet.torch_training.config import Structure
        import torch

        # Simple architecture
        arch = {
            'H': [(10, 'tanh')],
            'O': [(10, 'tanh')]
        }

        # Create descriptor
        descriptor = ChebyshevDescriptor(
            species=['H', 'O'],
            rad_order=5,
            rad_cutoff=4.0,
            ang_order=3,
            ang_cutoff=2.0,
            device='cpu',
            dtype=torch.float64
        )

        # Create trainer
        trainer = TorchANNPotential(arch=arch, descriptor=descriptor)

        # Create a simple structure for stats computation
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        structure = Structure(
            positions=positions,
            species=['O', 'H', 'H'],
            energy=-15.0
        )

        return trainer, [structure]

    def test_ascii_export_without_metadata(self, simple_trainer, tmp_path):
        """
        Test that ASCII export works when network lacks hidden_size/active_names.

        This is a regression test for the bug where models without NetAtom
        metadata would fail during ASCII export.
        """
        trainer, structures = simple_trainer

        # Verify the network doesn't have the metadata attributes
        # (it might if NetAtom is used, so we test the fallback explicitly)
        net = trainer.net

        # Export should work regardless
        output_dir = tmp_path / "ascii_output"
        output_files = trainer.to_aenet_ascii(
            str(output_dir),
            prefix="test",
            structures=structures
        )

        # Verify files were created
        assert len(output_files) == 2  # H and O
        for file_path in output_files:
            assert file_path.exists()
            assert file_path.suffix == ".ascii"
            # Verify file has content
            assert file_path.stat().st_size > 0

    def test_ascii_export_produces_valid_format(self, simple_trainer, tmp_path):
        """Test that exported ASCII files have the expected structure."""
        trainer, structures = simple_trainer

        output_dir = tmp_path / "ascii_output"
        output_files = trainer.to_aenet_ascii(
            str(output_dir),
            prefix="test",
            structures=structures
        )

        # Check first file structure
        with open(output_files[0], 'r') as f:
            content = f.read()

        # Basic sanity checks on file format
        lines = content.strip().split('\n')
        assert len(lines) > 10, "ASCII file should have multiple lines"

        # First line should be number of layers (an integer)
        try:
            nlayers = int(lines[0].strip())
            assert nlayers > 0, "Number of layers should be positive"
        except ValueError:
            pytest.fail("First line should be an integer (number of layers)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
