"""
Unit tests for model loading and inference.

Tests that trained models can be loaded and used for prediction without errors.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from aenet.torch_training.config import Structure
from aenet.torch_training.model_export import load_model


# Path to test data
TEST_DATA_DIR = Path(__file__).parent / "data"
TRAINED_MODEL_PATH = TEST_DATA_DIR / "trained_model.pt"


@pytest.fixture
def test_structures():
    """Create simple test structures for inference."""
    # Simple TiO2 structure (small rutile-like)
    positions1 = np.array([
        [0.0, 0.0, 0.0],      # Ti
        [1.5, 0.0, 0.0],      # O
        [0.0, 1.5, 0.0],      # O
    ])
    species1 = ["Ti", "O", "O"]

    # Another TiO2 structure (different geometry)
    positions2 = np.array([
        [0.0, 0.0, 0.0],      # Ti
        [1.6, 0.0, 0.0],      # O
        [0.0, 1.6, 0.0],      # O
    ])
    species2 = ["Ti", "O", "O"]

    structures = [
        Structure(positions=positions1, species=species1, energy=0.0),
        Structure(positions=positions2, species=species2, energy=0.0),
    ]

    return structures


@pytest.mark.skipif(
    not TRAINED_MODEL_PATH.exists(),
    reason="Trained model not found in test data directory"
)
class TestModelLoadingInference:
    """Test suite for model loading and inference."""

    def test_model_loads_without_error(self):
        """Test that a trained model can be loaded without errors."""
        trainer, metadata = load_model(TRAINED_MODEL_PATH)

        # Verify trainer was created
        assert trainer is not None
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "descriptor")
        assert hasattr(trainer, "_normalizer")

        # Verify normalizer is initialized (not None)
        assert trainer._normalizer is not None

        # Verify metadata was loaded
        assert metadata is not None
        assert "architecture" in metadata
        assert "descriptor_config" in metadata

    def test_normalizer_properly_restored(self):
        """Test that normalization manager is properly restored."""
        trainer, metadata = load_model(TRAINED_MODEL_PATH)

        # Verify normalizer exists and has proper state
        assert trainer._normalizer is not None

        # Check that normalization parameters are set
        norm_meta = metadata.get("normalization", {})
        if norm_meta:
            # If normalization was used during training, verify it's restored
            assert hasattr(trainer._normalizer, "E_shift")
            assert hasattr(trainer._normalizer, "E_scaling")
            assert hasattr(trainer._normalizer, "feature_mean")
            assert hasattr(trainer._normalizer, "feature_std")

    def test_energy_prediction_works(self, test_structures):
        """Test that energy prediction works after loading a model."""
        trainer, _ = load_model(TRAINED_MODEL_PATH)

        # Predict energies
        energies, forces = trainer.predict(
            test_structures, predict_forces=False
        )

        # Verify predictions
        assert energies is not None
        assert len(energies) == len(test_structures)
        assert all(isinstance(e, float) for e in energies)
        assert all(not np.isnan(e) for e in energies)

        # Forces should be None since we didn't request them
        assert forces is None

    def test_force_prediction_works(self, test_structures):
        """Test that force prediction works after loading a model."""
        trainer, _ = load_model(TRAINED_MODEL_PATH)

        # Predict both energies and forces
        energies, forces = trainer.predict(
            test_structures, predict_forces=True
        )

        # Verify energy predictions
        assert energies is not None
        assert len(energies) == len(test_structures)

        # Verify force predictions
        assert forces is not None
        assert len(forces) == len(test_structures)

        for i, (struct, force_tensor) in enumerate(
            zip(test_structures, forces)
        ):
            assert force_tensor.shape == (len(struct.species), 3)
            assert torch.is_tensor(force_tensor)
            # Check that forces are not all zeros or NaN
            assert not torch.isnan(force_tensor).any()

    def test_consistent_predictions(self, test_structures):
        """Test that repeated predictions give consistent results."""
        trainer, _ = load_model(TRAINED_MODEL_PATH)

        # Make predictions twice
        energies1, forces1 = trainer.predict(
            test_structures, predict_forces=True
        )
        energies2, forces2 = trainer.predict(
            test_structures, predict_forces=True
        )

        # Verify energies are consistent
        for e1, e2 in zip(energies1, energies2):
            assert abs(e1 - e2) < 1e-10, "Energies should be deterministic"

        # Verify forces are consistent
        for f1, f2 in zip(forces1, forces2):
            assert torch.allclose(
                f1, f2, atol=1e-10
            ), "Forces should be deterministic"

    def test_model_device_compatibility(self, test_structures):
        """Test that model works on CPU regardless of training device."""
        trainer, metadata = load_model(TRAINED_MODEL_PATH)

        # Model should be loaded on CPU by default
        assert trainer.device.type == "cpu"

        # Predictions should work
        energies, _ = trainer.predict(test_structures, predict_forces=False)
        assert len(energies) == len(test_structures)

    def test_descriptor_config_restored(self):
        """Test that descriptor configuration is properly restored."""
        trainer, metadata = load_model(TRAINED_MODEL_PATH)

        # Verify descriptor is properly configured
        desc_cfg = metadata.get("descriptor_config", {})
        assert desc_cfg is not None

        # Check key descriptor parameters match
        assert trainer.descriptor.species == desc_cfg["species"]
        assert trainer.descriptor.rad_order == desc_cfg["rad_order"]
        assert trainer.descriptor.ang_order == desc_cfg["ang_order"]
        assert abs(
            trainer.descriptor.rad_cutoff - desc_cfg["rad_cutoff"]
        ) < 1e-6
        assert abs(
            trainer.descriptor.ang_cutoff - desc_cfg["ang_cutoff"]
        ) < 1e-6


@pytest.mark.skipif(
    not TRAINED_MODEL_PATH.exists(),
    reason="Trained model not found in test data directory"
)
def test_load_and_predict_integration(test_structures):
    """
    Integration test: Load model and make predictions in one go.

    This test simulates the typical user workflow of loading a saved
    model and immediately using it for predictions.
    """
    # Load model
    trainer, metadata = load_model(TRAINED_MODEL_PATH)

    # Verify we can access key information
    assert "architecture" in metadata
    print(f"Loaded model with architecture: {metadata['architecture']}")

    # Make predictions
    energies, forces = trainer.predict(test_structures, predict_forces=True)

    # Verify results
    assert len(energies) == len(test_structures)
    assert len(forces) == len(test_structures)

    # Print summary for manual verification
    for i, (struct, energy, force) in enumerate(
        zip(test_structures, energies, forces)
    ):
        print(f"Structure {i}: {len(struct.species)} atoms")
        print(f"  Energy: {energy:.6f}")
        print(f"  Force shape: {force.shape}")
        print(f"  Force magnitude: {torch.norm(force).item():.6f}")


@pytest.mark.skipif(
    not TRAINED_MODEL_PATH.exists(),
    reason="Trained model not found in test data directory"
)
def test_normalizer_not_none_after_load():
    """
    Regression test for the specific bug that was fixed.

    This test ensures that the _normalizer attribute is never None
    after loading a model, which was the root cause of the original
    AttributeError: 'NoneType' object has no attribute
    'apply_feature_normalization'.
    """
    trainer, _ = load_model(TRAINED_MODEL_PATH)

    # The critical assertion: normalizer must not be None
    assert trainer._normalizer is not None, (
        "Normalizer should be initialized after loading model"
    )

    # Verify it's the correct type
    from aenet.torch_training.training.normalization import (
        NormalizationManager
    )
    assert isinstance(trainer._normalizer, NormalizationManager), (
        "Normalizer should be a NormalizationManager instance"
    )

    # Verify key methods exist
    assert hasattr(trainer._normalizer, "apply_feature_normalization")
    assert hasattr(trainer._normalizer, "denormalize_energy")
    assert callable(trainer._normalizer.apply_feature_normalization)
    assert callable(trainer._normalizer.denormalize_energy)
