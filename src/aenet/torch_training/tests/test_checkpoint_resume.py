"""
Tests for checkpoint saving and resumption functionality.

Verifies that training can be properly interrupted and resumed from
checkpoints, with all state properly restored.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import TorchANNPotential
from aenet.torch_training.config import Adam, Structure, TorchTrainingConfig


def _completed_epochs_from_payload(payload: dict) -> int:
    """Return completed epochs from checkpoint payload metadata."""
    extra_metadata = payload.get("extra_metadata", {}) or {}
    return int(extra_metadata["epoch"]) + 1


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_structures():
    """Create simple test structures."""
    structures = [
        Structure(
            species=["H", "H"],
            positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            energy=-1.0,
        ),
        Structure(
            species=["H", "H"],
            positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.76]]),
            energy=-0.95,
        ),
        Structure(
            species=["H", "H"],
            positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]]),
            energy=-0.85,
        ),
    ]
    return structures


@pytest.fixture
def descriptor():
    """Create a simple descriptor."""
    return ChebyshevDescriptor(
        species=["H"],
        rad_order=4,
        rad_cutoff=3.0,
        ang_order=0,
        ang_cutoff=3.0,
        device="cpu",
        dtype=torch.float64,
    )


@pytest.fixture
def architecture():
    """Simple network architecture."""
    return {"H": [(8, "tanh"), (8, "tanh")]}


def test_checkpoint_resume_basic(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test that resume runs additional epochs from the saved checkpoint."""
    # Phase 1: Train for 5 epochs with checkpointing
    pot1 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config1 = TorchTrainingConfig(
        iterations=5,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        max_checkpoints=3,
        save_best=False,
        show_progress=False,
    )

    results1 = pot1.train(structures=simple_structures, config=config1)
    assert results1 is not None
    assert len(results1.errors) == 5

    # Verify checkpoints were created
    checkpoint_files = list(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    assert len(checkpoint_files) > 0, "No checkpoints were created"

    # Get the last checkpoint
    last_checkpoint = sorted(checkpoint_files)[-1]
    payload = torch.load(
        last_checkpoint, map_location="cpu", weights_only=False
    )
    completed_epochs = _completed_epochs_from_payload(payload)

    # Phase 2: Create new trainer and resume from checkpoint
    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config2 = TorchTrainingConfig(
        iterations=4,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=None,
        checkpoint_interval=0,
        show_progress=False,
    )

    results2 = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(last_checkpoint),
    )

    # Verify training completed successfully (results object returned)
    assert results2 is not None
    assert len(results2.errors) == completed_epochs + 4


def test_checkpoint_resume_preserves_history(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test that training history is preserved across resume."""
    # Phase 1: Train for 3 epochs
    pot1 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config1 = TorchTrainingConfig(
        iterations=3,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=20,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        show_progress=False,
    )

    results1 = pot1.train(structures=simple_structures, config=config1)
    assert results1 is not None

    # Get last checkpoint
    checkpoints = sorted(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    last_checkpoint = checkpoints[-1]

    # Phase 2: Resume and train 2 more epochs
    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config2 = TorchTrainingConfig(
        iterations=2,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=20,
        checkpoint_dir=None,
        checkpoint_interval=0,
        show_progress=False,
    )

    results2 = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(last_checkpoint),
    )

    # Training should complete successfully
    assert results2 is not None
    assert len(results2.errors) == 5
    np.testing.assert_allclose(
        results2.errors["RMSE_train"].iloc[:3].to_numpy(),
        results1.errors["RMSE_train"].to_numpy(),
    )


def test_checkpoint_resume_optimizer_state(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test that optimizer state is properly restored."""
    # Train with a specific learning rate schedule
    pot1 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config1 = TorchTrainingConfig(
        iterations=3,
        method=Adam(mu=0.1, batchsize=2),
        testpercent=0,
        use_scheduler=False,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=2,
        show_progress=False,
    )

    pot1.train(structures=simple_structures, config=config1)

    # Get checkpoint
    checkpoint_path = sorted(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt")
    )[-1]

    # Load checkpoint and verify optimizer state exists
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in checkpoint
    assert len(checkpoint["optimizer_state_dict"]["state"]) > 0


def test_checkpoint_best_model(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test that ``best_model.pt`` resumes from saved checkpoint metadata."""
    pot = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config = TorchTrainingConfig(
        iterations=5,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=20,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        save_best=True,
        show_progress=False,
    )

    pot.train(structures=simple_structures, config=config)

    # Verify best_model.pt was created
    best_model_path = Path(temp_checkpoint_dir) / "best_model.pt"
    assert best_model_path.exists(), "best_model.pt was not created"

    # Verify it can be loaded
    best_payload = torch.load(
        best_model_path, map_location="cpu", weights_only=False
    )
    completed_epochs = _completed_epochs_from_payload(best_payload)

    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config2 = TorchTrainingConfig(
        iterations=2,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=20,
        checkpoint_dir=None,
        checkpoint_interval=0,
        show_progress=False,
    )

    results = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(best_model_path),
    )

    assert results is not None
    assert len(results.errors) == completed_epochs + 2


def test_checkpoint_rotation(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test that old checkpoints are properly rotated/deleted."""
    pot = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config = TorchTrainingConfig(
        iterations=10,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        max_checkpoints=3,
        save_best=False,
        show_progress=False,
    )

    pot.train(structures=simple_structures, config=config)

    # Should only keep 3 most recent checkpoints
    checkpoints = list(Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    assert len(checkpoints
               ) <= 3, f"Expected <=3 checkpoints, found {len(checkpoints)}"

    # Verify the most recent ones are kept
    if len(checkpoints) == 3:
        epochs = [
            int(p.stem.split("_")[-1])
            for p in checkpoints
        ]
        assert max(epochs) >= 8, "Most recent checkpoints should be kept"


def test_checkpoint_metadata_preservation(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test that metadata is preserved in checkpoints."""
    pot = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config = TorchTrainingConfig(
        iterations=3,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        sampling_policy="energy_weighted",
        atomic_energies={"H": -0.5},
        show_progress=False,
    )

    pot.train(structures=simple_structures, config=config)

    # Load checkpoint and verify metadata
    checkpoint_path = sorted(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt")
    )[-1]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)

    assert "atomic_energies" in checkpoint
    assert checkpoint["atomic_energies"]["H"] == -0.5
    assert "normalization" in checkpoint
    assert "training_config" in checkpoint
    assert checkpoint["training_config"]["sampling_policy"] == "energy_weighted"
    assert "architecture" in checkpoint


def test_checkpoint_resume_with_energy_weighted_sampling(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Static energy-weighted sampling should resume without extra state."""
    pot1 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config1 = TorchTrainingConfig(
        iterations=3,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        sampling_policy="energy_weighted",
        atomic_energies={"H": -0.5},
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        show_progress=False,
    )

    pot1.train(structures=simple_structures, config=config1)

    checkpoint_path = sorted(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt")
    )[-1]
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    completed_epochs = _completed_epochs_from_payload(payload)

    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config2 = TorchTrainingConfig(
        iterations=2,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        sampling_policy="energy_weighted",
        atomic_energies={"H": -0.5},
        checkpoint_dir=None,
        checkpoint_interval=0,
        show_progress=False,
    )

    results = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(checkpoint_path),
    )

    assert results is not None
    assert len(results.errors) == completed_epochs + 2


def test_checkpoint_resume_with_error_weighted_sampling(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Adaptive sampling should resume by bootstrapping from uniform again."""
    pot1 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config1 = TorchTrainingConfig(
        iterations=3,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        sampling_policy="error_weighted",
        atomic_energies={"H": -0.5},
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        show_progress=False,
    )

    pot1.train(structures=simple_structures, config=config1)

    checkpoint_path = sorted(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt")
    )[-1]
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    completed_epochs = _completed_epochs_from_payload(payload)

    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config2 = TorchTrainingConfig(
        iterations=2,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        sampling_policy="error_weighted",
        atomic_energies={"H": -0.5},
        checkpoint_dir=None,
        checkpoint_interval=0,
        show_progress=False,
    )

    results = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(checkpoint_path),
    )

    assert results is not None
    assert len(results.errors) == completed_epochs + 2


def test_checkpoint_resume_different_iterations(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test resuming with per-call iteration semantics."""
    # Train for 3 epochs
    pot1 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config1 = TorchTrainingConfig(
        iterations=3,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        show_progress=False,
    )
    pot1.train(structures=simple_structures, config=config1)

    checkpoint_path = sorted(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt")
    )[-1]

    # Resume and run 2 additional epochs
    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config2 = TorchTrainingConfig(
        iterations=2,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=None,
        checkpoint_interval=0,
        show_progress=False,
    )

    results = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(checkpoint_path),
    )

    # Training should complete successfully
    assert results is not None
    assert len(results.errors) == 5


def test_checkpoint_resume_zero_iterations_is_noop(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test that resuming with zero iterations only restores checkpoint state."""
    pot1 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config1 = TorchTrainingConfig(
        iterations=3,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=1,
        show_progress=False,
    )
    results1 = pot1.train(structures=simple_structures, config=config1)

    checkpoint_path = sorted(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt")
    )[-1]

    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config2 = TorchTrainingConfig(
        iterations=0,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=None,
        checkpoint_interval=0,
        show_progress=False,
    )

    results2 = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(checkpoint_path),
    )

    assert len(results2.errors) == 3
    np.testing.assert_allclose(
        results2.errors["RMSE_train"].to_numpy(),
        results1.errors["RMSE_train"].to_numpy(),
    )
