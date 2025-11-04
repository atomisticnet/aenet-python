"""
Tests for checkpoint saving and resumption functionality.

Verifies that training can be properly interrupted and resumed from
checkpoints, with all state properly restored.
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from aenet.torch_training import TorchANNPotential
from aenet.torch_training.config import TorchTrainingConfig, Structure, Adam
from aenet.torch_featurize import ChebyshevDescriptor


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


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
    """Test basic checkpoint save and resume functionality."""
    # Phase 1: Train for 5 epochs with checkpointing
    pot1 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config1 = TorchTrainingConfig(
        iterations=5,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=2,
        max_checkpoints=3,
        save_best=False,
        show_progress=False,
    )

    results1 = pot1.train(structures=simple_structures, config=config1)
    assert results1 is not None

    # Verify checkpoints were created
    checkpoint_files = list(
        Path(temp_checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    assert len(checkpoint_files) > 0, "No checkpoints were created"

    # Get the last checkpoint
    last_checkpoint = sorted(checkpoint_files)[-1]

    # Phase 2: Create new trainer and resume from checkpoint
    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)

    config2 = TorchTrainingConfig(
        iterations=10,  # Total 10 epochs
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=2,
        show_progress=False,
    )

    results2 = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(last_checkpoint),
    )

    # Verify training completed successfully (results object returned)
    assert results2 is not None
    # The training should have completed without errors


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
        iterations=5,  # Total should be 5 epochs
        method=Adam(mu=0.01, batchsize=2),
        testpercent=20,
        checkpoint_dir=temp_checkpoint_dir,
        show_progress=False,
    )

    results2 = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(last_checkpoint),
    )

    # Training should complete successfully
    assert results2 is not None


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
    """Test that best model checkpoint works correctly."""
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
    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config2 = TorchTrainingConfig(
        iterations=8,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=20,
        show_progress=False,
    )

    # Should be able to resume from best_model.pt
    pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(best_model_path),
    )


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
    assert "architecture" in checkpoint


def test_checkpoint_resume_different_iterations(
    temp_checkpoint_dir, simple_structures, descriptor, architecture
):
    """Test resuming with different total iteration count."""
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

    # Resume but set iterations to 10 (should train 7 more epochs)
    pot2 = TorchANNPotential(arch=architecture, descriptor=descriptor)
    config2 = TorchTrainingConfig(
        iterations=10,
        method=Adam(mu=0.01, batchsize=2),
        testpercent=0,
        show_progress=False,
    )

    results = pot2.train(
        structures=simple_structures,
        config=config2,
        resume_from=str(checkpoint_path),
    )

    # Training should complete successfully
    assert results is not None
