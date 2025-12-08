"""
Comprehensive tests for force training functionality.

This module tests that force training actually works and produces valid
force RMSE values, covering various sampling strategies and edge cases.
"""
import math
from pathlib import Path

import numpy as np
import torch
import pytest

from aenet.torch_training import (
    TorchTrainingConfig,
    Structure,
    TorchANNPotential,
)
from aenet.torch_featurize import ChebyshevDescriptor


def make_structures_with_forces(n_structures=10, n_atoms=5):
    """
    Create a set of simple structures with non-zero forces for testing.

    Parameters
    ----------
    n_structures : int
        Number of structures to generate
    n_atoms : int
        Number of atoms per structure

    Returns
    -------
    list[Structure]
        List of Structure objects with forces
    """
    structures = []
    np.random.seed(42)  # For reproducibility

    for i in range(n_structures):
        # Random positions within a box
        positions = np.random.uniform(-2.0, 2.0, size=(n_atoms, 3))
        species = ["H"] * n_atoms

        # Simple energy (distance-based)
        distances = np.linalg.norm(
            positions[None, :, :] - positions[:, None, :], axis=2)
        # Avoid self-interaction
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        energy = -1.0 / (min_dist + 0.5) + i * 0.1  # Vary by structure

        # Non-zero forces (random but physical-looking)
        forces = np.random.uniform(-0.5, 0.5, size=(n_atoms, 3))

        structures.append(
            Structure(
                positions=positions,
                species=species,
                energy=energy,
                forces=forces,
            )
        )

    return structures


def make_descriptor(dtype=torch.float64):
    """Create a simple descriptor for testing."""
    return ChebyshevDescriptor(
        species=["H"],
        rad_order=2,
        rad_cutoff=3.0,
        ang_order=1,
        ang_cutoff=2.5,
        min_cutoff=0.1,
        device="cpu",
        dtype=dtype,
    )


def make_arch(descriptor: ChebyshevDescriptor):
    """Create a simple architecture for testing."""
    return {
        "H": [(8, "tanh"), (4, "tanh")],
    }


@pytest.mark.cpu
def test_force_training_random_sampling_produces_valid_rmse(tmp_path: Path):
    """
    Test that force training with random sampling produces non-NaN force RMSE.

    This is the primary regression test for the bug where force training
    with force_sampling="random" and force_resample_num_epochs=0 (default)
    was not selecting any force structures, leading to NaN force RMSE.
    """
    structures = make_structures_with_forces(n_structures=20, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=5,
        testpercent=20,
        force_weight=0.5,  # Mixed energy/force training
        force_fraction=0.6,  # Use 60% of structures for forces
        force_sampling="random",  # This was broken!
        force_resample_num_epochs=0,  # Default: no periodic resampling
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpts"),
        checkpoint_interval=5,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Check that force RMSE is present and not NaN
    assert "RMSE_force_train" in result.errors.columns
    assert len(result.errors) == 5  # 5 epochs

    # This is the key assertion: force RMSE should NOT be NaN
    for idx in range(len(result.errors)):
        force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        assert not math.isnan(force_rmse), (
            f"Force RMSE is NaN at epoch {idx}. "
            "This indicates force training is not working!"
        )
        # Force RMSE should be positive
        assert force_rmse > 0, f"Force RMSE is non-positive: {force_rmse}"


@pytest.mark.cpu
def test_force_training_fixed_sampling_produces_valid_rmse(tmp_path: Path):
    """Test that fixed force sampling also produces valid force RMSE."""
    structures = make_structures_with_forces(n_structures=15, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=3,
        testpercent=0,
        force_weight=0.7,  # Heavy force weight
        force_fraction=0.5,
        force_sampling="fixed",  # Fixed subset
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Check force RMSE
    assert "RMSE_force_train" in result.errors.columns
    for idx in range(len(result.errors)):
        force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        assert not math.isnan(force_rmse)
        assert force_rmse > 0


@pytest.mark.cpu
def test_force_training_full_fraction(tmp_path: Path):
    """Test force training with force_fraction=1.0 (all structures)."""
    structures = make_structures_with_forces(n_structures=10, n_atoms=3)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=2,
        testpercent=0,
        force_weight=0.5,
        force_fraction=1.0,  # Use all structures
        force_sampling="random",
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Verify force training is active
    assert "RMSE_force_train" in result.errors.columns
    for idx in range(len(result.errors)):
        force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        assert not math.isnan(force_rmse)
        assert force_rmse > 0


@pytest.mark.cpu
def test_force_training_with_resampling(tmp_path: Path):
    """
    Test force training with periodic resampling.

    This tests that force_resample_num_epochs works correctly.
    """
    structures = make_structures_with_forces(n_structures=20, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=6,
        testpercent=0,
        force_weight=0.5,
        force_fraction=0.5,
        force_sampling="random",
        force_resample_num_epochs=3,  # Resample every 3 epochs
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Verify force training works across all epochs
    assert "RMSE_force_train" in result.errors.columns
    assert len(result.errors) == 6

    for idx in range(len(result.errors)):
        force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        assert not math.isnan(force_rmse), (
            f"Force RMSE is NaN at epoch {idx} with resampling enabled"
        )
        assert force_rmse > 0


@pytest.mark.cpu
def test_force_training_validation_split(tmp_path: Path):
    """
    Test force training with validation split to ensure test force RMSE
    is also computed.
    """
    structures = make_structures_with_forces(n_structures=20, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=3,
        testpercent=25,  # 25% validation
        force_weight=0.6,
        force_fraction=0.7,
        force_sampling="random",
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Check both train and test force RMSE
    assert "RMSE_force_train" in result.errors.columns
    assert "RMSE_force_test" in result.errors.columns

    for idx in range(len(result.errors)):
        train_force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        test_force_rmse = result.errors["RMSE_force_test"].iloc[idx]

        assert not math.isnan(train_force_rmse), "Train force RMSE is NaN"
        assert not math.isnan(test_force_rmse), "Test force RMSE is NaN"
        assert train_force_rmse > 0
        assert test_force_rmse > 0


@pytest.mark.cpu
def test_energy_only_training_has_nan_force_rmse(tmp_path: Path):
    """
    Test that energy-only training (force_weight=0) correctly shows NaN
    force RMSE, confirming the distinction from the bug case.
    """
    structures = make_structures_with_forces(n_structures=10, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=2,
        testpercent=0,
        force_weight=0.0,  # Energy-only
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Energy-only training should have NaN force RMSE
    assert "RMSE_force_train" in result.errors.columns
    for idx in range(len(result.errors)):
        force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        # In energy-only mode, force RMSE should be NaN
        assert math.isnan(force_rmse), (
            "Energy-only training should have NaN force RMSE"
        )


@pytest.mark.cpu
def test_force_training_with_caching(tmp_path: Path):
    """
    Test force training with various caching options enabled.
    """
    structures = make_structures_with_forces(n_structures=15, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=3,
        testpercent=0,
        force_weight=0.5,
        force_fraction=0.6,
        force_sampling="random",
        cache_features=True,
        cache_force_neighbors=True,
        cache_force_triplets=True,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Verify force training works with caching
    assert "RMSE_force_train" in result.errors.columns
    for idx in range(len(result.errors)):
        force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        assert not math.isnan(force_rmse)
        assert force_rmse > 0


@pytest.mark.cpu
def test_force_training_small_fraction(tmp_path: Path):
    """
    Test force training with a small force_fraction (e.g., 0.1).

    This ensures the sampling logic works even when only a few structures
    are selected.
    """
    structures = make_structures_with_forces(n_structures=20, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=3,
        testpercent=0,
        force_weight=0.5,
        force_fraction=0.15,  # Only ~3 structures with 20 total
        force_sampling="random",
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Even with small fraction, should have valid force RMSE
    assert "RMSE_force_train" in result.errors.columns
    for idx in range(len(result.errors)):
        force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        assert not math.isnan(force_rmse)
        assert force_rmse > 0


@pytest.mark.cpu
def test_force_only_training(tmp_path: Path):
    """
    Test pure force training (force_weight=1.0).

    This ensures the force-only training mode works correctly.
    """
    structures = make_structures_with_forces(n_structures=12, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=3,
        testpercent=0,
        force_weight=1.0,  # Force-only
        force_fraction=1.0,
        force_sampling="fixed",
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    # Force-only training should have valid force RMSE
    assert "RMSE_force_train" in result.errors.columns
    for idx in range(len(result.errors)):
        force_rmse = result.errors["RMSE_force_train"].iloc[idx]
        assert not math.isnan(force_rmse)
        assert force_rmse > 0


if __name__ == "__main__":
    # Allow running individual tests
    pytest.main([__file__, "-v"])
