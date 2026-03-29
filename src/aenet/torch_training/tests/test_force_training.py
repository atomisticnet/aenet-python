"""
Comprehensive tests for force training functionality.

This module tests that force training actually works and produces valid
force RMSE values, covering various sampling strategies and edge cases.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import aenet.torch_training.training.training_loop as training_loop_module
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    Adam,
    HDF5StructureDataset,
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
)
from aenet.torch_training.dataset import StructureDataset
from aenet.torch_training.sources import RecordSourceCollection, SourceRecord


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


def _source_collection_from_structures(
    source_ids: list[str],
    structures: list[Structure],
) -> RecordSourceCollection:
    """Build a simple record-backed source collection for test structures."""
    records = [
        SourceRecord(
            source_id=source_id,
            loader=(lambda struct=struct: struct),
            source_kind="test",
        )
        for source_id, struct in zip(source_ids, structures, strict=True)
    ]
    return RecordSourceCollection(records)


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
def test_force_training_error_weighted_sampling_produces_valid_rmse():
    """Adaptive error weighting should remain compatible with force training."""
    structures = make_structures_with_forces(n_structures=8, n_atoms=3)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)
    cfg = TorchTrainingConfig(
        iterations=2,
        testpercent=0,
        force_weight=0.5,
        force_fraction=1.0,
        force_sampling="fixed",
        sampling_policy="error_weighted",
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

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
        cache_neighbors=True,
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
def test_force_training_uses_graph_path_by_default(monkeypatch):
    """Default force training should pass graph payloads into the loss path."""
    structures = make_structures_with_forces(n_structures=6, n_atoms=4)
    descriptor = make_descriptor(dtype=torch.float64)
    arch = make_arch(descriptor)
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    calls = []
    original_compute_force_loss = training_loop_module.compute_force_loss

    def _wrapped_compute_force_loss(*args, **kwargs):
        calls.append(
            {
                "graph": kwargs.get("graph"),
                "triplets": kwargs.get("triplets"),
                "neighbor_info": kwargs.get("neighbor_info"),
            }
        )
        return original_compute_force_loss(*args, **kwargs)

    monkeypatch.setattr(
        training_loop_module,
        "compute_force_loss",
        _wrapped_compute_force_loss,
    )

    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        force_fraction=1.0,
        force_sampling="fixed",
        cache_neighbors=False,
        cache_force_triplets=False,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(structures=structures, config=cfg)

    assert "RMSE_force_train" in result.errors.columns
    assert calls, "compute_force_loss() was not reached during force training"
    assert all(call["graph"] is not None for call in calls)
    assert all(call["neighbor_info"] is None for call in calls)


@pytest.mark.cpu
def test_force_training_uses_persisted_hdf5_derivatives_when_available(
    tmp_path: Path,
    monkeypatch,
):
    """HDF5 training should pass persisted features and derivatives into the loss path."""
    structures = make_structures_with_forces(n_structures=6, n_atoms=4)
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structures))]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    descriptor = make_descriptor(dtype=torch.float64)
    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(tmp_path / "force_training_cached.h5"),
        sources=_source_collection_from_structures(file_paths, structures),
        mode="build",
    )
    dataset.build_database(
        show_progress=False,
        persist_features=True,
        persist_force_derivatives=True,
    )

    arch = make_arch(descriptor)
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    calls = []
    original_compute_force_loss = training_loop_module.compute_force_loss

    def _wrapped_compute_force_loss(*args, **kwargs):
        calls.append(
            {
                "features": kwargs.get("features"),
                "local_derivatives": kwargs.get("local_derivatives"),
                "graph": kwargs.get("graph"),
            }
        )
        return original_compute_force_loss(*args, **kwargs)

    monkeypatch.setattr(
        training_loop_module,
        "compute_force_loss",
        _wrapped_compute_force_loss,
    )

    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        force_fraction=1.0,
        force_sampling="fixed",
        memory_mode="cpu",
        device="cpu",
        num_workers=0,
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(dataset=dataset, config=cfg)

    assert "RMSE_force_train" in result.errors.columns
    assert calls, "compute_force_loss() was not reached during HDF5 training"
    assert any(call["local_derivatives"] is not None for call in calls)
    assert any(call["features"] is not None for call in calls)
    assert all(call["graph"] is None for call in calls)


@pytest.mark.cpu
def test_hdf5_force_training_random_sampling_initializes_force_selection(
    tmp_path: Path,
):
    """HDF5 training should initialize random force selection before epoch 0."""
    structures = make_structures_with_forces(n_structures=8, n_atoms=4)
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structures))]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    descriptor = make_descriptor(dtype=torch.float64)
    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(tmp_path / "force_training_random.h5"),
        sources=_source_collection_from_structures(file_paths, structures),
        mode="build",
        atomic_energies={"H": 0.0},
    )
    dataset.build_database(
        show_progress=False,
        persist_force_derivatives=True,
    )

    assert not hasattr(dataset, "selected_force_indices")

    pot = TorchANNPotential(arch=make_arch(descriptor), descriptor=descriptor)
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        force_fraction=0.5,
        force_sampling="random",
        memory_mode="cpu",
        device="cpu",
        num_workers=0,
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(dataset=dataset, config=cfg)

    force_rmse = result.errors["RMSE_force_train"].iloc[0]
    assert not math.isnan(force_rmse)
    assert force_rmse > 0


@pytest.mark.cpu
def test_hdf5_force_training_with_worker_restarts_smoke(tmp_path: Path):
    """HDF5 force training should remain correct with worker restarts."""
    structures = make_structures_with_forces(n_structures=8, n_atoms=4)
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structures))]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    descriptor = make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "force_training_worker_restart.h5"
    build_dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=_source_collection_from_structures(file_paths, structures),
        mode="build",
        atomic_energies={"H": 0.0},
    )
    build_dataset.build_database(
        show_progress=False,
        persist_force_derivatives=True,
    )
    dataset = HDF5StructureDataset(
        descriptor=make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        mode="load",
    )

    pot = TorchANNPotential(
        arch=make_arch(dataset.descriptor),
        descriptor=dataset.descriptor,
    )
    cfg = TorchTrainingConfig(
        iterations=2,
        testpercent=0,
        force_weight=0.5,
        force_fraction=0.5,
        force_sampling="random",
        force_resample_num_epochs=1,
        num_workers=2,
        persistent_workers=True,
        method=Adam(mu=0.001, batchsize=2),
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(dataset=dataset, config=cfg)

    force_rmse = result.errors["RMSE_force_train"].iloc[-1]
    assert not math.isnan(force_rmse)
    assert force_rmse > 0


@pytest.mark.cpu
def test_prebuilt_hdf5_dataset_uses_config_owned_runtime_policy(
    tmp_path: Path,
):
    """Prebuilt datasets should accept runtime policy only from the config."""
    structures = make_structures_with_forces(n_structures=4, n_atoms=3)
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structures))]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    descriptor = make_descriptor(dtype=torch.float64)
    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(tmp_path / "force_training_mismatch.h5"),
        sources=_source_collection_from_structures(file_paths, structures),
        mode="build",
        atomic_energies={"H": 0.0},
    )
    dataset.build_database(show_progress=False)

    pot = TorchANNPotential(arch=make_arch(descriptor), descriptor=descriptor)
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        force_fraction=1.0,
        force_sampling="random",
        cache_force_triplets=True,
        memory_mode="cpu",
        device="cpu",
        num_workers=0,
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(dataset=dataset, config=cfg)
    force_rmse = result.errors["RMSE_force_train"].iloc[0]
    assert not math.isnan(force_rmse)
    assert force_rmse > 0


@pytest.mark.cpu
def test_persisted_hdf5_force_training_matches_on_the_fly_training(
    tmp_path: Path,
):
    """Persisted-derivative HDF5 training should match the on-the-fly path."""
    structures = make_structures_with_forces(n_structures=8, n_atoms=4)
    train_structs = structures[:6]
    test_structs = structures[6:]

    descriptor_struct = make_descriptor(dtype=torch.float64)
    train_dataset = StructureDataset(
        structures=train_structs,
        descriptor=descriptor_struct,
        atomic_energies={"H": 0.0},
    )
    test_dataset = StructureDataset(
        structures=test_structs,
        descriptor=descriptor_struct,
        atomic_energies={"H": 0.0},
    )

    def _build_hdf5_dataset(
        subset_structures: list[Structure],
        database_name: str,
        descriptor: ChebyshevDescriptor,
    ) -> HDF5StructureDataset:
        file_paths = [
            str(tmp_path / f"{database_name}_{idx}.xsf")
            for idx in range(len(subset_structures))
        ]
        for path in file_paths:
            Path(path).write_text("placeholder", encoding="utf-8")

        dataset = HDF5StructureDataset(
            descriptor=descriptor,
            database_file=str(tmp_path / f"{database_name}.h5"),
            sources=_source_collection_from_structures(
                file_paths,
                subset_structures,
            ),
            mode="build",
            atomic_energies={"H": 0.0},
        )
        dataset.build_database(
            show_progress=False,
            persist_force_derivatives=True,
        )
        return dataset

    descriptor_hdf5 = make_descriptor(dtype=torch.float64)
    hdf5_train_dataset = _build_hdf5_dataset(
        train_structs,
        "equiv_train",
        descriptor_hdf5,
    )
    hdf5_test_dataset = _build_hdf5_dataset(
        test_structs,
        "equiv_test",
        descriptor_hdf5,
    )

    cfg = TorchTrainingConfig(
        iterations=3,
        testpercent=0,
        force_weight=0.5,
        force_fraction=1.0,
        force_sampling="fixed",
        cache_force_triplets=False,
        method=Adam(mu=0.001, batchsize=2),
        memory_mode="cpu",
        device="cpu",
        num_workers=0,
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    torch.manual_seed(17)
    pot_struct = TorchANNPotential(
        arch=make_arch(descriptor_struct),
        descriptor=descriptor_struct,
    )
    struct_result = pot_struct.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=cfg,
    )

    torch.manual_seed(17)
    pot_hdf5 = TorchANNPotential(
        arch=make_arch(descriptor_hdf5),
        descriptor=descriptor_hdf5,
    )
    hdf5_result = pot_hdf5.train(
        train_dataset=hdf5_train_dataset,
        test_dataset=hdf5_test_dataset,
        config=cfg,
    )

    metrics = [
        "MAE_train",
        "RMSE_train",
        "MAE_test",
        "RMSE_test",
        "RMSE_force_train",
        "RMSE_force_test",
    ]
    np.testing.assert_allclose(
        struct_result.errors[metrics].to_numpy(),
        hdf5_result.errors[metrics].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    struct_state = pot_struct.model.state_dict()
    hdf5_state = pot_hdf5.model.state_dict()
    assert struct_state.keys() == hdf5_state.keys()
    for key in struct_state:
        assert torch.allclose(
            struct_state[key],
            hdf5_state[key],
            rtol=1e-10,
            atol=1e-10,
        ), key


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
