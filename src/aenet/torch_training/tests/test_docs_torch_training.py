"""Docs-backed smoke tests for ``docs/source/usage/torch_training.rst``."""

import numpy as np
import pytest
import torch

from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    Adam,
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
)


def _make_structures(include_forces: bool) -> list[Structure]:
    """Create the tiny structures used by the training docs examples."""
    positions_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    positions_b = np.array(
        [
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    forces = np.zeros((3, 3), dtype=np.float64) if include_forces else None
    return [
        Structure(
            positions=positions_a,
            species=["H", "H", "H"],
            energy=0.0,
            forces=forces,
        ),
        Structure(
            positions=positions_b,
            species=["H", "H", "H"],
            energy=0.5,
            forces=forces,
        ),
    ]


def _make_descriptor(dtype=torch.float64) -> ChebyshevDescriptor:
    """Create the small descriptor used by the training docs examples."""
    return ChebyshevDescriptor(
        species=["H"],
        rad_order=1,
        rad_cutoff=2.0,
        ang_order=0,
        ang_cutoff=2.0,
        min_cutoff=0.1,
        device="cpu",
        dtype=dtype,
    )


def _make_potential() -> TorchANNPotential:
    """Build the tiny single-species potential used by the docs tests."""
    descriptor = _make_descriptor()
    arch = {"H": [(4, "tanh")]}
    return TorchANNPotential(arch, descriptor=descriptor)


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_energy_only_training_example():
    """The compact in-memory energy-only example should stay runnable."""
    potential = _make_potential()
    config = TorchTrainingConfig(
        iterations=1,
        method=Adam(mu=0.001, batchsize=1),
        testpercent=50,
        force_weight=0.0,
        atomic_energies={"H": 0.0},
        normalize_features=False,
        normalize_energy=False,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    results = potential.train(
        structures=_make_structures(include_forces=False),
        config=config,
    )

    assert len(results.errors) == 1
    assert "RMSE_train" in results.errors.columns
    assert "RMSE_test" in results.errors.columns
    assert results.errors["RMSE_force_train"].isna().all()
    assert results.errors["RMSE_force_test"].isna().all()


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_force_training_example():
    """The compact force-training example should stay runnable."""
    potential = _make_potential()
    config = TorchTrainingConfig(
        iterations=1,
        method=Adam(mu=0.001, batchsize=1),
        testpercent=50,
        force_weight=0.1,
        force_fraction=0.5,
        force_sampling="fixed",
        atomic_energies={"H": 0.0},
        normalize_features=False,
        normalize_energy=False,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    results = potential.train(
        structures=_make_structures(include_forces=True),
        config=config,
    )

    assert len(results.errors) == 1
    assert "RMSE_train" in results.errors.columns
    assert "RMSE_test" in results.errors.columns
    assert "RMSE_force_train" in results.errors.columns


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_sampling_policy_docs_example():
    """The docs sampling-policy examples should stay aligned with the API."""
    uniform_cfg = TorchTrainingConfig(sampling_policy="uniform")
    weighted_cfg = TorchTrainingConfig(
        sampling_policy="energy_weighted",
        atomic_energies={"H": 0.0},
    )
    adaptive_cfg = TorchTrainingConfig(
        sampling_policy="error_weighted",
        atomic_energies={"H": 0.0},
    )

    assert uniform_cfg.sampling_policy == "uniform"
    assert weighted_cfg.sampling_policy == "energy_weighted"
    assert adaptive_cfg.sampling_policy == "error_weighted"


@pytest.mark.cpu
def test_mutated_mixed_memory_mode_still_fails_fast():
    """Trainer should reject a mixed mode reintroduced after validation."""
    potential = _make_potential()
    config = TorchTrainingConfig(
        iterations=1,
        method=Adam(mu=0.001, batchsize=1),
        testpercent=50,
        force_weight=0.0,
        atomic_energies={"H": 0.0},
        normalize_features=False,
        normalize_energy=False,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )
    config.memory_mode = "mixed"

    with pytest.raises(
        NotImplementedError,
        match="reserved for a future real mixed-memory execution mode",
    ):
        potential.train(
            structures=_make_structures(include_forces=False),
            config=config,
        )
