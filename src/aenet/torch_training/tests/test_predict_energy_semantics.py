import numpy as np
import torch
import pytest

from aenet.torch_training import (
    TorchTrainingConfig,
    Structure,
    TorchANNPotential,
)
from aenet.torch_featurize import ChebyshevDescriptor


def _make_descriptor(dtype=torch.float64):
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


def _make_arch():
    return {"H": [(4, "tanh")]}


def _zero_out_model_weights(pot: TorchANNPotential):
    # Zero all parameters in underlying per-species MLPs so network outputs 0
    # EnergyModelAdapter wraps `net` with .functions per species
    for seq in pot.net.functions:
        for p in seq.parameters():
            with torch.no_grad():
                p.zero_()


@pytest.mark.cpu
def test_predict_returns_total_energy_when_trained_on_cohesive():
    # Single structure with 3 H atoms
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    species = ["H", "H", "H"]
    E_total_label = 0.0  # label value is not used by prediction in this test
    s = Structure(positions=positions, species=species,
                  energy=E_total_label, forces=None)

    descriptor = _make_descriptor(dtype=torch.float64)
    arch = _make_arch()
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    # Configure training to 'cohesive' with known atomic reference energies
    E_H = 1.23
    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        save_forces=False,
        normalize_features=False,
        normalize_energy=False,
        energy_target="cohesive",
        E_atomic={"H": E_H},
    )

    # Train with 0 iterations to set up internal state (_E_atomic, etc.)
    pot.train(
        structures=[s],
        config=cfg,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        resume_from=None,
        save_best=False,
        use_scheduler=False,
    )

    # Ensure model outputs zero cohesive energy by zeroing weights
    _zero_out_model_weights(pot)

    # Predict should return TOTAL energy = cohesive(=0) + sum(E_atomic)
    energies, _ = pot.predict([s], predict_forces=False)
    assert len(energies) == 1
    expected_total = 3 * E_H  # 3 H atoms
    assert pytest.approx(energies[0], rel=0, abs=1e-9) == expected_total


@pytest.mark.cpu
def test_cohesive_energy_helper_works_with_internal_atomic_energies():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    species = ["H", "H", "H"]
    E_H = 2.5
    # Total energy label for structure
    E_total = 10.0
    s = Structure(positions=positions, species=species,
                  energy=E_total, forces=None)

    descriptor = _make_descriptor(dtype=torch.float64)
    arch = _make_arch()
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    # Set training config to store E_atomic in the trainer
    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        energy_target="cohesive",
        E_atomic={"H": E_H},
        normalize_features=False,
        normalize_energy=False,
    )
    pot.train(
        structures=[s],
        config=cfg,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        resume_from=None,
        save_best=False,
        use_scheduler=False,
    )

    # Check helper: cohesive = total - sum(E_atomic)
    expected_coh = E_total - 3 * E_H
    coh = pot.cohesive_energy(s)
    assert pytest.approx(coh, rel=0, abs=1e-12) == expected_coh
