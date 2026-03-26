import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
)
from aenet.torch_training.dataset import (
    CachedStructureDataset,
    StructureDataset,
)


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

    # Configure training with known atomic reference energies
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
        atomic_energies={"H": E_H},
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    # Train with 0 iterations to set up internal state (_E_atomic, etc.)
    pot.train(
        structures=[s],
        config=cfg,
    )

    # Ensure model outputs zero cohesive energy by zeroing weights
    _zero_out_model_weights(pot)

    # Predict should return TOTAL energy = cohesive(=0) + sum(E_atomic)
    results = pot.predict([s], eval_forces=False)
    assert len(results.total_energy) == 1
    expected_total = 3 * E_H  # 3 H atoms
    assert pytest.approx(
        results.total_energy[0], rel=0, abs=1e-9) == expected_total


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
        atomic_energies={"H": E_H},
        normalize_features=False,
        normalize_energy=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )
    pot.train(
        structures=[s],
        config=cfg,
    )

    # Check helper: cohesive = total - sum(E_atomic)
    expected_coh = E_total - 3 * E_H
    coh = pot.cohesive_energy(s)
    assert pytest.approx(coh, rel=0, abs=1e-12) == expected_coh


@pytest.mark.cpu
def test_predict_dataset_reuses_cached_features(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    s1 = Structure(
        positions=positions,
        species=["H", "H", "H"],
        energy=1.0,
        forces=None,
        name="s1.xsf",
    )
    s2 = Structure(
        positions=positions + 0.05,
        species=["H", "H", "H"],
        energy=1.5,
        forces=None,
        name="s2.xsf",
    )

    descriptor = _make_descriptor(dtype=torch.float64)
    arch = _make_arch()
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        normalize_features=False,
        normalize_energy=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )
    pot.train(structures=[s1, s2], config=cfg)

    ds = CachedStructureDataset(
        structures=[s1, s2],
        descriptor=descriptor,
        show_progress=False,
    )

    def _fail(*args, **kwargs):
        raise AssertionError("forward_from_positions should not be called")

    monkeypatch.setattr(descriptor, "forward_from_positions", _fail)

    results = pot.predict_dataset(ds)
    assert len(results.total_energy) == 2
    assert results.paths == ["s1.xsf", "s2.xsf"]


@pytest.mark.cpu
def test_predict_dataset_supports_subset():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    s1 = Structure(positions=positions, species=["H", "H", "H"],
                   energy=1.0, forces=None, name="s1.xsf")
    s2 = Structure(positions=positions + 0.05, species=["H", "H", "H"],
                   energy=1.5, forces=None, name="s2.xsf")

    descriptor = _make_descriptor(dtype=torch.float64)
    arch = _make_arch()
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        normalize_features=False,
        normalize_energy=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )
    pot.train(structures=[s1, s2], config=cfg)

    ds = CachedStructureDataset(
        structures=[s1, s2],
        descriptor=descriptor,
        show_progress=False,
    )
    subset = Subset(ds, [1])

    results = pot.predict_dataset(subset)
    assert len(results.total_energy) == 1
    assert results.paths == ["s2.xsf"]
    assert results.atom_types[0] == ["H", "H", "H"]


@pytest.mark.cpu
def test_cached_dataset_matches_structure_dataset_energy_materialization():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    structures = [
        Structure(
            positions=positions,
            species=["H", "H", "H"],
            energy=1.0,
            forces=None,
            name="s1.xsf",
        ),
        Structure(
            positions=positions + 0.05,
            species=["H", "H", "H"],
            energy=1.5,
            forces=None,
            name="s2.xsf",
        ),
    ]

    descriptor = _make_descriptor(dtype=torch.float64)
    structure_ds = StructureDataset(structures=structures, descriptor=descriptor)
    cached_ds = CachedStructureDataset(
        structures=structures,
        descriptor=descriptor,
        show_progress=False,
    )

    expected = structure_ds.materialize_sample(0, use_forces=False)
    cached = cached_ds[0]

    assert torch.allclose(expected["features"], cached["features"])
    assert torch.equal(expected["species_indices"], cached["species_indices"])
    assert expected["n_atoms"] == cached["n_atoms"]
    assert expected["energy"] == cached["energy"]
    assert cached["use_forces"] is False
    assert cached["positions"] is None
