import math
from pathlib import Path

import numpy as np
import pytest
import torch

from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
)


def make_simple_structures_H_two():
    # Two small H-only structures with distances within cutoff
    # Structure A: triangle
    pos_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    # Structure B: shifted triangle
    pos_b = np.array(
        [
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    species_a = ["H", "H", "H"]
    species_b = ["H", "H", "H"]

    # Simple arbitrary energies (not physically meaningful)
    E_a = 0.0
    E_b = 0.5

    # Optional forces (zeros, just to exercise the path)
    F_a = np.zeros_like(pos_a)
    F_b = np.zeros_like(pos_b)

    sA = Structure(positions=pos_a, species=species_a, energy=E_a, forces=F_a)
    sB = Structure(positions=pos_b, species=species_b, energy=E_b, forces=F_b)
    return [sA, sB]


def make_descriptor_H(dtype=torch.float64):
    # Keep orders small to minimize compute; ensure within cutoffs
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


def make_arch_H(descriptor: ChebyshevDescriptor):
    # For single species, n_features = (rad_order+1) + (ang_order+1) = 2
    # Hidden size small; activations supported by NetAtom: linear/tanh/sigmoid
    return {
        "H": [(4, "tanh")],
    }


def zero_model_weights(pot: TorchANNPotential):
    for seq in pot.net.functions:
        for p in seq.parameters():
            with torch.no_grad():
                p.zero_()


@pytest.mark.cpu
def test_energy_only_smoke(tmp_path: Path):
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    ckpt_dir = tmp_path / "ckpts"
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=50,  # trigger validation + best model path
        force_weight=0.0,  # energy-only
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=None,
        save_best=True,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    # TrainOut object populated
    assert "RMSE_train" in result.errors.columns
    assert len(result.errors) == 1
    assert not math.isnan(result.errors["RMSE_train"].iloc[0])

    # Checkpoint saved
    assert ckpt_dir.exists()
    # Either checkpoint_epoch_0000.pt or best_model.pt (or both)
    files = {p.name for p in ckpt_dir.iterdir()}
    assert any(name.startswith("checkpoint_epoch_")
               and name.endswith(".pt") for name in files)


@pytest.mark.cpu
def test_warns_for_scheduler_with_tiny_validation_set():
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=1,
        method=None,
        testpercent=50,
        force_weight=0.0,
        atomic_energies={"H": 0.0},
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=True,
        show_progress=False,
    )

    with pytest.warns(
        UserWarning,
        match=r"use_scheduler=True with a validation set of only 1 structure",
    ):
        pot.train(
            structures=structures,
            config=cfg,
        )


@pytest.mark.cpu
def test_warns_for_save_best_with_tiny_validation_set(tmp_path: Path):
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    ckpt_dir = tmp_path / "ckpts"
    cfg = TorchTrainingConfig(
        iterations=1,
        method=None,
        testpercent=50,
        force_weight=0.0,
        atomic_energies={"H": 0.0},
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=None,
        save_best=True,
        use_scheduler=False,
        show_progress=False,
    )

    with pytest.warns(
        UserWarning,
        match=r"save_best=True with a validation set of only 1 structure",
    ):
        pot.train(
            structures=structures,
            config=cfg,
        )


@pytest.mark.cpu
def test_force_training_smoke(tmp_path: Path):
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    ckpt_dir = tmp_path / "ckpts"
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,   # no validation path here
        force_weight=0.5,  # include force term
        force_fraction=1.0,
        force_sampling="fixed",
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        save_forces=False,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=2,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    # TrainOut object populated
    assert "RMSE_force_train" in result.errors.columns
    assert len(result.errors) == 1
    # force rmse should be a number
    assert not math.isnan(result.errors["RMSE_force_train"].iloc[0])

    # Checkpoint saved and rotation does not error (single epoch anyway)
    assert ckpt_dir.exists()
    files = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    assert len(files) >= 1


@pytest.mark.cpu
def test_save_energies_writes_compatible_trainout_files(tmp_path: Path,
                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=50,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    assert (tmp_path / "energies.train.0").exists()
    assert (tmp_path / "energies.test.0").exists()
    assert result.energies is not None
    assert len(result.energies.energies_train) == 1
    assert len(result.energies.energies_test) == 1
    assert "ANN(eV/atom)" in result.energies.energies_train.columns
    assert "Ref(eV/atom)" in result.energies.energies_train.columns


@pytest.mark.cpu
def test_save_energies_uses_predict_dataset_for_cached_splits(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    call_count = {"forward_from_positions": 0}
    orig_forward = descriptor.forward_from_positions

    def _wrapped_forward(*args, **kwargs):
        call_count["forward_from_positions"] += 1
        return orig_forward(*args, **kwargs)

    monkeypatch.setattr(descriptor, "forward_from_positions", _wrapped_forward)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=50,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        cache_features=True,
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    pot.train(
        structures=structures,
        config=cfg,
    )

    # One featurization per structure during CachedStructureDataset build,
    # and no extra featurization during save_energies.
    assert call_count["forward_from_positions"] == 2


@pytest.mark.cpu
def test_save_energies_without_test_split_writes_train_only(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    assert (tmp_path / "energies.train.0").exists()
    assert not (tmp_path / "energies.test.0").exists()
    assert result.energies is not None
    assert len(result.energies.energies_train) == 2
    assert result.energies.energies_test is None


@pytest.mark.cpu
def test_save_energies_uses_total_energy_columns_with_atomic_references(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    for s in structures:
        s.energy = 3.69  # 3 H atoms with E_H = 1.23 each

    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)
    zero_model_weights(pot)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        atomic_energies={"H": 1.23},
        normalize_features=False,
        normalize_energy=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    train_df = result.energies.energies_train
    assert np.allclose(train_df["Ref(eV)"].values, 3.69)
    assert np.allclose(train_df["ANN(eV)"].values, 3.69)
    assert np.allclose(train_df["Ref-ANN(eV/atom)"].values, 0.0)
