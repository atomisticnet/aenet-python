import math
import os
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


@pytest.mark.cpu
def test_energy_only_smoke(tmp_path: Path):
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=50,  # trigger validation + best model path
        force_weight=0.0,  # energy-only
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
    )

    ckpt_dir = tmp_path / "ckpts"
    history = pot.train(
        structures=structures,
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=None,
        resume_from=None,
        save_best=True,
        use_scheduler=False,
    )

    # History keys populated
    assert "train_energy_rmse" in history
    assert len(history["train_energy_rmse"]) == 1
    assert not math.isnan(history["train_energy_rmse"][0])

    # Checkpoint saved
    assert ckpt_dir.exists()
    # Either checkpoint_epoch_0000.pt or best_model.pt (or both)
    files = {p.name for p in ckpt_dir.iterdir()}
    assert any(name.startswith("checkpoint_epoch_") and name.endswith(".pt") for name in files)


@pytest.mark.cpu
def test_force_training_smoke(tmp_path: Path):
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

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
    )

    ckpt_dir = tmp_path / "ckpts"
    history = pot.train(
        structures=structures,
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=2,
        resume_from=None,
        save_best=False,
        use_scheduler=False,
    )

    # History populated
    assert "train_force_rmse" in history
    assert len(history["train_force_rmse"]) == 1
    # force rmse should be a number
    assert not math.isnan(history["train_force_rmse"][0])

    # Checkpoint saved and rotation does not error (single epoch anyway)
    assert ckpt_dir.exists()
    files = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    assert len(files) >= 1
