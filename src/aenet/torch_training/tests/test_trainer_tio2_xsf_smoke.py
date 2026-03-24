import math
from pathlib import Path

import numpy as np
import pytest
import torch

from aenet.formats.xsf import XSFParser
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
)


def load_tio2_structures(n_structs: int = 3) -> list[Structure]:
    """
    Load a few TiO2 XSF fixtures and convert them to torch Structures.

    The repository fixtures include total energies and per-atom forces, so
    the returned structures preserve those labels when present.
    """
    # test file is at: src/aenet/torch_training/tests/this_file.py
    # xsf dir is at:  src/aenet/tests/data/xsf-TiO2
    root = Path(__file__).resolve().parents[4]
    xsf_dir = root / "src" / "aenet" / "tests" / "data" / "xsf-TiO2"
    assert xsf_dir.exists(), f"XSF directory not found: {xsf_dir}"

    files = sorted(xsf_dir.glob("*.xsf"))[:n_structs]
    assert len(files) > 0, f"No XSF files found in {xsf_dir}"

    parser = XSFParser()
    out: list[Structure] = []
    for p in files:
        s = parser.read(str(p))
        positions = np.array(s.coords[-1], dtype=np.float64)
        species = [str(t) for t in s.types]
        # XSFs may not have energy/forces populated; use zeros for smoke test
        energy = (
            float(s.energy[-1])
            if (s.energy and s.energy[-1] is not None)
            else 0.0
        )
        forces = (
            np.array(s.forces[-1], dtype=np.float64)
            if (s.forces and s.forces[-1] is not None)
            else None
        )
        cell = np.array(s.avec[-1]) if s.pbc else None
        pbc = np.array([True, True, True]) if s.pbc else None
        out.append(
            Structure(
                positions=positions,
                species=species,
                energy=energy,
                forces=forces,
                cell=cell,
                pbc=pbc,
                name=p.name,
            )
        )
    return out


def make_descriptor_tio2(dtype=torch.float64):
    # Keep fairly small orders for speed in smoke test
    return ChebyshevDescriptor(
        species=["Ti", "O"],
        rad_order=3,
        rad_cutoff=5.0,
        ang_order=1,
        ang_cutoff=3.0,
        min_cutoff=0.5,
        device="cpu",
        dtype=dtype,
    )


def make_arch_tio2():
    # Simple per-species 1-layer models using supported activations
    return {
        "Ti": [(8, "tanh")],
        "O": [(8, "tanh")],
    }


@pytest.mark.cpu
def test_energy_only_tio2_xsf_smoke(tmp_path: Path):
    # Load 3 TiO2 structures from repo data
    structures = load_tio2_structures(n_structs=3)
    descriptor = make_descriptor_tio2(dtype=torch.float64)
    arch = make_arch_tio2()

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    ckpt_dir = tmp_path / "ckpts"
    cfg = TorchTrainingConfig(
        iterations=1,         # single epoch
        testpercent=50,       # exercise validation + best model saving
        force_weight=0.0,     # energy-only
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

    # Checkpoint and/or best model saved
    assert ckpt_dir.exists()
    names = {p.name for p in ckpt_dir.iterdir()}
    assert (
        any(
            n.startswith("checkpoint_epoch_") and n.endswith(".pt")
            for n in names
        )
        or "best_model.pt" in names
    )


@pytest.mark.cpu
def test_force_training_tio2_xsf_sparse_smoke(tmp_path: Path):
    """
    Periodic TiO2 force training should work with the default sparse path.
    """
    structures = load_tio2_structures(n_structs=3)
    assert all(structure.forces is not None for structure in structures)
    assert any(
        np.linalg.norm(structure.forces) > 0.0
        for structure in structures
        if structure.forces is not None
    )
    descriptor = make_descriptor_tio2(dtype=torch.float64)
    arch = make_arch_tio2()

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        force_fraction=1.0,
        force_sampling="fixed",
        cache_force_neighbors=True,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    assert "RMSE_force_train" in result.errors.columns
    assert len(result.errors) == 1
    force_rmse = result.errors["RMSE_force_train"].iloc[0]
    assert not math.isnan(force_rmse)
    assert force_rmse >= 0.0
