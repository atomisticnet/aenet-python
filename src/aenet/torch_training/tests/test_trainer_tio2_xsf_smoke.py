import math
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from aenet.torch_training import (
    TorchANNPotential,
    TorchTrainingConfig,
    Structure,
)
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.formats.xsf import XSFParser


def load_tio2_structures(n_structs: int = 3) -> List[Structure]:
    """
    Load a few TiO2 XSF structures from repo data and convert
    to Torch Structures.

    Energies and forces are not guaranteed in these XSFs, so we
    set energy=0.0.
    """
    # test file is at: src/aenet/torch_training/tests/this_file.py
    # xsf dir is at:  src/aenet/tests/data/xsf-TiO2
    root = Path(__file__).resolve().parents[4]
    xsf_dir = root / "src" / "aenet" / "tests" / "data" / "xsf-TiO2"
    assert xsf_dir.exists(), f"XSF directory not found: {xsf_dir}"

    files = sorted(xsf_dir.glob("*.xsf"))[:n_structs]
    assert len(files) > 0, f"No XSF files found in {xsf_dir}"

    parser = XSFParser()
    out: List[Structure] = []
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
        forces = None
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

    cfg = TorchTrainingConfig(
        iterations=1,         # single epoch
        testpercent=50,       # exercise validation + best model saving
        force_weight=0.0,     # energy-only
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
