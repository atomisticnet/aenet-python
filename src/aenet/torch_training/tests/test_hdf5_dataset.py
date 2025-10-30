import math
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from aenet.torch_training import (
    HDF5StructureDataset,
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
)
from aenet.torch_featurize import ChebyshevDescriptor


def _make_struct(i: int) -> Structure:
    """
    Create a small H-only Structure for testing.

    Two variants are produced to vary energies slightly while keeping
    distances within the descriptor cutoff for neighbor construction.
    """
    if i % 2 == 0:
        pos = np.array(
            [[0.0, 0.0, 0.0],
             [0.9, 0.0, 0.0],
             [0.0, 0.9, 0.0]],
            dtype=np.float64,
        )
        energy = 0.0
    else:
        pos = np.array(
            [[0.1, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        energy = 0.5
    species = ["H", "H", "H"]
    forces = np.zeros_like(pos)
    return Structure(positions=pos, species=species,
                     energy=energy, forces=forces)


def _parser_factory(structs: List[Structure]):
    """
    Build a top-level, picklable parser callable that maps file path
    suffix index to a pre-generated Structure. This avoids I/O in tests.
    """
    def _parser(path: str) -> Structure:
        name = Path(path).name
        # Expect names like "s_0", "s_1", etc.
        try:
            idx = int(name.split("_")[-1])
        except Exception:
            idx = 0
        return structs[idx]
    return _parser


def _make_descriptor(dtype=torch.float64) -> ChebyshevDescriptor:
    # Keep orders small to minimize compute; ensure within cutoffs.
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


def _make_arch(descriptor: ChebyshevDescriptor):
    # Single-species small network; output layer implicit linear(1).
    return {"H": [(4, "tanh")]}


@pytest.mark.cpu
def test_hdf5_build_and_getitem(tmp_path: Path):
    # Prepare synthetic structures and dummy file paths.
    structs = [_make_struct(0), _make_struct(1)]
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structs))]
    for p in file_paths:
        Path(p).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        file_paths=file_paths,
        parser=_parser_factory(structs),
        mode="build",
        compression="zlib",
        compression_level=5,
    )
    # Build DB (serialize pickled Structures + metadata)
    ds.build_database(show_progress=False)

    # Basic checks
    assert len(ds) == len(structs)
    sample0 = ds[0]
    assert "features" in sample0
    assert "energy" in sample0
    assert "species_indices" in sample0
    assert sample0["n_atoms"] == 3
    # Features shape has 3 rows (atoms) x F columns
    assert sample0["features"].shape[0] == 3


@pytest.mark.cpu
def test_trainer_with_hdf5_dataset_smoke(tmp_path: Path):
    # Build small HDF5 dataset
    structs = [_make_struct(0), _make_struct(1)]
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structs))]
    for p in file_paths:
        Path(p).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        file_paths=file_paths,
        parser=_parser_factory(structs),
        mode="build",
        compression="zlib",
        compression_level=5,
        # Ensure forces are used (both structs have forces)
        force_fraction=1.0,
        force_sampling="fixed",
    )
    ds.build_database(show_progress=False)

    # Trainer
    arch = _make_arch(desc)
    pot = TorchANNPotential(arch=arch, descriptor=desc)

    # Force-supervised single-epoch smoke test
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        memory_mode="cpu",
        device="cpu",
        num_workers=0,
    )

    ckpt_dir = tmp_path / "ckpts"
    history = pot.train(
        dataset=ds,
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=1,
        resume_from=None,
        save_best=False,
        use_scheduler=False,
    )

    # Verify history keys and checkpoint dir created
    assert "train_force_rmse" in history
    assert len(history["train_force_rmse"]) == 1
    assert not math.isnan(history["train_force_rmse"][0])
    assert ckpt_dir.exists()
