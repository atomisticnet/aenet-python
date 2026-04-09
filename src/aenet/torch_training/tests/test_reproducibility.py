"""Tests for trainer reproducibility controls."""

from __future__ import annotations

from pathlib import Path

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
from aenet.torch_training.dataset import CachedStructureDataset


def _make_descriptor() -> ChebyshevDescriptor:
    """Create the small descriptor used by the reproducibility tests."""
    return ChebyshevDescriptor(
        species=["H"],
        rad_order=2,
        rad_cutoff=3.0,
        ang_order=0,
        ang_cutoff=3.0,
        min_cutoff=0.1,
        device="cpu",
        dtype=torch.float64,
    )


def _make_architecture() -> dict[str, list[tuple[int, str]]]:
    """Return a compact single-species architecture."""
    return {"H": [(6, "tanh"), (6, "tanh")]}


def _make_named_structures(*, include_forces: bool) -> list[Structure]:
    """Create a deterministic set of named hydrogen structures."""
    structures: list[Structure] = []
    base_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ],
        dtype=np.float64,
    )
    for idx in range(6):
        offset = 0.02 * idx
        positions = base_positions + np.array(
            [[offset, 0.0, 0.0], [0.0, offset, 0.0], [0.0, 0.0, offset]],
            dtype=np.float64,
        )
        forces = None
        if include_forces:
            forces = np.full((3, 3), 0.01 * (idx + 1), dtype=np.float64)
        structures.append(
            Structure(
                positions=positions,
                species=["H", "H", "H"],
                energy=0.1 * idx,
                forces=forces,
                name=f"struct_{idx}.xsf",
            )
        )
    return structures


def _read_energy_output_names(path: Path) -> list[str]:
    """Return the structure identifiers recorded in an energies.* file."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.split()[-1] for line in lines[1:]]


def _assert_matching_history(
    first: dict[str, list[float]],
    second: dict[str, list[float]],
) -> None:
    """Compare deterministic metric histories while ignoring timing noise."""
    assert first.keys() == second.keys()
    for key in first:
        if "time" in key:
            continue
        first_values = np.asarray(first[key], dtype=np.float64)
        second_values = np.asarray(second[key], dtype=np.float64)
        np.testing.assert_allclose(
            first_values,
            second_values,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


def _run_split_capture(
    tmp_path: Path,
    *,
    input_mode: str,
    split_seed: int,
    run_seed: int | None,
) -> tuple[list[str], list[str]]:
    """Train with zero epochs and return the saved train/test split names."""
    structures = _make_named_structures(include_forces=False)
    descriptor = _make_descriptor()
    potential = TorchANNPotential(_make_architecture(), descriptor=descriptor)

    config = TorchTrainingConfig(
        iterations=0,
        method=Adam(mu=0.001, batchsize=2),
        testpercent=34,
        seed=run_seed,
        split_seed=split_seed,
        atomic_energies={"H": 0.0},
        normalize_features=False,
        normalize_energy=False,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )

    if input_mode == "structures":
        potential.train(structures=structures, config=config)
    elif input_mode == "dataset":
        dataset = CachedStructureDataset(
            structures=structures,
            descriptor=descriptor,
            show_progress=False,
        )
        potential.train(dataset=dataset, config=config)
    else:
        raise ValueError(f"Unsupported input mode {input_mode!r}")

    train_names = _read_energy_output_names(tmp_path / "energies.train.0")
    test_names = _read_energy_output_names(tmp_path / "energies.test.0")
    return train_names, test_names


@pytest.mark.cpu
@pytest.mark.parametrize("input_mode", ["structures", "dataset"])
def test_split_seed_stabilizes_trainer_owned_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_mode: str,
):
    """Trainer-owned splits should depend on ``split_seed``, not ``seed``."""
    monkeypatch.chdir(tmp_path)

    first_train, first_test = _run_split_capture(
        tmp_path,
        input_mode=input_mode,
        split_seed=7,
        run_seed=101,
    )
    second_train, second_test = _run_split_capture(
        tmp_path,
        input_mode=input_mode,
        split_seed=7,
        run_seed=303,
    )

    assert first_train == second_train
    assert first_test == second_test

    changed_train, changed_test = _run_split_capture(
        tmp_path,
        input_mode=input_mode,
        split_seed=11,
        run_seed=101,
    )

    assert (changed_train, changed_test) != (first_train, first_test)


@pytest.mark.cpu
def test_seed_makes_single_model_training_reproducible(tmp_path: Path):
    """Fixed seeds should reproduce weights and history for one trainer run."""
    structures = _make_named_structures(include_forces=True)

    def _train_once(seed: int) -> TorchANNPotential:
        descriptor = _make_descriptor()
        potential = TorchANNPotential(_make_architecture(), descriptor=descriptor)
        config = TorchTrainingConfig(
            iterations=3,
            method=Adam(mu=0.01, batchsize=2),
            testpercent=0,
            seed=seed,
            split_seed=19,
            force_weight=0.5,
            force_fraction=0.5,
            force_sampling="random",
            force_resample_num_epochs=1,
            atomic_energies={"H": 0.0},
            normalize_features=False,
            normalize_energy=False,
            memory_mode="cpu",
            device="cpu",
            checkpoint_dir=str(tmp_path / f"ckpt-{seed}"),
            checkpoint_interval=0,
            max_checkpoints=None,
            save_best=False,
            use_scheduler=False,
            show_progress=False,
        )
        potential.train(structures=structures, config=config)
        return potential

    first = _train_once(17)
    second = _train_once(17)

    _assert_matching_history(first.history, second.history)
    first_state = first.model.state_dict()
    second_state = second.model.state_dict()
    assert first_state.keys() == second_state.keys()
    for key in first_state:
        torch.testing.assert_close(first_state[key], second_state[key])

    different = _train_once(18)
    assert any(
        not torch.equal(first_state[key], different.model.state_dict()[key])
        for key in first_state
    )
