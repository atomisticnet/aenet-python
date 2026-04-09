"""Docs-backed smoke tests for ``docs/source/usage/torch_inference.rst``."""

from pathlib import Path

import numpy as np
import pytest
import torch

from aenet.geometry import AtomicStructure
from aenet.mlip import PredictionConfig
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    Adam,
    Structure,
    TorchANNPotential,
    TorchCommitteeConfig,
    TorchCommitteePotential,
    TorchTrainingConfig,
)
from aenet.torch_training.dataset import CachedStructureDataset


def _make_descriptor(dtype=torch.float64) -> ChebyshevDescriptor:
    """Create the small single-species descriptor used by the docs tests."""
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


def _make_arch() -> dict:
    """Create the small single-species network used by the docs tests."""
    return {"H": [(4, "tanh")]}


def _make_structures() -> list[Structure]:
    """Create the tiny structures used by the inference docs examples."""
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
    return [
        Structure(
            positions=positions_a,
            species=["H", "H", "H"],
            energy=0.0,
            name="docs-inference-0.xsf",
        ),
        Structure(
            positions=positions_b,
            species=["H", "H", "H"],
            energy=0.5,
            name="docs-inference-1.xsf",
        ),
    ]


def _structures_to_atomic(
    structures: list[Structure],
) -> list[AtomicStructure]:
    """Convert docs test structures into AtomicStructure objects."""
    atomic_structures = []
    for structure in structures:
        atomic = AtomicStructure(
            structure.positions,
            structure.species,
            energy=structure.energy,
        )
        atomic.name = structure.name
        atomic_structures.append(atomic)
    return atomic_structures


def _write_xsf_files(
    atomic_structures: list[AtomicStructure],
    directory: Path,
) -> list[Path]:
    """Write temporary XSF files for the file-backed docs examples."""
    paths = []
    for index, structure in enumerate(atomic_structures):
        path = directory / f"docs-inference-{index}.xsf"
        structure.to_file(path)
        paths.append(path)
    return paths


def _make_trained_potential(
    structures: list[Structure],
) -> TorchANNPotential:
    """Create the tiny trained potential used by the inference docs tests."""
    potential = TorchANNPotential(_make_arch(), descriptor=_make_descriptor())
    config = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        atomic_energies={"H": 0.0},
        normalize_features=False,
        normalize_energy=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )
    potential.train(structures=structures, config=config)
    return potential


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_saved_model_and_structure_prediction_examples(tmp_path):
    """The saved-model inference example should stay runnable."""
    structures = _make_structures()
    atomic_structures = _structures_to_atomic(structures)
    structure_paths = _write_xsf_files(atomic_structures, tmp_path)

    model_path = tmp_path / "trained_model.pt"
    _make_trained_potential(structures).save(model_path)

    potential = TorchANNPotential.from_file(model_path)
    results = potential.predict(
        [structure_paths[0]],
        eval_forces=True,
        config=PredictionConfig(
            print_atomic_energies=True,
            timing=True,
        ),
    )

    assert model_path.exists()
    assert potential.metadata is not None
    assert len(results.total_energy) == 1
    assert len(results.cohesive_energy) == 1
    assert results.paths == [str(structure_paths[0])]
    assert results.forces is not None
    assert results.forces[0].shape == (3, 3)
    assert results.atom_energies is not None
    assert results.atom_energies[0].shape == (3,)
    assert results.timing is not None
    assert "featurization" in results.timing


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_cached_dataset_inference_example(tmp_path):
    """The cached-dataset inference example should stay runnable."""
    structures = _make_structures()
    atomic_structures = _structures_to_atomic(structures)
    structure_paths = _write_xsf_files(atomic_structures, tmp_path)

    model_path = tmp_path / "trained_model.pt"
    _make_trained_potential(structures).save(model_path)

    potential = TorchANNPotential.from_file(model_path)
    dataset = CachedStructureDataset(
        structures=[AtomicStructure.from_file(path) for path in structure_paths],
        descriptor=potential.descriptor,
        show_progress=False,
    )

    results = potential.predict_dataset(
        dataset,
        config=PredictionConfig(batch_size=1),
    )

    assert len(results.total_energy) == 2
    assert results.forces is None
    assert results.coords[0].shape == (3, 3)
    assert results.atom_types[0] == ["H", "H", "H"]


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_committee_ascii_export_example(tmp_path):
    """The committee export example in the inference docs should stay runnable."""
    structures = _make_structures()
    committee = TorchCommitteePotential(_make_arch(), descriptor=_make_descriptor())
    result = committee.train(
        structures=structures,
        config=TorchTrainingConfig(
            iterations=0,
            method=Adam(mu=0.001, batchsize=1),
            testpercent=0,
            force_weight=0.0,
            memory_mode="cpu",
            device="cpu",
            atomic_energies={"H": 0.0},
            normalize_features=False,
            normalize_energy=False,
            checkpoint_dir=None,
            checkpoint_interval=0,
            max_checkpoints=None,
            save_best=False,
            use_scheduler=False,
            show_progress=False,
        ),
        committee_config=TorchCommitteeConfig(
            num_members=2,
            base_seed=11,
            max_parallel=1,
            output_dir=tmp_path / "committee_run",
        ),
    )

    reloaded = TorchCommitteePotential.from_directory(result.output_dir)
    manifest = reloaded.to_aenet_ascii(
        tmp_path / "ascii_committee",
        prefix="committee",
        structures=structures,
    )

    assert len(manifest) == 2
    assert (tmp_path / "ascii_committee" / "member_000" / "committee.H.nn.ascii").exists()
    assert (tmp_path / "ascii_committee" / "member_001" / "committee.H.nn.ascii").exists()
