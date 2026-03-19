"""Docs-backed smoke tests for ``docs/source/usage/torch_datasets.rst``."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from aenet.geometry import AtomicStructure
from aenet.mlip import PredictionConfig
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    HDF5StructureDataset,
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
    train_test_split_dataset,
)
from aenet.torch_training.dataset import (
    CachedStructureDataset,
    StructureDataset,
    train_test_split,
)


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
    """Create the tiny force-labeled structures used by the docs examples."""
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
    forces = np.zeros((3, 3), dtype=np.float64)
    return [
        Structure(
            positions=positions_a,
            species=["H", "H", "H"],
            energy=0.0,
            forces=forces,
            name="docs-structure-0.xsf",
        ),
        Structure(
            positions=positions_b,
            species=["H", "H", "H"],
            energy=0.5,
            forces=forces,
            name="docs-structure-1.xsf",
        ),
    ]


def _structures_to_atomic(
    structures: list[Structure],
) -> list[AtomicStructure]:
    """Convert the torch-training structures into AtomicStructure objects."""
    atomic_structures = []
    for structure in structures:
        atomic = AtomicStructure(
            structure.positions,
            structure.species,
            energy=structure.energy,
            forces=structure.forces,
        )
        atomic.name = structure.name
        atomic_structures.append(atomic)
    return atomic_structures


def _write_xsf_files(
    atomic_structures: list[AtomicStructure],
    directory: Path,
) -> list[Path]:
    """Write temporary XSF files for the file-path docs examples."""
    paths = []
    for index, structure in enumerate(atomic_structures):
        path = directory / f"docs-structure-{index}.xsf"
        structure.to_file(path)
        paths.append(path)
    return paths


def _parse_xsf(path: str):
    """Parse an XSF file into the torch Structure entries stored in HDF5."""
    atomic = AtomicStructure.from_file(path)
    converted = atomic.to_TorchStructure()
    if isinstance(converted, list):
        return converted
    return [converted]


@pytest.fixture
def docs_training_structures():
    """Provide the small structures shared by the docs-backed tests."""
    return _make_structures()


@pytest.fixture
def docs_atomic_structures(docs_training_structures):
    """Provide AtomicStructure versions of the shared docs fixtures."""
    return _structures_to_atomic(docs_training_structures)


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_structure_input_formats_and_split_examples(
    docs_training_structures,
    docs_atomic_structures,
    tmp_path,
):
    """Input-format and split examples should stay aligned with the docs."""
    descriptor = _make_descriptor()
    file_paths = _write_xsf_files(docs_atomic_structures, tmp_path)

    dataset_from_paths = StructureDataset(
        structures=file_paths,
        descriptor=descriptor,
        force_fraction=1.0,
        force_sampling="fixed",
    )
    dataset_from_atomic = StructureDataset(
        structures=docs_atomic_structures,
        descriptor=descriptor,
        force_fraction=1.0,
        force_sampling="fixed",
    )
    dataset_from_torch = StructureDataset(
        structures=docs_training_structures,
        descriptor=descriptor,
        force_fraction=1.0,
        force_sampling="fixed",
    )

    assert len(dataset_from_paths) == 2
    assert len(dataset_from_atomic) == 2
    assert len(dataset_from_torch) == 2

    sample = dataset_from_torch[0]
    assert sample["features"].shape == (3, 3)
    assert sample["use_forces"] is True

    force_dataset = StructureDataset(
        structures=docs_training_structures,
        descriptor=descriptor,
        force_fraction=0.5,
        force_sampling="fixed",
        cache_features=True,
        cache_force_neighbors=True,
        cache_force_triplets=True,
        seed=7,
    )
    assert force_dataset.get_statistics()["n_force_selected"] == 1

    selected_idx = force_dataset.selected_force_indices[0]
    selected_sample = force_dataset[selected_idx]
    assert selected_sample["use_forces"] is True
    assert selected_sample["graph"] is not None
    assert selected_sample["triplets"] is not None

    train_ds, test_ds = train_test_split(
        dataset_from_torch,
        test_fraction=0.5,
        seed=42,
    )
    assert len(train_ds) == 1
    assert len(test_ds) == 1


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_cached_dataset_and_predict_dataset_examples(docs_training_structures):
    """Cached dataset examples should keep their documented behavior."""
    descriptor = _make_descriptor()
    cached_dataset = CachedStructureDataset(
        structures=docs_training_structures,
        descriptor=descriptor,
        show_progress=False,
    )

    sample = cached_dataset[0]
    assert sample["features"].shape == (3, 3)
    assert sample["use_forces"] is False

    train_ds, test_ds = train_test_split_dataset(
        cached_dataset,
        test_fraction=0.5,
        seed=1,
    )
    assert isinstance(train_ds, Subset)
    assert isinstance(test_ds, Subset)
    assert len(train_ds) == 1
    assert len(test_ds) == 1

    potential = TorchANNPotential(_make_arch(), descriptor=descriptor)
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
    potential.train(structures=docs_training_structures, config=config)

    prediction = potential.predict_dataset(
        test_ds,
        config=PredictionConfig(batch_size=1),
    )
    assert len(prediction.total_energy) == 1


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_hdf5_build_load_and_generic_split_examples(
    docs_atomic_structures,
    tmp_path,
):
    """The HDF5 docs workflow should remain valid on a tiny dataset."""
    descriptor = _make_descriptor()
    file_paths = _write_xsf_files(docs_atomic_structures, tmp_path)
    database_file = tmp_path / "training.h5"

    build_dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(database_file),
        file_paths=[str(path) for path in file_paths],
        parser=_parse_xsf,
        mode="build",
        force_fraction=0.5,
        force_sampling="fixed",
        cache_force_neighbors=True,
        cache_force_triplets=True,
        compression="zlib",
        compression_level=5,
    )
    build_dataset.build_database(show_progress=False)

    sample = build_dataset[0]
    assert len(build_dataset) == 2
    assert sample["features"].shape == (3, 3)

    load_dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(database_file),
        mode="load",
        force_fraction=0.5,
        force_sampling="fixed",
        cache_features=True,
        cache_force_neighbors=True,
        cache_force_triplets=True,
    )
    train_ds, test_ds = train_test_split_dataset(
        load_dataset,
        test_fraction=0.5,
        seed=42,
    )

    assert len(load_dataset) == 2
    assert isinstance(train_ds, Subset)
    assert isinstance(test_ds, Subset)
    assert len(train_ds) == 1
    assert len(test_ds) == 1

    load_dataset._close_handle()
    build_dataset._close_handle()
