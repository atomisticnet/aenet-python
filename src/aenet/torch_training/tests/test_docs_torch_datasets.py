"""Docs-backed smoke tests for ``docs/source/usage/torch_datasets.rst``."""

import tarfile
from pathlib import Path
from types import SimpleNamespace

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
from aenet.torch_training.sources import TarArchiveXSFSourceCollection


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


def _write_xsf_tar_bz2(
    atomic_structures: list[AtomicStructure],
    archive_path: Path,
) -> Path:
    """Write a tar.bz2 archive containing the provided XSF structures."""
    member_dir = archive_path.parent / f"{archive_path.stem}_members"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_paths = _write_xsf_files(
        atomic_structures,
        member_dir,
    )
    with tarfile.open(archive_path, mode="w:bz2") as archive:
        for member_path in member_paths:
            archive.add(
                member_path,
                arcname=f"dataset/{member_path.name}",
            )
    return archive_path

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
    )
    dataset_from_atomic = StructureDataset(
        structures=docs_atomic_structures,
        descriptor=descriptor,
    )
    dataset_from_torch = StructureDataset(
        structures=docs_training_structures,
        descriptor=descriptor,
    )

    assert len(dataset_from_paths) == 2
    assert len(dataset_from_atomic) == 2
    assert len(dataset_from_torch) == 2

    sample = dataset_from_torch[0]
    assert sample["features"].shape == (3, 3)
    assert sample["use_forces"] is True
    assert sample["graph"] is not None
    assert sample["triplets"] is not None

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
    """Cached dataset split and inference examples should keep working."""
    descriptor = _make_descriptor()
    cached_dataset = CachedStructureDataset(
        structures=docs_training_structures,
        descriptor=descriptor,
        atomic_energies={"H": 0.0},
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
        normalize_features=False,
        normalize_energy=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )
    potential.train(
        train_dataset=train_ds,
        test_dataset=test_ds,
        config=config,
    )

    prediction = potential.predict_dataset(
        test_ds,
        config=PredictionConfig(batch_size=1),
    )
    assert len(prediction.total_energy) == 1

    manual_train_ds = Subset(cached_dataset, [0])
    manual_test_ds = Subset(cached_dataset, [1])
    potential.train(
        train_dataset=manual_train_ds,
        test_dataset=manual_test_ds,
        config=config,
    )


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_hdf5_build_load_and_generic_split_examples(
    docs_atomic_structures,
    tmp_path,
    monkeypatch,
):
    """The HDF5 docs workflow should remain valid on a tiny dataset."""
    descriptor = _make_descriptor()
    file_paths = _write_xsf_files(docs_atomic_structures, tmp_path)
    database_file = tmp_path / "training.h5"

    build_dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(database_file),
        sources=[str(path) for path in file_paths],
        mode="build",
        compression="zlib",
        compression_level=5,
    )
    build_dataset.build_database(
        show_progress=False,
        persist_descriptor=True,
        persist_features=True,
        persist_force_derivatives=True,
    )
    assert build_dataset._h5 is None
    assert build_dataset.has_persisted_features() is True
    assert build_dataset.has_persisted_force_derivatives() is True

    sample = build_dataset[0]
    assert len(build_dataset) == 2
    assert sample["features"].shape == (3, 3)
    assert sample["local_derivatives"] is not None
    assert sample["graph"] is None
    assert sample["triplets"] is None

    load_dataset = HDF5StructureDataset(
        descriptor=None,
        database_file=str(database_file),
        mode="load",
    )
    loaded_sample = load_dataset[0]
    assert load_dataset.has_persisted_features() is True
    assert load_dataset.has_persisted_force_derivatives() is True
    assert load_dataset.get_persisted_feature_cache_info() is not None
    assert load_dataset.get_force_derivative_cache_info() is not None

    cache_state = SimpleNamespace(
        feature_cache={0: torch.full((3, 3), 7.0, dtype=descriptor.dtype)},
        neighbor_cache={},
        graph_cache={},
    )

    def _fail_persisted_feature_load(_idx: int):
        raise AssertionError("runtime cache should take precedence")

    monkeypatch.setattr(
        load_dataset,
        "load_persisted_features",
        _fail_persisted_feature_load,
    )
    cached_sample = load_dataset.materialize_sample(
        0,
        use_forces=False,
        cache_state=cache_state,
        cache_features=True,
    )
    assert torch.allclose(
        cached_sample["features"],
        cache_state.feature_cache[0],
    )

    monkeypatch.undo()
    force_sample = load_dataset.materialize_sample(
        0,
        use_forces=True,
        load_local_derivatives=True,
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
    assert loaded_sample["local_derivatives"] is not None
    assert loaded_sample["graph"] is None
    assert loaded_sample["triplets"] is None
    assert force_sample["local_derivatives"] is not None
    assert force_sample["graph"] is None
    assert force_sample["triplets"] is None

    load_dataset.close()
    build_dataset.close()
    assert load_dataset._h5 is None
    assert build_dataset._h5 is None


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_hdf5_tar_archive_source_example(
    docs_atomic_structures,
    tmp_path,
):
    """The tar-backed HDF5 docs example should remain runnable."""
    descriptor = _make_descriptor()
    archive_path = _write_xsf_tar_bz2(
        docs_atomic_structures,
        tmp_path / "training.tar.bz2",
    )
    database_file = tmp_path / "training_from_tar.h5"

    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(database_file),
        sources=TarArchiveXSFSourceCollection(archive_path),
        mode="build",
    )
    dataset.build_database(show_progress=False)

    assert len(dataset) == 2
    assert dataset.get_entry_metadata(0)["source_kind"] == "tar_xsf_member"
    assert dataset[0]["features"].shape == (3, 3)
