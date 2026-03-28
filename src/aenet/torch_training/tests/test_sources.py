import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from aenet.geometry import AtomicStructure
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import HDF5StructureDataset, Structure
from aenet.torch_training.sources import (
    FilePathSourceCollection,
    RecordSourceCollection,
    SourceCapabilities,
    SourceRecord,
    TarArchiveXSFSourceCollection,
    coerce_source_collection,
)


def _make_structure(energy: float = 0.0) -> Structure:
    """Create a tiny structure fixture for source-record tests."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0]],
        dtype=np.float64,
    )
    return Structure(
        positions=positions,
        species=["H", "H", "H"],
        energy=energy,
        forces=np.zeros_like(positions),
    )


def _make_descriptor() -> ChebyshevDescriptor:
    """Create a small H-only descriptor for HDF5 source tests."""
    return ChebyshevDescriptor(
        species=["H"],
        rad_order=1,
        rad_cutoff=2.0,
        ang_order=0,
        ang_cutoff=2.0,
        min_cutoff=0.1,
        device="cpu",
        dtype=torch.float64,
    )


def _write_xsf_tar_bz2(
    archive_path: Path,
    *,
    structures: list[Structure] | None = None,
    duplicate_member_names: bool = False,
) -> Path:
    """Create a tiny tar.bz2 archive containing XSF structure members."""
    if structures is None:
        structures = [_make_structure(0.0), _make_structure(1.0)]

    staging_dir = archive_path.parent / f"{archive_path.stem}_members"
    staging_dir.mkdir(parents=True, exist_ok=True)
    member_paths: list[Path] = []

    for index, structure in enumerate(structures):
        atomic = AtomicStructure(
            structure.positions,
            structure.species,
            energy=structure.energy,
            forces=structure.forces,
            avec=structure.cell,
        )
        member_path = staging_dir / f"structure-{index}.xsf"
        atomic.to_file(member_path)
        member_paths.append(member_path)

    with tarfile.open(archive_path, mode="w:bz2") as archive:
        for member_path in member_paths:
            archive_name = f"dataset/{member_path.name}"
            if duplicate_member_names:
                archive_name = "dataset/structure.xsf"
            archive.add(
                member_path,
                arcname=archive_name,
            )

    return archive_path


def _write_tar_bz2_members(
    archive_path: Path,
    members: list[tuple[str, bytes]],
) -> Path:
    """Create a tar.bz2 archive from explicit in-memory members."""
    with tarfile.open(archive_path, mode="w:bz2") as archive:
        for member_name, member_bytes in members:
            info = tarfile.TarInfo(member_name)
            info.size = len(member_bytes)
            archive.addfile(info, io.BytesIO(member_bytes))
    return archive_path


def _fixture_path(name: str) -> Path:
    """Return the path to one XSF fixture from the shared test suite."""
    return (
        Path(__file__).resolve().parents[2]
        / "formats"
        / "tests"
        / "fixtures"
        / name
    )


def _sample_signatures_match(left, right) -> None:
    """Assert two nested sample payloads are tensor-wise equivalent."""
    if left is None or right is None:
        assert left is right
        return
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.shape == right.shape
        assert torch.allclose(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _sample_signatures_match(left[key], right[key])
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
        return
    assert left == right


def test_source_record_load_structures_wraps_single_structure():
    """SourceRecord should normalize a single Structure to a list."""
    struct = _make_structure()
    record = SourceRecord(
        source_id="single",
        loader=lambda: struct,
        source_kind="memory",
    )

    loaded = record.load_structures()

    assert loaded == [struct]


def test_source_record_load_structures_preserves_multiple_structures():
    """SourceRecord should preserve multi-frame style structure lists."""
    structs = [_make_structure(0.0), _make_structure(1.0)]
    record = SourceRecord(
        source_id="multi",
        loader=lambda: structs,
        source_kind="memory",
    )

    loaded = record.load_structures()

    assert loaded == structs


def test_file_path_source_collection_reports_file_capabilities():
    """Plain file sources should advertise unrestricted file-backed access."""
    collection = FilePathSourceCollection([])

    assert collection.capabilities == SourceCapabilities(
        supports_multiple_passes=True,
        supports_random_access=True,
        supports_parallel_build=True,
    )


def test_file_path_source_collection_loads_xsf_fixture():
    """The built-in file-source adapter should parse XSF files to Structures."""
    fixture = (
        Path(__file__).resolve().parent / "data" / "TiO2-cell.xsf"
    )
    collection = FilePathSourceCollection([fixture])
    records = list(collection.iter_records())

    assert len(records) == 1
    assert records[0].source_id == str(fixture)
    assert records[0].source_kind == "file"

    structures = records[0].load_structures()

    assert len(structures) == 1
    struct = structures[0]
    assert struct.n_atoms == 23
    assert struct.species.count("Ti") == 8
    assert struct.species.count("O") == 15
    assert struct.has_forces() is True
    assert struct.cell is not None
    assert struct.pbc is not None


def test_coerce_source_collection_wraps_paths():
    """Path-like source lists should be wrapped automatically."""
    fixture = (
        Path(__file__).resolve().parent / "data" / "TiO2-cell.xsf"
    )

    collection = coerce_source_collection([fixture])

    assert isinstance(collection, FilePathSourceCollection)
    assert [record.source_id for record in collection.iter_records()] == [
        str(fixture)
    ]


def test_tar_archive_xsf_source_collection_reports_archive_capabilities(
    tmp_path: Path,
):
    """Archive-backed XSF sources should advertise conservative access flags."""
    archive_path = _write_xsf_tar_bz2(tmp_path / "structures.tar.bz2")

    collection = TarArchiveXSFSourceCollection(archive_path)

    assert collection.capabilities == SourceCapabilities(
        supports_multiple_passes=True,
        supports_random_access=False,
        supports_parallel_build=True,
    )
    assert len(collection) == 2


def test_tar_archive_xsf_source_collection_loads_archive_members(
    tmp_path: Path,
):
    """The tar-backed adapter should stream XSF members to Structures."""
    archive_path = _write_xsf_tar_bz2(tmp_path / "structures.tar.bz2")

    collection = TarArchiveXSFSourceCollection(archive_path)
    records = list(collection.iter_records())

    assert len(records) == 2
    assert records[0].source_kind == "tar_xsf_member"
    assert records[0].source_id.endswith("::member=0:dataset/structure-0.xsf")
    assert records[0].display_name == "structures.tar.bz2:dataset/structure-0.xsf"

    structures = records[0].load_structures()

    assert len(structures) == 1
    assert structures[0].n_atoms == 3
    assert structures[0].has_forces() is True


def test_hdf5_dataset_builds_from_path_sources_without_parser(tmp_path: Path):
    """HDF5StructureDataset should accept path-like sources directly."""
    fixture = (
        Path(__file__).resolve().parent / "data" / "TiO2-cell.xsf"
    )
    descriptor = ChebyshevDescriptor(
        species=["Ti", "O"],
        rad_order=1,
        rad_cutoff=6.0,
        ang_order=0,
        ang_cutoff=3.5,
        min_cutoff=0.5,
        device="cpu",
        dtype=torch.float64,
    )
    db_path = tmp_path / "path_sources.h5"

    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=[fixture],
        mode="build",
    )
    dataset.build_database(show_progress=False)

    assert len(dataset) == 1
    meta = dataset.get_entry_metadata(0)
    assert meta["source_id"] == str(fixture)
    assert meta["frame_idx"] == 0
    assert meta["source_kind"] == "file"
    assert meta["display_name"] == ""
    sample = dataset[0]
    assert sample["features"].ndim == 2


def test_hdf5_dataset_accepts_pathlike_database_file(tmp_path: Path):
    """PathLike database_file inputs should be accepted explicitly."""
    fixture = (
        Path(__file__).resolve().parent / "data" / "TiO2-cell.xsf"
    )
    descriptor = ChebyshevDescriptor(
        species=["Ti", "O"],
        rad_order=1,
        rad_cutoff=6.0,
        ang_order=0,
        ang_cutoff=3.5,
        min_cutoff=0.5,
        device="cpu",
        dtype=torch.float64,
    )
    db_path = tmp_path / "pathlike_database_file.h5"

    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=db_path,
        sources=[fixture],
        mode="build",
    )
    dataset.build_database(show_progress=False)

    assert len(dataset) == 1
    assert db_path.exists()


def test_hdf5_dataset_builds_from_record_source_collection(tmp_path: Path):
    """HDF5StructureDataset should build from explicit source records."""
    descriptor = _make_descriptor()
    records = [
        SourceRecord(
            source_id="mem_0",
            loader=lambda: _make_structure(0.0),
            source_kind="memory",
        ),
        SourceRecord(
            source_id="mem_1",
            loader=lambda: [_make_structure(0.5), _make_structure(1.0)],
            source_kind="memory",
        ),
    ]
    db_path = tmp_path / "record_sources.h5"

    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=RecordSourceCollection(records),
        mode="build",
    )
    dataset.build_database(show_progress=False)

    assert len(dataset) == 3
    assert dataset.get_entry_metadata(0)["source_id"] == "mem_0"
    assert dataset.get_entry_metadata(0)["source_kind"] == "memory"
    assert dataset.get_entry_metadata(1)["source_id"] == "mem_1"
    assert dataset.get_entry_metadata(2)["frame_idx"] == 1


def test_tar_archive_xsf_source_collection_disambiguates_duplicate_member_names(
    tmp_path: Path,
):
    """Duplicate archive member names should remain uniquely addressable."""
    structures = [_make_structure(0.0), _make_structure(1.0)]
    archive_path = _write_xsf_tar_bz2(
        tmp_path / "structures.tar.bz2",
        structures=structures,
        duplicate_member_names=True,
    )

    collection = TarArchiveXSFSourceCollection(archive_path)
    records = list(collection.iter_records())

    assert len(records) == 2
    assert records[0].source_id != records[1].source_id
    assert records[0].display_name != records[1].display_name
    assert records[0].display_name.endswith("@member=0")
    assert records[1].display_name.endswith("@member=1")
    assert records[0].load_structures()[0].energy == pytest.approx(0.0)
    assert records[1].load_structures()[0].energy == pytest.approx(1.0)


def test_hdf5_dataset_builds_from_tar_archive_xsf_sources(tmp_path: Path):
    """HDF5StructureDataset should build from the tar-backed XSF adapter."""
    archive_path = _write_xsf_tar_bz2(tmp_path / "structures.tar.bz2")
    descriptor = _make_descriptor()
    db_path = tmp_path / "tar_sources.h5"

    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=TarArchiveXSFSourceCollection(archive_path),
        mode="build",
    )
    dataset.build_database(show_progress=False, build_workers=2)

    assert len(dataset) == 2
    meta = dataset.get_entry_metadata(0)
    assert meta["source_id"].endswith("::member=0:dataset/structure-0.xsf")
    assert meta["frame_idx"] == 0
    assert meta["source_kind"] == "tar_xsf_member"
    assert meta["display_name"] == (
        "structures.tar.bz2:dataset/structure-0.xsf"
    )
    assert dataset.get_structure_identifier(0) == (
        "structures.tar.bz2:dataset/structure-0.xsf#frame=0"
    )


def test_hdf5_dataset_preserves_long_display_names_without_truncation(
    tmp_path: Path,
):
    """Long display names should round-trip exactly through HDF5 metadata."""
    descriptor = _make_descriptor()
    long_display_name = "archive.tar:" + ("member/" * 50) + "frame.xsf"
    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(tmp_path / "long_display_name.h5"),
        sources=RecordSourceCollection(
            [
                SourceRecord(
                    source_id="mem_0",
                    loader=lambda: _make_structure(0.0),
                    source_kind="memory",
                    display_name=long_display_name,
                )
            ]
        ),
        mode="build",
    )

    dataset.build_database(show_progress=False)

    meta = dataset.get_entry_metadata(0)
    assert meta["display_name"] == long_display_name
    assert dataset.get_structure_identifier(0) == (
        f"{long_display_name}#frame=0"
    )


def test_hdf5_dataset_rejects_overlong_display_names(
    tmp_path: Path,
):
    """Overlong source labels should fail fast instead of truncating."""
    descriptor = _make_descriptor()
    overlong_display_name = "x" * 3000
    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(tmp_path / "overlong_display_name.h5"),
        sources=RecordSourceCollection(
            [
                SourceRecord(
                    source_id="mem_0",
                    loader=lambda: _make_structure(0.0),
                    source_kind="memory",
                    display_name=overlong_display_name,
                )
            ]
        ),
        mode="build",
    )

    with pytest.raises(ValueError, match="display_name"):
        dataset.build_database(show_progress=False)


def test_hdf5_dataset_parallel_tar_build_keeps_duplicate_member_names_distinct(
    tmp_path: Path,
):
    """Duplicate archive members should remain distinct under parallel build."""
    archive_path = _write_xsf_tar_bz2(
        tmp_path / "duplicate_members.tar.bz2",
        structures=[_make_structure(0.0), _make_structure(1.0)],
        duplicate_member_names=True,
    )
    dataset = HDF5StructureDataset(
        descriptor=_make_descriptor(),
        database_file=str(tmp_path / "duplicate_members.h5"),
        sources=TarArchiveXSFSourceCollection(archive_path),
        mode="build",
    )

    dataset.build_database(show_progress=False, build_workers=2)

    meta0 = dataset.get_entry_metadata(0)
    meta1 = dataset.get_entry_metadata(1)
    assert meta0["source_id"] != meta1["source_id"]
    assert meta0["display_name"] != meta1["display_name"]
    assert str(meta0["display_name"]).endswith("@member=0")
    assert str(meta1["display_name"]).endswith("@member=1")


def test_hdf5_dataset_parallel_tar_build_matches_serial_persisted_caches(
    tmp_path: Path,
):
    """Parallel streamed tar builds should match serial cache payloads."""
    archive_path = _write_xsf_tar_bz2(
        tmp_path / "cache_compare.tar.bz2",
        structures=[_make_structure(0.0), _make_structure(1.0)],
    )
    descriptor = _make_descriptor()
    serial_db = tmp_path / "cache_compare_serial.h5"
    parallel_db = tmp_path / "cache_compare_parallel.h5"

    serial_dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=serial_db,
        sources=TarArchiveXSFSourceCollection(archive_path),
        mode="build",
    )
    parallel_dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=parallel_db,
        sources=TarArchiveXSFSourceCollection(archive_path),
        mode="build",
    )

    serial_dataset.build_database(
        show_progress=False,
        persist_features=True,
        persist_force_derivatives=True,
    )
    parallel_dataset.build_database(
        show_progress=False,
        build_workers=2,
        persist_features=True,
        persist_force_derivatives=True,
    )

    assert len(serial_dataset) == len(parallel_dataset) == 2
    for index in range(len(serial_dataset)):
        assert serial_dataset.get_entry_metadata(index) == (
            parallel_dataset.get_entry_metadata(index)
        )
        _sample_signatures_match(
            serial_dataset[index]["features"],
            parallel_dataset[index]["features"],
        )
        _sample_signatures_match(
            serial_dataset[index]["local_derivatives"],
            parallel_dataset[index]["local_derivatives"],
        )


def test_hdf5_dataset_parallel_tar_build_preserves_multiframe_member_order(
    tmp_path: Path,
):
    """Parallel tar builds should preserve member/frame ordering."""
    periodic_path = _fixture_path("periodic.xsf")
    trajectory_path = _fixture_path("trajectory.xsf")
    archive_path = _write_tar_bz2_members(
        tmp_path / "multiframe.tar.bz2",
        [
            ("dataset/periodic.xsf", periodic_path.read_bytes()),
            ("dataset/trajectory.xsf", trajectory_path.read_bytes()),
        ],
    )
    trajectory_atomic = AtomicStructure.from_file(trajectory_path)
    dataset = HDF5StructureDataset(
        descriptor=None,
        database_file=tmp_path / "multiframe_parallel.h5",
        sources=TarArchiveXSFSourceCollection(archive_path),
        mode="build",
    )

    dataset.build_database(show_progress=False, build_workers=2)

    assert len(dataset) == 1 + trajectory_atomic.nframes
    metadata = [dataset.get_entry_metadata(index) for index in range(len(dataset))]
    assert str(metadata[0]["source_id"]).endswith(
        "::member=0:dataset/periodic.xsf"
    )
    assert metadata[0]["frame_idx"] == 0
    for frame_idx, meta in enumerate(metadata[1:]):
        assert str(meta["source_id"]).endswith(
            "::member=1:dataset/trajectory.xsf"
        )
        assert meta["frame_idx"] == frame_idx


def test_hdf5_dataset_parallel_tar_build_failure_preserves_existing_database(
    tmp_path: Path,
):
    """Failed streamed tar builds should leave the prior database untouched."""
    initial_archive = _write_xsf_tar_bz2(
        tmp_path / "initial.tar.bz2",
        structures=[_make_structure(0.0), _make_structure(1.0)],
    )
    db_path = tmp_path / "preserve_existing.h5"
    initial_dataset = HDF5StructureDataset(
        descriptor=_make_descriptor(),
        database_file=db_path,
        sources=TarArchiveXSFSourceCollection(initial_archive),
        mode="build",
    )
    initial_dataset.build_database(show_progress=False, build_workers=2)
    initial_dataset.close()

    broken_archive = _write_tar_bz2_members(
        tmp_path / "broken.tar.bz2",
        [
            ("dataset/ok.xsf", _fixture_path("periodic.xsf").read_bytes()),
            ("dataset/bad.xsf", b"this is not a valid xsf payload"),
        ],
    )
    failing_dataset = HDF5StructureDataset(
        descriptor=None,
        database_file=db_path,
        sources=TarArchiveXSFSourceCollection(broken_archive),
        mode="build",
    )

    with pytest.raises(RuntimeError, match="Failed to load build source"):
        failing_dataset.build_database(show_progress=False, build_workers=2)

    load_dataset = HDF5StructureDataset(
        descriptor=_make_descriptor(),
        database_file=db_path,
        mode="load",
    )
    assert len(load_dataset) == 2
