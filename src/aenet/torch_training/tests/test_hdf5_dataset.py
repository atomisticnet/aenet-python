import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import tables
import torch

from aenet.formats.xsf import XSFParser
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    HDF5StructureDataset,
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
)
from aenet.torch_training.dataset import (
    StructureDataset,
    _build_force_graph_triplets,
)
from aenet.torch_training.sources import (
    RecordSourceCollection,
    SourceCapabilities,
    SourceRecord,
)
from aenet.torch_training.trainer import _collate_fn


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


def _write_pickled_build_inputs(
    directory: Path,
    payloads: list[object],
) -> list[str]:
    """Write serialized source payloads for process-based build tests."""
    directory.mkdir(parents=True, exist_ok=True)
    file_paths = []
    for index, payload in enumerate(payloads):
        path = directory / f"build_input_{index}.pkl"
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        file_paths.append(str(path))
    return file_paths


def _payload_loader(payload):
    """Create a record loader returning one in-memory payload."""

    def _load():
        return payload

    return _load


def _payload_source_collection(
    source_ids: list[str],
    payloads: list[Structure | list[Structure]],
    *,
    capabilities: SourceCapabilities | None = None,
) -> RecordSourceCollection:
    """Build a record-backed source collection for in-memory payloads."""
    records = [
        SourceRecord(
            source_id=source_id,
            loader=_payload_loader(payload),
            source_kind="test",
        )
        for source_id, payload in zip(source_ids, payloads, strict=True)
    ]
    return RecordSourceCollection(records, capabilities=capabilities)

def _pickled_source_collection(
    file_paths: list[str],
    *,
    capabilities: SourceCapabilities | None = None,
) -> RecordSourceCollection:
    """Build a source collection that loads payloads from pickle files."""

    def _make_loader(path: str):
        def _load():
            with Path(path).open("rb") as handle:
                payload = pickle.load(handle)
            if (
                isinstance(payload, tuple)
                and len(payload) == 2
                and payload[0] == "raise"
            ):
                raise RuntimeError(str(payload[1]))
            return payload

        return _load

    records = [
        SourceRecord(
            source_id=path,
            loader=_make_loader(path),
            source_kind="pickle",
        )
        for path in file_paths
    ]
    return RecordSourceCollection(records, capabilities=capabilities)


def _decode_table_string(value) -> str:
    """Decode PyTables string columns to plain Python strings."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_meta_rows(
    db_path: Path,
) -> list[tuple[str, int, str, str, bool, int, float, str]]:
    """Read the HDF5 metadata table for deterministic-order assertions."""
    with tables.open_file(str(db_path), mode="r") as h5:
        meta = h5.get_node("/entries/meta")
        rows = []
        for row in meta:  # type: ignore[assignment]
            rows.append(
                (
                    _decode_table_string(row["source_id"]),
                    int(row["frame_idx"]),
                    _decode_table_string(row["source_kind"]),
                    _decode_table_string(row["display_name"]),
                    bool(row["has_forces"]),
                    int(row["n_atoms"]),
                    float(row["energy"]),
                    _decode_table_string(row["name"]),
                )
            )
        return rows


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


def _build_legacy_v1_force_cache(
    db_path: Path,
    desc: ChebyshevDescriptor,
    structs: list[Structure],
) -> None:
    """Create a legacy v1 derivative-only HDF5 file for compatibility tests."""
    file_paths = [str(db_path.parent / f"s_{i}") for i in range(len(structs))]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )

    h5 = tables.open_file(
        str(db_path),
        mode="w",
        filters=ds._filters.to_tables_filters(),
    )
    try:
        entries_group = h5.create_group("/", "entries", "Serialized entries")
        vl_struct = h5.create_vlarray(
            entries_group,
            "structures",
            atom=tables.UInt8Atom(),
            title="Pickled Structures",
        )
        meta_table = h5.create_table(
            entries_group,
            "meta",
            description=ds._MetaRow,
        )
        ds._create_legacy_force_derivative_storage(h5)

        for entry_idx, struct in enumerate(structs):
            data_bytes = pickle.dumps(struct, protocol=pickle.HIGHEST_PROTOCOL)
            vl_struct.append(np.frombuffer(data_bytes, dtype=np.uint8))

            row = meta_table.row
            row["source_id"] = file_paths[entry_idx]
            row["frame_idx"] = 0
            row["source_kind"] = "legacy"
            row["display_name"] = ""
            row["has_forces"] = bool(struct.has_forces())
            row["n_atoms"] = int(struct.n_atoms)
            row["energy"] = float(struct.energy)
            row["name"] = str(struct.name) if struct.name is not None else ""
            row.append()

            if not struct.has_forces():
                continue

            positions = torch.from_numpy(struct.positions).to(desc.dtype)
            cell = (
                torch.from_numpy(struct.cell).to(desc.dtype)
                if struct.cell is not None
                else None
            )
            pbc = torch.from_numpy(struct.pbc) if struct.pbc is not None else None
            species_indices = torch.tensor(
                [desc.species_to_idx[s] for s in struct.species],
                dtype=torch.long,
            )
            graph_trip = _build_force_graph_triplets(
                descriptor=desc,
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
            _, local_derivatives = (
                desc.compute_features_and_local_derivatives_with_graph(
                    positions=positions,
                    species_indices=species_indices,
                    graph=graph_trip["graph"],
                    triplets=graph_trip["triplets"],
                    center_indices=None,
                )
            )
            ds._append_force_derivative_cache_entry(
                h5=h5,
                entry_idx=entry_idx,
                n_atoms=int(struct.n_atoms),
                local_derivatives=local_derivatives,
                node_paths=ds._legacy_force_derivative_node_paths(),
            )

        meta_table.flush()
        h5.flush()
    finally:
        h5.close()


def _make_incompatible_descriptor(dtype=torch.float64) -> ChebyshevDescriptor:
    """Return a descriptor that should reject an existing derivative cache."""
    return ChebyshevDescriptor(
        species=["H"],
        rad_order=1,
        rad_cutoff=2.5,
        ang_order=0,
        ang_cutoff=2.0,
        min_cutoff=0.1,
        device="cpu",
        dtype=dtype,
    )


def _make_arch(descriptor: ChebyshevDescriptor):
    # Single-species small network; output layer implicit linear(1).
    return {"H": [(4, "tanh")]}


def _make_multispecies_struct() -> Structure:
    """Create a small A/B structure that exercises typespin cache payloads."""
    pos = np.array(
        [[0.0, 0.0, 0.0],
         [0.9, 0.0, 0.0],
         [0.0, 0.8, 0.0]],
        dtype=np.float64,
    )
    species = ["A", "B", "A"]
    forces = np.zeros_like(pos)
    return Structure(positions=pos, species=species, energy=0.25, forces=forces)


def _make_multispecies_descriptor(dtype=torch.float64) -> ChebyshevDescriptor:
    """Return a descriptor with multi-species/typespin behavior enabled."""
    return ChebyshevDescriptor(
        species=["A", "B"],
        rad_order=1,
        rad_cutoff=2.0,
        ang_order=0,
        ang_cutoff=2.0,
        min_cutoff=0.1,
        device="cpu",
        dtype=dtype,
    )


def _make_tio2_struct() -> Structure:
    """Load one periodic TiO2 fixture with energies and forces."""
    root = Path(__file__).resolve().parents[4]
    xsf_path = root / "src" / "aenet" / "tests" / "data" / "xsf-TiO2"
    parser = XSFParser()
    parsed = parser.read(str(sorted(xsf_path.glob("*.xsf"))[0]))
    return Structure(
        positions=np.array(parsed.coords[-1], dtype=np.float64),
        species=[str(species) for species in parsed.types],
        energy=(
            float(parsed.energy[-1])
            if (parsed.energy and parsed.energy[-1] is not None)
            else 0.0
        ),
        forces=(
            np.array(parsed.forces[-1], dtype=np.float64)
            if (parsed.forces and parsed.forces[-1] is not None)
            else None
        ),
        cell=(
            np.array(parsed.avec[-1], dtype=np.float64)
            if parsed.pbc else None
        ),
        pbc=(
            np.array([True, True, True], dtype=bool)
            if parsed.pbc else None
        ),
        name=Path(parsed.path).name if getattr(parsed, "path", None) else None,
    )


def _make_tio2_descriptor(dtype=torch.float64) -> ChebyshevDescriptor:
    """Return the compact TiO2 descriptor used by torch-training smoke tests."""
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


def _make_energy_only_struct() -> Structure:
    """Create a structure without force labels for index-table coverage."""
    pos = np.array(
        [[0.0, 0.0, 0.0],
         [0.9, 0.0, 0.0]],
        dtype=np.float64,
    )
    species = ["H", "H"]
    return Structure(positions=pos, species=species, energy=0.1, forces=None)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("struct_factory", "descriptor_factory"),
    [
        (lambda: _make_struct(0), _make_descriptor),
        (_make_multispecies_struct, _make_multispecies_descriptor),
        (_make_tio2_struct, _make_tio2_descriptor),
    ],
    ids=["single_species", "multi_species", "periodic_tio2"],
)
def test_raw_feature_paths_are_equivalent(
    struct_factory,
    descriptor_factory,
):
    """
    Energy-view and graph-based force-view features should match.

    This locks in the Issue 14 design assumption that one raw ``(N, F)``
    tensor can serve as the canonical persisted-feature payload across the
    current feature and sparse-derivative paths.
    """
    struct = struct_factory()
    descriptor = descriptor_factory(dtype=torch.float64)

    positions = torch.as_tensor(struct.positions, dtype=descriptor.dtype)
    cell = (
        torch.as_tensor(struct.cell, dtype=descriptor.dtype)
        if struct.cell is not None
        else None
    )
    pbc = torch.as_tensor(struct.pbc) if struct.pbc is not None else None
    species_indices = torch.tensor(
        [descriptor.species_to_idx[species] for species in struct.species],
        dtype=torch.long,
    )
    graph_triplets = _build_force_graph_triplets(
        descriptor=descriptor,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )

    features_energy = descriptor.forward_from_positions(
        positions,
        struct.species,
        cell,
        pbc,
    )
    features_graph = descriptor.forward_with_graph(
        positions,
        species_indices,
        graph_triplets["graph"],
        graph_triplets["triplets"],
    )
    features_sparse, _ = (
        descriptor.compute_features_and_local_derivatives_with_graph(
            positions=positions,
            species_indices=species_indices,
            graph=graph_triplets["graph"],
            triplets=graph_triplets["triplets"],
            center_indices=None,
        )
    )

    assert torch.allclose(
        features_energy,
        features_graph,
        rtol=1e-8,
        atol=1e-10,
    )
    assert torch.allclose(
        features_energy,
        features_sparse,
        rtol=1e-8,
        atol=1e-10,
    )


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
        sources=_payload_source_collection(file_paths, structs),
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
def test_hdf5_parallel_build_preserves_order_for_multiframe_inputs(
    tmp_path: Path,
):
    """Parallel builds should preserve deterministic path/frame ordering."""
    struct_a = _make_struct(0)
    struct_a.name = "serial-a"
    struct_b0 = _make_struct(1)
    struct_b0.name = "multi-b0"
    struct_b1 = _make_struct(2)
    struct_b1.name = "multi-b1"
    struct_c = _make_energy_only_struct()
    struct_c.name = "serial-c"

    file_paths = _write_pickled_build_inputs(
        tmp_path,
        [struct_a, [struct_b0, struct_b1], struct_c],
    )
    db_path = tmp_path / "parallel_ordered.h5"
    ds = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        sources=_pickled_source_collection(file_paths),
        mode="build",
    )

    ds.build_database(show_progress=False, build_workers=2)

    expected_rows = [
        (
            file_paths[0],
            0,
            "pickle",
            "",
            True,
            3,
            float(struct_a.energy),
            "serial-a",
        ),
        (
            file_paths[1],
            0,
            "pickle",
            "",
            True,
            3,
            float(struct_b0.energy),
            "multi-b0",
        ),
        (
            file_paths[1],
            1,
            "pickle",
            "",
            True,
            3,
            float(struct_b1.energy),
            "multi-b1",
        ),
        (
            file_paths[2],
            0,
            "pickle",
            "",
            False,
            2,
            float(struct_c.energy),
            "serial-c",
        ),
    ]
    assert len(ds) == 4
    assert _read_meta_rows(db_path) == expected_rows
    assert [ds.get_structure(i).name for i in range(len(ds))] == [
        "serial-a",
        "multi-b0",
        "multi-b1",
        "serial-c",
    ]


@pytest.mark.cpu
def test_hdf5_parallel_build_matches_serial_for_persisted_caches(
    tmp_path: Path,
):
    """Parallel persisted-cache builds should match serial output exactly."""
    structs = [_make_struct(0), _make_struct(1), _make_struct(2)]
    for index, struct in enumerate(structs):
        struct.name = f"struct-{index}"

    file_paths = _write_pickled_build_inputs(tmp_path / "inputs", structs)
    serial_path = tmp_path / "serial_cache_build.h5"
    parallel_path = tmp_path / "parallel_cache_build.h5"

    serial_ds = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(serial_path),
        sources=_pickled_source_collection(file_paths),
        mode="build",
    )
    parallel_ds = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(parallel_path),
        sources=_pickled_source_collection(file_paths),
        mode="build",
    )

    serial_ds.build_database(
        show_progress=False,
        persist_features=True,
        persist_force_derivatives=True,
    )
    parallel_ds.build_database(
        show_progress=False,
        build_workers=2,
        persist_features=True,
        persist_force_derivatives=True,
    )

    assert _read_meta_rows(serial_path) == _read_meta_rows(parallel_path)
    for idx in range(len(serial_ds)):
        serial_features = serial_ds.load_persisted_features(idx)
        parallel_features = parallel_ds.load_persisted_features(idx)
        serial_derivatives = serial_ds.load_persisted_force_derivatives(idx)
        parallel_derivatives = parallel_ds.load_persisted_force_derivatives(idx)

        assert serial_features is not None
        assert parallel_features is not None
        assert torch.equal(serial_features, parallel_features)
        assert serial_derivatives is not None
        assert parallel_derivatives is not None
        for key in ("center_idx", "neighbor_idx"):
            assert torch.equal(
                serial_derivatives["radial"][key],
                parallel_derivatives["radial"][key],
            )
        assert torch.allclose(
            serial_derivatives["radial"]["dG_drij"],
            parallel_derivatives["radial"]["dG_drij"],
        )
        for key in ("center_idx", "neighbor_j_idx", "neighbor_k_idx"):
            assert torch.equal(
                serial_derivatives["angular"][key],
                parallel_derivatives["angular"][key],
            )
        for key in ("grads_i", "grads_j", "grads_k"):
            assert torch.allclose(
                serial_derivatives["angular"][key],
                parallel_derivatives["angular"][key],
            )


@pytest.mark.cpu
def test_hdf5_parallel_build_failure_preserves_existing_database(
    tmp_path: Path,
):
    """Failed parallel rebuilds should not replace an existing database."""
    initial_struct = _make_struct(0)
    ok_inputs = _write_pickled_build_inputs(tmp_path / "ok", [initial_struct])
    db_path = tmp_path / "atomic_parallel_build.h5"

    ds_ok = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        sources=_pickled_source_collection(ok_inputs),
        mode="build",
    )
    ds_ok.build_database(show_progress=False)

    failing_inputs = _write_pickled_build_inputs(
        tmp_path / "fail",
        [_make_struct(1), ("raise", "boom")],
    )
    ds_fail = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        sources=_pickled_source_collection(failing_inputs),
        mode="build",
    )

    with pytest.raises(RuntimeError, match="Failed to load build source"):
        ds_fail.build_database(show_progress=False, build_workers=2)

    ds_load = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        mode="load",
    )
    assert len(ds_load) == 1
    assert ds_load.get_structure(0).energy == pytest.approx(initial_struct.energy)


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
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
        compression="zlib",
        compression_level=5,
    )
    ds.build_database(show_progress=False)

    # Trainer
    arch = _make_arch(desc)
    pot = TorchANNPotential(arch=arch, descriptor=desc)

    # Force-supervised single-epoch smoke test
    ckpt_dir = tmp_path / "ckpts"
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        force_fraction=1.0,
        force_sampling="fixed",
        memory_mode="cpu",
        device="cpu",
        num_workers=0,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=1,
        save_best=False,
        use_scheduler=False,
    )

    history = pot.train(
        dataset=ds,
        config=cfg,
    )

    # Verify history attributes and checkpoint dir created
    assert "RMSE_force_train" in history.errors.columns
    assert len(history.errors["RMSE_force_train"]) == 1
    assert not math.isnan(history.errors["RMSE_force_train"].iloc[0])
    assert ckpt_dir.exists()


@pytest.mark.cpu
def test_hdf5_force_derivative_cache_round_trip(tmp_path: Path):
    """Schema v2 derivative-only caches should round-trip exactly."""
    structs = [_make_struct(0), _make_struct(1)]
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structs))]
    for p in file_paths:
        Path(p).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_cache.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_force_derivatives=True)

    assert ds.has_persisted_force_derivatives()
    assert ds.has_persisted_features() is False
    info = ds.get_force_derivative_cache_info()
    assert info is not None
    assert ds.get_persisted_feature_cache_info() is None
    assert info["schema_version"] == 2
    assert info["cache_format"] == "aenet.torch_training.cache.v2"
    assert info["payload_format"] == (
        "aenet.torch_training.local_derivatives.v1"
    )
    assert info["contains_features"] is False
    assert info["contains_force_derivatives"] is True
    assert info["n_radial_features"] == 2
    assert info["n_angular_features"] == 1
    assert info["multi"] is False

    sample = ds[0]
    _, expected = desc.compute_features_and_local_derivatives_with_graph(
        positions=sample["positions"],
        species_indices=sample["species_indices"],
        graph=sample["graph"],
        triplets=sample["triplets"],
        center_indices=None,
    )
    cached = ds.load_persisted_force_derivatives(0)
    assert cached is not None

    assert torch.equal(
        cached["radial"]["center_idx"], expected["radial"]["center_idx"]
    )
    assert torch.equal(
        cached["radial"]["neighbor_idx"], expected["radial"]["neighbor_idx"]
    )
    assert torch.allclose(
        cached["radial"]["dG_drij"], expected["radial"]["dG_drij"]
    )
    assert cached["radial"]["neighbor_typespin"] is None

    assert torch.equal(
        cached["angular"]["center_idx"], expected["angular"]["center_idx"]
    )
    assert torch.equal(
        cached["angular"]["neighbor_j_idx"],
        expected["angular"]["neighbor_j_idx"],
    )
    assert torch.equal(
        cached["angular"]["neighbor_k_idx"],
        expected["angular"]["neighbor_k_idx"],
    )
    assert torch.allclose(
        cached["angular"]["grads_i"], expected["angular"]["grads_i"]
    )
    assert torch.allclose(
        cached["angular"]["grads_j"], expected["angular"]["grads_j"]
    )
    assert torch.allclose(
        cached["angular"]["grads_k"], expected["angular"]["grads_k"]
    )
    assert cached["angular"]["triplet_typespin"] is None


@pytest.mark.cpu
def test_hdf5_persisted_feature_cache_round_trip(tmp_path: Path):
    """Persisted raw features should round-trip through the schema v2 helpers."""
    structs = [_make_struct(0), _make_struct(1)]
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structs))]
    for p in file_paths:
        Path(p).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_feature_cache.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_features=True)

    assert ds.has_persisted_features() is True
    assert ds.has_persisted_force_derivatives() is False

    info = ds.get_persisted_feature_cache_info()
    assert info is not None
    assert info["schema_version"] == 2
    assert info["cache_format"] == "aenet.torch_training.cache.v2"
    assert info["contains_features"] is True
    assert info["contains_force_derivatives"] is False
    assert info["storage_dtype"] == "float64"

    sample = ds[0]
    expected = desc.forward_from_positions(
        sample["positions"],
        sample["species"],
        sample["cell"],
        sample["pbc"],
    )
    cached = ds.load_persisted_features(0)
    assert cached is not None
    assert cached.dtype == torch.float64
    assert torch.allclose(cached, expected)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("struct_factory", "descriptor_factory", "build_dtype", "load_dtype", "atol"),
    [
        (lambda: _make_struct(0), _make_descriptor, torch.float64, torch.float64, 0.0),
        (_make_multispecies_struct, _make_multispecies_descriptor, torch.float64, torch.float64, 0.0),
        (_make_tio2_struct, _make_tio2_descriptor, torch.float64, torch.float64, 1.0e-12),
        (lambda: _make_struct(0), _make_descriptor, torch.float64, torch.float32, 1.0e-6),
        (lambda: _make_struct(0), _make_descriptor, torch.float32, torch.float64, 1.0e-6),
    ],
    ids=[
        "single_species_float64_exact",
        "multi_species_float64_exact",
        "periodic_tio2_float64",
        "float64_to_float32_cast",
        "float32_to_float64_cast",
    ],
)
def test_hdf5_persisted_feature_cache_matches_direct_features(
    tmp_path: Path,
    struct_factory,
    descriptor_factory,
    build_dtype: torch.dtype,
    load_dtype: torch.dtype,
    atol: float,
):
    """Persisted raw features should match direct features under active dtype semantics."""
    struct = struct_factory()
    file_path = tmp_path / "s_0"
    file_path.write_text("placeholder", encoding="utf-8")
    db_path = tmp_path / "structures_feature_cache_direct_match.h5"

    ds_build = HDF5StructureDataset(
        descriptor=descriptor_factory(dtype=build_dtype),
        database_file=str(db_path),
        sources=_payload_source_collection([str(file_path)], [struct]),
        mode="build",
    )
    ds_build.build_database(show_progress=False, persist_features=True)

    ds_load = HDF5StructureDataset(
        descriptor=descriptor_factory(dtype=load_dtype),
        database_file=str(db_path),
        mode="load",
    )
    cached = ds_load.load_persisted_features(0)

    positions = torch.as_tensor(struct.positions, dtype=load_dtype)
    cell = (
        torch.as_tensor(struct.cell, dtype=load_dtype)
        if struct.cell is not None
        else None
    )
    pbc = torch.as_tensor(struct.pbc) if struct.pbc is not None else None
    expected = ds_load.descriptor.forward_from_positions(
        positions,
        struct.species,
        cell,
        pbc,
    )

    assert cached is not None
    assert cached.dtype == load_dtype
    if atol == 0.0:
        assert torch.equal(cached, expected)
    else:
        assert torch.allclose(cached, expected, atol=atol, rtol=0.0)


@pytest.mark.cpu
def test_hdf5_energy_materialization_prefers_runtime_then_persisted_features(
    tmp_path: Path,
    monkeypatch,
):
    """Energy-view samples should prefer runtime cache over persisted features."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_feature_cache_runtime_precedence.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_features=True)

    def _fail_forward(*args, **kwargs):
        raise AssertionError("energy features should not be recomputed")

    monkeypatch.setattr(ds.descriptor, "forward_from_positions", _fail_forward)

    class _CacheState:
        def __init__(self) -> None:
            self.feature_cache = {}
            self.neighbor_cache = {}
            self.graph_cache = {}

    cache_state = _CacheState()
    sample = ds.materialize_sample(
        0,
        use_forces=False,
        cache_state=cache_state,
        cache_features=True,
    )

    assert 0 in cache_state.feature_cache
    assert torch.allclose(sample["features"], cache_state.feature_cache[0])

    def _fail_persisted_load(idx: int):
        raise AssertionError("runtime cache should take precedence")

    monkeypatch.setattr(ds, "load_persisted_features", _fail_persisted_load)
    sample_cached = ds.materialize_sample(
        0,
        use_forces=False,
        cache_state=cache_state,
        cache_features=True,
    )

    assert torch.allclose(sample_cached["features"], sample["features"])


@pytest.mark.cpu
def test_hdf5_energy_materialization_recomputes_when_persisted_features_absent(
    tmp_path: Path,
    monkeypatch,
):
    """Energy-view samples should fall back cleanly when no feature cache exists."""
    structs = [_make_energy_only_struct()]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_no_feature_cache_energy.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False)

    calls = {"forward_from_positions": 0}
    original_forward = ds.descriptor.forward_from_positions

    def _wrapped_forward(*args, **kwargs):
        calls["forward_from_positions"] += 1
        return original_forward(*args, **kwargs)

    def _fail_persisted_load(idx: int):
        raise AssertionError("no persisted feature cache should be loaded")

    monkeypatch.setattr(ds.descriptor, "forward_from_positions", _wrapped_forward)
    monkeypatch.setattr(ds, "load_persisted_features", _fail_persisted_load)

    sample = ds.materialize_sample(0, use_forces=False)
    expected = original_forward(
        sample["positions"],
        sample["species"],
        sample["cell"],
        sample["pbc"],
    )

    assert ds.has_persisted_features() is False
    assert calls["forward_from_positions"] == 1
    assert torch.equal(sample["features"], expected)


@pytest.mark.cpu
def test_hdf5_energy_materialization_uses_single_neighbor_build_on_cache_miss(
    tmp_path: Path,
    monkeypatch,
):
    """Neighbor-cache misses should not recompute energy features twice."""
    structs = [_make_energy_only_struct()]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_neighbor_cache_single_pass.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False)

    calls = {
        "forward_from_positions": 0,
        "featurize_with_neighbor_info": 0,
    }
    original_forward = ds.descriptor.forward_from_positions
    original_featurize = ds.descriptor.featurize_with_neighbor_info

    def _wrapped_forward(*args, **kwargs):
        calls["forward_from_positions"] += 1
        return original_forward(*args, **kwargs)

    def _wrapped_featurize(*args, **kwargs):
        calls["featurize_with_neighbor_info"] += 1
        return original_featurize(*args, **kwargs)

    class _CacheState:
        def __init__(self) -> None:
            self.feature_cache = {}
            self.neighbor_cache = {}
            self.graph_cache = {}

    monkeypatch.setattr(ds.descriptor, "forward_from_positions", _wrapped_forward)
    monkeypatch.setattr(
        ds.descriptor,
        "featurize_with_neighbor_info",
        _wrapped_featurize,
    )

    cache_state = _CacheState()
    sample = ds.materialize_sample(
        0,
        use_forces=False,
        cache_state=cache_state,
        cache_neighbors=True,
    )

    assert sample["features"] is not None
    assert calls["forward_from_positions"] == 0
    assert calls["featurize_with_neighbor_info"] == 1
    assert 0 in cache_state.neighbor_cache


@pytest.mark.cpu
def test_structure_and_hdf5_materialize_sample_match_without_persisted_payloads(
    tmp_path: Path,
):
    """Shared helpers should keep in-memory and HDF5 samples aligned."""
    structs = [_make_struct(0)]
    descriptor = _make_descriptor(dtype=torch.float64)
    structure_ds = StructureDataset(structures=structs, descriptor=descriptor)

    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")
    db_path = tmp_path / "structures_materialization_parity.h5"
    hdf5_ds = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    hdf5_ds.build_database(show_progress=False)

    class _CacheState:
        def __init__(self) -> None:
            self.feature_cache = {}
            self.neighbor_cache = {}
            self.graph_cache = {}

    energy_state_a = _CacheState()
    energy_state_b = _CacheState()
    energy_sample = structure_ds.materialize_sample(
        0,
        use_forces=False,
        cache_state=energy_state_a,
        cache_features=True,
        cache_neighbors=True,
    )
    energy_hdf5 = hdf5_ds.materialize_sample(
        0,
        use_forces=False,
        cache_state=energy_state_b,
        cache_features=True,
        cache_neighbors=True,
    )

    assert torch.allclose(energy_sample["features"], energy_hdf5["features"])
    assert torch.equal(
        energy_sample["species_indices"],
        energy_hdf5["species_indices"],
    )
    assert energy_sample["graph"] is None
    assert energy_hdf5["graph"] is None
    assert energy_sample["triplets"] is None
    assert energy_hdf5["triplets"] is None

    force_state_a = _CacheState()
    force_state_b = _CacheState()
    force_sample = structure_ds.materialize_sample(
        0,
        use_forces=True,
        cache_state=force_state_a,
        cache_force_triplets=True,
    )
    force_hdf5 = hdf5_ds.materialize_sample(
        0,
        use_forces=True,
        cache_state=force_state_b,
        cache_force_triplets=True,
        load_local_derivatives=True,
    )

    assert force_hdf5["local_derivatives"] is None
    assert torch.allclose(force_sample["features"], force_hdf5["features"])
    assert torch.equal(
        force_sample["graph"]["center_ptr"],
        force_hdf5["graph"]["center_ptr"],
    )
    assert torch.equal(
        force_sample["graph"]["nbr_idx"],
        force_hdf5["graph"]["nbr_idx"],
    )
    assert torch.allclose(force_sample["graph"]["r_ij"], force_hdf5["graph"]["r_ij"])
    assert torch.allclose(force_sample["graph"]["d_ij"], force_hdf5["graph"]["d_ij"])
    assert torch.equal(force_sample["triplets"]["tri_i"], force_hdf5["triplets"]["tri_i"])
    assert torch.equal(force_sample["triplets"]["tri_j"], force_hdf5["triplets"]["tri_j"])
    assert torch.equal(force_sample["triplets"]["tri_k"], force_hdf5["triplets"]["tri_k"])
    assert torch.equal(
        force_sample["triplets"]["tri_j_local"],
        force_hdf5["triplets"]["tri_j_local"],
    )
    assert torch.equal(
        force_sample["triplets"]["tri_k_local"],
        force_hdf5["triplets"]["tri_k_local"],
    )


@pytest.mark.cpu
def test_hdf5_combined_feature_and_derivative_cache_round_trip(tmp_path: Path):
    """Schema v2 should support feature and derivative sections together."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_combined_cache.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(
        show_progress=False,
        persist_features=True,
        persist_force_derivatives=True,
    )

    feature_info = ds.get_persisted_feature_cache_info()
    derivative_info = ds.get_force_derivative_cache_info()
    assert feature_info is not None
    assert derivative_info is not None
    assert feature_info["contains_features"] is True
    assert feature_info["contains_force_derivatives"] is True
    assert derivative_info["schema_version"] == 2

    sample = ds[0]
    cached_features = ds.load_persisted_features(0)
    cached_derivatives = ds.load_persisted_force_derivatives(0)

    assert cached_features is not None
    assert cached_derivatives is not None
    assert torch.allclose(cached_features, sample["features"])
    assert sample["graph"] is None
    assert sample["triplets"] is None
    assert torch.equal(
        cached_derivatives["radial"]["center_idx"],
        sample["local_derivatives"]["radial"]["center_idx"],
    )


@pytest.mark.cpu
def test_hdf5_force_materialization_uses_persisted_features_without_derivatives(
    tmp_path: Path,
    monkeypatch,
):
    """Force-view samples should reuse persisted features and still build graphs."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_uses_persisted_features.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_features=True)

    def _fail_forward(*args, **kwargs):
        raise AssertionError("force features should not be recomputed")

    monkeypatch.setattr(ds.descriptor, "forward_with_graph", _fail_forward)

    sample = ds.materialize_sample(
        0,
        use_forces=True,
        load_local_derivatives=True,
    )
    cached = ds.load_persisted_features(0)

    assert cached is not None
    assert torch.allclose(sample["features"], cached)
    assert sample["graph"] is not None
    assert sample["triplets"] is not None
    assert sample["local_derivatives"] is None


@pytest.mark.cpu
def test_hdf5_force_materialization_recomputes_features_when_feature_cache_absent(
    tmp_path: Path,
    monkeypatch,
):
    """Force-view samples should rebuild features when only derivatives are persisted."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_no_feature_cache_force.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_force_derivatives=True)

    calls = {"forward_with_graph": 0}
    original_forward = ds.descriptor.forward_with_graph

    def _wrapped_forward(*args, **kwargs):
        calls["forward_with_graph"] += 1
        return original_forward(*args, **kwargs)

    def _fail_persisted_load(idx: int):
        raise AssertionError("no persisted feature cache should be loaded")

    monkeypatch.setattr(ds.descriptor, "forward_with_graph", _wrapped_forward)
    monkeypatch.setattr(ds, "load_persisted_features", _fail_persisted_load)

    sample = ds.materialize_sample(
        0,
        use_forces=True,
        load_local_derivatives=True,
    )
    expected = original_forward(
        sample["positions"],
        sample["species_indices"],
        sample["graph"],
        sample["triplets"],
    )

    assert ds.has_persisted_features() is False
    assert sample["local_derivatives"] is not None
    assert sample["graph"] is not None
    assert sample["triplets"] is not None
    assert calls["forward_with_graph"] == 1
    assert torch.equal(sample["features"], expected)


@pytest.mark.cpu
def test_hdf5_load_recovers_persisted_descriptor_without_explicit_descriptor(
    tmp_path: Path,
):
    """Load mode should recover a persisted descriptor when none is provided."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_descriptor_manifest.h5"

    ds_build = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds_build.build_database(show_progress=False, persist_descriptor=True)

    ds_load = HDF5StructureDataset(
        descriptor=None,
        database_file=str(db_path),
        mode="load",
    )

    manifest = ds_load.get_descriptor_manifest()
    recovered = ds_load.load_persisted_descriptor()
    assert ds_load.has_persisted_descriptor() is True
    assert manifest is not None
    assert ds_load.descriptor is not None
    assert recovered is not None
    assert manifest["descriptor_class"].endswith("ChebyshevDescriptor")
    assert list(ds_load.descriptor.species) == ["H"]
    assert list(recovered.species) == ["H"]
    assert ds_load.descriptor.rad_order == 1
    assert ds_load.descriptor.ang_order == 0

    sample = ds_load[0]
    assert sample["features"].shape == (3, 3)
    assert sample["species_indices"].tolist() == [0, 0, 0]


@pytest.mark.cpu
def test_hdf5_feature_cache_load_recovers_descriptor_automatically(
    tmp_path: Path,
):
    """Feature-cache loads should recover the descriptor automatically."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_feature_cache_recover.h5"

    ds_build = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds_build.build_database(show_progress=False, persist_features=True)

    ds_load = HDF5StructureDataset(
        descriptor=None,
        database_file=str(db_path),
        mode="load",
    )
    cached = ds_load.load_persisted_features(0)

    assert ds_load.has_persisted_descriptor() is True
    assert ds_load.has_persisted_features() is True
    assert cached is not None
    assert ds_load.descriptor is not None
    assert list(ds_load.descriptor.species) == ["H"]
    assert cached.shape == (3, 3)


@pytest.mark.cpu
def test_hdf5_force_cache_load_recovers_descriptor_automatically(
    tmp_path: Path,
):
    """Force-cache loads should recover the descriptor automatically."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_cache_recover.h5"

    ds_build = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds_build.build_database(show_progress=False, persist_force_derivatives=True)

    ds_load = HDF5StructureDataset(
        descriptor=None,
        database_file=str(db_path),
        mode="load",
    )
    cached = ds_load.load_persisted_force_derivatives(0)
    sample = ds_load[0]

    assert ds_load.has_persisted_descriptor() is True
    assert cached is not None
    assert sample["local_derivatives"] is not None
    assert torch.equal(
        sample["local_derivatives"]["radial"]["center_idx"],
        cached["radial"]["center_idx"],
    )


@pytest.mark.cpu
def test_hdf5_persisted_feature_cache_rejects_incompatible_descriptor(
    tmp_path: Path,
):
    """Loading persisted features with different descriptor settings should fail."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    db_path = tmp_path / "structures_feature_cache_mismatch.h5"
    ds_build = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds_build.build_database(show_progress=False, persist_features=True)

    ds_load = HDF5StructureDataset(
        descriptor=None,
        database_file=str(db_path),
        mode="load",
    )
    ds_load.descriptor = _make_incompatible_descriptor(dtype=torch.float64)

    with pytest.raises(RuntimeError, match="incompatible"):
        ds_load.load_persisted_features(0)


@pytest.mark.cpu
def test_hdf5_materialization_rejects_incompatible_persisted_feature_descriptor(
    tmp_path: Path,
):
    """Public sample materialization should fail clearly on feature-cache mismatch."""
    structs = [_make_energy_only_struct()]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    db_path = tmp_path / "structures_feature_cache_materialize_mismatch.h5"
    ds_build = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds_build.build_database(show_progress=False, persist_features=True)

    ds_load = HDF5StructureDataset(
        descriptor=None,
        database_file=str(db_path),
        mode="load",
    )
    ds_load.descriptor = _make_incompatible_descriptor(dtype=torch.float64)

    with pytest.raises(RuntimeError, match="incompatible persisted cache"):
        ds_load.materialize_sample(0, use_forces=False)


@pytest.mark.cpu
def test_hdf5_persisted_feature_cache_casts_to_descriptor_dtype(tmp_path: Path):
    """Persisted features should cast to the active descriptor dtype on load."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    db_path = tmp_path / "structures_feature_cache_cast.h5"
    ds_build = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds_build.build_database(show_progress=False, persist_features=True)

    ds_load = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float32),
        database_file=str(db_path),
        mode="load",
    )
    cached = ds_load.load_persisted_features(0)
    sample = ds_load.materialize_sample(0, use_forces=False)
    expected = ds_load.descriptor.forward_from_positions(
        sample["positions"],
        sample["species"],
        sample["cell"],
        sample["pbc"],
    )

    assert cached is not None
    assert cached.dtype == torch.float32
    assert sample["features"].dtype == torch.float32
    assert torch.allclose(cached, expected, atol=1.0e-6, rtol=0.0)
    assert torch.allclose(sample["features"], expected, atol=1.0e-6, rtol=0.0)


@pytest.mark.cpu
def test_hdf5_getitem_exposes_persisted_force_derivatives(tmp_path: Path):
    """Force-supervised HDF5 samples should expose cached derivatives lazily."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_cache_getitem.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_force_derivatives=True)

    sample = ds[0]
    cached = ds.load_persisted_force_derivatives(0)

    assert sample["local_derivatives"] is not None
    assert cached is not None
    assert torch.equal(
        sample["local_derivatives"]["radial"]["center_idx"],
        cached["radial"]["center_idx"],
    )
    assert torch.allclose(
        sample["local_derivatives"]["radial"]["dG_drij"],
        cached["radial"]["dG_drij"],
    )
    assert torch.equal(
        sample["local_derivatives"]["angular"]["center_idx"],
        cached["angular"]["center_idx"],
    )


@pytest.mark.cpu
def test_hdf5_collate_batches_persisted_force_derivatives(tmp_path: Path):
    """Collation should batch cached local derivatives with atom offsets."""
    structs = [_make_struct(0), _make_struct(1)]
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structs))]
    for p in file_paths:
        Path(p).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_cache_collate.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_force_derivatives=True)

    sample0 = ds[0]
    sample1 = ds[1]
    collated = _collate_fn([sample0, sample1])

    local_derivatives_f = collated["local_derivatives_f"]
    assert local_derivatives_f is not None
    assert collated["features_f"] is not None

    offset = sample0["n_atoms"]
    expected_radial_center = torch.cat(
        [
            sample0["local_derivatives"]["radial"]["center_idx"],
            sample1["local_derivatives"]["radial"]["center_idx"] + offset,
        ]
    )
    expected_radial_neighbor = torch.cat(
        [
            sample0["local_derivatives"]["radial"]["neighbor_idx"],
            sample1["local_derivatives"]["radial"]["neighbor_idx"] + offset,
        ]
    )
    expected_radial_grads = torch.cat(
        [
            sample0["local_derivatives"]["radial"]["dG_drij"],
            sample1["local_derivatives"]["radial"]["dG_drij"],
        ],
        dim=0,
    )
    expected_angular_center = torch.cat(
        [
            sample0["local_derivatives"]["angular"]["center_idx"],
            sample1["local_derivatives"]["angular"]["center_idx"] + offset,
        ]
    )

    assert torch.equal(
        local_derivatives_f["radial"]["center_idx"],
        expected_radial_center,
    )
    assert torch.equal(
        local_derivatives_f["radial"]["neighbor_idx"],
        expected_radial_neighbor,
    )
    assert torch.allclose(
        local_derivatives_f["radial"]["dG_drij"],
        expected_radial_grads,
    )
    assert torch.equal(
        local_derivatives_f["angular"]["center_idx"],
        expected_angular_center,
    )


@pytest.mark.cpu
def test_hdf5_collate_allows_derivative_backed_force_batches_without_graphs(
    tmp_path: Path,
):
    """Collation should accept force samples backed only by local derivatives."""
    structs = [_make_struct(0), _make_struct(1)]
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structs))]
    for p in file_paths:
        Path(p).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_combined_cache_collate.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(
        show_progress=False,
        persist_features=True,
        persist_force_derivatives=True,
    )

    sample0 = ds[0]
    sample1 = ds[1]
    collated = _collate_fn([sample0, sample1])

    assert sample0["graph"] is None
    assert sample1["graph"] is None
    assert collated["features_f"] is not None
    assert collated["local_derivatives_f"] is not None
    assert collated["graph_f"] is None
    assert collated["triplets_f"] is None


@pytest.mark.cpu
def test_hdf5_force_derivative_cache_rejects_incompatible_descriptor(
    tmp_path: Path,
):
    """Loading a derivative cache with different descriptor settings should fail."""
    structs = [_make_struct(0), _make_struct(1)]
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structs))]
    for p in file_paths:
        Path(p).write_text("placeholder", encoding="utf-8")

    build_desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_cache_mismatch.h5"

    ds_build = HDF5StructureDataset(
        descriptor=build_desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds_build.build_database(show_progress=False, persist_force_derivatives=True)

    with pytest.raises(RuntimeError, match="incompatible"):
        HDF5StructureDataset(
            descriptor=_make_incompatible_descriptor(dtype=torch.float64),
            database_file=str(db_path),
            mode="load",
        )


@pytest.mark.cpu
def test_hdf5_legacy_v1_force_derivative_cache_still_loads(tmp_path: Path):
    """Legacy v1 derivative-only files should remain readable."""
    structs = [_make_struct(0)]
    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_cache_legacy_v1.h5"

    _build_legacy_v1_force_cache(db_path=db_path, desc=desc, structs=structs)

    ds = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        mode="load",
    )
    info = ds.get_force_derivative_cache_info()
    cached = ds.load_persisted_force_derivatives(0)

    assert info is not None
    assert info["schema_version"] == 1
    assert info["payload_format"] == (
        "aenet.torch_training.local_derivatives.v1"
    )
    assert ds.get_persisted_feature_cache_info() is None
    assert ds.has_persisted_features() is False
    assert cached is not None


@pytest.mark.cpu
def test_hdf5_force_derivative_cache_round_trip_multispecies_typespin(
    tmp_path: Path,
):
    """Multi-species caches should persist and reload typespin payloads."""
    structs = [_make_multispecies_struct()]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_multispecies_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_cache_multispecies.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_force_derivatives=True)

    info = ds.get_force_derivative_cache_info()
    assert info is not None
    assert info["multi"] is True

    sample = ds[0]
    _, expected = desc.compute_features_and_local_derivatives_with_graph(
        positions=sample["positions"],
        species_indices=sample["species_indices"],
        graph=sample["graph"],
        triplets=sample["triplets"],
        center_indices=None,
    )
    cached = ds.load_persisted_force_derivatives(0)
    assert cached is not None

    assert cached["radial"]["neighbor_typespin"] is not None
    assert expected["radial"]["neighbor_typespin"] is not None
    assert torch.allclose(
        cached["radial"]["neighbor_typespin"],
        expected["radial"]["neighbor_typespin"],
    )

    assert cached["angular"]["triplet_typespin"] is not None
    assert expected["angular"]["triplet_typespin"] is not None
    assert torch.allclose(
        cached["angular"]["triplet_typespin"],
        expected["angular"]["triplet_typespin"],
    )


@pytest.mark.cpu
def test_hdf5_force_derivative_cache_round_trip_float32(tmp_path: Path):
    """Persisted derivative payloads should preserve the configured float dtype."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float32)
    db_path = tmp_path / "structures_force_cache_float32.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_force_derivatives=True)

    info = ds.get_force_derivative_cache_info()
    assert info is not None
    assert info["storage_dtype"] == "float32"

    cached = ds.load_persisted_force_derivatives(0)
    assert cached is not None
    assert cached["radial"]["dG_drij"].dtype == torch.float32
    assert cached["angular"]["grads_i"].dtype == torch.float32


@pytest.mark.cpu
def test_hdf5_force_derivative_cache_skips_entries_without_forces(
    tmp_path: Path,
):
    """Only force-labeled structures should appear in the derivative-cache index."""
    structs = [_make_energy_only_struct()]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_force_cache_noforce.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_force_derivatives=True)

    assert ds.has_persisted_force_derivatives() is False
    info = ds.get_force_derivative_cache_info()
    assert info is not None
    assert ds.load_persisted_force_derivatives(0) is None


@pytest.mark.cpu
def test_hdf5_load_without_descriptor_or_manifest_rejects_getitem(
    tmp_path: Path,
):
    """Descriptor-dependent access should fail clearly without a manifest."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    db_path = tmp_path / "structures_no_manifest.h5"
    ds_build = HDF5StructureDataset(
        descriptor=_make_descriptor(dtype=torch.float64),
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds_build.build_database(show_progress=False)

    ds_load = HDF5StructureDataset(
        descriptor=None,
        database_file=str(db_path),
        mode="load",
    )

    assert ds_load.has_persisted_descriptor() is False
    assert ds_load.get_descriptor_manifest() is None
    assert len(ds_load) == 1
    assert ds_load.get_structure(0).n_atoms == 3
    with pytest.raises(RuntimeError, match="requires a descriptor"):
        _ = ds_load[0]


@pytest.mark.cpu
def test_hdf5_dataset_close_is_idempotent_and_context_manager_closes(
    tmp_path: Path,
):
    """Public handle cleanup should be deterministic and safe to repeat."""
    structs = [_make_struct(0)]
    file_paths = [str(tmp_path / "s_0")]
    Path(file_paths[0]).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_close.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False)
    assert ds._h5 is None

    sample = ds[0]
    assert sample["n_atoms"] == 3
    assert ds._h5 is not None

    ds.close()
    assert ds._h5 is None
    ds.close()
    assert ds._h5 is None

    with HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        mode="load",
    ) as load_ds:
        assert load_ds._h5 is not None
        sample = load_ds[0]
        assert sample["n_atoms"] == 3

    assert load_ds._h5 is None


@pytest.mark.cpu
def test_hdf5_persisted_derivatives_are_loaded_only_for_accessed_entries(
    tmp_path: Path,
    monkeypatch,
):
    """Derivative-cache loads should stay per-sample and lazy."""
    structs = [_make_struct(0), _make_struct(1), _make_struct(2)]
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structs))]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    desc = _make_descriptor(dtype=torch.float64)
    db_path = tmp_path / "structures_lazy_cache.h5"

    ds = HDF5StructureDataset(
        descriptor=desc,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structs),
        mode="build",
    )
    ds.build_database(show_progress=False, persist_force_derivatives=True)

    calls: list[int] = []
    original_loader = ds.load_persisted_force_derivatives

    def _wrapped_load(idx: int):
        calls.append(int(idx))
        return original_loader(idx)

    monkeypatch.setattr(ds, "load_persisted_force_derivatives", _wrapped_load)

    assert calls == []
    _ = ds[0]
    assert calls == [0]
    _ = ds[2]
    assert calls == [0, 2]
