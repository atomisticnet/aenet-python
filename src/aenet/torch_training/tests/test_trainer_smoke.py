import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tables
import torch
from torch.utils.data import Subset

import aenet.torch_training.trainer as trainer_module
from aenet.geometry import AtomicStructure
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    HDF5StructureDataset,
    Structure,
    TorchANNPotential,
    TorchTrainingConfig,
)
from aenet.torch_training.dataset import StructureDataset
from aenet.torch_training.sources import RecordSourceCollection, SourceRecord
from aenet.torch_training.trainer import _TrainingPolicyDataset


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


def make_simple_structures_H_many(n_structures=4):
    structures = []
    for idx in range(n_structures):
        shift = 0.05 * idx
        positions = np.array(
            [
                [shift, 0.0, 0.0],
                [0.9 + shift, 0.0, 0.0],
                [0.0, 0.9 + shift, 0.0],
            ],
            dtype=np.float64,
        )
        structures.append(
            Structure(
                positions=positions,
                species=["H", "H", "H"],
                energy=0.5 * idx,
                forces=np.zeros_like(positions),
            )
        )
    return structures


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


def _payload_source_collection(
    source_ids: list[str],
    payloads: list[Structure | list[Structure]],
) -> RecordSourceCollection:
    """Build a record-backed source collection for HDF5 smoke tests."""
    records = [
        SourceRecord(
            source_id=source_id,
            loader=(lambda payload=payload: payload),
            source_kind="test",
        )
        for source_id, payload in zip(source_ids, payloads, strict=True)
    ]
    return RecordSourceCollection(records)


def make_arch_H(descriptor: ChebyshevDescriptor):
    # For single species, n_features = (rad_order+1) + (ang_order+1) = 2
    # Hidden size small; activations supported by NetAtom: linear/tanh/sigmoid
    return {
        "H": [(4, "tanh")],
    }


def zero_model_weights(pot: TorchANNPotential):
    for seq in pot.net.functions:
        for p in seq.parameters():
            with torch.no_grad():
                p.zero_()


def write_xsf_structures(structures, directory: Path):
    paths = []
    for idx, structure in enumerate(structures):
        atomic = AtomicStructure(
            structure.positions,
            structure.species,
            energy=structure.energy,
            forces=structure.forces,
        )
        path = directory / f"structure-{idx}.xsf"
        atomic.to_file(path)
        paths.append(path)
    return paths


@pytest.mark.cpu
def test_energy_only_smoke(tmp_path: Path):
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    ckpt_dir = tmp_path / "ckpts"
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=50,  # trigger validation + best model path
        force_weight=0.0,  # energy-only
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

    # Checkpoint saved
    assert ckpt_dir.exists()
    # Either checkpoint_epoch_0000.pt or best_model.pt (or both)
    files = {p.name for p in ckpt_dir.iterdir()}
    assert any(name.startswith("checkpoint_epoch_")
               and name.endswith(".pt") for name in files)


@pytest.mark.cpu
def test_warns_for_scheduler_with_tiny_validation_set():
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=1,
        method=None,
        testpercent=50,
        force_weight=0.0,
        atomic_energies={"H": 0.0},
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=True,
        show_progress=False,
    )

    with pytest.warns(
        UserWarning,
        match=r"use_scheduler=True with a validation set of only 1 structure",
    ):
        pot.train(
            structures=structures,
            config=cfg,
        )


@pytest.mark.cpu
def test_warns_for_save_best_with_tiny_validation_set(tmp_path: Path):
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    ckpt_dir = tmp_path / "ckpts"
    cfg = TorchTrainingConfig(
        iterations=1,
        method=None,
        testpercent=50,
        force_weight=0.0,
        atomic_energies={"H": 0.0},
        memory_mode="cpu",
        device="cpu",
        save_energies=False,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=None,
        save_best=True,
        use_scheduler=False,
        show_progress=False,
    )

    with pytest.warns(
        UserWarning,
        match=r"save_best=True with a validation set of only 1 structure",
    ):
        pot.train(
            structures=structures,
            config=cfg,
        )


@pytest.mark.cpu
def test_force_training_smoke(tmp_path: Path):
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    ckpt_dir = tmp_path / "ckpts"
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
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=1,
        max_checkpoints=2,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    # TrainOut object populated
    assert "RMSE_force_train" in result.errors.columns
    assert len(result.errors) == 1
    # force rmse should be a number
    assert not math.isnan(result.errors["RMSE_force_train"].iloc[0])

    # Checkpoint saved and rotation does not error (single epoch anyway)
    assert ckpt_dir.exists()
    files = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    assert len(files) >= 1


@pytest.mark.cpu
def test_save_energies_writes_compatible_trainout_files(tmp_path: Path,
                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=50,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    assert (tmp_path / "energies.train.0").exists()
    assert (tmp_path / "energies.test.0").exists()
    assert result.energies is not None
    assert len(result.energies.energies_train) == 1
    assert len(result.energies.energies_test) == 1
    assert "ANN(eV/atom)" in result.energies.energies_train.columns
    assert "Ref(eV/atom)" in result.energies.energies_train.columns


@pytest.mark.cpu
def test_save_energies_uses_predict_dataset_for_cached_splits(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    call_count = {"forward_from_positions": 0}
    orig_forward = descriptor.forward_from_positions

    def _wrapped_forward(*args, **kwargs):
        call_count["forward_from_positions"] += 1
        return orig_forward(*args, **kwargs)

    monkeypatch.setattr(descriptor, "forward_from_positions", _wrapped_forward)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=50,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        cache_features=True,
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    pot.train(
        structures=structures,
        config=cfg,
    )

    # One featurization per structure during CachedStructureDataset build,
    # and no extra featurization during save_energies.
    assert call_count["forward_from_positions"] == 2


@pytest.mark.cpu
def test_save_energies_without_test_split_writes_train_only(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    assert (tmp_path / "energies.train.0").exists()
    assert not (tmp_path / "energies.test.0").exists()
    assert result.energies is not None
    assert len(result.energies.energies_train) == 2
    assert result.energies.energies_test is None


@pytest.mark.cpu
def test_save_energies_uses_total_energy_columns_with_atomic_references(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    for s in structures:
        s.energy = 3.69  # 3 H atoms with E_H = 1.23 each

    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)
    zero_model_weights(pot)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        atomic_energies={"H": 1.23},
        normalize_features=False,
        normalize_energy=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    train_df = result.energies.energies_train
    assert np.allclose(train_df["Ref(eV)"].values, 3.69)
    assert np.allclose(train_df["ANN(eV)"].values, 3.69)
    assert np.allclose(train_df["Ref-ANN(eV/atom)"].values, 0.0)


@pytest.mark.cpu
def test_save_energies_preserves_original_structure_indices_across_splits(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_many(n_structures=4)
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=50,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structures,
        config=cfg,
    )

    train_names = set(result.energies.energies_train["Path-of-input-file"])
    test_names = set(result.energies.energies_test["Path-of-input-file"])
    expected_names = {f"structure_{idx:06d}" for idx in range(4)}

    assert train_names.isdisjoint(test_names)
    assert train_names | test_names == expected_names


@pytest.mark.cpu
def test_save_energies_uses_input_paths_for_split_outputs(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_many(n_structures=4)
    structure_paths = write_xsf_structures(structures, tmp_path)
    descriptor = make_descriptor_H(dtype=torch.float64)
    arch = make_arch_H(descriptor)

    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=50,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        structures=structure_paths,
        config=cfg,
    )

    train_paths = set(result.energies.energies_train["Path-of-input-file"])
    test_paths = set(result.energies.energies_test["Path-of-input-file"])
    expected_paths = {str(path) for path in structure_paths}

    assert train_paths.isdisjoint(test_paths)
    assert train_paths | test_paths == expected_paths


@pytest.mark.cpu
def test_save_energies_hdf5_split_outputs_use_source_paths_and_frames(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    frame_groups = [
        make_simple_structures_H_two(),
        make_simple_structures_H_two(),
    ]
    frame_groups[1][0].energy = 1.0
    frame_groups[1][1].energy = 1.5

    file_paths = []
    for idx in range(len(frame_groups)):
        path = tmp_path / f"frames_{idx}"
        path.write_text("placeholder", encoding="utf-8")
        file_paths.append(str(path))

    descriptor = make_descriptor_H(dtype=torch.float64)
    db_path = tmp_path / "save_energies_multiframe.h5"
    build_ds = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, frame_groups),
        mode="build",
    )
    build_ds.build_database(show_progress=False)

    load_ds = HDF5StructureDataset(
        descriptor=make_descriptor_H(dtype=torch.float64),
        database_file=str(db_path),
        mode="load",
    )

    arch = make_arch_H(descriptor)
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=50,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        dataset=load_ds,
        config=cfg,
    )

    assert (tmp_path / "energies.train.0").exists()
    assert (tmp_path / "energies.test.0").exists()
    assert result.energies is not None

    train_paths = set(result.energies.energies_train["Path-of-input-file"])
    test_paths = set(result.energies.energies_test["Path-of-input-file"])
    expected_paths = {
        f"{file_path}#frame={frame_idx}"
        for file_path in file_paths
        for frame_idx in range(2)
    }

    assert train_paths.isdisjoint(test_paths)
    assert train_paths | test_paths == expected_paths


@pytest.mark.cpu
def test_save_energies_hdf5_identifier_precedence_uses_name_then_fallback(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    structures[0].name = "named_frame"
    structures[1].name = "will_be_cleared"

    file_path = tmp_path / "frames_0"
    file_path.write_text("placeholder", encoding="utf-8")

    descriptor = make_descriptor_H(dtype=torch.float64)
    db_path = tmp_path / "save_energies_identifier_precedence.h5"
    build_ds = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=_payload_source_collection([str(file_path)], [structures]),
        mode="build",
    )
    build_ds.build_database(show_progress=False)

    with tables.open_file(str(db_path), mode="a") as h5:
        meta = h5.get_node("/entries/meta")
        for row in meta.iterrows():
            row["source_id"] = ""
            if int(row["frame_idx"]) == 1:
                row["name"] = ""
            row.update()
        meta.flush()

    load_ds = HDF5StructureDataset(
        descriptor=make_descriptor_H(dtype=torch.float64),
        database_file=str(db_path),
        mode="load",
    )

    arch = make_arch_H(descriptor)
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        dataset=load_ds,
        config=cfg,
    )

    assert (tmp_path / "energies.train.0").exists()
    assert result.energies is not None

    identifiers = set(result.energies.energies_train["Path-of-input-file"])
    assert identifiers == {
        "named_frame#frame=0",
        "structure_000001#frame=1",
    }


@pytest.mark.cpu
def test_save_energies_hdf5_identifier_precedence_prefers_display_name(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    structures = make_simple_structures_H_two()
    structures[0].name = "struct-name-0"
    structures[1].name = "struct-name-1"

    file_path = tmp_path / "frames_0"
    file_path.write_text("placeholder", encoding="utf-8")

    descriptor = make_descriptor_H(dtype=torch.float64)
    db_path = tmp_path / "save_energies_display_name_precedence.h5"
    build_ds = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=_payload_source_collection([str(file_path)], [structures]),
        mode="build",
    )
    build_ds.build_database(show_progress=False)

    with tables.open_file(str(db_path), mode="a") as h5:
        meta = h5.get_node("/entries/meta")
        for row in meta.iterrows():
            row["display_name"] = "archive.tar:member.xsf"
            row.update()
        meta.flush()

    load_ds = HDF5StructureDataset(
        descriptor=make_descriptor_H(dtype=torch.float64),
        database_file=str(db_path),
        mode="load",
    )

    arch = make_arch_H(descriptor)
    pot = TorchANNPotential(arch=arch, descriptor=descriptor)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        save_energies=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    result = pot.train(
        dataset=load_ds,
        config=cfg,
    )

    assert (tmp_path / "energies.train.0").exists()
    assert result.energies is not None

    identifiers = set(result.energies.energies_train["Path-of-input-file"])
    assert identifiers == {
        "archive.tar:member.xsf#frame=0",
        "archive.tar:member.xsf#frame=1",
    }


@pytest.mark.cpu
def test_training_policy_cache_features_preserves_hdf5_feature_values(
    tmp_path: Path,
):
    """Trainer-owned runtime caches should preserve HDF5 feature values."""
    structures = make_simple_structures_H_two()
    file_paths = [str(tmp_path / f"s_{i}") for i in range(len(structures))]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    db_persisted = tmp_path / "policy_cached_features_persisted.h5"
    ds_persisted = HDF5StructureDataset(
        descriptor=make_descriptor_H(dtype=torch.float64),
        database_file=str(db_persisted),
        sources=_payload_source_collection(file_paths, structures),
        mode="build",
    )
    ds_persisted.build_database(show_progress=False, persist_features=True)

    db_fallback = tmp_path / "policy_cached_features_fallback.h5"
    ds_fallback = HDF5StructureDataset(
        descriptor=make_descriptor_H(dtype=torch.float64),
        database_file=str(db_fallback),
        sources=_payload_source_collection(file_paths, structures),
        mode="build",
    )
    ds_fallback.build_database(show_progress=False)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        cache_features=True,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )

    persisted_policy = _TrainingPolicyDataset(ds_persisted, cfg, split="train")
    fallback_policy = _TrainingPolicyDataset(ds_fallback, cfg, split="train")

    persisted_first = persisted_policy[0]
    persisted_second = persisted_policy[0]
    fallback_first = fallback_policy[0]
    fallback_second = fallback_policy[0]

    positions = torch.as_tensor(structures[0].positions, dtype=torch.float64)
    expected = ds_persisted.descriptor.forward_from_positions(
        positions,
        structures[0].species,
        None,
        None,
    )

    assert 0 in persisted_policy._cache_state.feature_cache
    assert 0 in fallback_policy._cache_state.feature_cache
    assert torch.equal(persisted_first["features"], expected)
    assert torch.equal(persisted_second["features"], expected)
    assert torch.equal(fallback_first["features"], expected)
    assert torch.equal(fallback_second["features"], expected)
    assert torch.equal(persisted_first["features"], fallback_first["features"])


@pytest.mark.cpu
def test_training_policy_feature_cache_eviction_uses_lru_order():
    """Trainer-owned feature caches should evict the least-recent entry."""
    structures = make_simple_structures_H_many(n_structures=3)
    descriptor = make_descriptor_H(dtype=torch.float64)
    dataset = StructureDataset(structures=structures, descriptor=descriptor)
    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        cache_features=True,
        cache_feature_max_entries=2,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )

    policy = _TrainingPolicyDataset(dataset, cfg, split="train")
    _ = policy[0]
    _ = policy[1]
    _ = policy[0]
    _ = policy[2]

    cache = policy._cache_state.feature_cache
    assert len(cache) == 2
    assert 0 in cache
    assert 1 not in cache
    assert 2 in cache


@pytest.mark.cpu
def test_training_policy_warmup_stops_when_bounded_caches_are_full(
    monkeypatch,
):
    """Warmup should stop once all enabled bounded caches reach capacity."""
    structures = make_simple_structures_H_many(n_structures=6)
    descriptor = make_descriptor_H(dtype=torch.float64)
    dataset = StructureDataset(structures=structures, descriptor=descriptor)
    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        cache_features=True,
        cache_feature_max_entries=2,
        cache_warmup=True,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )

    policy = _TrainingPolicyDataset(dataset, cfg, split="train")
    calls = {"count": 0}
    original_getitem = _TrainingPolicyDataset.__getitem__

    def _counting_getitem(self, idx):
        calls["count"] += 1
        return original_getitem(self, idx)

    monkeypatch.setattr(_TrainingPolicyDataset, "__getitem__", _counting_getitem)

    policy.warmup_caches(show_progress=False)

    assert calls["count"] == 2
    assert len(policy._cache_state.feature_cache) == 2


class _StopAfterLoaderConfig(RuntimeError):
    """Sentinel used to stop trainer setup after DataLoader configuration."""


@pytest.mark.cpu
def test_find_hdf5_root_datasets_walks_policy_and_subset_wrappers(
    tmp_path: Path,
):
    """HDF5 cleanup should find one root dataset through wrapper layers."""
    structures = make_simple_structures_H_many(n_structures=4)
    file_paths = [str(tmp_path / f"s_{idx}.xsf") for idx in range(4)]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    descriptor = make_descriptor_H(dtype=torch.float64)
    db_path = tmp_path / "worker_cleanup_walk.h5"
    build_ds = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structures),
        mode="build",
    )
    build_ds.build_database(show_progress=False)
    load_ds = HDF5StructureDataset(
        descriptor=make_descriptor_H(dtype=torch.float64),
        database_file=str(db_path),
        mode="load",
    )

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )
    wrapped = Subset(
        _TrainingPolicyDataset(Subset(load_ds, [0, 1, 2]), cfg, split="train"),
        [0, 1],
    )

    discovered = trainer_module._find_hdf5_root_datasets(wrapped)

    assert discovered == [load_ds]


@pytest.mark.cpu
def test_register_hdf5_worker_cleanup_closes_reachable_roots(
    tmp_path: Path,
    monkeypatch,
):
    """Worker cleanup registration should close reachable HDF5 roots."""
    structures = make_simple_structures_H_many(n_structures=3)
    file_paths = [str(tmp_path / f"s_{idx}.xsf") for idx in range(3)]
    for path in file_paths:
        Path(path).write_text("placeholder", encoding="utf-8")

    descriptor = make_descriptor_H(dtype=torch.float64)
    db_path = tmp_path / "worker_cleanup_register.h5"
    build_ds = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=_payload_source_collection(file_paths, structures),
        mode="build",
    )
    build_ds.build_database(show_progress=False)
    load_ds = HDF5StructureDataset(
        descriptor=make_descriptor_H(dtype=torch.float64),
        database_file=str(db_path),
        mode="load",
    )

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )
    wrapped = Subset(_TrainingPolicyDataset(load_ds, cfg, split="train"), [0, 1])

    callbacks = []
    close_calls = []

    monkeypatch.setattr(
        trainer_module.atexit,
        "register",
        lambda callback: callbacks.append(callback),
    )
    monkeypatch.setattr(
        trainer_module.torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(dataset=wrapped),
    )
    monkeypatch.setattr(load_ds, "close", lambda: close_calls.append("closed"))

    trainer_module._register_hdf5_worker_cleanup(worker_id=0)

    assert len(callbacks) == 1
    callbacks[0]()
    assert close_calls == ["closed"]


@pytest.mark.cpu
def test_random_force_resampling_disables_persistent_train_workers(
    monkeypatch,
):
    """Epoch-level random resampling should restart training workers."""
    records = []

    def _record_train_loader(*args, **kwargs):
        records.append(kwargs)
        raise _StopAfterLoaderConfig

    monkeypatch.setattr(trainer_module, "DataLoader", _record_train_loader)

    descriptor = make_descriptor_H(dtype=torch.float64)
    pot = TorchANNPotential(arch=make_arch_H(descriptor), descriptor=descriptor)
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        force_fraction=0.5,
        force_sampling="random",
        force_resample_num_epochs=1,
        num_workers=2,
        persistent_workers=True,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    with pytest.warns(
        UserWarning,
        match="Disabling persistent_workers for the training DataLoader",
    ):
        with pytest.raises(_StopAfterLoaderConfig):
            pot.train(
                structures=make_simple_structures_H_many(n_structures=6),
                config=cfg,
            )

    assert len(records) == 1
    assert records[0]["num_workers"] == 2
    assert records[0]["persistent_workers"] is False
    assert callable(records[0]["worker_init_fn"])


@pytest.mark.cpu
def test_fixed_or_static_force_sampling_keeps_persistent_train_workers(
    monkeypatch,
):
    """Training should preserve persistent workers when no restart is needed."""
    records = []

    def _record_train_loader(*args, **kwargs):
        records.append(kwargs)
        raise _StopAfterLoaderConfig

    monkeypatch.setattr(trainer_module, "DataLoader", _record_train_loader)

    descriptor = make_descriptor_H(dtype=torch.float64)
    pot = TorchANNPotential(arch=make_arch_H(descriptor), descriptor=descriptor)
    cfg = TorchTrainingConfig(
        iterations=1,
        testpercent=0,
        force_weight=0.5,
        force_fraction=0.5,
        force_sampling="random",
        force_resample_num_epochs=0,
        num_workers=2,
        persistent_workers=True,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        use_scheduler=False,
        show_progress=False,
    )

    with pytest.raises(_StopAfterLoaderConfig):
        pot.train(
            structures=make_simple_structures_H_many(n_structures=6),
            config=cfg,
        )

    assert len(records) == 1
    assert records[0]["num_workers"] == 2
    assert records[0]["persistent_workers"] is True
    assert callable(records[0]["worker_init_fn"])


@pytest.mark.cpu
def test_cache_warmup_is_disabled_by_default(monkeypatch):
    """Stats collection should not prefill runtime caches by default."""
    structures = make_simple_structures_H_many(n_structures=4)
    descriptor = make_descriptor_H(dtype=torch.float64)
    dataset = StructureDataset(structures=structures, descriptor=descriptor)
    pot = TorchANNPotential(arch=make_arch_H(descriptor), descriptor=descriptor)

    def _unexpected_cached_getitem(self, idx):
        raise AssertionError(
            "runtime-cache-backed dataset access should not happen before "
            "epoch 0 when cache_warmup=False"
        )

    monkeypatch.setattr(
        _TrainingPolicyDataset,
        "__getitem__",
        _unexpected_cached_getitem,
    )

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        cache_features=True,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )

    pot.train(dataset=dataset, config=cfg)


@pytest.mark.cpu
def test_cache_warmup_runs_when_enabled_for_single_process_training(
    monkeypatch,
):
    """Explicit warmup should run before epoch 0 for single-process loaders."""
    structures = make_simple_structures_H_many(n_structures=4)
    descriptor = make_descriptor_H(dtype=torch.float64)
    dataset = StructureDataset(structures=structures, descriptor=descriptor)
    pot = TorchANNPotential(arch=make_arch_H(descriptor), descriptor=descriptor)

    calls = []

    def _record_warmup(self, show_progress=True):
        calls.append(self._split)
        return None

    monkeypatch.setattr(_TrainingPolicyDataset, "warmup_caches", _record_warmup)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        cache_features=True,
        cache_warmup=True,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )

    pot.train(dataset=dataset, config=cfg)

    assert calls == ["train"]


@pytest.mark.cpu
def test_cache_warmup_warns_and_skips_for_worker_dataloaders(
    monkeypatch,
):
    """Configured warmup should be skipped when DataLoader workers are used."""
    structures = make_simple_structures_H_many(n_structures=4)
    descriptor = make_descriptor_H(dtype=torch.float64)
    dataset = StructureDataset(structures=structures, descriptor=descriptor)
    pot = TorchANNPotential(arch=make_arch_H(descriptor), descriptor=descriptor)

    calls = []

    def _record_warmup(self, show_progress=True):
        calls.append(self._split)
        return None

    monkeypatch.setattr(_TrainingPolicyDataset, "warmup_caches", _record_warmup)

    cfg = TorchTrainingConfig(
        iterations=0,
        testpercent=0,
        force_weight=0.0,
        cache_features=True,
        cache_warmup=True,
        num_workers=2,
        persistent_workers=False,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )

    with pytest.warns(
        UserWarning,
        match="Skipping trainer-owned runtime cache warmup",
    ):
        pot.train(dataset=dataset, config=cfg)

    assert calls == []
