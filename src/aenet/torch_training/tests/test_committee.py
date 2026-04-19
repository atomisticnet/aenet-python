"""Tests for committee-training orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aenet.mlip.ensemble import AenetEnsembleResult, normalize_ensemble_members
from aenet.torch_featurize import ChebyshevDescriptor
from aenet.torch_training import (
    Adam,
    HDF5StructureDataset,
    Structure,
    TorchCommitteeConfig,
    TorchCommitteeMemberResult,
    TorchCommitteePotential,
    TorchCommitteePredictResult,
    TorchCommitteeTrainResult,
    TorchTrainingConfig,
)
from aenet.torch_training.dataset import CachedStructureDataset
from aenet.torch_training.sources import RecordSourceCollection, SourceRecord


def _make_descriptor() -> ChebyshevDescriptor:
    """Create the compact descriptor used by committee tests."""
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


def _make_architecture() -> dict[str, list[tuple[int, str]]]:
    """Return the small architecture used by committee tests."""
    return {"H": [(4, "tanh")]}


def _make_structures(count: int = 6) -> list[Structure]:
    """Create small named H-only structures for committee tests."""
    structures: list[Structure] = []
    base_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    for index in range(count):
        offset = 0.03 * index
        positions = base_positions + np.array(
            [
                [offset, 0.0, 0.0],
                [0.0, offset, 0.0],
                [0.0, 0.0, offset],
            ],
            dtype=np.float64,
        )
        structures.append(
            Structure(
                positions=positions,
                species=["H", "H", "H"],
                energy=0.2 * index,
                forces=None,
                name=f"struct_{index}.xsf",
            )
        )
    return structures


def _base_training_config() -> TorchTrainingConfig:
    """Return the shared training config used by committee tests."""
    return TorchTrainingConfig(
        iterations=1,
        method=Adam(mu=0.001, batchsize=2),
        testpercent=34,
        split_seed=7,
        atomic_energies={"H": 0.0},
        normalize_features=False,
        normalize_energy=False,
        memory_mode="cpu",
        device="cpu",
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
        show_progress=False,
    )


def _member_names(energy_file: Path) -> list[str]:
    """Read the structure identifiers written into one energies.* file."""
    lines = energy_file.read_text(encoding="utf-8").strip().splitlines()
    return [line.split()[-1] for line in lines[1:]]


@pytest.mark.cpu
def test_committee_config_validates_member_seeds():
    """Committee config should validate explicit per-member seed lists."""
    with pytest.raises(
        ValueError,
        match="member_seeds must have length equal to num_members",
    ):
        TorchCommitteeConfig(num_members=2, member_seeds=[11])


@pytest.mark.cpu
def test_committee_result_summary_uses_completed_finite_metrics(tmp_path: Path):
    """Committee result summaries should skip failed and unavailable metrics."""
    result = TorchCommitteeTrainResult(
        output_dir=tmp_path,
        metadata_path=tmp_path / "committee_metadata.json",
        execution_mode="sequential",
        members=[
            TorchCommitteeMemberResult(
                member_index=0,
                seed=1,
                split_seed=None,
                device="cpu",
                member_dir=tmp_path / "member_000",
                model_path=tmp_path / "member_000" / "model.pt",
                history_json_path=None,
                history_csv_path=None,
                summary_path=None,
                checkpoint_dir=None,
                status="completed",
                metrics={
                    "final_MAE_train": 1.0,
                    "final_RMSE_train": 2.0,
                    "final_MAE_test": None,
                    "final_RMSE_test": float("nan"),
                },
            ),
            TorchCommitteeMemberResult(
                member_index=1,
                seed=2,
                split_seed=None,
                device="cpu",
                member_dir=tmp_path / "member_001",
                model_path=tmp_path / "member_001" / "model.pt",
                history_json_path=None,
                history_csv_path=None,
                summary_path=None,
                checkpoint_dir=None,
                status="completed",
                metrics={
                    "final_MAE_train": 3.0,
                    "final_RMSE_train": 4.0,
                },
            ),
            TorchCommitteeMemberResult(
                member_index=2,
                seed=3,
                split_seed=None,
                device="cpu",
                member_dir=tmp_path / "member_002",
                model_path=None,
                history_json_path=None,
                history_csv_path=None,
                summary_path=None,
                checkpoint_dir=None,
                status="failed",
                metrics={"final_MAE_train": 100.0},
                error="boom",
            ),
        ],
    )

    stats = result.stats
    assert stats["final_MAE_train"]["mean"] == pytest.approx(2.0)
    assert stats["final_MAE_train"]["std"] == pytest.approx(1.0)
    assert stats["final_MAE_train"]["n"] == 2
    assert stats["final_RMSE_train"]["mean"] == pytest.approx(3.0)
    assert "final_MAE_test" not in stats
    assert "final_RMSE_test" not in stats

    summary = str(result)
    assert "final_MAE_train:" in summary
    assert "final_MAE_test:" not in summary
    assert "completed_members: 2" in summary
    assert "failed_members: 1" in summary
    assert result.trainouts == []


@pytest.mark.cpu
def test_sequential_committee_training_writes_metadata_and_models(tmp_path: Path):
    """Sequential committee runs should persist the standardized layout."""
    descriptor = _make_descriptor()
    committee = TorchCommitteePotential(_make_architecture(), descriptor)
    config = _base_training_config()
    committee_config = TorchCommitteeConfig(
        num_members=2,
        base_seed=11,
        max_parallel=1,
        output_dir=tmp_path / "committee",
    )

    result = committee.train(
        structures=_make_structures(),
        config=config,
        committee_config=committee_config,
    )

    assert result.execution_mode == "sequential"
    assert result.metadata_path.exists()
    assert len(result.members) == 2
    assert [member.seed for member in result.members] == [11, 12]

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["execution_mode"] == "sequential"
    assert metadata["resolved_train_dataset"] == "StructureDataset"
    assert metadata["resolved_test_dataset"] == "StructureDataset"

    for index, member in enumerate(result.members):
        assert member.status == "completed"
        assert member.member_dir == tmp_path / "committee" / f"member_{index:03d}"
        assert member.model_path is not None and member.model_path.exists()
        assert member.history_json_path is not None and member.history_json_path.exists()
        assert member.history_csv_path is not None and member.history_csv_path.exists()
        assert member.summary_path is not None and member.summary_path.exists()

    assert len(result.completed_members) == 2
    assert len(result.failed_members) == 0

    stats = result.stats
    assert stats["final_MAE_train"]["n"] == 2
    assert stats["final_RMSE_train"]["n"] == 2
    assert stats["final_MAE_test"]["n"] == 2
    assert stats["final_RMSE_test"]["n"] == 2

    summary = str(result)
    assert "Training statistics:" in summary
    assert "final_MAE_train:" in summary
    assert "final_RMSE_train:" in summary
    assert "\u00b1" in summary
    assert "completed_members: 2" in summary
    assert "failed_members: 0" in summary

    table = result.to_dataframe()
    assert len(table) == 2
    assert "final_MAE_train" in table.columns
    assert "final_RMSE_train" in table.columns
    assert table["status"].tolist() == ["completed", "completed"]

    trainouts = result.trainouts
    assert len(trainouts) == 2
    assert "RMSE_train" in trainouts[0].errors.columns
    assert result.members[0].trainout is not None
    assert result.members[0].stats["final_RMSE_train"] == pytest.approx(
        result.members[0].trainout.stats["final_RMSE_train"]
    )


@pytest.mark.cpu
def test_committee_training_allows_default_device_auto_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Committee training should honor the trainer's default device logic."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    descriptor = _make_descriptor()
    committee = TorchCommitteePotential(_make_architecture(), descriptor)
    config = _base_training_config()
    config.device = None

    result = committee.train(
        structures=_make_structures(),
        config=config,
        committee_config=TorchCommitteeConfig(
            num_members=1,
            base_seed=13,
            max_parallel=1,
            output_dir=tmp_path / "committee_auto_device",
        ),
    )

    assert result.members[0].status == "completed"
    assert result.members[0].device == "cpu"


@pytest.mark.cpu
def test_committee_preserves_explicit_train_test_split(tmp_path: Path):
    """Explicit train/test datasets should be reused across all members."""
    descriptor = _make_descriptor()
    dataset = CachedStructureDataset(
        structures=_make_structures(),
        descriptor=descriptor,
        show_progress=False,
    )
    train_dataset = torch.utils.data.Subset(dataset, [0, 2, 4])
    test_dataset = torch.utils.data.Subset(dataset, [1, 3, 5])

    committee = TorchCommitteePotential(_make_architecture(), descriptor)
    config = _base_training_config()
    config.testpercent = 0
    config.save_energies = True

    result = committee.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        committee_config=TorchCommitteeConfig(
            num_members=2,
            base_seed=21,
            max_parallel=1,
            output_dir=tmp_path / "committee",
        ),
    )

    expected_train = ["struct_0.xsf", "struct_2.xsf", "struct_4.xsf"]
    expected_test = ["struct_1.xsf", "struct_3.xsf", "struct_5.xsf"]
    for member in result.members:
        assert member.status == "completed"
        train_names = _member_names(member.member_dir / "energies.train.0")
        test_names = _member_names(member.member_dir / "energies.test.0")
        assert train_names == expected_train
        assert test_names == expected_test


@pytest.mark.cpu
def test_parallel_committee_training_supports_hdf5_dataset(tmp_path: Path):
    """Parallel committee mode should work with the built-in HDF5 dataset."""
    descriptor = _make_descriptor()
    db_path = tmp_path / "committee.h5"
    structures = _make_structures()
    sources = RecordSourceCollection(
        [
            SourceRecord(
                source_id=f"struct_{index}",
                loader=(lambda struct=struct: struct),
                source_kind="test",
            )
            for index, struct in enumerate(structures)
        ]
    )
    build_dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        sources=sources,
        mode="build",
    )
    build_dataset.build_database(show_progress=False)

    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=str(db_path),
        mode="load",
    )

    committee = TorchCommitteePotential(_make_architecture(), descriptor)
    result = committee.train(
        dataset=dataset,
        config=_base_training_config(),
        committee_config=TorchCommitteeConfig(
            num_members=2,
            base_seed=31,
            max_parallel=2,
            output_dir=tmp_path / "committee_parallel",
        ),
    )

    assert result.execution_mode == "parallel"
    assert len(result.members) == 2
    for member in result.members:
        assert member.status == "completed"
        assert member.model_path is not None and member.model_path.exists()
        assert member.history_json_path is not None and member.history_json_path.exists()


@pytest.mark.cpu
def test_parallel_committee_rejects_external_dataset(tmp_path: Path):
    """Parallel committee mode should fail fast for unsupported datasets."""

    class ExternalDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            raise AssertionError("This test should fail before iteration.")

    descriptor = _make_descriptor()
    committee = TorchCommitteePotential(_make_architecture(), descriptor)

    with pytest.raises(
        ValueError,
        match="Parallel committee training only supports the built-in",
    ):
        committee.train(
            dataset=ExternalDataset(),
            config=_base_training_config(),
            committee_config=TorchCommitteeConfig(
                num_members=2,
                base_seed=41,
                max_parallel=2,
                output_dir=tmp_path / "committee_invalid",
            ),
        )


@pytest.mark.cpu
def test_committee_can_load_saved_members_and_aggregate_predictions(tmp_path: Path):
    """Saved committee outputs should round-trip into aggregated inference."""
    descriptor = _make_descriptor()
    committee = TorchCommitteePotential(_make_architecture(), descriptor)
    result = committee.train(
        structures=_make_structures(),
        config=_base_training_config(),
        committee_config=TorchCommitteeConfig(
            num_members=2,
            base_seed=51,
            max_parallel=1,
            output_dir=tmp_path / "committee_predict",
        ),
    )

    reloaded = TorchCommitteePotential.from_directory(result.output_dir)
    predictions = reloaded.predict(_make_structures(count=2), eval_forces=True)

    assert reloaded.member_model_paths() == [
        member.model_path for member in result.members if member.model_path is not None
    ]
    assert isinstance(predictions, TorchCommitteePredictResult)
    assert len(predictions.member_outputs) == 2
    assert len(predictions.member_outputs[0].total_energy) == 2
    assert predictions.indices == [0, 1]
    assert predictions.source_indices == [None, None]
    assert len(predictions) == 2
    assert isinstance(predictions[0], AenetEnsembleResult)
    assert predictions[0].num_members == 2
    assert predictions[0].forces is not None
    assert predictions[0].member_forces is not None
    assert predictions[0].member_forces.shape[0] == 2
    assert predictions.energy_std.shape == (2,)
    table = predictions.to_dataframe()
    assert list(table["index"]) == [0, 1]
    assert "energy_std" in table.columns
    assert "member_energies" in table.columns


@pytest.mark.cpu
def test_committee_predict_dataset_supports_subset_metadata(tmp_path: Path):
    """Dataset-backed committee prediction should preserve subset metadata."""
    descriptor = _make_descriptor()
    structures = _make_structures()
    dataset = CachedStructureDataset(
        structures=structures,
        descriptor=descriptor,
        show_progress=False,
    )
    subset = torch.utils.data.Subset(dataset, [1, 3, 5])

    committee = TorchCommitteePotential(_make_architecture(), descriptor)
    committee.train(
        structures=structures,
        config=_base_training_config(),
        committee_config=TorchCommitteeConfig(
            num_members=2,
            base_seed=52,
            max_parallel=1,
            output_dir=tmp_path / "committee_predict_dataset",
        ),
    )

    predictions = committee.predict_dataset(subset, eval_forces=False)

    assert isinstance(predictions, TorchCommitteePredictResult)
    assert len(predictions) == 3
    assert len(predictions.member_outputs) == 2
    assert len(predictions.member_outputs[0].total_energy) == 3
    assert predictions.indices == [0, 1, 2]
    assert predictions.source_indices == [1, 3, 5]
    assert predictions.identifiers == [
        "struct_1.xsf",
        "struct_3.xsf",
        "struct_5.xsf",
    ]
    assert predictions.energy_mean.shape == (3,)
    assert predictions.energy_std.shape == (3,)

    table = predictions.to_dataframe()
    assert list(table["source_index"]) == [1, 3, 5]
    assert list(table["identifier"]) == predictions.identifiers
    sorted_table = predictions.sort_by("energy_std")
    top_table = predictions.top_uncertain(n=2)
    assert len(sorted_table) == 3
    assert len(top_table) == 2
    assert sorted_table["energy_std"].is_monotonic_decreasing

    force_predictions = committee.predict_dataset(subset, eval_forces=True)
    assert len(force_predictions) == 3
    assert len(force_predictions.member_outputs) == 2
    assert force_predictions.member_outputs[0].forces is not None
    assert force_predictions[0].forces is not None
    assert force_predictions[0].member_forces is not None
    assert force_predictions[0].force_uncertainty is not None
    assert force_predictions.max_force_uncertainty.shape == (3,)
    force_table = force_predictions.to_dataframe()
    assert "max_force_uncertainty" in force_table.columns
    assert force_table["max_force_uncertainty"].notna().all()


@pytest.mark.cpu
def test_duplicate_member_committee_has_zero_uncertainty(tmp_path: Path):
    """Duplicate member models should yield zero committee uncertainty."""
    descriptor = _make_descriptor()
    committee = TorchCommitteePotential(_make_architecture(), descriptor)
    result = committee.train(
        structures=_make_structures(),
        config=_base_training_config(),
        committee_config=TorchCommitteeConfig(
            num_members=1,
            base_seed=61,
            max_parallel=1,
            output_dir=tmp_path / "committee_duplicate",
        ),
    )

    model_path = result.members[0].model_path
    assert model_path is not None

    duplicate_committee = TorchCommitteePotential.from_files([model_path, model_path])
    prediction = duplicate_committee.predict(
        _make_structures(count=1),
        eval_forces=True,
    )[0]

    assert prediction.energy_std == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(prediction.member_energies[0], prediction.member_energies[1])
    np.testing.assert_allclose(prediction.forces_std, 0.0, atol=1e-12)
    np.testing.assert_allclose(prediction.force_uncertainty, 0.0, atol=1e-12)


@pytest.mark.cpu
def test_committee_ascii_export_returns_ensemble_manifest(tmp_path: Path):
    """Committee ASCII export should produce the manifest expected by ensemble APIs."""
    descriptor = _make_descriptor()
    committee = TorchCommitteePotential(_make_architecture(), descriptor)
    structures = _make_structures()
    result = committee.train(
        structures=structures,
        config=_base_training_config(),
        committee_config=TorchCommitteeConfig(
            num_members=2,
            base_seed=71,
            max_parallel=1,
            output_dir=tmp_path / "committee_ascii",
        ),
    )

    manifest = committee.to_aenet_ascii(
        tmp_path / "ascii_committee",
        prefix="committee",
        structures=structures,
    )

    assert committee.member_model_paths() == [
        member.model_path for member in result.members if member.model_path is not None
    ]
    assert normalize_ensemble_members(manifest) == manifest
    assert len(manifest) == 2
    assert set(manifest[0].keys()) == {"H"}
    for index, member_manifest in enumerate(manifest):
        exported = Path(member_manifest["H"])
        assert exported.exists()
        assert exported.parent == tmp_path / "ascii_committee" / f"member_{index:03d}"
