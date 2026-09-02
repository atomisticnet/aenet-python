"""Tests for force-informed Taylor energy augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from aenet.geometry.sampling import (
    TaylorExpansionConfig as GeometryTaylorExpansionConfig,
)
from aenet.geometry.sampling import taylor_energy as geometry_taylor_energy
from aenet.geometry.transformations import (
    AtomDisplacementTransformation,
    DOptimalDisplacementTransformation,
    RandomDisplacementTransformation,
)
from aenet.torch_training import (
    Adam,
    Structure,
    TaylorExpansionConfig,
    TaylorSourceCollection,
    TorchANNPotential,
    TorchTrainingConfig,
    generate_taylor_samples,
    split_reference_structures,
    taylor_energy,
)
from aenet.torch_training.dataset import StructureDataset
from aenet.torch_training.hdf5_dataset import HDF5StructureDataset
from aenet.torch_training.sources import RecordSourceCollection, SourceRecord

_DEFAULT_FORCES = object()


def test_legacy_torch_imports_are_compatibility_adapters():
    """Published torch imports should remain usable after the core move."""
    assert TaylorExpansionConfig is GeometryTaylorExpansionConfig
    assert taylor_energy is geometry_taylor_energy

    parent = _parent(name="legacy-parent")
    result = generate_taylor_samples([parent], _random_config(n_structures=1))

    assert all(isinstance(item, Structure) for item in result.structures)
    assert [record.label_origin for record in result.records] == [
        "exact",
        "taylor",
    ]


def _parent(
    *,
    name: str = "parent-a",
    energy: float | None = 1.25,
    forces: np.ndarray | None | object = _DEFAULT_FORCES,
) -> Structure:
    if forces is _DEFAULT_FORCES:
        forces = np.array(
            [
                [0.80, -0.20, 0.10],
                [-0.35, 0.15, -0.05],
            ],
            dtype=np.float64,
        )
    assert forces is None or isinstance(forces, np.ndarray)
    return Structure(
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]],
            dtype=np.float64,
        ),
        species=["H", "H"],
        energy=energy,
        forces=forces,
        name=name,
    )


def _random_config(*, n_structures: int = 3) -> TaylorExpansionConfig:
    return TaylorExpansionConfig(
        transformation=RandomDisplacementTransformation(
            rms=0.02,
            max_structures=n_structures,
            random_state=17,
            orthonormalize=False,
            remove_translations=False,
        ),
        include_reference=True,
    )


def _small_descriptor():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_cluster")
    pytest.importorskip("torch_scatter")
    from aenet.torch_featurize import ChebyshevDescriptor

    return ChebyshevDescriptor(
        species=["H"],
        rad_order=2,
        rad_cutoff=2.5,
        ang_order=0,
        ang_cutoff=2.5,
        min_cutoff=0.1,
        device="cpu",
        dtype=torch.float64,
    )


def test_taylor_energy_uses_force_sign_and_validates_shapes():
    forces = np.array([[2.0, -1.0, 0.5], [-0.5, 0.0, 1.0]])
    displacement = np.array([[0.1, 0.2, -0.1], [0.0, 0.4, 0.2]])

    energy = taylor_energy(4.0, forces, displacement)

    assert energy == pytest.approx(3.85)
    with pytest.raises(ValueError, match="same shape"):
        taylor_energy(4.0, forces, displacement[:1])
    with pytest.raises(ValueError, match="finite"):
        taylor_energy(np.nan, forces, displacement)


def test_random_sampling_reuses_applied_displacements_and_clears_forces():
    parent = _parent()
    original_positions = parent.positions.copy()
    original_forces = parent.forces.copy()

    result = generate_taylor_samples([parent], _random_config())

    assert result.n_parents == 1
    assert result.n_exact == 1
    assert result.n_derived == 3
    assert result.n_skipped == 0
    assert len(result.structures) == 4
    assert result.records[0].label_origin == "exact"
    assert result.records[0].structure.forces is not None

    for record in result.records[1:]:
        displacement = record.structure.positions - parent.positions
        expected = taylor_energy(parent.energy, parent.forces, displacement)
        assert record.strategy == "random"
        assert record.label_origin == "taylor"
        assert record.structure.energy == pytest.approx(expected)
        assert record.structure.forces is None
        assert record.displacement_rms == pytest.approx(0.02, abs=1e-6)
        assert record.structure.positions is not parent.positions
        assert record.structure.name.startswith("parent-a::taylor:random:")

    assert np.array_equal(parent.positions, original_positions)
    assert np.array_equal(parent.forces, original_forces)


def test_random_sampling_is_reproducible_without_advancing_prototype_rng():
    parent = _parent()
    config = _random_config(n_structures=2)

    first = generate_taylor_samples([parent], config)
    second = generate_taylor_samples([parent], config)

    first_positions = [record.structure.positions for record in first.records]
    second_positions = [
        record.structure.positions for record in second.records
    ]
    for positions_a, positions_b in zip(first_positions, second_positions):
        assert np.array_equal(positions_a, positions_b)


def test_adapter_parent_stream_is_stable_under_unrelated_insertion():
    config = _random_config(n_structures=2)

    original = generate_taylor_samples(
        [_parent(name="a"), _parent(name="b")],
        config,
    )
    inserted = generate_taylor_samples(
        [_parent(name="x"), _parent(name="a"), _parent(name="b")],
        config,
    )

    original_b = [
        record.structure.positions
        for record in original.records
        if record.parent_id == "b" and record.label_origin == "taylor"
    ]
    inserted_b = [
        record.structure.positions
        for record in inserted.records
        if record.parent_id == "b" and record.label_origin == "taylor"
    ]
    assert len(original_b) == len(inserted_b)
    assert all(
        np.array_equal(before, after)
        for before, after in zip(original_b, inserted_b)
    )


def test_doptimal_sampling_preserves_native_constraints_and_labels():
    parent = _parent()
    config = TaylorExpansionConfig(
        transformation=DOptimalDisplacementTransformation(
            rms=0.025,
            n_structures=4,
            max_iter=20,
            random_state=23,
            remove_translations=True,
            enforce_zero_mean=True,
        ),
        include_reference=False,
    )

    result = generate_taylor_samples([parent], config)

    assert result.n_exact == 0
    assert result.n_derived == 4
    displacements = []
    for record in result.records:
        displacement = record.structure.positions - parent.positions
        displacements.append(displacement)
        assert record.strategy == "d_optimal"
        assert record.structure.energy == pytest.approx(
            taylor_energy(parent.energy, parent.forces, displacement)
        )
        assert record.displacement_rms == pytest.approx(0.025, abs=1e-3)
        assert np.linalg.norm(displacement.mean(axis=0)) < 1e-9

    assert np.linalg.norm(np.mean(displacements, axis=0)) < 1e-6


@pytest.mark.parametrize(
    ("energy", "forces", "match"),
    [
        (None, np.ones((2, 3)), "finite energy"),
        (np.inf, np.ones((2, 3)), "finite energy"),
        (1.0, None, "force array"),
        (1.0, np.array([[np.nan, 0.0, 0.0]] * 2), "finite forces"),
    ],
)
def test_parent_validation_happens_before_generation(energy, forces, match):
    parent = _parent(energy=energy, forces=forces)

    with pytest.raises(ValueError, match=match):
        generate_taylor_samples([parent], _random_config())


def test_zero_force_policy_can_skip_or_keep_children():
    parent = _parent(forces=np.zeros((2, 3)))
    skip_config = _random_config(n_structures=2)
    keep_config = TaylorExpansionConfig(
        transformation=skip_config.transformation,
        include_reference=True,
        zero_force_policy="keep",
    )

    skipped = generate_taylor_samples([parent], skip_config)
    kept = generate_taylor_samples([parent], keep_config)

    assert skipped.n_exact == 1
    assert skipped.n_derived == 0
    assert skipped.n_skipped == 2
    assert kept.n_derived == 2
    assert all(
        record.structure.energy == pytest.approx(parent.energy)
        for record in kept.records[1:]
    )


def test_config_rejects_nonstatistical_displacement_transformation():
    with pytest.raises(TypeError, match="RandomDisplacementTransformation"):
        TaylorExpansionConfig(
            transformation=AtomDisplacementTransformation(0.01)
        )


def test_translation_removal_rejects_one_atom_degenerate_case():
    parent = Structure(
        positions=np.zeros((1, 3)),
        species=["H"],
        energy=0.0,
        forces=np.ones((1, 3)),
        name="single-atom",
    )
    config = TaylorExpansionConfig(
        transformation=RandomDisplacementTransformation(
            rms=0.01,
            max_structures=2,
            random_state=3,
            orthonormalize=False,
            remove_translations=True,
        )
    )

    with pytest.raises(ValueError, match="no internal displacement"):
        generate_taylor_samples([parent], config)


def test_reference_split_is_reproducible_disjoint_and_order_preserving():
    parents = [_parent(name=f"parent-{index}") for index in range(10)]

    first = split_reference_structures(
        parents,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=99,
    )
    second = split_reference_structures(
        parents,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=99,
    )

    assert first == second
    train, validation, test = first
    assert [len(train), len(validation), len(test)] == [6, 2, 2]
    names = [{structure.name for structure in split} for split in first]
    assert not names[0] & names[1]
    assert not names[0] & names[2]
    assert not names[1] & names[2]
    for split in first:
        indices = [
            int(structure.name.rsplit("-", 1)[-1]) for structure in split
        ]
        assert indices == sorted(indices)


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_taylor_source_collection_preserves_parent_mapping_in_hdf5(tmp_path):
    parent = _parent(name="source-parent")
    sources = RecordSourceCollection(
        [
            SourceRecord(
                source_id="reference/source-parent.xsf",
                loader=lambda: parent,
                source_kind="memory",
                display_name="source-parent.xsf",
            )
        ]
    )
    taylor_sources = TaylorSourceCollection(
        sources,
        _random_config(n_structures=2),
    )
    descriptor = _small_descriptor()
    database = tmp_path / "taylor.h5"
    dataset = HDF5StructureDataset(
        descriptor=descriptor,
        database_file=database,
        sources=taylor_sources,
        mode="build",
    )

    dataset.build_database(
        show_progress=False,
        persist_features=True,
        persist_force_derivatives=False,
    )

    assert len(dataset) == 3
    assert dataset.has_persisted_features()
    assert not dataset.has_persisted_force_derivatives()
    metadata = [dataset.get_entry_metadata(index) for index in range(3)]
    assert {item["source_id"] for item in metadata} == {
        "reference/source-parent.xsf"
    }
    assert [item["frame_idx"] for item in metadata] == [0, 1, 2]
    assert [item["has_forces"] for item in metadata] == [True, False, False]
    assert metadata[0]["name"].endswith("::exact")
    assert metadata[0]["taylor_parent_id"] == (
        "reference/source-parent.xsf#frame=0"
    )
    assert metadata[0]["taylor_child_index"] is None
    assert metadata[0]["taylor_strategy"] == "random"
    assert metadata[0]["taylor_label_origin"] == "exact"
    assert metadata[0]["source_frame_idx"] == 0
    assert all(
        item["name"].startswith("source-parent::taylor:random:")
        for item in metadata[1:]
    )
    assert [item["taylor_child_index"] for item in metadata[1:]] == [0, 1]
    assert all(
        item["taylor_parent_id"] == "reference/source-parent.xsf#frame=0"
        for item in metadata[1:]
    )
    assert all(
        item["taylor_label_origin"] == "taylor" for item in metadata[1:]
    )
    assert len({item["taylor_generation_id"] for item in metadata}) == 1
    dataset.close()

    reopened = HDF5StructureDataset(
        descriptor=None,
        database_file=database,
        mode="load",
    )
    assert len(reopened) == 3
    assert reopened.has_persisted_features()
    assert not reopened.has_persisted_force_derivatives()
    reopened_metadata = [
        reopened.get_entry_metadata(index) for index in range(3)
    ]
    assert reopened_metadata == metadata
    reopened.close()


def test_taylor_hdf5_energy_filter_rejects_complete_parent_family(tmp_path):
    parent = _parent(
        name="high-energy",
        energy=10.0,
        forces=np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]),
    )
    sources = RecordSourceCollection(
        [SourceRecord("high.xsf", lambda: parent, source_kind="memory")]
    )
    dataset = HDF5StructureDataset(
        descriptor=None,
        database_file=tmp_path / "filtered.h5",
        sources=TaylorSourceCollection(
            sources,
            _random_config(n_structures=20),
        ),
        mode="build",
        max_energy=0.0,
    )

    dataset.build_database(show_progress=False)

    assert len(dataset) == 0
    dataset.close()


def test_taylor_hdf5_keeps_all_children_of_accepted_parent(tmp_path):
    parent = _parent(
        name="accepted",
        energy=-1.0,
        forces=np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]),
    )
    sources = RecordSourceCollection(
        [SourceRecord("accepted.xsf", lambda: parent, source_kind="memory")]
    )
    dataset = HDF5StructureDataset(
        descriptor=None,
        database_file=tmp_path / "accepted.h5",
        sources=TaylorSourceCollection(
            sources,
            _random_config(n_structures=20),
        ),
        mode="build",
        max_energy=0.0,
    )

    dataset.build_database(show_progress=False)

    assert len(dataset) == 21
    assert any(
        dataset.get_entry_metadata(index)["energy"] > 0.0
        for index in range(len(dataset))
    )
    dataset.close()


def test_taylor_hdf5_force_filter_rejects_complete_parent_family(tmp_path):
    parent = _parent(
        name="high-force",
        energy=-1.0,
        forces=np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
    )
    sources = RecordSourceCollection(
        [SourceRecord("force.xsf", lambda: parent, source_kind="memory")]
    )
    dataset = HDF5StructureDataset(
        descriptor=None,
        database_file=tmp_path / "force-filtered.h5",
        sources=TaylorSourceCollection(sources, _random_config()),
        mode="build",
        max_forces=1.0,
    )

    dataset.build_database(show_progress=False)

    assert len(dataset) == 0
    dataset.close()


def test_taylor_hdf5_preserves_original_frame_identity(tmp_path):
    parents = [_parent(name="same-name"), _parent(name="same-name")]
    sources = RecordSourceCollection(
        [SourceRecord("trajectory.xsf", lambda: parents, source_kind="memory")]
    )
    dataset = HDF5StructureDataset(
        descriptor=None,
        database_file=tmp_path / "frames.h5",
        sources=TaylorSourceCollection(
            sources,
            _random_config(n_structures=1),
        ),
        mode="build",
    )

    dataset.build_database(show_progress=False)

    metadata = [dataset.get_entry_metadata(index) for index in range(4)]
    assert [item["source_frame_idx"] for item in metadata] == [0, 0, 1, 1]
    assert [item["taylor_parent_id"] for item in metadata] == [
        "trajectory.xsf#frame=0",
        "trajectory.xsf#frame=0",
        "trajectory.xsf#frame=1",
        "trajectory.xsf#frame=1",
    ]
    assert [item["taylor_label_origin"] for item in metadata] == [
        "exact",
        "taylor",
        "exact",
        "taylor",
    ]
    dataset.close()


def test_taylor_source_collection_preserves_chunked_streaming():
    parent_a = _parent(name="parent-a")
    parent_b = _parent(name="parent-b")

    class ChunkedSources(RecordSourceCollection):
        def __init__(self, records):
            super().__init__(records)
            self.requested_chunk_sizes = []

        def iter_record_chunks(self, chunk_size):
            self.requested_chunk_sizes.append(chunk_size)
            records = list(self.iter_records())
            for start in range(0, len(records), chunk_size):
                yield records[start : start + chunk_size]

    sources = ChunkedSources(
        [
            SourceRecord("a", lambda: parent_a, source_kind="memory"),
            SourceRecord("b", lambda: parent_b, source_kind="memory"),
        ]
    )
    taylor_sources = TaylorSourceCollection(
        sources,
        _random_config(n_structures=2),
    )

    chunks = list(taylor_sources.iter_record_chunks(chunk_size=1))

    assert sources.requested_chunk_sizes == [1]
    assert [[record.source_id for record in chunk] for chunk in chunks] == [
        ["a"],
        ["b"],
    ]
    assert [
        len(record.load_structures()) for chunk in chunks for record in chunk
    ] == [3, 3]


@pytest.mark.cpu
@pytest.mark.docs_examples
def test_energy_only_training_does_not_enter_force_loss(monkeypatch):
    descriptor = _small_descriptor()
    parents = [
        _parent(name="train-a", energy=1.0),
        _parent(name="train-b", energy=1.1),
    ]
    samples = generate_taylor_samples(
        parents,
        _random_config(n_structures=2),
    ).structures
    dataset = StructureDataset(
        samples,
        descriptor,
        atomic_energies={"H": 0.0},
    )

    def _unexpected_force_path(*args, **kwargs):
        raise AssertionError("force-loss path must not run")

    monkeypatch.setattr(
        "aenet.torch_training.training.training_loop.compute_force_loss",
        _unexpected_force_path,
    )
    model = TorchANNPotential({"H": [(4, "tanh")]}, descriptor=descriptor)
    config = TorchTrainingConfig(
        iterations=1,
        method=Adam(mu=0.001, batchsize=2),
        testpercent=0,
        force_weight=0.0,
        atomic_energies={"H": 0.0},
        normalize_features=False,
        normalize_energy=False,
        memory_mode="cpu",
        device="cpu",
        show_progress=False,
        checkpoint_dir=None,
        checkpoint_interval=0,
        max_checkpoints=None,
        save_best=False,
        use_scheduler=False,
    )

    output = model.train(train_dataset=dataset, config=config)

    assert len(output.errors) == 1
