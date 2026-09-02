"""Tests for backend-neutral Taylor energy sampling."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from aenet.geometry import AtomicStructure
from aenet.geometry.sampling import (
    TaylorExpansionConfig,
    TaylorReference,
    generate_taylor_samples,
    split_reference_structures,
    taylor_energy,
)
from aenet.geometry.transformations import (
    DOptimalDisplacementTransformation,
    RandomDisplacementTransformation,
)


def _structure(*, energy: float = 1.25, forces=None) -> AtomicStructure:
    if forces is None:
        forces = np.array(
            [[0.8, -0.2, 0.1], [-0.35, 0.15, -0.05]],
            dtype=np.float64,
        )
    return AtomicStructure(
        coords=np.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]]),
        types=["H", "H"],
        energy=energy,
        forces=forces,
    )


def _reference(parent_id: str) -> TaylorReference:
    return TaylorReference(parent_id=parent_id, structure=_structure())


def _random_config(
    *,
    n_structures: int = 3,
    remove_translations: bool = False,
) -> TaylorExpansionConfig:
    return TaylorExpansionConfig(
        transformation=RandomDisplacementTransformation(
            rms=0.02,
            max_structures=n_structures,
            random_state=17,
            orthonormalize=False,
            remove_translations=remove_translations,
        )
    )


def _child_positions(result, parent_id: str) -> list[np.ndarray]:
    return [
        record.structure.coords[0]
        for record in result.records
        if record.parent_id == parent_id and record.label_origin == "taylor"
    ]


def test_taylor_energy_uses_force_sign_and_validates_shapes():
    """The neutral primitive should implement the package force convention."""
    forces = np.array([[2.0, -1.0, 0.5], [-0.5, 0.0, 1.0]])
    displacement = np.array([[0.1, 0.2, -0.1], [0.0, 0.4, 0.2]])

    assert taylor_energy(4.0, forces, displacement) == pytest.approx(3.85)
    with pytest.raises(ValueError, match="same shape"):
        taylor_energy(4.0, forces, displacement[:1])


def test_random_sampling_clears_child_forces_and_preserves_parent():
    """Derived records should own coordinates and discard stale labels."""
    reference = _reference("parent-a")
    original_coords = reference.structure.coords[0].copy()
    original_forces = reference.structure.forces[0].copy()

    result = generate_taylor_samples([reference], _random_config())

    assert result.n_exact == 1
    assert result.n_derived == 3
    assert result.requested_children == 3
    assert result.accepted_children == 3
    assert result.n_skipped == 0
    for record in result.records[1:]:
        displacement = record.structure.coords[0] - original_coords
        assert record.structure.energy[0] == pytest.approx(
            taylor_energy(1.25, original_forces, displacement)
        )
        assert record.structure.forces[0] in (None, [])
        assert record.structure.coords[0] is not reference.structure.coords[0]
    assert np.array_equal(reference.structure.coords[0], original_coords)
    assert np.array_equal(reference.structure.forces[0], original_forces)


def test_parent_stream_is_stable_under_insertion_and_reordering():
    """Stable identity, rather than collection position, should seed a parent."""
    config = _random_config(n_structures=2)

    original = generate_taylor_samples(
        [_reference("a"), _reference("b")],
        config,
    )
    inserted = generate_taylor_samples(
        [_reference("x"), _reference("a"), _reference("b")],
        config,
    )
    reordered = generate_taylor_samples(
        [_reference("b"), _reference("a")],
        config,
    )

    expected = _child_positions(original, "b")
    expected_records = [
        record
        for record in original.records
        if record.parent_id == "b" and record.label_origin == "taylor"
    ]
    for result in (inserted, reordered):
        actual = _child_positions(result, "b")
        assert len(actual) == len(expected)
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
        actual_records = [
            record
            for record in result.records
            if record.parent_id == "b" and record.label_origin == "taylor"
        ]
        assert [record.child_index for record in actual_records] == [
            record.child_index for record in expected_records
        ]
        assert [record.structure.name for record in actual_records] == [
            record.structure.name for record in expected_records
        ]
        assert [record.structure.energy[0] for record in actual_records] == [
            record.structure.energy[0] for record in expected_records
        ]


def test_caller_owned_generator_is_not_advanced():
    """Configuration and generation should leave a caller's RNG untouched."""
    caller_rng = np.random.default_rng(123)
    comparison_rng = np.random.default_rng(123)
    config = TaylorExpansionConfig(
        transformation=RandomDisplacementTransformation(
            rms=0.02,
            max_structures=2,
            random_state=caller_rng,
            orthonormalize=False,
            remove_translations=False,
        )
    )

    generate_taylor_samples([_reference("parent")], config)

    assert caller_rng.random() == comparison_rng.random()


def test_parent_identity_must_be_nonempty_and_unique():
    """Ambiguous parent identities should fail before sampling."""
    with pytest.raises(ValueError, match="non-empty"):
        TaylorReference(parent_id="", structure=_structure())
    with pytest.raises(ValueError, match="unique"):
        generate_taylor_samples(
            [_reference("same"), _reference("same")],
            _random_config(),
        )


def test_multiframe_reference_is_rejected_explicitly():
    """The core API should never silently select the last trajectory frame."""
    structure = _structure()
    structure.add_frame(
        structure.coords[0] + 0.1,
        energy=1.5,
        forces=structure.forces[0],
    )

    with pytest.raises(ValueError, match="exactly one frame"):
        TaylorReference(parent_id="trajectory", structure=structure)


def test_orthonormal_shortfall_is_reported_by_cause():
    """Dimensionality-limited output should be visible in result accounting."""
    config = TaylorExpansionConfig(
        transformation=RandomDisplacementTransformation(
            rms=0.01,
            max_structures=10,
            random_state=7,
            orthonormalize=True,
            remove_translations=True,
        )
    )

    with pytest.warns(RuntimeWarning, match="only 3 orthonormal"):
        result = generate_taylor_samples([_reference("parent")], config)

    assert result.requested_children == 10
    assert result.accepted_children == 3
    assert result.unavailable_children == 7
    assert result.duplicate_skipped == 0
    assert result.zero_force_skipped == 0
    assert result.n_skipped == 7


def test_zero_force_shortfall_is_reported_by_cause():
    """Zero-force policy should account for every omitted request."""
    reference = TaylorReference(
        parent_id="stationary",
        structure=_structure(forces=np.zeros((2, 3))),
    )

    result = generate_taylor_samples(
        [reference],
        _random_config(n_structures=4),
    )

    assert result.requested_children == 4
    assert result.accepted_children == 0
    assert result.zero_force_skipped == 4
    assert result.unavailable_children == 0
    assert result.n_skipped == 4


def test_duplicate_displacements_are_reported_by_cause(monkeypatch):
    """Produced duplicates should not be conflated with unavailable output."""
    def duplicate_children(self, structure):
        for _ in range(2):
            child = structure.copy()
            child.coords[0] = structure.coords[0] + 0.01
            yield child

    monkeypatch.setattr(
        RandomDisplacementTransformation,
        "apply_transformation",
        duplicate_children,
    )

    result = generate_taylor_samples(
        [_reference("parent")],
        _random_config(n_structures=2),
    )

    assert result.requested_children == 2
    assert result.accepted_children == 1
    assert result.duplicate_skipped == 1
    assert result.unavailable_children == 0
    assert result.n_skipped == 1


def test_periodic_cell_and_metadata_are_preserved():
    """Sampling should retain periodic geometry while owning output arrays."""
    cell = np.diag([5.0, 6.0, 7.0])
    structure = AtomicStructure(
        coords=np.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]]),
        types=["H", "H"],
        avec=cell,
        energy=1.25,
        forces=np.array([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]),
    )
    reference = TaylorReference(parent_id="periodic", structure=structure)

    result = generate_taylor_samples(
        [reference],
        _random_config(n_structures=1),
    )

    assert all(record.structure.pbc for record in result.records)
    assert all(
        np.array_equal(record.structure.avec[0], cell)
        for record in result.records
    )
    assert all(record.structure.avec[0] is not cell for record in result.records)


def test_doptimal_sampling_is_backend_neutral():
    """The neutral core should orchestrate the existing D-optimal transform."""
    config = TaylorExpansionConfig(
        transformation=DOptimalDisplacementTransformation(
            rms=0.025,
            n_structures=4,
            max_iter=10,
            random_state=23,
            remove_translations=True,
            enforce_zero_mean=True,
        ),
        include_reference=False,
    )

    result = generate_taylor_samples([_reference("parent")], config)

    assert result.n_exact == 0
    assert result.accepted_children == 4
    displacements = np.array(
        [record.structure.coords[0] - _structure().coords[0]
         for record in result.records]
    )
    assert np.linalg.norm(displacements.mean(axis=0)) < 1.0e-6
    assert np.allclose(
        np.sqrt(np.mean(displacements**2, axis=(1, 2))),
        0.025,
        atol=1.0e-3,
    )


def test_reference_split_is_disjoint_and_preserves_source_order():
    """Parents should be split before augmentation without family leakage."""
    references = [_reference(f"parent-{index}") for index in range(10)]

    train, validation, test = split_reference_structures(
        references,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=99,
    )

    assert [len(train), len(validation), len(test)] == [6, 2, 2]
    ids = [{reference.parent_id for reference in split}
           for split in (train, validation, test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    for split in (train, validation, test):
        indices = [int(reference.parent_id.rsplit("-", 1)[-1])
                   for reference in split]
        assert indices == sorted(indices)


def test_neutral_taylor_api_imports_when_torch_is_blocked():
    """The authoritative geometry import path must not require PyTorch."""
    code = """
import builtins

original_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("torch intentionally blocked")
    return original_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from aenet.geometry.sampling import taylor_energy
assert taylor_energy(1.0, [[1.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]]) == 0.9
"""
    subprocess.run([sys.executable, "-c", code], check=True)
