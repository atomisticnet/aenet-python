"""Tests for regression-based reference-energy helpers."""

from pathlib import Path

import numpy as np
import pytest

from aenet.geometry.structure import AtomicStructure
from aenet.reference_energies import (
    ReferenceEnergies,
    iter_composition_energy_samples_from_files,
)


def _sample(composition: dict[str, int], energy: float):
    """Return one public regression sample."""
    return composition, energy


def _fixture_path(name: str) -> Path:
    """Return one parser fixture path."""
    return (
        Path(__file__).resolve().parents[1]
        / "formats"
        / "tests"
        / "fixtures"
        / name
    )


def test_reference_energies_from_regression_recovers_exact_solution():
    samples = [
        _sample({"A": 2}, 2.0),
        _sample({"B": 2}, 4.0),
        _sample({"A": 1, "B": 1}, 3.0),
        _sample({"A": 2, "B": 1}, 4.0),
    ]

    refs = ReferenceEnergies.from_regression(samples)

    assert refs.method == "regression"
    assert refs.atomic_energies == pytest.approx({"A": 1.0, "B": 2.0})
    assert refs.metadata["rank"] == 2
    assert refs.metadata["n_samples_total"] == 4
    assert refs.metadata["n_samples_used"] == 4
    assert refs.metadata["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_reference_energies_from_regression_uses_fixed_species_constraints():
    samples = [
        _sample({"A": 1, "C": 2}, 11.0),
        _sample({"B": 1, "C": 2}, 12.0),
    ]

    refs = ReferenceEnergies.from_regression(
        samples,
        fixed_atomic_energies={"C": 5.0},
    )

    assert refs.atomic_energies == pytest.approx(
        {"A": 1.0, "B": 2.0, "C": 5.0}
    )
    assert refs.metadata["fixed_atomic_energies"] == {"C": 5.0}
    assert refs.metadata["free_species"] == ["A", "B"]


def test_reference_energies_from_regression_raises_for_underdetermined_system():
    samples = [
        _sample({"A": 1, "C": 2}, 11.0),
        _sample({"B": 1, "C": 2}, 12.0),
    ]

    with pytest.raises(ValueError, match="underdetermined or rank-deficient"):
        ReferenceEnergies.from_regression(samples)


def test_reference_energies_from_regression_subset_selection_is_deterministic():
    samples = [
        _sample({"A": 1}, 0.95),
        _sample({"B": 1}, 2.05),
        _sample({"A": 1, "B": 1}, 3.02),
        _sample({"A": 2, "B": 1}, 3.91),
        _sample({"A": 1, "B": 2}, 5.10),
        _sample({"A": 2, "B": 2}, 6.00),
    ]

    refs_seed_a = ReferenceEnergies.from_regression(
        samples,
        subset_size=4,
        random_seed=7,
    )
    refs_seed_b = ReferenceEnergies.from_regression(
        samples,
        subset_size=4,
        random_seed=7,
    )
    refs_other_seed = ReferenceEnergies.from_regression(
        samples,
        subset_size=4,
        random_seed=13,
    )

    assert refs_seed_a.atomic_energies == pytest.approx(
        refs_seed_b.atomic_energies
    )
    assert refs_seed_a.metadata["n_samples_used"] == 4
    assert refs_seed_a.metadata["random_seed"] == 7
    assert refs_seed_a.atomic_energies != pytest.approx(
        refs_other_seed.atomic_energies
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"subset_size": 0}, "subset_size must be a positive integer"),
        ({"subset_size": 1.5}, "subset_size must be a positive integer"),
        (
            {"subset_size": 1, "subset_fraction": 0.5},
            "Specify at most one of subset_size and subset_fraction",
        ),
        (
            {"subset_fraction": 0.0},
            "subset_fraction must be a finite float in the interval",
        ),
        (
            {"subset_fraction": 0.5},
            "subset_fraction requires a sized sample collection",
        ),
    ],
)
def test_reference_energies_from_regression_validates_subset_arguments(
    kwargs,
    match,
):
    if "subset_fraction" in kwargs and kwargs.get("subset_size") is None:
        samples = (sample for sample in [_sample({"A": 1}, 1.0)])
    else:
        samples = [
            _sample({"A": 1}, 1.0),
            _sample({"B": 1}, 2.0),
        ]

    with pytest.raises(ValueError, match=match):
        ReferenceEnergies.from_regression(samples, **kwargs)


def test_reference_energies_from_regression_raises_for_oversized_subset():
    samples = [
        _sample({"A": 1}, 1.0),
        _sample({"B": 1}, 2.0),
    ]

    with pytest.raises(
        ValueError,
        match="subset_size cannot exceed the number of available samples",
    ):
        ReferenceEnergies.from_regression(samples, subset_size=3)


def test_reference_energies_from_regression_accepts_lazy_iterators():
    samples = (
        _sample(composition, energy)
        for composition, energy in [
            ({"H": 2}, 2.0),
            ({"H": 3}, 3.0),
        ]
    )

    refs = ReferenceEnergies.from_regression(samples)

    assert refs.atomic_energies == pytest.approx({"H": 1.0})
    assert refs.metadata["species_order"] == ["H"]


def test_iter_composition_energy_samples_from_files_yields_all_frames():
    path = _fixture_path("fhiaims-03.out")
    samples = list(iter_composition_energy_samples_from_files([path]))

    assert len(samples) == 2
    assert samples[0][0] == samples[1][0]
    assert samples[0][1] == pytest.approx(-23087739.7518042)
    assert samples[1][1] == pytest.approx(-23087739.7518038)


def test_reference_energies_from_regression_can_fit_file_samples(tmp_path):
    path_a = tmp_path / "sample_a.xsf"
    path_b = tmp_path / "sample_b.xsf"
    AtomicStructure(
        coords=np.zeros((1, 3), dtype=np.float64),
        types=["H"],
        energy=1.0,
    ).to_file(path_a)
    AtomicStructure(
        coords=np.zeros((1, 3), dtype=np.float64),
        types=["H"],
        energy=1.0,
    ).to_file(path_b)

    refs = ReferenceEnergies.from_regression(
        iter_composition_energy_samples_from_files([path_a, path_b])
    )

    assert refs.atomic_energies == pytest.approx({"H": 1.0})
    assert refs.metadata["n_samples_total"] == 2


def test_reference_energies_atomic_energies_property_returns_copy():
    refs = ReferenceEnergies.from_regression(
        [
            _sample({"A": 1}, 1.0),
            _sample({"A": 2}, 2.0),
        ]
    )

    mapping = refs.atomic_energies
    mapping["A"] = 99.0

    assert refs.atomic_energies["A"] == pytest.approx(1.0)


def test_reference_energies_from_regression_rejects_fractional_compositions():
    samples = [
        _sample({"A": 1.5}, 1.0),
        _sample({"A": 3}, 2.0),
    ]

    with pytest.raises(ValueError, match="Composition counts must be integer-valued"):
        ReferenceEnergies.from_regression(samples)


def test_reference_energies_from_reference_compounds_solves_exact_system():
    samples = [
        _sample({"A": 2}, 2.0),
        _sample({"B": 2}, 4.0),
    ]

    refs = ReferenceEnergies.from_reference_compounds(
        samples,
        reference_compounds=["A2", "B2"],
    )

    assert refs.method == "reference_compounds"
    assert refs.atomic_energies == pytest.approx({"A": 1.0, "B": 2.0})
    assert refs.metadata["reference_compounds"] == [
        {"A": 2},
        {"B": 2},
    ]
    assert refs.metadata["reference_candidate_counts"] == [1, 1]


def test_reference_energies_from_reference_compounds_uses_lowest_energy():
    samples = [
        _sample({"A": 2}, 2.5),
        _sample({"A": 2}, 2.0),
        _sample({"B": 2}, 4.0),
    ]

    refs = ReferenceEnergies.from_reference_compounds(
        samples,
        reference_compounds=["A2", "B2"],
    )

    assert refs.atomic_energies == pytest.approx({"A": 1.0, "B": 2.0})
    assert refs.metadata["reference_candidate_counts"] == [2, 1]
    assert refs.metadata["selected_reference_samples"][0]["energy"] == pytest.approx(
        2.0
    )


def test_reference_energies_from_reference_compounds_supports_fixed_species():
    samples = [
        _sample({"A": 1, "C": 2}, 11.0),
        _sample({"B": 1, "C": 2}, 12.0),
    ]

    refs = ReferenceEnergies.from_reference_compounds(
        samples,
        reference_compounds=["AC2", "BC2"],
        fixed_atomic_energies={"C": 5.0},
    )

    assert refs.atomic_energies == pytest.approx(
        {"A": 1.0, "B": 2.0, "C": 5.0}
    )
    assert refs.metadata["free_species"] == ["A", "B"]


def test_reference_energies_from_reference_compounds_raises_when_underdetermined():
    samples = [
        _sample({"A": 1, "C": 2}, 11.0),
        _sample({"B": 1, "C": 2}, 12.0),
    ]

    with pytest.raises(ValueError, match="underdetermined or rank-deficient"):
        ReferenceEnergies.from_reference_compounds(
            samples,
            reference_compounds=["AC2", "BC2"],
        )


def test_reference_energies_from_reference_compounds_raises_for_missing_compound():
    samples = [_sample({"A": 2}, 2.0)]

    with pytest.raises(ValueError, match="Missing requested reference compounds"):
        ReferenceEnergies.from_reference_compounds(
            samples,
            reference_compounds=["A2", "B2"],
        )


def test_reference_energies_from_reference_compounds_rejects_duplicates():
    samples = [
        _sample({"A": 2}, 2.0),
        _sample({"B": 2}, 4.0),
    ]

    with pytest.raises(ValueError, match="duplicate or equivalent"):
        ReferenceEnergies.from_reference_compounds(
            samples,
            reference_compounds=["A2", {"A": 2}],
        )


def test_reference_energies_from_reference_compounds_rejects_fractional_mapping():
    samples = [
        _sample({"A": 2}, 2.0),
        _sample({"B": 2}, 4.0),
    ]

    with pytest.raises(ValueError, match="Composition counts must be integer-valued"):
        ReferenceEnergies.from_reference_compounds(
            samples,
            reference_compounds=[{"A": 1.5}, "B2"],
        )
