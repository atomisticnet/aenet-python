"""Helpers for constructing atomic reference energies.

This module centers the regression workflow on lightweight
``(composition, energy)`` samples, so users can stream data from files,
databases, or custom parsers without materializing full structure objects in
memory. File-backed convenience adapters are provided separately for workflows
that already rely on :mod:`aenet.io.structure`.
"""

from __future__ import annotations

import copy
import math
import random
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from os import PathLike
from typing import Any

import numpy as np

from .geometry.structure import AtomicStructure
from .io.structure import read as read_structure

__all__ = ["ReferenceEnergies", "iter_composition_energy_samples_from_files"]


@dataclass(frozen=True)
class _EnergySample:
    """Lightweight composition/energy sample used by regression helpers."""

    composition: dict[str, int]
    energy: float


@dataclass(frozen=True)
class _SolveResult:
    """Internal result of one constrained reference-energy solve."""

    atomic_energies: dict[str, float]
    species_order: list[str]
    fixed_atomic_energies: dict[str, float]
    free_species: list[str]
    rank: int
    residual_sum_squares: float
    rmse: float
    singular_values: list[float]


def _validated_energy(value: Any) -> float:
    """Return one finite energy value."""
    try:
        energy = float(value)
    except Exception as exc:
        raise ValueError(
            "Regression requires finite total energies for every sample."
        ) from exc

    if not math.isfinite(energy):
        raise ValueError(
            "Regression requires finite total energies for every sample."
        )
    return energy


def _normalize_composition(
    composition: Mapping[Any, Any],
) -> dict[str, int]:
    """Validate and normalize one composition mapping."""
    if not isinstance(composition, Mapping):
        raise TypeError("Each sample composition must be a mapping.")

    normalized: dict[str, int] = {}
    for species, count in composition.items():
        key = str(species)
        if isinstance(count, bool):
            raise ValueError("Composition counts must be integer-valued.")
        if isinstance(count, Integral):
            value = int(count)
        elif isinstance(count, Real):
            numeric = float(count)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError("Composition counts must be integer-valued.")
            value = int(numeric)
        else:
            raise ValueError(
                "Composition counts must be integer-valued."
            )
        if value < 0:
            raise ValueError("Composition counts must be non-negative.")
        normalized[key] = value

    if sum(normalized.values()) <= 0:
        raise ValueError(
            "Each sample composition must contain at least one atom."
        )
    return normalized


def _normalize_energy_sample(sample: Any) -> _EnergySample:
    """Normalize one public regression sample."""
    if isinstance(sample, _EnergySample):
        return sample

    if isinstance(sample, tuple) and len(sample) == 2:
        composition, energy = sample
        return _EnergySample(
            composition=_normalize_composition(composition),
            energy=_validated_energy(energy),
        )

    raise TypeError(
        "Each regression sample must be a ``(composition, energy)`` pair."
    )


def _formula_to_composition(formula: str) -> dict[str, int]:
    """Parse a simple chemical formula into a composition mapping."""
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(
            "Reference compounds must be non-empty formula strings or "
            "composition mappings."
        )

    token_pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
    normalized = formula.strip()
    composition: dict[str, int] = {}
    position = 0

    for match in token_pattern.finditer(normalized):
        if match.start() != position:
            raise ValueError(
                f"Invalid reference compound formula {formula!r}."
            )
        species = match.group(1)
        count_text = match.group(2)
        count = int(count_text) if count_text else 1
        if count <= 0:
            raise ValueError(
                f"Invalid reference compound formula {formula!r}."
            )
        composition[species] = composition.get(species, 0) + count
        position = match.end()

    if position != len(normalized) or not composition:
        raise ValueError(
            f"Invalid reference compound formula {formula!r}."
        )

    return composition


def _normalize_reference_compound(compound: Any) -> dict[str, int]:
    """Normalize one reference-compound specification."""
    if isinstance(compound, Mapping):
        return _normalize_composition(compound)
    if isinstance(compound, str):
        return _normalize_composition(_formula_to_composition(compound))
    raise TypeError(
        "Each reference compound must be a formula string or a composition "
        "mapping."
    )


def _composition_key(
    composition: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    """Return a hashable canonical key for one composition."""
    return tuple(
        (species, int(count))
        for species, count in sorted(composition.items())
        if int(count) != 0
    )


def _solve_atomic_energies(
    samples: Sequence[_EnergySample],
    *,
    fixed_atomic_energies: dict[str, float] | None,
) -> _SolveResult:
    """Solve one constrained atomic-energy system from normalized samples."""
    if len(samples) == 0:
        raise ValueError("At least one regression sample is required.")

    seen_species: set[str] = set()
    for sample in samples:
        seen_species.update(sample.composition.keys())
    species_order = sorted(seen_species)

    fixed = _normalize_fixed_atomic_energies(
        fixed_atomic_energies,
        species_order=species_order,
    )

    composition_rows = [
        _composition_row(sample.composition, species_order=species_order)
        for sample in samples
    ]
    energies = [sample.energy for sample in samples]

    composition_matrix = np.stack(composition_rows)
    energy_vector = np.array(energies, dtype=np.float64)

    fixed_species = [species for species in species_order if species in fixed]
    free_species = [species for species in species_order if species not in fixed]

    energy_offset = np.zeros(len(samples), dtype=np.float64)
    if fixed_species:
        fixed_indices = [species_order.index(species) for species in fixed_species]
        fixed_values = np.array(
            [fixed[species] for species in fixed_species],
            dtype=np.float64,
        )
        energy_offset = composition_matrix[:, fixed_indices] @ fixed_values

    reduced_targets = energy_vector - energy_offset

    rank = 0
    singular_values: list[float] = []
    residual_sum_squares = 0.0
    resolved = dict(fixed)

    if free_species:
        free_indices = [species_order.index(species) for species in free_species]
        free_matrix = composition_matrix[:, free_indices]
        solution, residuals, rank, singular = np.linalg.lstsq(
            free_matrix,
            reduced_targets,
            rcond=None,
        )
        singular_values = [float(value) for value in singular.tolist()]

        if int(rank) < len(free_species):
            raise ValueError(
                "Reference-energy system is underdetermined or rank-deficient. "
                "Provide fixed_atomic_energies for one or more species."
            )

        predicted = free_matrix @ solution
        residual_sum_squares = float(np.sum((predicted - reduced_targets) ** 2))
        for species, energy in zip(free_species, solution.tolist()):
            resolved[species] = float(energy)
    else:
        predicted = np.zeros(len(samples), dtype=np.float64)
        rank = 0
        residual_sum_squares = float(np.sum((predicted - reduced_targets) ** 2))

    rmse = math.sqrt(residual_sum_squares / float(len(samples)))

    return _SolveResult(
        atomic_energies=resolved,
        species_order=list(species_order),
        fixed_atomic_energies=dict(fixed),
        free_species=list(free_species),
        rank=int(rank),
        residual_sum_squares=float(residual_sum_squares),
        rmse=float(rmse),
        singular_values=singular_values,
    )


def _select_reference_compound_samples(
    samples,
    *,
    reference_compounds: Sequence[Any],
) -> tuple[list[_EnergySample], int, list[dict[str, int]], list[int]]:
    """Select minimum-energy samples for the requested reference compounds."""
    if len(reference_compounds) == 0:
        raise ValueError("At least one reference compound is required.")

    normalized_references = [
        _normalize_reference_compound(compound)
        for compound in reference_compounds
    ]
    reference_keys = [_composition_key(comp) for comp in normalized_references]

    duplicates = {
        key for key in reference_keys if reference_keys.count(key) > 1
    }
    if duplicates:
        raise ValueError(
            "reference_compounds contains duplicate or equivalent "
            "compositions."
        )

    best_samples: dict[tuple[tuple[str, int], ...], _EnergySample] = {}
    candidate_counts = {key: 0 for key in reference_keys}
    n_samples_total = 0

    for raw_sample in samples:
        sample = _normalize_energy_sample(raw_sample)
        n_samples_total += 1
        sample_key = _composition_key(sample.composition)
        if sample_key not in candidate_counts:
            continue

        candidate_counts[sample_key] += 1
        previous = best_samples.get(sample_key)
        if previous is None or sample.energy < previous.energy:
            best_samples[sample_key] = sample

    if n_samples_total == 0:
        raise ValueError("At least one regression sample is required.")

    missing = [
        normalized_references[index]
        for index, key in enumerate(reference_keys)
        if key not in best_samples
    ]
    if missing:
        raise ValueError(
            "Missing requested reference compounds in the provided samples: "
            f"{missing!r}."
        )

    selected_samples = [best_samples[key] for key in reference_keys]
    counts_in_order = [candidate_counts[key] for key in reference_keys]
    return (
        selected_samples,
        n_samples_total,
        normalized_references,
        counts_in_order,
    )


def _composition_from_species_sequence(species: Sequence[Any]) -> dict[str, int]:
    """Return a composition mapping from a species list."""
    composition: dict[str, int] = {}
    for species_name in species:
        key = str(species_name)
        composition[key] = composition.get(key, 0) + 1
    return composition


def _iter_composition_energy_samples_from_atomic_structure(
    structure: AtomicStructure,
) -> Iterator[tuple[dict[str, int], float]]:
    """Yield one ``(composition, energy)`` sample per frame."""
    composition = _normalize_composition(structure.composition)
    for frame in range(structure.nframes):
        yield dict(composition), _validated_energy(structure.energy[frame])


def iter_composition_energy_samples_from_files(
    paths: PathLike[str] | str | Iterable[PathLike[str] | str],
    *,
    frmt: str | None = None,
    **read_kwargs,
) -> Iterator[tuple[dict[str, int], float]]:
    """Yield lazy ``(composition, energy)`` samples from structure files.

    Parameters
    ----------
    paths : path-like or iterable of path-like
        One path or an iterable of paths readable by :func:`aenet.io.structure.read`.
        Files containing multiple frames yield one sample per frame.
    frmt : str, optional
        Explicit input format forwarded to :func:`aenet.io.structure.read`.
    **read_kwargs
        Additional keyword arguments forwarded to
        :func:`aenet.io.structure.read`.

    Yields
    ------
    tuple[dict[str, int], float]
        One ``(composition, energy)`` pair per frame.
    """
    if isinstance(paths, (str, PathLike)):
        iterable: Iterable[PathLike[str] | str] = [paths]
    else:
        iterable = paths

    for path in iterable:
        structure = read_structure(path, frmt=frmt, **read_kwargs)
        yield from _iter_composition_energy_samples_from_atomic_structure(
            structure
        )


def _normalize_fixed_atomic_energies(
    fixed_atomic_energies: dict[str, float] | None,
    *,
    species_order: list[str],
) -> dict[str, float]:
    """Validate user-supplied fixed species energies."""
    if fixed_atomic_energies is None:
        return {}

    normalized = {
        str(species): float(energy)
        for species, energy in fixed_atomic_energies.items()
    }
    for species, energy in normalized.items():
        if species not in species_order:
            raise ValueError(
                f"fixed_atomic_energies contains unknown species {species!r}."
            )
        if not math.isfinite(energy):
            raise ValueError(
                "fixed_atomic_energies must contain only finite values."
            )
    return normalized


def _composition_row(
    composition: dict[str, int],
    *,
    species_order: list[str],
) -> np.ndarray:
    """Return the species-count row for one composition."""
    return np.array(
        [composition.get(species, 0) for species in species_order],
        dtype=np.float64,
    )


def _normalize_subset_size(
    *,
    subset_size: int | None,
    subset_fraction: float | None,
    n_hint: int | None = None,
) -> int | None:
    """Validate and normalize optional subset selection arguments."""
    if subset_size is not None and subset_fraction is not None:
        raise ValueError(
            "Specify at most one of subset_size and subset_fraction."
        )

    if subset_size is not None:
        if isinstance(subset_size, bool) or not isinstance(
            subset_size, Integral
        ):
            raise ValueError("subset_size must be a positive integer.")
        size = int(subset_size)
        if size <= 0:
            raise ValueError("subset_size must be a positive integer.")
        if n_hint is not None and size > n_hint:
            raise ValueError(
                "subset_size cannot exceed the number of available samples."
            )
        return size

    if subset_fraction is None:
        return None

    fraction = float(subset_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError(
            "subset_fraction must be a finite float in the interval (0, 1]."
        )
    if n_hint is None:
        raise ValueError(
            "subset_fraction requires a sized sample collection. Use "
            "subset_size for lazy iterators."
        )
    return max(1, int(math.ceil(n_hint * fraction)))


def _sample_count_hint(samples: Any) -> int | None:
    """Return an optional item-count hint for a sample iterable."""
    try:
        return len(samples)  # type: ignore[arg-type]
    except Exception:
        return None


def _select_regression_samples(
    samples,
    *,
    subset_size: int | None,
    subset_fraction: float | None,
    random_seed: int | None,
) -> tuple[list[_EnergySample], int, set[str]]:
    """Return the chosen regression samples and the total sample count."""
    normalized_subset_size = _normalize_subset_size(
        subset_size=subset_size,
        subset_fraction=subset_fraction,
        n_hint=_sample_count_hint(samples),
    )

    rng = random.Random(random_seed)
    selected: list[_EnergySample] = []
    n_samples_total = 0
    seen_species: set[str] = set()

    for raw_sample in samples:
        sample = _normalize_energy_sample(raw_sample)
        n_samples_total += 1
        seen_species.update(sample.composition.keys())

        if normalized_subset_size is None:
            selected.append(sample)
            continue

        if len(selected) < normalized_subset_size:
            selected.append(sample)
            continue

        replace_idx = rng.randint(0, n_samples_total - 1)
        if replace_idx < normalized_subset_size:
            selected[replace_idx] = sample

    if n_samples_total == 0:
        raise ValueError("At least one regression sample is required.")

    if (
        normalized_subset_size is not None
        and n_samples_total < normalized_subset_size
    ):
        raise ValueError(
            "subset_size cannot exceed the number of available samples."
        )

    return selected, n_samples_total, seen_species


@dataclass(frozen=True)
class ReferenceEnergies:
    """Resolved atomic reference energies with provenance metadata.

    Parameters
    ----------
    _atomic_energies : dict[str, float]
        Internal storage for the resolved species-energy mapping.
    method : str
        Name of the construction workflow.
    _metadata : dict, optional
        Workflow metadata and diagnostics.
    """

    _atomic_energies: dict[str, float]
    method: str
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Normalize stored mappings to plain finite floats."""
        normalized = {
            str(species): float(energy)
            for species, energy in self._atomic_energies.items()
        }
        for energy in normalized.values():
            if not math.isfinite(energy):
                raise ValueError(
                    "atomic_energies must contain only finite values."
                )
        object.__setattr__(self, "_atomic_energies", normalized)
        object.__setattr__(self, "_metadata", copy.deepcopy(self._metadata))

    @property
    def atomic_energies(self) -> dict[str, float]:
        """Return a copy of the resolved atomic reference energies."""
        return dict(self._atomic_energies)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy of the workflow metadata and diagnostics."""
        return copy.deepcopy(self._metadata)

    @classmethod
    def from_regression(
        cls,
        samples,
        *,
        fixed_atomic_energies: dict[str, float] | None = None,
        subset_size: int | None = None,
        subset_fraction: float | None = None,
        random_seed: int | None = None,
    ) -> ReferenceEnergies:
        """Estimate atomic energies from total-energy regression.

        Parameters
        ----------
        samples : iterable
            Iterable of ``(composition, energy)`` pairs. ``composition`` must
            be a mapping from species labels to integer counts, and ``energy``
            must be a finite total energy. The iterable may be lazy.
        fixed_atomic_energies : dict[str, float], optional
            User-specified species energies held fixed during the fit. This is
            required for underdetermined composition spaces unless the user
            explicitly chooses a different reference convention upstream.
        subset_size : int, optional
            Number of regression samples to draw uniformly without replacement
            before fitting. Mutually exclusive with ``subset_fraction``.
        subset_fraction : float, optional
            Fraction of regression samples to draw uniformly without
            replacement before fitting. Mutually exclusive with
            ``subset_size``.
        random_seed : int, optional
            Seed used for deterministic subset selection.

        Returns
        -------
        ReferenceEnergies
            Resolved atomic reference energies and fit metadata.

        Raises
        ------
        ValueError
            If the inputs are empty, contain invalid samples, specify invalid
            subset arguments, contain unknown fixed species, or remain
            underdetermined after applying user constraints.
        """
        selected_samples, n_samples_total, _ = _select_regression_samples(
            samples,
            subset_size=subset_size,
            subset_fraction=subset_fraction,
            random_seed=random_seed,
        )
        solve = _solve_atomic_energies(
            selected_samples,
            fixed_atomic_energies=fixed_atomic_energies,
        )
        metadata = {
            "species_order": list(solve.species_order),
            "fixed_atomic_energies": dict(solve.fixed_atomic_energies),
            "free_species": list(solve.free_species),
            "n_samples_total": int(n_samples_total),
            "n_samples_used": int(len(selected_samples)),
            "subset_size": int(len(selected_samples))
            if subset_size is not None or subset_fraction is not None
            else None,
            "subset_fraction": (
                float(subset_fraction)
                if subset_fraction is not None
                else None
            ),
            "random_seed": random_seed,
            "rank": int(solve.rank),
            "residual_sum_squares": float(solve.residual_sum_squares),
            "rmse": float(solve.rmse),
            "singular_values": list(solve.singular_values),
        }

        return cls(
            _atomic_energies=solve.atomic_energies,
            method="regression",
            _metadata=metadata,
        )

    @classmethod
    def from_reference_compounds(
        cls,
        samples,
        *,
        reference_compounds: Sequence[Any],
        fixed_atomic_energies: dict[str, float] | None = None,
    ) -> ReferenceEnergies:
        """Construct atomic energies from user-chosen reference compounds.

        Parameters
        ----------
        samples : iterable
            Iterable of ``(composition, energy)`` pairs. ``composition`` must
            be a mapping from species labels to integer counts, and ``energy``
            must be a finite total energy. The iterable may be lazy.
        reference_compounds : sequence
            Requested reference compositions, specified either as formula
            strings such as ``"TiO2"`` or as explicit composition mappings.
            If multiple matching samples exist for one requested composition,
            the lowest-energy sample is used.
        fixed_atomic_energies : dict[str, float], optional
            User-specified species energies held fixed during the solve.

        Returns
        -------
        ReferenceEnergies
            Resolved atomic reference energies and solver metadata.

        Raises
        ------
        ValueError
            If the inputs are empty, contain invalid samples, request missing
            or duplicate reference compounds, contain unknown fixed species, or
            remain underdetermined after applying user constraints.
        """
        (
            selected_samples,
            n_samples_total,
            normalized_references,
            candidate_counts,
        ) = _select_reference_compound_samples(
            samples,
            reference_compounds=reference_compounds,
        )

        solve = _solve_atomic_energies(
            selected_samples,
            fixed_atomic_energies=fixed_atomic_energies,
        )

        metadata = {
            "species_order": list(solve.species_order),
            "fixed_atomic_energies": dict(solve.fixed_atomic_energies),
            "free_species": list(solve.free_species),
            "n_samples_total": int(n_samples_total),
            "n_reference_compounds": int(len(normalized_references)),
            "reference_compounds": [dict(comp) for comp in normalized_references],
            "reference_candidate_counts": list(candidate_counts),
            "selected_reference_samples": [
                {
                    "composition": dict(sample.composition),
                    "energy": float(sample.energy),
                }
                for sample in selected_samples
            ],
            "rank": int(solve.rank),
            "residual_sum_squares": float(solve.residual_sum_squares),
            "rmse": float(solve.rmse),
            "singular_values": list(solve.singular_values),
        }

        return cls(
            _atomic_energies=solve.atomic_energies,
            method="reference_compounds",
            _metadata=metadata,
        )
