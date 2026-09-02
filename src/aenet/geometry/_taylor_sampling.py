"""Backend-neutral force-informed Taylor energy sampling."""

from __future__ import annotations

import copy
import hashlib
import pickle
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

from .structure import AtomicStructure
from .transformations import (
    DOptimalDisplacementTransformation,
    RandomDisplacementTransformation,
)

_SupportedTransformation = Union[
    RandomDisplacementTransformation,
    DOptimalDisplacementTransformation,
]
_ZeroForcePolicy = Literal["skip", "keep", "error"]
_DuplicatePolicy = Literal["skip", "keep", "error"]


def taylor_energy(
    parent_energy: float,
    forces: np.ndarray,
    displacement: np.ndarray,
) -> float:
    """Return a first-order Taylor energy for an applied displacement.

    The force convention is ``F = -dE/dR``, giving
    ``E_child = E_parent - sum(delta_R * F)``.

    Parameters
    ----------
    parent_energy
        Exact reference energy in eV.
    forces
        Reference forces with shape ``(N, 3)`` in eV/Angstrom.
    displacement
        Applied Cartesian displacement with shape ``(N, 3)`` in Angstrom.

    Returns
    -------
    float
        Taylor-expanded total energy in eV.

    Raises
    ------
    ValueError
        If the energy, forces, or displacement are non-finite or the arrays do
        not have matching ``(N, 3)`` shapes.
    """
    try:
        energy = float(parent_energy)
    except (TypeError, ValueError) as exc:
        raise ValueError("parent_energy must be a finite energy") from exc
    if not np.isfinite(energy):
        raise ValueError("parent_energy must be a finite energy")

    force_array = np.asarray(forces, dtype=np.float64)
    displacement_array = np.asarray(displacement, dtype=np.float64)
    if force_array.shape != displacement_array.shape:
        raise ValueError("forces and displacement must have the same shape")
    if force_array.ndim != 2 or force_array.shape[1] != 3:
        raise ValueError(
            "forces and displacement must both have shape (N, 3)"
        )
    if not np.isfinite(force_array).all():
        raise ValueError("forces must contain only finite values")
    if not np.isfinite(displacement_array).all():
        raise ValueError("displacement must contain only finite values")

    correction = -np.sum(displacement_array * force_array, dtype=np.float64)
    child_energy = energy + float(correction)
    if not np.isfinite(child_energy):
        raise ValueError("Taylor-expanded energy is not finite")
    return child_energy


@dataclass(frozen=True)
class TaylorReference:
    """Identify one exact, single-frame reference configuration.

    Parameters
    ----------
    parent_id
        Non-empty stable identifier. It defines provenance and the independent
        per-parent random stream, so it must not depend on collection position.
    structure
        One-frame atomic structure carrying an exact total energy and forces.

    Notes
    -----
    The contained structure remains caller-owned and mutable. Sample generation
    never mutates it and returns independently owned structures.
    """

    parent_id: str
    structure: AtomicStructure

    def __post_init__(self) -> None:
        if not isinstance(self.parent_id, str) or not self.parent_id.strip():
            raise ValueError("parent_id must be a non-empty string")
        if not isinstance(self.structure, AtomicStructure):
            raise TypeError("structure must be an AtomicStructure")
        if self.structure.nframes != 1:
            raise ValueError("Taylor references must contain exactly one frame")


@dataclass(frozen=True)
class TaylorExpansionConfig:
    """Configure transformation-backed Taylor augmentation.

    Parameters
    ----------
    transformation
        Random or D-optimal displacement transformation. It is copied at
        construction and treated as an immutable random-state prototype.
    include_reference
        Include an independent exact-parent copy before its derived children.
    zero_force_tolerance
        Euclidean norm in eV/Angstrom at or below which ``zero_force_policy``
        applies.
    zero_force_policy
        ``"skip"`` omits derived children, ``"keep"`` assigns unchanged
        first-order energies, and ``"error"`` rejects the collection.
    duplicate_tolerance
        Absolute Cartesian tolerance in Angstrom used to detect duplicate
        displacements within one parent.
    duplicate_policy
        ``"skip"``, ``"keep"``, or ``"error"`` for duplicate children.

    Notes
    -----
    Per-parent streams are derived from the prototype generator state, optional
    namespace, and stable parent ID. The prototype and caller-owned generator
    are not advanced. Reproducibility is scoped to compatible NumPy, SciPy, and
    transformation implementations.
    """

    transformation: _SupportedTransformation
    include_reference: bool = True
    zero_force_tolerance: float = 0.0
    zero_force_policy: _ZeroForcePolicy = "skip"
    duplicate_tolerance: float = 1.0e-12
    duplicate_policy: _DuplicatePolicy = "skip"

    def __post_init__(self) -> None:
        if not isinstance(
            self.transformation,
            (
                RandomDisplacementTransformation,
                DOptimalDisplacementTransformation,
            ),
        ):
            raise TypeError(
                "transformation must be RandomDisplacementTransformation or "
                "DOptimalDisplacementTransformation"
            )
        zero_force_tolerance = float(self.zero_force_tolerance)
        duplicate_tolerance = float(self.duplicate_tolerance)
        if not np.isfinite(zero_force_tolerance) or zero_force_tolerance < 0.0:
            raise ValueError("zero_force_tolerance must be finite and non-negative")
        if not np.isfinite(duplicate_tolerance) or duplicate_tolerance < 0.0:
            raise ValueError("duplicate_tolerance must be finite and non-negative")
        if self.zero_force_policy not in {"skip", "keep", "error"}:
            raise ValueError(
                "zero_force_policy must be 'skip', 'keep', or 'error'"
            )
        if self.duplicate_policy not in {"skip", "keep", "error"}:
            raise ValueError(
                "duplicate_policy must be 'skip', 'keep', or 'error'"
            )
        object.__setattr__(self, "zero_force_tolerance", zero_force_tolerance)
        object.__setattr__(self, "duplicate_tolerance", duplicate_tolerance)
        object.__setattr__(
            self,
            "transformation",
            copy.deepcopy(self.transformation),
        )

    @property
    def strategy(self) -> Literal["random", "d_optimal"]:
        """Return the stable sampling-strategy label."""
        if isinstance(
            self.transformation,
            DOptimalDisplacementTransformation,
        ):
            return "d_optimal"
        return "random"

    @property
    def requested_children(self) -> int | None:
        """Return the explicitly requested child count, if configured."""
        if isinstance(
            self.transformation,
            DOptimalDisplacementTransformation,
        ):
            return int(self.transformation.n_structures)
        if self.transformation.max_structures is None:
            return None
        return int(self.transformation.max_structures)


@dataclass(frozen=True)
class TaylorSampleRecord:
    """Describe one exact or Taylor-derived configuration.

    Parameters
    ----------
    structure
        Independently owned one-frame structure. Derived structures carry an
        approximate energy and no force label.
    parent_id
        Stable identity of the exact reference family.
    parent_index
        Parent's position in this generation call. This records output order
        but is not used for random-state derivation.
    child_index
        Transformation output position, or ``None`` for the exact parent.
    strategy
        Stable random or D-optimal sampling label.
    label_origin
        ``"exact"`` for a retained parent or ``"taylor"`` for an approximate
        child.
    delta_energy
        Child-minus-parent total energy in eV.
    displacement_rms
        RMS over all Cartesian displacement components in Angstrom.
    maximum_displacement
        Maximum per-atom displacement norm in Angstrom.
    """

    structure: AtomicStructure
    parent_id: str
    parent_index: int
    child_index: int | None
    strategy: Literal["random", "d_optimal"]
    label_origin: Literal["exact", "taylor"]
    delta_energy: float
    displacement_rms: float
    maximum_displacement: float


@dataclass(frozen=True)
class TaylorSamplingResult:
    """Hold generated records and auditable collection-level counts.

    Parameters
    ----------
    records
        Exact and derived records in deterministic parent/child order.
    config
        Configuration snapshot used for generation.
    n_parents
        Number of exact reference parents processed.
    requested_children
        Total native transformation output count requested across parents.
    accepted_children
        Number of derived records retained.
    duplicate_skipped
        Number of produced duplicate displacements omitted by policy.
    zero_force_skipped
        Number of requested children omitted by zero-force policy.
    unavailable_children
        Number of requested children not yielded by a transformation.
    """

    records: tuple[TaylorSampleRecord, ...]
    config: TaylorExpansionConfig
    n_parents: int
    requested_children: int
    accepted_children: int
    duplicate_skipped: int = 0
    zero_force_skipped: int = 0
    unavailable_children: int = 0

    @property
    def structures(self) -> list[AtomicStructure]:
        """Return generated structures in stable record order."""
        return [record.structure for record in self.records]

    @property
    def n_exact(self) -> int:
        """Return the number of retained exact-parent records."""
        return sum(record.label_origin == "exact" for record in self.records)

    @property
    def n_derived(self) -> int:
        """Return the number of Taylor-derived records."""
        return self.accepted_children

    @property
    def n_skipped(self) -> int:
        """Return the total number of requested children not retained."""
        return (
            self.duplicate_skipped
            + self.zero_force_skipped
            + self.unavailable_children
        )


def _frame_forces(structure: AtomicStructure, index: int) -> np.ndarray:
    """Return one validated force array."""
    try:
        raw_forces = structure.forces[0]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"parent {index} must have a force array") from exc
    if raw_forces is None or np.asarray(raw_forces).size == 0:
        raise ValueError(f"parent {index} must have a force array")
    return np.asarray(raw_forces)


def _validate_reference(reference: TaylorReference, index: int) -> None:
    """Validate labels and coordinates required for augmentation."""
    structure = reference.structure
    try:
        energy = float(structure.energy[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"parent {index} must have a finite energy") from exc
    if not np.isfinite(energy):
        raise ValueError(f"parent {index} must have a finite energy")

    positions = np.asarray(structure.coords[0])
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError(f"parent {index} positions must have shape (N, 3)")
    if not np.isfinite(positions).all():
        raise ValueError(f"parent {index} positions must be finite")

    forces = _frame_forces(structure, index)
    if forces.shape != positions.shape:
        raise ValueError(f"parent {index} force array must match positions shape")
    if not np.isfinite(forces).all():
        raise ValueError(f"parent {index} must have finite forces")

    if structure.pbc:
        cell = np.asarray(structure.avec[0])
        if cell.shape != (3, 3) or not np.isfinite(cell).all():
            raise ValueError(f"parent {index} cell must be a finite (3, 3) array")


def _validate_references(
    references: Sequence[TaylorReference],
    config: TaylorExpansionConfig,
) -> None:
    """Validate a complete reference collection before generation."""
    if not all(isinstance(reference, TaylorReference) for reference in references):
        raise TypeError("parents must contain only TaylorReference objects")
    counts = Counter(reference.parent_id for reference in references)
    duplicates = sorted(parent_id for parent_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "parent_id values must be unique; duplicates: " + ", ".join(duplicates)
        )
    for index, reference in enumerate(references):
        _validate_reference(reference, index)
        if (
            reference.structure.natoms == 1
            and config.transformation.remove_translations
        ):
            raise ValueError(
                f"parent {index} has no internal displacement after translation "
                "removal; disable remove_translations or provide multiple atoms"
            )
        force_norm = float(np.linalg.norm(_frame_forces(reference.structure, index)))
        if (
            force_norm <= config.zero_force_tolerance
            and config.zero_force_policy == "error"
        ):
            raise ValueError(
                f"parent {reference.parent_id!r} is at or below the zero-force "
                "tolerance"
            )


def _requested_children(
    structure: AtomicStructure,
    config: TaylorExpansionConfig,
) -> int:
    """Resolve the native output request for one parent."""
    requested = config.requested_children
    if requested is not None:
        return requested
    dimensions = 3 * structure.natoms
    if config.transformation.remove_translations:
        return max(0, dimensions - 3)
    return dimensions


def _seeded_transformation(
    config: TaylorExpansionConfig,
    *,
    parent_id: str,
    parent_namespace: str | None,
) -> _SupportedTransformation:
    """Return an independently seeded transformation for one stable ID."""
    transformation = copy.deepcopy(config.transformation)
    rng = transformation.rng
    state = pickle.dumps(
        rng.bit_generator.state,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    identity = f"{parent_namespace or ''}\0{parent_id}".encode()
    digest = hashlib.sha256(state + identity).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    bit_generator_type = type(rng.bit_generator)
    try:
        bit_generator = bit_generator_type(seed)
    except TypeError:
        bit_generator = np.random.PCG64(seed)
    transformation.rng = np.random.Generator(bit_generator)
    return transformation


def _copy_structure(
    structure: AtomicStructure,
    *,
    energy: float,
    forces: np.ndarray | None,
    name: str,
) -> AtomicStructure:
    """Create an independent one-frame structure with explicit labels."""
    copied = AtomicStructure(
        coords=np.array(structure.coords[0], dtype=np.float64, copy=True),
        types=list(structure.types),
        avec=(
            np.array(structure.avec[0], dtype=np.float64, copy=True)
            if structure.pbc
            else None
        ),
        energy=float(energy),
        forces=(
            None
            if forces is None
            else np.array(forces, dtype=np.float64, copy=True)
        ),
        fixed=np.array(structure.fixed, dtype=bool, copy=True),
    )
    copied.comments = copy.deepcopy(structure.comments)
    copied.name = name
    return copied


def generate_taylor_samples(
    parents: Sequence[TaylorReference],
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
) -> TaylorSamplingResult:
    """Generate exact and Taylor-derived records for reference parents.

    Parameters
    ----------
    parents
        Ordered, uniquely identified, single-frame exact references. The
        collection is fully validated before any transformation runs.
    config
        Transformation and label-handling policy.
    parent_namespace
        Optional stable namespace used only to separate random streams for
        identical parent IDs in independently managed collections.

    Returns
    -------
    TaylorSamplingResult
        Independently owned records and auditable generation counts.

    Raises
    ------
    TypeError
        If the collection contains values other than `TaylorReference`.
    ValueError
        If identities, reference labels, coordinates, transformation output,
        or configured policies are invalid.
    """
    references = list(parents)
    _validate_references(references, config)
    records: list[TaylorSampleRecord] = []
    requested_total = 0
    accepted_total = 0
    duplicate_skipped = 0
    zero_force_skipped = 0
    unavailable_total = 0

    for parent_index, reference in enumerate(references):
        parent = reference.structure
        parent_id = reference.parent_id
        parent_energy = float(parent.energy[0])
        parent_forces = np.asarray(parent.forces[0], dtype=np.float64)
        requested = _requested_children(parent, config)
        requested_total += requested

        if config.include_reference:
            records.append(
                TaylorSampleRecord(
                    structure=_copy_structure(
                        parent,
                        energy=parent_energy,
                        forces=parent_forces,
                        name=f"{parent_id}::exact",
                    ),
                    parent_id=parent_id,
                    parent_index=parent_index,
                    child_index=None,
                    strategy=config.strategy,
                    label_origin="exact",
                    delta_energy=0.0,
                    displacement_rms=0.0,
                    maximum_displacement=0.0,
                )
            )

        force_norm = float(np.linalg.norm(parent_forces))
        if (
            force_norm <= config.zero_force_tolerance
            and config.zero_force_policy == "skip"
        ):
            zero_force_skipped += requested
            continue

        transformation = _seeded_transformation(
            config,
            parent_id=parent_id,
            parent_namespace=parent_namespace,
        )
        produced = 0
        accepted_displacements: list[np.ndarray] = []
        children = transformation.apply_transformation(parent)
        for child_index, transformed in enumerate(children):
            produced += 1
            child_positions = np.asarray(transformed.coords[-1], dtype=np.float64)
            displacement = child_positions - np.asarray(parent.coords[0])
            if displacement.shape != np.asarray(parent.coords[0]).shape:
                raise ValueError(
                    "transformation returned coordinates with a changed atom "
                    "count or shape"
                )
            if not np.isfinite(displacement).all():
                raise ValueError("transformation returned a non-finite displacement")

            duplicate = any(
                np.allclose(
                    displacement,
                    accepted,
                    rtol=0.0,
                    atol=config.duplicate_tolerance,
                )
                for accepted in accepted_displacements
            )
            if duplicate and config.duplicate_policy == "error":
                raise ValueError(
                    f"transformation returned duplicate child {child_index} "
                    f"for parent {parent_id!r}"
                )
            if duplicate and config.duplicate_policy == "skip":
                duplicate_skipped += 1
                continue
            accepted_displacements.append(np.array(displacement, copy=True))

            child_energy = taylor_energy(
                parent_energy,
                parent_forces,
                displacement,
            )
            atom_norms = np.linalg.norm(displacement, axis=1)
            child = _copy_structure(
                parent,
                energy=child_energy,
                forces=None,
                name=(
                    f"{parent_id}::taylor:{config.strategy}:"
                    f"{child_index:06d}"
                ),
            )
            child.coords[0] = np.array(child_positions, dtype=np.float64, copy=True)
            records.append(
                TaylorSampleRecord(
                    structure=child,
                    parent_id=parent_id,
                    parent_index=parent_index,
                    child_index=child_index,
                    strategy=config.strategy,
                    label_origin="taylor",
                    delta_energy=child_energy - parent_energy,
                    displacement_rms=float(np.sqrt(np.mean(displacement**2))),
                    maximum_displacement=float(np.max(atom_norms)),
                )
            )
            accepted_total += 1
        unavailable_total += max(0, requested - produced)

    return TaylorSamplingResult(
        records=tuple(records),
        config=copy.deepcopy(config),
        n_parents=len(references),
        requested_children=requested_total,
        accepted_children=accepted_total,
        duplicate_skipped=duplicate_skipped,
        zero_force_skipped=zero_force_skipped,
        unavailable_children=unavailable_total,
    )


def iter_taylor_records(
    parents: Sequence[TaylorReference],
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
) -> Iterator[TaylorSampleRecord]:
    """Yield Taylor records in deterministic order.

    Parameters
    ----------
    parents
        Ordered exact reference records.
    config
        Transformation and label-handling policy.
    parent_namespace
        Optional stable random-stream namespace.

    Yields
    ------
    TaylorSampleRecord
        Exact and derived records after complete collection validation and
        materialization.
    """
    yield from generate_taylor_samples(
        parents,
        config,
        parent_namespace=parent_namespace,
    ).records


def iter_taylor_structures(
    parents: Sequence[TaylorReference],
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
) -> Iterator[AtomicStructure]:
    """Yield generated atomic structures in deterministic order.

    Parameters
    ----------
    parents
        Ordered exact reference records.
    config
        Transformation and label-handling policy.
    parent_namespace
        Optional stable random-stream namespace.

    Yields
    ------
    AtomicStructure
        Independently owned exact and Taylor-derived structures.
    """
    for record in iter_taylor_records(
        parents,
        config,
        parent_namespace=parent_namespace,
    ):
        yield record.structure


def split_reference_structures(
    parents: Sequence[TaylorReference],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int | None = None,
) -> tuple[list[TaylorReference], list[TaylorReference], list[TaylorReference]]:
    """Split exact parents before augmentation while preserving source order.

    Parameters
    ----------
    parents
        Ordered, uniquely identified exact references.
    validation_fraction
        Fraction assigned to validation, in ``[0, 1)``.
    test_fraction
        Fraction assigned to the held-out test split, in ``[0, 1)``. The sum
        with ``validation_fraction`` must be less than one.
    seed
        Seed for the independent NumPy split generator.

    Returns
    -------
    tuple of list
        Training, validation, and test references. Each list retains source
        order and no parent family crosses a split.

    Raises
    ------
    TypeError
        If a value is not a `TaylorReference`.
    ValueError
        If identities or fractions are invalid or no training parent remains.
    """
    references = list(parents)
    if not all(isinstance(reference, TaylorReference) for reference in references):
        raise TypeError("parents must contain only TaylorReference objects")
    counts = Counter(reference.parent_id for reference in references)
    if any(count > 1 for count in counts.values()):
        raise ValueError("parent_id values must be unique")

    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validation_fraction + test_fraction must be less than 1"
        )

    n_parents = len(references)
    permutation = np.random.default_rng(seed).permutation(n_parents)
    n_validation = int(round(n_parents * validation_fraction))
    n_test = int(round(n_parents * test_fraction))
    if n_validation + n_test >= n_parents and n_parents > 0:
        raise ValueError("split fractions leave no training parents")

    validation_indices = set(permutation[:n_validation].tolist())
    test_indices = set(
        permutation[n_validation:n_validation + n_test].tolist()
    )
    train_indices = set(range(n_parents)) - validation_indices - test_indices

    def _select(indices: set[int]) -> list[TaylorReference]:
        return [references[index] for index in sorted(indices)]

    return (
        _select(train_indices),
        _select(validation_indices),
        _select(test_indices),
    )
