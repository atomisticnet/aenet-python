"""Force-informed local Taylor energy augmentation.

This module turns force-bearing reference structures into ordinary
energy-labelled structures. Coordinate generation is delegated to the existing
random and D-optimal transformations in :mod:`aenet.geometry.transformations`.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from aenet.geometry import AtomicStructure
from aenet.geometry.transformations import (
    DOptimalDisplacementTransformation,
    RandomDisplacementTransformation,
)

from .config import Structure
from .dataset import convert_to_structures
from .sources import (
    SourceCapabilities,
    SourceCollection,
    SourceRecord,
    coerce_source_collection,
)

__all__ = [
    "TaylorExpansionConfig",
    "TaylorSampleRecord",
    "TaylorSamplingResult",
    "TaylorSourceCollection",
    "generate_taylor_samples",
    "iter_taylor_records",
    "iter_taylor_structures",
    "split_reference_structures",
    "taylor_energy",
]

_SupportedTransformation = (
    RandomDisplacementTransformation | DOptimalDisplacementTransformation
)
_ZeroForcePolicy = Literal["skip", "keep", "error"]
_DuplicatePolicy = Literal["skip", "keep", "error"]


def taylor_energy(
    parent_energy: float,
    forces: np.ndarray,
    displacement: np.ndarray,
) -> float:
    """Return the first-order Taylor energy for an applied displacement.

    The package force convention is ``F = -dE/dR``, so the correction is
    ``-sum(displacement * forces)``.

    Parameters
    ----------
    parent_energy
        Exact reference energy in eV.
    forces
        Reference forces with shape ``(N, 3)`` in eV/Angstrom.
    displacement
        Applied Cartesian displacement with shape ``(N, 3)`` in Angstrom.
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
class TaylorExpansionConfig:
    """Configuration for transformation-backed Taylor augmentation.

    The transformation is treated as an immutable prototype. Each parent gets
    a deterministic clone with an independent random stream derived from the
    prototype state and stable parent identity. The prototype's generator is
    therefore not advanced by augmentation.

    Parameters
    ----------
    transformation
        A random or D-optimal displacement transformation.
    include_reference
        Include an independent copy of each exact parent before its children.
    zero_force_tolerance
        Euclidean force-array norm at or below which the zero-force policy is
        applied.
    zero_force_policy
        ``"skip"`` omits derived children, ``"keep"`` generates them with a
        zero first-order correction, and ``"error"`` rejects the parent.
    duplicate_tolerance
        Absolute coordinate tolerance used to recognize duplicate child
        displacements within one parent.
    duplicate_policy
        ``"skip"``, ``"keep"``, or ``"error"`` for duplicate children.
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
        if self.zero_force_tolerance < 0.0:
            raise ValueError("zero_force_tolerance must be non-negative")
        if self.zero_force_policy not in {"skip", "keep", "error"}:
            raise ValueError(
                "zero_force_policy must be 'skip', 'keep', or 'error'"
            )
        if self.duplicate_tolerance < 0.0:
            raise ValueError("duplicate_tolerance must be non-negative")
        if self.duplicate_policy not in {"skip", "keep", "error"}:
            raise ValueError(
                "duplicate_policy must be 'skip', 'keep', or 'error'"
            )
        object.__setattr__(
            self,
            "transformation",
            copy.deepcopy(self.transformation),
        )

    @property
    def strategy(self) -> Literal["random", "d_optimal"]:
        """Return the stable strategy label for provenance."""
        if isinstance(
            self.transformation,
            DOptimalDisplacementTransformation,
        ):
            return "d_optimal"
        return "random"

    @property
    def requested_children(self) -> int | None:
        """Return the transformation's requested child count, if explicit."""
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
    """One exact parent or force-derived Taylor sample with provenance."""

    structure: Structure
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
    """Materialized Taylor samples and compact generation counts."""

    records: tuple[TaylorSampleRecord, ...]
    n_parents: int
    n_skipped: int = 0

    @property
    def structures(self) -> list[Structure]:
        """Return the generated structures in stable record order."""
        return [record.structure for record in self.records]

    @property
    def n_exact(self) -> int:
        """Return the number of retained exact-parent records."""
        return sum(record.label_origin == "exact" for record in self.records)

    @property
    def n_derived(self) -> int:
        """Return the number of Taylor-derived records."""
        return sum(record.label_origin == "taylor" for record in self.records)


def _copy_structure(structure: Structure, *, name: str) -> Structure:
    """Return an independent copy of a torch-training structure."""
    return Structure(
        positions=np.array(structure.positions, dtype=np.float64, copy=True),
        species=list(structure.species),
        energy=float(structure.energy),
        forces=(
            None
            if structure.forces is None
            else np.array(structure.forces, dtype=np.float64, copy=True)
        ),
        cell=(
            None
            if structure.cell is None
            else np.array(structure.cell, dtype=np.float64, copy=True)
        ),
        pbc=(
            None
            if structure.pbc is None
            else np.array(structure.pbc, dtype=bool, copy=True)
        ),
        name=name,
    )


def _validate_parent(structure: Structure, index: int) -> None:
    """Validate all labels required before transforming one parent."""
    if structure.energy is None:
        raise ValueError(f"parent {index} must have a finite energy")
    try:
        energy = float(structure.energy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"parent {index} must have a finite energy") from exc
    if not np.isfinite(energy):
        raise ValueError(f"parent {index} must have a finite energy")

    positions = np.asarray(structure.positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"parent {index} positions must have shape (N, 3)")
    if not np.isfinite(positions).all():
        raise ValueError(f"parent {index} positions must be finite")

    if structure.forces is None:
        raise ValueError(f"parent {index} must have a force array")
    forces = np.asarray(structure.forces)
    if forces.shape != positions.shape:
        raise ValueError(
            f"parent {index} force array must match positions shape"
        )
    if not np.isfinite(forces).all():
        raise ValueError(f"parent {index} must have finite forces")

    if structure.cell is not None and not np.isfinite(structure.cell).all():
        raise ValueError(f"parent {index} cell must be finite")


def _validate_transformation_for_parent(
    structure: Structure,
    config: TaylorExpansionConfig,
    index: int,
) -> None:
    """Reject transformation constraints with no internal displacement."""
    if (
        structure.n_atoms == 1
        and config.transformation.remove_translations
    ):
        raise ValueError(
            f"parent {index} has no internal displacement after translation "
            "removal; disable remove_translations or provide multiple atoms"
        )


def _requested_children_for_parent(
    structure: Structure,
    config: TaylorExpansionConfig,
) -> int:
    """Resolve the native transformation output request for one parent."""
    requested = config.requested_children
    if requested is not None:
        return requested
    dimensions = 3 * structure.n_atoms
    if config.transformation.remove_translations:
        return max(0, dimensions - 3)
    return dimensions


def _parent_ids(parents: Sequence[Structure]) -> list[str]:
    """Return stable, collision-free identifiers in parent order."""
    base_ids = [
        str(parent.name) if parent.name not in (None, "") else f"parent-{i:06d}"
        for i, parent in enumerate(parents)
    ]
    counts = Counter(base_ids)
    return [
        base_id if counts[base_id] == 1 else f"{base_id}::parent={index}"
        for index, base_id in enumerate(base_ids)
    ]


def _seeded_transformation(
    config: TaylorExpansionConfig,
    *,
    parent_id: str,
    parent_index: int,
    parent_namespace: str | None,
) -> _SupportedTransformation:
    """Clone and seed a transformation deterministically for one parent."""
    transformation = copy.deepcopy(config.transformation)
    rng = transformation.rng
    state = pickle.dumps(
        rng.bit_generator.state,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    identity = (
        f"{parent_namespace or ''}\0{parent_id}\0{parent_index}"
    ).encode()
    digest = hashlib.sha256(state + identity).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    bit_generator_type = type(rng.bit_generator)
    try:
        bit_generator = bit_generator_type(seed)
    except TypeError:
        bit_generator = np.random.PCG64(seed)
    transformation.rng = np.random.Generator(bit_generator)
    return transformation


def _child_structure(
    parent: Structure,
    positions: np.ndarray,
    *,
    energy: float,
    name: str,
) -> Structure:
    """Create one energy-only child while preserving structural metadata."""
    return Structure(
        positions=np.array(positions, dtype=np.float64, copy=True),
        species=list(parent.species),
        energy=float(energy),
        forces=None,
        cell=(
            None
            if parent.cell is None
            else np.array(parent.cell, dtype=np.float64, copy=True)
        ),
        pbc=(
            None
            if parent.pbc is None
            else np.array(parent.pbc, dtype=bool, copy=True)
        ),
        name=name,
    )


def generate_taylor_samples(
    parents: Sequence,
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
) -> TaylorSamplingResult:
    """Generate exact and Taylor-derived structures for reference parents.

    Parent labels are validated as a complete collection before any
    transformation runs. Coordinate generation is performed exclusively by
    the configured aenet transformation.
    """
    parent_structures = convert_to_structures(list(parents))
    for index, parent in enumerate(parent_structures):
        _validate_parent(parent, index)
        _validate_transformation_for_parent(parent, config, index)

    parent_ids = _parent_ids(parent_structures)
    records: list[TaylorSampleRecord] = []
    skipped = 0

    for parent_index, (parent, parent_id) in enumerate(
        zip(parent_structures, parent_ids)
    ):
        if config.include_reference:
            records.append(
                TaylorSampleRecord(
                    structure=_copy_structure(
                        parent,
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

        force_norm = float(np.linalg.norm(parent.forces))
        if force_norm <= config.zero_force_tolerance:
            if config.zero_force_policy == "error":
                raise ValueError(
                    f"parent {parent_id!r} is at or below the zero-force "
                    "tolerance"
                )
            if config.zero_force_policy == "skip":
                skipped += _requested_children_for_parent(parent, config)
                continue

        transformation = _seeded_transformation(
            config,
            parent_id=parent_id,
            parent_index=parent_index,
            parent_namespace=parent_namespace,
        )
        atomic_parent = AtomicStructure.from_TorchStructure(parent)
        child_atomics = transformation.apply_transformation(atomic_parent)
        accepted_displacements: list[np.ndarray] = []

        for child_index, child_atomic in enumerate(child_atomics):
            child_positions = np.asarray(
                child_atomic.coords[-1],
                dtype=np.float64,
            )
            displacement = child_positions - parent.positions
            if displacement.shape != parent.positions.shape:
                raise ValueError(
                    "transformation returned coordinates with a changed "
                    "atom count or shape"
                )
            if not np.isfinite(displacement).all():
                raise ValueError(
                    "transformation returned a non-finite displacement"
                )

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
                skipped += 1
                continue
            accepted_displacements.append(np.array(displacement, copy=True))

            child_energy = taylor_energy(
                parent.energy,
                parent.forces,
                displacement,
            )
            atom_norms = np.linalg.norm(displacement, axis=1)
            record = TaylorSampleRecord(
                structure=_child_structure(
                    parent,
                    child_positions,
                    energy=child_energy,
                    name=(
                        f"{parent_id}::taylor:{config.strategy}:"
                        f"{child_index:06d}"
                    ),
                ),
                parent_id=parent_id,
                parent_index=parent_index,
                child_index=child_index,
                strategy=config.strategy,
                label_origin="taylor",
                delta_energy=child_energy - float(parent.energy),
                displacement_rms=float(np.sqrt(np.mean(displacement**2))),
                maximum_displacement=float(np.max(atom_norms)),
            )
            records.append(record)

    return TaylorSamplingResult(
        records=tuple(records),
        n_parents=len(parent_structures),
        n_skipped=skipped,
    )


def iter_taylor_records(
    parents: Sequence,
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
) -> Iterator[TaylorSampleRecord]:
    """Yield materialized Taylor records in deterministic order."""
    yield from generate_taylor_samples(
        parents,
        config,
        parent_namespace=parent_namespace,
    ).records


def iter_taylor_structures(
    parents: Sequence,
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
) -> Iterator[Structure]:
    """Yield exact and derived structures in deterministic order."""
    for record in iter_taylor_records(
        parents,
        config,
        parent_namespace=parent_namespace,
    ):
        yield record.structure


def split_reference_structures(
    parents: Sequence,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int | None = None,
) -> tuple[list[Structure], list[Structure], list[Structure]]:
    """Split exact parents before augmentation while preserving source order."""
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

    structures = convert_to_structures(list(parents))
    n_parents = len(structures)
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

    def _select(indices: set[int]) -> list[Structure]:
        return [structures[index] for index in sorted(indices)]

    return (
        _select(train_indices),
        _select(validation_indices),
        _select(test_indices),
    )


class TaylorSourceCollection:
    """Wrap source records with parent-local Taylor augmentation.

    Each upstream source ID remains the HDF5 parent identity. The exact record
    is returned first when enabled, followed by transformation-derived
    children. Structure names retain exact-versus-Taylor provenance across an
    HDF5 close/reopen cycle.
    """

    def __init__(
        self,
        sources: SourceCollection | Sequence,
        config: TaylorExpansionConfig,
    ) -> None:
        self._sources = coerce_source_collection(sources)
        self.config = config

    @property
    def capabilities(self) -> SourceCapabilities:
        """Return the traversal capabilities of the wrapped collection."""
        return self._sources.capabilities

    def iter_records(self) -> Iterator[SourceRecord]:
        """Yield deterministic augmentation-aware source records."""
        for record in self._sources.iter_records():
            yield self._wrap_record(record)

    def iter_record_chunks(
        self,
        chunk_size: int,
    ) -> Iterator[list[SourceRecord]]:
        """Yield wrapped records while preserving upstream chunk streaming.

        Archive-backed sources use this path to read compressed members in a
        single sequential pass before Taylor augmentation. Sources without a
        native chunk iterator are grouped deterministically as a fallback.
        """
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be >= 1")

        upstream_chunks = getattr(self._sources, "iter_record_chunks", None)
        if callable(upstream_chunks):
            for chunk in upstream_chunks(chunk_size=chunk_size):
                yield [self._wrap_record(record) for record in chunk]
            return

        chunk: list[SourceRecord] = []
        for record in self._sources.iter_records():
            chunk.append(self._wrap_record(record))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def __len__(self) -> int:
        """Return the number of logical parent source records."""
        try:
            return len(self._sources)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("wrapped source collection has no length") from exc

    def _make_loader(self, record: SourceRecord):
        """Create an independent loader for one logical parent source."""
        config = self.config

        def _load() -> list[Structure]:
            parents = record.load_structures()
            return generate_taylor_samples(
                parents,
                config,
                parent_namespace=str(record.source_id),
            ).structures

        return _load

    def _wrap_record(self, record: SourceRecord) -> SourceRecord:
        """Return one augmentation-aware view of an upstream record."""
        return SourceRecord(
            source_id=record.source_id,
            loader=self._make_loader(record),
            source_kind=(
                f"{record.source_kind}:taylor_{self.config.strategy}"
                if record.source_kind
                else f"taylor_{self.config.strategy}"
            ),
            display_name=record.display_name,
        )
