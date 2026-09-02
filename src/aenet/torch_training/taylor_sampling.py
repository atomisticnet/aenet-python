"""PyTorch and source-collection adapters for neutral Taylor sampling."""

from __future__ import annotations

import copy
import hashlib
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from aenet.geometry import AtomicStructure
from aenet.geometry.sampling import (
    TaylorExpansionConfig,
    TaylorReference,
    taylor_energy,
)
from aenet.geometry.sampling import (
    generate_taylor_samples as generate_atomic_taylor_samples,
)
from aenet.geometry.sampling import (
    split_reference_structures as split_atomic_references,
)

from ._materialization import filter_structures
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


@dataclass(frozen=True)
class TaylorSampleRecord:
    """One torch-training structure and its Taylor provenance."""

    structure: Structure
    parent_id: str
    parent_index: int
    child_index: int | None
    strategy: str
    label_origin: str
    delta_energy: float
    displacement_rms: float
    maximum_displacement: float


@dataclass(frozen=True)
class TaylorSamplingResult:
    """Torch-adapted records and neutral generation counts."""

    records: tuple[TaylorSampleRecord, ...]
    config: TaylorExpansionConfig
    n_parents: int
    requested_children: int
    accepted_children: int
    duplicate_skipped: int = 0
    zero_force_skipped: int = 0
    unavailable_children: int = 0

    @property
    def structures(self) -> list[Structure]:
        """Return structures in stable record order."""
        return [record.structure for record in self.records]

    @property
    def n_exact(self) -> int:
        """Return the number of exact-parent records."""
        return sum(record.label_origin == "exact" for record in self.records)

    @property
    def n_derived(self) -> int:
        """Return the number of derived records."""
        return self.accepted_children

    @property
    def n_skipped(self) -> int:
        """Return the number of requested children not retained."""
        return (
            self.duplicate_skipped
            + self.zero_force_skipped
            + self.unavailable_children
        )


def _parent_ids(
    parents: Sequence[Structure], supplied: Sequence[str] | None
) -> list[str]:
    if supplied is None:
        resolved = [str(parent.name or "") for parent in parents]
        if any(not value.strip() for value in resolved):
            raise ValueError(
                "each parent needs a non-empty name or explicit parent_id"
            )
    else:
        resolved = [str(value) for value in supplied]
        if len(resolved) != len(parents):
            raise ValueError("parent_ids must have the same length as parents")
    if len(set(resolved)) != len(resolved):
        raise ValueError("parent IDs must be unique")
    return resolved


def _references(
    parents: Sequence[Structure], ids: Sequence[str]
) -> list[TaylorReference]:
    return [
        TaylorReference(
            parent_id=i, structure=AtomicStructure.from_TorchStructure(p)
        )
        for p, i in zip(parents, ids)
    ]


def _adapt(record) -> TaylorSampleRecord:
    return TaylorSampleRecord(
        structure=record.structure.to_TorchStructure(frame=0),
        parent_id=record.parent_id,
        parent_index=record.parent_index,
        child_index=record.child_index,
        strategy=record.strategy,
        label_origin=record.label_origin,
        delta_energy=record.delta_energy,
        displacement_rms=record.displacement_rms,
        maximum_displacement=record.maximum_displacement,
    )


def generate_taylor_samples(
    parents: Sequence,
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
    parent_ids: Sequence[str] | None = None,
) -> TaylorSamplingResult:
    """Generate torch structures through the backend-neutral Taylor core."""
    structures = convert_to_structures(list(parents))
    result = generate_atomic_taylor_samples(
        _references(structures, _parent_ids(structures, parent_ids)),
        config,
        parent_namespace=parent_namespace,
    )
    return TaylorSamplingResult(
        records=tuple(_adapt(record) for record in result.records),
        config=result.config,
        n_parents=result.n_parents,
        requested_children=result.requested_children,
        accepted_children=result.accepted_children,
        duplicate_skipped=result.duplicate_skipped,
        zero_force_skipped=result.zero_force_skipped,
        unavailable_children=result.unavailable_children,
    )


def iter_taylor_records(
    parents: Sequence,
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
    parent_ids: Sequence[str] | None = None,
) -> Iterator[TaylorSampleRecord]:
    """Yield adapted records in deterministic order."""
    yield from generate_taylor_samples(
        parents,
        config,
        parent_namespace=parent_namespace,
        parent_ids=parent_ids,
    ).records


def iter_taylor_structures(
    parents: Sequence,
    config: TaylorExpansionConfig,
    *,
    parent_namespace: str | None = None,
    parent_ids: Sequence[str] | None = None,
) -> Iterator[Structure]:
    """Yield structures in deterministic order."""
    for record in iter_taylor_records(
        parents,
        config,
        parent_namespace=parent_namespace,
        parent_ids=parent_ids,
    ):
        yield record.structure


def split_reference_structures(
    parents: Sequence,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int | None = None,
) -> tuple[list[Structure], list[Structure], list[Structure]]:
    """Split exact parents before augmentation."""
    structures = convert_to_structures(list(parents))
    splits = split_atomic_references(
        _references(structures, [str(i) for i in range(len(structures))]),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return tuple(
        [structures[int(ref.parent_id)] for ref in split] for split in splits
    )


class TaylorSourceCollection:
    """Augment source records after filtering exact parents."""

    def __init__(
        self,
        sources: SourceCollection | Sequence,
        config: TaylorExpansionConfig,
        *,
        max_energy: float | None = None,
        max_forces: float | None = None,
        atomic_energies: dict[str, float] | None = None,
    ) -> None:
        self._sources = coerce_source_collection(sources)
        self.config = config
        self.max_energy = max_energy
        self.max_forces = max_forces
        self.atomic_energies = copy.deepcopy(atomic_energies)

    @property
    def capabilities(self) -> SourceCapabilities:
        """Return upstream traversal capabilities."""
        return self._sources.capabilities

    def with_parent_filters(
        self,
        *,
        max_energy: float | None,
        max_forces: float | None,
        atomic_energies: dict[str, float] | None,
    ) -> TaylorSourceCollection:
        """Return a wrapper configured for pre-expansion filtering."""
        return type(self)(
            self._sources,
            self.config,
            max_energy=max_energy,
            max_forces=max_forces,
            atomic_energies=atomic_energies,
        )

    def iter_records(self) -> Iterator[SourceRecord]:
        """Yield wrapped source records."""
        for record in self._sources.iter_records():
            yield self._wrap_record(record)

    def iter_record_chunks(
        self, chunk_size: int
    ) -> Iterator[list[SourceRecord]]:
        """Yield wrapped records while preserving chunk streaming."""
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be >= 1")
        upstream = getattr(self._sources, "iter_record_chunks", None)
        if callable(upstream):
            for chunk in upstream(chunk_size=int(chunk_size)):
                yield [self._wrap_record(record) for record in chunk]
            return
        chunk = []
        for record in self._sources.iter_records():
            chunk.append(self._wrap_record(record))
            if len(chunk) == int(chunk_size):
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def __len__(self) -> int:
        """Return the number of upstream source records."""
        return len(self._sources)  # type: ignore[arg-type]

    def _generation_id(self, source_id: str) -> str:
        payload = pickle.dumps(
            (source_id, self.config),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return hashlib.sha256(payload).hexdigest()

    def _make_loader(self, record: SourceRecord):
        config, source_id = self.config, str(record.source_id)
        generation_id = self._generation_id(source_id)

        def _load() -> list[Structure]:
            output = []
            for source_frame_idx, parent in enumerate(
                record.load_structures()
            ):
                if not filter_structures(
                    [parent],
                    max_energy=self.max_energy,
                    max_forces=self.max_forces,
                    atomic_energies=self.atomic_energies,
                ):
                    continue
                parent_id = f"{source_id}#frame={source_frame_idx}"
                result = generate_taylor_samples(
                    [parent],
                    config,
                    parent_namespace=source_id,
                    parent_ids=[parent_id],
                )
                for sample in result.records:
                    base = str(parent.name or parent_id)
                    sample.structure.name = (
                        f"{base}::exact"
                        if sample.label_origin == "exact"
                        else f"{base}::taylor:{sample.strategy}:{sample.child_index:06d}"
                    )
                    sample.structure._aenet_taylor_provenance = {
                        "parent_id": sample.parent_id,
                        "child_index": sample.child_index,
                        "strategy": sample.strategy,
                        "label_origin": sample.label_origin,
                        "generation_id": generation_id,
                        "source_frame_idx": source_frame_idx,
                    }
                    output.append(sample.structure)
            return output

        return _load

    def _wrap_record(self, record: SourceRecord) -> SourceRecord:
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
