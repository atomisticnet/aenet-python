"""
Source abstractions for HDF5-backed torch-training dataset construction.

This module introduces a small source-oriented layer that decouples HDF5
dataset building from filesystem-path-specific assumptions.

The first implementation is intentionally narrow:

- ``SourceRecord`` represents one logical build input and owns source
  identity plus structure loading.
- ``SourceCapabilities`` describes source-collection traversal properties
  relevant to build-time execution strategy.
- ``FilePathSourceCollection`` adapts ordinary structure-file paths into the
  source model using ``AtomicStructure.from_file(...).to_TorchStructure()``.

Future HDF5 build refactors can consume these source objects directly instead
of taking ``file_paths`` plus a separate parser callback.
"""

from __future__ import annotations

import io
import os
import tarfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from aenet.formats.xsf import XSFParser
from aenet.geometry import AtomicStructure

from .config import Structure

__all__ = [
    "FilePathSourceCollection",
    "RecordSourceCollection",
    "SourceCapabilities",
    "SourceCollection",
    "SourceRecord",
    "TarArchiveXSFSourceCollection",
    "coerce_source_collection",
]


def _normalize_structures(
    loaded: Structure | Sequence[Structure],
) -> list[Structure]:
    """Normalize one or many loaded structures to a concrete list."""
    if isinstance(loaded, (list, tuple)):
        structs = list(loaded)
    else:
        structs = [loaded]

    for struct in structs:
        if not isinstance(struct, Structure):
            raise TypeError(
                "Source adapters must load torch-training Structure "
                f"objects, got {type(struct)!r}."
            )
    return structs


@dataclass(frozen=True)
class SourceCapabilities:
    """Traversal and execution capabilities of a source collection."""

    supports_multiple_passes: bool = True
    supports_random_access: bool = True
    supports_parallel_build: bool = True


@dataclass(frozen=True)
class SourceRecord:
    """
    One logical build input for HDF5 dataset construction.

    Parameters
    ----------
    source_id
        Stable source identifier used to describe the logical input. This is
        not required to be a filesystem path.
    loader
        Callable that loads one or more torch-training ``Structure`` objects
        from this record.
    source_kind
        Optional short source type label such as ``"file"`` or
        ``"tar_member"``.
    display_name
        Optional user-facing label. When omitted, callers can fall back to
        ``source_id``.
    """

    source_id: str
    loader: Callable[[], Structure | Sequence[Structure]]
    source_kind: str = ""
    display_name: str = ""

    def load_structures(self) -> list[Structure]:
        """Load and normalize this record's structure payload."""
        return _normalize_structures(self.loader())


@runtime_checkable
class SourceCollection(Protocol):
    """Protocol for collections of HDF5 build source records."""

    @property
    def capabilities(self) -> SourceCapabilities:
        """Return traversal/execution capabilities for the collection."""

    def iter_records(self) -> Iterator[SourceRecord]:
        """Yield source records in deterministic build order."""


@runtime_checkable
class _ChunkedSourceCollection(Protocol):
    """
    Optional internal protocol for streamed chunked HDF5 builds.

    Sources implementing this protocol can load the next chunk of logical
    records serially in the parent process and then hand those in-memory
    records to worker threads for parallel payload preparation.
    """

    def iter_record_chunks(
        self,
        chunk_size: int,
    ) -> Iterator[list[SourceRecord]]:
        """Yield deterministic chunks of ready-to-load source records."""


class FilePathSourceCollection:
    """
    Source collection backed by ordinary structure files on disk.

    Each input path becomes one ``SourceRecord`` whose ``source_id`` is the
    normalized path string and whose loader parses the file through
    ``AtomicStructure.from_file(...).to_TorchStructure()``.
    """

    def __init__(self, sources: Sequence[str | os.PathLike]):
        self._sources = [os.fspath(source) for source in sources]
        self._capabilities = SourceCapabilities(
            supports_multiple_passes=True,
            supports_random_access=True,
            supports_parallel_build=True,
        )

    @property
    def capabilities(self) -> SourceCapabilities:
        """Return capabilities for ordinary file-backed sources."""
        return self._capabilities

    def iter_records(self) -> Iterator[SourceRecord]:
        """Yield one source record per input file path."""
        for source_path in self._sources:
            yield SourceRecord(
                source_id=source_path,
                loader=self._make_loader(source_path),
                source_kind="file",
                display_name="",
            )

    @staticmethod
    def _make_loader(source_path: str) -> Callable[[], list[Structure]]:
        """Create a loader that parses one path-backed structure source."""

        def _load() -> list[Structure]:
            atomic_struct = AtomicStructure.from_file(source_path)
            return _normalize_structures(atomic_struct.to_TorchStructure())

        return _load


class RecordSourceCollection:
    """
    Simple in-memory source collection backed by explicit source records.

    This is primarily useful for tests and for programmatic callers that
    already own the record-level loading logic.
    """

    def __init__(
        self,
        records: Sequence[SourceRecord],
        *,
        capabilities: SourceCapabilities | None = None,
    ):
        self._records = list(records)
        self._capabilities = capabilities or SourceCapabilities()

    @property
    def capabilities(self) -> SourceCapabilities:
        """Return the declared capabilities for this record collection."""
        return self._capabilities

    def iter_records(self) -> Iterator[SourceRecord]:
        """Yield the stored records in deterministic order."""
        yield from self._records

    def __len__(self) -> int:
        """Return the number of explicit source records."""
        return len(self._records)


@dataclass(frozen=True)
class _TarArchiveMemberSpec:
    """Deterministic identity for one matching member in a tar archive."""

    archive_index: int
    member_name: str
    display_name: str


class TarArchiveXSFSourceCollection:
    """
    Source collection backed by XSF members inside a tar archive.

    Parameters
    ----------
    archive_path
        Path to a tar archive such as ``.tar``, ``.tar.gz``, or
        ``.tar.bz2``.
    member_suffixes
        Archive member suffixes to treat as XSF structure payloads.

    Notes
    -----
    This adapter supports deterministic repeated traversal by reopening the
    archive. It reports ``supports_random_access=False`` because individual
    member loads may require sequential scans through the archive stream.
    For HDF5 builds with ``build_workers > 1``, the adapter exposes an
    internal chunked-streaming path that reads archive members serially in
    the parent process and parallelizes only downstream parsing and cache
    preparation.
    """

    def __init__(
        self,
        archive_path: str | os.PathLike,
        *,
        member_suffixes: Sequence[str] = (".xsf",),
    ):
        self._archive_path = os.fspath(archive_path)
        self._member_suffixes = tuple(
            suffix.lower() for suffix in member_suffixes
        )
        self._member_specs = self._collect_member_specs()
        self._capabilities = SourceCapabilities(
            supports_multiple_passes=True,
            supports_random_access=False,
            supports_parallel_build=True,
        )

    @property
    def capabilities(self) -> SourceCapabilities:
        """Return capabilities for archive-backed XSF sources."""
        return self._capabilities

    def iter_records(self) -> Iterator[SourceRecord]:
        """Yield one source record per matching archive member."""
        for spec in self._member_specs:
            yield SourceRecord(
                source_id=(
                    f"{self._archive_path}::member={spec.archive_index}:"
                    f"{spec.member_name}"
                ),
                loader=self._make_loader(spec.archive_index),
                source_kind="tar_xsf_member",
                display_name=spec.display_name,
            )

    def __len__(self) -> int:
        """Return the number of matching archive members."""
        return len(self._member_specs)

    def iter_record_chunks(
        self,
        chunk_size: int,
    ) -> Iterator[list[SourceRecord]]:
        """Yield archive-backed records in deterministic streamed chunks."""
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be >= 1")

        if not self._member_specs:
            return

        spec_iter = iter(self._member_specs)
        next_spec = next(spec_iter, None)
        chunk: list[SourceRecord] = []
        with tarfile.open(self._archive_path, mode="r:*") as archive:
            for archive_index, member in enumerate(archive):
                if next_spec is None:
                    break
                if archive_index != next_spec.archive_index:
                    continue

                extracted = archive.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(
                        f"Archive member {member.name!r} was not found in "
                        f"{self._archive_path!r}."
                    )
                with extracted:
                    payload_bytes = extracted.read()

                chunk.append(
                    SourceRecord(
                        source_id=(
                            f"{self._archive_path}::member="
                            f"{next_spec.archive_index}:"
                            f"{next_spec.member_name}"
                        ),
                        loader=self._make_bytes_loader(payload_bytes),
                        source_kind="tar_xsf_member",
                        display_name=next_spec.display_name,
                    )
                )
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
                next_spec = next(spec_iter, None)

        if next_spec is not None:
            raise FileNotFoundError(
                f"Archive member #{next_spec.archive_index} was not found in "
                f"{self._archive_path!r}."
            )
        if chunk:
            yield chunk

    def _collect_member_specs(self) -> list[_TarArchiveMemberSpec]:
        """Return matching archive members with deterministic identities."""
        archive_label = Path(self._archive_path).name
        matching_members: list[tuple[int, str]] = []
        with tarfile.open(self._archive_path, mode="r:*") as archive:
            for archive_index, member in enumerate(archive):
                if (
                    member.isfile()
                    and member.name.lower().endswith(self._member_suffixes)
                ):
                    matching_members.append((archive_index, member.name))

        duplicate_counts: dict[str, int] = {}
        for _, member_name in matching_members:
            duplicate_counts[member_name] = (
                duplicate_counts.get(member_name, 0) + 1
            )

        member_specs: list[_TarArchiveMemberSpec] = []
        for archive_index, member_name in matching_members:
            display_name = f"{archive_label}:{member_name}"
            if duplicate_counts[member_name] > 1:
                display_name = (
                    f"{display_name}@member={archive_index}"
                )
            member_specs.append(
                _TarArchiveMemberSpec(
                    archive_index=archive_index,
                    member_name=member_name,
                    display_name=display_name,
                )
            )
        return member_specs

    def _make_loader(
        self,
        archive_index: int,
    ) -> Callable[[], list[Structure]]:
        """Create a loader that parses one XSF archive member."""

        def _load() -> list[Structure]:
            with tarfile.open(self._archive_path, mode="r:*") as archive:
                member = None
                for current_index, current_member in enumerate(archive):
                    if current_index == archive_index:
                        member = current_member
                        break
                if member is None:
                    raise FileNotFoundError(
                        f"Archive member #{archive_index} was not found in "
                        f"{self._archive_path!r}."
                    )

                extracted = archive.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(
                        f"Archive member {member.name!r} was not found in "
                        f"{self._archive_path!r}."
                    )
                with extracted, io.TextIOWrapper(
                    extracted,
                    encoding="utf-8",
                ) as text_stream:
                    atomic_struct = XSFParser().read(text_stream)
            return _normalize_structures(atomic_struct.to_TorchStructure())

        return _load

    @staticmethod
    def _make_bytes_loader(
        payload_bytes: bytes,
    ) -> Callable[[], list[Structure]]:
        """Create a loader that parses one in-memory XSF payload."""

        def _load() -> list[Structure]:
            with io.StringIO(payload_bytes.decode("utf-8")) as text_stream:
                atomic_struct = XSFParser().read(text_stream)
            return _normalize_structures(atomic_struct.to_TorchStructure())

        return _load


def coerce_source_collection(
    sources: Sequence[str | os.PathLike] | SourceCollection,
) -> SourceCollection:
    """
    Normalize supported source inputs to a source collection.

    Sequences of ordinary path-like inputs are wrapped in
    ``FilePathSourceCollection``. Existing source-collection objects are
    returned as-is.
    """
    if isinstance(sources, SourceCollection):
        return sources

    normalized = list(sources)
    if not normalized:
        return FilePathSourceCollection([])

    first = normalized[0]
    if isinstance(first, SourceRecord):
        if not all(isinstance(item, SourceRecord) for item in normalized):
            raise TypeError(
                "Mixed source lists are not supported. Provide either only "
                "source records, only path-like inputs, or a source "
                "collection object."
            )
        return RecordSourceCollection(normalized)
    if isinstance(first, (str, os.PathLike, Path)):
        if not all(isinstance(item, (str, os.PathLike, Path)) for item in normalized):
            raise TypeError(
                "Mixed source lists are not supported. Provide either only "
                "path-like inputs or a source collection object."
            )
        return FilePathSourceCollection(normalized)

    raise TypeError(
        "Unsupported sources input. Provide a sequence of path-like values "
        "or a source collection object."
    )
