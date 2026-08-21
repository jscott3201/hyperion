#!/usr/bin/env python3
"""Model-free evidence-tree inventory and atomic no-replace publication.

This module is the P2-NEXT-01 publication foundation. It implements the
content-addressed regular-file inventory and the durable same-filesystem
atomic no-replace transition frozen in the Qwen producer publication
requirements. It is not the raw-bundle semantic verifier, does not emit a
detached seal, and never authors pass, agreement, acceptance, or support
fields.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from qwen38_contract import canonical_json


HASH_CHUNK_BYTES = 1024 * 1024
INVENTORY_RECORD_FIELDS = frozenset({"byte_length", "path", "sha256"})
_SHA256_HEX = frozenset("0123456789abcdef")

_MACOS_RENAME_EXCL = 0x4
_LINUX_RENAME_NOREPLACE = 0x1


class EvidenceIOError(RuntimeError):
    """A fail-closed evidence I/O operation refused or lost a race."""

    kind = "evidence_io_error"


class UnsafePathError(EvidenceIOError):
    """A path is not a canonical relative POSIX path."""

    kind = "unsafe_path"


class UnsafeTreeError(EvidenceIOError):
    """A tree contains a symlink, special file, hard link, or unsafe name."""

    kind = "unsafe_tree"


class TreeMutationError(EvidenceIOError):
    """A tree entry changed identity or content during the operation."""

    kind = "tree_mutation"


class MarkerError(EvidenceIOError):
    """The final marker could not be created exactly as specified."""

    kind = "marker_error"


class CrossDeviceError(EvidenceIOError):
    """Staging and the target parent are on different filesystems."""

    kind = "cross_device"


class TargetExistsError(EvidenceIOError):
    """The publication target already existed at the instant of rename."""

    kind = "target_exists"


class DurablePublicationUncertainError(EvidenceIOError):
    """The publication outcome could not be verified as a durable success.

    The target may be visible at its final path. No rollback is attempted;
    an operator must inspect the target before any retry or reuse. The
    ``kind`` distinguishes a rename that took effect but could not be
    synced from an at_rename hook outcome that could not be verified.
    """

    def __init__(self, message: str, *, kind: str = "publication_outcome_uncertain"):
        super().__init__(message)
        self.kind = kind


class PublicationFailedError(EvidenceIOError):
    """The publication failed before the rename took effect."""

    kind = "publication_failed"


class UnsupportedPlatformError(EvidenceIOError):
    """No atomic no-replace rename primitive is available on this platform."""

    kind = "unsupported_platform"


@dataclass(frozen=True)
class PublicationReceipt:
    """Implementation-owned publication facts, not an evidence receipt."""

    target_path: str
    tree_inventory_sha256: str
    file_count: int
    tree_bytes: int
    marker_path: str
    marker_sha256: str
    platform_operation: str


Hooks = dict[str, Callable[..., object]]


def _run_hook(hooks: Hooks | None, name: str, *arguments: Any) -> None:
    hook = (hooks or {}).get(name)
    if hook is not None:
        hook(*arguments)


def _canonical_relative(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise UnsafePathError(f"{context} must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise UnsafePathError(f"{context} contains a backslash or NUL: {value!r}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise UnsafePathError(f"{context} is not a canonical relative path: {value!r}")
    for part in relative.parts:
        try:
            part.encode("utf-8")
        except UnicodeEncodeError as error:
            raise UnsafePathError(
                f"{context} is not representable as canonical UTF-8: {value!r}"
            ) from error
    return relative


def _safe_entry_name(name: str, context: str) -> None:
    if name in {"", ".", ".."} or "\\" in name or "\x00" in name:
        raise UnsafeTreeError(f"{context} is not a safe entry name: {name!r}")
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as error:
        raise UnsafeTreeError(
            f"{context} is not representable as canonical UTF-8: {name!r}"
        ) from error


def _open_root_directory(root: Path, context: str) -> int:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise UnsafeTreeError(f"{context} is unavailable: {root}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise UnsafeTreeError(f"{context} must not be a symlink: {root}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise UnsafeTreeError(f"{context} must be a real directory: {root}")
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise UnsafeTreeError(f"{context} could not be opened: {root}") from error
    try:
        try:
            opened = os.fstat(descriptor)
        except OSError as error:
            raise TreeMutationError(f"{context} could not be inspected") from error
        if not stat.S_ISDIR(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            raise TreeMutationError(f"{context} changed identity while being opened")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_resolved_directory(path: Path, context: str) -> int:
    """Open an absolute resolved directory chain with per-component identity checks."""

    resolved = Path(os.path.realpath(path))
    try:
        descriptor = os.open(resolved.anchor, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as error:
        raise UnsafePathError(f"{context} anchor could not be opened: {resolved}") from error
    metadata: os.stat_result | None = None
    try:
        for part in resolved.parts[1:]:
            try:
                metadata = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise UnsafePathError(
                    f"{context} is unavailable: {part!r} in {resolved}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise UnsafePathError(
                    f"{context} must be a chain of real directories: {resolved}"
                )
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise UnsafePathError(
                    f"{context} could not be opened: {part!r} in {resolved}"
                ) from error
            try:
                opened = os.fstat(child)
            except OSError as error:
                os.close(child)
                raise TreeMutationError(
                    f"{context} component could not be inspected: {part!r}"
                ) from error
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                os.close(child)
                raise TreeMutationError(
                    f"{context} changed identity while being opened: {part!r}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _assert_same_identity(
    before: os.stat_result,
    after: os.stat_result,
    relative: str,
) -> None:
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise TreeMutationError(f"entry changed identity while being read: {relative}")
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise TreeMutationError(f"entry changed content while being read: {relative}")


def _inventory_file(
    directory_fd: int,
    name: str,
    relative: str,
    metadata: os.stat_result,
    seen_inodes: set[tuple[int, int]],
    hooks: Hooks | None,
    identities: dict[str, os.stat_result] | None = None,
) -> dict[str, Any]:
    _run_hook(hooks, "before_entry_open", relative)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
    except FileNotFoundError as error:
        raise TreeMutationError(f"entry vanished while being inventoried: {relative}") from error
    except OSError as error:
        raise UnsafeTreeError(f"entry could not be opened: {relative}") from error
    try:
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            raise TreeMutationError(f"entry could not be inspected: {relative}") from error
        if not stat.S_ISREG(before.st_mode):
            raise UnsafeTreeError(f"entry is not a regular file: {relative}")
        if before.st_nlink != 1:
            raise UnsafeTreeError(f"entry has additional hard links: {relative}")
        if (before.st_dev, before.st_ino) in seen_inodes:
            raise UnsafeTreeError(f"entry repeats a device/inode identity: {relative}")
        _assert_same_identity(metadata, before, relative)
        digest = hashlib.sha256()
        first_chunk = True
        while True:
            try:
                chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            except OSError as error:
                raise TreeMutationError(f"entry could not be read: {relative}") from error
            if first_chunk:
                _run_hook(hooks, "during_file_read", relative)
                first_chunk = False
            if not chunk:
                break
            digest.update(chunk)
        try:
            after = os.fstat(descriptor)
        except OSError as error:
            raise TreeMutationError(f"entry could not be inspected: {relative}") from error
        _assert_same_identity(before, after, relative)
    finally:
        os.close(descriptor)
    seen_inodes.add((before.st_dev, before.st_ino))
    if identities is not None:
        identities[relative] = before
    return {
        "path": relative,
        "byte_length": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _inventory_directory(
    directory_fd: int,
    prefix: str,
    entries: list[dict[str, Any]],
    seen_inodes: set[tuple[int, int]],
    hooks: Hooks | None,
    identities: dict[str, os.stat_result] | None = None,
) -> None:
    try:
        scanner = os.scandir(directory_fd)
    except OSError as error:
        raise UnsafeTreeError(f"tree enumeration failed closed: {error}") from error
    with scanner:
        for entry in scanner:
            _safe_entry_name(entry.name, "tree entry")
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise TreeMutationError(
                    f"tree entry changed while being enumerated: {prefix}{entry.name}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise UnsafeTreeError(f"tree contains a symlink: {prefix}{entry.name}")
            relative = f"{prefix}{entry.name}"
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except FileNotFoundError as error:
                    raise TreeMutationError(
                        f"directory vanished while being inventoried: {relative}"
                    ) from error
                except OSError as error:
                    raise UnsafeTreeError(
                        f"directory could not be opened: {relative}"
                    ) from error
                try:
                    opened = os.fstat(child_fd)
                except OSError as error:
                    raise TreeMutationError(
                        f"directory could not be inspected: {relative}"
                    ) from error
                if not stat.S_ISDIR(opened.st_mode) or (
                    opened.st_dev,
                    opened.st_ino,
                ) != (metadata.st_dev, metadata.st_ino):
                    raise TreeMutationError(
                        f"directory changed identity while being read: {relative}"
                    )
                try:
                    _inventory_directory(
                        child_fd, f"{relative}/", entries, seen_inodes, hooks, identities
                    )
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                entries.append(
                    _inventory_file(
                        directory_fd,
                        entry.name,
                        relative,
                        metadata,
                        seen_inodes,
                        hooks,
                        identities,
                    )
                )
            else:
                raise UnsafeTreeError(
                    f"tree contains a special or non-regular file: {relative}"
                )


def inventory_tree(
    root: Path,
    *,
    hooks: Hooks | None = None,
    identities: dict[str, os.stat_result] | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic content-addressed inventory of every regular file."""

    entries: list[dict[str, Any]] = []
    seen_inodes: set[tuple[int, int]] = set()
    root_fd = _open_root_directory(root, "inventory root")
    try:
        _inventory_directory(root_fd, "", entries, seen_inodes, hooks, identities)
    finally:
        os.close(root_fd)
    entries.sort(key=lambda record: record["path"])
    return entries


def tree_inventory_sha256(entries: list[dict[str, Any]]) -> str:
    """Digest the canonical JSON of a sorted closed inventory collection."""

    for record in entries:
        if not isinstance(record, dict) or set(record) != INVENTORY_RECORD_FIELDS:
            raise EvidenceIOError(f"tree inventory record is not closed: {record!r}")
        if not isinstance(record["path"], str) or type(record["byte_length"]) is not int:
            raise EvidenceIOError(f"tree inventory record is malformed: {record!r}")
        digest = record["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or not set(digest) <= _SHA256_HEX:
            raise EvidenceIOError(f"tree inventory record digest is malformed: {record!r}")
    if entries != sorted(entries, key=lambda record: record["path"]):
        raise EvidenceIOError("tree inventory must be sorted by canonical path")
    return hashlib.sha256(canonical_json(entries)).hexdigest()


def create_final_marker(
    staging_root: Path,
    marker_relative_path: str,
    marker_bytes: bytes,
) -> dict[str, Any]:
    """Create the exact marker file inside staging with exclusive no-follow semantics."""

    relative = _canonical_relative(marker_relative_path, "final marker")
    root_fd = _open_root_directory(staging_root, "staging root")
    parent_fd = root_fd
    try:
        for part in relative.parts[:-1]:
            try:
                os.mkdir(part, dir_fd=parent_fd, mode=0o755)
            except FileExistsError:
                pass
            except OSError as error:
                raise MarkerError(
                    f"final marker parent could not be created: {relative}"
                ) from error
            try:
                child_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except OSError as error:
                raise UnsafeTreeError(
                    f"final marker parent is not a real directory: {relative}"
                ) from error
            if parent_fd != root_fd:
                os.close(parent_fd)
            parent_fd = child_fd
        try:
            marker_fd = os.open(
                relative.parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                dir_fd=parent_fd,
                mode=0o644,
            )
        except FileExistsError as error:
            raise MarkerError(
                f"final marker or an alias already exists: {relative}"
            ) from error
        except OSError as error:
            raise MarkerError(f"final marker creation failed: {relative}") from error
        try:
            try:
                written = 0
                while written < len(marker_bytes):
                    written += os.write(marker_fd, marker_bytes[written:])
                os.fsync(marker_fd)
            except OSError as error:
                raise MarkerError(f"final marker could not be written: {relative}") from error
            try:
                final = os.fstat(marker_fd)
            except OSError as error:
                raise MarkerError(f"final marker could not be inspected: {relative}") from error
            if final.st_size != len(marker_bytes):
                raise MarkerError(f"final marker byte length drifted: {relative}")
        finally:
            os.close(marker_fd)
    finally:
        if parent_fd != root_fd:
            os.close(parent_fd)
        os.close(root_fd)
    return {
        "path": relative.as_posix(),
        "byte_length": len(marker_bytes),
        "sha256": hashlib.sha256(marker_bytes).hexdigest(),
    }


def _sync_files_walk(
    directory_fd: int,
    prefix: str,
    hooks: Hooks | None,
    identities: dict[str, os.stat_result],
) -> list[tuple[str, int]]:
    observed: list[tuple[str, int]] = []
    try:
        scanner = os.scandir(directory_fd)
    except OSError as error:
        raise PublicationFailedError(f"staging enumeration failed closed: {error}") from error
    with scanner:
        for entry in scanner:
            _safe_entry_name(entry.name, "staging entry")
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise TreeMutationError(
                    f"staging entry changed while being enumerated: {prefix}{entry.name}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise UnsafeTreeError(f"staging contains a symlink: {prefix}{entry.name}")
            relative = f"{prefix}{entry.name}"
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    raise PublicationFailedError(
                        f"staging directory could not be opened: {relative}"
                    ) from error
                try:
                    observed.extend(
                        _sync_files_walk(child_fd, f"{relative}/", hooks, identities)
                    )
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                observed.append((relative, metadata.st_size))
                _run_hook(hooks, "during_file_sync", relative)
                try:
                    descriptor = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    raise PublicationFailedError(
                        f"staging file could not be opened for sync: {relative}"
                    ) from error
                try:
                    try:
                        before = os.fstat(descriptor)
                    except OSError as error:
                        raise PublicationFailedError(
                            f"staging file could not be inspected: {relative}"
                        ) from error
                    if not stat.S_ISREG(before.st_mode):
                        raise UnsafeTreeError(
                            f"staging entry is not a regular file: {relative}"
                        )
                    if before.st_nlink != 1:
                        raise UnsafeTreeError(
                            f"staging entry has additional hard links: {relative}"
                        )
                    _assert_same_identity(metadata, before, relative)
                    accepted = identities.get(relative)
                    if accepted is None or (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ) != (
                        accepted.st_dev,
                        accepted.st_ino,
                        accepted.st_size,
                        accepted.st_mtime_ns,
                        accepted.st_ctime_ns,
                    ):
                        raise TreeMutationError(
                            f"staging entry differs from the accepted inventory: {relative}"
                        )
                    try:
                        os.fsync(descriptor)
                    except OSError as error:
                        raise PublicationFailedError(
                            f"staging file could not be synced: {relative}"
                        ) from error
                    try:
                        after_sync = os.fstat(descriptor)
                    except OSError as error:
                        raise PublicationFailedError(
                            f"staging file could not be inspected: {relative}"
                        ) from error
                    _assert_same_identity(before, after_sync, relative)
                finally:
                    os.close(descriptor)
            else:
                raise UnsafeTreeError(
                    f"staging contains a special or non-regular file: {relative}"
                )
    return observed


def _fsync_files(
    directory_fd: int,
    prefix: str,
    hooks: Hooks | None,
    expected: dict[str, int],
    identities: dict[str, os.stat_result],
) -> list[tuple[str, int]]:
    """Sync every regular file and prove the observed set matches the accepted inventory."""

    observed = _sync_files_walk(directory_fd, prefix, hooks, identities)
    if sorted(observed) != sorted(expected.items()):
        raise TreeMutationError(
            "staging no longer matches the accepted inventory during the sync pass"
        )
    return observed


def _fsync_directories_bottom_up(
    directory_fd: int,
    prefix: str,
    hooks: Hooks | None,
) -> None:
    try:
        scanner = os.scandir(directory_fd)
        with scanner:
            for entry in scanner:
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise UnsafeTreeError(
                        f"staging contains a symlink: {prefix}{entry.name}"
                    )
                relative = f"{prefix}{entry.name}"
                if stat.S_ISDIR(metadata.st_mode):
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        _fsync_directories_bottom_up(child_fd, f"{relative}/", hooks)
                    finally:
                        os.close(child_fd)
                elif not stat.S_ISREG(metadata.st_mode):
                    raise UnsafeTreeError(
                        f"staging contains a special or non-regular file: {relative}"
                    )
    except OSError as error:
        raise PublicationFailedError(f"staging directory sync failed closed: {error}") from error
    _run_hook(hooks, "during_directory_sync", prefix)
    try:
        os.fsync(directory_fd)
    except OSError as error:
        raise PublicationFailedError(f"staging directory could not be synced: {error}") from error


def _stat_tree_walk(
    directory_fd: int,
    prefix: str,
    identities: dict[str, os.stat_result],
) -> list[tuple[str, int]]:
    """Identity-bound structural walk used as the last reconciliation before the rename."""

    observed: list[tuple[str, int]] = []
    try:
        scanner = os.scandir(directory_fd)
        with scanner:
            for entry in scanner:
                _safe_entry_name(entry.name, "staging entry")
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise TreeMutationError(
                        f"staging entry changed while being reconciled: {prefix}{entry.name}"
                    ) from error
                if stat.S_ISLNK(metadata.st_mode):
                    raise UnsafeTreeError(
                        f"staging contains a symlink: {prefix}{entry.name}"
                    )
                relative = f"{prefix}{entry.name}"
                if stat.S_ISDIR(metadata.st_mode):
                    try:
                        child_fd = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory_fd,
                        )
                    except OSError as error:
                        raise TreeMutationError(
                            f"staging directory could not be reopened: {relative}"
                        ) from error
                    try:
                        opened = os.fstat(child_fd)
                        if (opened.st_dev, opened.st_ino) != (
                            metadata.st_dev,
                            metadata.st_ino,
                        ):
                            raise TreeMutationError(
                                f"staging directory changed identity while being "
                                f"reconciled: {relative}"
                            )
                        observed.extend(
                            _stat_tree_walk(child_fd, f"{relative}/", identities)
                        )
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(metadata.st_mode):
                    if metadata.st_nlink != 1:
                        raise UnsafeTreeError(
                            f"staging entry has additional hard links: {relative}"
                        )
                    accepted = identities.get(relative)
                    if accepted is None or (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    ) != (
                        accepted.st_dev,
                        accepted.st_ino,
                        accepted.st_size,
                        accepted.st_mtime_ns,
                        accepted.st_ctime_ns,
                    ):
                        raise TreeMutationError(
                            f"staging entry differs from the accepted inventory: {relative}"
                        )
                    observed.append((relative, metadata.st_size))
                    try:
                        descriptor = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory_fd,
                        )
                    except OSError as error:
                        raise TreeMutationError(
                            f"staging file could not be reopened: {relative}"
                        ) from error
                    try:
                        try:
                            before = os.fstat(descriptor)
                        except OSError as error:
                            raise TreeMutationError(
                                f"staging file could not be inspected: {relative}"
                            ) from error
                        if before.st_nlink != 1:
                            raise UnsafeTreeError(
                                f"staging entry has additional hard links: {relative}"
                            )
                        _assert_same_identity(metadata, before, relative)
                        try:
                            os.fsync(descriptor)
                        except OSError as error:
                            raise PublicationFailedError(
                                f"staging file could not be synced: {relative}"
                            ) from error
                        try:
                            after = os.fstat(descriptor)
                        except OSError as error:
                            raise PublicationFailedError(
                                f"staging file could not be inspected: {relative}"
                            ) from error
                        _assert_same_identity(before, after, relative)
                        if (after.st_mtime_ns, after.st_ctime_ns) != (
                            accepted.st_mtime_ns,
                            accepted.st_ctime_ns,
                        ):
                            raise TreeMutationError(
                                f"staging entry differs from the accepted inventory: {relative}"
                            )
                    finally:
                        os.close(descriptor)
                else:
                    raise UnsafeTreeError(
                        f"staging contains a special or non-regular file: {relative}"
                    )
    except EvidenceIOError:
        raise
    except OSError as error:
        raise TreeMutationError(f"staging reconciliation walk failed closed: {error}") from error
    return observed


_LIBC: ctypes.CDLL | None = None


def _libc() -> ctypes.CDLL:
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(None, use_errno=True)
    return _LIBC


def _no_replace_primitive() -> tuple[Any, int, str]:
    if sys.platform == "darwin":
        try:
            primitive = _libc().renameatx_np
        except AttributeError as error:
            raise UnsupportedPlatformError(
                "renameatx_np is unavailable on this macOS libc"
            ) from error
        return primitive, _MACOS_RENAME_EXCL, "renameatx_np(RENAME_EXCL)"
    if sys.platform == "linux":
        try:
            primitive = _libc().renameat2
        except AttributeError as error:
            raise UnsupportedPlatformError(
                "renameat2 is unavailable on this Linux libc"
            ) from error
        return primitive, _LINUX_RENAME_NOREPLACE, "renameat2(RENAME_NOREPLACE)"
    raise UnsupportedPlatformError(
        f"no atomic no-replace rename primitive on {sys.platform}"
    )


def _rename_no_replace(
    source_dir_fd: int,
    source_name: str,
    target_parent_fd: int,
    target_name: str,
) -> str:
    primitive, flags, operation = _no_replace_primitive()
    primitive.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    primitive.restype = ctypes.c_int
    ctypes.set_errno(0)
    status = primitive(
        source_dir_fd,
        os.fsencode(source_name),
        target_parent_fd,
        os.fsencode(target_name),
        flags,
    )
    if status == 0:
        return operation
    failure = ctypes.get_errno()
    if failure in (errno.EEXIST, errno.ENOTEMPTY):
        raise TargetExistsError(
            f"publication target already exists at the instant of rename: {target_name!r}"
        )
    if failure in (errno.ENOSYS, errno.EINVAL):
        raise UnsupportedPlatformError(
            f"{operation} is not supported by this kernel or filesystem: "
            f"{os.strerror(failure)}"
        )
    raise PublicationFailedError(
        f"{operation} failed: {os.strerror(failure)}"
    )


def _descriptor_device(descriptor: int) -> int:
    try:
        return os.fstat(descriptor).st_dev
    except OSError as error:
        raise PublicationFailedError("target parent could not be inspected") from error


def _open_target_parent(target: Path, staging: Path) -> int:
    if target.name in {"", ".", ".."} or "\\" in target.name or "\x00" in target.name:
        raise UnsafePathError(f"publication target has an unsafe final component: {target!r}")
    try:
        target.name.encode("utf-8")
    except UnicodeEncodeError as error:
        raise UnsafePathError(
            f"publication target is not representable as canonical UTF-8: {target!r}"
        ) from error
    if ".." in target.parts:
        raise UnsafePathError(f"publication target escapes upward: {target!r}")
    if PurePosixPath(os.path.realpath(target)).parts == PurePosixPath(
        os.path.realpath(staging)
    ).parts:
        raise UnsafePathError(
            "the publication target must not be the staging root itself"
        )
    original_parent = Path(os.path.abspath(target.parent))
    try:
        final_parent_metadata = os.lstat(original_parent)
    except OSError as error:
        raise UnsafePathError(
            f"target parent is unavailable: {original_parent}"
        ) from error
    if stat.S_ISLNK(final_parent_metadata.st_mode):
        raise UnsafePathError(
            f"target parent must not be a symlink alias: {original_parent}"
        )
    staging_components = PurePosixPath(os.path.realpath(staging)).parts
    parent = Path(os.path.realpath(target.parent))
    parent_components = parent.parts
    if parent_components[: len(staging_components)] == staging_components:
        raise UnsafePathError(
            "the target parent is staging or lives inside it; the target must "
            "not be published below the staging root"
        )
    return _open_resolved_directory(parent, "target parent")


def publish_tree(
    staging_root: Path,
    target: Path,
    marker_relative_path: str,
    marker_bytes: bytes,
    *,
    hooks: Hooks | None = None,
    remove_staging_on_failure: bool = False,
) -> PublicationReceipt:
    """Publish a prepared staging tree through one atomic no-replace transition.

    The staging tree is caller-owned. Nothing is deleted on failure unless
    ``remove_staging_on_failure`` is set, and even then only after the
    staging root is proven to still hold the identity observed at the start.
    The rename is sourced from an identity-checked parent descriptor, the
    sync pass compares every file's full identity (device, inode, size,
    mtime, ctime) against snapshots from the accepted inventory, and the
    final pre-rename reconciliation re-opens, flushes, and re-verifies that
    same identity for every file after the ``before_rename`` hook, so
    deferred-writeback mutations are exposed before the transition, and the
    staging root is re-synced so directory entries created during the hook
    are persisted. A staging-root swap or content mutation landing in the
    irreducible window between that final reconciliation and the rename
    syscall itself cannot be detected in userspace and is a documented
    residual, as is a mutation that fits inside one timestamp-granularity
    tick and a target-parent swap after the parent descriptor is opened.
    If anything fails after the rename took effect, the target may be
    visible but durability is not claimed; the error is surfaced and no
    rollback is attempted.

    Deterministic fault-injection hooks, invoked when present in ``hooks``:
    ``after_initial_inventory``, ``after_final_marker``,
    ``after_final_inventory``, ``during_file_sync(relative)``,
    ``during_directory_sync(prefix)``, ``before_rename``,
    ``at_rename(rename)``, ``after_rename``, and ``during_parent_sync``.
    ``before_rename`` runs before the final identity-bound reconciliation,
    so hook-driven regular-file changes are still refused; empty-directory
    structure is not part of the regular-file inventory. An ``at_rename``
    hook must actually perform the platform rename and return its operation
    name: hook failures, non-rename returns, and arrivals that do not hold
    the staged tree are conservatively classified as
    durable-publication-uncertain outcomes and never yield a receipt.
    """

    staging_root = Path(staging_root)
    target = Path(target)
    try:
        root_metadata = staging_root.lstat()
    except OSError as error:
        raise UnsafeTreeError(f"staging root is unavailable: {staging_root}") from error
    staging_identity = (root_metadata.st_dev, root_metadata.st_ino)
    try:
        entries_before = inventory_tree(staging_root)
        _run_hook(hooks, "after_initial_inventory")
        marker = create_final_marker(staging_root, marker_relative_path, marker_bytes)
        _run_hook(hooks, "after_final_marker")
        accepted_identities: dict[str, os.stat_result] = {}
        entries_after = inventory_tree(staging_root, identities=accepted_identities)
        expected = sorted(entries_before + [marker], key=lambda record: record["path"])
        if entries_after != expected:
            raise TreeMutationError(
                "staging changed between inventories by more than the final marker"
            )
        _run_hook(hooks, "after_final_inventory")
        expected_items = [(record["path"], record["byte_length"]) for record in entries_after]
        staging_root_fd = _open_root_directory(staging_root, "staging root")
        try:
            _fsync_files(
                staging_root_fd,
                "",
                hooks,
                {record["path"]: record["byte_length"] for record in entries_after},
                accepted_identities,
            )
            _fsync_directories_bottom_up(staging_root_fd, "", hooks)
        finally:
            os.close(staging_root_fd)
        resolved_staging = Path(os.path.realpath(staging_root))
        staging_parent_fd = _open_resolved_directory(
            resolved_staging.parent, "staging parent"
        )
        try:
            target_parent_fd = _open_target_parent(target, staging_root)
        except Exception:
            os.close(staging_parent_fd)
            raise
        try:
            if _descriptor_device(target_parent_fd) != root_metadata.st_dev:
                raise CrossDeviceError(
                    "staging and the target parent are on different filesystems"
                )
            _run_hook(hooks, "before_rename")
            staging_root_fd = _open_root_directory(staging_root, "staging root")
            try:
                reconciled = _stat_tree_walk(staging_root_fd, "", accepted_identities)
                try:
                    os.fsync(staging_root_fd)
                except OSError as error:
                    raise PublicationFailedError(
                        "staging root could not be synced after reconciliation"
                    ) from error
            finally:
                os.close(staging_root_fd)
            if sorted(reconciled) != sorted(expected_items):
                raise TreeMutationError(
                    "staging no longer matches the accepted inventory before the transition"
                )
            try:
                source_metadata = os.stat(
                    resolved_staging.name,
                    dir_fd=staging_parent_fd,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise TreeMutationError(
                    "staging root vanished before the atomic transition"
                ) from error
            if (source_metadata.st_dev, source_metadata.st_ino) != staging_identity:
                raise TreeMutationError(
                    "staging root was replaced before the atomic transition"
                )

            def rename_callable() -> str:
                return _rename_no_replace(
                    staging_parent_fd,
                    resolved_staging.name,
                    target_parent_fd,
                    target.name,
                )

            rename_hook = (hooks or {}).get("at_rename")
            if rename_hook is not None:
                _, _, expected_operation = _no_replace_primitive()
                try:
                    operation = rename_hook(rename_callable)
                except BaseException as error:
                    raise DurablePublicationUncertainError(
                        "the at_rename hook failed after possibly renaming; the "
                        "target state requires operator inspection",
                        kind="at_rename_outcome_unverified",
                    ) from error
                if not isinstance(operation, str) or operation != expected_operation:
                    raise DurablePublicationUncertainError(
                        "the at_rename hook did not return the platform rename "
                        "operation; no verified publication occurred",
                        kind="at_rename_outcome_unverified",
                    )
                try:
                    arrival = os.stat(
                        target.name,
                        dir_fd=target_parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError as error:
                    raise DurablePublicationUncertainError(
                        "the target does not currently hold the staged tree "
                        "after the at_rename hook; the target state requires "
                        "operator inspection",
                        kind="at_rename_outcome_unverified",
                    ) from error
                except OSError as error:
                    raise DurablePublicationUncertainError(
                        "the at_rename outcome could not be verified; the target "
                        "state requires operator inspection",
                        kind="at_rename_outcome_unverified",
                    ) from error
                if (arrival.st_dev, arrival.st_ino) != staging_identity:
                    raise DurablePublicationUncertainError(
                        "the target does not hold the staged tree after the "
                        "at_rename hook; the target state requires operator "
                        "inspection",
                        kind="at_rename_outcome_unverified",
                    )
            else:
                operation = rename_callable()
            try:
                _run_hook(hooks, "after_rename")
                _run_hook(hooks, "during_parent_sync")
                os.fsync(target_parent_fd)
                arrival = os.stat(
                    target.name,
                    dir_fd=target_parent_fd,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise DurablePublicationUncertainError(
                    "rename completed but the post-rename verification failed; "
                    "the target may be visible and requires operator inspection",
                    kind="post_rename_parent_sync_failed",
                ) from error
            except BaseException as error:
                raise DurablePublicationUncertainError(
                    "rename completed but the post-rename sync failed; the target "
                    "may be visible and requires operator inspection",
                    kind="post_rename_parent_sync_failed",
                ) from error
            if (arrival.st_dev, arrival.st_ino) != staging_identity:
                raise DurablePublicationUncertainError(
                    "the target changed after the rename; the target state "
                    "requires operator inspection",
                    kind="post_rename_parent_sync_failed",
                )
        finally:
            os.close(staging_parent_fd)
            os.close(target_parent_fd)
    except Exception as failure:
        if remove_staging_on_failure:
            try:
                _remove_owned_staging(staging_root, staging_identity)
            except (EvidenceIOError, OSError) as cleanup_error:
                raise failure from cleanup_error
        raise
    return PublicationReceipt(
        target_path=str(target.absolute()),
        tree_inventory_sha256=tree_inventory_sha256(entries_after),
        file_count=len(entries_after),
        tree_bytes=sum(record["byte_length"] for record in entries_after),
        marker_path=marker["path"],
        marker_sha256=marker["sha256"],
        platform_operation=operation,
    )


def _remove_tree_at(directory_fd: int) -> None:
    scanner = os.scandir(directory_fd)
    with scanner:
        for entry in scanner:
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise UnsafeTreeError(
                    f"staging contains a symlink or special file: {entry.name!r}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    _remove_tree_at(child_fd)
                finally:
                    os.close(child_fd)
                os.rmdir(entry.name, dir_fd=directory_fd)
            else:
                os.unlink(entry.name, dir_fd=directory_fd)


def _remove_owned_staging(staging_root: Path, staging_identity: tuple[int, int]) -> None:
    try:
        descriptor = os.open(
            staging_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError:
        return
    try:
        try:
            opened = os.fstat(descriptor)
        except OSError as error:
            raise PublicationFailedError(
                "staging root could not be inspected during cleanup"
            ) from error
        if (opened.st_dev, opened.st_ino) != staging_identity:
            raise UnsafeTreeError(
                "staging root changed identity; refusing to remove a caller path"
            )
        if not stat.S_ISDIR(opened.st_mode):
            raise UnsafeTreeError("staging root is no longer a real directory")
        _remove_tree_at(descriptor)
    finally:
        os.close(descriptor)
    resolved_parent = Path(os.path.realpath(staging_root)).parent
    parent_fd = _open_resolved_directory(resolved_parent, "staging parent")
    try:
        name = PurePosixPath(os.path.realpath(staging_root)).name
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != staging_identity:
            raise UnsafeTreeError(
                "staging root changed identity; refusing to remove a caller path"
            )
        os.rmdir(name, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
