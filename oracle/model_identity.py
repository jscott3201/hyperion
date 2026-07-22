#!/usr/bin/env python3
"""Verify that a model payload exactly matches its immutable SHA256SUMS inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Callable, TypeVar


MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  \./([^\r\n]+)$")
SUSPICIOUS_CACHE_PAYLOAD_SUFFIXES = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
Loaded = TypeVar("Loaded")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def transport_cache_identity(root: Path) -> tuple[str, int]:
    """Bind directory topology and regular-file bytes; return digest and file count."""

    cache_root = root / ".cache"
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise RuntimeError("source model .cache must be a real directory")
    cache_children = list(cache_root.iterdir())
    if len(cache_children) != 1 or cache_children[0].name != "huggingface":
        raise RuntimeError("source model .cache must contain only huggingface")
    huggingface = cache_children[0]
    if huggingface.is_symlink() or not huggingface.is_dir():
        raise RuntimeError("source model .cache/huggingface must be a real directory")

    entries: list[str] = []
    file_count = 0

    def traversal_failed(error: OSError) -> None:
        raise RuntimeError(f"transport cache traversal failed closed: {error}") from error

    try:
        for directory, directories, files in os.walk(
            huggingface,
            followlinks=False,
            onerror=traversal_failed,
        ):
            directories.sort()
            files.sort()
            directory_path = Path(directory)
            for name in directories:
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix()
                mode = candidate.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise RuntimeError(
                        f"transport cache contains a directory symlink: {relative}"
                    )
                if not stat.S_ISDIR(mode):
                    raise RuntimeError(
                        f"transport cache contains a special directory: {relative}"
                    )
                entries.append(f"D {relative}\n")
            for name in files:
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix()
                metadata = candidate.lstat()
                mode = metadata.st_mode
                if stat.S_ISLNK(mode):
                    raise RuntimeError(f"transport cache contains a file symlink: {relative}")
                if not stat.S_ISREG(mode):
                    raise RuntimeError(f"transport cache contains a special file: {relative}")
                if candidate.suffix.lower() in SUSPICIOUS_CACHE_PAYLOAD_SUFFIXES:
                    raise RuntimeError(
                        f"transport cache contains an unexpected model payload: {relative}"
                    )
                entries.append(
                    f"F {sha256_file(candidate)} {metadata.st_size} {relative}\n"
                )
                file_count += 1
    except OSError as error:
        raise RuntimeError(f"transport cache inspection failed closed: {error}") from error
    entries.sort()
    if file_count == 0:
        raise RuntimeError("source model transport cache is empty")
    return hashlib.sha256("".join(entries).encode()).hexdigest(), file_count


def verify_model_tree(
    root: Path,
    expected_manifest_sha256: str,
    *,
    allow_huggingface_cache: bool = False,
    expected_transport_cache_sha256: str | None = None,
) -> dict[str, Any]:
    if allow_huggingface_cache:
        if expected_transport_cache_sha256 is None or not re.fullmatch(
            r"[0-9a-f]{64}", expected_transport_cache_sha256
        ):
            raise RuntimeError("source transport cache requires a canonical SHA-256")
    elif expected_transport_cache_sha256 is not None:
        raise RuntimeError("converted model verification cannot accept a transport-cache digest")
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"model root must be a real directory, not a symlink: {root}")
    manifest = root / "SHA256SUMS"
    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(f"model checksum manifest is missing or a symlink: {manifest}")
    manifest_sha256 = sha256_file(manifest)
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            f"model manifest mismatch: expected {expected_manifest_sha256}, found {manifest_sha256}"
        )

    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise RuntimeError(f"model manifest contains a malformed line: {line!r}")
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in expected:
            raise RuntimeError(f"model manifest contains an unsafe or duplicate path: {name!r}")
        expected[name] = digest
    if not expected:
        raise RuntimeError("model manifest has no payload files")

    transport_cache_sha256: str | None = None
    transport_cache_file_count = 0
    actual: set[str] = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        kept: list[str] = []
        for name in directories:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                raise RuntimeError(f"model tree contains a directory symlink: {relative}")
            if relative == ".cache" and allow_huggingface_cache:
                if expected_transport_cache_sha256 is None:
                    raise RuntimeError("source transport cache requires an expected tree digest")
                transport_cache_sha256, transport_cache_file_count = transport_cache_identity(
                    root
                )
                if transport_cache_sha256 != expected_transport_cache_sha256:
                    raise RuntimeError(
                        "source transport cache differs: expected "
                        f"{expected_transport_cache_sha256}, found {transport_cache_sha256}"
                    )
                continue
            kept.append(name)
        directories[:] = kept
        for name in files:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink() or not candidate.is_file():
                raise RuntimeError(f"model tree contains a symlink or special file: {relative}")
            if relative == "SHA256SUMS":
                continue
            actual.add(relative)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise RuntimeError(f"model tree inventory differs: missing={missing}, extra={extra}")
    if allow_huggingface_cache and transport_cache_sha256 is None:
        raise RuntimeError("source model is missing its bound .cache/huggingface tree")

    entries: list[str] = []
    for name, expected_sha256 in sorted(expected.items()):
        path = root / name
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"model payload hash differs for {name}: expected {expected_sha256}, found {actual_sha256}"
            )
        entries.append(f"{actual_sha256}  {path.stat().st_size}  {name}\n")
    return {
        "schema": "hyperion.model-tree-identity.v1",
        "manifest_sha256": manifest_sha256,
        "payload_tree_sha256": hashlib.sha256("".join(entries).encode()).hexdigest(),
        "payload_file_count": len(entries),
        "exact_inventory": True,
        "symlinks_rejected": True,
        "transport_cache_excluded": allow_huggingface_cache,
        "transport_cache_separately_bound": allow_huggingface_cache,
        "transport_cache_tree_sha256": transport_cache_sha256,
        "transport_cache_file_count": transport_cache_file_count,
    }


def verified_model_load(
    root: Path,
    expected_manifest_sha256: str,
    expected_identity: dict[str, Any],
    loader: Callable[..., Loaded],
    /,
    *args: Any,
    **kwargs: Any,
) -> Loaded:
    """Recheck a converted model and cross the load boundary without intervening work."""

    boundary_identity = verify_model_tree(root, expected_manifest_sha256)
    if boundary_identity != expected_identity:
        raise RuntimeError("model identity changed before the actual load boundary")
    try:
        loaded = loader(str(root), *args, **kwargs)
    finally:
        try:
            loaded_identity = verify_model_tree(root, expected_manifest_sha256)
        except Exception as error:
            raise RuntimeError("model identity changed while the actual loader ran") from error
        if loaded_identity != boundary_identity:
            raise RuntimeError("model identity changed while the actual loader ran")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--allow-huggingface-cache", action="store_true")
    parser.add_argument("--expected-transport-cache-sha256")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_model_tree(
                args.model,
                args.manifest_sha256,
                allow_huggingface_cache=args.allow_huggingface_cache,
                expected_transport_cache_sha256=args.expected_transport_cache_sha256,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
