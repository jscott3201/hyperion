#!/usr/bin/env python3
"""Verify that a model payload exactly matches its immutable SHA256SUMS inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  \./([^\r\n]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_tree(
    root: Path, expected_manifest_sha256: str, *, allow_huggingface_cache: bool = False
) -> dict[str, Any]:
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
                cache_children = list(candidate.iterdir())
                if any(child.name != "huggingface" for child in cache_children):
                    raise RuntimeError("source model .cache contains non-Hugging-Face entries")
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--allow-huggingface-cache", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_model_tree(
                args.model,
                args.manifest_sha256,
                allow_huggingface_cache=args.allow_huggingface_cache,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
