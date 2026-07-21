#!/usr/bin/env python3
"""Fail-closed identity checks for the immutable M1 stock oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_PYTHON = "3.12.13"
EXPECTED_MLX_VERSION = "0.32.0"
EXPECTED_MLX_METAL_VERSION = "0.32.0"
EXPECTED_MLX_LM_VERSION = "0.31.3"
EXPECTED_MLX_LM_COMMIT = "8239c72de5a0e42c539e30489021db73c7fe258c"

# Derived from the reviewed oracle/uv.lock environment by the canonical algorithm below.
EXPECTED_MLX_TREE_SHA256 = "bacebd4f46680155a129301ffefc516402142183584f2b47673bc91b561f0cd9"
EXPECTED_MLX_LM_TREE_SHA256 = (
    "40dc49399a07cdf22e3516070cfe222e89ec2f0ff29cd6e257e1b069edc3472f"
)
EXPECTED_MLX_METAL_TREE_SHA256 = (
    "628a99548b65855148fb03f71cac83ce46eae42140f119fa8d1b51285c2abefd"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _included_distribution_file(distribution_name: str, relative: PurePosixPath) -> bool:
    if not relative.parts or "__pycache__" in relative.parts or relative.suffix == ".pyc":
        return False
    first = relative.parts[0]
    if distribution_name == "mlx":
        package = first == "mlx"
        metadata = first.startswith("mlx-") and first.endswith(".dist-info")
    elif distribution_name == "mlx-metal":
        package = first == "mlx"
        metadata = first.startswith("mlx_metal-") and first.endswith(".dist-info")
    elif distribution_name == "mlx-lm":
        package = first == "mlx_lm"
        metadata = first.startswith("mlx_lm-") and first.endswith(".dist-info")
    else:  # pragma: no cover - all callers are constants in this file
        raise RuntimeError(f"unsupported oracle distribution: {distribution_name}")
    return (package or metadata) and relative.name != "RECORD"


def distribution_tree(distribution_name: str) -> tuple[str, list[dict[str, Any]]]:
    """Hash every immutable package/metadata payload using installation-relative names."""

    distribution = importlib.metadata.distribution(distribution_name)
    files = distribution.files
    if files is None:
        raise RuntimeError(f"{distribution_name} has no installed-file inventory")
    entries: list[dict[str, Any]] = []
    for package_path in files:
        relative = PurePosixPath(str(package_path))
        if not _included_distribution_file(distribution_name, relative):
            continue
        installed = Path(distribution.locate_file(package_path)).resolve()
        if not installed.is_file():
            raise RuntimeError(
                f"{distribution_name} installed payload is missing: {relative.as_posix()}"
            )
        entries.append(
            {
                "path": relative.as_posix(),
                "size": installed.stat().st_size,
                "sha256": sha256_file(installed),
            }
        )
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise RuntimeError(f"{distribution_name} canonical tree is empty")
    canonical = b"".join(
        (
            f"{item['sha256']}  {item['size']}  {item['path']}\n".encode("utf-8")
            for item in entries
        )
    )
    return hashlib.sha256(canonical).hexdigest(), entries


def verify_identity(*, allow_unset_digests: bool = False) -> dict[str, Any]:
    python = platform.python_version()
    if python != EXPECTED_PYTHON:
        raise RuntimeError(f"expected Python {EXPECTED_PYTHON}, found {python}")

    mlx = importlib.metadata.distribution("mlx")
    mlx_metal = importlib.metadata.distribution("mlx-metal")
    mlx_lm = importlib.metadata.distribution("mlx-lm")
    if mlx.version != EXPECTED_MLX_VERSION:
        raise RuntimeError(f"expected mlx {EXPECTED_MLX_VERSION}, found {mlx.version}")
    if mlx_metal.version != EXPECTED_MLX_METAL_VERSION:
        raise RuntimeError(
            f"expected mlx-metal {EXPECTED_MLX_METAL_VERSION}, found {mlx_metal.version}"
        )
    if mlx_lm.version != EXPECTED_MLX_LM_VERSION:
        raise RuntimeError(
            f"expected mlx-lm {EXPECTED_MLX_LM_VERSION}, found {mlx_lm.version}"
        )
    direct_url_text = mlx_lm.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("mlx-lm direct_url.json is missing")
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != EXPECTED_MLX_LM_COMMIT:
        raise RuntimeError(
            f"expected mlx-lm commit {EXPECTED_MLX_LM_COMMIT}, found {commit}"
        )

    mlx_tree_sha256, mlx_files = distribution_tree("mlx")
    mlx_metal_tree_sha256, mlx_metal_files = distribution_tree("mlx-metal")
    mlx_lm_tree_sha256, mlx_lm_files = distribution_tree("mlx-lm")
    if not allow_unset_digests:
        if mlx_tree_sha256 != EXPECTED_MLX_TREE_SHA256:
            raise RuntimeError(
                "installed mlx tree differs from the reviewed oracle: "
                f"expected {EXPECTED_MLX_TREE_SHA256}, found {mlx_tree_sha256}"
            )
        if mlx_metal_tree_sha256 != EXPECTED_MLX_METAL_TREE_SHA256:
            raise RuntimeError(
                "installed mlx-metal tree differs from the reviewed oracle: "
                f"expected {EXPECTED_MLX_METAL_TREE_SHA256}, found {mlx_metal_tree_sha256}"
            )
        if mlx_lm_tree_sha256 != EXPECTED_MLX_LM_TREE_SHA256:
            raise RuntimeError(
                "installed mlx-lm tree differs from the reviewed oracle: "
                f"expected {EXPECTED_MLX_LM_TREE_SHA256}, found {mlx_lm_tree_sha256}"
            )
    return {
        "schema": "hyperion.m1-oracle-identity.v1",
        "python": python,
        "mlx_version": mlx.version,
        "mlx_metal_version": mlx_metal.version,
        "mlx_lm_version": mlx_lm.version,
        "mlx_lm_commit": commit,
        "mlx_tree_sha256": mlx_tree_sha256,
        "mlx_tree_file_count": len(mlx_files),
        "mlx_metal_tree_sha256": mlx_metal_tree_sha256,
        "mlx_metal_tree_file_count": len(mlx_metal_files),
        "mlx_lm_tree_sha256": mlx_lm_tree_sha256,
        "mlx_lm_tree_file_count": len(mlx_lm_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--derive",
        action="store_true",
        help="print canonical digests without comparing the three tree constants",
    )
    args = parser.parse_args()
    print(json.dumps(verify_identity(allow_unset_digests=args.derive), sort_keys=True))


if __name__ == "__main__":
    main()
