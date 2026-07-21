#!/usr/bin/env python3
"""Launch the M1 oracle without executing ambient Python startup hooks."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import types
from pathlib import Path
from typing import Any


EXPECTED_PYTHON = "3.12.13"
EXPECTED_PYTHON_EXECUTABLE_SHA256 = (
    "01564940172b2811e1f39a4dc90e84c7a26a19cf071bbc5de67e456d82627bec"
)
EXPECTED_PYTHON_RUNTIME_TREE_SHA256 = (
    "01a580d385a91f4b8bc195c8b2f56c4c2d156f6c1e1ad8768fc4501987c4e12f"
)
EXPECTED_PYTHON_RUNTIME_FILE_COUNT = 1897
EXPECTED_SITE_PACKAGES_TREE_SHA256 = (
    "db258e22404a3937d46d72ff44083400aafcf34636b8444a91a29c858b297006"
)
EXPECTED_SITE_PACKAGES_FILE_COUNT = 5470


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tree(
    root: Path, *, allow_symlinks: bool, ignored_names: frozenset[str] = frozenset()
) -> tuple[str, int]:
    """Hash a complete executable tree using location-independent relative names."""

    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"identity root is not a directory: {root}")
    entries: list[str] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in directories:
            candidate = directory_path / name
            if name == "__pycache__":
                continue
            if candidate.is_symlink():
                if not allow_symlinks:
                    raise RuntimeError(f"identity tree contains a directory symlink: {candidate}")
                target = os.readlink(candidate)
                resolved = (candidate.parent / target).resolve()
                if not resolved.is_relative_to(root):
                    raise RuntimeError(f"identity symlink escapes its tree: {candidate}")
                relative = candidate.relative_to(root).as_posix()
                entries.append(f"L {hashlib.sha256(target.encode()).hexdigest()} {len(target.encode())} {relative}\n")
                continue
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in files:
            candidate = directory_path / name
            if name in ignored_names or name.endswith(".pyc"):
                continue
            relative = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                if not allow_symlinks:
                    raise RuntimeError(f"identity tree contains a file symlink: {candidate}")
                target = os.readlink(candidate)
                resolved = (candidate.parent / target).resolve()
                if not resolved.is_relative_to(root):
                    raise RuntimeError(f"identity symlink escapes its tree: {candidate}")
                entries.append(f"L {hashlib.sha256(target.encode()).hexdigest()} {len(target.encode())} {relative}\n")
            elif candidate.is_file():
                entries.append(
                    f"F {sha256_file(candidate)} {candidate.stat().st_size} {relative}\n"
                )
            else:
                raise RuntimeError(f"identity tree contains a special file: {candidate}")
    entries.sort()
    if not entries:
        raise RuntimeError(f"identity tree is empty: {root}")
    return hashlib.sha256("".join(entries).encode()).hexdigest(), len(entries)


def startup_identity() -> dict[str, Any]:
    if sys.version.split()[0] != EXPECTED_PYTHON:
        raise RuntimeError(f"expected Python {EXPECTED_PYTHON}, found {sys.version.split()[0]}")
    flags = {
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "ignore_environment": sys.flags.ignore_environment,
        "safe_path": bool(sys.flags.safe_path),
        "no_user_site": sys.flags.no_user_site,
    }
    if flags != {
        "isolated": 1,
        "no_site": 1,
        "ignore_environment": 1,
        "safe_path": True,
        "no_user_site": 1,
    }:
        raise RuntimeError(f"oracle requires python -I -S, found flags {flags}")
    if "site" in sys.modules or "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise RuntimeError("ambient Python startup hooks executed before oracle isolation")

    oracle_dir = Path(__file__).resolve().parent
    expected_executable = oracle_dir / ".venv/bin/python"
    if Path(sys.executable) != expected_executable:
        raise RuntimeError(
            f"oracle executable must be {expected_executable}, found {sys.executable}"
        )
    site_packages = oracle_dir / ".venv/lib/python3.12/site-packages"
    runtime_root = Path(sys.executable).resolve().parent.parent
    executable_sha256 = sha256_file(Path(sys.executable))
    runtime_sha256, runtime_count = canonical_tree(
        runtime_root, allow_symlinks=True, ignored_names=frozenset({".DS_Store"})
    )
    site_sha256, site_count = canonical_tree(
        # Wheel RECORD files are non-executable installer receipts and uv rewrites
        # their entry-script rows with the environment path. Every referenced
        # payload byte, plus every extra file, is still covered by this full tree.
        site_packages,
        allow_symlinks=False,
        ignored_names=frozenset({".DS_Store", "RECORD"}),
    )
    return {
        "python_executable_sha256": executable_sha256,
        "python_runtime_tree_sha256": runtime_sha256,
        "python_runtime_file_count": runtime_count,
        "site_packages_tree_sha256": site_sha256,
        "site_packages_file_count": site_count,
        "isolated_flags": flags,
        "site_packages": site_packages,
    }


def validate_expected(identity: dict[str, Any]) -> None:
    expected = {
        "python_executable_sha256": EXPECTED_PYTHON_EXECUTABLE_SHA256,
        "python_runtime_tree_sha256": EXPECTED_PYTHON_RUNTIME_TREE_SHA256,
        "python_runtime_file_count": EXPECTED_PYTHON_RUNTIME_FILE_COUNT,
        "site_packages_tree_sha256": EXPECTED_SITE_PACKAGES_TREE_SHA256,
        "site_packages_file_count": EXPECTED_SITE_PACKAGES_FILE_COUNT,
    }
    actual = {key: identity[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"oracle startup substrate differs: expected {expected}, found {actual}")


def install_identity_module(identity: dict[str, Any]) -> None:
    module = types.ModuleType("_hyperion_isolated_identity")
    module.ORACLE_STARTUP_IDENTITY = {
        key: value for key, value in identity.items() if key != "site_packages"
    }
    sys.modules[module.__name__] = module


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: isolated_oracle.py identity [--derive] | script PATH [ARGS...] | module NAME [ARGS...]")
    action = sys.argv[1]
    identity = startup_identity()
    if action == "identity":
        derive = sys.argv[2:] == ["--derive"]
        if sys.argv[2:] not in ([], ["--derive"]):
            raise SystemExit("usage: isolated_oracle.py identity [--derive]")
        if not derive:
            validate_expected(identity)
        print(
            json.dumps(
                {key: value for key, value in identity.items() if key != "site_packages"},
                sort_keys=True,
            )
        )
        return

    validate_expected(identity)
    install_identity_module(identity)
    oracle_dir = Path(__file__).resolve().parent
    sys.path.extend([str(oracle_dir), str(identity["site_packages"])])
    if action == "script":
        if len(sys.argv) < 3:
            raise SystemExit("isolated_oracle.py script requires a target")
        target = Path(sys.argv[2]).resolve()
        if target.parent != oracle_dir or target.suffix != ".py":
            raise RuntimeError("isolated oracle script target must be a direct oracle/*.py file")
        sys.argv = [str(target), *sys.argv[3:]]
        runpy.run_path(str(target), run_name="__main__")
    elif action == "module":
        if len(sys.argv) < 3 or sys.argv[2] != "mlx_lm.server":
            raise RuntimeError("isolated oracle permits only the mlx_lm.server module")
        sys.argv = [sys.argv[2], *sys.argv[3:]]
        runpy.run_module("mlx_lm.server", run_name="__main__", alter_sys=True)
    else:
        raise SystemExit(f"unsupported isolated oracle action: {action}")


if __name__ == "__main__":
    main()
