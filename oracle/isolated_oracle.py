#!/usr/bin/env python3
"""Launch the M1 oracle without executing ambient Python startup hooks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import struct
import sys
import types
from pathlib import Path
from typing import Any


EXPECTED_PYTHON = "3.12.13"
EXPECTED_PYTHON_EXECUTABLE_SHA256 = (
    "01564940172b2811e1f39a4dc90e84c7a26a19cf071bbc5de67e456d82627bec"
)
EXPECTED_PYTHON_RUNTIME_TREE_SHA256 = (
    "84fdd9dcc811d7dab39be0d36dcb375526287b8b033b663864d3fd896a67efcb"
)
EXPECTED_PYTHON_RUNTIME_FILE_COUNT = 1897
EXPECTED_SITE_PACKAGES_TREE_SHA256 = (
    "db258e22404a3937d46d72ff44083400aafcf34636b8444a91a29c858b297006"
)
EXPECTED_SITE_PACKAGES_FILE_COUNT = 5470
EXPECTED_HASH_SEED_PROBE = 1244036990071903237
EXPECTED_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TZ": "UTC",
}

# Fixed replacement for the resolved interpreter install prefix when hashing the
# runtime tree. The uv base prefix is relocated per machine (HOME-embedded in
# libpython3.12.dylib and _sysconfigdata), so the prefix bytes are stripped from
# each file's content before hashing; code tampering is still detected because
# every non-prefix byte is bound.
PATH_PREFIX_TOKEN = b"<<HYPERION_PREFIX>>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# Mach-O magic numbers (the first four bytes on disk). The runtime tree excludes
# embedded code signatures because uv re-signs every Mach-O it relocates adhoc,
# and the adhoc CodeDirectory signs the install-name page, so signature bytes
# differ per machine even for the same release.
_MACHO_MAGICS = frozenset(
    {
        b"\xcf\xfa\xed\xfe",  # MH_MAGIC_64 (arm64)
        b"\xce\xfa\xed\xfe",  # MH_MAGIC
        b"\xca\xfe\xba\xbe",  # FAT_MAGIC
        b"\xbe\xba\xfe\xca",  # FAT_MAGIC_64
    }
)

# LC_CODE_SIGNATURE load command id; its payload is (cmd, cmdsize, dataoff, datasize).
_LC_CODE_SIGNATURE = 0x1D


def _macho_codesignature_range(path: Path) -> tuple[int, int] | None:
    """Return ``(offset, size)`` of ``path``'s embedded code signature, or None.

    Mach-O thin binaries only (the arm64 base-prefix binaries); fat archives and
    unrecognized magics return None so the caller hashes the raw bytes. ``codesign
    --remove-signature`` rewrites ``__LINKEDIT`` in a tool-version-dependent way (a
    2-byte drift between the dev and CI runners broke the prior approach), so we
    instead parse the ``LC_CODE_SIGNATURE`` load command and exclude exactly that
    byte range from the hash — no rewrite, no subprocess, no tool drift.
    """
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic == b"\xcf\xfa\xed\xfe":  # MH_MAGIC_64, little-endian
            header_size = 32
        elif magic == b"\xce\xfa\xed\xfe":  # MH_MAGIC, little-endian
            header_size = 28
        else:
            return None
        handle.seek(16)  # ncmds follows magic, cputype, cpusubtype, filetype
        (ncmds,) = struct.unpack("<I", handle.read(4))
        offset = header_size
        for _ in range(ncmds):
            handle.seek(offset)
            cmd, cmdsize = struct.unpack("<II", handle.read(8))
            if cmd == _LC_CODE_SIGNATURE:
                if cmdsize < 16:
                    return None
                dataoff, datasize = struct.unpack("<II", handle.read(8))
                return dataoff, datasize
            offset += cmdsize
    return None


def _normalized_file_bytes(
    path: Path,
    *,
    path_prefix: str | None,
    strip_codesignature: bool,
) -> bytes:
    """Read ``path`` for hashing, optionally signature-excluded and prefix-normalized."""
    data = path.read_bytes()
    if strip_codesignature:
        signature = _macho_codesignature_range(path)
        if signature is not None:
            offset, size = signature
            if size > 0 and offset + size <= len(data):
                data = data[:offset] + data[offset + size :]
    if path_prefix:
        prefix_bytes = path_prefix.encode()
        if prefix_bytes and prefix_bytes in data:
            data = data.replace(prefix_bytes, PATH_PREFIX_TOKEN)
    return data


def canonical_tree(
    root: Path,
    *,
    allow_symlinks: bool,
    ignored_names: frozenset[str] = frozenset(),
    reject_bytecode: bool = False,
    ignore_bytecode: bool = False,
    ignore_wheel_records: bool = False,
    path_prefix: str | None = None,
    strip_codesignature: bool = False,
    entries_out: list[str] | None = None,
) -> tuple[str, int]:
    """Hash a complete tree using location-independent relative names.

    ``reject_bytecode`` and ``ignore_bytecode`` are mutually exclusive in intent.
    The site-packages tree rejects bytecode so any post-clean ``.pyc`` insertion
    fails loudly. The runtime tree ignores bytecode instead, because the runtime
    root is the shared uv base interpreter prefix and its ``__pycache__`` content
    accumulates non-deterministically as other tools import the same interpreter.

    ``path_prefix`` (runtime tree only) is the resolved interpreter install prefix
    stripped from each file's content before hashing. The uv base prefix is
    relocated per machine — the prefix is HOME-embedded in ``libpython3.12.dylib``
    (the dynamically-linked interpreter core) and ``_sysconfigdata`` — so the raw
    bytes differ across machines even for the same release. Stripping the prefix
    pins the interpreter code while leaving the install location unbound.

    ``strip_codesignature`` (runtime tree only) hashes Mach-O files over their
    unsigned content. Relocation re-signs every patched Mach-O adhoc, and the adhoc
    CodeDirectory signs the install-name page, so signature bytes differ per
    machine for the same release; removing the signature (then normalizing the
    prefix) makes the interpreter code reproducible across machines.
    """

    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"identity root is not a directory: {root}")
    entries: list[str] = entries_out if entries_out is not None else []
    for directory, directories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in directories:
            candidate = directory_path / name
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
            if not candidate.is_dir():
                raise RuntimeError(f"identity tree contains a special directory: {candidate}")
            if name == "__pycache__":
                if reject_bytecode:
                    raise RuntimeError(f"identity tree contains a bytecode directory: {candidate}")
                if ignore_bytecode:
                    continue
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in files:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if ignore_bytecode and name.endswith(".pyc"):
                continue
            if candidate.is_symlink():
                if not allow_symlinks:
                    raise RuntimeError(f"identity tree contains a file symlink: {candidate}")
                target = os.readlink(candidate)
                resolved = (candidate.parent / target).resolve()
                if not resolved.is_relative_to(root):
                    raise RuntimeError(f"identity symlink escapes its tree: {candidate}")
                entries.append(f"L {hashlib.sha256(target.encode()).hexdigest()} {len(target.encode())} {relative}\n")
            elif candidate.is_file():
                if reject_bytecode and name.endswith(".pyc"):
                    raise RuntimeError(f"identity tree contains a bytecode file: {candidate}")
                wheel_record = (
                    ignore_wheel_records
                    and name == "RECORD"
                    and candidate.parent.name.endswith(".dist-info")
                )
                if name in ignored_names or wheel_record:
                    continue
                if path_prefix is None and not strip_codesignature:
                    digest_hex = sha256_file(candidate)
                    size = candidate.stat().st_size
                else:
                    data = _normalized_file_bytes(
                        candidate,
                        path_prefix=path_prefix,
                        strip_codesignature=strip_codesignature,
                    )
                    digest_hex = hashlib.sha256(data).hexdigest()
                    size = len(data)
                entries.append(f"F {digest_hex} {size} {relative}\n")
            else:
                raise RuntimeError(f"identity tree contains a special file: {candidate}")
    entries.sort()
    if not entries:
        raise RuntimeError(f"identity tree is empty: {root}")
    return hashlib.sha256("".join(entries).encode()).hexdigest(), len(entries)


def scrubbed_environment() -> dict[str, str]:
    """Require the caller to have removed every ambient environment variable."""

    actual = dict(os.environ)
    # macOS injects this process-local CoreFoundation value even through env -i.
    core_foundation = actual.pop("__CF_USER_TEXT_ENCODING", None)
    if core_foundation is not None and not re.fullmatch(
        r"0x[0-9A-Fa-f]+:0x0:0x0", core_foundation
    ):
        raise RuntimeError("oracle CoreFoundation text environment is malformed")
    if actual != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            f"oracle environment differs: expected {EXPECTED_ENVIRONMENT}, found {actual}"
        )
    os.environ.clear()
    os.environ.update(EXPECTED_ENVIRONMENT)
    return dict(EXPECTED_ENVIRONMENT)


def startup_identity(manifest_out: list[str] | None = None) -> dict[str, Any]:
    if sys.version.split()[0] != EXPECTED_PYTHON:
        raise RuntimeError(f"expected Python {EXPECTED_PYTHON}, found {sys.version.split()[0]}")
    environment = scrubbed_environment()
    flags = {
        "bytes_warning": sys.flags.bytes_warning,
        "debug": sys.flags.debug,
        "dev_mode": bool(sys.flags.dev_mode),
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "hash_randomization": sys.flags.hash_randomization,
        "ignore_environment": sys.flags.ignore_environment,
        "inspect": sys.flags.inspect,
        "int_max_str_digits": sys.flags.int_max_str_digits,
        "interactive": sys.flags.interactive,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "optimize": sys.flags.optimize,
        "quiet": sys.flags.quiet,
        "safe_path": bool(sys.flags.safe_path),
        "utf8_mode": sys.flags.utf8_mode,
        "verbose": sys.flags.verbose,
        "warn_default_encoding": sys.flags.warn_default_encoding,
    }
    if flags != {
        "bytes_warning": 0,
        "debug": 0,
        "dev_mode": False,
        "dont_write_bytecode": 1,
        "hash_randomization": 0,
        "ignore_environment": 0,
        "inspect": 0,
        "int_max_str_digits": 4300,
        "interactive": 0,
        "isolated": 0,
        "no_site": 1,
        "no_user_site": 1,
        "optimize": 0,
        "quiet": 0,
        "safe_path": True,
        "utf8_mode": 1,
        "verbose": 0,
        "warn_default_encoding": 0,
    }:
        raise RuntimeError(f"oracle requires scrubbed deterministic Python flags, found {flags}")
    if sys.pycache_prefix != "/dev/null":
        raise RuntimeError(
            f"oracle requires an inert pycache prefix, found {sys.pycache_prefix!r}"
        )
    if "site" in sys.modules or "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise RuntimeError("ambient Python startup hooks executed before oracle isolation")
    hash_seed_probe = hash("hyperion-m1-fixed-hash-probe")
    if hash_seed_probe != EXPECTED_HASH_SEED_PROBE:
        raise RuntimeError(
            f"oracle hash seed probe differs: expected {EXPECTED_HASH_SEED_PROBE}, "
            f"found {hash_seed_probe}"
        )

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
        runtime_root,
        allow_symlinks=True,
        ignored_names=frozenset({".DS_Store"}),
        ignore_bytecode=True,
        path_prefix=str(runtime_root),
        strip_codesignature=True,
        entries_out=manifest_out,
    )
    site_sha256, site_count = canonical_tree(
        # Wheel RECORD files are non-executable installer receipts and uv rewrites
        # their entry-script rows with the environment path. Every referenced
        # payload byte, plus every extra file, is still covered by this full tree.
        site_packages,
        allow_symlinks=False,
        ignored_names=frozenset({".DS_Store"}),
        reject_bytecode=True,
        ignore_wheel_records=True,
    )
    return {
        "python_executable_sha256": executable_sha256,
        "python_runtime_tree_sha256": runtime_sha256,
        "python_runtime_file_count": runtime_count,
        "site_packages_tree_sha256": site_sha256,
        "site_packages_file_count": site_count,
        "startup_flags": flags,
        "pycache_prefix": sys.pycache_prefix,
        "hash_seed_probe": hash_seed_probe,
        "environment": environment,
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
        raise SystemExit(
            "usage: isolated_oracle.py identity [--derive [--manifest]] | script PATH [ARGS...]"
        )
    action = sys.argv[1]
    if action == "identity":
        args = sys.argv[2:]
        derive = "--derive" in args
        manifest = "--manifest" in args
        if args and not (derive or manifest):
            raise SystemExit("usage: isolated_oracle.py identity [--derive [--manifest]]")
        manifest_lines: list[str] = []
        identity = startup_identity(manifest_out=manifest_lines if manifest else None)
        if not derive:
            validate_expected(identity)
        if manifest:
            for line in sorted(manifest_lines):
                print(line, end="")
            return
        print(
            json.dumps(
                {key: value for key, value in identity.items() if key != "site_packages"},
                sort_keys=True,
            )
        )
        return

    identity = startup_identity()
    validate_expected(identity)
    install_identity_module(identity)
    oracle_dir = Path(__file__).resolve().parent
    sys.path.extend([str(oracle_dir), str(identity["site_packages"])])
    if action == "script":
        if len(sys.argv) < 3:
            raise SystemExit("isolated_oracle.py script requires a target")

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
    else:
        raise SystemExit(f"unsupported isolated oracle action: {action}")


if __name__ == "__main__":
    main()
