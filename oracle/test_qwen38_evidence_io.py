#!/usr/bin/env python3
"""Model-free positive, hostile, race, and fault-injection controls for evidence I/O."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import socket
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Callable
from unittest import mock

import qwen38_evidence_io as evidence_io
from qwen38_contract import canonical_json

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_CONTRACT_FILES = [
    "oracle/qwen38/cases.jsonl",
    "oracle/qwen38/contract.json",
    "oracle/qwen38/producer-contracts.json",
    "oracle/qwen38/source-manifest.json",
    "oracle/qwen38/trace-schema.json",
]
MARKER_PATH = "manifest/final.json"
MARKER_BYTES = b'{"marker":true}\n'


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_tree(root: Path, files: dict[str, bytes]) -> None:
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


class FailingHook(Exception):
    """Deterministic fault injected at a named boundary."""


def fail(name: str) -> Callable[..., None]:
    def hook(*_arguments: object) -> None:
        raise FailingHook(name)

    return hook


def raise_at_rename(rename: Callable[[], str]) -> str:
    raise FailingHook("at_rename")


class EvidenceIOTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.workspace = Path(self._temporary.name)
        self.staging = self.workspace / "staging"
        self.target_parent = self.workspace / "publish"
        self.staging.mkdir()
        self.target_parent.mkdir()
        self.target = self.target_parent / "evidence-bundle"

    def prepare_tree(self, files: dict[str, bytes] | None = None) -> None:
        write_tree(
            self.staging,
            files
            if files is not None
            else {
                "alpha.json": b'{"first":1}\n',
                "nested/beta.bin": bytes(range(256)),
                "nested/deep/gamma.txt": b"gamma\n",
            },
        )

    def publish(self, **kwargs: object) -> evidence_io.PublicationReceipt:
        return evidence_io.publish_tree(
            self.staging,
            self.target,
            MARKER_PATH,
            MARKER_BYTES,
            **kwargs,  # type: ignore[arg-type]
        )


class InventoryPositiveTests(EvidenceIOTestBase):
    def test_inventory_is_deterministic_across_creation_order(self) -> None:
        files = {
            "a/one.txt": b"one\n",
            "b/c/two.txt": b"two\n",
            "three.txt": b"three\n",
            "empty.txt": b"",
        }
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first_root = Path(left)
            second_root = Path(right)
            write_tree(first_root, files)
            write_tree(second_root, dict(reversed(list(files.items()))))
            first = evidence_io.inventory_tree(first_root)
            second = evidence_io.inventory_tree(second_root)
        self.assertEqual(first, second)
        self.assertEqual(
            [record["path"] for record in first],
            sorted(record["path"] for record in first),
        )
        self.assertEqual(
            evidence_io.tree_inventory_sha256(first),
            hashlib.sha256(canonical_json(first)).hexdigest(),
        )

    def test_inventory_records_are_exact(self) -> None:
        self.prepare_tree({"empty.txt": b"", "data.bin": bytes(range(256))})
        entries = evidence_io.inventory_tree(self.staging)
        self.assertEqual(
            entries,
            [
                {
                    "path": "data.bin",
                    "byte_length": 256,
                    "sha256": sha256_bytes(bytes(range(256))),
                },
                {"path": "empty.txt", "byte_length": 0, "sha256": sha256_bytes(b"")},
            ],
        )

    def test_large_file_is_hashed_in_chunks(self) -> None:
        payload = os.urandom(2 * evidence_io.HASH_CHUNK_BYTES + 128)
        self.prepare_tree({"big.bin": payload})
        entries = evidence_io.inventory_tree(self.staging)
        self.assertEqual(entries[0]["sha256"], sha256_bytes(payload))
        self.assertEqual(entries[0]["byte_length"], len(payload))

    def test_inventory_of_empty_tree_is_an_empty_sorted_collection(self) -> None:
        self.assertEqual(evidence_io.inventory_tree(self.staging), [])
        self.assertEqual(
            evidence_io.tree_inventory_sha256([]),
            hashlib.sha256(canonical_json([])).hexdigest(),
        )


class InventoryHostileTests(EvidenceIOTestBase):
    def assert_hostile(self, build: Callable[[Path], None]) -> None:
        build(self.staging)
        with self.assertRaises(evidence_io.EvidenceIOError):
            evidence_io.inventory_tree(self.staging)

    def test_symlink_root_is_rejected(self) -> None:
        link = self.workspace / "link"
        link.symlink_to(self.staging, target_is_directory=True)
        with self.assertRaisesRegex(evidence_io.UnsafeTreeError, "symlink"):
            evidence_io.inventory_tree(link)

    def test_non_directory_root_is_rejected(self) -> None:
        file_root = self.workspace / "file-root"
        file_root.write_bytes(b"not a directory")
        with self.assertRaisesRegex(evidence_io.UnsafeTreeError, "real directory"):
            evidence_io.inventory_tree(file_root)

    def test_missing_root_fails_closed(self) -> None:
        with self.assertRaisesRegex(evidence_io.UnsafeTreeError, "unavailable"):
            evidence_io.inventory_tree(self.workspace / "absent")

    def test_symlinked_file_is_rejected(self) -> None:
        def build(root: Path) -> None:
            write_tree(root, {"real.txt": b"data"})
            (root / "link.txt").symlink_to(root / "real.txt")

        self.assert_hostile(build)

    def test_symlinked_directory_is_rejected(self) -> None:
        def build(root: Path) -> None:
            write_tree(root, {"real/inner.txt": b"data"})
            (root / "alias").symlink_to(root / "real", target_is_directory=True)

        self.assert_hostile(build)

    def test_dangling_symlink_is_rejected(self) -> None:
        self.assert_hostile(lambda root: (root / "dangling").symlink_to(root / "nowhere"))

    def test_fifo_is_rejected(self) -> None:
        self.assert_hostile(lambda root: os.mkfifo(root / "pipe"))

    def test_unix_socket_is_rejected(self) -> None:
        def build(root: Path) -> None:
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.addCleanup(server.close)
            server.bind(str(root / "socket"))

        self.assert_hostile(build)

    def test_hard_link_alias_inside_tree_is_rejected(self) -> None:
        def build(root: Path) -> None:
            write_tree(root, {"original.txt": b"shared"})
            os.link(root / "original.txt", root / "alias.txt")

        self.assert_hostile(build)

    def test_file_with_external_hard_link_is_rejected(self) -> None:
        def build(root: Path) -> None:
            write_tree(root, {"inside.txt": b"shared"})
            os.link(root / "inside.txt", self.workspace / "outside.txt")

        self.assert_hostile(build)

    def test_non_utf8_entry_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(evidence_io.UnsafeTreeError, "UTF-8"):
            evidence_io._safe_entry_name(os.fsdecode(b"bad\xff.txt"), "tree entry")

    def test_backslash_entry_name_is_rejected(self) -> None:
        self.assert_hostile(lambda root: write_tree(root, {"back\\slash.txt": b"data"}))


class InventoryRaceTests(EvidenceIOTestBase):
    def test_file_replaced_between_enumeration_and_open_is_rejected(self) -> None:
        self.prepare_tree({"victim.txt": b"original"})
        victim = self.staging / "victim.txt"
        seen: set[str] = set()

        def replace(relative: str) -> None:
            if relative == "victim.txt" and relative not in seen:
                seen.add(relative)
                payload = victim.read_bytes() + b"replaced\n"
                victim.unlink()
                victim.write_bytes(payload)

        with self.assertRaisesRegex(evidence_io.TreeMutationError, "changed identity"):
            evidence_io.inventory_tree(self.staging, hooks={"before_entry_open": replace})

    def test_append_while_hashing_is_rejected(self) -> None:
        payload = b"x" * (evidence_io.HASH_CHUNK_BYTES + 16)
        self.prepare_tree({"growing.bin": payload})
        growing = self.staging / "growing.bin"
        seen: set[str] = set()

        def append(relative: str) -> None:
            if relative == "growing.bin" and relative not in seen:
                seen.add(relative)
                with growing.open("ab") as handle:
                    handle.write(b"tail")

        with self.assertRaisesRegex(evidence_io.TreeMutationError, "changed content"):
            evidence_io.inventory_tree(self.staging, hooks={"during_file_read": append})

    def test_truncate_while_hashing_is_rejected(self) -> None:
        payload = b"y" * (evidence_io.HASH_CHUNK_BYTES + 16)
        self.prepare_tree({"shrinking.bin": payload})
        shrinking = self.staging / "shrinking.bin"
        seen: set[str] = set()

        def truncate(relative: str) -> None:
            if relative == "shrinking.bin" and relative not in seen:
                seen.add(relative)
                with shrinking.open("r+b") as handle:
                    handle.truncate(4)

        with self.assertRaisesRegex(evidence_io.TreeMutationError, "changed content"):
            evidence_io.inventory_tree(self.staging, hooks={"during_file_read": truncate})


class TreeDigestTests(unittest.TestCase):
    def test_digest_requires_sorted_closed_records(self) -> None:
        record = {"path": "a", "byte_length": 1, "sha256": "0" * 64}
        with self.assertRaisesRegex(evidence_io.EvidenceIOError, "sorted"):
            evidence_io.tree_inventory_sha256(
                [{"path": "b", "byte_length": 1, "sha256": "0" * 64}, record]
            )
        with self.assertRaisesRegex(evidence_io.EvidenceIOError, "closed"):
            evidence_io.tree_inventory_sha256([{"path": "a", "sha256": "0" * 64}])

    def test_digest_is_not_labeled_as_a_bundle_or_seal_digest(self) -> None:
        self.assertFalse(hasattr(evidence_io, "bundle_inventory_sha256"))
        module_source = Path(evidence_io.__file__).read_text()
        self.assertNotIn("bundle_inventory_sha256", module_source)
        self.assertNotIn("hyperion.qwen38-detached-seal.v1", module_source)


class PublicationPositiveTests(EvidenceIOTestBase):
    def test_successful_publication_is_atomic_and_exact(self) -> None:
        self.prepare_tree()
        receipt = self.publish()
        self.assertTrue(self.target.is_dir())
        self.assertFalse(self.staging.exists())
        entries = evidence_io.inventory_tree(self.target)
        self.assertEqual(
            {record["path"] for record in entries},
            {"alpha.json", "nested/beta.bin", "nested/deep/gamma.txt", MARKER_PATH},
        )
        self.assertEqual((self.target / MARKER_PATH).read_bytes(), MARKER_BYTES)
        self.assertEqual(receipt.marker_path, MARKER_PATH)
        self.assertEqual(receipt.marker_sha256, sha256_bytes(MARKER_BYTES))
        self.assertEqual(receipt.file_count, 4)
        self.assertEqual(
            receipt.tree_inventory_sha256,
            evidence_io.tree_inventory_sha256(entries),
        )
        self.assertIn("RENAME", receipt.platform_operation)
        self.assertIn("EXCL", receipt.platform_operation)

    def test_receipt_carries_only_implementation_owned_facts(self) -> None:
        self.prepare_tree()
        receipt = self.publish()
        self.assertEqual(
            set(receipt.__dataclass_fields__),
            {
                "target_path",
                "tree_inventory_sha256",
                "file_count",
                "tree_bytes",
                "marker_path",
                "marker_sha256",
                "platform_operation",
            },
        )
        serialized = json.dumps(receipt.__dict__)
        for forbidden in ("pass", "accepted", "agreement", "support", "seal"):
            self.assertNotIn(forbidden, serialized)

    def test_second_publication_is_refused_and_first_target_unchanged(self) -> None:
        self.prepare_tree()
        first = self.publish()
        before = evidence_io.inventory_tree(self.target)
        self.staging.mkdir()
        self.prepare_tree({"alpha.json": b"second attempt\n"})
        with self.assertRaisesRegex(evidence_io.TargetExistsError, "already exists"):
            self.publish()
        self.assertEqual(evidence_io.inventory_tree(self.target), before)
        self.assertEqual(first.target_path, str(self.target.absolute()))

    def test_empty_staging_tree_publishes_with_only_the_marker(self) -> None:
        receipt = self.publish()
        entries = evidence_io.inventory_tree(self.target)
        self.assertEqual([record["path"] for record in entries], [MARKER_PATH])
        self.assertEqual(receipt.file_count, 1)

    def test_marker_bytes_are_written_exactly(self) -> None:
        self.prepare_tree()
        exact = b"\x00\xffno-newline\x80"
        evidence_io.publish_tree(self.staging, self.target, "m/marker.bin", exact)
        self.assertEqual((self.target / "m" / "marker.bin").read_bytes(), exact)


class PublicationHostileTests(EvidenceIOTestBase):
    def test_existing_target_contents_survive_a_refused_publication(self) -> None:
        self.prepare_tree()
        self.target.mkdir()
        (self.target / "occupied").write_bytes(b"occupied")
        with self.assertRaises(evidence_io.TargetExistsError):
            self.publish()
        self.assertEqual((self.target / "occupied").read_bytes(), b"occupied")

    def test_target_created_immediately_before_rename_does_not_overwrite(self) -> None:
        self.prepare_tree()

        def race() -> None:
            self.target.mkdir()
            (self.target / "racer").write_bytes(b"racer")

        with self.assertRaises(evidence_io.TargetExistsError):
            self.publish(hooks={"before_rename": lambda: race()})
        self.assertEqual((self.target / "racer").read_bytes(), b"racer")
        self.assertTrue(self.staging.is_dir())

    def test_noncanonical_marker_paths_are_rejected(self) -> None:
        self.prepare_tree()
        for bad in (
            "/absolute/marker.json",
            "../escape.json",
            "nested/../../escape.json",
            "back\\slash.json",
            "",
            "a//double.json",
            "trailing/",
        ):
            with self.subTest(marker=bad):
                staging = self.workspace / f"staging-{uuid.uuid4().hex}"
                staging.mkdir()
                write_tree(staging, {"payload.txt": b"payload"})
                target = self.target_parent / f"target-{uuid.uuid4().hex}"
                with self.assertRaises(evidence_io.UnsafePathError):
                    evidence_io.publish_tree(staging, target, bad, MARKER_BYTES)
                self.assertFalse(target.exists())

    def test_marker_collision_is_refused(self) -> None:
        self.prepare_tree()
        write_tree(self.staging, {MARKER_PATH: b"pre-existing"})
        with self.assertRaisesRegex(evidence_io.MarkerError, "already exists"):
            self.publish()
        self.assertFalse(self.target.exists())

    def test_marker_parent_symlink_is_refused(self) -> None:
        self.prepare_tree()
        outside = self.workspace / "outside"
        outside.mkdir()
        (self.staging / "manifest").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(evidence_io.EvidenceIOError):
            self.publish()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(outside.iterdir()), [])

    def test_non_directory_target_parent_is_rejected(self) -> None:
        self.prepare_tree()
        blocked = self.workspace / "blocked"
        blocked.write_bytes(b"not a directory")
        with self.assertRaisesRegex(evidence_io.UnsafePathError, "real director"):
            evidence_io.publish_tree(
                self.staging, blocked / "bundle", MARKER_PATH, MARKER_BYTES
            )
        self.assertFalse((blocked / "bundle").exists())

    def test_target_parent_symlink_is_rejected(self) -> None:
        self.prepare_tree()
        real = self.target_parent / "real"
        real.mkdir()
        link = self.workspace / "parent-link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaises(evidence_io.UnsafePathError):
            evidence_io.publish_tree(self.staging, link / "bundle", MARKER_PATH, MARKER_BYTES)
        self.assertFalse((real / "bundle").exists())

    def test_target_inside_or_containing_staging_is_rejected(self) -> None:
        cases = [
            ("inside staging", lambda staging: staging / "inner", "sibling"),
            ("contains staging", lambda staging: self.workspace / "outer", "sibling"),
        ]
        for label, make_target, expected in cases:
            with self.subTest(target=label):
                staging = self.workspace / f"staging-{uuid.uuid4().hex}"
                staging.mkdir()
                write_tree(staging, {"payload.txt": b"payload"})
                with self.assertRaisesRegex(evidence_io.UnsafePathError, expected):
                    evidence_io.publish_tree(
                        staging, make_target(staging), MARKER_PATH, MARKER_BYTES
                    )
        self.assertFalse((self.workspace / "outer").exists())

    def test_target_traversal_forms_are_rejected(self) -> None:
        cases = [
            ("relative traversal", self.target_parent / ".." / "escape", "escapes upward"),
            (
                "backslash name",
                self.target_parent / "with\\backslash",
                "unsafe final component",
            ),
        ]
        for label, bad_target, expected in cases:
            with self.subTest(target=label):
                staging = self.workspace / f"staging-{uuid.uuid4().hex}"
                staging.mkdir()
                write_tree(staging, {"payload.txt": b"payload"})
                with self.assertRaisesRegex(evidence_io.UnsafePathError, expected):
                    evidence_io.publish_tree(
                        staging, bad_target, MARKER_PATH, MARKER_BYTES
                    )
        self.assertFalse((self.workspace / "escape").exists())
        self.assertFalse((self.target_parent / "with\\backslash").exists())

    def test_cross_filesystem_publication_is_refused(self) -> None:
        self.prepare_tree()
        original = evidence_io._descriptor_device
        with mock.patch.object(
            evidence_io, "_descriptor_device", lambda descriptor: original(descriptor) + 1
        ):
            with self.assertRaisesRegex(
                evidence_io.CrossDeviceError, "different filesystems"
            ):
                self.publish()
        self.assertFalse(self.target.exists())
        self.assertTrue(self.staging.is_dir())

    def test_unsupported_platform_fails_closed(self) -> None:
        self.prepare_tree()
        with mock.patch.object(evidence_io.sys, "platform", "plan9"):
            with self.assertRaisesRegex(
                evidence_io.UnsupportedPlatformError, "no atomic no-replace"
            ):
                self.publish()
        self.assertFalse(self.target.exists())
        self.assertTrue(self.staging.is_dir())

    def test_linux_enosys_rename_is_an_unsupported_platform(self) -> None:
        self.prepare_tree()

        class FakePrimitive:
            argtypes: list[object] = []
            restype: object = None

            def __call__(self, *_arguments: object) -> int:
                ctypes.set_errno(errno.ENOSYS)
                return -1

        class FakeLibc:
            def __init__(self) -> None:
                self.renameat2 = FakePrimitive()

        original_libc = evidence_io._LIBC
        evidence_io._LIBC = FakeLibc()
        try:
            with mock.patch.object(evidence_io.sys, "platform", "linux"):
                with self.assertRaisesRegex(
                    evidence_io.UnsupportedPlatformError, "not supported by this kernel"
                ):
                    self.publish()
        finally:
            evidence_io._LIBC = original_libc
        self.assertFalse(self.target.exists())
        self.assertTrue(self.staging.is_dir())


class PublicationFaultInjectionTests(EvidenceIOTestBase):
    def boundary_hooks(self) -> dict[str, Callable[..., object]]:
        return {
            "after_initial_inventory": fail("after_initial_inventory"),
            "after_final_marker": fail("after_final_marker"),
            "after_final_inventory": fail("after_final_inventory"),
            "during_file_sync": fail("during_file_sync"),
            "during_directory_sync": fail("during_directory_sync"),
            "before_rename": fail("before_rename"),
        }

    def test_pre_rename_failures_leave_the_target_absent(self) -> None:
        for boundary, hook in self.boundary_hooks().items():
            with self.subTest(boundary=boundary):
                staging = self.workspace / f"staging-{uuid.uuid4().hex}"
                staging.mkdir()
                write_tree(staging, {"payload.txt": b"payload"})
                target = self.target_parent / f"target-{uuid.uuid4().hex}"
                sibling = self.target_parent / f"sibling-{uuid.uuid4().hex}"
                sibling.write_bytes(b"untouched")
                with self.assertRaises(FailingHook):
                    evidence_io.publish_tree(
                        staging, target, MARKER_PATH, MARKER_BYTES, hooks={boundary: hook}
                    )
                self.assertFalse(target.exists())
                self.assertEqual(sibling.read_bytes(), b"untouched")

    def test_at_rename_hook_failure_is_conservatively_uncertain(self) -> None:
        self.prepare_tree()
        with self.assertRaises(evidence_io.DurablePublicationUncertainError):
            self.publish(hooks={"at_rename": raise_at_rename})
        self.assertFalse(self.target.exists())

    def test_post_rename_parent_sync_failure_is_not_durable_success(self) -> None:
        self.prepare_tree()

        def parent_sync_failure() -> None:
            raise OSError(5, "injected parent fsync failure")

        with self.assertRaises(evidence_io.DurablePublicationUncertainError):
            self.publish(hooks={"during_parent_sync": parent_sync_failure})
        self.assertTrue(self.target.is_dir())
        self.assertEqual((self.target / MARKER_PATH).read_bytes(), MARKER_BYTES)

    def test_after_rename_hook_failure_is_classified_as_uncertain(self) -> None:
        self.prepare_tree()
        with self.assertRaises(evidence_io.DurablePublicationUncertainError) as caught:
            self.publish(hooks={"after_rename": fail("after_rename")})
        self.assertIsInstance(caught.exception.__cause__, FailingHook)
        self.assertTrue(self.target.is_dir())

    def test_staging_mutation_between_inventories_is_rejected(self) -> None:
        self.prepare_tree()
        seen: set[None] = set()

        def mutate() -> None:
            if not seen:
                seen.add(None)
                (self.staging / "alpha.json").write_bytes(b"mutated\n")

        with self.assertRaisesRegex(evidence_io.TreeMutationError, "between inventories"):
            self.publish(hooks={"after_final_marker": mutate})
        self.assertFalse(self.target.exists())

    def test_optional_staging_cleanup_removes_only_the_proven_root(self) -> None:
        self.prepare_tree()
        with self.assertRaises(FailingHook):
            self.publish(
                hooks={"before_rename": fail("before_rename")},
                remove_staging_on_failure=True,
            )
        self.assertFalse(self.staging.exists())
        self.assertFalse(self.target.exists())
        self.assertTrue(self.target_parent.is_dir())

    def test_cleanup_refuses_a_swapped_staging_root(self) -> None:
        self.prepare_tree()
        seen: set[None] = set()

        def swap() -> None:
            if not seen:
                seen.add(None)
                shutil.rmtree(self.staging)
                self.staging.mkdir()

        with self.assertRaises(evidence_io.TreeMutationError) as caught:
            self.publish(
                hooks={"after_initial_inventory": swap, "before_rename": fail("before_rename")},
                remove_staging_on_failure=True,
            )
        self.assertIsInstance(caught.exception.__cause__, evidence_io.UnsafeTreeError)
        self.assertTrue(self.staging.is_dir())
        self.assertFalse(self.target.exists())

    def test_staging_root_replaced_after_sync_is_refused_before_the_rename(self) -> None:
        self.prepare_tree()
        seen: set[None] = set()

        def swap() -> None:
            if not seen:
                seen.add(None)
                os.rename(self.staging, self.workspace / "staging-superseded")
                self.staging.mkdir()
                (self.staging / "imposter.txt").write_bytes(b"imposter")

        with self.assertRaisesRegex(
            evidence_io.TreeMutationError, "replaced before the atomic transition"
        ):
            self.publish(hooks={"before_rename": swap})
        self.assertFalse(self.target.exists())
        self.assertTrue((self.staging / "imposter.txt").exists())

    def test_file_added_after_the_final_inventory_is_refused(self) -> None:
        self.prepare_tree()
        seen: set[None] = set()

        def smuggle() -> None:
            if not seen:
                seen.add(None)
                (self.staging / "smuggled.txt").write_bytes(b"smuggled")

        with self.assertRaisesRegex(
            evidence_io.TreeMutationError, "differs from the accepted inventory"
        ):
            self.publish(hooks={"after_final_inventory": smuggle})
        self.assertFalse(self.target.exists())

    def test_file_deleted_after_the_final_inventory_is_refused(self) -> None:
        self.prepare_tree()
        seen: set[None] = set()

        def remove_pending() -> None:
            if not seen:
                seen.add(None)
                (self.staging / "nested" / "deep" / "gamma.txt").unlink()

        with self.assertRaisesRegex(
            evidence_io.TreeMutationError, "no longer matches the accepted inventory"
        ):
            self.publish(hooks={"after_final_inventory": remove_pending})
        self.assertFalse(self.target.exists())

    def test_same_size_rewrite_after_the_final_inventory_is_refused(self) -> None:
        self.prepare_tree({"alpha.json": b'{"first":1}\n'})
        seen: set[None] = set()

        def rewrite() -> None:
            if not seen:
                seen.add(None)
                (self.staging / "alpha.json").write_bytes(b'{"first":2}\n')

        with self.assertRaisesRegex(
            evidence_io.TreeMutationError, "differs from the accepted inventory"
        ):
            self.publish(hooks={"after_final_inventory": rewrite})
        self.assertFalse(self.target.exists())

    def test_file_added_during_the_directory_sync_is_refused(self) -> None:
        self.prepare_tree()
        seen: set[None] = set()

        def smuggle(prefix: str) -> None:
            if prefix == "" and not seen:
                seen.add(None)
                (self.staging / "smuggled.txt").write_bytes(b"smuggled")

        with self.assertRaisesRegex(
            evidence_io.TreeMutationError, "no longer matches the accepted inventory"
        ):
            self.publish(hooks={"during_directory_sync": smuggle})
        self.assertFalse(self.target.exists())


class FrozenContractTests(unittest.TestCase):
    def test_frozen_qwen_contract_files_are_regular_and_present(self) -> None:
        for relative in FROZEN_CONTRACT_FILES:
            path = REPO_ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertFalse(path.is_symlink(), relative)
            self.assertRegex(sha256_bytes(path.read_bytes()), r"[0-9a-f]{64}", relative)

    def test_module_makes_no_verdict_or_support_claim(self) -> None:
        module_source = Path(evidence_io.__file__).read_text()
        for forbidden in (
            "bundle_inventory_sha256",
            "hyperion.qwen38-detached-seal.v1",
            "support_accepted",
            "reference_agreement",
        ):
            self.assertNotIn(forbidden, module_source)


if __name__ == "__main__":
    unittest.main()
