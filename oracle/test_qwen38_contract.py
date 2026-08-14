#!/usr/bin/env python3
"""Model-free positive and hostile controls for the Qwen3.8 oracle contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import qwen38_contract


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "oracle" / "qwen38" / "contract.json"
SOURCE_MANIFEST_PATH = REPO_ROOT / "oracle" / "qwen38" / "source-manifest.json"
CASES_PATH = REPO_ROOT / "oracle" / "qwen38" / "cases.jsonl"


class Qwen38ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = qwen38_contract.load_json_no_duplicates(CONTRACT_PATH)

    def test_committed_contract_is_explicitly_unexecuted(self) -> None:
        result = qwen38_contract.validate_contract(REPO_ROOT, CONTRACT_PATH)
        self.assertTrue(result["contract_only_unexecuted"])
        self.assertFalse(result["native_executable"])
        self.assertFalse(result["support_accepted"])
        self.assertEqual(result["oracle_ids"], ["mlx_lm", "transformers"])
        self.assertEqual(result["case_count"], 20)

    def test_source_identity_matches_rust_and_recognition_fixture(self) -> None:
        model_lib = (REPO_ROOT / "crates" / "hyperion-model" / "src" / "lib.rs").read_text()
        self.assertIn('"Qwen/Qwen3.8-27B"', model_lib)
        self.assertIn(qwen38_contract.SOURCE_REVISION, model_lib)
        fixture = json.loads(
            (
                REPO_ROOT
                / "crates"
                / "hyperion-model"
                / "fixtures"
                / "qwen35-recognized-only"
                / "config.json"
            ).read_text()
        )
        self.assertEqual(fixture["model_type"], "qwen3_5")
        self.assertEqual(fixture["architectures"], ["Qwen3_5ForConditionalGeneration"])

    def test_source_inventory_is_complete_and_content_addressed(self) -> None:
        manifest = qwen38_contract.load_json_no_duplicates(SOURCE_MANIFEST_PATH)
        files = qwen38_contract.validate_source_manifest_value(manifest)
        self.assertEqual(set(files), qwen38_contract.EXPECTED_SOURCE_PATHS)
        self.assertEqual(len(files), 32)
        self.assertEqual(manifest["payload_bytes"], sum(item["size"] for item in files.values()))
        self.assertEqual(
            qwen38_contract.sha256_file(SOURCE_MANIFEST_PATH),
            qwen38_contract.SOURCE_MANIFEST_SHA256,
        )

    def test_cases_are_inputs_not_fake_oracle_outputs(self) -> None:
        cases = qwen38_contract.load_cases(CASES_PATH)
        forbidden = {
            "rendered",
            "input_ids",
            "logits",
            "hidden_states",
            "generated_tokens",
            "oracle_output",
            "agreement",
        }
        for case in cases:
            self.assertTrue(forbidden.isdisjoint(case))
            for tool in case["tools"]:
                self.assertEqual(set(tool), {"function", "type"})
                self.assertEqual(tool["type"], "function")
            for message in case["messages"]:
                if message["role"] == "assistant":
                    for call in message["tool_calls"]:
                        self.assertEqual(set(call), {"function", "type"})
                        self.assertEqual(call["type"], "function")
        coverage = {tag for case in cases for tag in case["coverage"]}
        self.assertEqual(coverage, set(qwen38_contract.REQUIRED_COVERAGE))
        stop_ids = {token for case in cases for token in case["stop_token_ids"]}
        self.assertEqual(stop_ids, {248044, 248046})
        dual_stop = next(case for case in cases if case["id"] == "generate-canonical-dual-stop")
        self.assertEqual(dual_stop["stop_token_ids"], [248046, 248044])
        self.assertEqual(dual_stop["max_steps"], 16)

    def test_overclaims_and_identity_drift_fail_closed(self) -> None:
        mutations = []

        support = copy.deepcopy(self.contract)
        support["status"]["support_accepted"] = True
        mutations.append(("support claim", support))

        source = copy.deepcopy(self.contract)
        source["source"]["revision"] = "0" * 40
        mutations.append(("source drift", source))

        unknown = copy.deepcopy(self.contract)
        unknown["accepted"] = True
        mutations.append(("unknown field", unknown))

        tolerance = copy.deepcopy(self.contract)
        tolerance["trace"]["tolerances"]["max_abs"] = 0.01
        mutations.append(("invented tolerance", tolerance))

        same_oracle = copy.deepcopy(self.contract)
        same_oracle["oracles"][1] = copy.deepcopy(same_oracle["oracles"][0])
        same_oracle["oracles"][1]["id"] = "mlx_lm"
        mutations.append(("non-independent oracle", same_oracle))

        missing_channel = copy.deepcopy(self.contract)
        missing_channel["trace"]["required_channels"].pop()
        mutations.append(("missing trace channel", missing_channel))

        missing_evidence = copy.deepcopy(self.contract)
        missing_evidence["required_evidence_identities"].pop()
        mutations.append(("missing evidence identity", missing_evidence))

        missing_coverage = copy.deepcopy(self.contract)
        missing_coverage["required_coverage"].remove("thinking_low")
        mutations.append(("missing coverage", missing_coverage))

        wrong_cases_hash = copy.deepcopy(self.contract)
        wrong_cases_hash["conversation"]["cases_sha256"] = "0" * 64
        mutations.append(("case drift", wrong_cases_hash))

        for name, mutated in mutations:
            with self.subTest(name=name):
                with self.assertRaises(qwen38_contract.ContractError):
                    qwen38_contract.validate_contract_value(REPO_ROOT, mutated)

    def test_duplicate_json_key_and_noncanonical_case_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate_json = root / "duplicate.json"
            duplicate_json.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
            with self.assertRaisesRegex(qwen38_contract.ContractError, "duplicate key"):
                qwen38_contract.load_json_no_duplicates(duplicate_json)

            first_line = CASES_PATH.read_text(encoding="utf-8").splitlines()[0]
            noncanonical = root / "noncanonical.jsonl"
            noncanonical.write_text(first_line.replace('"coverage":', '"coverage" :', 1) + "\n")
            with self.assertRaisesRegex(qwen38_contract.ContractError, "not canonical JSON"):
                qwen38_contract.load_cases(noncanonical)

            duplicate_case = root / "duplicate-case.jsonl"
            duplicate_case.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(qwen38_contract.ContractError, "repeats ID"):
                qwen38_contract.load_cases(duplicate_case)

            case = json.loads(first_line)
            case["unexpected"] = True
            unknown = root / "unknown.jsonl"
            unknown.write_bytes(qwen38_contract.canonical_json(case))
            with self.assertRaisesRegex(qwen38_contract.ContractError, "unknown"):
                qwen38_contract.load_cases(unknown)

            fake_coverage_case = json.loads(first_line)
            fake_coverage_case["coverage"] = qwen38_contract.REQUIRED_COVERAGE
            fake_coverage = root / "fake-coverage.jsonl"
            fake_coverage.write_bytes(qwen38_contract.canonical_json(fake_coverage_case))
            with self.assertRaisesRegex(qwen38_contract.ContractError, "not supported"):
                qwen38_contract.load_cases(fake_coverage)

    def test_case_semantics_reject_falsy_content_and_self_attested_coverage(self) -> None:
        cases = {case["id"]: case for case in qwen38_contract.load_cases(CASES_PATH)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            falsy_content = copy.deepcopy(cases["render-parallel-tool-calls"])
            falsy_content["messages"][1]["content"] = False
            path = root / "falsy-content.jsonl"
            path.write_bytes(qwen38_contract.canonical_json(falsy_content))
            with self.assertRaisesRegex(qwen38_contract.ContractError, "content must be a string"):
                qwen38_contract.load_cases(path)

            fake_preserve = copy.deepcopy(cases["render-thinking-default-xhigh"])
            fake_preserve["coverage"] = ["preserve_thinking_true"]
            fake_preserve["template_args"]["preserve_thinking"] = True
            path = root / "fake-preserve.jsonl"
            path.write_bytes(qwen38_contract.canonical_json(fake_preserve))
            with self.assertRaisesRegex(qwen38_contract.ContractError, "not supported"):
                qwen38_contract.load_cases(path)

            fake_parser = copy.deepcopy(cases["parser-single-call"])
            fake_parser["parser_input"] = "ordinary text"
            fake_parser["expected_text"] = "ordinary text"
            path = root / "fake-parser.jsonl"
            path.write_bytes(qwen38_contract.canonical_json(fake_parser))
            with self.assertRaisesRegex(qwen38_contract.ContractError, "not supported"):
                qwen38_contract.load_cases(path)

            changed_fallback = copy.deepcopy(cases["parser-malformed-fallback"])
            changed_fallback["expected_text"] = "truncated"
            path = root / "changed-fallback.jsonl"
            path.write_bytes(qwen38_contract.canonical_json(changed_fallback))
            with self.assertRaisesRegex(qwen38_contract.ContractError, "preserve the exact input"):
                qwen38_contract.load_cases(path)

    def test_noncanonical_relative_paths_are_rejected(self) -> None:
        for value in (
            "oracle//qwen38/cases.jsonl",
            "oracle/qwen38/./cases.jsonl",
            "oracle/qwen38/",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(qwen38_contract.ContractError, "not a canonical"):
                    qwen38_contract._safe_relative_path(value, "test path")

    def test_source_manifest_rejects_inventory_substitution(self) -> None:
        manifest = qwen38_contract.load_json_no_duplicates(SOURCE_MANIFEST_PATH)
        extra = copy.deepcopy(manifest)
        extra["files"].append(
            {
                "path": "unreviewed.bin",
                "size": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
                "storage": "git",
            }
        )
        extra["files"].sort(key=lambda item: item["path"])
        extra["payload_file_count"] += 1
        extra["payload_bytes"] += 1
        with self.assertRaisesRegex(qwen38_contract.ContractError, "path set differs"):
            qwen38_contract.validate_source_manifest_value(extra)

        duplicate = copy.deepcopy(manifest)
        duplicate["files"].append(copy.deepcopy(duplicate["files"][-1]))
        duplicate["payload_file_count"] += 1
        duplicate["payload_bytes"] += duplicate["files"][-1]["size"]
        with self.assertRaisesRegex(qwen38_contract.ContractError, "repeats path"):
            qwen38_contract.validate_source_manifest_value(duplicate)


class Qwen38SourceTreeTests(unittest.TestCase):
    @staticmethod
    def write_payload(root: Path, relative: str, payload: bytes) -> dict[str, object]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {
            "path": relative,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "storage": "git",
        }

    def make_tree(self, root: Path) -> dict[str, object]:
        files = [
            self.write_payload(root, "config.json", b"{}\n"),
            self.write_payload(root, "weights/model.safetensors", b"synthetic\n"),
        ]
        return {
            "repository": "synthetic/test",
            "revision": "1" * 40,
            "files": files,
        }

    def test_exact_synthetic_source_tree_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_tree(root)
            result = qwen38_contract._verify_exact_tree(root, manifest)
            self.assertTrue(result["exact_inventory"])
            self.assertTrue(result["symlinks_and_hardlinks_rejected"])
            self.assertEqual(result["payload_file_count"], 2)

    def test_extra_missing_mutated_and_symlinked_files_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_tree(root)

            extra = root / "extra.bin"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(qwen38_contract.ContractError, "inventory differs"):
                qwen38_contract._verify_exact_tree(root, manifest)
            extra.unlink()

            weights = root / "weights" / "model.safetensors"
            original = weights.read_bytes()
            weights.write_bytes(b"mutated\n")
            with self.assertRaisesRegex(qwen38_contract.ContractError, "size differs|hash differs"):
                qwen38_contract._verify_exact_tree(root, manifest)
            weights.write_bytes(original)

            config = root / "config.json"
            config.unlink()
            config.symlink_to(weights)
            with self.assertRaisesRegex(qwen38_contract.ContractError, "unsafe file"):
                qwen38_contract._verify_exact_tree(root, manifest)

    def test_hard_link_alias_is_rejected_even_when_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.bin"
            payload.write_bytes(b"same inode")
            alias = root / "alias.bin"
            try:
                os.link(payload, alias)
            except OSError as error:
                self.skipTest(f"hard links unavailable: {error}")
            item = {
                "size": len(b"same inode"),
                "sha256": hashlib.sha256(b"same inode").hexdigest(),
                "storage": "git",
            }
            manifest = {
                "repository": "synthetic/test",
                "revision": "1" * 40,
                "files": [
                    {"path": "alias.bin", **item},
                    {"path": "payload.bin", **item},
                ],
            }
            with self.assertRaisesRegex(qwen38_contract.ContractError, "hard-link alias"):
                qwen38_contract._verify_exact_tree(root, manifest)

    def test_missing_source_root_is_not_a_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with self.assertRaisesRegex(qwen38_contract.ContractError, "real directory"):
                qwen38_contract._verify_exact_tree(
                    missing,
                    {"repository": "synthetic/test", "revision": "1" * 40, "files": []},
                )

    def test_public_verifier_cannot_accept_an_arbitrary_empty_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(qwen38_contract.ContractError, "inventory differs"):
                qwen38_contract.verify_source_tree(root, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
