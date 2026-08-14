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
TRACE_SCHEMA_PATH = REPO_ROOT / "oracle" / "qwen38" / "trace-schema.json"
PRODUCER_CONTRACTS_PATH = REPO_ROOT / "oracle" / "qwen38" / "producer-contracts.json"


class Qwen38ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = qwen38_contract.load_json_no_duplicates(CONTRACT_PATH)
        self.trace_schema = qwen38_contract.load_json_no_duplicates(TRACE_SCHEMA_PATH)
        self.producer_contracts = qwen38_contract.load_json_no_duplicates(
            PRODUCER_CONTRACTS_PATH
        )
        self.cases = qwen38_contract.load_cases(CASES_PATH)

    def test_committed_contract_is_explicitly_unexecuted(self) -> None:
        result = qwen38_contract.validate_contract(REPO_ROOT, CONTRACT_PATH)
        self.assertTrue(result["contract_only_unexecuted"])
        self.assertFalse(result["native_executable"])
        self.assertFalse(result["support_accepted"])
        self.assertEqual(result["oracle_ids"], ["mlx_lm", "transformers"])
        self.assertEqual(result["case_count"], 20)
        self.assertTrue(result["trace_schema_frozen"])
        self.assertTrue(result["producer_semantics_frozen"])
        self.assertFalse(result["producer_environments_frozen"])
        self.assertEqual(
            result["trace_schema_sha256"], qwen38_contract.TRACE_SCHEMA_SHA256
        )
        self.assertEqual(
            result["producer_contracts_sha256"],
            qwen38_contract.PRODUCER_CONTRACTS_SHA256,
        )

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

        wrong_trace_schema = copy.deepcopy(self.contract)
        wrong_trace_schema["trace"]["schema_sha256"] = "0" * 64
        mutations.append(("trace schema substitution", wrong_trace_schema))

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

    def test_trace_recipe_covers_free_running_thinking_tools_and_history(self) -> None:
        validated = qwen38_contract.validate_trace_schema_value(
            self.trace_schema,
            self.cases,
        )
        plan = validated["case_plan"]
        behavior_ids = [item["case_id"] for item in plan["behavior_cases"]]
        self.assertEqual(behavior_ids, qwen38_contract.BEHAVIOR_CASE_IDS)
        self.assertIn("render-thinking-default-xhigh", behavior_ids)
        self.assertIn("render-system-first-tools", behavior_ids)
        self.assertIn("render-tool-results-preserve-thinking-true", behavior_ids)
        self.assertEqual(
            [run["id"] for run in validated["runs"]],
            qwen38_contract.REQUIRED_TRACE_RUNS,
        )
        self.assertEqual(validated["topk"]["width"], 32)
        self.assertIsNone(validated["tolerances"]["numeric_thresholds"])

    def test_trace_schema_rejects_discretionary_coverage_and_layout_drift(self) -> None:
        mutations = []

        missing_behavior = copy.deepcopy(self.trace_schema)
        missing_behavior["case_plan"]["behavior_cases"].pop()
        mutations.append(("missing free-running behavior", missing_behavior))

        wrong_layer = copy.deepcopy(self.trace_schema)
        wrong_layer["selections"]["gdn_state_layers"][0] = 3
        mutations.append(("full-attention layer selected as GDN", wrong_layer))

        hidden_transpose = copy.deepcopy(self.trace_schema)
        hidden_transpose["canonical_normalization"]["gdn_recurrent_transform"][
            "mlx_lm"
        ] = "identity"
        mutations.append(("equal-shaped recurrent transpose omitted", hidden_transpose))

        conv_width = copy.deepcopy(self.trace_schema)
        conv_width["native_layouts"]["transformers"]["gdn_conv_shape"][-1] = 3
        mutations.append(("Transformers convolution cache truncated by producer", conv_width))

        kv_tail = copy.deepcopy(self.trace_schema)
        kv_tail["native_layouts"]["mlx_lm"]["logical_offset_required"] = False
        mutations.append(("mlx-lm allocated KV tail accepted", kv_tail))

        topk = copy.deepcopy(self.trace_schema)
        topk["topk"]["internal_rank_count"] = 32
        mutations.append(("rank-33 cutoff removed", topk))

        unknown = copy.deepcopy(self.trace_schema)
        unknown["channels"][0]["producer_may_skip"] = True
        mutations.append(("unknown channel policy", unknown))

        for name, mutated in mutations:
            with self.subTest(name=name):
                with self.assertRaises(qwen38_contract.ContractError):
                    qwen38_contract.validate_trace_schema_value(mutated, self.cases)

    def test_every_reviewed_trace_semantic_is_content_addressed(self) -> None:
        mutations = []

        def mutate(name, editor):
            candidate = copy.deepcopy(self.trace_schema)
            editor(candidate)
            mutations.append((name, candidate))

        mutate("empty run case set", lambda value: value["runs"][0].update(cases="none"))
        mutate(
            "disabled instrumentation",
            lambda value: value["runs"][1].update(instrumented=False),
        )
        mutate("sampling decode", lambda value: value["runs"][2].update(decode="sample"))
        mutate("missing snapshots", lambda value: value["runs"][3].update(state_snapshots=[]))
        mutate(
            "prediction frame drift",
            lambda value: value["prediction_frame"].update(prompt_call="ambiguous"),
        )
        mutate(
            "hidden prefix ambiguity",
            lambda value: value["selections"].update(
                prompt_hidden_positions="sorted_unique(0,2,3,4,T-1)"
            ),
        )
        mutate(
            "logits dtype drift",
            lambda value: value["channels"][5].update(stored_dtype="u32le"),
        )
        mutate(
            "native axes drift",
            lambda value: value["native_layouts"]["transformers"].update(
                full_attention_kv_axes=["wrong"]
            ),
        )
        mutate(
            "descriptor identity removed",
            lambda value: value["serialization"]["tensor_descriptor_required_fields"].remove(
                "sha256"
            ),
        )
        mutate("empty arm inventory", lambda value: value["bundle"].update(arm_files=[]))
        mutate(
            "bundle traversal",
            lambda value: value["bundle"].update(arm_root="../../escape"),
        )
        mutate(
            "metric formula drift",
            lambda value: value["comparison_metrics"]["numeric"].update(max_abs="mean"),
        )
        mutate(
            "open event records",
            lambda value: value["record_schemas"]["common_constraints"].update(
                records_are_closed=False
            ),
        )
        mutate(
            "parser evidence moved into producer arm",
            lambda value: value["record_schemas"]["events_jsonl"]["record_types"].insert(
                2, "parser_outcome"
            ),
        )
        mutate(
            "rejection payload disconnected",
            lambda value: value["record_schemas"]["events_jsonl"][
                "required_fields_by_type"
            ]["render_rejection"].remove("rejection_payload_path"),
        )
        mutate(
            "wrong KV capacity",
            lambda value: value["native_layouts"]["mlx_lm"].update(
                full_attention_kv_shape=[1, 4, "ceil_div(L,256)*256", 256]
            ),
        )
        mutate(
            "KV recurrence drift",
            lambda value: value["native_layouts"]["mlx_lm"]["capacity_recurrence"].update(
                step=128
            ),
        )
        mutate(
            "frame grammar opened",
            lambda value: value["record_schemas"]["common_constraints"].update(
                frame_ids="arbitrary"
            ),
        )
        mutate(
            "shared verifier evidence unsealed",
            lambda value: value["record_schemas"]["detached_seal_json"][
                "required_fields"
            ].remove("bundle_inventory_sha256"),
        )
        mutate(
            "top-k payload opened",
            lambda value: value["record_schemas"]["payload_manifest_jsonl"][
                "canonical_json_payload_schemas"
            ]["topk"].update(closed_object=False),
        )
        mutate(
            "stop coverage overclaim",
            lambda value: value["case_plan"]["stop_coverage_semantics"].update(
                stop_endoftext="observed stop termination"
            ),
        )
        mutate(
            "run-end ownership opened",
            lambda value: value["record_schemas"]["events_jsonl"][
                "cardinality_and_sequence"
            ].pop(),
        )
        mutate("missing comparison edge", lambda value: value.update(validation_edges=[]))

        for name, mutated in mutations:
            with self.subTest(name=name):
                with self.assertRaises(qwen38_contract.ContractError):
                    qwen38_contract.validate_trace_schema_value(mutated, self.cases)

    def test_producer_contracts_are_isolated_unbuilt_and_kernel_explicit(self) -> None:
        validated = qwen38_contract.validate_producer_contracts_value(
            REPO_ROOT,
            self.producer_contracts,
        )
        arms = {arm["id"]: arm for arm in validated["arms"]}
        self.assertEqual(
            arms["transformers"]["environment"]["lock_state"],
            "separate_lock_required_before_execution",
        )
        self.assertEqual(
            arms["transformers"]["execution"]["optional_imports_must_be_absent"],
            ["causal_conv1d", "fla", "flash_attn", "kernels", "xformers"],
        )
        self.assertEqual(
            arms["mlx_lm"]["execution"]["gdn_primary_path"],
            "stock_metal_eval_kernel",
        )
        self.assertEqual(
            validated["publication"]["state"],
            "required_behavior_not_implemented",
        )
        self.assertIn("numeric_tolerances", validated["unfrozen"])

    def test_producer_contract_mutations_fail_closed(self) -> None:
        mutations = []

        shared_lock = copy.deepcopy(self.producer_contracts)
        shared_lock["arms"][0]["environment"]["lock_state"] = (
            "existing_lock_candidate_unexecuted_for_qwen"
        )
        shared_lock["arms"][0]["environment"]["current_oracle_lock_may_not_be_used"] = False
        mutations.append(("shared oracle lock", shared_lock))

        optional_fla = copy.deepcopy(self.producer_contracts)
        optional_fla["arms"][0]["execution"]["optional_imports_must_be_absent"].remove(
            "fla"
        )
        mutations.append(("optional FLA path", optional_fla))

        silent_ops = copy.deepcopy(self.producer_contracts)
        silent_ops["arms"][1]["execution"]["gdn_ops_fallback"] = "allowed"
        mutations.append(("silent MLX ops fallback", silent_ops))

        overclaim = copy.deepcopy(self.producer_contracts)
        overclaim["state"] = "executed"
        mutations.append(("execution overclaim", overclaim))

        random_text = copy.deepcopy(self.producer_contracts)
        random_text["arms"][0]["load"][
            "randomly_initialized_text_parameters_allowed"
        ] = True
        mutations.append(("random text parameters", random_text))

        for name, mutated in mutations:
            with self.subTest(name=name):
                with self.assertRaises(qwen38_contract.ContractError):
                    qwen38_contract.validate_producer_contracts_value(REPO_ROOT, mutated)

    def test_every_reviewed_producer_semantic_is_content_addressed(self) -> None:
        mutations = []

        def mutate(name, editor):
            candidate = copy.deepcopy(self.producer_contracts)
            editor(candidate)
            mutations.append((name, candidate))

        mutate(
            "repository substitution",
            lambda value: value["arms"][0].update(repository="attacker/fork"),
        )
        mutate(
            "load source substitution",
            lambda value: value["arms"][0]["load"].update(source="network"),
        )
        mutate(
            "pre-pruned MLX source substitution",
            lambda value: value["arms"][1]["load"].update(
                source="prepruned_text_only_tree"
            ),
        )
        mutate(
            "GDN route substitution",
            lambda value: value["arms"][0]["execution"].update(gdn_prefill_path="fla"),
        )
        mutate(
            "Python substitution",
            lambda value: value["arms"][0]["environment"].update(python="latest"),
        )
        mutate(
            "accelerator substitution",
            lambda value: value["arms"][0]["environment"].update(accelerator="cpu"),
        )
        mutate(
            "MLX version substitution",
            lambda value: value["arms"][1]["environment"].update(mlx="9.9.9"),
        )
        mutate(
            "MLX stream substitution",
            lambda value: value["arms"][1]["execution"].update(stream="cpu"),
        )
        mutate(
            "seed substitution",
            lambda value: value["arms"][1]["determinism"].update(
                python_numpy_mlx_seed=999
            ),
        )
        mutate(
            "receipt field removed",
            lambda value: value["receipt_required_fields"].remove("argv"),
        )
        mutate(
            "receipt value contract removed",
            lambda value: value["receipt_field_contracts"][
                "nonempty_array_fields"
            ].pop("per_call_kernel_route_evidence"),
        )
        mutate(
            "loader provenance removed",
            lambda value: value["arms"][1]["critical_sources"].pop(),
        )
        mutate(
            "input preparation drift",
            lambda value: value["input_preparation"].update(
                tokenizer_loader="AutoTokenizer.from_pretrained(source_root,use_fast=True)"
            ),
        )
        mutate(
            "parameter closure bypass",
            lambda value: value["parameter_closure"].update(
                execution_may_begin_before_implementation=True
            ),
        )
        mutate(
            "source tensor count drift",
            lambda value: value["parameter_closure"]["expected_source_tensor_counts"].update(
                total=1200
            ),
        )
        mutate(
            "Transformers input dtype drift",
            lambda value: value["arms"][0]["execution"].update(
                input_ids_dtype="torch.int32"
            ),
        )
        mutate(
            "runner trust overclaim",
            lambda value: value["trust_boundary"].update(
                remote_hardware_attestation_provided=True
            ),
        )
        mutate(
            "process evidence removed",
            lambda value: value["receipt_required_fields"].remove("process_inventory"),
        )
        mutate(
            "nested receipt type coverage removed",
            lambda value: value["receipt_field_contracts"]["nested_field_types"][
                "sha256"
            ].remove("process_entry.environment_values_sha256"),
        )
        mutate(
            "lazy MLX capture",
            lambda value: value["arms"][1]["execution"].update(
                mx_eval_logits_and_all_cache_leaves_after_every_forward=False
            ),
        )
        mutate(
            "split-K drift",
            lambda value: value["arms"][0]["determinism"].update(
                bfloat16_reduced_precision_and_split_k=[False, True]
            ),
        )
        mutate(
            "instrumented route removed",
            lambda value: value["receipt_field_contracts"]["nonempty_array_fields"].pop(
                "effective_forward_routes_and_static_kwargs"
            ),
        )
        mutate(
            "memory diagnostics made comparable",
            lambda value: value["receipt_field_contracts"]["nested_field_types"][
                "boolean"
            ].remove(
                "memory_and_swap_before_peak_after.diagnostic_only_noncomparable"
            ),
        )
        mutate(
            "memory authority drift",
            lambda value: value["receipt_field_contracts"][
                "per_arm_memory_diagnostic_profiles"
            ]["mlx_lm"].update(accelerator_memory_authority="process_rss"),
        )

        for name, mutated in mutations:
            with self.subTest(name=name):
                with self.assertRaises(qwen38_contract.ContractError):
                    qwen38_contract.validate_producer_contracts_value(REPO_ROOT, mutated)

    def test_asymmetric_recurrent_layout_proves_the_invisible_transpose(self) -> None:
        shape = [1, 2, 3, 5]
        values = list(range(30))
        transformed, transformed_shape = qwen38_contract.normalize_synthetic_layout(
            values,
            shape,
            "transpose_last_two",
        )
        self.assertEqual(transformed_shape, [1, 2, 5, 3])
        self.assertNotEqual(transformed, values)
        self.assertEqual(transformed[:6], [0, 5, 10, 1, 6, 11])
        round_trip, original_shape = qwen38_contract.normalize_synthetic_layout(
            transformed,
            transformed_shape,
            "transpose_last_two",
        )
        self.assertEqual(original_shape, shape)
        self.assertEqual(round_trip, values)

    def test_convolution_history_and_logical_kv_tail_normalize_independently(self) -> None:
        conv, conv_shape = qwen38_contract.normalize_synthetic_layout(
            list(range(8)),
            [1, 2, 4],
            "take_last_3_then_transpose_0_2_1",
        )
        self.assertEqual(conv_shape, [1, 3, 2])
        self.assertEqual(conv, [1, 5, 2, 6, 3, 7])

        kv, kv_shape = qwen38_contract.normalize_synthetic_layout(
            list(range(512)),
            [1, 1, 256, 2],
            "slice_token_axis_to_logical_offset",
            logical_offset=3,
            call_token_counts=[3],
        )
        self.assertEqual(kv_shape, [1, 1, 3, 2])
        self.assertEqual(kv, [0, 1, 2, 3, 4, 5])
        self.assertEqual(
            qwen38_contract.mlx_kv_capacity_after_calls([3, 1, 1, 295]),
            (300, 517),
        )

        with self.assertRaisesRegex(qwen38_contract.ContractError, "pinned mlx-lm recurrence"):
            qwen38_contract.normalize_synthetic_layout(
                list(range(16)),
                [1, 1, 8, 2],
                "slice_token_axis_to_logical_offset",
                logical_offset=3,
                call_token_counts=[3],
            )

        with self.assertRaisesRegex(qwen38_contract.ContractError, r"L<=C<L\+256"):
            qwen38_contract.normalize_synthetic_layout(
                list(range(259)),
                [1, 1, 259, 1],
                "slice_token_axis_to_logical_offset",
                logical_offset=3,
                call_token_counts=[3],
            )

        with self.assertRaisesRegex(qwen38_contract.ContractError, "positive integer"):
            qwen38_contract.mlx_kv_capacity_after_calls([3, 0])

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
