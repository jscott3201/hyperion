#!/usr/bin/env python3
"""Fail-closed validation for the unexecuted Qwen3.8 oracle contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


CONTRACT_SCHEMA = "hyperion.qwen38-oracle-contract.v2"
PREVIOUS_CONTRACT_SCHEMA = "hyperion.qwen38-oracle-contract.v1"
PREVIOUS_CONTRACT_SHA256 = "d837027aadc58c530f996841c98172f5a6b7307bf5f9fb22b46d333375bd2a4d"
SOURCE_SCHEMA = "hyperion.qwen38-source-manifest.v1"
TRACE_SCHEMA = "hyperion.qwen38-trace-schema.v1"
PRODUCER_SCHEMA = "hyperion.qwen38-producer-contracts.v1"
VALIDATION_SCHEMA = "hyperion.qwen38-oracle-contract-validation.v2"
SOURCE_REPOSITORY = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_MANIFEST_SHA256 = "450ff5ada441702f54e8da2cd3c92a207ab1b3bdbaa6254427556d9afa4ff91a"
CASES_SHA256 = "2a1c23a1baa9ea8566fce71d452272202eddf71e83190c930389fce94097fda3"
TRACE_SCHEMA_SHA256 = "41afbb8fbe9c15f4b08f1e3196ea2c8920be166e7666bf639a28e83f100980f8"
TRACE_SCHEMA_CANONICAL_SHA256 = "17c3fbbca3272443ac15e979445438e2d0e10b6e52d9a48c2d8d211f3d05e699"
PRODUCER_CONTRACTS_SHA256 = "6b828775a275ee5f8829b314120591be482e19d9750db3f10890b17c5d5dca60"
PRODUCER_CONTRACTS_CANONICAL_SHA256 = (
    "3d55072241cfbe589662a3ece5786f9fe5e7de909e46b1f57f85f90ca93e09bc"
)
MLX_ORACLE_LOCK_SHA256 = "b59b9022be34f429962150b4aa6a8339b7d4250c6ec814294b5d0d47663c540e"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
CASE_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
WEIGHT_SHARD_RE = re.compile(r"model-([0-9]{5})-of-00018\.safetensors")
PARSER_TOOL_BLOCK_RE = re.compile(
    r"<tool_call>\n<function=([A-Za-z_][A-Za-z0-9_.-]*)>\n(.*?)\n</function>\n</tool_call>",
    re.DOTALL,
)

STATUS = {
    "oracle_reference_only": True,
    "trace_schema_frozen": True,
    "producer_semantics_frozen": True,
    "producer_environments_frozen": False,
    "weights_acquired": False,
    "local_model_tree_verified": False,
    "oracle_executed": False,
    "real_weight_traces_produced": False,
    "tolerances_frozen": False,
    "reference_agreement_measured": False,
    "native_executable": False,
    "artifact_accepted": False,
    "support_accepted": False,
}

SPECIAL_TOKENS = {
    "endoftext": 248044,
    "im_start": 248045,
    "im_end": 248046,
    "tool_call_start": 248058,
    "tool_call_end": 248059,
    "tool_response_start": 248066,
    "tool_response_end": 248067,
    "think_start": 248068,
    "think_end": 248069,
}

REQUIRED_SOURCE_FILES = {
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}

EXPECTED_SOURCE_PATHS = REQUIRED_SOURCE_FILES | {
    ".gitattributes",
    "crc32.txt",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    *(f"model-{index:05d}-of-00018.safetensors" for index in range(1, 19)),
}

REQUIRED_COVERAGE = [
    "canonical_dual_stop",
    "consecutive_tool_results",
    "invalid_reasoning_effort_rejection",
    "multi_turn",
    "parallel_tool_calls",
    "parser_every_byte_split",
    "parser_malformed_fallback",
    "parser_parallel_calls",
    "parser_utf8_split",
    "parser_valid_call",
    "preserve_thinking_false",
    "preserve_thinking_true",
    "stop_endoftext",
    "stop_im_end",
    "system_first_rejection",
    "system_first_valid",
    "system_with_tools",
    "thinking_default",
    "thinking_disabled",
    "thinking_low",
    "thinking_medium",
    "thinking_xhigh",
    "tool_declaration",
]

REQUIRED_TRACE_CHANNELS = [
    "rendered_utf8",
    "render_rejection",
    "parser_outcome",
    "input_ids",
    "position_ids",
    "full_vocab_logits",
    "topk",
    "greedy_token_ids",
    "free_running_utf8",
    "selected_hidden_states",
    "native_gdn_recurrent_state",
    "native_gdn_conv_state",
    "native_full_attention_kv",
]

REQUIRED_TRACE_RUNS = [
    "behavior_monolithic",
    "trace_monolithic",
    "trace_token_serial_prefill",
    "trace_conv_boundary_prefill",
]

BEHAVIOR_CASE_IDS = [
    "generate-canonical-dual-stop",
    "generate-stop-endoftext",
    "generate-stop-im-end",
    "render-system-first-tools",
    "render-thinking-default-xhigh",
    "render-thinking-low",
    "render-thinking-medium",
    "render-tool-results-preserve-thinking-false",
    "render-tool-results-preserve-thinking-true",
]

GDN_STATE_LAYERS = [0, 2, 30, 62]
FULL_ATTENTION_KV_LAYERS = [3, 31, 63]
HIDDEN_SITES = [
    "embedding_output",
    "decoder_output_0",
    "decoder_output_2",
    "decoder_output_3",
    "decoder_output_30",
    "decoder_output_31",
    "decoder_output_32",
    "decoder_output_62",
    "decoder_output_63",
    "final_norm_output",
]

REQUIRED_EVIDENCE_IDENTITIES = [
    "source_manifest",
    "source_tree",
    "oracle_implementation",
    "oracle_environment",
    "environment_tree",
    "python_runtime",
    "device_backend",
    "machine_profile",
    "producer_contracts",
    "execution_mode",
    "producer_command",
    "producer_executable",
    "process_inventory",
    "case_corpus",
    "conversation_profile",
    "trace_schema",
    "parameter_inventory",
    "cache_topology",
    "kernel_route_inventory",
    "raw_payload_inventory",
    "comparison_harness",
    "detached_seal",
]

EXPECTED_ORACLES = [
    {
        "id": "transformers",
        "repository": "huggingface/transformers",
        "revision": "95940bf8775059a42f047256f076e4f607bc43ec",
        "implementation_path": "src/transformers/models/qwen3_5/modeling_qwen3_5.py",
        "implementation_sha256": "90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a",
        "state": "source_pinned_unexecuted",
        "execution_mode_status": "semantic_mode_frozen_environment_unbuilt",
        "execution_intent": {
            "model_class": "Qwen3_5ForConditionalGeneration",
            "component": "text_inputs_only",
            "weight_dtype": "bfloat16",
            "attention": "eager",
            "hub_kernels": False,
            "trust_remote_code": False,
            "local_files_only": True,
            "generation": "fixed_step_greedy_direct_forward",
            "max_steps": 16,
        },
    },
    {
        "id": "mlx_lm",
        "repository": "ml-explore/mlx-lm",
        "revision": "8239c72de5a0e42c539e30489021db73c7fe258c",
        "implementation_path": "mlx_lm/models/qwen3_5.py",
        "implementation_sha256": "cdcfbf22681d2005f4bdff53ab9ab06da7aeb71c0e89378f65f893beeaf0b47c",
        "state": "source_pinned_unexecuted",
        "execution_mode_status": "semantic_mode_frozen_environment_unbuilt",
        "execution_intent": {
            "model_class": "mlx_lm.models.qwen3_5.Model",
            "component": "text_only_sanitized",
            "weight_dtype": "bfloat16",
            "mlx": "0.32.0",
            "model_training": False,
            "generation": "fixed_step_greedy_direct_forward",
            "max_steps": 16,
        },
    },
]


class ContractError(RuntimeError):
    """The committed oracle contract or a candidate source tree is invalid."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ContractError(f"JSON contains a non-finite number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"JSON contains a duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_surrogates(value: Any, context: str) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ContractError(f"{context} contains a Unicode surrogate")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_surrogates(item, f"{context}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key, f"{context} key")
            _reject_surrogates(item, f"{context}.{key}")


def load_json_no_duplicates(path: Path) -> Any:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ContractError(f"JSON must not contain a UTF-8 BOM: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"JSON is not valid UTF-8: {path}") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ContractError(f"malformed JSON in {path}: {error}") from error
    _reject_surrogates(value, str(path))
    return value


def _exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{context} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{context} must be a non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{context} must be an integer >= {minimum}")
    return value


def _string_list(value: Any, context: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ContractError(f"{context} must be a {'non-empty ' if nonempty else ''}list")
    if not all(isinstance(item, str) and item for item in value):
        raise ContractError(f"{context} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise ContractError(f"{context} contains duplicates")
    return value


def _sha256(value: Any, context: str) -> str:
    value = _string(value, context)
    if SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{context} must be a lowercase SHA-256")
    return value


def _commit(value: Any, context: str) -> str:
    value = _string(value, context)
    if COMMIT_RE.fullmatch(value) is None:
        raise ContractError(f"{context} must be a full lowercase Git commit")
    return value


def _safe_relative_path(value: Any, context: str) -> PurePosixPath:
    value = _string(value, context)
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in value
        or relative.as_posix() != value
    ):
        raise ContractError(f"{context} is not a canonical relative path: {value!r}")
    return relative


def _regular_repo_file(repo_root: Path, relative_value: Any, context: str) -> Path:
    relative = _safe_relative_path(relative_value, context)
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise ContractError("repository root must be a real directory")
    current = repo_root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ContractError(f"{context} is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ContractError(f"{context} crosses a symlink: {relative}")
        if index + 1 == len(relative.parts):
            if not stat.S_ISREG(mode):
                raise ContractError(f"{context} must be a regular file: {relative}")
        elif not stat.S_ISDIR(mode):
            raise ContractError(f"{context} parent is not a directory: {relative}")
    return current


def validate_source_manifest_value(manifest: Any) -> dict[str, Any]:
    manifest = _exact_keys(
        manifest,
        {
            "schema",
            "repository",
            "revision",
            "license",
            "observed_at_utc",
            "inventory_source",
            "payload_file_count",
            "payload_bytes",
            "files",
        },
        "source manifest",
    )
    if manifest["schema"] != SOURCE_SCHEMA:
        raise ContractError("source manifest schema differs")
    if manifest["repository"] != SOURCE_REPOSITORY:
        raise ContractError("source repository differs")
    if manifest["revision"] != SOURCE_REVISION:
        raise ContractError("source revision differs")
    if manifest["license"] != "Apache-2.0":
        raise ContractError("source license differs")
    if re.fullmatch(r"2026-08-14T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", manifest["observed_at_utc"]) is None:
        raise ContractError("source observation timestamp is not the reviewed UTC date")
    expected_inventory_url = (
        "https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/"
        f"{SOURCE_REVISION}?blobs=true"
    )
    if manifest["inventory_source"] != expected_inventory_url:
        raise ContractError("source inventory URL is not revision-pinned")

    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ContractError("source manifest files must be a non-empty list")
    parsed: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    weight_shards: set[int] = set()
    for index, item in enumerate(files):
        item = _exact_keys(item, {"path", "size", "sha256", "storage"}, f"source file {index}")
        path = _safe_relative_path(item["path"], f"source file {index} path").as_posix()
        if path in parsed:
            raise ContractError(f"source manifest repeats path: {path}")
        size = _integer(item["size"], f"source file {path} size", minimum=1)
        _sha256(item["sha256"], f"source file {path} sha256")
        if item["storage"] not in {"git", "lfs"}:
            raise ContractError(f"source file {path} has an unknown storage class")
        shard_match = WEIGHT_SHARD_RE.fullmatch(path)
        if shard_match is not None:
            if item["storage"] != "lfs":
                raise ContractError(f"weight shard is not LFS-backed: {path}")
            weight_shards.add(int(shard_match.group(1)))
        parsed[path] = item
        total_bytes += size

    if list(parsed) != sorted(parsed):
        raise ContractError("source manifest paths must be sorted")
    if set(range(1, 19)) != weight_shards:
        raise ContractError("source manifest does not contain the exact 18-shard sequence")
    if set(parsed) != EXPECTED_SOURCE_PATHS:
        raise ContractError(
            "source manifest path set differs: "
            f"missing={sorted(EXPECTED_SOURCE_PATHS - set(parsed))}, "
            f"extra={sorted(set(parsed) - EXPECTED_SOURCE_PATHS)}"
        )
    if parsed["tokenizer.json"]["storage"] != "lfs":
        raise ContractError("tokenizer.json storage class differs")
    if manifest["payload_file_count"] != len(files):
        raise ContractError("source payload_file_count differs from the inventory")
    if manifest["payload_bytes"] != total_bytes:
        raise ContractError("source payload_bytes differs from the inventory")
    return parsed


def _validate_expected_tool_calls(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError(f"{context} must be a list")
    result = []
    for index, call in enumerate(value):
        call = _exact_keys(call, {"arguments", "name"}, f"{context}[{index}]")
        _string(call["name"], f"{context}[{index}].name")
        if not isinstance(call["arguments"], dict):
            raise ContractError(f"{context}[{index}].arguments must be an object")
        result.append(call)
    return result


def _validate_message_tool_calls(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError(f"{context} must be a list")
    for index, call in enumerate(value):
        call = _exact_keys(call, {"function", "type"}, f"{context}[{index}]")
        if call["type"] != "function":
            raise ContractError(f"{context}[{index}].type must be function")
        function = _exact_keys(
            call["function"],
            {"arguments", "name"},
            f"{context}[{index}].function",
        )
        _string(function["name"], f"{context}[{index}].function.name")
        if not isinstance(function["arguments"], dict):
            raise ContractError(f"{context}[{index}].function.arguments must be an object")
    return value


def _validate_messages(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError(f"{context} must be a list")
    for index, message in enumerate(value):
        if not isinstance(message, dict) or "role" not in message:
            raise ContractError(f"{context}[{index}] must be a role-bearing object")
        role = message["role"]
        if role == "assistant":
            message = _exact_keys(
                message,
                {"content", "reasoning_content", "role", "tool_calls"},
                f"{context}[{index}]",
            )
            if not isinstance(message["content"], str):
                raise ContractError(f"{context}[{index}].content must be a string")
            if not isinstance(message["reasoning_content"], str):
                raise ContractError(f"{context}[{index}].reasoning_content must be a string")
            _validate_message_tool_calls(message["tool_calls"], f"{context}[{index}].tool_calls")
        elif role in {"system", "user", "tool"}:
            message = _exact_keys(message, {"content", "role"}, f"{context}[{index}]")
            _string(message["content"], f"{context}[{index}].content")
        else:
            raise ContractError(f"{context}[{index}] has an unsupported role: {role!r}")
    return value


def _validate_tools(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError(f"{context} must be a list")
    for index, tool in enumerate(value):
        tool = _exact_keys(tool, {"function", "type"}, f"{context}[{index}]")
        if tool["type"] != "function":
            raise ContractError(f"{context}[{index}].type must be function")
        function = _exact_keys(
            tool["function"],
            {"description", "name", "parameters"},
            f"{context}[{index}].function",
        )
        _string(function["description"], f"{context}[{index}].function.description")
        _string(function["name"], f"{context}[{index}].function.name")
        if not isinstance(function["parameters"], dict):
            raise ContractError(f"{context}[{index}].function.parameters must be an object")
    return value


def _validate_template_args(value: Any, context: str, *, rejection_case: bool) -> None:
    if not isinstance(value, dict):
        raise ContractError(f"{context} must be an object")
    allowed = {"add_generation_prompt", "enable_thinking", "reasoning_effort", "preserve_thinking"}
    if not set(value).issubset(allowed):
        raise ContractError(f"{context} contains an unknown template argument")
    for boolean_key in {"add_generation_prompt", "enable_thinking", "preserve_thinking"}:
        if boolean_key in value and type(value[boolean_key]) is not bool:
            raise ContractError(f"{context}.{boolean_key} must be boolean")
    if "reasoning_effort" in value:
        valid = {"xhigh", "medium", "low"}
        if rejection_case:
            if value["reasoning_effort"] in valid:
                raise ContractError(f"{context} rejection case uses a valid reasoning effort")
        elif value["reasoning_effort"] not in valid:
            raise ContractError(f"{context} has an unsupported reasoning effort")


def _supported_coverage(case: dict[str, Any]) -> set[str]:
    """Derive coverage that the case structure can actually exercise."""

    supported: set[str] = set()
    kind = case["kind"]
    messages = case["messages"]
    template_args = case["template_args"]

    if kind in {"render", "render_reject", "generation"}:
        enable_thinking = template_args.get("enable_thinking", True)
        effort = template_args.get("reasoning_effort", "xhigh")
        if enable_thinking:
            if "enable_thinking" not in template_args and "reasoning_effort" not in template_args:
                supported.add("thinking_default")
            supported.add(f"thinking_{effort}")
        else:
            supported.add("thinking_disabled")

        roles = [message["role"] for message in messages]
        if len(roles) >= 3 and "assistant" in roles[:-1] and roles[-1] == "user":
            supported.add("multi_turn")
        last_user_index = max(
            (index for index, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )
        has_prior_reasoning = any(
            index < last_user_index
            and message["role"] == "assistant"
            and bool(message["reasoning_content"].strip())
            for index, message in enumerate(messages)
        )
        if template_args.get("preserve_thinking") is True and has_prior_reasoning:
            supported.add("preserve_thinking_true")
        if template_args.get("preserve_thinking") is False and has_prior_reasoning:
            supported.add("preserve_thinking_false")
        if case["tools"]:
            supported.add("tool_declaration")
        if any(
            message["role"] == "assistant" and len(message["tool_calls"]) >= 2
            for message in messages
        ):
            supported.add("parallel_tool_calls")
        if any(left == right == "tool" for left, right in zip(roles, roles[1:])):
            supported.add("consecutive_tool_results")
        if kind == "render" and roles and roles[0] == "system":
            supported.add("system_first_valid")
            if case["tools"]:
                supported.add("system_with_tools")
        if kind == "render_reject" and "system" in roles[1:]:
            supported.add("system_first_rejection")
        if (
            kind == "render_reject"
            and "reasoning_effort" in template_args
            and template_args["reasoning_effort"] not in {"xhigh", "medium", "low"}
        ):
            supported.add("invalid_reasoning_effort_rejection")
        if kind == "generation" and case["stop_token_ids"] == [248046, 248044]:
            supported.add("canonical_dual_stop")
        if kind == "generation" and case["stop_token_ids"] == [248046]:
            supported.add("stop_im_end")
        if kind == "generation" and case["stop_token_ids"] == [248044]:
            supported.add("stop_endoftext")

    if kind == "parser":
        call_count = len(case["expected_tool_calls"])
        blocks = list(PARSER_TOOL_BLOCK_RE.finditer(case["parser_input"]))
        parsed_names = [match.group(1) for match in blocks]
        expected_names = [call["name"] for call in case["expected_tool_calls"]]
        parameter_markers_match = all(
            all(f"<parameter={key}>\n" in match.group(2) for key in call["arguments"])
            for match, call in zip(blocks, case["expected_tool_calls"])
        )
        syntax_matches = (
            len(blocks) == call_count
            and parsed_names == expected_names
            and parameter_markers_match
        )
        if case["expected_outcome"] == "parse_calls" and call_count == 1 and syntax_matches:
            supported.add("parser_valid_call")
        if case["expected_outcome"] == "parse_calls" and call_count >= 2 and syntax_matches:
            supported.add("parser_parallel_calls")
        if case["split_policy"] == "every_utf8_byte":
            supported.add("parser_every_byte_split")
            if any(ord(character) > 127 for character in case["parser_input"]):
                supported.add("parser_utf8_split")
        if case["expected_outcome"] == "fallback_text":
            supported.add("parser_malformed_fallback")

    return supported


def _validate_case(case: Any, line_number: int) -> dict[str, Any]:
    context = f"case line {line_number}"
    case = _exact_keys(
        case,
        {
            "coverage",
            "expected_outcome",
            "expected_text",
            "expected_tool_calls",
            "id",
            "kind",
            "max_steps",
            "messages",
            "parser_input",
            "split_policy",
            "stop_token_ids",
            "template_args",
            "tools",
        },
        context,
    )
    case_id = _string(case["id"], f"{context}.id")
    if CASE_ID_RE.fullmatch(case_id) is None or len(case_id) > 80:
        raise ContractError(f"{context}.id is not canonical")
    coverage = _string_list(case["coverage"], f"{context}.coverage")
    if coverage != sorted(coverage):
        raise ContractError(f"{context}.coverage must be sorted")
    kind = case["kind"]
    if kind not in {"render", "render_reject", "generation", "parser"}:
        raise ContractError(f"{context}.kind is unsupported")
    _validate_messages(case["messages"], f"{context}.messages")
    _validate_tools(case["tools"], f"{context}.tools")
    expected_calls = _validate_expected_tool_calls(
        case["expected_tool_calls"],
        f"{context}.expected_tool_calls",
    )
    message_calls = [
        call["function"]
        for message in case["messages"]
        if message["role"] == "assistant"
        for call in message["tool_calls"]
    ]
    declared_tool_names = {tool["function"]["name"] for tool in case["tools"]}
    if any(call["name"] not in declared_tool_names for call in message_calls):
        raise ContractError(f"{context} calls a function absent from its tool declarations")
    rejection_case = kind == "render_reject"
    _validate_template_args(case["template_args"], f"{context}.template_args", rejection_case=rejection_case)

    stop_ids = case["stop_token_ids"]
    if not isinstance(stop_ids, list) or any(type(item) is not int for item in stop_ids):
        raise ContractError(f"{context}.stop_token_ids must be an integer list")
    if len(stop_ids) != len(set(stop_ids)) or not set(stop_ids).issubset({248044, 248046}):
        raise ContractError(f"{context}.stop_token_ids are invalid")
    if case["split_policy"] not in {"none", "every_utf8_byte"}:
        raise ContractError(f"{context}.split_policy is unsupported")

    if kind == "parser":
        if case["messages"] or case["tools"] or case["template_args"] or stop_ids:
            raise ContractError(f"{context} parser case contains renderer inputs")
        _string(case["parser_input"], f"{context}.parser_input")
        if case["expected_outcome"] not in {"parse_calls", "fallback_text"}:
            raise ContractError(f"{context} parser outcome is invalid")
        if case["expected_outcome"] == "parse_calls" and not expected_calls:
            raise ContractError(f"{context} parse_calls case has no expected calls")
        if case["expected_outcome"] == "fallback_text" and expected_calls:
            raise ContractError(f"{context} fallback case unexpectedly declares calls")
        if not isinstance(case["expected_text"], str):
            raise ContractError(f"{context}.expected_text must be a string")
        if case["expected_outcome"] == "fallback_text":
            if case["expected_text"] != case["parser_input"]:
                raise ContractError(f"{context} fallback text must preserve the exact input")
        else:
            blocks = list(PARSER_TOOL_BLOCK_RE.finditer(case["parser_input"]))
            cursor = 0
            surrounding = []
            for match in blocks:
                surrounding.append(case["parser_input"][cursor : match.start()])
                cursor = match.end()
            surrounding.append(case["parser_input"][cursor:])
            if case["expected_text"] != "".join(surrounding):
                raise ContractError(f"{context} expected surrounding text differs")
    else:
        if case["expected_text"] is not None:
            raise ContractError(f"{context}.expected_text is only valid for parser cases")
        if message_calls != expected_calls:
            raise ContractError(f"{context}.expected_tool_calls differ from transcript inputs")
        if case["parser_input"] is not None or case["split_policy"] != "none":
            raise ContractError(f"{context} renderer case contains parser input")
        if not case["messages"]:
            raise ContractError(f"{context} renderer case has no messages")
        expected = "reject" if rejection_case else "capture"
        if case["expected_outcome"] != expected:
            raise ContractError(f"{context} outcome differs for {kind}")
        if kind == "generation" and stop_ids not in ([248046, 248044], [248046], [248044]):
            raise ContractError(f"{context} generation case has a noncanonical stop policy")
        if kind != "generation" and stop_ids:
            raise ContractError(f"{context} non-generation case declares stop IDs")
    if kind == "generation":
        if case["max_steps"] != 16:
            raise ContractError(f"{context}.max_steps must be the frozen 16-step bound")
    elif case["max_steps"] is not None:
        raise ContractError(f"{context}.max_steps is only valid for generation cases")
    unsupported_coverage = set(coverage) - _supported_coverage(case)
    if unsupported_coverage:
        raise ContractError(
            f"{context}.coverage is not supported by the case structure: "
            f"{sorted(unsupported_coverage)}"
        )
    return case


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ContractError("cases JSONL must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("cases JSONL is not valid UTF-8") from error
    if not text.endswith("\n"):
        raise ContractError("cases JSONL must end with a newline")
    cases = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ContractError(f"cases JSONL contains a blank line at {line_number}")
        try:
            case = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as error:
            raise ContractError(f"malformed case JSON at line {line_number}: {error}") from error
        _reject_surrogates(case, f"case line {line_number}")
        if canonical_json(case).decode("utf-8").removesuffix("\n") != line:
            raise ContractError(f"case line {line_number} is not canonical JSON")
        case = _validate_case(case, line_number)
        if case["id"] in seen:
            raise ContractError(f"cases JSONL repeats ID: {case['id']}")
        seen.add(case["id"])
        cases.append(case)
    if not cases:
        raise ContractError("cases JSONL is empty")
    return cases


def _require_exact_value(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ContractError(f"{context} differs from the reviewed contract")


def validate_trace_schema_value(schema: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    schema = _exact_keys(
        schema,
        {
            "schema",
            "state",
            "binds",
            "scope",
            "case_plan",
            "runs",
            "prefill_schedules",
            "prediction_frame",
            "frame_kinds",
            "control_pair",
            "selections",
            "channels",
            "native_layouts",
            "canonical_normalization",
            "topk",
            "serialization",
            "record_schemas",
            "validation_edges",
            "cross_oracle_comparison",
            "bundle",
            "comparison_metrics",
            "tolerances",
        },
        "trace schema",
    )
    if schema["schema"] != TRACE_SCHEMA or schema["state"] != "schema_frozen_no_payload":
        raise ContractError("trace schema identity or state differs")
    _require_exact_value(
        schema["binds"],
        {
            "case_corpus_sha256": CASES_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_revision": SOURCE_REVISION,
        },
        "trace schema bindings",
    )
    _require_exact_value(
        schema["scope"],
        {
            "batch_size": 1,
            "padding": False,
            "prompt_tokens_min": 6,
            "prompt_tokens_max": 512,
            "max_generated_tokens": 16,
            "weight_dtype": "bfloat16",
            "gdn_recurrent_dtype": "float32",
            "quantization": False,
            "vision": False,
            "mtp": False,
            "adapters": False,
        },
        "trace scope",
    )

    case_by_id = {case["id"]: case for case in cases}
    plan = _exact_keys(
        schema["case_plan"],
        {
            "renderer_case_ids",
            "parser_case_ids",
            "behavior_cases",
            "render_rejection_codes",
            "stop_coverage_semantics",
            "state_probe_case_id",
            "free_running_gap_closed_by_promoted_render_cases",
        },
        "trace case plan",
    )
    renderer_ids = _string_list(plan["renderer_case_ids"], "trace renderer cases")
    parser_ids = _string_list(plan["parser_case_ids"], "trace parser cases")
    if renderer_ids != sorted(case["id"] for case in cases if case["kind"] != "parser"):
        raise ContractError("trace renderer cases do not cover the exact non-parser corpus")
    if parser_ids != sorted(case["id"] for case in cases if case["kind"] == "parser"):
        raise ContractError("trace parser cases do not cover the exact parser corpus")
    behavior_cases = plan["behavior_cases"]
    if not isinstance(behavior_cases, list) or len(behavior_cases) != len(BEHAVIOR_CASE_IDS):
        raise ContractError("trace behavior case count differs")
    behavior_ids = []
    for index, behavior in enumerate(behavior_cases):
        behavior = _exact_keys(
            behavior,
            {"case_id", "max_steps", "stop_token_ids"},
            f"trace behavior case {index}",
        )
        case_id = _string(behavior["case_id"], f"trace behavior case {index} ID")
        if case_id not in case_by_id:
            raise ContractError(f"trace behavior case is absent from the corpus: {case_id}")
        case = case_by_id[case_id]
        if case["kind"] not in {"render", "generation"}:
            raise ContractError(f"trace behavior case cannot be executed: {case_id}")
        if case["template_args"].get("add_generation_prompt") is not True:
            raise ContractError(f"trace behavior case lacks a generation prompt: {case_id}")
        if behavior["max_steps"] != 16:
            raise ContractError(f"trace behavior case changes the generation bound: {case_id}")
        if behavior["stop_token_ids"] not in ([248046, 248044], [248046], [248044]):
            raise ContractError(f"trace behavior case has an invalid stop policy: {case_id}")
        if case["kind"] == "generation" and (
            behavior["max_steps"] != case["max_steps"]
            or behavior["stop_token_ids"] != case["stop_token_ids"]
        ):
            raise ContractError(f"trace behavior case changes a committed generation input: {case_id}")
        behavior_ids.append(case_id)
    if behavior_ids != BEHAVIOR_CASE_IDS:
        raise ContractError("trace behavior cases or ordering differ")
    _require_exact_value(
        plan["render_rejection_codes"],
        {
            "reject-invalid-reasoning-effort": "invalid_reasoning_effort",
            "reject-system-not-first": "system_message_must_be_first",
        },
        "trace render rejection codes",
    )
    _require_exact_value(
        plan["stop_coverage_semantics"],
        {
            "canonical_dual_stop": (
                "configured input and stop-policy coverage only; observed termination is separately "
                "receipted and this label does not claim either token was emitted"
            ),
            "stop_endoftext": (
                "configured input and stop-policy coverage only; observed termination is separately "
                "receipted and this label does not claim token 248044 was emitted"
            ),
            "stop_im_end": (
                "configured input and stop-policy coverage only; observed termination is separately "
                "receipted and this label does not claim token 248046 was emitted"
            ),
        },
        "trace stop coverage semantics",
    )
    required_behavior_coverage = {
        "thinking_default",
        "thinking_low",
        "thinking_medium",
        "system_with_tools",
        "preserve_thinking_false",
        "preserve_thinking_true",
    }
    behavior_coverage = {
        tag for case_id in behavior_ids for tag in case_by_id[case_id]["coverage"]
    }
    if not required_behavior_coverage.issubset(behavior_coverage):
        raise ContractError("trace behavior plan omits thinking, tool, or preserved-history generation")
    if plan["state_probe_case_id"] != "generate-canonical-dual-stop":
        raise ContractError("trace state-probe case differs")
    if plan["free_running_gap_closed_by_promoted_render_cases"] is not True:
        raise ContractError("trace schema does not close free-running behavior coverage")

    runs = schema["runs"]
    if not isinstance(runs, list) or len(runs) != 4:
        raise ContractError("trace run matrix differs")
    validated_runs = []
    for index, run in enumerate(runs):
        run = _exact_keys(
            run,
            {
                "id",
                "cases",
                "fresh_process",
                "fresh_cache_per_case",
                "instrumented",
                "prefill_schedule",
                "decode",
                "state_snapshots",
            },
            f"trace run {index}",
        )
        if run["fresh_process"] is not True or run["fresh_cache_per_case"] is not True:
            raise ContractError(f"trace run can reuse mutable state: {run['id']}")
        if not isinstance(run["state_snapshots"], list):
            raise ContractError(f"trace run snapshots must be a list: {run['id']}")
        validated_runs.append(run)
    runs = validated_runs
    run_ids = [_string(run["id"], f"trace run {index} ID") for index, run in enumerate(runs)]
    if run_ids != REQUIRED_TRACE_RUNS:
        raise ContractError("trace run IDs or ordering differ")
    _require_exact_value(
        [run["prefill_schedule"] for run in runs],
        ["whole_prompt", "whole_prompt", "one_token_calls", "chunks_3_1_1_remainder"],
        "trace prefill run matrix",
    )
    _require_exact_value(
        schema["prefill_schedules"],
        {
            "whole_prompt": "one call consuming positions [0,T)",
            "one_token_calls": "T calls consuming [i,i+1) for i=0..T-1 without sampling between calls",
            "chunks_3_1_1_remainder": "calls consuming [0,3), [3,4), [4,5), and [5,T), without sampling between calls",
        },
        "trace prefill schedules",
    )
    prediction = _exact_keys(
        schema["prediction_frame"],
        {
            "prompt_call",
            "decode_call",
            "cache_length_after_call",
            "stop_policy",
            "terminal_cache",
            "high_level_generate_api",
        },
        "prediction frame",
    )
    if prediction["high_level_generate_api"] is not False:
        raise ContractError("trace schema permits a high-level generation API")
    if prediction["cache_length_after_call"] != (
        "T+i for call i, before the token predicted by that call enters the cache"
    ):
        raise ContractError("prediction-frame cache semantics differ")
    if prediction["stop_policy"] != (
        "include a predicted stop token in emitted IDs and UTF-8, then perform no later forward call"
    ):
        raise ContractError("prediction-frame stop semantics differ")
    _require_exact_value(
        schema["frame_kinds"],
        {
            "generation_prediction": {
                "sampled": True,
                "call_span": "prompt [0,T) for call 0; one emitted token for each later call",
                "prediction_index": "zero-based emitted-token index",
                "boundary_tags": ["post_prefill", "post_decode", "terminal"],
                "termination": "none_until_terminal_then_stop_token_or_max_steps",
            },
            "teacher_forced_prefill": {
                "sampled": False,
                "call_span": "explicit half-open prompt-token range [start,end)",
                "prediction_index": None,
                "boundary_tags": [
                    "logical_prefix_3",
                    "logical_prefix_4",
                    "logical_prefix_5",
                    "post_prefill",
                ],
                "termination": "not_applicable",
            },
        },
        "trace frame kinds",
    )
    _require_exact_value(
        schema["control_pair"],
        {
            "uninstrumented_run": "behavior_monolithic",
            "instrumented_run": "trace_monolithic",
            "case_id": "generate-canonical-dual-stop",
            "fresh_state_for_each": True,
            "comparison_authority": "third_party_verifier_after_measured_tolerances",
        },
        "instrumentation control pair",
    )

    selections = _exact_keys(
        schema["selections"],
        {
            "hidden_sites",
            "prompt_hidden_positions",
            "decode_hidden_positions",
            "hidden_site_boundaries",
            "gdn_state_layers",
            "full_attention_kv_layers",
            "gdn_layer_rule",
            "full_attention_layer_rule",
        },
        "trace selections",
    )
    if selections["hidden_sites"] != HIDDEN_SITES:
        raise ContractError("trace hidden-state sites differ")
    if selections["gdn_state_layers"] != GDN_STATE_LAYERS:
        raise ContractError("trace GDN state layers differ")
    if selections["full_attention_kv_layers"] != FULL_ATTENTION_KV_LAYERS:
        raise ContractError("trace full-attention state layers differ")
    if any(layer % 4 == 3 for layer in selections["gdn_state_layers"]):
        raise ContractError("trace selects a full-attention layer as GDN")
    if any(layer % 4 != 3 for layer in selections["full_attention_kv_layers"]):
        raise ContractError("trace selects a GDN layer as full attention")
    if selections["prompt_hidden_positions"] != (
        "ascending(sorted_unique(0,2,3,4,T-1) intersect [0,logical_prefix)) at each "
        "snapshot; descriptor stores the exact concrete IDs"
    ):
        raise ContractError("trace hidden positions differ")
    _require_exact_value(
        selections["hidden_site_boundaries"],
        {
            "embedding_output": (
                "token embedding lookup output before decoder layer 0 and before any "
                "decoder-layer normalization"
            ),
            "decoder_output_<i>": (
                "post-MLP-residual output returned by the complete decoder block i, after both "
                "attention-or-GDN and MLP residual updates and before layer i+1"
            ),
            "final_norm_output": (
                "output of the final text-model RMSNorm applied after decoder layer 63 and "
                "before lm_head"
            ),
        },
        "trace hidden-site boundaries",
    )

    channels = schema["channels"]
    if not isinstance(channels, list):
        raise ContractError("trace channels must be a list")
    channel_ids = []
    for index, channel in enumerate(channels):
        if not isinstance(channel, dict):
            raise ContractError(f"trace channel {index} must be an object")
        channel_ids.append(_string(channel.get("id"), f"trace channel {index} ID"))
    if channel_ids != REQUIRED_TRACE_CHANNELS:
        raise ContractError("trace channels or ordering differ")
    if len(set(channel_ids)) != len(channel_ids):
        raise ContractError("trace channels contain duplicates")
    channel_fields = {
        "rendered_utf8": {"id", "applies_to", "encoding"},
        "render_rejection": {"id", "applies_to", "encoding"},
        "parser_outcome": {"id", "applies_to", "encoding"},
        "input_ids": {"id", "applies_to", "stored_dtype", "axes", "shape"},
        "position_ids": {"id", "applies_to", "stored_dtype", "axes", "shape"},
        "full_vocab_logits": {
            "id",
            "applies_to",
            "stored_dtype",
            "axes",
            "shape",
            "conversion",
        },
        "topk": {"id", "applies_to", "encoding"},
        "greedy_token_ids": {"id", "applies_to", "stored_dtype", "axes", "shape"},
        "free_running_utf8": {"id", "applies_to", "encoding"},
        "selected_hidden_states": {
            "id",
            "applies_to",
            "stored_dtype",
            "native_axes",
            "shape",
            "snapshot_rule",
        },
        "native_gdn_recurrent_state": {
            "id",
            "applies_to",
            "stored_dtype",
            "shape",
            "snapshot_rule",
        },
        "native_gdn_conv_state": {
            "id",
            "applies_to",
            "stored_dtype",
            "shape",
            "snapshot_rule",
        },
        "native_full_attention_kv": {
            "id",
            "applies_to",
            "stored_dtype",
            "shape",
            "snapshot_rule",
        },
    }
    for index, channel in enumerate(channels):
        _exact_keys(channel, channel_fields[channel["id"]], f"trace channel {index}")
    channel_by_id = {channel["id"]: channel for channel in channels}
    if channel_by_id["full_vocab_logits"].get("shape") != [1, 248320]:
        raise ContractError("full-vocabulary trace shape differs")
    if channel_by_id["native_gdn_recurrent_state"].get("shape") != [1, 48, 128, 128]:
        raise ContractError("GDN recurrent trace shape differs")
    for channel_id in (
        "selected_hidden_states",
        "native_gdn_recurrent_state",
        "native_gdn_conv_state",
        "native_full_attention_kv",
    ):
        if channel_by_id[channel_id].get("snapshot_rule") != (
            "materialize_before_later_cache_mutation"
        ):
            raise ContractError(f"trace snapshot can alias mutable cache state: {channel_id}")

    layouts = _exact_keys(schema["native_layouts"], {"transformers", "mlx_lm"}, "native layouts")
    transformers_layout = _exact_keys(
        layouts["transformers"],
        {
            "gdn_recurrent_axes",
            "gdn_recurrent_shape",
            "gdn_conv_axes",
            "gdn_conv_shape",
            "full_attention_kv_axes",
            "full_attention_kv_shape",
        },
        "Transformers native layout",
    )
    mlx_layout = _exact_keys(
        layouts["mlx_lm"],
        {
            "gdn_recurrent_axes",
            "gdn_recurrent_shape",
            "gdn_conv_axes",
            "gdn_conv_shape",
            "full_attention_kv_axes",
            "full_attention_kv_shape",
            "logical_offset_required",
            "capacity_constraint",
            "capacity_recurrence",
        },
        "mlx-lm native layout",
    )
    if transformers_layout["gdn_recurrent_axes"] != [
        "batch",
        "value_head",
        "key_feature",
        "value_feature",
    ]:
        raise ContractError("Transformers recurrent axes differ")
    if mlx_layout["gdn_recurrent_axes"] != [
        "batch",
        "value_head",
        "value_feature",
        "key_feature",
    ]:
        raise ContractError("mlx-lm recurrent axes differ")
    if transformers_layout["gdn_conv_shape"] != [1, 10240, 4]:
        raise ContractError("Transformers convolution-cache shape differs")
    if mlx_layout["gdn_conv_shape"] != [1, 3, 10240]:
        raise ContractError("mlx-lm convolution-cache shape differs")
    if mlx_layout["full_attention_kv_shape"] != [1, 4, "C", 256]:
        raise ContractError("mlx-lm allocated KV shape differs")
    if mlx_layout["logical_offset_required"] is not True:
        raise ContractError("mlx-lm logical KV offset is not required")
    if mlx_layout["capacity_constraint"] != (
        "L<=C<L+256; record actual C because nonaligned chunk growth need not be a multiple of 256"
    ):
        raise ContractError("mlx-lm allocated KV capacity constraint differs")
    _require_exact_value(
        mlx_layout["capacity_recurrence"],
        {
            "initial_state": "L=0,C=0",
            "no_growth": "when L+N<=C, set L=L+N and leave C unchanged",
            "growth": (
                "when L+N>C, first set C=L if prior C>0 and L mod 256 != 0, "
                "then set C=C+ceil(N/256)*256 and L=L+N"
            ),
            "call_token_count": "N is the positive token count in the current forward call",
            "step": 256,
        },
        "mlx-lm allocated KV capacity recurrence",
    )

    normalization = _exact_keys(
        schema["canonical_normalization"],
        {
            "owner",
            "producer_canonical_state_is_not_raw_evidence",
            "gdn_recurrent_axes",
            "gdn_recurrent_transform",
            "gdn_conv_axes",
            "gdn_conv_shape",
            "gdn_conv_transform",
            "full_attention_kv_axes",
            "full_attention_kv_transform",
            "kv_semantics",
        },
        "canonical normalization",
    )
    if normalization["owner"] != "third_party_verifier":
        raise ContractError("a producer owns trace normalization")
    if normalization["producer_canonical_state_is_not_raw_evidence"] is not True:
        raise ContractError("producer-normalized state can replace raw evidence")
    _require_exact_value(
        normalization["gdn_recurrent_transform"],
        {"transformers": "identity", "mlx_lm": "transpose_last_two"},
        "GDN recurrent normalization",
    )
    _require_exact_value(
        normalization["gdn_conv_transform"],
        {
            "transformers": "take_last_3_then_transpose_0_2_1",
            "mlx_lm": "identity",
        },
        "GDN convolution normalization",
    )
    _require_exact_value(
        normalization["full_attention_kv_transform"],
        {"transformers": "identity", "mlx_lm": "slice_token_axis_to_logical_offset"},
        "full-attention KV normalization",
    )

    _require_exact_value(
        schema["topk"],
        {
            "width": 32,
            "internal_rank_count": 33,
            "ranking": "descending_float32_value_then_ascending_token_id",
            "ids": 32,
            "values": 32,
            "adjacent_margins": 31,
            "cutoff_margin": "rank_32_value_minus_rank_33_value",
            "greedy": "rank_1_token_id",
            "verifier_recomputes_from_full_logits": True,
        },
        "trace top-k contract",
    )
    serialization = _exact_keys(
        schema["serialization"],
        {
            "numeric_payload",
            "allowed_numeric_dtypes",
            "json",
            "jsonl",
            "text",
            "compression",
            "archives",
            "executable_payloads",
            "pickle_or_object_arrays",
            "tensor_descriptor_required_fields",
        },
        "trace serialization",
    )
    if serialization.get("numeric_payload") != (
        "one_headerless_c_contiguous_little_endian_tensor_per_regular_file"
    ):
        raise ContractError("numeric trace serialization differs")
    if serialization.get("allowed_numeric_dtypes") != ["f32le", "u32le"]:
        raise ContractError("numeric trace dtypes differ")
    for forbidden_flag in (
        "compression",
        "archives",
        "executable_payloads",
        "pickle_or_object_arrays",
    ):
        if serialization.get(forbidden_flag) is not False:
            raise ContractError(f"trace serialization permits {forbidden_flag}")
    descriptor_fields = _string_list(
        serialization.get("tensor_descriptor_required_fields"),
        "tensor descriptor required fields",
    )
    if descriptor_fields != sorted(descriptor_fields):
        raise ContractError("tensor descriptor fields must be sorted")
    records = _exact_keys(
        schema["record_schemas"],
        {
            "common_constraints",
            "events_jsonl",
            "payload_manifest_jsonl",
            "verifier_parser_outcomes_jsonl",
            "receipt_json",
            "detached_seal_json",
            "required_payload_inventory",
        },
        "trace record schemas",
    )
    common_records = _exact_keys(
        records["common_constraints"],
        {
            "records_are_closed",
            "unknown_fields",
            "duplicate_records",
            "paths",
            "sha256",
            "identifiers",
            "frame_ids",
            "integers",
            "numbers",
        },
        "trace common record constraints",
    )
    if common_records["records_are_closed"] is not True:
        raise ContractError("trace records are not closed")
    if common_records["frame_ids"] != (
        "exactly <run_instance_id>/gen/<prediction_index> or "
        "<run_instance_id>/prefill/<logical_prefix>, with canonical unsigned decimal suffix "
        "and no leading zero except zero"
    ):
        raise ContractError("trace frame-ID grammar differs")
    event_records = _exact_keys(
        records["events_jsonl"],
        {
            "record_types",
            "required_fields_by_type",
            "field_types",
            "digest_preimages",
            "nullable_fields",
            "ordering",
            "parser_records",
            "uniqueness_keys",
            "foreign_key_rules",
            "call_conditionals",
            "snapshot_specification_mapping",
            "cardinality_and_sequence",
        },
        "trace event records",
    )
    if event_records["record_types"] != [
        "case_input",
        "render_rejection",
        "run_start",
        "call",
        "snapshot",
        "run_end",
    ]:
        raise ContractError("trace event record types differ")
    _require_exact_value(
        event_records["digest_preimages"],
        {
            "input_ids_sha256": (
                "SHA-256 of the exact bound input_ids payload bytes serialized as headerless "
                "C-contiguous u32le"
            ),
            "template_arguments_sha256": (
                "SHA-256 of canonical JSON for the exact case template_args object before any "
                "runtime default expansion"
            ),
        },
        "trace event digest preimages",
    )
    _require_exact_value(
        event_records["snapshot_specification_mapping"],
        {
            "post_first_decode_call_if_executed": {
                "boundary_tag": "post_decode",
                "condition": "emit only on the first generation decode call after the prompt call",
            },
            "terminal_call_if_distinct": {
                "boundary_tag": "terminal",
                "condition": (
                    "emit only when the terminal call is distinct from the calls already tagged "
                    "post_prefill or post_decode"
                ),
            },
        },
        "trace snapshot specification mapping",
    )
    payload_records = _exact_keys(
        records["payload_manifest_jsonl"],
        {
            "record_types",
            "tensor_required_fields_source",
            "bytes_required_fields",
            "field_types",
            "tensor_nullable_fields",
            "bytes_nullable_fields",
            "channel_rules",
            "foreign_key_rules",
            "canonical_json_payload_schemas",
            "closed_records",
            "path_uniqueness_key",
            "path_must_be_beneath",
            "shape_rule",
            "tensor_byte_length_rule",
            "payload_byte_length_rule",
            "hash_rule",
            "one_record_per_payload",
            "manifest_covers_every_payload_regular_file_exactly_once",
        },
        "trace payload records",
    )
    for flag in (
        "closed_records",
        "one_record_per_payload",
        "manifest_covers_every_payload_regular_file_exactly_once",
    ):
        if payload_records[flag] is not True:
            raise ContractError(f"trace payload records weaken {flag}")
    canonical_payloads = _exact_keys(
        payload_records["canonical_json_payload_schemas"],
        {"topk", "render_rejection"},
        "trace canonical JSON payload schemas",
    )
    topk_payload = _exact_keys(
        canonical_payloads["topk"],
        {"closed_object", "required_fields", "field_types", "verifier_rule"},
        "trace top-k payload schema",
    )
    if topk_payload["closed_object"] is not True or topk_payload["required_fields"] != [
        "adjacent_margins_f32",
        "cutoff_margin_f32",
        "greedy_token_id",
        "rank_33_token_id",
        "rank_33_value_f32",
        "ranked_token_ids",
        "ranked_values_f32",
    ]:
        raise ContractError("trace top-k payload schema differs")
    rejection_payload = _exact_keys(
        canonical_payloads["render_rejection"],
        {"closed_object", "required_fields", "field_types", "stable_code_mapping"},
        "trace render-rejection payload schema",
    )
    if rejection_payload["closed_object"] is not True or rejection_payload[
        "required_fields"
    ] != ["case_id", "native_exception_text", "stable_rejection_code"]:
        raise ContractError("trace render-rejection payload schema differs")
    _require_exact_value(
        rejection_payload["stable_code_mapping"],
        {
            "reject-invalid-reasoning-effort": "invalid_reasoning_effort",
            "reject-system-not-first": "system_message_must_be_first",
        },
        "trace render-rejection stable-code mapping",
    )
    parser_records = _exact_keys(
        records["verifier_parser_outcomes_jsonl"],
        {
            "owner",
            "location",
            "closed_records",
            "required_fields",
            "record_type",
            "field_types",
            "cardinality",
            "comparison",
            "parser_harness_binding",
            "arm_producers_may_write",
        },
        "trace verifier parser records",
    )
    if parser_records["arm_producers_may_write"] is not False:
        raise ContractError("an arm producer may author parser evidence")
    receipt_records = _exact_keys(
        records["receipt_json"],
        {
            "closed_schema",
            "required_fields_source",
            "value_schema_source",
            "producer_verdict_fields_forbidden",
            "all_identity_hashes_verified_before_payload_acceptance",
        },
        "trace receipt records",
    )
    if not all(
        receipt_records[field] is True
        for field in (
            "closed_schema",
            "producer_verdict_fields_forbidden",
            "all_identity_hashes_verified_before_payload_acceptance",
        )
    ):
        raise ContractError("trace receipt schema is not fail-closed")
    detached_seal = _exact_keys(
        records["detached_seal_json"],
        {
            "owner",
            "location",
            "closed_schema",
            "required_fields",
            "field_types",
            "producer_may_write",
            "arm_inventory_sha256_rule",
            "bundle_inventory_sha256_rule",
            "seal_covers_events_manifest_receipt_every_payload_and_shared_verifier_outputs",
            "integrity_binding_not_authentication_or_remote_attestation",
            "trusted_verifier_controlled_path_and_runner_required",
        },
        "trace detached seal",
    )
    if (
        detached_seal["owner"] != "third_party_verifier"
        or detached_seal["producer_may_write"] is not False
        or detached_seal["closed_schema"] is not True
    ):
        raise ContractError("trace detached seal ownership differs")
    if detached_seal["required_fields"] != [
        "arm_inventory_sha256",
        "bundle_inventory_sha256",
        "comparison_harness_sha256",
        "parser_harness_path",
        "parser_harness_sha256",
        "producer_contracts_sha256",
        "schema",
        "trace_schema_sha256",
    ]:
        raise ContractError("trace detached seal field set differs")
    if (
        detached_seal[
            "seal_covers_events_manifest_receipt_every_payload_and_shared_verifier_outputs"
        ]
        is not True
    ):
        raise ContractError("trace detached seal omits shared verifier evidence")
    if (
        detached_seal["integrity_binding_not_authentication_or_remote_attestation"] is not True
        or detached_seal["trusted_verifier_controlled_path_and_runner_required"] is not True
    ):
        raise ContractError("trace detached seal overclaims authentication")
    if set(detached_seal["field_types"]) != set(detached_seal["required_fields"]):
        raise ContractError("trace detached seal field types do not close its field set")
    validation_edges = schema["validation_edges"]
    if not isinstance(validation_edges, list) or [edge.get("id") for edge in validation_edges] != [
        "instrumentation_control",
        "post_prefill_all_schedules",
        "conv_boundary_prefixes",
    ]:
        raise ContractError("trace validation edges differ")
    cross_oracle = _exact_keys(
        schema["cross_oracle_comparison"],
        {
            "arms",
            "logical_join_key",
            "cardinality",
            "descriptor_rule",
            "coverage",
            "parser_rule",
            "producer_may_filter_or_select_pairs",
        },
        "trace cross-oracle comparison",
    )
    if cross_oracle["arms"] != ["transformers", "mlx_lm"]:
        raise ContractError("trace cross-oracle arms differ")
    if cross_oracle["logical_join_key"] != [
        "case_id",
        "run_id",
        "record_or_frame_kind",
        "call_index",
        "boundary_tag",
        "logical_prefix",
        "channel",
        "site_or_role",
        "layer",
        "selected_position_ids",
    ]:
        raise ContractError("trace cross-oracle join key differs")
    if cross_oracle["producer_may_filter_or_select_pairs"] is not False:
        raise ContractError("a trace producer may select cross-oracle pairs")
    bundle = _exact_keys(
        schema["bundle"],
        {
            "arm_root",
            "arm_files",
            "payload_root",
            "shared_verifier_root",
            "shared_verifier_files",
            "comparison_report",
            "arms_publish_independently",
            "producer_reads_other_arm",
            "producer_authored_pass_fields_forbidden",
            "third_party_verifier_required",
            "detached_external_seal_required",
            "exact_regular_file_inventory_required",
            "symlinks_hardlinks_and_special_files_forbidden",
            "atomic_publication_required_but_unimplemented",
        },
        "trace bundle",
    )
    for required_flag in (
        "arms_publish_independently",
        "producer_authored_pass_fields_forbidden",
        "third_party_verifier_required",
        "detached_external_seal_required",
        "exact_regular_file_inventory_required",
        "symlinks_hardlinks_and_special_files_forbidden",
        "atomic_publication_required_but_unimplemented",
    ):
        if bundle.get(required_flag) is not True:
            raise ContractError(f"trace bundle rule is not fail-closed: {required_flag}")
    if bundle.get("producer_reads_other_arm") is not False:
        raise ContractError("a trace producer may read the other arm")
    if bundle["comparison_report"] != (
        "absent_from_v1_raw_evidence_bundle; tolerance_and_comparison_report_schema_requires_a_"
        "reviewed_superseding_contract_after_clean_repeats_and_fault_injection"
    ):
        raise ContractError("trace bundle overclaims a comparison report")
    _require_exact_value(
        {
            "arm_root": bundle["arm_root"],
            "arm_files": bundle["arm_files"],
            "payload_root": bundle["payload_root"],
            "shared_verifier_root": bundle["shared_verifier_root"],
            "shared_verifier_files": bundle["shared_verifier_files"],
        },
        {
            "arm_root": "arms/<oracle_id>",
            "arm_files": ["events.jsonl", "payload-manifest.jsonl", "receipt.json"],
            "payload_root": "payload",
            "shared_verifier_root": "verifier",
            "shared_verifier_files": ["parser-outcomes.jsonl"],
        },
        "trace bundle paths",
    )
    comparison_metrics = _exact_keys(
        schema["comparison_metrics"],
        {"exact", "numeric", "state_diagnostics", "verifier_derived"},
        "trace comparison metrics",
    )
    for key in ("exact", "verifier_derived"):
        _string_list(comparison_metrics[key], f"trace comparison metrics {key}")
    numeric_metrics = _exact_keys(
        comparison_metrics["numeric"],
        {
            "accumulation",
            "nonfinite",
            "max_abs",
            "max_relative",
            "root_mean_square",
            "cosine_distance",
            "topk_overlap",
            "greedy_margin",
            "cutoff_margin",
            "reduction_axes",
        },
        "trace numeric metric formulas",
    )
    if numeric_metrics["accumulation"] != "float64_in_canonical_row_major_order":
        raise ContractError("trace numeric accumulation order differs")
    state_diagnostics = _exact_keys(
        comparison_metrics["state_diagnostics"],
        {"accumulation", "l2_norm", "state_max_magnitude", "finite_count", "scope"},
        "trace state diagnostic formulas",
    )
    if state_diagnostics["finite_count"] != (
        "sum_i(isfinite(x_i)); must equal product(shape)"
    ):
        raise ContractError("trace state finite-count rule differs")
    if schema["tolerances"] != {
        "status": "unfrozen_requires_clean_repeat_and_injected_fault_measurement",
        "numeric_thresholds": None,
        "producer_may_decide_pass": False,
    }:
        raise ContractError("trace schema invents a tolerance or producer verdict")
    if hashlib.sha256(canonical_json(schema)).hexdigest() != TRACE_SCHEMA_CANONICAL_SHA256:
        raise ContractError("trace schema differs from the reviewed canonical semantics")
    return schema


def _validate_critical_sources(value: Any, expected: list[dict[str, str]], context: str) -> None:
    if not isinstance(value, list):
        raise ContractError(f"{context} must be a list")
    for index, source in enumerate(value):
        source = _exact_keys(source, {"path", "sha256"}, f"{context}[{index}]")
        _safe_relative_path(source["path"], f"{context}[{index}].path")
        _sha256(source["sha256"], f"{context}[{index}].sha256")
    if value != expected:
        raise ContractError(f"{context} differs from the reviewed source pins")


def validate_producer_contracts_value(repo_root: Path, value: Any) -> dict[str, Any]:
    value = _exact_keys(
        value,
        {
            "schema",
            "state",
            "binds",
            "common",
            "input_preparation",
            "separation",
            "trust_boundary",
            "arms",
            "parameter_closure",
            "receipt_required_fields",
            "receipt_field_contracts",
            "publication",
            "unfrozen",
        },
        "producer contracts",
    )
    if value["schema"] != PRODUCER_SCHEMA:
        raise ContractError("producer-contract schema differs")
    if value["state"] != "semantic_modes_frozen_environments_unbuilt":
        raise ContractError("producer contracts claim an environment or execution")
    _require_exact_value(
        value["binds"],
        {
            "case_corpus_sha256": CASES_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_revision": SOURCE_REVISION,
            "trace_schema_path": "oracle/qwen38/trace-schema.json",
            "trace_schema_sha256": TRACE_SCHEMA_SHA256,
        },
        "producer bindings",
    )
    _require_exact_value(
        value["common"],
        {
            "batch_size": 1,
            "padding": False,
            "weights": "verified_checkpoint_bfloat16",
            "gdn_recurrent_state": "float32",
            "mode": "evaluation_inference_only",
            "generation": "direct_forward_fixed_step_greedy_no_generate_api",
            "argmax": "float32_logits_descending_value_then_ascending_token_id",
            "stop": "include_stop_then_no_later_forward",
            "max_steps": 16,
            "fresh_process_per_run": True,
            "fresh_cache_per_case": True,
            "attention_mask": None,
            "logical_positions": "absolute_arange_from_cache_offset_with_no_padding",
            "quantization": False,
            "vision": False,
            "mtp": False,
            "adapters": False,
            "network_during_execution": False,
            "source_verified_before_and_after": True,
            "producer_reads_other_arm": False,
            "producer_compares_or_decides_pass": False,
            "snapshots_materialized_before_cache_mutation": True,
            "forward_result_materialized_before_next_call": True,
        },
        "common producer semantics",
    )
    input_preparation = _exact_keys(
        value["input_preparation"],
        {
            "state",
            "ownership",
            "tokenizer_loader",
            "expected_class",
            "render_call",
            "tokenize_call",
            "rendered_encoding",
            "input_ids",
            "rejection_mapping",
            "parser_owner",
            "arm_may_consume_other_arm_input_bundle",
            "exact_input_bundle_sha256",
            "tokenizer_sources",
        },
        "producer input preparation",
    )
    if input_preparation["arm_may_consume_other_arm_input_bundle"] is not False:
        raise ContractError("a producer may consume the other arm input bundle")
    tokenizer_sources = _exact_keys(
        input_preparation["tokenizer_sources"],
        {"transformers", "mlx_lm"},
        "producer tokenizer sources",
    )
    for arm_id, source in tokenizer_sources.items():
        if not isinstance(source, dict):
            raise ContractError(f"producer tokenizer source {arm_id} must be an object")
        _sha256(source.get("tokenization_auto_sha256"), f"{arm_id} tokenization_auto hash")
        _sha256(source.get("tokenization_qwen2_sha256"), f"{arm_id} tokenization_qwen2 hash")
    _require_exact_value(
        value["separation"],
        {
            "dedicated_environment_per_arm": True,
            "dedicated_entrypoint_per_arm": True,
            "distinct_process_or_host": True,
            "shared_current_oracle_environment_as_two_arms_forbidden": True,
            "environment_and_entrypoint_aliases_forbidden": True,
        },
        "producer separation",
    )
    _require_exact_value(
        value["trust_boundary"],
        {
            "trusted_isolated_runner_required": True,
            "remote_hardware_attestation_provided": False,
            "source_and_environment_mounted_immutable_during_each_child": True,
            "concurrent_mutation_allowed": False,
            "producer_receipt_is_evidence_not_self_authenticating_proof": True,
            "external_verifier_recomputes_every_available_identity": True,
        },
        "producer trust boundary",
    )
    arms = value["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise ContractError("producer contracts must define exactly two arms")
    by_id: dict[str, dict[str, Any]] = {}
    arm_keys = {
        "id",
        "repository",
        "revision",
        "state",
        "critical_sources",
        "environment",
        "load",
        "execution",
        "instrumentation",
        "determinism",
        "snapshot",
    }
    for index, arm in enumerate(arms):
        arm = _exact_keys(arm, arm_keys, f"producer arm {index}")
        arm_id = _string(arm["id"], f"producer arm {index} ID")
        if arm_id in by_id:
            raise ContractError(f"producer arm repeats ID: {arm_id}")
        _commit(arm["revision"], f"producer arm {arm_id} revision")
        by_id[arm_id] = arm
    if list(by_id) != ["transformers", "mlx_lm"]:
        raise ContractError("producer arm IDs or ordering differ")
    if by_id["transformers"]["repository"] == by_id["mlx_lm"]["repository"]:
        raise ContractError("producer arms use the same implementation source")

    transformers = by_id["transformers"]
    if transformers["revision"] != "95940bf8775059a42f047256f076e4f607bc43ec":
        raise ContractError("Transformers producer revision differs")
    if transformers["state"] != "semantic_mode_frozen_environment_missing":
        raise ContractError("Transformers producer overclaims an environment")
    _validate_critical_sources(
        transformers["critical_sources"],
        [
            {
                "path": "src/transformers/cache_utils.py",
                "sha256": "0d5fd6901ce2b7108eff40e06d7ce29e9b0f9cc8ed2f40d2fb3a2e4d4f43e630",
            },
            {
                "path": "src/transformers/integrations/hub_kernels.py",
                "sha256": "50e5b5f938cdb2c5a2f7e90ae1ab3933cb2d505ab38c6c4f4d0226320df4b94a",
            },
            {
                "path": "src/transformers/masking_utils.py",
                "sha256": "e8c497af6979274fc6ae78980ad9893e7850bdb750e46d459f09178123992196",
            },
            {
                "path": "src/transformers/modeling_rope_utils.py",
                "sha256": "a8bf3f6a53760366fb5fa51cecc06a8707d3cded36fd8f3ac51e140c0718af21",
            },
            {
                "path": "src/transformers/modeling_utils.py",
                "sha256": "a7392c26dd2f005383cc3c1f8562638a6039d393ca03fb1933513479a6d264e4",
            },
            {
                "path": "src/transformers/models/auto/tokenization_auto.py",
                "sha256": "06115e3944dd73a2379a440f8131208dfaf1639a3992f29dce2d17f0e34785e0",
            },
            {
                "path": "src/transformers/models/qwen2/tokenization_qwen2.py",
                "sha256": "fac4e6576bfe2369731be147a4e530f262bdf32f2ac50436f96f0d8bdd2fc628",
            },
            {
                "path": "src/transformers/models/qwen3_5/configuration_qwen3_5.py",
                "sha256": "3c01b3cdcff8d77cbafac9841bc48c41e5a5b38637231f1bde3d843cd198dbaf",
            },
            {
                "path": "src/transformers/models/qwen3_5/modeling_qwen3_5.py",
                "sha256": "90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a",
            },
        ],
        "Transformers critical sources",
    )
    tf_environment = _exact_keys(
        transformers["environment"],
        {
            "lock_state",
            "current_oracle_lock_may_not_be_used",
            "python",
            "torch_wheel_and_backend",
            "accelerator",
            "cpu_or_disk_offload",
            "distributed_or_sharded",
        },
        "Transformers producer environment",
    )
    if tf_environment.get("lock_state") != "separate_lock_required_before_execution":
        raise ContractError("Transformers producer does not require a separate lock")
    if tf_environment.get("current_oracle_lock_may_not_be_used") is not True:
        raise ContractError("Transformers producer may reuse the mlx-lm oracle lock")
    if tf_environment.get("torch_wheel_and_backend") != "unfrozen":
        raise ContractError("Transformers producer invents a Torch/backend identity")
    if tf_environment.get("cpu_or_disk_offload") is not False:
        raise ContractError("Transformers producer permits offload")
    if tf_environment.get("distributed_or_sharded") is not False:
        raise ContractError("Transformers producer permits sharding")
    tf_load = _exact_keys(
        transformers["load"],
        {
            "class",
            "loader",
            "source",
            "torch_dtype",
            "attn_implementation",
            "device_placement",
            "output_loading_info",
            "local_files_only",
            "trust_remote_code",
            "strict_text_parameter_closure",
            "executed_prefixes",
            "allowed_loaded_but_unexecuted_prefixes",
            "required_ignored_prefixes",
            "randomly_initialized_text_parameters_allowed",
        },
        "Transformers producer load",
    )
    if tf_load.get("class") != "Qwen3_5ForConditionalGeneration":
        raise ContractError("Transformers producer class differs")
    if tf_load.get("source") != "verified_local_tree_only":
        raise ContractError("Transformers producer source differs")
    if tf_load.get("strict_text_parameter_closure") is not True:
        raise ContractError("Transformers producer lacks strict text-parameter closure")
    if tf_load.get("executed_prefixes") != ["model.language_model.", "lm_head."]:
        raise ContractError("Transformers executed tensor scope differs")
    if tf_load.get("allowed_loaded_but_unexecuted_prefixes") != ["model.visual."]:
        raise ContractError("Transformers vision scope differs")
    if tf_load.get("required_ignored_prefixes") != ["mtp."]:
        raise ContractError("Transformers MTP scope differs")
    if tf_load.get("local_files_only") is not True or tf_load.get("trust_remote_code") is not False:
        raise ContractError("Transformers load is not offline and code-closed")
    if tf_load.get("randomly_initialized_text_parameters_allowed") is not False:
        raise ContractError("Transformers load permits random text parameters")
    tf_execution = _exact_keys(
        transformers["execution"],
        {
            "forward_target",
            "forward_kwargs",
            "model_eval",
            "torch_inference_mode",
            "autocast",
            "torch_compile",
            "attention_implementation",
            "cache",
            "use_cache",
            "logits_to_keep",
            "logits_selection",
            "input_ids_dtype",
            "position_ids_dtype",
            "position_mapping",
            "hub_kernel_environment",
            "optional_imports_must_be_absent",
            "gdn_prefill_path",
            "gdn_decode_path",
            "conv_path",
            "resolved_callable_origins_must_be_receipted",
        },
        "Transformers producer execution",
    )
    if tf_execution.get("attention_implementation") != "eager":
        raise ContractError("Transformers attention mode differs")
    if tf_execution.get("hub_kernel_environment") != "USE_HUB_KERNELS=NO_before_python_import":
        raise ContractError("Transformers hub kernels are not disabled before import")
    if tf_execution.get("optional_imports_must_be_absent") != [
        "causal_conv1d",
        "fla",
        "flash_attn",
        "kernels",
        "xformers",
    ]:
        raise ContractError("Transformers optional-kernel denial list differs")
    if tf_execution.get("cache") != "DynamicCache(config=model.config)":
        raise ContractError("Transformers cache mode differs")
    if (
        tf_execution.get("model_eval") is not True
        or tf_execution.get("torch_inference_mode") is not True
        or tf_execution.get("use_cache") is not True
        or tf_execution.get("logits_to_keep") != 1
    ):
        raise ContractError("Transformers direct-forward mode differs")
    if tf_execution.get("position_mapping") != (
        "four_equal_planes_[text,time,height,width]_shape_[4,1,call_tokens]_from_canonical_absolute_positions"
    ):
        raise ContractError("Transformers position mapping differs")
    if tf_execution.get("input_ids_dtype") != "torch.int64" or tf_execution.get(
        "position_ids_dtype"
    ) != "torch.int64":
        raise ContractError("Transformers input dtype differs")
    if not all(
        tf_execution.get(key) is False
        for key in ("autocast", "torch_compile")
    ):
        raise ContractError("Transformers producer permits a hidden execution transform")
    if tf_execution.get("resolved_callable_origins_must_be_receipted") is not True:
        raise ContractError("Transformers callable origins are not receipted")
    _require_exact_value(
        transformers["instrumentation"],
        {
            "state": "algorithm_frozen_entrypoint_unbuilt",
            "stock_forward_target": "Qwen3_5ForConditionalGeneration.forward",
            "capture_route": "temporary_read_only_torch_forward_hooks",
            "module_mapping": {
                "embedding_output": "model.model.language_model.embed_tokens output",
                "decoder_output_<i>": (
                    "model.model.language_model.layers[i] returned hidden tensor"
                ),
                "final_norm_output": "model.model.language_model.norm output",
            },
            "selected_layers_only": [0, 2, 3, 30, 31, 32, 62, 63],
            "hook_returns_none": True,
            "hooks_removed_after_each_forward_including_error": True,
            "output_hidden_states_kwarg_used": False,
            "global_monkeypatch": False,
            "fresh_cache_stock_control": "behavior_monolithic_vs_trace_monolithic",
        },
        "Transformers instrumentation",
    )
    _require_exact_value(
        transformers["determinism"],
        {
            "pythonhashseed": "0",
            "cublas_workspace_config": ":4096:8",
            "python_numpy_torch_seed": 0,
            "deterministic_algorithms": "enabled_warn_only_false",
            "cudnn_benchmark": False,
            "float32_precision_apis": (
                "torch.backends.fp32_precision='ieee';"
                "torch.backends.cuda.matmul.fp32_precision='ieee'"
            ),
            "bfloat16_reduced_precision_and_split_k": [False, False],
            "fp16_reduced_precision_and_split_k": [False, False],
            "unsupported_deterministic_operation": "fatal",
        },
        "Transformers determinism",
    )
    if transformers["snapshot"] != "detach_clone_contiguous_device_sync_host_copy_then_hash":
        raise ContractError("Transformers snapshot semantics differ")

    mlx_lm = by_id["mlx_lm"]
    if mlx_lm["revision"] != "8239c72de5a0e42c539e30489021db73c7fe258c":
        raise ContractError("mlx-lm producer revision differs")
    if mlx_lm["state"] != "semantic_mode_frozen_environment_candidate_unexecuted":
        raise ContractError("mlx-lm producer overclaims execution")
    _validate_critical_sources(
        mlx_lm["critical_sources"],
        [
            {
                "path": "mlx_lm/models/base.py",
                "sha256": "61330e1c065739cd712bfeb09d673f33797cde7e613e95bf6d9ebbee9006f373",
            },
            {
                "path": "mlx_lm/models/cache.py",
                "sha256": "819ed95dcbf755652363cfdb15a639890447abb534a06dcefd52c7fff5055750",
            },
            {
                "path": "mlx_lm/models/gated_delta.py",
                "sha256": "79c8376a51c694b03e54d2f996ced6ea6c8c42868b8571529f97334db165a3e1",
            },
            {
                "path": "mlx_lm/models/qwen3_5.py",
                "sha256": "cdcfbf22681d2005f4bdff53ab9ab06da7aeb71c0e89378f65f893beeaf0b47c",
            },
            {
                "path": "mlx_lm/models/qwen3_next.py",
                "sha256": "3c572fe3fbb36721efab4d80d1bb6af11beb4ad1caae18deefc9fc84cbcd9b79",
            },
            {
                "path": "mlx_lm/models/rope_utils.py",
                "sha256": "9f68c938c040fa111d13f2ed95c70e8261515fb3b54f8a0a474c096baf4e087a",
            },
            {
                "path": "mlx_lm/utils.py",
                "sha256": "9473634d92dbba39d5133a7a92062c00a642aeb3b7477478d87a67ce5f34c22c",
            },
        ],
        "mlx-lm critical sources",
    )
    mlx_environment = _exact_keys(
        mlx_lm["environment"],
        {
            "lock_state",
            "lock_path",
            "lock_sha256",
            "python",
            "mlx",
            "mlx_lm_revision",
            "selected_mlx_metal_wheel",
            "accelerator",
            "pipeline_or_distributed",
        },
        "mlx-lm producer environment",
    )
    if mlx_environment.get("lock_state") != "existing_lock_candidate_unexecuted_for_qwen":
        raise ContractError("mlx-lm environment state differs")
    lock_path = _regular_repo_file(repo_root, mlx_environment.get("lock_path"), "mlx-lm lock")
    lock_hash = _sha256(mlx_environment.get("lock_sha256"), "mlx-lm lock hash")
    if lock_hash != MLX_ORACLE_LOCK_SHA256 or sha256_file(lock_path) != lock_hash:
        raise ContractError("mlx-lm candidate lock identity differs")
    if mlx_environment.get("selected_mlx_metal_wheel") != "unfrozen_until_host_selection":
        raise ContractError("mlx-lm producer invents a host wheel")
    if mlx_environment.get("pipeline_or_distributed") is not False:
        raise ContractError("mlx-lm producer permits pipeline or distributed execution")
    mlx_load = _exact_keys(
        mlx_lm["load"],
        {
            "class",
            "loader",
            "source",
            "model_file_config_field_must_be_absent",
            "lazy",
            "strict",
            "strict_text_parameter_closure",
            "source_selected_prefixes",
            "runtime_parameter_prefixes",
            "required_omitted_prefixes",
            "randomly_initialized_text_parameters_allowed",
        },
        "mlx-lm producer load",
    )
    if mlx_load.get("class") != "mlx_lm.models.qwen3_5.Model":
        raise ContractError("mlx-lm producer class differs")
    if mlx_load.get("source") != (
        "verified_full_unmodified_checkpoint_tree_then_pinned_Model.sanitize_to_text"
    ):
        raise ContractError("mlx-lm producer source or sanitize order differs")
    if mlx_load.get("strict_text_parameter_closure") is not True:
        raise ContractError("mlx-lm producer lacks strict text-parameter closure")
    if mlx_load.get("source_selected_prefixes") != ["model.language_model.", "lm_head."]:
        raise ContractError("mlx-lm source tensor scope differs")
    if mlx_load.get("runtime_parameter_prefixes") != [
        "language_model.lm_head.",
        "language_model.model.",
    ]:
        raise ContractError("mlx-lm runtime parameter scope differs")
    if mlx_load.get("required_omitted_prefixes") != ["model.visual.", "mtp."]:
        raise ContractError("mlx-lm omitted tensor scope differs")
    if mlx_load.get("lazy") is not False or mlx_load.get("strict") is not True:
        raise ContractError("mlx-lm load is not eager and strict")
    if mlx_load.get("randomly_initialized_text_parameters_allowed") is not False:
        raise ContractError("mlx-lm load permits random text parameters")
    mlx_execution = _exact_keys(
        mlx_lm["execution"],
        {
            "forward_target",
            "forward_kwargs",
            "model_training",
            "device",
            "stream",
            "global_compile_wrapper",
            "cache",
            "logits_selection",
            "input_ids_dtype",
            "position_mapping",
            "gdn_primary_path",
            "gdn_ops_fallback",
            "mx_eval_logits_and_all_cache_leaves_after_every_forward",
            "mx_gpu_synchronize_before_host_copy",
            "default_device_gpu_assertion",
            "metal_available_assertion",
            "stock_kernel_route_assertion_per_call",
            "resolved_kernel_and_callable_origins_must_be_receipted",
        },
        "mlx-lm producer execution",
    )
    if mlx_execution.get("cache") != "model.make_cache_48_arrayscache_16_kvcache":
        raise ContractError("mlx-lm cache mode differs")
    if mlx_execution.get("input_ids_dtype") != "mx.int32":
        raise ContractError("mlx-lm input dtype differs")
    if mlx_execution.get("model_training") is not False or mlx_execution.get("device") != "mlx_gpu_metal":
        raise ContractError("mlx-lm primary device or evaluation mode differs")
    if mlx_execution.get("gdn_primary_path") != "stock_metal_eval_kernel":
        raise ContractError("mlx-lm primary GDN path differs")
    if mlx_execution.get("gdn_ops_fallback") != "diagnostic_only_never_silent_primary":
        raise ContractError("mlx-lm ops fallback can silently become primary")
    for required_flag in (
        "mx_eval_logits_and_all_cache_leaves_after_every_forward",
        "mx_gpu_synchronize_before_host_copy",
        "default_device_gpu_assertion",
        "metal_available_assertion",
        "stock_kernel_route_assertion_per_call",
        "resolved_kernel_and_callable_origins_must_be_receipted",
    ):
        if mlx_execution.get(required_flag) is not True:
            raise ContractError(f"mlx-lm execution omits {required_flag}")
    if mlx_execution.get("global_compile_wrapper") is not False:
        raise ContractError("mlx-lm producer permits a global compile wrapper")
    _require_exact_value(
        mlx_lm["instrumentation"],
        {
            "state": "algorithm_frozen_entrypoint_unbuilt",
            "stock_forward_target": "mlx_lm.models.qwen3_5.Model.__call__",
            "capture_route": (
                "producer_entrypoint_explicit_text_orchestration_without_module_monkeypatch"
            ),
            "object_paths": {
                "embed_tokens": "model.language_model.model.embed_tokens",
                "layers": "model.language_model.model.pipeline_layers",
                "final_norm": "model.language_model.model.norm",
                "lm_head": "model.language_model.lm_head",
            },
            "orchestration": [
                (
                    "assert pipeline_size=1 pipeline_rank=0 ssm_idx=0 fa_idx=3 and exactly 64 "
                    "pipeline_layers with the frozen layer-kind pattern"
                ),
                "embed inputs once and capture embedding_output",
                (
                    "build fa_mask with pinned create_attention_mask(hidden,cache[fa_idx]) and "
                    "ssm_mask with pinned create_ssm_mask(hidden,cache[ssm_idx]) exactly once "
                    "per call"
                ),
                (
                    "for each ordered layer and cache entry choose ssm_mask for GDN or fa_mask "
                    "for full attention, call the pinned layer, and capture selected post-block "
                    "decoder outputs"
                ),
                (
                    "apply the pinned final norm once, capture final_norm_output, then apply the "
                    "untied pinned lm_head once"
                ),
                (
                    "evaluate logits selected captures and every cache leaf together, synchronize "
                    "the default Metal GPU stream, then host-copy snapshots"
                ),
            ],
            "global_monkeypatch": False,
            "fresh_cache_stock_control": "behavior_monolithic_vs_trace_monolithic",
        },
        "mlx-lm instrumentation",
    )
    _require_exact_value(
        mlx_lm["determinism"],
        {
            "pythonhashseed": "0",
            "python_numpy_mlx_seed": 0,
            "greedy_only": True,
            "unexpected_device_or_ops_fallback": "fatal",
        },
        "mlx-lm determinism",
    )
    if mlx_lm["snapshot"] != "copy_materialize_mx_eval_gpu_sync_host_copy_then_hash":
        raise ContractError("mlx-lm snapshot semantics differ")

    receipt_fields = _string_list(value["receipt_required_fields"], "producer receipt fields")
    if receipt_fields != sorted(receipt_fields):
        raise ContractError("producer receipt fields must be sorted")
    required_receipt_fields = {
        "argv",
        "backend_and_device",
        "cache_inventory",
        "case_corpus_sha256",
        "command_environment_allowlist",
        "contract_sha256",
        "conversation_and_input_bundle_sha256",
        "critical_import_origins_and_hashes",
        "determinism_controls",
        "effective_forward_routes_and_static_kwargs",
        "environment_lock_sha256",
        "environment_tree_identity",
        "installed_distribution_inventory",
        "machine_profile_without_serial_numbers",
        "model_source_revision",
        "oracle_id",
        "oracle_implementation_revision",
        "oracle_repository",
        "parameter_dtype_device_and_prefix_inventory",
        "per_call_kernel_route_evidence",
        "process_inventory",
        "producer_contracts_sha256",
        "producer_entrypoint_sha256",
        "publication_target",
        "python_executable_and_abi",
        "raw_payload_inventory_sha256",
        "resolved_kernel_inventory",
        "selected_distribution_artifact_hashes_and_direct_urls",
        "source_manifest_sha256",
        "source_pre_verification",
        "source_post_verification",
        "source_tree_identity",
        "start_and_end_utc",
        "trace_schema_sha256",
    }
    if set(receipt_fields) != required_receipt_fields:
        raise ContractError("producer receipt field set differs")
    receipt_contracts = _exact_keys(
        value["receipt_field_contracts"],
        {
            "closed_schema",
            "sha256_fields",
            "commit_fields",
            "string_fields",
            "nonempty_array_fields",
            "nonempty_object_fields",
            "nested_required_fields",
            "nested_field_types",
            "nested_nullable_fields",
            "nested_digest_preimages",
            "nested_semantic_rules",
            "per_arm_environment_keys",
            "per_arm_memory_diagnostic_profiles",
            "digest_preimages",
            "semantic_rules",
        },
        "producer receipt field contracts",
    )
    if receipt_contracts["closed_schema"] is not True:
        raise ContractError("producer receipt schema is open")
    typed_receipt_fields = set(
        _string_list(receipt_contracts["sha256_fields"], "receipt SHA-256 fields")
    ) | set(_string_list(receipt_contracts["commit_fields"], "receipt commit fields"))
    for key in ("string_fields", "nonempty_array_fields", "nonempty_object_fields"):
        mapping = receipt_contracts[key]
        if not isinstance(mapping, dict) or not mapping:
            raise ContractError(f"producer receipt {key} must be a nonempty object")
        typed_receipt_fields.update(mapping)
    if typed_receipt_fields != required_receipt_fields:
        raise ContractError("producer receipt value types do not cover the exact field set")
    nested_fields = receipt_contracts["nested_required_fields"]
    if not isinstance(nested_fields, dict) or not nested_fields:
        raise ContractError("producer receipt nested schemas are absent")
    for key, fields in nested_fields.items():
        if _string_list(fields, f"producer receipt nested schema {key}") != sorted(fields):
            raise ContractError(f"producer receipt nested schema {key} is not sorted")
    _require_exact_value(
        nested_fields["memory_and_swap_before_peak_after"],
        [
            "accelerator_memory_authority",
            "after_execution_bytes",
            "after_load_bytes",
            "before_load_bytes",
            "diagnostic_only_noncomparable",
            "memory_scope",
            "peak_execution_bytes",
            "sampling_interval_milliseconds",
            "swap_after_bytes",
            "swap_authority",
            "swap_before_bytes",
            "swap_peak_bytes",
            "swap_scope",
            "wired_memory_policy",
        ],
        "producer memory diagnostic field set",
    )
    nested_field_types = _exact_keys(
        receipt_contracts["nested_field_types"],
        {
            "boolean",
            "closed_object",
            "nonnegative_integer",
            "positive_integer_array",
            "sha256",
            "string",
            "string_array",
        },
        "producer receipt nested field types",
    )
    expected_nested_paths = {
        f"{schema_name}.{field}"
        for schema_name, fields in nested_fields.items()
        for field in fields
    }
    typed_nested_paths: list[str] = []
    for field_type, paths in nested_field_types.items():
        typed_paths = _string_list(paths, f"producer receipt nested {field_type} fields")
        if typed_paths != sorted(typed_paths):
            raise ContractError(f"producer receipt nested {field_type} fields are not sorted")
        typed_nested_paths.extend(typed_paths)
    if len(typed_nested_paths) != len(set(typed_nested_paths)) or set(
        typed_nested_paths
    ) != expected_nested_paths:
        raise ContractError("producer receipt nested field types do not close the field set")
    nullable_nested_paths = _string_list(
        receipt_contracts["nested_nullable_fields"],
        "producer receipt nested nullable fields",
    )
    if nullable_nested_paths != sorted(nullable_nested_paths) or not set(
        nullable_nested_paths
    ).issubset(expected_nested_paths):
        raise ContractError("producer receipt nested nullability differs")
    nested_digest_preimages = receipt_contracts["nested_digest_preimages"]
    if not isinstance(nested_digest_preimages, dict) or set(nested_digest_preimages) != set(
        nested_field_types["sha256"]
    ):
        raise ContractError("producer receipt nested digest preimages differ")
    for path, preimage in nested_digest_preimages.items():
        _string(preimage, f"producer receipt nested digest preimage {path}")
    _string_list(
        receipt_contracts["nested_semantic_rules"],
        "producer receipt nested semantic rules",
    )
    environment_keys = _exact_keys(
        receipt_contracts["per_arm_environment_keys"],
        {"transformers", "mlx_lm"},
        "producer receipt environment keys",
    )
    for arm_id, fields in environment_keys.items():
        if _string_list(fields, f"producer receipt environment {arm_id}") != sorted(fields):
            raise ContractError(f"producer receipt environment {arm_id} is not sorted")
    _require_exact_value(
        environment_keys,
        {
            "transformers": [
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED",
                "PYTHONNOUSERSITE",
                "PYTHONSAFEPATH",
                "TRANSFORMERS_OFFLINE",
                "USE_HUB_KERNELS",
            ],
            "mlx_lm": [
                "HF_HUB_OFFLINE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED",
                "PYTHONNOUSERSITE",
                "PYTHONSAFEPATH",
                "TRANSFORMERS_OFFLINE",
            ],
        },
        "producer receipt environment allowlists",
    )
    memory_profiles = _exact_keys(
        receipt_contracts["per_arm_memory_diagnostic_profiles"],
        {"transformers", "mlx_lm"},
        "producer receipt memory diagnostic profiles",
    )
    _require_exact_value(
        memory_profiles,
        {
            "transformers": {
                "accelerator_memory_authority": (
                    "torch.cuda.memory_allocated_and_max_memory_allocated_with_peak_reset_"
                    "immediately_after_after_load_sample"
                ),
                "diagnostic_only_noncomparable": True,
                "memory_scope": "model_process_accelerator_allocator_active_bytes",
                "sampling_interval_milliseconds": 10,
                "swap_authority": (
                    "linux_proc_meminfo_swap_total_minus_swap_free_at_boundaries_and_parent_"
                    "10ms_samples"
                ),
                "swap_scope": "host_global_used_swap_bytes",
            },
            "mlx_lm": {
                "accelerator_memory_authority": (
                    "mx.metal.get_active_memory_and_get_peak_memory_with_peak_reset_"
                    "immediately_after_after_load_sample"
                ),
                "diagnostic_only_noncomparable": True,
                "memory_scope": "model_process_accelerator_allocator_active_bytes",
                "sampling_interval_milliseconds": 10,
                "swap_authority": (
                    "macos_sysctlbyname_vm.swapusage_used_at_boundaries_and_parent_10ms_"
                    "samples"
                ),
                "swap_scope": "host_global_used_swap_bytes",
            },
        },
        "producer receipt memory diagnostic profiles",
    )
    digest_preimages = receipt_contracts["digest_preimages"]
    if not isinstance(digest_preimages, dict) or set(digest_preimages) != set(
        receipt_contracts["sha256_fields"]
    ):
        raise ContractError("producer receipt digest preimages differ from its digest fields")
    _string_list(receipt_contracts["semantic_rules"], "producer receipt semantic rules")
    parameter_closure = _exact_keys(
        value["parameter_closure"],
        {
            "state",
            "expected_source_tensor_counts",
            "requirements",
            "execution_may_begin_before_implementation",
        },
        "producer parameter closure",
    )
    if (
        parameter_closure["state"] != "required_algorithm_frozen_implementation_pending"
        or parameter_closure["execution_may_begin_before_implementation"] is not False
    ):
        raise ContractError("producer parameter closure overclaims or permits execution")
    _require_exact_value(
        parameter_closure["expected_source_tensor_counts"],
        {
            "mtp_omitted": 15,
            "selected_text_and_lm_head": 851,
            "total": 1199,
            "vision_loaded_or_omitted_by_arm": 333,
        },
        "producer source tensor counts",
    )
    _string_list(parameter_closure["requirements"], "producer parameter closure requirements")
    publication = _exact_keys(
        value["publication"],
        {
            "state",
            "sibling_staging_directory",
            "files_and_directories_fsynced",
            "same_filesystem_atomic_rename",
            "parent_directory_fsynced",
            "existing_target_refused",
            "manifest_written_last",
            "no_skip_or_partial_success",
        },
        "producer publication",
    )
    if publication.get("state") != "required_behavior_not_implemented":
        raise ContractError("producer publication overclaims implementation")
    for required_flag in (
        "sibling_staging_directory",
        "files_and_directories_fsynced",
        "same_filesystem_atomic_rename",
        "parent_directory_fsynced",
        "existing_target_refused",
        "manifest_written_last",
        "no_skip_or_partial_success",
    ):
        if publication.get(required_flag) is not True:
            raise ContractError(f"producer publication weakens {required_flag}")
    unfrozen = _string_list(value["unfrozen"], "producer unfrozen identities")
    required_unfrozen = {
        "transformers_dependency_lock_and_wheel_hashes",
        "transformers_torch_cuda_and_cudnn_builds",
        "transformers_gpu_model_driver_and_uuid",
        "mlx_selected_wheel_machine_and_macos_build",
        "both_installed_package_tree_digests",
        "both_producer_entrypoints",
        "both_clean_execution_receipts",
        "both_exact_python_builds_and_executables",
        "run_to_run_variance",
        "numeric_tolerances",
        "raw_payload_identities",
        "rendered_and_tokenized_input_bundle_identity",
        "cross_oracle_agreement",
        "native_support",
    }
    if set(unfrozen) != required_unfrozen:
        raise ContractError("producer unfrozen identity set differs")
    if (
        hashlib.sha256(canonical_json(value)).hexdigest()
        != PRODUCER_CONTRACTS_CANONICAL_SHA256
    ):
        raise ContractError("producer contracts differ from the reviewed canonical semantics")
    return value


def mlx_kv_capacity_after_calls(call_token_counts: list[int]) -> tuple[int, int]:
    """Return the pinned mlx-lm KVCache logical length and allocated capacity."""

    logical_offset = 0
    capacity = 0
    for index, call_tokens in enumerate(call_token_counts):
        if type(call_tokens) is not int or call_tokens <= 0:
            raise ContractError(f"MLX KV call {index} token count must be a positive integer")
        if logical_offset + call_tokens > capacity:
            if capacity > 0 and logical_offset % 256 != 0:
                capacity = logical_offset
            capacity += ((call_tokens + 255) // 256) * 256
        logical_offset += call_tokens
    return logical_offset, capacity


def normalize_synthetic_layout(
    values: list[int],
    shape: list[int],
    transform: str,
    *,
    logical_offset: int | None = None,
    call_token_counts: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Exercise reviewed axis transforms without accepting a real trace payload."""

    if not shape or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ContractError("synthetic tensor shape must contain positive integers")
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    if len(values) != element_count:
        raise ContractError("synthetic tensor element count differs from its shape")
    if transform == "identity":
        return list(values), list(shape)
    if transform == "transpose_last_two":
        if len(shape) < 2:
            raise ContractError("last-two transpose requires rank >= 2")
        rows, columns = shape[-2:]
        outer = element_count // (rows * columns)
        output = []
        for outer_index in range(outer):
            base = outer_index * rows * columns
            for column in range(columns):
                for row in range(rows):
                    output.append(values[base + row * columns + column])
        return output, [*shape[:-2], columns, rows]
    if transform == "take_last_3_then_transpose_0_2_1":
        if len(shape) != 3 or shape[-1] != 4:
            raise ContractError("convolution normalization requires [B,C,4]")
        batch, channels, retained = shape
        output = []
        for batch_index in range(batch):
            for history in range(1, retained):
                for channel in range(channels):
                    output.append(values[(batch_index * channels + channel) * retained + history])
        return output, [batch, 3, channels]
    if transform == "slice_token_axis_to_logical_offset":
        if len(shape) != 4 or logical_offset is None or call_token_counts is None:
            raise ContractError(
                "KV normalization requires rank 4, a logical offset, and call token counts"
            )
        batch, heads, capacity, features = shape
        if (
            logical_offset <= 0
            or logical_offset > capacity
            or capacity >= logical_offset + 256
        ):
            raise ContractError("KV capacity does not satisfy L<=C<L+256")
        expected_offset, expected_capacity = mlx_kv_capacity_after_calls(call_token_counts)
        if (logical_offset, capacity) != (expected_offset, expected_capacity):
            raise ContractError("KV capacity does not satisfy the pinned mlx-lm recurrence")
        output = []
        for batch_index in range(batch):
            for head in range(heads):
                base = (batch_index * heads + head) * capacity * features
                output.extend(values[base : base + logical_offset * features])
        return output, [batch, heads, logical_offset, features]
    raise ContractError(f"unsupported synthetic layout transform: {transform}")


def _validate_contract_value(repo_root: Path, contract: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = _exact_keys(
        contract,
        {
            "schema",
            "supersedes",
            "status",
            "source",
            "component",
            "oracles",
            "producer_contracts",
            "conversation",
            "required_coverage",
            "trace",
            "required_evidence_identities",
        },
        "oracle contract",
    )
    if contract["schema"] != CONTRACT_SCHEMA:
        raise ContractError("oracle contract schema differs")
    if contract["supersedes"] != {
        "schema": PREVIOUS_CONTRACT_SCHEMA,
        "sha256": PREVIOUS_CONTRACT_SHA256,
    }:
        raise ContractError("oracle contract does not bind the reviewed v1 predecessor")
    if contract["status"] != STATUS:
        raise ContractError("oracle contract status overclaims or differs")

    source = _exact_keys(
        contract["source"],
        {"repository", "revision", "manifest_path", "manifest_sha256"},
        "oracle contract source",
    )
    if source["repository"] != SOURCE_REPOSITORY or source["revision"] != SOURCE_REVISION:
        raise ContractError("oracle contract source identity differs")
    manifest_path = _regular_repo_file(repo_root, source["manifest_path"], "source manifest")
    manifest_hash = _sha256(source["manifest_sha256"], "source manifest hash")
    if manifest_hash != SOURCE_MANIFEST_SHA256:
        raise ContractError("source manifest hash is not the reviewed pin")
    if sha256_file(manifest_path) != manifest_hash:
        raise ContractError("source manifest hash differs from the contract")
    source_manifest = load_json_no_duplicates(manifest_path)
    source_files = validate_source_manifest_value(source_manifest)

    component = _exact_keys(
        contract["component"],
        {
            "scope",
            "outer_architecture",
            "outer_model_type",
            "text_model_type",
            "included_tensor_prefixes",
            "omitted_tensor_prefixes",
            "text_parameters",
            "num_hidden_layers",
            "linear_attention_layers",
            "full_attention_layers",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "tie_word_embeddings",
            "recurrent_state_dtype",
        },
        "oracle contract component",
    )
    expected_component = {
        "scope": "text-only",
        "outer_architecture": "Qwen3_5ForConditionalGeneration",
        "outer_model_type": "qwen3_5",
        "text_model_type": "qwen3_5_text",
        "included_tensor_prefixes": ["model.language_model.", "lm_head."],
        "omitted_tensor_prefixes": ["model.visual.", "mtp."],
        "text_parameters": 26895998464,
        "num_hidden_layers": 64,
        "linear_attention_layers": 48,
        "full_attention_layers": 16,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "tie_word_embeddings": False,
        "recurrent_state_dtype": "float32",
    }
    if component != expected_component:
        raise ContractError("Qwen text component geometry or scope differs")

    oracles = contract["oracles"]
    if not isinstance(oracles, list) or len(oracles) != 2:
        raise ContractError("oracle contract must pin exactly two implementations")
    oracle_ids: set[str] = set()
    oracle_repositories: set[str] = set()
    oracle_revisions: set[str] = set()
    for index, oracle in enumerate(oracles):
        oracle = _exact_keys(
            oracle,
            {
                "id",
                "repository",
                "revision",
                "implementation_path",
                "implementation_sha256",
                "state",
                "execution_mode_status",
                "execution_intent",
            },
            f"oracle {index}",
        )
        oracle_id = _string(oracle["id"], f"oracle {index} id")
        repository = _string(oracle["repository"], f"oracle {oracle_id} repository")
        revision = _commit(oracle["revision"], f"oracle {oracle_id} revision")
        _safe_relative_path(oracle["implementation_path"], f"oracle {oracle_id} implementation path")
        _sha256(oracle["implementation_sha256"], f"oracle {oracle_id} implementation hash")
        if oracle["state"] != "source_pinned_unexecuted":
            raise ContractError(f"oracle {oracle_id} claims execution")
        if oracle["execution_mode_status"] != "semantic_mode_frozen_environment_unbuilt":
            raise ContractError(f"oracle {oracle_id} execution-mode state differs")
        if not isinstance(oracle["execution_intent"], dict) or not oracle["execution_intent"]:
            raise ContractError(f"oracle {oracle_id} lacks an execution intent")
        oracle_ids.add(oracle_id)
        oracle_repositories.add(repository)
        oracle_revisions.add(revision)
    if oracle_ids != {"transformers", "mlx_lm"}:
        raise ContractError("oracle IDs differ")
    if len(oracle_repositories) != 2 or len(oracle_revisions) != 2:
        raise ContractError("oracle implementations are not independent source pins")
    if oracles != EXPECTED_ORACLES:
        raise ContractError("oracle source pins or execution modes differ")

    producer_reference = _exact_keys(
        contract["producer_contracts"],
        {"state", "path", "sha256"},
        "producer-contract reference",
    )
    if producer_reference["state"] != "semantic_modes_frozen_environments_unbuilt":
        raise ContractError("producer-contract reference overclaims execution")
    producer_path = _regular_repo_file(
        repo_root,
        producer_reference["path"],
        "Qwen producer contracts",
    )
    producer_hash = _sha256(producer_reference["sha256"], "producer-contract hash")
    if producer_hash != PRODUCER_CONTRACTS_SHA256:
        raise ContractError("producer-contract hash is not the reviewed pin")
    if sha256_file(producer_path) != producer_hash:
        raise ContractError("producer-contract file hash differs")
    producer_contracts = load_json_no_duplicates(producer_path)
    validate_producer_contracts_value(repo_root, producer_contracts)

    conversation = _exact_keys(
        contract["conversation"],
        {
            "cases_path",
            "cases_sha256",
            "case_count",
            "tokenizer_class",
            "tokenizer_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
            "generation_config_sha256",
            "special_tokens",
            "stop_token_ids",
            "shared_dependency_disclosure",
        },
        "conversation contract",
    )
    cases_path = _regular_repo_file(repo_root, conversation["cases_path"], "Qwen cases")
    cases_hash = _sha256(conversation["cases_sha256"], "Qwen cases hash")
    if cases_hash != CASES_SHA256:
        raise ContractError("Qwen cases hash is not the reviewed pin")
    if sha256_file(cases_path) != cases_hash:
        raise ContractError("Qwen cases hash differs from the contract")
    cases = load_cases(cases_path)
    if conversation["case_count"] != len(cases):
        raise ContractError("Qwen case_count differs")
    if conversation["tokenizer_class"] != "Qwen2Tokenizer":
        raise ContractError("Qwen tokenizer class differs")
    critical_hashes = {
        "tokenizer.json": "tokenizer_sha256",
        "tokenizer_config.json": "tokenizer_config_sha256",
        "chat_template.jinja": "chat_template_sha256",
        "generation_config.json": "generation_config_sha256",
    }
    for source_path, conversation_key in critical_hashes.items():
        expected = _sha256(conversation[conversation_key], f"conversation {conversation_key}")
        if source_files[source_path]["sha256"] != expected:
            raise ContractError(f"conversation {conversation_key} differs from source inventory")
    if conversation["special_tokens"] != SPECIAL_TOKENS:
        raise ContractError("Qwen special-token map differs")
    if conversation["stop_token_ids"] != [248046, 248044]:
        raise ContractError("Qwen stop-token ordering differs")
    disclosure = _string(conversation["shared_dependency_disclosure"], "shared dependency disclosure")
    if "not an independent model-math vote" not in disclosure:
        raise ContractError("shared tokenizer/template dependency is not disclosed")

    required_coverage = _string_list(contract["required_coverage"], "required coverage")
    if required_coverage != sorted(required_coverage):
        raise ContractError("required coverage must be sorted")
    if required_coverage != REQUIRED_COVERAGE:
        raise ContractError("required coverage differs from the reviewed contract")
    actual_coverage = {tag for case in cases for tag in case["coverage"]}
    if actual_coverage != set(required_coverage):
        raise ContractError(
            "case coverage differs: "
            f"missing={sorted(set(required_coverage) - actual_coverage)}, "
            f"extra={sorted(actual_coverage - set(required_coverage))}"
        )

    trace = _exact_keys(
        contract["trace"],
        {"state", "schema_path", "schema_sha256", "bundle_rules", "tolerances"},
        "trace contract",
    )
    if trace["state"] != "schema_frozen_no_payload":
        raise ContractError("trace contract claims a payload")
    trace_path = _regular_repo_file(repo_root, trace["schema_path"], "Qwen trace schema")
    trace_hash = _sha256(trace["schema_sha256"], "Qwen trace-schema hash")
    if trace_hash != TRACE_SCHEMA_SHA256:
        raise ContractError("trace-schema hash is not the reviewed pin")
    if sha256_file(trace_path) != trace_hash:
        raise ContractError("trace-schema file hash differs")
    trace_schema = load_json_no_duplicates(trace_path)
    validate_trace_schema_value(trace_schema, cases)
    expected_rules = {
        "raw_oracle_outputs_separate": True,
        "producer_authored_pass_forbidden": True,
        "third_party_comparison_required": True,
        "exact_inventory_required": True,
        "external_inventory_digest_required": True,
        "shared_verifier_outputs_sealed": True,
        "atomic_publication_required": True,
        "source_verified_before_and_after": True,
    }
    if trace["bundle_rules"] != expected_rules:
        raise ContractError("trace bundle rules differ")
    if trace["tolerances"] != {
        "status": "unfrozen_requires_clean_repeat_and_injected_fault_measurement"
    }:
        raise ContractError("trace contract invents or changes tolerances")
    if (
        _string_list(contract["required_evidence_identities"], "required evidence identities")
        != REQUIRED_EVIDENCE_IDENTITIES
    ):
        raise ContractError("required evidence identities differ from the reviewed contract")
    return contract, cases


def validate_contract_value(repo_root: Path, contract: Any) -> dict[str, Any]:
    validated, cases = _validate_contract_value(repo_root, contract)
    return {
        "schema": VALIDATION_SCHEMA,
        "contract_schema": validated["schema"],
        "source_revision": validated["source"]["revision"],
        "source_manifest_sha256": validated["source"]["manifest_sha256"],
        "case_corpus_sha256": validated["conversation"]["cases_sha256"],
        "trace_schema_sha256": validated["trace"]["schema_sha256"],
        "producer_contracts_sha256": validated["producer_contracts"]["sha256"],
        "case_count": len(cases),
        "oracle_ids": sorted(oracle["id"] for oracle in validated["oracles"]),
        "contract_only_unexecuted": True,
        "trace_schema_frozen": True,
        "producer_semantics_frozen": True,
        "producer_environments_frozen": False,
        "native_executable": False,
        "support_accepted": False,
    }


def validate_contract(repo_root: Path, contract_path: Path) -> dict[str, Any]:
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ContractError("oracle contract must be a real regular file")
    return validate_contract_value(repo_root, load_json_no_duplicates(contract_path))


def _verify_exact_tree(root: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError("source root must be a real directory")
    expected = {item["path"]: item for item in source_manifest["files"]}
    actual: dict[str, Path] = {}
    seen_inodes: dict[tuple[int, int], str] = {}

    def traversal_failed(error: OSError) -> None:
        raise ContractError(f"source tree traversal failed closed: {error}") from error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=traversal_failed):
        directories.sort()
        files.sort()
        directory_path = Path(directory)
        for name in directories:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ContractError(f"source tree contains an unsafe directory: {relative}")
        for name in files:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ContractError(f"source tree contains an unsafe file: {relative}")
            inode = (metadata.st_dev, metadata.st_ino)
            if metadata.st_nlink != 1 or inode in seen_inodes:
                raise ContractError(f"source tree contains a hard-link alias: {relative}")
            seen_inodes[inode] = relative
            actual[relative] = candidate

    if set(actual) != set(expected):
        raise ContractError(
            "source tree inventory differs: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    identity_lines = []
    for relative, item in sorted(expected.items()):
        candidate = actual[relative]
        size = candidate.stat().st_size
        if size != item["size"]:
            raise ContractError(f"source size differs for {relative}")
        digest = sha256_file(candidate)
        if digest != item["sha256"]:
            raise ContractError(f"source hash differs for {relative}")
        identity_lines.append(f"{digest}  {size}  {relative}\n")
    return {
        "schema": "hyperion.qwen38-source-tree-identity.v1",
        "repository": source_manifest["repository"],
        "revision": source_manifest["revision"],
        "payload_file_count": len(expected),
        "payload_bytes": sum(item["size"] for item in expected.values()),
        "payload_tree_sha256": hashlib.sha256("".join(identity_lines).encode()).hexdigest(),
        "exact_inventory": True,
        "symlinks_and_hardlinks_rejected": True,
    }


def verify_source_tree(root: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Verify a complete BF16 source checkout against the committed pinned inventory."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    manifest_path = _regular_repo_file(
        repo_root,
        "oracle/qwen38/source-manifest.json",
        "pinned source manifest",
    )
    if sha256_file(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ContractError("pinned source manifest hash differs")
    source_manifest = load_json_no_duplicates(manifest_path)
    validate_source_manifest_value(source_manifest)
    return _verify_exact_tree(root, source_manifest)


def main() -> None:
    repo_root_default = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument(
        "--contract",
        type=Path,
        default=repo_root_default / "oracle" / "qwen38" / "contract.json",
    )
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    result = validate_contract(args.repo_root, args.contract)
    if args.source_root is not None:
        result["source_tree"] = verify_source_tree(args.source_root, repo_root=args.repo_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
