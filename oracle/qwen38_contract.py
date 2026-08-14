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


CONTRACT_SCHEMA = "hyperion.qwen38-oracle-contract.v1"
SOURCE_SCHEMA = "hyperion.qwen38-source-manifest.v1"
VALIDATION_SCHEMA = "hyperion.qwen38-oracle-contract-validation.v1"
SOURCE_REPOSITORY = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_MANIFEST_SHA256 = "450ff5ada441702f54e8da2cd3c92a207ab1b3bdbaa6254427556d9afa4ff91a"
CASES_SHA256 = "2a1c23a1baa9ea8566fce71d452272202eddf71e83190c930389fce94097fda3"
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
    "input_ids",
    "position_ids",
    "full_vocab_logits_f32",
    "topk_ids_values_and_margins",
    "greedy_token_ids",
    "free_running_utf8",
    "selected_hidden_states",
    "selected_gdn_recurrent_state_f32",
    "selected_gdn_conv_state",
    "selected_full_attention_kv",
    "instrumented_uninstrumented_control",
]

REQUIRED_TRACE_BOUNDARIES = [
    "initial",
    "post_prefill",
    "post_decode",
    "token_step",
    "chunk_before_conv_width",
    "chunk_at_conv_width",
    "chunk_after_conv_width",
]

REQUIRED_EVIDENCE_IDENTITIES = [
    "source_manifest",
    "source_tree",
    "oracle_implementation",
    "oracle_environment",
    "python_runtime",
    "device_backend",
    "execution_mode",
    "case_corpus",
    "conversation_profile",
    "trace_schema",
    "producer_command",
    "producer_executable",
    "machine_profile",
    "raw_payload_inventory",
    "comparison_harness",
]

EXPECTED_ORACLES = [
    {
        "id": "transformers",
        "repository": "huggingface/transformers",
        "revision": "95940bf8775059a42f047256f076e4f607bc43ec",
        "implementation_path": "src/transformers/models/qwen3_5/modeling_qwen3_5.py",
        "implementation_sha256": "90d929129ffc835d2652c604925c4f3842bc6e401e174ec6f0db2285dfb8f85a",
        "state": "source_pinned_unexecuted",
        "execution_mode_status": "partial_intent_environment_and_producer_unfrozen",
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
        "execution_mode_status": "partial_intent_environment_and_producer_unfrozen",
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


def _validate_contract_value(repo_root: Path, contract: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = _exact_keys(
        contract,
        {
            "schema",
            "status",
            "source",
            "component",
            "oracles",
            "conversation",
            "required_coverage",
            "trace",
            "required_evidence_identities",
        },
        "oracle contract",
    )
    if contract["schema"] != CONTRACT_SCHEMA:
        raise ContractError("oracle contract schema differs")
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
        if oracle["execution_mode_status"] != "partial_intent_environment_and_producer_unfrozen":
            raise ContractError(f"oracle {oracle_id} overclaims a frozen execution mode")
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
        {"state", "required_channels", "required_boundaries", "bundle_rules", "tolerances"},
        "trace contract",
    )
    if trace["state"] != "channel_requirements_preregistered_no_payload":
        raise ContractError("trace contract claims a payload")
    if _string_list(trace["required_channels"], "trace required channels") != REQUIRED_TRACE_CHANNELS:
        raise ContractError("trace channels differ from the reviewed contract")
    if _string_list(trace["required_boundaries"], "trace required boundaries") != REQUIRED_TRACE_BOUNDARIES:
        raise ContractError("trace boundaries differ from the reviewed contract")
    expected_rules = {
        "raw_oracle_outputs_separate": True,
        "producer_authored_pass_ignored": True,
        "third_party_comparison_required": True,
        "exact_inventory_required": True,
        "external_inventory_digest_required": True,
        "atomic_publication_required": True,
        "source_verified_before_and_after": True,
    }
    if trace["bundle_rules"] != expected_rules:
        raise ContractError("trace bundle rules differ")
    if trace["tolerances"] != {"status": "unfrozen_requires_clean_and_fault_measurement"}:
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
        "case_count": len(cases),
        "oracle_ids": sorted(oracle["id"] for oracle in validated["oracles"]),
        "contract_only_unexecuted": True,
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
