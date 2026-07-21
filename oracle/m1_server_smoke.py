#!/usr/bin/env python3
"""Run the preregistered stock mlx-lm loopback tool-use smoke."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from oracle_identity import verify_identity


SCHEMA = "hyperion.m1-server-smoke.v1"
RUN_MANIFEST_SCHEMA = "hyperion.m1-run-manifest.v1"
SERVER_SOURCE_SHA256 = "cdfcb4ac848636f9927851a0ec7a951584526530cb7832ba58049e4a9144db8b"
SERVER_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TZ": "UTC",
}
FILTER = "site:HQ AND equip:AHU-01"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_points",
            "description": "Return normalized building points matching a bounded filter.",
            "parameters": {
                "type": "object",
                "properties": {"filter": {"type": "string"}},
                "required": ["filter"],
                "additionalProperties": False,
            },
        },
    }
]
USER_MESSAGE = {
    "role": "user",
    "content": (
        "Deterministic interoperability check. Call get_points exactly once with filter "
        f"{FILTER!r}. Do not answer in prose before the tool result."
    ),
}
TOOL_RESULT = [
    {
        "id": "hq.ahu01.sat",
        "label": "AHU-01 Supply Air Temperature",
        "kind": "analogInput",
        "unit": "degF",
        "tags": ["site:HQ", "equip:AHU-01", "measurement:supply-air-temperature"],
    }
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def write_journal_line(output: Any, value: dict[str, Any]) -> None:
    output.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    output.flush()


def sanitize_text(text: str, repo_root: Path, model_path: Path) -> str:
    for original, replacement in (
        (str(model_path), "<MODEL>"),
        (str(repo_root), "<REPO>"),
        (str(Path.home()), "<HOME>"),
    ):
        text = text.replace(original, replacement)
    return text


def sanitize_server_log(path: Path, repo_root: Path, model_path: Path) -> None:
    text = sanitize_text(
        path.read_text(encoding="utf-8", errors="replace"), repo_root, model_path
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-key", choices=("12b", "e4b"), required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=18080)
    return parser.parse_args()


def request_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "default_model",
        "messages": messages,
        "tools": TOOLS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def stream_request(port: int, body: dict[str, Any], journal_path: Path) -> dict[str, Any]:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with journal_path.open("x", encoding="utf-8") as journal:
        write_journal_line(
            journal,
            {"kind": "request", "unix_ns": time.time_ns(), "body": body},
        )
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        started = time.perf_counter_ns()
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=encoded,
            headers={"Content-Type": "application/json", "Content-Length": str(len(encoded))},
        )
        response = connection.getresponse()
        write_journal_line(
            journal,
            {
                "kind": "response_start",
                "unix_ns": time.time_ns(),
                "status": response.status,
                "content_type": response.getheader("Content-Type"),
                "cache_control": response.getheader("Cache-Control"),
            },
        )
        if response.status != 200:
            payload = response.read().decode(errors="replace")
            write_journal_line(journal, {"kind": "http_failure", "payload": payload})
            raise RuntimeError(f"server returned HTTP {response.status}: {payload}")

        chunks: list[dict[str, Any]] = []
        emissions_ns: list[int] = []
        raw_lines: list[str] = []
        saw_done = False
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode(errors="strict").rstrip("\r\n")
            if not line:
                continue
            offset_ns = time.perf_counter_ns() - started
            raw_lines.append(line)
            write_journal_line(
                journal,
                {"kind": "sse_line", "emission_offset_ns": offset_ns, "line": line},
            )
            if line.startswith(":"):
                continue
            if not line.startswith("data: "):
                raise RuntimeError(f"unexpected SSE line: {line}")
            data = line[6:]
            if data == "[DONE]":
                saw_done = True
                break
            emissions_ns.append(offset_ns)
            chunks.append(json.loads(data))
        connection.close()
        write_journal_line(
            journal,
            {"kind": "response_end", "unix_ns": time.time_ns(), "saw_done": saw_done},
        )
    if not saw_done:
        raise RuntimeError("stream ended without [DONE]")

    content = ""
    reasoning = ""
    tool_calls: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    usage = None
    for chunk in chunks:
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
            tool_calls.extend(delta.get("tool_calls") or [])
            if choice.get("finish_reason") is not None:
                finish_reasons.append(choice["finish_reason"])
    inter_emission_ns = [
        later - earlier for earlier, later in zip(emissions_ns, emissions_ns[1:])
    ]
    return {
        "request": body,
        "status": response.status,
        "headers": {
            "content-type": response.getheader("Content-Type"),
            "cache-control": response.getheader("Cache-Control"),
        },
        "raw_lines": raw_lines,
        "chunks": chunks,
        "emission_offsets_ns": emissions_ns,
        "http_inter_emission_ns": inter_emission_ns,
        "assembled": {
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "finish_reasons": finish_reasons,
            "usage": usage,
        },
    }


def validate_stream(result: dict[str, Any], *, expect_tool_call: bool) -> None:
    assembled = result["assembled"]
    if assembled["reasoning_content"]:
        raise RuntimeError("thinking-disabled server response emitted reasoning")
    usage = assembled["usage"]
    if not isinstance(usage, dict):
        raise RuntimeError("stream omitted the requested usage object")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(usage.get(field), int) or usage[field] < 0:
            raise RuntimeError(f"stream usage field {field} is invalid")
    reasons = assembled["finish_reasons"]
    if not reasons:
        raise RuntimeError("stream emitted no finish reason")
    expected = {"tool_calls", "stop"} if expect_tool_call else {"stop"}
    if reasons[-1] not in expected:
        raise RuntimeError(f"unexpected terminal finish reason: {reasons[-1]!r}")


def wait_for_health(port: int, process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"mlx-lm server exited during startup with {process.returncode}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.25)
    raise RuntimeError("timed out waiting for mlx-lm server health")


def validate_tool_call(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tool_calls) != 1:
        raise RuntimeError(f"expected exactly one tool call, received {len(tool_calls)}")
    call = tool_calls[0]
    function = call.get("function") or {}
    if function.get("name") != "get_points":
        raise RuntimeError(f"unexpected tool function: {function.get('name')!r}")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if arguments != {"filter": FILTER}:
        raise RuntimeError(f"unexpected get_points arguments: {arguments!r}")
    if not isinstance(call.get("id"), str) or not call["id"]:
        raise RuntimeError("tool call has no non-empty ID")
    return call


def canonical_result(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    tool_call = first["assembled"]["tool_calls"][0]
    function = tool_call["function"]
    arguments = function["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return {
        "tool_call": {
            "id": "call_1",
            "type": tool_call.get("type"),
            "function": {"name": function["name"], "arguments": arguments},
        },
        "first_content": first["assembled"]["content"],
        "first_reasoning": first["assembled"]["reasoning_content"],
        "first_finish_reasons": first["assembled"]["finish_reasons"],
        "first_usage": first["assembled"]["usage"],
        "final_content": second["assembled"]["content"],
        "final_reasoning": second["assembled"]["reasoning_content"],
        "final_finish_reasons": second["assembled"]["finish_reasons"],
        "final_usage": second["assembled"]["usage"],
    }


def run_repeat(args: argparse.Namespace, repeat: int, port: int) -> dict[str, Any]:
    log_path = args.output_dir / f"{args.model_key}-repeat-{repeat + 1}.server.log"
    with log_path.open("xb") as log:
        command = [
            str(args.repo_root / "oracle/.venv/bin/python"),
            "-m",
            "mlx_lm.server",
            "--model",
            str(args.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--temp",
            "0",
            "--max-tokens",
            "128",
            "--chat-template-args",
            '{"enable_thinking":false}',
            "--decode-concurrency",
            "1",
            "--prompt-concurrency",
            "1",
            "--prefill-step-size",
            "2048",
            "--prompt-cache-size",
            "0",
        ]
        evidence_command = [
            "oracle/.venv/bin/python" if item == command[0] else item
            for item in command
        ]
        evidence_command = [
            "<MODEL>" if item == str(args.model_path) else item for item in evidence_command
        ]
        process = subprocess.Popen(
            command,
            cwd=args.repo_root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=SERVER_ENVIRONMENT,
        )
        result: dict[str, Any] | None = None
        try:
            wait_for_health(port, process)
            prefix = f"{args.model_key}-repeat-{repeat + 1}"
            warmup = stream_request(
                port,
                request_body([USER_MESSAGE]),
                args.output_dir / f"{prefix}.warmup.jsonl",
            )
            validate_stream(warmup, expect_tool_call=True)
            first = stream_request(
                port,
                request_body([USER_MESSAGE]),
                args.output_dir / f"{prefix}.tool-call.jsonl",
            )
            validate_stream(first, expect_tool_call=True)
            call = validate_tool_call(first["assembled"]["tool_calls"])
            assistant_message = {
                "role": "assistant",
                "content": first["assembled"]["content"] or None,
                "tool_calls": [call],
            }
            tool_message = {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": "get_points",
                "content": json.dumps(TOOL_RESULT, sort_keys=True, separators=(",", ":")),
            }
            second = stream_request(
                port,
                request_body([USER_MESSAGE, assistant_message, tool_message]),
                args.output_dir / f"{prefix}.final.jsonl",
            )
            validate_stream(second, expect_tool_call=False)
            if second["assembled"]["tool_calls"]:
                raise RuntimeError("final turn unexpectedly emitted another tool call")
            if not second["assembled"]["content"].strip():
                raise RuntimeError("final turn emitted no user-visible content")
            canonical = canonical_result(first, second)
            result = {
                "repeat": repeat + 1,
                "port": port,
                "command": evidence_command,
                "warmup": warmup,
                "first_turn": first,
                "tool_result": TOOL_RESULT,
                "final_turn": second,
                "canonical": canonical,
                "server_log": log_path.name,
            }
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            log.flush()
            os.fsync(log.fileno())
            sanitize_server_log(log_path, args.repo_root, args.model_path)
        if result is None:
            raise RuntimeError("server repeat completed without a result")
        if process.returncode not in (0, -signal.SIGTERM, -signal.SIGKILL):
            raise RuntimeError(f"server exited unexpectedly with {process.returncode}")
        prefix = f"{args.model_key}-repeat-{repeat + 1}"
        journals = {
            turn: {
                "file": f"{prefix}.{suffix}.jsonl",
                "sha256": sha256_file(args.output_dir / f"{prefix}.{suffix}.jsonl"),
            }
            for turn, suffix in (
                ("warmup", "warmup"),
                ("first_turn", "tool-call"),
                ("final_turn", "final"),
            )
        }
        result["server_exit"] = {
            "returncode": process.returncode,
            "expected_termination": process.returncode in (-signal.SIGTERM, -signal.SIGKILL),
        }
        result["server_log_sha256"] = sha256_file(log_path)
        result["journals"] = journals
        write_json_exclusive(
            args.output_dir / f"{args.model_key}-repeat-{repeat + 1}.json",
            result,
        )
        return result


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.model_path = args.model_path.resolve()
    args.run_manifest = args.run_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_unix_ns = time.time_ns()
    try:
        if not 1024 <= args.base_port <= 65533:
            raise RuntimeError("base port must leave room for two loopback repeats")
        run_manifest = json.loads(args.run_manifest.read_text())
        run_manifest_sha256 = sha256_file(args.run_manifest)
        if run_manifest.get("schema") != RUN_MANIFEST_SCHEMA:
            raise RuntimeError("server-smoke run manifest schema is invalid")
        if args.output_dir != args.run_manifest.parent / "server":
            raise RuntimeError("server-smoke output is not bound to the run-manifest root")
        if sha256_file(Path(__file__).resolve()) != run_manifest.get("server_worker_sha256"):
            raise RuntimeError("server-smoke worker differs from the run manifest")
        model_receipt = args.run_manifest.parent / "preflight/model-verification.log"
        if sha256_file(model_receipt) != run_manifest.get("model_verification_pre_sha256"):
            raise RuntimeError("server-smoke model receipt differs from the run manifest")
        oracle_identity = verify_identity()
        actual_manifest = sha256_file(args.model_path / "SHA256SUMS")
        if actual_manifest != args.model_manifest_sha256:
            raise RuntimeError("server-smoke model manifest differs from the accepted identity")
        server_path = (
            args.repo_root
            / "oracle/.venv/lib/python3.12/site-packages/mlx_lm/server.py"
        )
        server_source_sha256 = sha256_file(server_path)
        if server_source_sha256 != SERVER_SOURCE_SHA256:
            raise RuntimeError("installed mlx-lm server source differs from the accepted identity")
        repeats = [run_repeat(args, repeat, args.base_port + repeat) for repeat in range(2)]
        canonical_sha256 = [
            hashlib.sha256(
                json.dumps(
                    item["canonical"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            for item in repeats
        ]
        deterministic = len(set(canonical_sha256)) == 1
        if not deterministic:
            raise RuntimeError(
                "fresh-server greedy smoke repeats produced different canonical loops"
            )
        result = {
            "schema": SCHEMA,
            "success": True,
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": time.time_ns(),
            "source_commit": run_manifest["source_commit"],
            "run_id": run_manifest["run_id"],
            "run_manifest_sha256": run_manifest_sha256,
            "model_verification_pre_sha256": run_manifest[
                "model_verification_pre_sha256"
            ],
            "model_key": args.model_key,
            "model_label": args.model_label,
            "model_manifest_sha256": actual_manifest,
            "server_source_sha256": server_source_sha256,
            "worker_sha256": sha256_file(Path(__file__).resolve()),
            "oracle_identity": oracle_identity,
            "server_environment": SERVER_ENVIRONMENT,
            "thinking_enabled": False,
            "reasoning_validated_empty": True,
            "usage_validated": True,
            "finish_reasons_validated": True,
            "process_exit_validated": True,
            "temperature": 0.0,
            "seed": 0,
            "prompt_cache_size": 0,
            "decode_concurrency": 1,
            "prompt_concurrency": 1,
            "prefill_step_size": 2048,
            "repeat_count": len(repeats),
            "canonical_sha256": canonical_sha256,
            "deterministic": deterministic,
            "quality_floor": False,
            "http_inter_emission_is_token_itl": False,
            "repeats": repeats,
        }
        output_path = args.output_dir / f"{args.model_key}.server-smoke.json"
        write_json_exclusive(output_path, result)
    except BaseException as error:
        failure_path = args.output_dir / f"{args.model_key}.server-smoke.failure.json"
        write_json_exclusive(
            failure_path,
            {
                "schema": SCHEMA,
                "started_unix_ns": started_unix_ns,
                "finished_unix_ns": time.time_ns(),
                "run_manifest": "run-manifest.json",
                "run_manifest_sha256": (
                    sha256_file(args.run_manifest) if args.run_manifest.is_file() else None
                ),
                "model_key": args.model_key,
                "model_manifest_sha256": args.model_manifest_sha256,
                "success": False,
                "error_type": type(error).__name__,
                "message": sanitize_text(str(error), args.repo_root, args.model_path),
                "traceback": sanitize_text(
                    traceback.format_exc(), args.repo_root, args.model_path
                ).splitlines(),
            },
        )
        raise SystemExit(1) from None
    print(
        "M1_SERVER_SMOKE_PASS "
        f"model={args.model_key} output={output_path.relative_to(args.repo_root)}"
    )


if __name__ == "__main__":
    main()
