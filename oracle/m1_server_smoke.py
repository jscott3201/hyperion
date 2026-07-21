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


SCHEMA = "hyperion.m1-server-smoke.v1"
SERVER_SHA256 = "cdfcb4ac848636f9927851a0ec7a951584526530cb7832ba58049e4a9144db8b"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-key", choices=("12b", "e4b"), required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
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


def stream_request(port: int, body: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    started = time.perf_counter_ns()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=encoded,
        headers={"Content-Type": "application/json", "Content-Length": str(len(encoded))},
    )
    response = connection.getresponse()
    if response.status != 200:
        payload = response.read().decode(errors="replace")
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
        raw_lines.append(line)
        if line.startswith(":"):
            continue
        if not line.startswith("data: "):
            raise RuntimeError(f"unexpected SSE line: {line}")
        data = line[6:]
        if data == "[DONE]":
            saw_done = True
            break
        emissions_ns.append(time.perf_counter_ns() - started)
        chunks.append(json.loads(data))
    connection.close()
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
            reasoning += delta.get("reasoning_content") or ""
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
        "first_finish_reasons": first["assembled"]["finish_reasons"],
        "final_content": second["assembled"]["content"],
        "final_finish_reasons": second["assembled"]["finish_reasons"],
    }


def run_repeat(args: argparse.Namespace, repeat: int, port: int) -> dict[str, Any]:
    log_path = args.output_dir / f"{args.model_key}-repeat-{repeat + 1}.server.log"
    with log_path.open("xb") as log:
        command = [
            str(args.repo_root / "oracle/.venv/bin/mlx_lm.server"),
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
        process = subprocess.Popen(
            command,
            cwd=args.repo_root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_health(port, process)
            warmup = stream_request(port, request_body([USER_MESSAGE]))
            first = stream_request(port, request_body([USER_MESSAGE]))
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
            )
            if second["assembled"]["tool_calls"]:
                raise RuntimeError("final turn unexpectedly emitted another tool call")
            if not second["assembled"]["content"].strip():
                raise RuntimeError("final turn emitted no user-visible content")
            canonical = canonical_result(first, second)
            return {
                "repeat": repeat + 1,
                "port": port,
                "command": command,
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


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 1024 <= args.base_port <= 65533:
        raise RuntimeError("base port must leave room for two loopback repeats")
    actual_manifest = sha256_file(args.model_path / "SHA256SUMS")
    if actual_manifest != args.model_manifest_sha256:
        raise RuntimeError("server-smoke model manifest differs from the accepted identity")
    server_path = args.repo_root / "oracle/.venv/lib/python3.12/site-packages/mlx_lm/server.py"
    if sha256_file(server_path) != SERVER_SHA256:
        raise RuntimeError("installed mlx-lm server source differs from the accepted identity")

    try:
        repeats = [run_repeat(args, repeat, args.base_port + repeat) for repeat in range(2)]
    except BaseException as error:
        failure_path = args.output_dir / f"{args.model_key}.server-smoke.failure.json"
        with failure_path.open("x", encoding="utf-8") as output:
            json.dump(
                {
                    "schema": SCHEMA,
                    "source_commit": args.source_commit,
                    "model_key": args.model_key,
                    "model_manifest_sha256": actual_manifest,
                    "success": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc().splitlines(),
                },
                output,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
        raise
    canonical_sha256 = [
        hashlib.sha256(
            json.dumps(item["canonical"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for item in repeats
    ]
    deterministic = len(set(canonical_sha256)) == 1
    if not deterministic:
        raise RuntimeError("fresh-server greedy smoke repeats produced different canonical loops")
    result = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "model_key": args.model_key,
        "model_label": args.model_label,
        "model_manifest_sha256": actual_manifest,
        "server_source_sha256": SERVER_SHA256,
        "worker_sha256": sha256_file(Path(__file__).resolve()),
        "thinking_enabled": False,
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
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"M1_SERVER_SMOKE_PASS model={args.model_key} output={output_path}")


if __name__ == "__main__":
    main()
