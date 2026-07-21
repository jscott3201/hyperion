#!/usr/bin/env python3
"""Run the pinned stock mlx-lm server with an in-process model-load guard."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import server

from model_identity import verify_model_tree


SERVER_SOURCE_SHA256 = "cdfcb4ac848636f9927851a0ec7a951584526530cb7832ba58049e4a9144db8b"
LOAD_RECEIPT_PREFIX = "HYPERION_M1_MODEL_LOAD_IDENTITY "


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifiedModelProvider(server.ModelProvider):
    """Add only an identity guard around the unchanged stock provider load."""

    def __init__(
        self,
        cli_args: argparse.Namespace,
        *,
        model_path: Path,
        manifest_sha256: str,
        payload_tree_sha256: str,
        payload_file_count: int,
    ) -> None:
        super().__init__(cli_args)
        self._verified_path = model_path.resolve()
        self._manifest_sha256 = manifest_sha256
        self._payload_tree_sha256 = payload_tree_sha256
        self._payload_file_count = payload_file_count

    def _load(
        self,
        model_path: str,
        adapter_path: str | None = None,
        draft_model_path: str | None = None,
    ) -> None:
        if (
            Path(model_path).resolve() != self._verified_path
            or adapter_path is not None
            or draft_model_path is not None
        ):
            raise RuntimeError("verified server refused an unregistered model load")
        identity = verify_model_tree(self._verified_path, self._manifest_sha256)
        if (
            identity["payload_tree_sha256"] != self._payload_tree_sha256
            or identity["payload_file_count"] != self._payload_file_count
        ):
            raise RuntimeError("server model identity differs at the stock load boundary")

        # The next operation enters the pinned provider's unchanged load path.
        super()._load(model_path, adapter_path, draft_model_path)
        if verify_model_tree(self._verified_path, self._manifest_sha256) != identity:
            raise RuntimeError("server model identity changed while the stock loader ran")
        print(
            LOAD_RECEIPT_PREFIX + json.dumps(identity, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
            flush=True,
        )


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Verified Hyperion M1 mlx-lm server")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--model-payload-tree-sha256", required=True)
    parser.add_argument("--model-payload-file-count", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--temp", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--chat-template-args", type=json.loads, required=True)
    parser.add_argument("--decode-concurrency", type=int, required=True)
    parser.add_argument("--prompt-concurrency", type=int, required=True)
    parser.add_argument("--prefill-step-size", type=int, required=True)
    parser.add_argument("--prompt-cache-size", type=int, required=True)
    verified = parser.parse_args()
    if (
        verified.host != "127.0.0.1"
        or not 1024 <= verified.port <= 65535
        or verified.temp != 0.0
        or verified.max_tokens != 128
        or verified.chat_template_args != {"enable_thinking": False}
        or verified.decode_concurrency != 1
        or verified.prompt_concurrency != 1
        or verified.prefill_step_size != 2048
        or verified.prompt_cache_size != 0
        or verified.model_payload_file_count <= 0
    ):
        raise RuntimeError("verified server arguments differ from the preregistered smoke")

    stock = argparse.Namespace(
        model=str(verified.model.resolve()),
        adapter_path=None,
        host=verified.host,
        port=verified.port,
        allowed_origins="*",
        draft_model=None,
        num_draft_tokens=3,
        trust_remote_code=False,
        log_level="INFO",
        chat_template="",
        use_default_chat_template=False,
        temp=verified.temp,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=verified.max_tokens,
        chat_template_args=verified.chat_template_args,
        decode_concurrency=verified.decode_concurrency,
        prompt_concurrency=verified.prompt_concurrency,
        prefill_step_size=verified.prefill_step_size,
        prompt_cache_size=verified.prompt_cache_size,
        prompt_cache_bytes=None,
        pipeline=False,
    )
    return verified, vars(stock)


def main() -> None:
    verified, stock_values = parse_args()
    server_path = Path(server.__file__).resolve()
    if sha256_file(server_path) != SERVER_SOURCE_SHA256:
        raise RuntimeError("installed mlx-lm server source differs from the accepted identity")
    stock = argparse.Namespace(**stock_values)
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    provider = VerifiedModelProvider(
        stock,
        model_path=verified.model,
        manifest_sha256=verified.model_manifest_sha256,
        payload_tree_sha256=verified.model_payload_tree_sha256,
        payload_file_count=verified.model_payload_file_count,
    )
    server.run(stock.host, stock.port, provider)


if __name__ == "__main__":
    main()
