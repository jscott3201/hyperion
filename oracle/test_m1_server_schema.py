#!/usr/bin/env python3
"""Model-free live-validator controls for nested M1 server SSE schemas."""

from __future__ import annotations

import copy
import sys
import types
import unittest


identity_module = types.ModuleType("_hyperion_isolated_identity")
identity_module.ORACLE_STARTUP_IDENTITY = {}
sys.modules[identity_module.__name__] = identity_module

from m1_server_smoke import validate_stream  # noqa: E402


def valid_result() -> dict:
    tool_call = {
        "index": 0,
        "id": "call-synthetic",
        "type": "function",
        "function": {
            "name": "get_points",
            "arguments": '{"filter":"site:HQ AND equip:AHU-01"}',
        },
    }
    return {
        "chunks": [
            {
                "id": "chatcmpl-synthetic",
                "system_fingerprint": "mlx-lm-0.31.3",
                "object": "chat.completion.chunk",
                "model": "default_model",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "delta": {"role": "assistant", "tool_calls": [tool_call]},
                    }
                ],
            },
            {
                "id": "chatcmpl-synthetic",
                "system_fingerprint": "mlx-lm-0.31.3",
                "object": "chat.completion",
                "model": "default_model",
                "created": 1,
                "choices": [],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            },
        ],
        "assembled": {
            "reasoning_content": "",
            "finish_reasons": ["tool_calls"],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        },
    }


class ServerSchemaTests(unittest.TestCase):
    def test_valid_nested_tool_call(self) -> None:
        validate_stream(valid_result(), expect_tool_call=True)

    def assert_nested_rejected(self, mutate) -> None:
        candidate = copy.deepcopy(valid_result())
        call = candidate["chunks"][0]["choices"][0]["delta"]["tool_calls"][0]
        mutate(call)
        with self.assertRaises(RuntimeError):
            validate_stream(candidate, expect_tool_call=True)

    def test_extra_tool_call_field_rejected(self) -> None:
        self.assert_nested_rejected(lambda call: call.__setitem__("extra", True))

    def test_extra_function_field_rejected(self) -> None:
        self.assert_nested_rejected(
            lambda call: call["function"].__setitem__("extra", True)
        )

    def test_mistyped_tool_call_index_rejected(self) -> None:
        self.assert_nested_rejected(lambda call: call.__setitem__("index", "0"))

    def test_mistyped_function_arguments_rejected(self) -> None:
        self.assert_nested_rejected(
            lambda call: call["function"].__setitem__("arguments", {})
        )


if __name__ == "__main__":
    unittest.main()
