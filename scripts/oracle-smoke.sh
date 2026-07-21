#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${HYPERION_M0_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-mlx-g64-b4}
generator="$repo_root/oracle/.venv/bin/mlx_lm.generate"

if [[ ! -x "$generator" ]]; then
    echo "oracle environment is missing; run scripts/setup-oracle.sh" >&2
    exit 2
fi
if [[ ! -f "$model_dir/config.json" ]]; then
    echo "M0 oracle model is missing at $model_dir" >&2
    exit 2
fi

prompt=$(tr -d '\r\n' <"$repo_root/oracle/m0-prompt.txt")
response=$(
    "$generator" \
        --model "$model_dir" \
        --prompt "$prompt" \
        --max-tokens 128 \
        --temp 0.0 \
        --seed 0 \
        --chat-template-config '{"enable_thinking":false}' \
        --verbose False
)
if [[ "$response" != "HYPERION_M0_OK" ]]; then
    echo "oracle generated a response but did not exactly match the frozen M0 marker" >&2
    printf '%s\n' "$response" >&2
    exit 1
fi

printf 'MEASURED {"schema":"hyperion.oracle-smoke.v1","model":"gemma-4-12B-QAT-Q4-g64-affine","prompt":"HYPERION_M0_OK","generated":true}\n'
printf '%s\n' "$response"
