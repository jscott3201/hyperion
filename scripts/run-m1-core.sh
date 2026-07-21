#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-core.sh RUN_ID}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "M1 run ID contains unsafe characters" >&2
    exit 64
fi
cd "$repo_root"
binary="$repo_root/target/release/hyperion-bench"
if [[ ! -x "$binary" ]]; then
    echo "release benchmark binary is missing; run scripts/m1-preflight.sh" >&2
    exit 2
fi

for model in 12b e4b; do
    for context in 512 1024 4096 8192 16384 32768; do
        output="benchmarks/raw/m1/$run_id/core/$model-$context.jsonl"
        "$binary" m1 run-cell \
            --model "$model" \
            --context "$context" \
            --arm core-default \
            --wired-limit default \
            --output "$output"
    done
done

"$binary" m1 summarize --input-dir "benchmarks/raw/m1/$run_id/core" \
    >"benchmarks/raw/m1/$run_id/core-summary.json"
echo "m1-core-pass: benchmarks/raw/m1/$run_id/core-summary.json"
