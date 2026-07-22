#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-nightly.sh RUN_ID}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "M1 nightly run ID contains unsafe characters" >&2
    exit 64
fi
cd "$repo_root"
binary="$repo_root/target/m1-release/release/hyperion-bench"
run_manifest="benchmarks/raw/m1/$run_id/run-manifest.json"
for model in 12b e4b; do
    "$binary" m1 run-cell \
        --model "$model" \
        --context 512 \
        --arm nightly-512x128 \
        --wired-limit default \
        --generated-tokens 129 \
        --run-manifest "$run_manifest" \
        --output "benchmarks/raw/m1/$run_id/nightly/$model-512x128.jsonl"
done
"$binary" m1 summarize --input-dir "benchmarks/raw/m1/$run_id/nightly" \
    >"benchmarks/raw/m1/$run_id/nightly-summary.json"
echo "m1-nightly-pass: benchmarks/raw/m1/$run_id/nightly-summary.json"
