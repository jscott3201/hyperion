#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-stretch.sh RUN_ID 12b|e4b TRIALS}
model=${2:?usage: scripts/run-m1-stretch.sh RUN_ID 12b|e4b TRIALS}
trials=${3:?usage: scripts/run-m1-stretch.sh RUN_ID 12b|e4b TRIALS}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ \
    || ! "$model" =~ ^(12b|e4b)$ \
    || ! "$trials" =~ ^[1-4]$ ]]; then
    echo "M1 128K stretch requires a safe run ID, model 12b|e4b, and one to four trials" >&2
    exit 64
fi
cd "$repo_root"
binary="$repo_root/target/m1-release/release/hyperion-bench"
run_manifest="benchmarks/raw/m1/$run_id/run-manifest.json"
run_root="benchmarks/raw/m1/$run_id/stretch/$model-n$trials"
mkdir -p "$run_root"

"$binary" m1 run-cell \
    --model "$model" \
    --context 131072 \
    --arm stretch-128k \
    --wired-limit default \
    --generated-tokens 129 \
    --trials "$trials" \
    --run-manifest "$run_manifest" \
    --output "$run_root/cell.jsonl"
"$binary" m1 summarize --input-dir "$run_root" >"$run_root/summary.json"
echo "m1-stretch-low-n-pass: model=$model trials=$trials evidence=$run_root"
