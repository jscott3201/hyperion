#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-acca.sh RUN_ID C_BYTES}
candidate_bytes=${2:?usage: scripts/run-m1-acca.sh RUN_ID C_BYTES}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ || ! "$candidate_bytes" =~ ^[0-9]+$ ]]; then
    echo "M1 A-C-C-A arguments are invalid" >&2
    exit 64
fi
cd "$repo_root"
binary="$repo_root/target/release/hyperion-bench"
run_root="benchmarks/raw/m1/$run_id/acca"
mkdir -p "$run_root"

"$binary" m1 run-cell --model 12b --context 4096 --arm acca-a1 \
    --wired-limit default --output "$run_root/01-a1.jsonl"
"$binary" m1 run-cell --model 12b --context 4096 --arm acca-c1 \
    --wired-limit "$candidate_bytes" --output "$run_root/02-c1.jsonl"
"$binary" m1 run-cell --model 12b --context 4096 --arm acca-c2 \
    --wired-limit "$candidate_bytes" --output "$run_root/03-c2.jsonl"
"$binary" m1 run-cell --model 12b --context 4096 --arm acca-a2 \
    --wired-limit default --output "$run_root/04-a2.jsonl"

"$binary" m1 check-acca --input-dir "$run_root" >"$run_root/confirmation.json"
echo "m1-acca-pass: $run_root/confirmation.json"
