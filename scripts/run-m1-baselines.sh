#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-baselines.sh RUN_ID}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "M1 run ID contains unsafe characters" >&2
    exit 64
fi
cd "$repo_root"
run_root="benchmarks/raw/m1/$run_id"
if [[ -e "$run_root" ]]; then
    echo "refusing to overwrite existing M1 run: $run_root" >&2
    exit 1
fi

scripts/m1-preflight.sh
mkdir -p "$run_root"
scripts/run-m1-core.sh "$run_id"
scripts/run-m1-budget-sweep.sh "$run_id"
candidate_bytes=$(jq -er '.selected_c_bytes' "$run_root/budget/selection.json")
scripts/run-m1-acca.sh "$run_id" "$candidate_bytes"
scripts/run-m1-server-smoke.sh "$run_id"
target/release/hyperion-bench m1 summarize --input-dir "$run_root" >"$run_root/summary.json"
(
    cd "$run_root"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 >SHA256SUMS
)
printf 'm1-baselines-pass: run=%s manifest_sha256=%s\n' \
    "$run_id" \
    "$(shasum -a 256 "$run_root/SHA256SUMS" | awk '{print $1}')"
