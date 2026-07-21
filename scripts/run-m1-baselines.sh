#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-baselines.sh RUN_ID}
archive_mode=${2:-archive}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "M1 run ID contains unsafe characters" >&2
    exit 64
fi
if [[ "$archive_mode" != archive && "$archive_mode" != --defer-archive ]]; then
    echo "usage: scripts/run-m1-baselines.sh RUN_ID [--defer-archive]" >&2
    exit 64
fi
cd "$repo_root"
run_root="benchmarks/raw/m1/$run_id"
if [[ -e "$run_root" ]]; then
    echo "refusing to overwrite existing M1 run: $run_root" >&2
    exit 1
fi

mkdir -p "$run_root"
scripts/m1-preflight.sh "$run_root"
binary="$repo_root/target/m1-release/release/hyperion-bench"
"$binary" m1 begin-run --output-dir "$run_root" --run-id "$run_id"
scripts/run-m1-core.sh "$run_id"
scripts/run-m1-budget-sweep.sh "$run_id"
candidate_bytes=$(jq -er '.selected_c_bytes' "$run_root/budget/selection.json")
scripts/run-m1-acca.sh "$run_id" "$candidate_bytes"
scripts/run-m1-server-smoke.sh "$run_id"
mkdir -p "$run_root/postflight"
scripts/verify-oracle.sh | tee "$run_root/postflight/oracle-verification.log"
{
    scripts/verify-m0-models.sh
    scripts/verify-m1-e4b-models.sh
} | tee "$run_root/postflight/model-verification.log"
"$binary" m1 verify-run --input-dir "$run_root" >"$run_root/verification.json"
"$binary" m1 summarize --input-dir "$run_root/core" >"$run_root/summary.json"
(
    cd "$run_root"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 >SHA256SUMS
)
if [[ "$archive_mode" == archive ]]; then
    scripts/archive-m1-evidence.sh "$run_id"
else
    echo "m1-archive-deferred: run=$run_id"
fi
printf 'm1-baselines-pass: run=%s manifest_sha256=%s\n' \
    "$run_id" \
    "$(shasum -a 256 "$run_root/SHA256SUMS" | awk '{print $1}')"
