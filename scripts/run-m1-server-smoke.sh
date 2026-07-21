#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-server-smoke.sh RUN_ID}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "M1 run ID contains unsafe characters" >&2
    exit 64
fi
cd "$repo_root"
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "M1 server smoke requires a clean tracked worktree" >&2
    exit 1
fi
output_dir="$repo_root/benchmarks/raw/m1/$run_id/server"
evidence_output_dir="benchmarks/raw/m1/$run_id/server"
run_manifest="$repo_root/benchmarks/raw/m1/$run_id/run-manifest.json"
model_12b=${HYPERION_M1_12B_ORACLE_MODEL:-${HYPERION_M0_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-mlx-g64-b4}}
model_e4b=${HYPERION_M1_E4B_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-e4b-qat-mlx-g64-b4}

"$repo_root/scripts/run-isolated-oracle.sh" \
    script "$repo_root/oracle/m1_server_smoke.py" \
    --repo-root "$repo_root" \
    --model-path "$model_12b" \
    --model-key 12b \
    --model-label gemma-4-12B-QAT-Q4-g64-affine \
    --model-manifest-sha256 9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144 \
    --run-manifest "$run_manifest" \
    --output-dir "$output_dir" \
    --base-port 18080
"$repo_root/scripts/run-isolated-oracle.sh" \
    script "$repo_root/oracle/m1_server_smoke.py" \
    --repo-root "$repo_root" \
    --model-path "$model_e4b" \
    --model-key e4b \
    --model-label gemma-4-E4B-QAT-Q4-g64-affine \
    --model-manifest-sha256 9ba65423d3b2bab1e7c52ea88a1a2b0a33c1f51909b1df66330bf872b7a6c2b0 \
    --run-manifest "$run_manifest" \
    --output-dir "$output_dir" \
    --base-port 18090

jq -s -e 'length == 2 and all(
    .schema == "hyperion.m1-server-smoke.v1" and
    .success == true and
    .deterministic == true and
    .repeat_count == 2 and
    .reasoning_validated_empty == true and
    .usage_validated == true and
    .finish_reasons_validated == true and
    .process_exit_validated == true and
    .quality_floor == false
)' "$output_dir/12b.server-smoke.json" "$output_dir/e4b.server-smoke.json" >/dev/null
echo "m1-server-smoke-pass: $evidence_output_dir"
