#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-budget-sweep.sh RUN_ID}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "M1 run ID contains unsafe characters" >&2
    exit 64
fi
cd "$repo_root"
binary="$repo_root/target/m1-release/release/hyperion-bench"
run_manifest="benchmarks/raw/m1/$run_id/run-manifest.json"
run_root="benchmarks/raw/m1/$run_id/budget"
mkdir -p "$run_root/coarse" "$run_root/refine"

gib=1073741824
failed_points=0
for whole_gib in 4 5 6 7 8 9 10 11; do
    cap_bytes=$((whole_gib * gib))
    if ! "$binary" m1 run-cell \
        --model 12b \
        --context 4096 \
        --arm "discovery-coarse-${whole_gib}g" \
        --wired-limit "$cap_bytes" \
        --run-manifest "$run_manifest" \
        --output "$run_root/coarse/$cap_bytes.jsonl"; then
        failed_points=$((failed_points + 1))
    fi
done

"$binary" m1 select-budget --input-dir "$run_root/coarse" \
    >"$run_root/coarse-selection.json"
while IFS= read -r cap_bytes; do
    if ! "$binary" m1 run-cell \
        --model 12b \
        --context 4096 \
        --arm "discovery-refine-$cap_bytes" \
        --wired-limit "$cap_bytes" \
        --run-manifest "$run_manifest" \
        --output "$run_root/refine/$cap_bytes.jsonl"; then
        failed_points=$((failed_points + 1))
    fi
done < <(jq -r '.refinement_candidates_bytes[]' "$run_root/coarse-selection.json")

"$binary" m1 select-budget --input-dir "$run_root" >"$run_root/selection.json"
selected=$(jq -er '.selected_c_bytes' "$run_root/selection.json")
printf 'm1-budget-pass: selected_c_bytes=%s failed_points=%s evidence=%s\n' \
    "$selected" "$failed_points" "$run_root/selection.json"
