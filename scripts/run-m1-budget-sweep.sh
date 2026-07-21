#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/run-m1-budget-sweep.sh RUN_ID}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "M1 run ID contains unsafe characters" >&2
    exit 64
fi
cd "$repo_root"
binary="$repo_root/target/release/hyperion-bench"
run_root="benchmarks/raw/m1/$run_id/budget"
mkdir -p "$run_root/coarse" "$run_root/refine"

gib=1073741824
for whole_gib in 4 5 6 7 8 9 10 11; do
    cap_bytes=$((whole_gib * gib))
    "$binary" m1 run-cell \
        --model 12b \
        --context 4096 \
        --arm "discovery-coarse-${whole_gib}g" \
        --wired-limit "$cap_bytes" \
        --output "$run_root/coarse/$cap_bytes.jsonl"
done

"$binary" m1 select-budget --input-dir "$run_root/coarse" \
    >"$run_root/coarse-selection.json"
while IFS= read -r cap_bytes; do
    "$binary" m1 run-cell \
        --model 12b \
        --context 4096 \
        --arm "discovery-refine-$cap_bytes" \
        --wired-limit "$cap_bytes" \
        --output "$run_root/refine/$cap_bytes.jsonl"
done < <(jq -r '.refinement_candidates_bytes[]' "$run_root/coarse-selection.json")

"$binary" m1 select-budget --input-dir "$run_root" >"$run_root/selection.json"
selected=$(jq -er '.selected_c_bytes' "$run_root/selection.json")
printf 'm1-budget-pass: selected_c_bytes=%s evidence=%s\n' "$selected" "$run_root/selection.json"
