#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
binary="$repo_root/target/debug/hyperion-bench"
fixture="$repo_root/benchmarks/m1/fixtures/valid-cell.jsonl"
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-harness.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT

python3 oracle/build_m1_synthetic_fixtures.py \
    --output "$scratch_dir/valid-cell.jsonl"
cmp "$fixture" "$scratch_dir/valid-cell.jsonl"
cmp "${fixture%.jsonl}.stderr.log" "$scratch_dir/valid-cell.stderr.log"
"$binary" m1 verify-trace "$fixture" >/dev/null

python3 oracle/mutate_m1_synthetic_fixture.py \
    --input "$fixture" --output "$scratch_dir/controlled.jsonl" \
    --mutation controlled-failure
"$binary" m1 verify-trace "$scratch_dir/controlled.jsonl" >/dev/null

for mutation in \
    missing-controller-start \
    missing-pre-boundary \
    cross-trial-hash \
    oracle-tree-drift \
    controller-envelope-drift \
    worker-source-drift \
    os-summary-drift \
    missing-warmup-boundary \
    controller-sample-count \
    trial-event-reorder \
    missing-stderr \
    stderr-hash-drift \
    worker-exit-drift \
    non-capacity-failure \
    capacity-protocol-error \
    alloc-substring-failure \
    traceback-non-string \
    failure-missing-field \
    failure-extra-field \
    sigkill-relabel \
    uncontrolled-oom; do
    candidate="$scratch_dir/$mutation.jsonl"
    python3 oracle/mutate_m1_synthetic_fixture.py \
        --input "$fixture" --output "$candidate" --mutation "$mutation"
    if "$binary" m1 verify-trace "$candidate" >"$scratch_dir/$mutation.out" 2>&1; then
        echo "M1 harness negative control accepted $mutation" >&2
        exit 1
    fi
done

echo "m1-harness-model-free-controls: pass"
