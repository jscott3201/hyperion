#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
fixture="$repo_root/benchmarks/m1/fixtures/valid-cell.jsonl"
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-harness.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT

oracle_lock_sha256=$(shasum -a 256 "$repo_root/oracle/uv.lock" | awk '{print $1}')
if ! rg -Fq \
    "const ORACLE_LOCK_SHA256: &str = \"$oracle_lock_sha256\";" \
    crates/hyperion-bench/src/m1.rs; then
    echo "M1 Rust controller oracle lock pin is stale" >&2
    exit 1
fi
if ! rg -Fq \
    "ORACLE_LOCK_SHA256 = \"$oracle_lock_sha256\"" \
    oracle/build_m1_synthetic_fixtures.py; then
    echo "M1 synthetic fixture generator oracle lock pin is stale" >&2
    exit 1
fi
if ! rg -Fq \
    "expected_lock_sha256=$oracle_lock_sha256" \
    scripts/verify-oracle.sh; then
    echo "oracle verifier lock pin is stale" >&2
    exit 1
fi
if ! jq -e --arg lock "$oracle_lock_sha256" \
    '.oracle.lock_sha256 == $lock' benchmarks/m1/schedule.json >/dev/null; then
    echo "M1 schedule oracle lock pin is stale" >&2
    exit 1
fi
if ! jq -s -e --arg lock "$oracle_lock_sha256" \
    '[.[] | select(has("oracle_lock_sha256")) | .oracle_lock_sha256] == [$lock]' \
    "$fixture" >/dev/null; then
    echo "M1 synthetic fixture oracle lock receipt is stale" >&2
    exit 1
fi

cargo build --locked -p hyperion-bench
target_dir=$(cargo metadata --locked --no-deps --format-version 1 | jq -er '.target_directory')
binary="$target_dir/debug/hyperion-bench"
if [[ ! -x "$binary" ]]; then
    echo "M1 harness build did not produce $binary" >&2
    exit 1
fi

python3 -B oracle/test_m1_worker_envelope.py

python3 -B oracle/build_m1_synthetic_fixtures.py \
    --output "$scratch_dir/valid-cell.jsonl"
cmp "$fixture" "$scratch_dir/valid-cell.jsonl"
cmp "${fixture%.jsonl}.stderr.log" "$scratch_dir/valid-cell.stderr.log"
"$binary" m1 verify-trace "$fixture" >/dev/null

python3 -B oracle/mutate_m1_synthetic_fixture.py \
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
    worker-environment-drift \
    worker-platform-drift \
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
    python3 -B oracle/mutate_m1_synthetic_fixture.py \
        --input "$fixture" --output "$candidate" --mutation "$mutation"
    if "$binary" m1 verify-trace "$candidate" >"$scratch_dir/$mutation.out" 2>&1; then
        echo "M1 harness negative control accepted $mutation" >&2
        exit 1
    fi
done

echo "m1-harness-model-free-controls: pass"
