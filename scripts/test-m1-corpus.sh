#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
corpus_dir="$repo_root/benchmarks/m1/corpus"

"$repo_root/scripts/verify-m1-corpus.sh"

scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-corpus-negative.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
cp -R "$corpus_dir" "$scratch_dir/corpus"
printf '\000' >>"$scratch_dir/corpus/12b-512.tokens.u32le"
if HYPERION_M1_CORPUS_DIR="$scratch_dir/corpus" \
    "$repo_root/scripts/verify-m1-corpus.sh" >"$scratch_dir/output" 2>&1; then
    echo "M1 corpus negative control accepted a modified token fixture" >&2
    exit 1
fi
if ! rg -q 'token fixture hash mismatch' "$scratch_dir/output"; then
    echo "M1 corpus negative control failed for an unexpected reason" >&2
    sed -n '1,120p' "$scratch_dir/output" >&2
    exit 1
fi

echo "m1-corpus-negative-control: modified token bytes rejected"
