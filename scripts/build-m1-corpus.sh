#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:---check}
corpus_dir="$repo_root/benchmarks/m1/corpus"

case "$mode" in
    --check | --write) ;;
    *)
        echo "usage: scripts/build-m1-corpus.sh [--check|--write]" >&2
        exit 64
        ;;
esac

"$repo_root/scripts/verify-oracle.sh"
"$repo_root/scripts/verify-m0-models.sh"
"$repo_root/scripts/verify-m1-e4b-models.sh"

if [[ "$mode" == "--write" ]]; then
    mkdir -p "$corpus_dir"
    exec "$repo_root/oracle/.venv/bin/python" \
        "$repo_root/oracle/build_m1_corpus.py" \
        --repo-root "$repo_root" \
        --output "$corpus_dir"
fi

if [[ ! -d "$corpus_dir" ]]; then
    echo "tracked M1 corpus is missing at $corpus_dir" >&2
    exit 2
fi
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-corpus-check.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
"$repo_root/oracle/.venv/bin/python" \
    "$repo_root/oracle/build_m1_corpus.py" \
    --repo-root "$repo_root" \
    --output "$scratch_dir/generated"
if ! diff -qr "$corpus_dir" "$scratch_dir/generated"; then
    echo "tracked M1 corpus differs from a clean regeneration" >&2
    exit 1
fi

echo "m1-corpus-regeneration: exact"
