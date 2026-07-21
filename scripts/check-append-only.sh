#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
base_ref=${HYPERION_BASE_REF:-origin/development}

if ! git rev-parse --verify --quiet "$base_ref^{commit}" >/dev/null; then
    echo "append-only: required base $base_ref is unavailable" >&2
    exit 1
fi
if [[ "$(git rev-parse "$base_ref^{commit}")" == "$(git rev-parse "HEAD^{commit}")" ]]; then
    echo "append-only: base $base_ref resolves to HEAD; refusing a self-comparison" >&2
    exit 1
fi

scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-append-only.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT

tracked_files=(benchmarks/BENCHMARKS.md eval-results/EVAL_RESULTS.md)
while IFS= read -r decision; do
    if [[ "$decision" == docs/decisions/*.md ]]; then
        tracked_files+=("$decision")
    fi
done < <(git ls-tree -r --name-only "$base_ref" -- docs/decisions)

for path in "${tracked_files[@]}"; do
    if ! git cat-file -e "$base_ref:$path" 2>/dev/null; then
        continue
    fi
    if [[ ! -f "$path" ]]; then
        echo "append-only file was deleted: $path" >&2
        exit 1
    fi
    baseline="$scratch_dir/baseline"
    candidate_prefix="$scratch_dir/candidate-prefix"
    git show "$base_ref:$path" >"$baseline"
    if [[ "$path" == docs/decisions/*.md ]] && \
        ! grep -Eiq '^- Status:[[:space:]]*accepted($|[[:space:];])' "$baseline"; then
        continue
    fi
    baseline_bytes=$(wc -c <"$baseline" | tr -d ' ')
    if [[ "$baseline_bytes" -eq 0 ]]; then
        : >"$candidate_prefix"
    else
        dd if="$path" of="$candidate_prefix" bs="$baseline_bytes" count=1 2>/dev/null
    fi
    if ! cmp -s "$baseline" "$candidate_prefix"; then
        echo "accepted content was modified instead of appended: $path" >&2
        exit 1
    fi
done

echo "append-only: accepted ledger and decision prefixes are unchanged"
