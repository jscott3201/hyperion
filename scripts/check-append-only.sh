#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
base_ref=${HYPERION_BASE_REF:-origin/development}

if ! git rev-parse --verify --quiet "$base_ref^{commit}" >/dev/null; then
    echo "append-only: base $base_ref is unavailable; nothing accepted can be compared"
    exit 0
fi

scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-append-only.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT

tracked_files=(benchmarks/BENCHMARKS.md eval-results/EVAL_RESULTS.md)
while IFS= read -r decision; do
    tracked_files+=("$decision")
done < <(git ls-tree -r --name-only "$base_ref" -- 'docs/decisions/*.md')

for path in "${tracked_files[@]}"; do
    if ! git cat-file -e "$base_ref:$path" 2>/dev/null; then
        continue
    fi
    if [[ ! -f "$path" ]]; then
        echo "append-only file was deleted: $path" >&2
        exit 1
    fi
    baseline="$scratch_dir/baseline"
    git show "$base_ref:$path" >"$baseline"
    baseline_bytes=$(wc -c <"$baseline" | tr -d ' ')
    if ! cmp -s -n "$baseline_bytes" "$baseline" "$path"; then
        echo "accepted content was modified instead of appended: $path" >&2
        exit 1
    fi
done

echo "append-only: accepted ledger and decision prefixes are unchanged"
