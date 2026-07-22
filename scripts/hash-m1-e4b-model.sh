#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${HYPERION_M1_E4B_SOURCE_MODEL:-$repo_root/artifacts/models/gemma4-e4b-qat-source}
converted_dir=${HYPERION_M1_E4B_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-e4b-qat-mlx-g64-b4}

hash_tree() {
    local tree=$1
    if [[ ! -d "$tree" ]]; then
        echo "model directory is missing: $tree" >&2
        exit 2
    fi
    local scratch
    scratch=$(mktemp "${TMPDIR:-/tmp}/hyperion-m1-e4b-sha256.XXXXXX")
    (
        cd "$tree"
        find . -type f ! -name SHA256SUMS ! -path './.cache/*' -print0 |
            sort -z |
            xargs -0 shasum -a 256
    ) >"$scratch"
    mv "$scratch" "$tree/SHA256SUMS"
    shasum -a 256 "$tree/SHA256SUMS"
}

hash_tree "$source_dir"
hash_tree "$converted_dir"
