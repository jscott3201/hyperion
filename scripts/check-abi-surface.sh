#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
header="$repo_root/native/hyperion_mlx/include/hyperion_mlx.h"
function_count=$(rg -c '^HypStatus hyp_[a-z0-9_]+\(' "$header")

if (( function_count > 25 )); then
    echo "native ABI has $function_count functions; the v1 limit is 25" >&2
    exit 1
fi
if (( function_count != 2 )); then
    echo "M0 expects exactly 2 native ABI functions, found $function_count" >&2
    exit 1
fi

echo "abi-surface: $function_count/25 functions"
