#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
header="$repo_root/native/hyperion_mlx/include/hyperion_mlx.h"
function_count=$(rg -c '^HypStatus hyp_[a-z0-9_]+\(' "$header")

if (( function_count > 25 )); then
    echo "native ABI has $function_count functions; the v1 limit is 25" >&2
    exit 1
fi
# Per-milestone ratchet: M0 = 2 (canary + last_error); M2-1.3 adds the 9 model/step
# lifecycle + stub functions. M3 sampler adds hyp_prefill_chunk_sampled +
# hyp_decode_block_sampled (11 → 13; abi_version 1 → 2). Bump this when an ADR
# approves an ABI addition.
expected_count=13
if (( function_count != expected_count )); then
    echo "M3 expects exactly $expected_count native ABI functions, found $function_count" >&2
    exit 1
fi

echo "abi-surface: $function_count/25 functions"
