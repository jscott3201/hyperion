#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${HYPERION_M0_SOURCE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-source}
output_dir=${HYPERION_M0_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-mlx-g64-b4}

if [[ ! -x "$repo_root/oracle/.venv/bin/mlx_lm.convert" ]]; then
    echo "oracle environment is missing; run scripts/setup-oracle.sh" >&2
    exit 2
fi

"$repo_root/oracle/.venv/bin/mlx_lm.convert" \
    --hf-path "$source_dir" \
    --mlx-path "$output_dir" \
    --quantize \
    --q-group-size 64 \
    --q-bits 4 \
    --q-mode affine
