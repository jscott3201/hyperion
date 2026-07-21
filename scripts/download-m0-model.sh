#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_id=google/gemma-4-12B-it-qat-q4_0-unquantized
revision=b6ed86275a6a5735884e208bfed95b445a684ca2
source_dir="$repo_root/artifacts/models/gemma4-12b-qat-source"

hf download "$model_id" \
    --revision "$revision" \
    --local-dir "$source_dir"
hf cache verify "$model_id" \
    --revision "$revision" \
    --local-dir "$source_dir" \
    --fail-on-missing-files \
    --fail-on-extra-files
