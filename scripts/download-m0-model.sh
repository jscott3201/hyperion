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
    --fail-on-missing-files

# `hf download --local-dir` creates `.cache/huggingface` under the destination,
# and `hf cache verify --fail-on-extra-files` counts its own metadata as extra.
# Enforce the reviewed repository root independently while leaving that cache intact.
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-hf-files.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
printf '%s\n' \
    .gitattributes \
    README.md \
    chat_template.jinja \
    config.json \
    generation_config.json \
    model.safetensors \
    processor_config.json \
    tokenizer.json \
    tokenizer_config.json | sort >"$scratch_dir/expected"
find "$source_dir" -maxdepth 1 -type f ! -name SHA256SUMS -exec basename {} \; | sort \
    >"$scratch_dir/actual"
if ! cmp -s "$scratch_dir/expected" "$scratch_dir/actual"; then
    echo "M0 source root differs from the reviewed nine-file snapshot:" >&2
    diff -u "$scratch_dir/expected" "$scratch_dir/actual" >&2 || true
    exit 1
fi
