#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_id=google/gemma-4-E4B-it-qat-q4_0-unquantized
revision=476025a01dbf99361c062bbeca3d6a76bb4c4566
source_dir=${HYPERION_M1_E4B_SOURCE_MODEL:-$repo_root/artifacts/models/gemma4-e4b-qat-source}

hf download "$model_id" \
    --revision "$revision" \
    --local-dir "$source_dir"
hf cache verify "$model_id" \
    --revision "$revision" \
    --local-dir "$source_dir" \
    --fail-on-missing-files

# `hf download --local-dir` owns `.cache/huggingface`; separately reject any
# unexpected files at the repository root while allowing our checksum manifest.
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-e4b-files.XXXXXX")
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
    echo "M1 E4B source root differs from the reviewed nine-file snapshot:" >&2
    diff -u "$scratch_dir/expected" "$scratch_dir/actual" >&2 || true
    exit 1
fi
