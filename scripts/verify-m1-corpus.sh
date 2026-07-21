#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
corpus_dir=${HYPERION_M1_CORPUS_DIR:-$repo_root/benchmarks/m1/corpus}
manifest="$corpus_dir/manifest.json"

if [[ ! -f "$manifest" ]]; then
    echo "M1 corpus manifest is missing at $manifest" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "M1 corpus verification requires jq" >&2
    exit 2
fi

generator_sha256=$(shasum -a 256 "$repo_root/oracle/build_m1_corpus.py" | awk '{print $1}')
records_file=$(jq -er '.records_file' "$manifest")
records_sha256=$(jq -er '.records_sha256' "$manifest")
record_count=$(jq -er '.record_count' "$manifest")

jq -e \
    --arg generator_sha256 "$generator_sha256" '
    .schema == "hyperion.m1-corpus.v1" and
    .generator == "oracle/build_m1_corpus.py" and
    .generator_sha256 == $generator_sha256 and
    .record_count == 1024 and
    .targets == [512, 1024, 4096, 8192, 16384, 32768] and
    .families == [
        "fdd_explanation",
        "energy_recommendation",
        "point_tagging",
        "operator_copilot",
        "long_transcript",
        "document_qa",
        "short_chat",
        "compound_parallel_tool_chain"
    ] and
    (.fixtures | length) == 12 and
    ([.fixtures[] | [.model_key, .target_tokens]] | unique | length) == 12 and
    ([.fixtures[] | select(
        (.model_key == "12b" or .model_key == "e4b") and
        (.target_tokens == 512 or .target_tokens == 1024 or
         .target_tokens == 4096 or .target_tokens == 8192 or
         .target_tokens == 16384 or .target_tokens == 32768) and
        .actual_tokens == .target_tokens and
        .token_encoding == "little-endian-u32" and
        .record_count >= 8
    )] | length) == 12
    ' "$manifest" >/dev/null

case "$records_file" in
    */* | .* | *..*)
        echo "M1 corpus records path is not a plain relative filename" >&2
        exit 1
        ;;
esac
records_path="$corpus_dir/$records_file"
if [[ ! -f "$records_path" ]]; then
    echo "M1 corpus records file is missing: $records_path" >&2
    exit 1
fi
if [[ "$(shasum -a 256 "$records_path" | awk '{print $1}')" != "$records_sha256" ]]; then
    echo "M1 corpus records hash mismatch" >&2
    exit 1
fi
jq -s -e \
    --argjson count "$record_count" \
    --argjson families "$(jq -c '.families' "$manifest")" '
    length == $count and
    ([.[].family] | unique | sort) == ($families | sort) and
    ([.[].index] | unique | length) == $count
    ' "$records_path" >/dev/null

while IFS=$'\t' read -r model_key target actual rendered_file rendered_sha token_file token_sha; do
    for name in "$rendered_file" "$token_file"; do
        case "$name" in
            */* | .* | *..*)
                echo "M1 fixture path is not a plain relative filename: $name" >&2
                exit 1
                ;;
        esac
    done
    rendered_path="$corpus_dir/$rendered_file"
    token_path="$corpus_dir/$token_file"
    if [[ ! -s "$rendered_path" || ! -s "$token_path" ]]; then
        echo "M1 fixture payload is missing for $model_key/$target" >&2
        exit 1
    fi
    if [[ "$(shasum -a 256 "$rendered_path" | awk '{print $1}')" != "$rendered_sha" ]]; then
        echo "M1 rendered fixture hash mismatch for $model_key/$target" >&2
        exit 1
    fi
    if [[ "$(shasum -a 256 "$token_path" | awk '{print $1}')" != "$token_sha" ]]; then
        echo "M1 token fixture hash mismatch for $model_key/$target" >&2
        exit 1
    fi
    token_bytes=$(stat -f '%z' "$token_path")
    if [[ "$token_bytes" -ne $((actual * 4)) ]]; then
        echo "M1 token fixture byte count mismatch for $model_key/$target" >&2
        exit 1
    fi
    if od -An -tu4 -v "$token_path" | awk '{for (i = 1; i <= NF; i++) if ($i >= 262144) exit 1}'; then
        :
    else
        echo "M1 token fixture contains an out-of-vocabulary ID for $model_key/$target" >&2
        exit 1
    fi
done < <(
    jq -r '.fixtures[] | [
        .model_key,
        .target_tokens,
        .actual_tokens,
        .rendered_file,
        .rendered_sha256,
        .token_file,
        .token_sha256
    ] | @tsv' "$manifest"
)

printf 'm1-corpus-verified: manifest=%s fixtures=12 tokens=512,1024,4096,8192,16384,32768\n' \
    "$(shasum -a 256 "$manifest" | awk '{print $1}')"
