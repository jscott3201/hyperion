#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${HYPERION_M0_SOURCE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-source}
converted_dir=${HYPERION_M0_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-mlx-g64-b4}

verify_tree() {
    local label=$1
    local tree=$2
    local expected_manifest_sha256=$3
    local allow_cache=${4:-false}
    local expected_cache_sha256=${5:-}
    local arguments=(
        script oracle/model_identity.py
        --model "$tree"
        --manifest-sha256 "$expected_manifest_sha256"
    )
    if [[ "$allow_cache" == true ]]; then
        arguments+=(
            --allow-huggingface-cache
            --expected-transport-cache-sha256 "$expected_cache_sha256"
        )
    fi
    local identity
    identity=$(scripts/run-isolated-oracle.sh "${arguments[@]}")
    jq -e --arg manifest "$expected_manifest_sha256" --arg cache "$expected_cache_sha256" '
        .schema == "hyperion.model-tree-identity.v1" and
        .manifest_sha256 == $manifest and
        .exact_inventory == true and
        .symlinks_rejected == true and
        (if $cache == "" then
            .transport_cache_excluded == false and
            .transport_cache_separately_bound == false and
            .transport_cache_tree_sha256 == null and
            .transport_cache_file_count == 0
        else
            .transport_cache_excluded == true and
            .transport_cache_separately_bound == true and
            .transport_cache_tree_sha256 == $cache and
            .transport_cache_file_count > 0
        end) and
        (.payload_file_count > 0)
    ' <<<"$identity" >/dev/null
    printf 'model-tree-verified: label=%s identity=%s\n' "$label" "$identity"
}

verify_tree \
    "M0 source" \
    "$source_dir" \
    6a07a92df9260b71117b113a8ad0b305432a48f895abd850a7616241a636ebed \
    true \
    8ee7b68d0ece0fd7a1d281f3c1e9c9d82ece65bcb1cacfbaa04c5864baba9be7
verify_tree \
    "M0 converted model" \
    "$converted_dir" \
    9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144

jq -e '
    .model_type == "gemma4_unified" and
    .text_config.hidden_size == 3840 and
    .text_config.num_hidden_layers == 48 and
    .text_config.vocab_size == 262144
' "$source_dir/config.json" >/dev/null
jq -e '
    .model_type == "gemma4_unified" and
    .text_config.hidden_size == 3840 and
    .text_config.num_hidden_layers == 48 and
    .text_config.vocab_size == 262144 and
    .quantization.group_size == 64 and
    .quantization.bits == 4 and
    .quantization.mode == "affine" and
    .quantization_config == .quantization
' "$converted_dir/config.json" >/dev/null

printf 'models-verified: source=%s converted=%s quant=affine-q4-g64\n' \
    6a07a92df9260b71117b113a8ad0b305432a48f895abd850a7616241a636ebed \
    9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144
