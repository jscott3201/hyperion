#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${HYPERION_M1_E4B_SOURCE_MODEL:-$repo_root/artifacts/models/gemma4-e4b-qat-source}
converted_dir=${HYPERION_M1_E4B_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-e4b-qat-mlx-g64-b4}
source_manifest_sha256=d86886b83233724bbfd0bbf6f033bcd7b2862c16206f62b6d7c01826fef44c95
converted_manifest_sha256=9ba65423d3b2bab1e7c52ea88a1a2b0a33c1f51909b1df66330bf872b7a6c2b0

verify_tree() {
    local label=$1
    local tree=$2
    local expected_manifest_sha256=$3
    local manifest="$tree/SHA256SUMS"
    if [[ ! -f "$manifest" ]]; then
        echo "$label checksum manifest is missing at $manifest" >&2
        exit 2
    fi
    local actual_manifest_sha256
    actual_manifest_sha256=$(shasum -a 256 "$manifest" | awk '{print $1}')
    if [[ "$actual_manifest_sha256" != "$expected_manifest_sha256" ]]; then
        echo "$label checksum-manifest identity differs from the reviewed artifact" >&2
        exit 1
    fi
    (
        cd "$tree"
        shasum -a 256 -c SHA256SUMS
    )
}

verify_e4b_geometry() {
    local config=$1
    jq -e '
        .model_type == "gemma4" and
        .text_config.hidden_size == 2560 and
        .text_config.num_hidden_layers == 42 and
        .text_config.vocab_size == 262144 and
        .text_config.max_position_embeddings == 131072 and
        .text_config.sliding_window == 512 and
        .text_config.num_attention_heads == 8 and
        .text_config.num_key_value_heads == 2 and
        .text_config.head_dim == 256 and
        .text_config.global_head_dim == 512 and
        .text_config.attention_k_eq_v == false and
        .text_config.hidden_size_per_layer_input == 256 and
        .text_config.num_kv_shared_layers == 18 and
        (.text_config.layer_types | length) == 42 and
        ([.text_config.layer_types[] | select(. == "full_attention")] | length) == 7 and
        ([.text_config.layer_types[] | select(. == "sliding_attention")] | length) == 35
    ' "$config" >/dev/null
}

verify_tree "M1 E4B source" "$source_dir" "$source_manifest_sha256"
verify_tree "M1 E4B converted model" "$converted_dir" "$converted_manifest_sha256"
verify_e4b_geometry "$source_dir/config.json"
verify_e4b_geometry "$converted_dir/config.json"
jq -e '
    .quantization.group_size == 64 and
    .quantization.bits == 4 and
    .quantization.mode == "affine" and
    .quantization_config == .quantization
' "$converted_dir/config.json" >/dev/null
if ! rg -q '^license: apache-2\.0$' "$source_dir/README.md"; then
    echo "M1 E4B source model card does not declare apache-2.0" >&2
    exit 1
fi

printf 'm1-e4b-models-verified: source=%s converted=%s quant=affine-q4-g64\n' \
    "$source_manifest_sha256" \
    "$converted_manifest_sha256"
