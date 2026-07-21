#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${HYPERION_M0_SOURCE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-source}
converted_dir=${HYPERION_M0_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-mlx-g64-b4}

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

verify_tree \
    "M0 source" \
    "$source_dir" \
    6a07a92df9260b71117b113a8ad0b305432a48f895abd850a7616241a636ebed
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
