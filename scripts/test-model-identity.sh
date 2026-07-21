#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-model-identity.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
model="$scratch_dir/model"
mkdir -p "$model"
printf 'synthetic weights\n' >"$model/model.safetensors"
(
    cd "$model"
    shasum -a 256 ./model.safetensors >SHA256SUMS
)
manifest_sha256=$(shasum -a 256 "$model/SHA256SUMS" | awk '{print $1}')
verify=(python3 -I -S oracle/model_identity.py --model "$model" \
    --manifest-sha256 "$manifest_sha256")
"${verify[@]}" | jq -e '.exact_inventory and .symlinks_rejected' >/dev/null

printf 'unmanifested override\n' >"$model/model-00002-of-00002.safetensors"
if "${verify[@]}" >"$scratch_dir/extra.out" 2>&1; then
    echo "model identity accepted an unmanifested weight shard" >&2
    exit 1
fi
mv "$model/model-00002-of-00002.safetensors" "$scratch_dir/extra"

ln -s model.safetensors "$model/model-alias.safetensors"
if "${verify[@]}" >"$scratch_dir/symlink.out" 2>&1; then
    echo "model identity accepted a model-tree symlink" >&2
    exit 1
fi
mv "$model/model-alias.safetensors" "$scratch_dir/model-alias.safetensors"

ln -s model "$scratch_dir/model-link"
if python3 -I -S oracle/model_identity.py --model "$scratch_dir/model-link" \
    --manifest-sha256 "$manifest_sha256" >"$scratch_dir/root-symlink.out" 2>&1; then
    echo "model identity accepted a symlinked model root" >&2
    exit 1
fi

echo "model-identity-exact-inventory-controls: pass"
