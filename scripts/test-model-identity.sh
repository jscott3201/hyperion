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

mkdir -p "$model/.cache/huggingface/trees" "$model/.cache/huggingface/download"
printf '{"revision":"synthetic"}\n' >"$model/.cache/huggingface/trees/revision.json"
printf 'metadata\n' >"$model/.cache/huggingface/download/model.safetensors.metadata"
cache_sha256=$(python3 -B -c '
import sys
from pathlib import Path
sys.path.insert(0, "oracle")
from model_identity import transport_cache_identity
print(transport_cache_identity(Path(sys.argv[1]))[0])
' "$model")
cache_verify=(
    python3 -I -S oracle/model_identity.py
    --model "$model"
    --manifest-sha256 "$manifest_sha256"
    --allow-huggingface-cache
    --expected-transport-cache-sha256 "$cache_sha256"
)
"${cache_verify[@]}" | jq -e \
    --arg cache "$cache_sha256" \
    '.transport_cache_separately_bound and .transport_cache_tree_sha256 == $cache' \
    >/dev/null

mv "$model/.cache/huggingface/trees/revision.json" "$scratch_dir/revision.json"
ln -s "$scratch_dir/revision.json" "$model/.cache/huggingface/trees/revision.json"
if "${cache_verify[@]}" >"$scratch_dir/cache-file-symlink.out" 2>&1; then
    echo "model identity accepted a transport-cache file symlink" >&2
    exit 1
fi
rm "$model/.cache/huggingface/trees/revision.json"
mv "$scratch_dir/revision.json" "$model/.cache/huggingface/trees/revision.json"

ln -s trees "$model/.cache/huggingface/tree-alias"
if "${cache_verify[@]}" >"$scratch_dir/cache-directory-symlink.out" 2>&1; then
    echo "model identity accepted a transport-cache directory symlink" >&2
    exit 1
fi
mv "$model/.cache/huggingface/tree-alias" "$scratch_dir/tree-alias"

mkfifo "$model/.cache/huggingface/download/special.metadata"
if "${cache_verify[@]}" >"$scratch_dir/cache-special.out" 2>&1; then
    echo "model identity accepted a transport-cache special file" >&2
    exit 1
fi
rm "$model/.cache/huggingface/download/special.metadata"

printf 'unmanifested cache weights\n' \
    >"$model/.cache/huggingface/download/model-00002-of-00002.safetensors"
if "${cache_verify[@]}" >"$scratch_dir/cache-shard.out" 2>&1; then
    echo "model identity accepted an extra transport-cache weight shard" >&2
    exit 1
fi

python3 -B oracle/test_model_load_boundary.py

echo "model-identity-exact-inventory-controls: pass"
