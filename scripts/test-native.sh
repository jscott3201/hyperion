#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:---model-free}
case "$mode" in
    --model-free|--all) ;;
    *) echo "usage: scripts/test-native.sh [--model-free|--all]" >&2; exit 64 ;;
esac

hyp_mlx_root=${MLX_ROOT:-/opt/homebrew/opt/mlx}
hyp_build_dir=${HYPERION_NATIVE_BUILD_DIR:-$repo_root/build/native}

cmake \
    -S "$repo_root/native/hyperion_mlx" \
    -B "$hyp_build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 \
    -DCMAKE_PREFIX_PATH="$hyp_mlx_root" \
    -DBUILD_TESTING=ON
cmake --build "$hyp_build_dir" --config Release --parallel

if [[ ! -s "$hyp_build_dir/hyperion_canary.metallib" ]]; then
    echo "native build did not produce hyperion_canary.metallib" >&2
    exit 1
fi

if [[ "$mode" == "--all" ]]; then
    ctest --test-dir "$hyp_build_dir" --output-on-failure -C Release
else
    ctest \
        --test-dir "$hyp_build_dir" \
        --output-on-failure \
        -C Release \
        -R 'hyperion_(platform_policy(_negative_control)?|abi_contract)$'
fi

metallib_sha256=$(shasum -a 256 "$hyp_build_dir/hyperion_canary.metallib" | awk '{print $1}')
printf 'MEASURED {"schema":"hyperion.native-metallib.v1","build":"cmake","sha256":"%s"}\n' \
    "$metallib_sha256"
