#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "release gate requires a clean tracked worktree" >&2
    exit 1
fi

scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-release.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
export CARGO_TARGET_DIR="$scratch_dir/target"
export HYPERION_NATIVE_BUILD_DIR="$scratch_dir/native"
export HYPERION_REPO_ROOT="$repo_root"
export HYPERION_TINY_FIXTURE="$repo_root/crates/hyperion-model/fixtures/gemma4-unified-tiny"
export HYPERION_12B_ARTIFACT="${HYPERION_M0_ORACLE_MODEL:-$repo_root/artifacts/models/gemma4-12b-qat-mlx-g64-b4}"

required_release_files=(
    "$HYPERION_TINY_FIXTURE/model-00001-of-00002.safetensors"
    "$HYPERION_TINY_FIXTURE/model-00002-of-00002.safetensors"
    "$HYPERION_12B_ARTIFACT/model-00001-of-00002.safetensors"
    "$HYPERION_12B_ARTIFACT/model-00002-of-00002.safetensors"
    "$repo_root/native/hyperion_mlx/tests/fixtures/12b_hidden_golden.safetensors"
    "$repo_root/native/hyperion_mlx/tests/fixtures/12b_greedy_golden.safetensors"
    "$repo_root/native/hyperion_mlx/tests/fixtures/12b_long_greedy_golden.safetensors"
    "$repo_root/native/hyperion_mlx/tests/fixtures/12b_faulted_greedy_golden.safetensors"
)
for required_release_file in "${required_release_files[@]}"; do
    if [[ ! -f "$required_release_file" || -L "$required_release_file" ]]; then
        echo "release gate requires a real regular input: $required_release_file" >&2
        exit 1
    fi
done

scripts/verify-oracle.sh
scripts/test-oracle-startup.sh
scripts/verify-m0-models.sh
scripts/ci-pr.sh
scripts/test-native.sh --all
cargo test --locked -p hyperion-server \
    engine::tests::engine_drives_greedy_on_tiny_fixture \
    -- --ignored --exact --nocapture
metallib_path="$CARGO_TARGET_DIR/debug/hyperion_canary.metallib"
if [[ ! -s "$metallib_path" ]]; then
    echo "fresh Cargo build did not produce its metallib sidecar" >&2
    exit 1
fi
metallib_sha256=$(shasum -a 256 "$metallib_path" | awk '{print $1}')
export HYPERION_METALLIB_PATH="$metallib_path"
cargo test --locked -p hyperion-ffi -- --ignored --exact tests::real_m5_canary
cargo run --locked -p hyperion-bench -- canary
post_canary_sha256=$(shasum -a 256 "$metallib_path" | awk '{print $1}')
if [[ "$post_canary_sha256" != "$metallib_sha256" ]]; then
    echo "fresh metallib changed while the canary was executing" >&2
    exit 1
fi
printf 'MEASURED {"schema":"hyperion.metallib.v1","source":"fresh-cargo-sidecar","sha256":"%s","stable_during_canary":true}\n' \
    "$metallib_sha256"
unset HYPERION_METALLIB_PATH
env -u CARGO_TARGET_DIR -u MLX_ROOT \
    HYPERION_12B_ARTIFACT="$HYPERION_12B_ARTIFACT" \
    scripts/run-m3-stream-parity.sh "$scratch_dir/m3-stream-parity"
scripts/oracle-smoke.sh
