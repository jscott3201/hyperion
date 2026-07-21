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

scripts/verify-oracle.sh
scripts/verify-m0-models.sh
scripts/ci-pr.sh
scripts/test-native.sh --all
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
scripts/oracle-smoke.sh
