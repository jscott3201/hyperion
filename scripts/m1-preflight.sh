#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${1:?usage: scripts/m1-preflight.sh benchmarks/raw/m1/RUN}
cd "$repo_root"

evidence_home=${HOME:?HOME is required for evidence redaction}
sanitize_log() {
    sed -e "s|$repo_root|<REPO>|g" -e "s|$evidence_home|<HOME>|g"
}

case "$run_root" in
    benchmarks/raw/m1/*) ;;
    *) echo "M1 preflight run root must be under benchmarks/raw/m1" >&2; exit 64 ;;
esac
mkdir -p "$run_root/preflight"
exec > >(sanitize_log | tee "$run_root/preflight/preflight.log") 2>&1

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "M1 preflight requires a clean tracked worktree" >&2
    exit 1
fi
for variable in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS; do
    if [[ -n "${!variable:-}" ]]; then
        echo "M1 preflight rejects ambient $variable" >&2
        exit 1
    fi
done
if env | awk -F= '$1 ~ /^MLX_/ && $1 != "MLX_ROOT" { found=1 } $1 ~ /^CARGO_PROFILE_/ { found=1 } END { exit !found }'; then
    echo "M1 preflight rejects ambient MLX_* (except MLX_ROOT) and CARGO_PROFILE_* overrides" >&2
    exit 1
fi

scripts/verify-oracle.sh | tee "$run_root/preflight/oracle-verification.log"
{
    scripts/verify-m0-models.sh
    scripts/verify-m1-e4b-models.sh
} | tee "$run_root/preflight/model-verification.log"
scripts/ci-pr.sh 2>&1 | sanitize_log | tee "$run_root/preflight/ci.log"
scripts/build-m1-corpus.sh --check 2>&1 | sanitize_log | tee "$run_root/preflight/corpus.log"
export CARGO_TARGET_DIR="$repo_root/target/m1-release"
cargo build --locked --release -p hyperion-bench 2>&1 | sanitize_log | tee "$run_root/preflight/build.log"
binary="$CARGO_TARGET_DIR/release/hyperion-bench"

printf 'm1-preflight-pass: commit=%s executable_sha256=%s corpus_manifest_sha256=%s\n' \
    "$(git rev-parse HEAD)" \
    "$(shasum -a 256 "$binary" | awk '{print $1}')" \
    "$(shasum -a 256 benchmarks/m1/corpus/manifest.json | awk '{print $1}')"
