#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "M1 preflight requires a clean tracked worktree" >&2
    exit 1
fi

scripts/ci-pr.sh
scripts/build-m1-corpus.sh --check
cargo build --locked --release -p hyperion-bench

printf 'm1-preflight-pass: commit=%s executable_sha256=%s corpus_manifest_sha256=%s\n' \
    "$(git rev-parse HEAD)" \
    "$(shasum -a 256 target/release/hyperion-bench | awk '{print $1}')" \
    "$(shasum -a 256 benchmarks/m1/corpus/manifest.json | awk '{print $1}')"
