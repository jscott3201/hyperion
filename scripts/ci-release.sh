#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

scripts/ci-pr.sh
scripts/test-native.sh --all
cargo test --locked -p hyperion-ffi -- --ignored --exact tests::real_m5_canary
cargo run --locked -p hyperion-bench -- canary
scripts/oracle-smoke.sh
