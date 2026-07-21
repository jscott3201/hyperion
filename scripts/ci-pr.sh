#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

actual_rust=$(rustc --version | awk '{print $2}')
if [[ "$actual_rust" != "1.95.0" ]]; then
    echo "tier1 requires rustc 1.95.0 exactly; found $actual_rust" >&2
    exit 1
fi

cargo fmt --all -- --check
cargo build --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
scripts/no-orphan-crates.sh
scripts/check-unsafe-confinement.sh
scripts/check-abi-surface.sh
scripts/check-append-only.sh
scripts/test-native.sh --model-free
