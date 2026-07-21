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
scripts/test-append-only.sh
scripts/check-append-only.sh
scripts/verify-m1-corpus.sh
scripts/test-m1-corpus.sh
scripts/test-model-identity.sh
scripts/test-m1-harness.sh
scripts/test-m1-archive-layout.sh
python3 -c 'import ast, pathlib; [ast.parse(pathlib.Path(path).read_text()) for path in ("oracle/build_m1_corpus.py", "oracle/build_m1_synthetic_fixtures.py", "oracle/mutate_m1_synthetic_fixture.py", "oracle/isolated_oracle.py", "oracle/model_identity.py", "oracle/oracle_identity.py", "oracle/m1_bench_worker.py", "oracle/m1_server_smoke.py")]'
bash -n scripts/m1-preflight.sh scripts/run-m1-core.sh scripts/run-m1-budget-sweep.sh \
    scripts/run-m1-acca.sh scripts/run-m1-server-smoke.sh scripts/run-m1-baselines.sh \
    scripts/run-m1-nightly.sh scripts/run-m1-stretch.sh scripts/archive-m1-evidence.sh \
    scripts/test-m1-harness.sh scripts/test-model-identity.sh scripts/setup-oracle.sh \
    scripts/verify-oracle.sh scripts/verify-m0-models.sh scripts/verify-m1-e4b-models.sh \
    scripts/verify-m1-archive-layout.sh scripts/test-m1-archive-layout.sh
scripts/test-native.sh --model-free
