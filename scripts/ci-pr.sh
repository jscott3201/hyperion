#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

actual_rust=$(rustc --version | awk '{print $2}')
if [[ "$actual_rust" != "1.95.0" ]]; then
    echo "tier1 requires rustc 1.95.0 exactly; found $actual_rust" >&2
    exit 1
fi

git diff --check "${HYPERION_BASE_REF:-origin/development}" HEAD --
cargo fmt --all -- --check
cargo build --locked --workspace --all-targets
scripts/test-m1-harness.sh
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
scripts/no-orphan-crates.sh
scripts/check-unsafe-confinement.sh
scripts/check-abi-surface.sh
scripts/test-append-only.sh
scripts/check-append-only.sh
scripts/test-m1-corpus.sh
scripts/test-model-identity.sh
case "${HYPERION_RUN_M3_RUNNER_TEST:-true}" in
    true) scripts/test-m3-stream-parity-runner.sh ;;
    false) echo "m3-stream-parity-runner-regression: skipped (runner unchanged)" ;;
    *) echo "HYPERION_RUN_M3_RUNNER_TEST must be true or false" >&2; exit 1 ;;
esac
scripts/test-m1-archive-layout.sh
scripts/test-workflow-checkouts.sh
python3 -B oracle/test_isolated_oracle.py
python3 -B oracle/test_m1_server_schema.py
python3 -B oracle/test_qwen38_contract.py
python3 -B oracle/test_qwen38_evidence_io.py
python3 -B oracle/test_verified_mlx_server.py
python3 -B -c 'import ast, pathlib; [ast.parse(pathlib.Path(path).read_text()) for path in ("oracle/build_m1_corpus.py", "oracle/build_m1_synthetic_fixtures.py", "oracle/mutate_m1_synthetic_fixture.py", "oracle/isolated_oracle.py", "oracle/model_identity.py", "oracle/oracle_identity.py", "oracle/m1_bench_worker.py", "oracle/m1_server_smoke.py", "oracle/qwen38_contract.py", "oracle/qwen38_evidence_io.py", "oracle/verified_mlx_server.py", "oracle/test_isolated_oracle.py", "oracle/test_m1_worker_envelope.py", "oracle/test_model_load_boundary.py", "oracle/test_m1_server_schema.py", "oracle/test_qwen38_contract.py", "oracle/test_qwen38_evidence_io.py", "oracle/test_verified_mlx_server.py")]'
bash -n scripts/ci-pr.sh scripts/ci-release.sh scripts/test-native.sh \
    scripts/m1-preflight.sh scripts/run-m1-core.sh scripts/run-m1-budget-sweep.sh \
    scripts/run-m1-acca.sh scripts/run-m1-server-smoke.sh scripts/run-m1-baselines.sh \
    scripts/run-m1-nightly.sh scripts/run-m1-stretch.sh scripts/archive-m1-evidence.sh \
    scripts/test-m1-harness.sh scripts/test-model-identity.sh scripts/setup-oracle.sh \
    scripts/verify-oracle.sh scripts/verify-m0-models.sh scripts/verify-m1-e4b-models.sh \
    scripts/verify-m1-archive-layout.sh scripts/test-m1-archive-layout.sh \
    scripts/run-isolated-oracle.sh scripts/test-oracle-startup.sh \
    scripts/test-workflow-checkouts.sh scripts/run-m3-stream-parity.sh \
    scripts/test-m3-stream-parity-runner.sh
scripts/test-native.sh --model-free
