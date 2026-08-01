#!/usr/bin/env bash
set -euo pipefail

m3_test_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$m3_test_repo_root"

m3_test_scratch=$(mktemp -d)
m3_test_cleanup() {
    rm -rf -- "$m3_test_scratch"
}
trap m3_test_cleanup EXIT

m3_test_real_rustup_home=${RUSTUP_HOME:-$HOME/.rustup}

# The regression controls each execution-shaping variable explicitly. Remove any
# inherited copies so host configuration cannot change which control is under test.
while IFS= read -r m3_test_inherited_override; do
    unset "$m3_test_inherited_override"
done < <(python3 -I -S -c '
import os

exact = {
    "CARGO_HOME",
    "CARGO_TARGET_DIR",
    "CARGO_BUILD_TARGET",
    "CARGO_BUILD_TARGET_DIR",
    "CARGO_RUNNER",
    "CARGO_TARGET_RUNNER",
    "CARGO_BUILD_RUNNER",
    "CARGO_BUILD_RUSTFLAGS",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
}
prefixes = ("MLX_", "CARGO_PROFILE_", "CARGO_TARGET_")
for name in sorted(os.environ):
    if name in exact or name.startswith(prefixes):
        print(name)
')

m3_test_real_cargo=$(command -v cargo)
m3_test_real_git=$(command -v git)
m3_test_real_shasum=$(command -v shasum)
m3_test_heavy_marker="$m3_test_scratch/heavy-command-started"
m3_test_payload_marker="$m3_test_scratch/payload-checksum-started"
mkdir -p "$m3_test_scratch/bin"
# shellcheck disable=SC2016 # The generated shim expands these at execution time.
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'if (( $# == 1 )) && [[ "$1" == --version ]]; then' \
    '    exec "${M3_TEST_REAL_CARGO:?}" --version' \
    'fi' \
    ': >"${M3_TEST_HEAVY_MARKER:?}"' \
    'exit 97' \
    >"$m3_test_scratch/bin/cargo"
chmod +x "$m3_test_scratch/bin/cargo"

m3_test_assert_log_hashes() {
    local m3_test_assert_evidence=$1
    local m3_test_log_name
    local m3_test_actual_log_sha
    local m3_test_manifest_log_sha
    for m3_test_log_name in preflight identity build test; do
        if [[ ! -f "$m3_test_assert_evidence/$m3_test_log_name.log" ]]; then
            echo "runner did not retain $m3_test_log_name.log" >&2
            return 1
        fi
        m3_test_actual_log_sha=$(shasum -a 256 \
            "$m3_test_assert_evidence/$m3_test_log_name.log" | awk '{print $1}')
        m3_test_manifest_log_sha=$(jq -r \
            ".logs.$m3_test_log_name.sha256" \
            "$m3_test_assert_evidence/manifest.json")
        if [[ "$m3_test_actual_log_sha" != "$m3_test_manifest_log_sha" ]]; then
            echo "manifest hash for $m3_test_log_name.log is stale" >&2
            return 1
        fi
    done
}

m3_test_evidence="$m3_test_scratch/evidence"
set +e
env -u HYPERION_12B_ARTIFACT \
    PATH="$m3_test_scratch/bin:$PATH" \
    M3_TEST_HEAVY_MARKER="$m3_test_heavy_marker" \
    M3_TEST_REAL_CARGO="$m3_test_real_cargo" \
    scripts/run-m3-stream-parity.sh "$m3_test_evidence" \
    >"$m3_test_scratch/invocation.log" 2>&1
m3_test_status=$?
set -e

if (( m3_test_status != 64 )); then
    echo "missing-artifact runner exit was $m3_test_status, expected 64" >&2
    sed -n '1,200p' "$m3_test_scratch/invocation.log" >&2
    exit 1
fi
if [[ -e "$m3_test_heavy_marker" ]]; then
    echo "missing-artifact runner started a Cargo build/test/run" >&2
    exit 1
fi
if [[ ! -f "$m3_test_evidence/preflight.log" || \
      ! -f "$m3_test_evidence/identity.log" || \
      ! -f "$m3_test_evidence/build.log" || \
      ! -f "$m3_test_evidence/test.log" || \
      ! -f "$m3_test_evidence/manifest.json" ]]; then
    echo "missing-artifact runner did not leave all required evidence files" >&2
    exit 1
fi
if ! grep -q \
    'HYPERION_12B_ARTIFACT must be set explicitly to the pinned real artifact' \
    "$m3_test_evidence/preflight.log"
then
    echo "missing-artifact preflight did not identify the explicit artifact requirement" >&2
    exit 1
fi
if [[ -s "$m3_test_evidence/identity.log" || \
      -s "$m3_test_evidence/build.log" || \
      -s "$m3_test_evidence/test.log" ]]; then
    echo "missing-artifact runner wrote identity/build/test output before rejecting the artifact" >&2
    exit 1
fi
jq -e '
    .schema == "hyperion.m3-stream-parity-evidence.v1" and
    .status == "failed" and
    .failure_stage == "artifact_environment" and
    .exit_status == 64 and
    .source.clean == false and
    .fixture.expected_sha256 == "086ca72232de415973564b2c6028c98a7063f7d024b85411512071650c86cf3d" and
    .fixture.actual_sha256 == null and
    .artifact.expected_historical_manifest_sha256 == "9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144" and
    .artifact.expected_owner_payload_manifest_sha256 == "3cee7e9c21051eb6e6857ef485b355b4a2e847901930d64620348cbc7f56c806" and
    .artifact.identity_kind == null and
    .artifact.manifest_sha256 == null and
    .artifact.identity == null and
    .command.build_argv == [
      "cargo", "test", "--locked", "--offline",
      "-p", "hyperion-server", "--test", "contract",
      "--no-run", "--message-format=json-render-diagnostics"
    ] and
    .command.direct_test_argv == [
      "<contract-test-executable>",
      "real_http_sse_matches_m2_golden",
      "--ignored", "--exact", "--nocapture"
    ] and
    .command.build_environment.CARGO_TARGET_DIR == "<fresh-runner-target-root>" and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null and
    .execution.test_binary_sha256 == null
' "$m3_test_evidence/manifest.json" >/dev/null
m3_test_assert_log_hashes "$m3_test_evidence"

m3_test_unknown_artifact="$m3_test_scratch/unknown-artifact"
m3_test_unknown_evidence="$m3_test_scratch/unknown-evidence"
mkdir -p "$m3_test_unknown_artifact"
printf 'unapproved reconstructed payload\n' >"$m3_test_unknown_artifact/SHA256SUMS"
m3_test_unknown_manifest_sha=$(shasum -a 256 "$m3_test_unknown_artifact/SHA256SUMS" | awk '{print $1}')
set +e
PATH="$m3_test_scratch/bin:$PATH" \
    M3_TEST_HEAVY_MARKER="$m3_test_heavy_marker" \
    M3_TEST_REAL_CARGO="$m3_test_real_cargo" \
    HYPERION_12B_ARTIFACT="$m3_test_unknown_artifact" \
    scripts/run-m3-stream-parity.sh "$m3_test_unknown_evidence" \
    >"$m3_test_scratch/unknown-invocation.log" 2>&1
m3_test_unknown_status=$?
set -e
if (( m3_test_unknown_status != 64 )); then
    echo "unknown-manifest runner exit was $m3_test_unknown_status, expected 64" >&2
    sed -n '1,200p' "$m3_test_scratch/unknown-invocation.log" >&2
    exit 1
fi
if [[ -e "$m3_test_heavy_marker" ]]; then
    echo "unknown-manifest runner started a Cargo build/test/run" >&2
    exit 1
fi
if ! grep -q \
    'artifact identity manifest digest is not an approved immutable payload' \
    "$m3_test_unknown_evidence/preflight.log"
then
    echo "unknown-manifest preflight did not report the immutable allowlist failure" >&2
    exit 1
fi
jq -e \
    --arg actual "$m3_test_unknown_manifest_sha" \
    '.status == "failed" and
     .failure_stage == "artifact_environment" and
     .exit_status == 64 and
     .artifact.identity_kind == "historical_sha256sums" and
     .artifact.manifest_sha256 == $actual and
     .artifact.identity == null' \
    "$m3_test_unknown_evidence/manifest.json" >/dev/null
m3_test_assert_log_hashes "$m3_test_unknown_evidence"

for m3_test_manifest_case in missing_manifest ambiguous_manifests; do
    m3_test_case_artifact="$m3_test_scratch/$m3_test_manifest_case-artifact"
    m3_test_case_evidence="$m3_test_scratch/$m3_test_manifest_case-evidence"
    mkdir -p "$m3_test_case_artifact"
    if [[ "$m3_test_manifest_case" == ambiguous_manifests ]]; then
        : >"$m3_test_case_artifact/SHA256SUMS"
        : >"$m3_test_case_artifact/PAYLOAD_SHA256SUMS"
    fi
    set +e
    PATH="$m3_test_scratch/bin:$PATH" \
        M3_TEST_HEAVY_MARKER="$m3_test_heavy_marker" \
        M3_TEST_REAL_CARGO="$m3_test_real_cargo" \
        HYPERION_12B_ARTIFACT="$m3_test_case_artifact" \
        scripts/run-m3-stream-parity.sh "$m3_test_case_evidence" \
        >"$m3_test_scratch/$m3_test_manifest_case-invocation.log" 2>&1
    m3_test_case_status=$?
    set -e
    if (( m3_test_case_status != 64 )); then
        echo "$m3_test_manifest_case runner exit was $m3_test_case_status, expected 64" >&2
        exit 1
    fi
    if ! grep -q \
        'artifact must contain exactly one recognized identity manifest' \
        "$m3_test_case_evidence/preflight.log"
    then
        echo "$m3_test_manifest_case did not fail the manifest ambiguity gate" >&2
        exit 1
    fi
    jq -e '
        .status == "failed" and
        .failure_stage == "artifact_environment" and
        .exit_status == 64
    ' "$m3_test_case_evidence/manifest.json" >/dev/null
    m3_test_assert_log_hashes "$m3_test_case_evidence"
done

# Drive the approved owner-manifest branch with synthetic files while the
# shasum shim accepts only the manifest digest and fails payload verification.
# This proves a failed `shasum -c` cannot be overwritten by later JSON output.
# shellcheck disable=SC2016 # Generated shim expands variables when invoked.
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'if [[ "${1:-}" == status ]]; then exit 0; fi' \
    'exec "${M3_TEST_REAL_GIT:?}" "$@"' \
    >"$m3_test_scratch/bin/git"
# shellcheck disable=SC2016 # Generated shim expands variables when invoked.
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'if (( $# == 3 )) && [[ "$1" == -a && "$2" == 256 && "$3" == */PAYLOAD_SHA256SUMS ]]; then' \
    '    printf "%s  %s\n" "3cee7e9c21051eb6e6857ef485b355b4a2e847901930d64620348cbc7f56c806" "$3"' \
    '    exit 0' \
    'fi' \
    'if (( $# == 4 )) && [[ "$1" == -a && "$2" == 256 && "$3" == -c && "$4" == PAYLOAD_SHA256SUMS ]]; then' \
    '    : >"${M3_TEST_PAYLOAD_MARKER:?}"' \
    '    echo "generation_config.json: FAILED"' \
    '    exit 1' \
    'fi' \
    'exec "${M3_TEST_REAL_SHASUM:?}" "$@"' \
    >"$m3_test_scratch/bin/shasum"
printf '%s\n' '#!/usr/bin/env bash' 'printf "arm64\n"' >"$m3_test_scratch/bin/uname"
# shellcheck disable=SC2016 # Generated shim expands variables when invoked.
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "${2:-}" in' \
    '    hw.model) printf "TestMac\n" ;;' \
    '    hw.memsize) printf "17179869184\n" ;;' \
    '    *) exit 1 ;;' \
    'esac' \
    >"$m3_test_scratch/bin/sysctl"
# shellcheck disable=SC2016 # Generated shim expands variables when invoked.
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "${1:-}" in' \
    '    -productVersion) printf "26.2\n" ;;' \
    '    -buildVersion) printf "test-build\n" ;;' \
    '    *) exit 1 ;;' \
    'esac' \
    >"$m3_test_scratch/bin/sw_vers"
chmod +x \
    "$m3_test_scratch/bin/git" \
    "$m3_test_scratch/bin/shasum" \
    "$m3_test_scratch/bin/uname" \
    "$m3_test_scratch/bin/sysctl" \
    "$m3_test_scratch/bin/sw_vers"

m3_test_owner_artifact="$m3_test_scratch/owner-artifact"
m3_test_owner_evidence="$m3_test_scratch/owner-evidence"
mkdir -p "$m3_test_owner_artifact"
for m3_test_owner_file in \
    chat_template.jinja \
    config.json \
    generation_config.json \
    model-00001-of-00002.safetensors \
    model-00002-of-00002.safetensors \
    model.safetensors.index.json \
    tokenizer.json \
    tokenizer_config.json
do
    : >"$m3_test_owner_artifact/$m3_test_owner_file"
    printf '%064d  %s\n' 0 "$m3_test_owner_file" >>"$m3_test_owner_artifact/PAYLOAD_SHA256SUMS"
done
printf 'allowed release metadata\n' >"$m3_test_owner_artifact/README.md"
m3_test_clean_home="$m3_test_scratch/clean-home"
mkdir -p "$m3_test_clean_home"

# An ambient Cargo target runner must be rejected by name before the payload
# checksum, Cargo version/build, fixture checksum, or direct test can execute.
m3_test_override_evidence="$m3_test_scratch/override-evidence"
set +e
PATH="$m3_test_scratch/bin:$PATH" \
    M3_TEST_HEAVY_MARKER="$m3_test_heavy_marker" \
    M3_TEST_PAYLOAD_MARKER="$m3_test_payload_marker" \
    M3_TEST_REAL_CARGO="$m3_test_real_cargo" \
    M3_TEST_REAL_GIT="$m3_test_real_git" \
    M3_TEST_REAL_SHASUM="$m3_test_real_shasum" \
    HOME="$m3_test_clean_home" \
    RUSTUP_HOME="$m3_test_real_rustup_home" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER="$m3_test_scratch/fake-runner" \
    scripts/run-m3-stream-parity.sh "$m3_test_override_evidence" \
    >"$m3_test_scratch/override-invocation.log" 2>&1
m3_test_override_status=$?
set -e
if (( m3_test_override_status != 1 )); then
    echo "ambient target-runner exit was $m3_test_override_status, expected 1" >&2
    sed -n '1,240p' "$m3_test_scratch/override-invocation.log" >&2
    exit 1
fi
if [[ -e "$m3_test_heavy_marker" ]]; then
    echo "ambient target-runner control invoked Cargo beyond the exact version probe" >&2
    exit 1
fi
if [[ -e "$m3_test_payload_marker" ]]; then
    echo "ambient target-runner control started payload checksum verification" >&2
    exit 1
fi
if ! grep -q \
    'M3 stream parity rejects ambient CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER' \
    "$m3_test_override_evidence/preflight.log"
then
    echo "ambient target-runner gate did not report the exact variable name" >&2
    exit 1
fi
if [[ -s "$m3_test_override_evidence/identity.log" || \
      -s "$m3_test_override_evidence/build.log" || \
      -s "$m3_test_override_evidence/test.log" ]]; then
    echo "ambient target-runner control wrote identity/build/test output" >&2
    exit 1
fi
jq -e '
    .status == "failed" and
    .failure_stage == "ambient_execution_overrides" and
    .exit_status == 1 and
    .source.clean == true and
    .fixture.actual_sha256 == null and
    .artifact.identity_kind == "owner_payload_sha256sums" and
    .artifact.manifest_sha256 == "3cee7e9c21051eb6e6857ef485b355b4a2e847901930d64620348cbc7f56c806" and
    .artifact.identity == null and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null and
    .execution.test_binary_sha256 == null
' "$m3_test_override_evidence/manifest.json" >/dev/null
m3_test_assert_log_hashes "$m3_test_override_evidence"

# Default user Cargo configuration is ambient even when CARGO_HOME is unset. A
# rustc wrapper there could replace the final executable after a real compile.
m3_test_config_home="$m3_test_scratch/config-home"
m3_test_config_evidence="$m3_test_scratch/config-evidence"
mkdir -p "$m3_test_config_home/.cargo"
printf '%s\n' \
    '[build]' \
    'rustc-wrapper = "/definitely-not-an-approved-wrapper"' \
    >"$m3_test_config_home/.cargo/config.toml"
set +e
PATH="$m3_test_scratch/bin:$PATH" \
    M3_TEST_HEAVY_MARKER="$m3_test_heavy_marker" \
    M3_TEST_PAYLOAD_MARKER="$m3_test_payload_marker" \
    M3_TEST_REAL_CARGO="$m3_test_real_cargo" \
    M3_TEST_REAL_GIT="$m3_test_real_git" \
    M3_TEST_REAL_SHASUM="$m3_test_real_shasum" \
    HOME="$m3_test_config_home" \
    RUSTUP_HOME="$m3_test_real_rustup_home" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    scripts/run-m3-stream-parity.sh "$m3_test_config_evidence" \
    >"$m3_test_scratch/config-invocation.log" 2>&1
m3_test_config_status=$?
set -e
if (( m3_test_config_status != 1 )); then
    echo "default Cargo config exit was $m3_test_config_status, expected 1" >&2
    sed -n '1,240p' "$m3_test_scratch/config-invocation.log" >&2
    exit 1
fi
if [[ -e "$m3_test_heavy_marker" || -e "$m3_test_payload_marker" ]]; then
    echo "default Cargo config control reached Cargo or payload verification" >&2
    exit 1
fi
if ! grep -q \
    'M3 stream parity rejects default user Cargo config.toml' \
    "$m3_test_config_evidence/preflight.log"
then
    echo "default Cargo config control did not fail the configuration gate" >&2
    exit 1
fi
jq -e '
    .status == "failed" and
    .failure_stage == "cargo_configuration_preflight" and
    .exit_status == 1 and
    .source.clean == true and
    .fixture.actual_sha256 == null and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null and
    .execution.test_binary_sha256 == null
' "$m3_test_config_evidence/manifest.json" >/dev/null
m3_test_assert_log_hashes "$m3_test_config_evidence"

set +e
PATH="$m3_test_scratch/bin:$PATH" \
    M3_TEST_HEAVY_MARKER="$m3_test_heavy_marker" \
    M3_TEST_PAYLOAD_MARKER="$m3_test_payload_marker" \
    M3_TEST_REAL_CARGO="$m3_test_real_cargo" \
    M3_TEST_REAL_GIT="$m3_test_real_git" \
    M3_TEST_REAL_SHASUM="$m3_test_real_shasum" \
    HOME="$m3_test_clean_home" \
    RUSTUP_HOME="$m3_test_real_rustup_home" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    scripts/run-m3-stream-parity.sh "$m3_test_owner_evidence" \
    >"$m3_test_scratch/owner-invocation.log" 2>&1
m3_test_owner_status=$?
set -e
if (( m3_test_owner_status != 1 )); then
    echo "owner checksum-failure exit was $m3_test_owner_status, expected 1" >&2
    sed -n '1,240p' "$m3_test_scratch/owner-invocation.log" >&2
    exit 1
fi
if [[ -e "$m3_test_heavy_marker" ]]; then
    echo "owner checksum failure started the Cargo test" >&2
    exit 1
fi
if [[ ! -e "$m3_test_payload_marker" ]]; then
    echo "owner checksum failure did not reach payload verification" >&2
    sed -n '1,240p' "$m3_test_scratch/owner-invocation.log" >&2
    exit 1
fi
if ! grep -q 'generation_config.json: FAILED' "$m3_test_owner_evidence/identity.log"; then
    echo "owner checksum failure was not retained in identity.log" >&2
    exit 1
fi
jq -e '
    .status == "failed" and
    .failure_stage == "artifact_identity_preflight" and
    .exit_status == 1 and
    .artifact.identity_kind == "owner_payload_sha256sums" and
    .artifact.manifest_sha256 == "3cee7e9c21051eb6e6857ef485b355b4a2e847901930d64620348cbc7f56c806" and
    .artifact.identity == null and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null and
    .execution.test_binary_sha256 == null
' "$m3_test_owner_evidence/manifest.json" >/dev/null
m3_test_assert_log_hashes "$m3_test_owner_evidence"

# Exercise the runner's exact embedded resolver without compiling. These cases
# fail if Cargo JSON is ambiguous or points at a symlink/path escape.
m3_test_resolver="$m3_test_scratch/contract-resolver.py"
awk '
    /^# M3_CONTRACT_RESOLVER_PYTHON_BEGIN$/ { capture=1; next }
    /^# M3_CONTRACT_RESOLVER_PYTHON_END$/ { capture=0; found=1; exit }
    capture { print }
    END { if (!found) exit 1 }
' scripts/run-m3-stream-parity.sh >"$m3_test_resolver"
if [[ ! -s "$m3_test_resolver" ]]; then
    echo "could not extract the runner contract-executable resolver" >&2
    exit 1
fi
m3_test_discovery_root="$m3_test_scratch/discovery target"
mkdir -p "$m3_test_discovery_root/debug/deps"
m3_test_discovery_root=$(cd "$m3_test_discovery_root" && pwd -P)
m3_test_valid_executable="$m3_test_discovery_root/debug/deps/contract valid"
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' >"$m3_test_valid_executable"
chmod +x "$m3_test_valid_executable"
m3_test_valid_json="$m3_test_scratch/discovery-valid.jsonl"
jq -cn --arg executable "$m3_test_valid_executable" '
    {
      reason: "compiler-artifact",
      target: {name: "contract", kind: ["test"]},
      executable: $executable
    }
' >"$m3_test_valid_json"
m3_test_discovered_executable=$(python3 -I -S "$m3_test_resolver" \
    discover "$m3_test_discovery_root" "$m3_test_valid_json")
if [[ "$m3_test_discovered_executable" != "$m3_test_valid_executable" ]]; then
    echo "resolver did not return the one valid Cargo-emitted executable" >&2
    exit 1
fi

m3_test_ambiguous_json="$m3_test_scratch/discovery-ambiguous.jsonl"
jq -cn --arg executable "$m3_test_valid_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >"$m3_test_ambiguous_json"
jq -cn --arg executable "$m3_test_valid_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >>"$m3_test_ambiguous_json"
if python3 -I -S "$m3_test_resolver" \
    discover "$m3_test_discovery_root" "$m3_test_ambiguous_json" \
    >"$m3_test_scratch/discovery-ambiguous.out" 2>&1
then
    echo "resolver accepted ambiguous Cargo executable records" >&2
    exit 1
fi

m3_test_symlink_executable="$m3_test_discovery_root/debug/deps/contract-symlink"
ln -s "$m3_test_valid_executable" "$m3_test_symlink_executable"
m3_test_symlink_json="$m3_test_scratch/discovery-symlink.jsonl"
jq -cn --arg executable "$m3_test_symlink_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >"$m3_test_symlink_json"
if python3 -I -S "$m3_test_resolver" \
    discover "$m3_test_discovery_root" "$m3_test_symlink_json" \
    >"$m3_test_scratch/discovery-symlink.out" 2>&1
then
    echo "resolver accepted a symlink contract executable" >&2
    exit 1
fi

m3_test_escape_executable="$m3_test_scratch/contract-escape"
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' >"$m3_test_escape_executable"
chmod +x "$m3_test_escape_executable"
m3_test_escape_executable=$(cd "${m3_test_escape_executable%/*}" && pwd -P)/${m3_test_escape_executable##*/}
m3_test_escape_json="$m3_test_scratch/discovery-escape.jsonl"
jq -cn --arg executable "$m3_test_escape_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >"$m3_test_escape_json"
if python3 -I -S "$m3_test_resolver" \
    discover "$m3_test_discovery_root" "$m3_test_escape_json" \
    >"$m3_test_scratch/discovery-escape.out" 2>&1
then
    echo "resolver accepted a contract executable outside the fresh target root" >&2
    exit 1
fi

printf 'm3-stream-parity-runner-regression-pass: environment/artifact/discovery controls passed model-free\n'
