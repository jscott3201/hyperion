#!/bin/bash
set -euo pipefail

m3_test_repo_root=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$m3_test_repo_root"

m3_test_scratch=$(/usr/bin/mktemp -d)
m3_test_cleanup_normal=
m3_test_cleanup_normal_sibling=
m3_test_cleanup_race=
m3_test_cleanup_original=
m3_test_cleanup() {
    local m3_test_cleanup_path
    for m3_test_cleanup_path in \
        "${m3_test_cleanup_normal:-}" \
        "${m3_test_cleanup_normal_sibling:-}" \
        "${m3_test_cleanup_race:-}" \
        "${m3_test_cleanup_original:-}"
    do
        case "$m3_test_cleanup_path" in
            /private/tmp/hyperion-m3-contract.cleanup-test.*)
                /bin/rm -rf -- "$m3_test_cleanup_path"
                ;;
        esac
    done
    /bin/rm -rf -- "$m3_test_scratch"
}
trap m3_test_cleanup EXIT

# Remove inherited execution-shaping variables so each negative control selects
# exactly one rejection path. The runner itself also rejects all of these names.
while IFS= read -r m3_test_inherited_override; do
    unset "$m3_test_inherited_override"
done < <(/usr/bin/python3 -I -S -c '
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
    "RUSTUP_HOME",
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

m3_test_fake_bin="$m3_test_scratch/fake-bin"
m3_test_fake_marker="$m3_test_scratch/fake-tool-executed"
/bin/mkdir -p "$m3_test_fake_bin"
for m3_test_fake_name in \
    dirname date mkdir mv tee shasum awk jq python3 git sort head tail \
    uname sysctl sw_vers env mktemp getconf cargo rustc rustup cmake clang clang++ \
    ar ranlib xcrun make
do
    # shellcheck disable=SC2016 # The generated shim expands at execution time.
    printf '%s\n' \
        '#!/bin/bash' \
        'printf "%s %s\n" "$0" "$*" >>"${M3_TEST_FAKE_MARKER:?}"' \
        'exit 97' \
        >"$m3_test_fake_bin/$m3_test_fake_name"
    /bin/chmod +x "$m3_test_fake_bin/$m3_test_fake_name"
done

m3_test_assert_fake_unused() {
    if [[ -e "$m3_test_fake_marker" ]]; then
        echo "runner executed a caller-PATH shim" >&2
        /usr/bin/sed -n '1,80p' "$m3_test_fake_marker" >&2
        return 1
    fi
}

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
        m3_test_actual_log_sha=$(/usr/bin/shasum -a 256 \
            "$m3_test_assert_evidence/$m3_test_log_name.log" | /usr/bin/awk '{print $1}')
        m3_test_manifest_log_sha=$(/usr/bin/jq -r \
            ".logs.$m3_test_log_name.sha256" \
            "$m3_test_assert_evidence/manifest.json")
        if [[ "$m3_test_actual_log_sha" != "$m3_test_manifest_log_sha" ]]; then
            echo "manifest hash for $m3_test_log_name.log is stale" >&2
            return 1
        fi
    done
}

m3_test_assert_static_identity() {
    local m3_test_assert_evidence=$1
    /usr/bin/jq -e '
        .trust_boundary.active_same_uid_mutation_excluded == true and
        (.trust_boundary.statement | contains("active same-UID mutation")) and
        .source.repository_root_binding == "<physical-repository-root>" and
        (.source.git_dir_discovery | contains("linked-worktree gitfile")) and
        .command.source_git.environment_launcher == "/usr/bin/env -i" and
        .command.source_git.executable == "/usr/bin/git" and
        .command.source_git.inherited_environment == "cleared" and
        .command.source_git.environment.HOME == "<canonical-login-home>" and
        .command.source_git.environment.PATH == "/usr/bin:/bin:/usr/sbin:/sbin" and
        .command.source_git.environment.GIT_CONFIG_NOSYSTEM == "1" and
        .command.source_git.environment.GIT_CONFIG_SYSTEM == "/dev/null" and
        .command.source_git.environment.GIT_CONFIG_GLOBAL == "/dev/null" and
        .command.source_git.environment.GIT_ATTR_NOSYSTEM == "1" and
        .command.source_git.environment.GIT_TERMINAL_PROMPT == "0" and
        .command.source_git.environment.GIT_PAGER == "" and
        .command.source_git.environment.GIT_OPTIONAL_LOCKS == "0" and
        .command.source_git.environment.GIT_NO_LAZY_FETCH == "1" and
        .command.source_git.repository_binding.working_directory ==
          "<physical-repository-root>" and
        .command.source_git.repository_binding.work_tree ==
          "<physical-repository-root>" and
        .command.source_git.repository_binding.git_dir ==
          "repository-native-linked-worktree-aware" and
        (.command.source_git.global_options | index("--no-optional-locks")) != null and
        (.command.source_git.command_line_config_overrides |
          index("core.fsmonitor=false")) != null and
        (.command.source_git.command_line_config_overrides |
          index("core.untrackedCache=false")) != null and
        (.command.source_git.command_line_config_overrides |
          index("core.ignoreStat=false")) != null and
        (.command.source_git.command_line_config_overrides |
          index("core.hooksPath=/dev/null")) != null and
        (.command.source_git.command_line_config_overrides |
          index("diff.external=")) != null and
        (.command.source_git.local_repository_config |
          contains("command-line safety overrides")) and
        (.execution_identity.controlled_path |
          startswith("/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:"))
    ' "$m3_test_assert_evidence/manifest.json" >/dev/null
    /usr/bin/jq -e '
        .execution_identity.tools as $tools |
        [
          "bash", "dirname", "date", "mkdir", "mv", "tee", "shasum",
          "perl", "awk", "jq", "python3", "git", "sort", "head", "tail",
          "uname", "sysctl", "sw_vers", "env", "mktemp", "getconf", "cargo", "rustc",
          "rustup", "cmake", "clang", "clangxx", "ar", "ranlib", "xcrun",
          "make", "sh", "cc", "ld", "libtool", "install_name_tool",
          "metal_driver", "metal"
        ] |
        all(. as $name |
          ($tools[$name].invocation_path | startswith("/")) and
          ($tools[$name].canonical_path | startswith("/")) and
          ($tools[$name].sha256 | test("^[0-9a-f]{64}$"))
        )
    ' "$m3_test_assert_evidence/manifest.json" >/dev/null
}

# This invocation reaches fixed dirname/date/mkdir/python/shasum/jq work in the
# new runner. The prior runner would execute at least the prepended dirname shim.
m3_test_evidence="$m3_test_scratch/evidence"
set +e
/usr/bin/env -u HYPERION_12B_ARTIFACT \
    PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    scripts/run-m3-stream-parity.sh "$m3_test_evidence" \
    >"$m3_test_scratch/invocation.log" 2>&1
m3_test_status=$?
set -e
if (( m3_test_status != 64 )); then
    echo "missing-artifact runner exit was $m3_test_status, expected 64" >&2
    /usr/bin/sed -n '1,200p' "$m3_test_scratch/invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if ! /usr/bin/grep -q \
    'HYPERION_12B_ARTIFACT must be set explicitly to the pinned real artifact' \
    "$m3_test_evidence/preflight.log"
then
    echo "missing-artifact preflight did not identify the explicit artifact requirement" >&2
    exit 1
fi
if [[ -s "$m3_test_evidence/identity.log" || \
      -s "$m3_test_evidence/build.log" || \
      -s "$m3_test_evidence/test.log" ]]; then
    echo "missing-artifact runner wrote identity/build/test output before rejection" >&2
    exit 1
fi
/usr/bin/jq -e '
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
    .command.build_environment.HOME == "<fresh-runner-target-root>/build-home" and
    .command.direct_test_environment.HOME == "<fresh-runner-target-root>/test-home" and
    (.command.build_environment.TMPDIR | startswith("/private/var/folders/")) and
    .command.direct_test_environment.LC_ALL == "C" and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null and
    .execution.test_binary_path == null and
    .execution.test_binary_sha256 == null
' "$m3_test_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_evidence"
m3_test_assert_log_hashes "$m3_test_evidence"

m3_test_unknown_artifact="$m3_test_scratch/unknown-artifact"
m3_test_unknown_evidence="$m3_test_scratch/unknown-evidence"
/bin/mkdir -p "$m3_test_unknown_artifact"
printf 'unapproved reconstructed payload\n' >"$m3_test_unknown_artifact/SHA256SUMS"
m3_test_unknown_manifest_sha=$(/usr/bin/shasum -a 256 \
    "$m3_test_unknown_artifact/SHA256SUMS" | /usr/bin/awk '{print $1}')
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_unknown_artifact" \
    scripts/run-m3-stream-parity.sh "$m3_test_unknown_evidence" \
    >"$m3_test_scratch/unknown-invocation.log" 2>&1
m3_test_unknown_status=$?
set -e
if (( m3_test_unknown_status != 64 )); then
    echo "unknown-manifest runner exit was $m3_test_unknown_status, expected 64" >&2
    /usr/bin/sed -n '1,200p' "$m3_test_scratch/unknown-invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if ! /usr/bin/grep -q \
    'artifact identity manifest digest is not an approved immutable payload' \
    "$m3_test_unknown_evidence/preflight.log"
then
    echo "unknown-manifest preflight did not report the immutable allowlist failure" >&2
    exit 1
fi
/usr/bin/jq -e --arg actual "$m3_test_unknown_manifest_sha" '
    .status == "failed" and
    .failure_stage == "artifact_environment" and
    .exit_status == 64 and
    .artifact.identity_kind == "historical_sha256sums" and
    .artifact.manifest_sha256 == $actual and
    .artifact.identity == null
' "$m3_test_unknown_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_unknown_evidence"
m3_test_assert_log_hashes "$m3_test_unknown_evidence"

for m3_test_manifest_case in missing_manifest ambiguous_manifests; do
    m3_test_case_artifact="$m3_test_scratch/$m3_test_manifest_case-artifact"
    m3_test_case_evidence="$m3_test_scratch/$m3_test_manifest_case-evidence"
    /bin/mkdir -p "$m3_test_case_artifact"
    if [[ "$m3_test_manifest_case" == ambiguous_manifests ]]; then
        : >"$m3_test_case_artifact/SHA256SUMS"
        : >"$m3_test_case_artifact/PAYLOAD_SHA256SUMS"
    fi
    set +e
    PATH="$m3_test_fake_bin:$PATH" \
        M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
        HYPERION_12B_ARTIFACT="$m3_test_case_artifact" \
        scripts/run-m3-stream-parity.sh "$m3_test_case_evidence" \
        >"$m3_test_scratch/$m3_test_manifest_case-invocation.log" 2>&1
    m3_test_case_status=$?
    set -e
    if (( m3_test_case_status != 64 )); then
        echo "$m3_test_manifest_case runner exit was $m3_test_case_status, expected 64" >&2
        exit 1
    fi
    m3_test_assert_fake_unused
    if ! /usr/bin/grep -q \
        'artifact must contain exactly one recognized identity manifest' \
        "$m3_test_case_evidence/preflight.log"
    then
        echo "$m3_test_manifest_case did not fail the manifest ambiguity gate" >&2
        exit 1
    fi
    /usr/bin/jq -e '
        .status == "failed" and
        .failure_stage == "artifact_environment" and
        .exit_status == 64
    ' "$m3_test_case_evidence/manifest.json" >/dev/null
    m3_test_assert_static_identity "$m3_test_case_evidence"
    m3_test_assert_log_hashes "$m3_test_case_evidence"
done

# An ambient Cargo target runner must be rejected by name before source,
# payload verification, toolchain probes, Cargo, or the direct test can run.
m3_test_override_evidence="$m3_test_scratch/override-evidence"
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_unknown_artifact" \
    CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER="$m3_test_scratch/fake-runner" \
    scripts/run-m3-stream-parity.sh "$m3_test_override_evidence" \
    >"$m3_test_scratch/override-invocation.log" 2>&1
m3_test_override_status=$?
set -e
if (( m3_test_override_status != 1 )); then
    echo "ambient target-runner exit was $m3_test_override_status, expected 1" >&2
    /usr/bin/sed -n '1,200p' "$m3_test_scratch/override-invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if ! /usr/bin/grep -q \
    'M3 stream parity rejects ambient CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER' \
    "$m3_test_override_evidence/preflight.log"
then
    echo "ambient target-runner gate did not report the exact variable name" >&2
    exit 1
fi
/usr/bin/jq -e '
    .status == "failed" and
    .failure_stage == "ambient_execution_overrides" and
    .exit_status == 1 and
    .source.clean == false and
    .artifact.identity_kind == null and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null
' "$m3_test_override_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_override_evidence"
m3_test_assert_log_hashes "$m3_test_override_evidence"

# A caller cannot redirect Cargo configuration through HOME. The runner binds
# HOME to the canonical passwd entry before inspecting or executing Cargo.
m3_test_config_home="$m3_test_scratch/config-home"
m3_test_config_evidence="$m3_test_scratch/config-evidence"
/bin/mkdir -p "$m3_test_config_home/.cargo"
printf '%s\n' \
    '[build]' \
    'rustc-wrapper = "/definitely-not-an-approved-wrapper"' \
    >"$m3_test_config_home/.cargo/config.toml"
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HOME="$m3_test_config_home" \
    HYPERION_12B_ARTIFACT="$m3_test_unknown_artifact" \
    scripts/run-m3-stream-parity.sh "$m3_test_config_evidence" \
    >"$m3_test_scratch/config-invocation.log" 2>&1
m3_test_config_status=$?
set -e
if (( m3_test_config_status != 1 )); then
    echo "redirected HOME exit was $m3_test_config_status, expected 1" >&2
    /usr/bin/sed -n '1,200p' "$m3_test_scratch/config-invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if ! /usr/bin/grep -q 'requires HOME to equal the canonical login home' \
    "$m3_test_config_evidence/preflight.log"
then
    echo "redirected HOME did not fail the static execution identity gate" >&2
    exit 1
fi
/usr/bin/jq -e '
    .status == "failed" and
    .failure_stage == "static_execution_identity" and
    .exit_status == 1 and
    .source.clean == false and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null
' "$m3_test_config_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_config_evidence"
m3_test_assert_log_hashes "$m3_test_config_evidence"

# Use the exact approved public eight-line manifest, but empty payload files.
# The fixed shasum validates the manifest allowlist and then fails payload
# verification before any Cargo build or direct model test.
m3_test_owner_artifact="$m3_test_scratch/owner-artifact"
m3_test_owner_evidence="$m3_test_scratch/owner-evidence"
/bin/mkdir -p "$m3_test_owner_artifact"
while IFS= read -r m3_test_owner_line; do
    m3_test_owner_file=${m3_test_owner_line#*  }
    : >"$m3_test_owner_artifact/$m3_test_owner_file"
done <<'PAYLOAD_FILES'
ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4  chat_template.jinja
257501c3412dd0c5645c56a47b6c5752fbc416c586d97534bca696668644b7b0  config.json
a8349d9bd64cc5841297fcb5002f0fdc4749c473c8f1b10ea337f9ce4ee7014e  generation_config.json
318f06775a7c234e0c31c1f9971a38b6c3217d5c5afe2be8a8286fdfe4015dd9  model-00001-of-00002.safetensors
755c80994e9c8dc7c9491d5d01c1472c152da3055ce9fd35832f0c2c12f3c39f  model-00002-of-00002.safetensors
0352c33d9baee674195c874b42687e0afa0fb68b42f5c2e1a8a2fff44b125b7a  model.safetensors.index.json
cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f  tokenizer.json
a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9  tokenizer_config.json
PAYLOAD_FILES
printf '%s\n' \
    'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4  chat_template.jinja' \
    '257501c3412dd0c5645c56a47b6c5752fbc416c586d97534bca696668644b7b0  config.json' \
    'a8349d9bd64cc5841297fcb5002f0fdc4749c473c8f1b10ea337f9ce4ee7014e  generation_config.json' \
    '318f06775a7c234e0c31c1f9971a38b6c3217d5c5afe2be8a8286fdfe4015dd9  model-00001-of-00002.safetensors' \
    '755c80994e9c8dc7c9491d5d01c1472c152da3055ce9fd35832f0c2c12f3c39f  model-00002-of-00002.safetensors' \
    '0352c33d9baee674195c874b42687e0afa0fb68b42f5c2e1a8a2fff44b125b7a  model.safetensors.index.json' \
    'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f  tokenizer.json' \
    'a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9  tokenizer_config.json' \
    >"$m3_test_owner_artifact/PAYLOAD_SHA256SUMS"

# Hostile inherited Git routing, object, index, config, pager, prompt, and
# helper state must not divert the physical source preflight. The invocation
# must bind the real HEAD/tree, report a clean source, and reach the later
# approved-manifest payload checksum failure without invoking a shim/helper.
m3_test_hostile_git_evidence="$m3_test_scratch/hostile-git-evidence"
m3_test_hostile_git_dir="$m3_test_scratch/hostile-git-dir"
m3_test_hostile_git_work_tree="$m3_test_scratch/hostile-git-work-tree"
m3_test_hostile_git_objects="$m3_test_scratch/hostile-git-objects"
m3_test_hostile_git_alternates="$m3_test_scratch/hostile-git-alternates"
m3_test_hostile_git_index="$m3_test_scratch/hostile-git-index"
m3_test_hostile_git_global="$m3_test_scratch/hostile-git-global"
m3_test_hostile_git_system="$m3_test_scratch/hostile-git-system"
m3_test_hostile_git_helper="$m3_test_scratch/hostile-git-helper"
m3_test_hostile_git_marker="$m3_test_scratch/hostile-git-helper-executed"
/bin/mkdir -p \
    "$m3_test_hostile_git_dir" \
    "$m3_test_hostile_git_work_tree" \
    "$m3_test_hostile_git_objects" \
    "$m3_test_hostile_git_alternates"
: >"$m3_test_hostile_git_index"
printf '%s\n' \
    '#!/bin/bash' \
    "printf '%s\\n' 'hostile git helper executed' >'$m3_test_hostile_git_marker'" \
    'exit 97' \
    >"$m3_test_hostile_git_helper"
/bin/chmod +x "$m3_test_hostile_git_helper"
printf '%s\n' \
    '[core]' \
    "worktree = $m3_test_hostile_git_work_tree" \
    "fsmonitor = $m3_test_hostile_git_helper" \
    '[diff]' \
    "external = $m3_test_hostile_git_helper" \
    >"$m3_test_hostile_git_global"
printf '%s\n' \
    '[core]' \
    "worktree = $m3_test_hostile_git_work_tree" \
    >"$m3_test_hostile_git_system"
m3_test_expected_source_sha=$(/usr/bin/env -i \
    HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="$TMPDIR" \
    LANG=C LC_ALL=C \
    /usr/bin/git --no-optional-locks -C "$m3_test_repo_root" \
    --work-tree="$m3_test_repo_root" rev-parse --verify 'HEAD^{commit}')
m3_test_expected_source_tree_sha=$(/usr/bin/env -i \
    HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="$TMPDIR" \
    LANG=C LC_ALL=C \
    /usr/bin/git --no-optional-locks -C "$m3_test_repo_root" \
    --work-tree="$m3_test_repo_root" rev-parse --verify 'HEAD^{tree}')
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    GIT_DIR="$m3_test_hostile_git_dir" \
    GIT_COMMON_DIR="$m3_test_hostile_git_dir" \
    GIT_WORK_TREE="$m3_test_hostile_git_work_tree" \
    GIT_INDEX_FILE="$m3_test_hostile_git_index" \
    GIT_OBJECT_DIRECTORY="$m3_test_hostile_git_objects" \
    GIT_ALTERNATE_OBJECT_DIRECTORIES="$m3_test_hostile_git_alternates" \
    GIT_CONFIG="$m3_test_hostile_git_global" \
    GIT_CONFIG_GLOBAL="$m3_test_hostile_git_global" \
    GIT_CONFIG_SYSTEM="$m3_test_hostile_git_system" \
    GIT_CONFIG_NOSYSTEM=0 \
    GIT_CONFIG_COUNT=2 \
    GIT_CONFIG_KEY_0=core.fsmonitor \
    GIT_CONFIG_VALUE_0="$m3_test_hostile_git_helper" \
    GIT_CONFIG_KEY_1=core.worktree \
    GIT_CONFIG_VALUE_1="$m3_test_hostile_git_work_tree" \
    GIT_EXEC_PATH="$m3_test_fake_bin" \
    GIT_EXTERNAL_DIFF="$m3_test_hostile_git_helper" \
    GIT_PAGER="$m3_test_hostile_git_helper" \
    GIT_ASKPASS="$m3_test_hostile_git_helper" \
    GIT_TERMINAL_PROMPT=1 \
    GIT_OPTIONAL_LOCKS=1 \
    GIT_NO_LAZY_FETCH=0 \
    scripts/run-m3-stream-parity.sh "$m3_test_hostile_git_evidence" \
    >"$m3_test_scratch/hostile-git-invocation.log" 2>&1
m3_test_hostile_git_status=$?
set -e
if (( m3_test_hostile_git_status != 1 )); then
    echo "hostile-Git runner exit was $m3_test_hostile_git_status, expected 1" >&2
    /usr/bin/sed -n '1,240p' "$m3_test_scratch/hostile-git-invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if [[ -e "$m3_test_hostile_git_marker" ]]; then
    echo "source preflight invoked a hostile inherited Git helper" >&2
    exit 1
fi
if ! /usr/bin/grep -q 'FAILED' "$m3_test_hostile_git_evidence/identity.log"; then
    echo "hostile-Git control did not reach the later artifact checksum failure" >&2
    exit 1
fi
/usr/bin/jq -e \
    --arg source_sha "$m3_test_expected_source_sha" \
    --arg source_tree_sha "$m3_test_expected_source_tree_sha" '
    .status == "failed" and
    .failure_stage == "artifact_identity_preflight" and
    .exit_status == 1 and
    .source.sha == $source_sha and
    .source.tree_sha == $source_tree_sha and
    .source.clean == true and
    .artifact.identity_kind == "owner_payload_sha256sums" and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null
' "$m3_test_hostile_git_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_hostile_git_evidence"
m3_test_assert_log_hashes "$m3_test_hostile_git_evidence"

set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    scripts/run-m3-stream-parity.sh "$m3_test_owner_evidence" \
    >"$m3_test_scratch/owner-invocation.log" 2>&1
m3_test_owner_status=$?
set -e
if (( m3_test_owner_status != 1 )); then
    echo "owner checksum-failure exit was $m3_test_owner_status, expected 1" >&2
    /usr/bin/sed -n '1,240p' "$m3_test_scratch/owner-invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if ! /usr/bin/grep -q 'FAILED' "$m3_test_owner_evidence/identity.log"; then
    echo "owner checksum failure was not retained in identity.log" >&2
    exit 1
fi
/usr/bin/jq -e '
    .status == "failed" and
    .failure_stage == "artifact_identity_preflight" and
    .exit_status == 1 and
    .source.clean == true and
    .artifact.identity_kind == "owner_payload_sha256sums" and
    .artifact.manifest_sha256 == "3cee7e9c21051eb6e6857ef485b355b4a2e847901930d64620348cbc7f56c806" and
    .artifact.identity == null and
    .toolchain.rustc == null and
    .toolchain.cargo == null and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null and
    .execution.test_binary_sha256 == null
' "$m3_test_owner_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_owner_evidence"
m3_test_assert_log_hashes "$m3_test_owner_evidence"

# Exercise the exact embedded Cargo executable resolver without compiling.
m3_test_resolver="$m3_test_scratch/contract-resolver.py"
/usr/bin/awk '
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
/bin/mkdir -p "$m3_test_discovery_root/debug/deps"
m3_test_discovery_root=$(cd "$m3_test_discovery_root" && pwd -P)
m3_test_valid_executable="$m3_test_discovery_root/debug/deps/contract valid"
printf '%s\n' '#!/bin/bash' 'exit 0' >"$m3_test_valid_executable"
/bin/chmod +x "$m3_test_valid_executable"
m3_test_valid_json="$m3_test_scratch/discovery-valid.jsonl"
/usr/bin/jq -cn --arg executable "$m3_test_valid_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >"$m3_test_valid_json"
m3_test_discovered_executable=$(/usr/bin/python3 -I -S "$m3_test_resolver" \
    discover "$m3_test_discovery_root" "$m3_test_valid_json")
if [[ "$m3_test_discovered_executable" != "$m3_test_valid_executable" ]]; then
    echo "resolver did not return the one valid Cargo-emitted executable" >&2
    exit 1
fi

m3_test_ambiguous_json="$m3_test_scratch/discovery-ambiguous.jsonl"
/usr/bin/jq -cn --arg executable "$m3_test_valid_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >"$m3_test_ambiguous_json"
/usr/bin/jq -cn --arg executable "$m3_test_valid_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >>"$m3_test_ambiguous_json"
if /usr/bin/python3 -I -S "$m3_test_resolver" \
    discover "$m3_test_discovery_root" "$m3_test_ambiguous_json" \
    >"$m3_test_scratch/discovery-ambiguous.out" 2>&1
then
    echo "resolver accepted ambiguous Cargo executable records" >&2
    exit 1
fi

m3_test_symlink_executable="$m3_test_discovery_root/debug/deps/contract-symlink"
/bin/ln -s "$m3_test_valid_executable" "$m3_test_symlink_executable"
m3_test_symlink_json="$m3_test_scratch/discovery-symlink.jsonl"
/usr/bin/jq -cn --arg executable "$m3_test_symlink_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >"$m3_test_symlink_json"
if /usr/bin/python3 -I -S "$m3_test_resolver" \
    discover "$m3_test_discovery_root" "$m3_test_symlink_json" \
    >"$m3_test_scratch/discovery-symlink.out" 2>&1
then
    echo "resolver accepted a symlink contract executable" >&2
    exit 1
fi

m3_test_escape_executable="$m3_test_scratch/contract-escape"
printf '%s\n' '#!/bin/bash' 'exit 0' >"$m3_test_escape_executable"
/bin/chmod +x "$m3_test_escape_executable"
m3_test_escape_executable=$(cd "${m3_test_escape_executable%/*}" && pwd -P)/${m3_test_escape_executable##*/}
m3_test_escape_json="$m3_test_scratch/discovery-escape.jsonl"
/usr/bin/jq -cn --arg executable "$m3_test_escape_executable" '
    {reason: "compiler-artifact", target: {name: "contract", kind: ["test"]}, executable: $executable}
' >"$m3_test_escape_json"
if /usr/bin/python3 -I -S "$m3_test_resolver" \
    discover "$m3_test_discovery_root" "$m3_test_escape_json" \
    >"$m3_test_scratch/discovery-escape.out" 2>&1
then
    echo "resolver accepted a contract executable outside the fresh target root" >&2
    exit 1
fi

# Exercise the runner's exact embedded cleanup helper. Normal cleanup removes
# only its bound root and unlinks, rather than follows, an internal symlink.
m3_test_cleanup_helper="$m3_test_scratch/build-root-cleanup.py"
/usr/bin/awk '
    /^# M3_BUILD_ROOT_CLEANUP_PYTHON_BEGIN$/ { capture=1; next }
    /^# M3_BUILD_ROOT_CLEANUP_PYTHON_END$/ { capture=0; found=1; exit }
    capture { print }
    END { if (!found) exit 1 }
' scripts/run-m3-stream-parity.sh >"$m3_test_cleanup_helper"
if [[ ! -s "$m3_test_cleanup_helper" ]]; then
    echo "could not extract the runner build-root cleanup helper" >&2
    exit 1
fi
m3_test_stat_identity() {
    /usr/bin/python3 -I -S -c '
import os
import sys
metadata = os.stat(sys.argv[1], follow_symlinks=False)
print(metadata.st_dev, metadata.st_ino)
' "$1"
}

m3_test_cleanup_normal=$(/usr/bin/mktemp -d \
    /private/tmp/hyperion-m3-contract.cleanup-test.normal.XXXXXXXX)
m3_test_cleanup_normal_sibling="${m3_test_cleanup_normal}.sibling"
/bin/mkdir -p "$m3_test_cleanup_normal/nested" "$m3_test_cleanup_normal_sibling"
printf 'root sentinel\n' >"$m3_test_cleanup_normal/nested/root-sentinel"
printf 'sibling sentinel\n' >"$m3_test_cleanup_normal_sibling/sentinel"
/bin/ln -s "$m3_test_cleanup_normal_sibling" \
    "$m3_test_cleanup_normal/nested/sibling-link"
read -r m3_test_normal_dev m3_test_normal_ino \
    <<<"$(m3_test_stat_identity "$m3_test_cleanup_normal")"
/usr/bin/python3 -I -S "$m3_test_cleanup_helper" \
    /private/tmp "${m3_test_cleanup_normal##*/}" \
    "$m3_test_normal_dev" "$m3_test_normal_ino"
if [[ -e "$m3_test_cleanup_normal" || -L "$m3_test_cleanup_normal" ]]; then
    echo "normal cleanup did not remove its bound root" >&2
    exit 1
fi
if [[ ! -f "$m3_test_cleanup_normal_sibling/sentinel" ]]; then
    echo "normal cleanup followed an internal symlink or removed a sibling" >&2
    exit 1
fi
/bin/rm -rf -- "$m3_test_cleanup_normal_sibling"
m3_test_cleanup_normal=
m3_test_cleanup_normal_sibling=

# Replace the pathname after binding its dev+inode. The exact cleanup helper
# must fail before recursive traversal and leave both replacement and original.
m3_test_cleanup_race=$(/usr/bin/mktemp -d \
    /private/tmp/hyperion-m3-contract.cleanup-test.race.XXXXXXXX)
read -r m3_test_race_dev m3_test_race_ino \
    <<<"$(m3_test_stat_identity "$m3_test_cleanup_race")"
m3_test_cleanup_original="${m3_test_cleanup_race}.original"
/bin/mv "$m3_test_cleanup_race" "$m3_test_cleanup_original"
/bin/mkdir "$m3_test_cleanup_race"
printf 'replacement sentinel\n' >"$m3_test_cleanup_race/replacement-sentinel"
printf 'original sentinel\n' >"$m3_test_cleanup_original/original-sentinel"
set +e
/usr/bin/python3 -I -S "$m3_test_cleanup_helper" \
    /private/tmp "${m3_test_cleanup_race##*/}" \
    "$m3_test_race_dev" "$m3_test_race_ino" \
    >"$m3_test_scratch/cleanup-race.out" 2>&1
m3_test_cleanup_race_status=$?
set -e
if (( m3_test_cleanup_race_status == 0 )); then
    echo "cleanup accepted a replacement build-root entry" >&2
    exit 1
fi
if [[ ! -f "$m3_test_cleanup_race/replacement-sentinel" || \
      ! -f "$m3_test_cleanup_original/original-sentinel" ]]; then
    echo "cleanup recursively deleted replacement or original material" >&2
    exit 1
fi
if ! /usr/bin/grep -q 'root identity mismatch' "$m3_test_scratch/cleanup-race.out"; then
    echo "cleanup replacement control did not report identity mismatch" >&2
    exit 1
fi
/bin/rm -rf -- "$m3_test_cleanup_race" "$m3_test_cleanup_original"
m3_test_cleanup_race=
m3_test_cleanup_original=

printf '%s\n' \
    'm3-stream-parity-runner-regression-pass: identity/environment/git-routing/artifact/discovery/cleanup controls passed model-free'
