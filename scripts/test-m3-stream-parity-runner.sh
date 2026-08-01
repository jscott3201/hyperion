#!/bin/bash -p
# shellcheck disable=SC2329 # Exported probe functions are inspected by child Bash processes.
if [[ ${BASH:-} != /bin/bash || $- != *p* ]]; then
    printf '%s\n' \
        'M3 stream parity runner harness requires direct execution by fixed /bin/bash in privileged mode' >&2
    exit 127
fi
set -euo pipefail

if [[ ${M3_TEST_PRIVILEGED_SELF_PROBE:-} == 1 ]]; then
    if [[ -e ${M3_TEST_STARTUP_MARKER:?} ]]; then
        echo "privileged harness startup processed inherited BASH_ENV" >&2
        exit 1
    fi
    if declare -F m3_test_imported_function >/dev/null; then
        echo "privileged harness startup imported a caller function" >&2
        exit 1
    fi
    printf '%s\n' 'm3-privileged-harness-self-probe-pass'
    exit 0
fi

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

m3_test_startup_marker="$m3_test_scratch/inherited-startup-executed"
m3_test_startup_file="$m3_test_scratch/inherited-startup.sh"
printf '%s\n' \
    "printf '%s\\n' 'inherited shell startup executed' >'$m3_test_startup_marker'" \
    >"$m3_test_startup_file"
m3_test_imported_function() {
    printf '%s\n' 'imported caller function executed' >"$m3_test_startup_marker"
}
export -f m3_test_imported_function
M3_TEST_PRIVILEGED_SELF_PROBE=1 \
    M3_TEST_STARTUP_MARKER="$m3_test_startup_marker" \
    BASH_ENV="$m3_test_startup_file" \
    scripts/test-m3-stream-parity-runner.sh \
    >"$m3_test_scratch/privileged-harness-self-probe.log" 2>&1
if [[ -e "$m3_test_startup_marker" ]]; then
    echo "direct harness execution processed inherited shell startup state" >&2
    exit 1
fi
if ! /usr/bin/grep -qx 'm3-privileged-harness-self-probe-pass' \
    "$m3_test_scratch/privileged-harness-self-probe.log"
then
    echo "direct harness execution did not confirm privileged startup" >&2
    exit 1
fi

set +e
M3_TEST_PRIVILEGED_SELF_PROBE=1 \
    M3_TEST_STARTUP_MARKER="$m3_test_startup_marker" \
    BASH_ENV="$m3_test_startup_file" \
    /bin/bash scripts/test-m3-stream-parity-runner.sh \
    >"$m3_test_scratch/nonprivileged-harness.log" 2>&1
m3_test_nonprivileged_harness_status=$?
set -e
unset -f m3_test_imported_function
if (( m3_test_nonprivileged_harness_status != 127 )); then
    echo "non-privileged explicit harness exit was $m3_test_nonprivileged_harness_status, expected 127" >&2
    exit 1
fi
if [[ ! -e "$m3_test_startup_marker" ]]; then
    echo "explicit /bin/bash harness control did not demonstrate inherited BASH_ENV timing" >&2
    exit 1
fi
/bin/rm -f "$m3_test_startup_marker"

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
    ar ranlib xcrun xcode-select make
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

m3_test_sha256() {
    /usr/bin/python3 -I -S -c '
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
' "$1"
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
        m3_test_actual_log_sha=$(m3_test_sha256 \
            "$m3_test_assert_evidence/$m3_test_log_name.log")
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
        .command.shell.interpreter == "/bin/bash" and
        .command.shell.invocation == "direct executable" and
        .command.shell.privileged_mode_required == true and
        (.command.shell.inherited_startup_state | contains("BASH_ENV")) and
        (.command.shell.inherited_startup_state | contains("imported functions")) and
        .command.developer_tools.selection.environment_launcher == "/usr/bin/env -i" and
        .command.developer_tools.selection.executable == "/usr/bin/xcode-select" and
        .command.developer_tools.selection.argv ==
          ["/usr/bin/xcode-select", "--print-path"] and
        .command.developer_tools.selection.inherited_environment == "cleared" and
        .command.developer_tools.selection.DEVELOPER_DIR == null and
        .command.developer_tools.pin.variable == "DEVELOPER_DIR" and
        .command.developer_tools.pin.scopes == [
          "xcrun_tool_discovery",
          "source_git",
          "runner_python_helpers",
          "toolchain_queries",
          "cargo_build",
          "direct_test"
        ] and
        .command.developer_tools.xcrun.environment_launcher == "/usr/bin/env -i" and
        .command.developer_tools.xcrun.executable == "/usr/bin/xcrun" and
        .command.developer_tools.xcrun.inherited_environment == "cleared" and
        .command.developer_tools.python_helpers.executable == "/usr/bin/python3" and
        .command.hashing.environment_launcher == "/usr/bin/env -i" and
        .command.hashing.executable == "/usr/bin/shasum" and
        .command.hashing.runtime == "/usr/bin/perl" and
        .command.hashing.inherited_environment == "cleared" and
        .command.hashing.working_directory == "preserved" and
        .command.hashing.environment.HOME == "<canonical-login-home>" and
        .command.hashing.environment.PATH == "/usr/bin:/bin:/usr/sbin:/sbin" and
        (.command.hashing.environment.TMPDIR | startswith("/private/var/folders/")) and
        .command.hashing.environment.LANG == "C" and
        .command.hashing.environment.LC_ALL == "C" and
        .source.repository_root_binding == "<physical-repository-root>" and
        (.source.git_dir_discovery | contains("linked-worktree gitfile")) and
        (.source.git_dir_discovery | contains("no repository config loaded")) and
        (.source.repository_native_metadata_policy.inspection |
          contains("includes disabled")) and
        (.source.repository_native_metadata_policy.prohibited |
          index("filter.* helpers")) != null and
        .source.repository_native_metadata_policy.default_info_exclude ==
          "comment-only content allowed" and
        .source.native_index_policy.index ==
          "repository-native-linked-worktree-aware" and
        (.source.native_index_policy.required_for_clean |
          contains("exactly equal HEAD")) and
        (.source.native_index_policy.required_for_clean |
          contains("assume-unchanged and skip-worktree absent")) and
        .source.native_index_policy.prohibited_entry_flags ==
          ["assume-unchanged", "skip-worktree"] and
        (.source.native_index_policy.inspection |
          contains("ls-files --cached --stage -v -z")) and
        (.source.native_index_policy.inspection |
          contains("NUL-delimited bytes")) and
        (.source.native_index_policy.mutation |
          contains("not refreshed or modified")) and
        (.source.raw_worktree_policy.tracked_bytes |
          contains("attributes, clean/smudge filters, and textconv are not used")) and
        (.source.raw_worktree_policy.tracked_modes |
          contains("symlink modes")) and
        (.source.raw_worktree_policy.path_inventory |
          contains("untracked or ignored")) and
        (.source.raw_worktree_policy.path_inventory |
          contains("Git excludes are not used")) and
        (.source.raw_worktree_policy.generated_root_allowances["target/"] |
          contains("fresh runner-owned external directory")) and
        (.source.raw_worktree_policy.generated_root_allowances["build/"] |
          contains("Cargo OUT_DIR")) and
        (.source.raw_worktree_policy.generated_root_allowances["oracle/.venv/"] |
          contains("does not invoke the oracle environment")) and
        .command.source_git.environment_launcher == "/usr/bin/env -i" and
        .command.source_git.executable == "/usr/bin/git" and
        .command.source_git.inherited_environment == "cleared" and
        .command.source_git.environment.HOME == "<canonical-login-home>" and
        .command.source_git.environment.DEVELOPER_DIR ==
          .command.developer_tools.pin.canonical_value and
        .command.source_git.environment.PATH == "/usr/bin:/bin:/usr/sbin:/sbin" and
        .command.source_git.environment.GIT_CONFIG_NOSYSTEM == "1" and
        .command.source_git.environment.GIT_CONFIG_SYSTEM == "/dev/null" and
        .command.source_git.environment.GIT_CONFIG_GLOBAL == "/dev/null" and
        .command.source_git.environment.GIT_ATTR_NOSYSTEM == "1" and
        .command.source_git.environment.GIT_TERMINAL_PROMPT == "0" and
        .command.source_git.environment.GIT_PAGER == "" and
        .command.source_git.environment.GIT_OPTIONAL_LOCKS == "0" and
        .command.source_git.environment.GIT_NO_LAZY_FETCH == "1" and
        .command.source_git.environment.GIT_NO_REPLACE_OBJECTS == "1" and
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
          index("core.attributesFile=/dev/null")) != null and
        (.command.source_git.command_line_config_overrides |
          index("core.excludesFile=/dev/null")) != null and
        (.command.source_git.command_line_config_overrides |
          index("diff.external=")) != null and
        (.command.source_git.local_repository_config |
          contains("helper-capable routing rejected")) and
        (.command.developer_tools.xcrun.metal_resolution |
          contains("MobileAsset/cryptex")) and
        .command.toolchain_query_environment.DEVELOPER_DIR ==
          .command.developer_tools.pin.canonical_value and
        .command.build_environment.DEVELOPER_DIR ==
          .command.developer_tools.pin.canonical_value and
        .command.direct_test_environment.DEVELOPER_DIR ==
          .command.developer_tools.pin.canonical_value and
        .command.developer_tools.xcrun.DEVELOPER_DIR ==
          .command.developer_tools.pin.canonical_value and
        .command.developer_tools.python_helpers.DEVELOPER_DIR ==
          .command.developer_tools.pin.canonical_value and
        (.execution_identity.controlled_path |
          startswith("/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:"))
    ' "$m3_test_assert_evidence/manifest.json" >/dev/null
    /usr/bin/jq -e '
        .execution_identity.tools as $tools |
        ($tools.shasum.invocation_path == "/usr/bin/shasum") and
        ($tools.perl.invocation_path == "/usr/bin/perl") and
        ($tools.xcode_select.invocation_path == "/usr/bin/xcode-select") and
        (.execution_identity.xcode_binding as $binding |
          ($binding.developer_dir.invocation_path | startswith("/")) and
          ($binding.developer_dir.canonical_path | startswith("/")) and
          $binding.environment_pin.DEVELOPER_DIR ==
            $binding.developer_dir.canonical_path and
          ($binding.postflight_verified | type) == "boolean" and
          .command.developer_tools.pin.canonical_value ==
            $binding.developer_dir.canonical_path and
          $binding.metal_driver.invocation_path ==
            $tools.metal_driver.invocation_path and
          $binding.metal_driver.canonical_path ==
            $tools.metal_driver.canonical_path and
          $binding.metal_driver.sha256 == $tools.metal_driver.sha256 and
          ($binding.metal_driver.sha256 | test("^[0-9a-f]{64}$")) and
          ($binding.metal_driver.resolution |
            contains("fixed xcrun under canonical DEVELOPER_DIR pin"))) and
        ([
          "bash", "dirname", "date", "mkdir", "mv", "tee", "shasum",
          "perl", "awk", "jq", "python3", "git", "sort", "head", "tail",
          "uname", "sysctl", "sw_vers", "env", "mktemp", "getconf", "cargo", "rustc",
          "rustup", "cmake", "clang", "clangxx", "ar", "ranlib", "xcrun",
          "xcode_select",
          "make", "sh", "cc", "ld", "libtool", "install_name_tool",
          "metal_driver", "metal"
        ] |
        all(. as $name |
          ($tools[$name].invocation_path | startswith("/")) and
          ($tools[$name].canonical_path | startswith("/")) and
          ($tools[$name].sha256 | test("^[0-9a-f]{64}$"))
        ))
    ' "$m3_test_assert_evidence/manifest.json" >/dev/null
}

# This invocation reaches fixed dirname/date/mkdir/python/shasum/jq work in the
# new runner. The prior runner would execute at least the prepended dirname shim.
m3_test_evidence="$m3_test_scratch/evidence"
m3_test_hostile_developer_dir="$m3_test_scratch/hostile-developer-dir"
set +e
/usr/bin/env -u HYPERION_12B_ARTIFACT \
    PATH="$m3_test_fake_bin:$PATH" \
    DEVELOPER_DIR="$m3_test_hostile_developer_dir" \
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
/usr/bin/jq -e --arg hostile_developer_dir "$m3_test_hostile_developer_dir" '
    .schema == "hyperion.m3-stream-parity-evidence.v1" and
    .status == "failed" and
    .failure_stage == "artifact_environment" and
    .exit_status == 64 and
    .source.clean == true and
    .fixture.expected_sha256 == "086ca72232de415973564b2c6028c98a7063f7d024b85411512071650c86cf3d" and
    .fixture.actual_sha256 == null and
    .artifact.expected_historical_manifest_sha256 == "9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144" and
    .artifact.expected_owner_payload_manifest_sha256 == "3cee7e9c21051eb6e6857ef485b355b4a2e847901930d64620348cbc7f56c806" and
    .artifact.identity_kind == null and
    .artifact.manifest_sha256 == null and
    .artifact.identity == null and
    .command.developer_tools.pin.canonical_value != $hostile_developer_dir and
    .execution_identity.xcode_binding.postflight_verified == false and
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

# Explicit /bin/bash invocation bypasses the required privileged shebang mode.
# BASH_ENV executes before runner code in this unsupported form, but the runner
# must then fail closed before it initializes an evidence receipt.
m3_test_nonprivileged_runner_evidence="$m3_test_scratch/nonprivileged-runner-evidence"
set +e
BASH_ENV="$m3_test_startup_file" \
    /bin/bash scripts/run-m3-stream-parity.sh \
    "$m3_test_nonprivileged_runner_evidence" \
    >"$m3_test_scratch/nonprivileged-runner.log" 2>&1
m3_test_nonprivileged_runner_status=$?
set -e
if (( m3_test_nonprivileged_runner_status != 127 )); then
    echo "non-privileged explicit runner exit was $m3_test_nonprivileged_runner_status, expected 127" >&2
    exit 1
fi
if [[ ! -e "$m3_test_startup_marker" ]]; then
    echo "explicit /bin/bash runner control did not demonstrate inherited BASH_ENV timing" >&2
    exit 1
fi
if [[ -e "$m3_test_nonprivileged_runner_evidence/manifest.json" ]] && \
    /usr/bin/jq -e '.status == "passed" and .exit_status == 0' \
        "$m3_test_nonprivileged_runner_evidence/manifest.json" >/dev/null 2>&1
then
    echo "non-privileged explicit runner created a passing receipt" >&2
    exit 1
fi
if [[ -e "$m3_test_nonprivileged_runner_evidence" ]]; then
    echo "non-privileged explicit runner initialized evidence before failing closed" >&2
    exit 1
fi
/bin/rm -f "$m3_test_startup_marker"

m3_test_unknown_artifact="$m3_test_scratch/unknown-artifact"
m3_test_unknown_evidence="$m3_test_scratch/unknown-evidence"
/bin/mkdir -p "$m3_test_unknown_artifact"
printf 'unapproved reconstructed payload\n' >"$m3_test_unknown_artifact/SHA256SUMS"
m3_test_unknown_manifest_sha=$(m3_test_sha256 \
    "$m3_test_unknown_artifact/SHA256SUMS")
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

# Build a disposable repository plus linked worktree containing an exact copy
# of the runner. Two unusual tracked pathname byte sequences are changed only
# after their native linked-worktree index entries receive the two source-hiding
# flags. The ordinary status gate is therefore empty, but the exact runner must
# reject both flags before artifact identity, fixture, build, or test work.
m3_test_index_git() {
    local m3_test_index_git_root=$1
    shift
    /usr/bin/env -i \
        HOME="$HOME" \
        PATH=/usr/bin:/bin:/usr/sbin:/sbin \
        TMPDIR="$TMPDIR" \
        LANG=C LC_ALL=C \
        GIT_CONFIG_NOSYSTEM=1 \
        GIT_CONFIG_SYSTEM=/dev/null \
        GIT_CONFIG_GLOBAL=/dev/null \
        GIT_ATTR_NOSYSTEM=1 \
        GIT_TERMINAL_PROMPT=0 \
        GIT_PAGER= PAGER= \
        GIT_OPTIONAL_LOCKS=0 \
        GIT_NO_LAZY_FETCH=1 \
        /usr/bin/git \
        --no-optional-locks \
        -C "$m3_test_index_git_root" \
        -c core.fsmonitor=false \
        -c core.untrackedCache=false \
        -c core.ignoreStat=false \
        -c core.trustctime=true \
        -c core.checkStat=default \
        -c core.fileMode=true \
        -c core.symlinks=true \
        -c core.hooksPath=/dev/null \
        -c core.pager= \
        -c pager.status=false \
        -c diff.external= \
        -c diff.trustExitCode=false \
        "$@"
}

m3_test_index_repo="$m3_test_scratch/index-flags-repository"
m3_test_index_worktree="$m3_test_scratch/index-flags-linked-worktree"
m3_test_index_evidence="$m3_test_scratch/index-flags-evidence"
m3_test_assume_name=$'sentinels/assume-unchanged\nsentinel.txt'
m3_test_skip_name=$'sentinels/skip-worktree\tsentinel.txt'
/bin/mkdir -p "$m3_test_index_repo/scripts" "$m3_test_index_repo/sentinels"
m3_test_index_git "$m3_test_index_repo" init --quiet
/bin/cp scripts/run-m3-stream-parity.sh \
    "$m3_test_index_repo/scripts/run-m3-stream-parity.sh"
/bin/chmod +x "$m3_test_index_repo/scripts/run-m3-stream-parity.sh"
printf '%s\n' 'assume-unchanged committed bytes' \
    >"$m3_test_index_repo/$m3_test_assume_name"
printf '%s\n' 'skip-worktree committed bytes' \
    >"$m3_test_index_repo/$m3_test_skip_name"
m3_test_index_git "$m3_test_index_repo" add -- \
    scripts/run-m3-stream-parity.sh \
    "$m3_test_assume_name" \
    "$m3_test_skip_name"
m3_test_index_git "$m3_test_index_repo" \
    -c user.name=m3-index-control \
    -c user.email=m3-index-control.invalid \
    commit --quiet -m 'test: seed hidden-index control'
m3_test_index_git "$m3_test_index_repo" worktree add --quiet \
    -b index-flags-control "$m3_test_index_worktree"

if [[ "$(m3_test_sha256 scripts/run-m3-stream-parity.sh)" != \
      "$(m3_test_sha256 "$m3_test_index_worktree/scripts/run-m3-stream-parity.sh")" ]]; then
    echo "disposable linked worktree does not contain the exact runner copy" >&2
    exit 1
fi
m3_test_assume_baseline_sha=$(m3_test_sha256 \
    "$m3_test_index_worktree/$m3_test_assume_name")
m3_test_skip_baseline_sha=$(m3_test_sha256 \
    "$m3_test_index_worktree/$m3_test_skip_name")
m3_test_index_git "$m3_test_index_worktree" update-index \
    --assume-unchanged -- "$m3_test_assume_name"
m3_test_index_git "$m3_test_index_worktree" update-index \
    --skip-worktree -- "$m3_test_skip_name"
printf '%s\n' 'assume-unchanged changed bytes compiled from disk' \
    >"$m3_test_index_worktree/$m3_test_assume_name"
printf '%s\n' 'skip-worktree changed bytes compiled from disk' \
    >"$m3_test_index_worktree/$m3_test_skip_name"
if [[ "$m3_test_assume_baseline_sha" == \
      "$(m3_test_sha256 "$m3_test_index_worktree/$m3_test_assume_name")" || \
      "$m3_test_skip_baseline_sha" == \
      "$(m3_test_sha256 "$m3_test_index_worktree/$m3_test_skip_name")" ]]; then
    echo "disposable index control did not change both tracked sentinels" >&2
    exit 1
fi

m3_test_index_listing="$m3_test_scratch/index-flags-listing.bin"
m3_test_index_git "$m3_test_index_worktree" \
    ls-files --cached --stage -v -z >"$m3_test_index_listing"
/usr/bin/python3 -I -S - \
    "$m3_test_index_listing" "$m3_test_assume_name" "$m3_test_skip_name" <<'PY'
import os
import sys

with open(sys.argv[1], "rb") as stream:
    data = stream.read()
if not data or not data.endswith(b"\0"):
    raise SystemExit("disposable index listing is not a complete NUL stream")

tags = {}
for record in data[:-1].split(b"\0"):
    separator = record.find(b"\t", 2)
    if len(record) < 4 or record[1:2] != b" " or separator < 0:
        raise SystemExit("disposable index listing contains a malformed record")
    tags[record[separator + 1 :]] = record[0:1]

assume_name = os.fsencode(sys.argv[2])
skip_name = os.fsencode(sys.argv[3])
if tags.get(assume_name) != b"h":
    raise SystemExit("assume-unchanged sentinel is not present with tag h")
if tags.get(skip_name) != b"S":
    raise SystemExit("skip-worktree sentinel is not present with tag S")
PY

m3_test_index_status="$m3_test_scratch/index-flags-status.bin"
m3_test_index_git "$m3_test_index_worktree" status \
    --porcelain=v1 -z --untracked-files=all --ignore-submodules=none \
    >"$m3_test_index_status"
if [[ -s "$m3_test_index_status" ]]; then
    echo "disposable status did not hide both changed flagged sentinels" >&2
    exit 1
fi

m3_test_imported_function() {
    printf '%s\n' 'imported caller function executed' >"$m3_test_startup_marker"
}
export -f m3_test_imported_function
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    BASH_ENV="$m3_test_startup_file" \
    ENV="$m3_test_startup_file" \
    "$m3_test_index_worktree/scripts/run-m3-stream-parity.sh" \
    "$m3_test_index_evidence" \
    >"$m3_test_scratch/index-flags-invocation.log" 2>&1
m3_test_index_status_code=$?
set -e
unset -f m3_test_imported_function
if (( m3_test_index_status_code != 1 )); then
    echo "native-index flags runner exit was $m3_test_index_status_code, expected 1" >&2
    /usr/bin/sed -n '1,240p' "$m3_test_scratch/index-flags-invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if [[ -e "$m3_test_startup_marker" ]]; then
    echo "native-index flags control processed inherited shell startup state" >&2
    exit 1
fi
if ! /usr/bin/grep -q \
    'source preflight exact HEAD/index/raw-worktree attestation failed' \
    "$m3_test_index_evidence/preflight.log" || \
   ! /usr/bin/grep -q 'flags=assume-unchanged path_hex=' \
    "$m3_test_index_evidence/preflight.log" || \
   ! /usr/bin/grep -q 'flags=skip-worktree path_hex=' \
    "$m3_test_index_evidence/preflight.log"
then
    echo "native-index flags control did not report both prohibited flag classes" >&2
    exit 1
fi
if [[ -s "$m3_test_index_evidence/identity.log" || \
      -s "$m3_test_index_evidence/build.log" || \
      -s "$m3_test_index_evidence/test.log" ]]; then
    echo "native-index flags control reached artifact identity, build, or test work" >&2
    exit 1
fi
/usr/bin/jq -e '
    .status == "failed" and
    .failure_stage == "source_preflight" and
    .exit_status == 1 and
    .source.sha == null and
    .source.tree_sha == null and
    .source.clean == false and
    .fixture.actual_sha256 == null and
    .artifact.identity_kind == null and
    .artifact.identity == null and
    .toolchain.rustc == null and
    .toolchain.cargo == null and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null and
    .execution.test_binary_path == null and
    .execution.test_binary_sha256 == null
' "$m3_test_index_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_index_evidence"
m3_test_assert_log_hashes "$m3_test_index_evidence"

# Poison repository-native config, include routing, filter/fsmonitor helpers,
# info attributes/excludes, and an ignored project Cargo config in a disposable
# linked worktree. The runner must reject the native metadata before artifact,
# build, or test work and must not execute either the Git helper or Cargo wrapper.
m3_test_native_repo="$m3_test_scratch/native-metadata-repository"
m3_test_native_worktree="$m3_test_scratch/native-metadata-linked-worktree"
m3_test_native_evidence="$m3_test_scratch/native-metadata-evidence"
m3_test_native_marker="$m3_test_scratch/native-metadata-helper-executed"
/bin/mkdir -p "$m3_test_native_repo/scripts"
m3_test_index_git "$m3_test_native_repo" init --quiet
/bin/cp scripts/run-m3-stream-parity.sh \
    "$m3_test_native_repo/scripts/run-m3-stream-parity.sh"
/bin/chmod +x "$m3_test_native_repo/scripts/run-m3-stream-parity.sh"
printf '%s\n' '/.cargo/config' >"$m3_test_native_repo/.gitignore"
m3_test_index_git "$m3_test_native_repo" add -- \
    scripts/run-m3-stream-parity.sh .gitignore
m3_test_index_git "$m3_test_native_repo" \
    -c user.name=m3-native-metadata-control \
    -c user.email=m3-native-metadata-control.invalid \
    commit --quiet -m 'test: seed native metadata control'
m3_test_index_git "$m3_test_native_repo" worktree add --quiet \
    -b native-metadata-control "$m3_test_native_worktree"

m3_test_native_common="$m3_test_native_repo/.git"
m3_test_native_helper="$m3_test_native_common/hostile-helper"
m3_test_native_include="$m3_test_native_common/hostile-included-config"
m3_test_native_external_excludes="$m3_test_native_common/hostile-excludes"
printf '%s\n' \
    '#!/bin/bash' \
    "printf '%s\\n' 'hostile native helper executed' >>'$m3_test_native_marker'" \
    "printf '%s\\n' 'canonical hostile filter output'" \
    'exit 0' \
    >"$m3_test_native_helper"
/bin/chmod +x "$m3_test_native_helper"
/bin/mkdir -p "$m3_test_native_worktree/.cargo"
printf '%s\n' '[build]' "rustc-wrapper = \"$m3_test_native_helper\"" \
    >"$m3_test_native_worktree/.cargo/config"

# The tracked .gitignore hides this project-level Cargo config from ordinary
# status. With otherwise benign repository metadata, physical inventory must
# still report the byte path and reject before the wrapper can execute.
m3_test_ignored_cargo_evidence="$m3_test_scratch/ignored-cargo-evidence"
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    "$m3_test_native_worktree/scripts/run-m3-stream-parity.sh" \
    "$m3_test_ignored_cargo_evidence" \
    >"$m3_test_scratch/ignored-cargo-invocation.log" 2>&1
m3_test_ignored_cargo_status=$?
set -e
if (( m3_test_ignored_cargo_status != 1 )); then
    echo "ignored-Cargo runner exit was $m3_test_ignored_cargo_status, expected 1" >&2
    /usr/bin/sed -n '1,240p' \
        "$m3_test_scratch/ignored-cargo-invocation.log" >&2
    exit 1
fi
if [[ -e "$m3_test_native_marker" ]]; then
    echo "ignored-Cargo source preflight executed the hostile wrapper" >&2
    exit 1
fi
if ! /usr/bin/grep -q \
    'source preflight exact HEAD/index/raw-worktree attestation failed' \
    "$m3_test_ignored_cargo_evidence/preflight.log" || \
   ! /usr/bin/grep -q \
    'untracked or ignored physical source path path_hex=2e636172676f2f636f6e666967' \
    "$m3_test_ignored_cargo_evidence/preflight.log"
then
    echo "ignored-Cargo control did not report the physical .cargo/config path" >&2
    exit 1
fi
if [[ -s "$m3_test_ignored_cargo_evidence/identity.log" || \
      -s "$m3_test_ignored_cargo_evidence/build.log" || \
      -s "$m3_test_ignored_cargo_evidence/test.log" ]]; then
    echo "ignored-Cargo control reached artifact identity, build, or test work" >&2
    exit 1
fi
/usr/bin/jq -e '
    .status == "failed" and
    .failure_stage == "source_preflight" and
    .source.sha == null and
    .source.clean == false and
    .artifact.identity_kind == null and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null
' "$m3_test_ignored_cargo_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_ignored_cargo_evidence"
m3_test_assert_log_hashes "$m3_test_ignored_cargo_evidence"

printf '%s\n' \
    '[filter "included-hostile"]' \
    "clean = $m3_test_native_helper" \
    "smudge = $m3_test_native_helper" \
    '[diff "included-hostile"]' \
    "textconv = $m3_test_native_helper" \
    >"$m3_test_native_include"
printf '%s\n' '/.cargo/config' >"$m3_test_native_external_excludes"
m3_test_index_git "$m3_test_native_worktree" config --local \
    core.excludesFile "$m3_test_native_external_excludes"
m3_test_index_git "$m3_test_native_worktree" config --local \
    filter.direct-hostile.clean "$m3_test_native_helper"
m3_test_index_git "$m3_test_native_worktree" config --local \
    include.path "$m3_test_native_include"
m3_test_index_git "$m3_test_native_worktree" config --local \
    core.fsmonitor "$m3_test_native_helper"
printf '%s\n' '* filter=included-hostile diff=included-hostile' \
    >"$m3_test_native_common/info/attributes"
printf '%s\n' '# hostile active native exclude follows' '/.cargo/config' \
    >"$m3_test_native_common/info/exclude"

# Exercise the poison once during setup, then remove the marker. The runner
# itself must neither execute the helper nor reach Cargo.
set +e
m3_test_index_git "$m3_test_native_worktree" status \
    --porcelain=v1 -z --untracked-files=all --ignore-submodules=none \
    >"$m3_test_scratch/native-metadata-status.bin" 2>/dev/null
set -e
/bin/rm -f -- "$m3_test_native_marker"
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    "$m3_test_native_worktree/scripts/run-m3-stream-parity.sh" \
    "$m3_test_native_evidence" \
    >"$m3_test_scratch/native-metadata-invocation.log" 2>&1
m3_test_native_status=$?
set -e
if (( m3_test_native_status != 1 )); then
    echo "native-metadata runner exit was $m3_test_native_status, expected 1" >&2
    /usr/bin/sed -n '1,240p' \
        "$m3_test_scratch/native-metadata-invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if [[ -e "$m3_test_native_marker" ]]; then
    echo "native-metadata preflight executed a hostile Git/Cargo helper" >&2
    exit 1
fi
if ! /usr/bin/grep -q \
    'rejected dangerous or unreadable repository-native Git metadata' \
    "$m3_test_native_evidence/preflight.log" || \
   ! /usr/bin/grep -Eq \
    'include\.path|filter\.direct-hostile\.clean|core\.excludesfile|core\.fsmonitor' \
    "$m3_test_native_evidence/preflight.log"
then
    echo "native-metadata control did not report prohibited local config" >&2
    exit 1
fi
if [[ -s "$m3_test_native_evidence/identity.log" || \
      -s "$m3_test_native_evidence/build.log" || \
      -s "$m3_test_native_evidence/test.log" ]]; then
    echo "native-metadata control reached artifact identity, build, or test work" >&2
    exit 1
fi
/usr/bin/jq -e '
    .status == "failed" and
    .failure_stage == "source_preflight" and
    .exit_status == 1 and
    .source.sha == null and
    .source.tree_sha == null and
    .source.clean == false and
    .artifact.identity_kind == null and
    .artifact.identity == null and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null
' "$m3_test_native_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_native_evidence"
m3_test_assert_log_hashes "$m3_test_native_evidence"

# Demonstrate the original tracked-byte bypass independently: Git status is
# empty because a configured clean filter canonicalizes a poisoned raw file,
# but the runner rejects the helper-capable native config before it can execute.
m3_test_filter_repo="$m3_test_scratch/clean-filter-repository"
m3_test_filter_worktree="$m3_test_scratch/clean-filter-linked-worktree"
m3_test_filter_evidence="$m3_test_scratch/clean-filter-evidence"
m3_test_filter_marker="$m3_test_scratch/clean-filter-helper-executed"
m3_test_filter_helper="$m3_test_scratch/clean-filter-helper"
/bin/mkdir -p "$m3_test_filter_repo/scripts"
m3_test_index_git "$m3_test_filter_repo" init --quiet
printf '%s\n' \
    '#!/bin/bash' \
    "printf '%s\\n' 'clean filter executed' >>'$m3_test_filter_marker'" \
    "printf '%s\\n' 'canonical tracked bytes'" \
    >"$m3_test_filter_helper"
/bin/chmod +x "$m3_test_filter_helper"
/bin/cp scripts/run-m3-stream-parity.sh \
    "$m3_test_filter_repo/scripts/run-m3-stream-parity.sh"
/bin/chmod +x "$m3_test_filter_repo/scripts/run-m3-stream-parity.sh"
printf '%s\n' 'sentinel.txt filter=hostile-clean' \
    >"$m3_test_filter_repo/.gitattributes"
printf '%s\n' 'canonical tracked bytes' >"$m3_test_filter_repo/sentinel.txt"
m3_test_index_git "$m3_test_filter_repo" config --local \
    filter.hostile-clean.clean "$m3_test_filter_helper"
m3_test_index_git "$m3_test_filter_repo" add -- \
    scripts/run-m3-stream-parity.sh .gitattributes sentinel.txt
m3_test_index_git "$m3_test_filter_repo" \
    -c user.name=m3-clean-filter-control \
    -c user.email=m3-clean-filter-control.invalid \
    commit --quiet -m 'test: seed clean-filter control'
m3_test_index_git "$m3_test_filter_repo" worktree add --quiet \
    -b clean-filter-control "$m3_test_filter_worktree"
printf '%s\n' 'poisoned tracked bytes!' \
    >"$m3_test_filter_worktree/sentinel.txt"
m3_test_filter_raw_sha=$(m3_test_sha256 \
    "$m3_test_filter_worktree/sentinel.txt")
m3_test_filter_head_sha=$(m3_test_index_git "$m3_test_filter_worktree" \
    show HEAD:sentinel.txt | /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}')
if [[ "$m3_test_filter_raw_sha" == "$m3_test_filter_head_sha" ]]; then
    echo "clean-filter control did not change raw tracked bytes" >&2
    exit 1
fi
m3_test_filter_status="$m3_test_scratch/clean-filter-status.bin"
m3_test_index_git "$m3_test_filter_worktree" status \
    --porcelain=v1 -z --untracked-files=all --ignore-submodules=none \
    >"$m3_test_filter_status"
if [[ -s "$m3_test_filter_status" ]]; then
    echo "configured clean filter did not hide the raw tracked-byte change" >&2
    exit 1
fi
/bin/rm -f -- "$m3_test_filter_marker"
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    "$m3_test_filter_worktree/scripts/run-m3-stream-parity.sh" \
    "$m3_test_filter_evidence" \
    >"$m3_test_scratch/clean-filter-invocation.log" 2>&1
m3_test_filter_runner_status=$?
set -e
if (( m3_test_filter_runner_status != 1 )); then
    echo "clean-filter runner exit was $m3_test_filter_runner_status, expected 1" >&2
    /usr/bin/sed -n '1,240p' "$m3_test_scratch/clean-filter-invocation.log" >&2
    exit 1
fi
if [[ -e "$m3_test_filter_marker" ]]; then
    echo "source preflight executed the configured hostile clean filter" >&2
    exit 1
fi
if ! /usr/bin/grep -q 'filter.hostile-clean.clean' \
    "$m3_test_filter_evidence/preflight.log"; then
    echo "clean-filter control did not report the prohibited local filter" >&2
    exit 1
fi
if [[ -s "$m3_test_filter_evidence/identity.log" || \
      -s "$m3_test_filter_evidence/build.log" || \
      -s "$m3_test_filter_evidence/test.log" ]]; then
    echo "clean-filter control reached artifact identity, build, or test work" >&2
    exit 1
fi
/usr/bin/jq -e '
    .status == "failed" and
    .failure_stage == "source_preflight" and
    .source.sha == null and
    .source.clean == false and
    .execution.build_exit_status == null and
    .execution.direct_test_exit_status == null
' "$m3_test_filter_evidence/manifest.json" >/dev/null
m3_test_assert_static_identity "$m3_test_filter_evidence"
m3_test_assert_log_hashes "$m3_test_filter_evidence"

m3_test_hostile_perl_dir="$m3_test_scratch/hostile-perl"
m3_test_hostile_perl_marker="$m3_test_scratch/hostile-perl-executed"
/bin/mkdir -p "$m3_test_hostile_perl_dir"
printf '%s\n' \
    'package M3Hostile;' \
    'use strict;' \
    'use warnings;' \
    "BEGIN { open my \$marker, '>', '$m3_test_hostile_perl_marker' or die \$!; print {\$marker} \"hostile Perl startup executed\\n\"; close \$marker or die \$!; }" \
    '1;' \
    >"$m3_test_hostile_perl_dir/M3Hostile.pm"

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
m3_test_imported_function() {
    printf '%s\n' 'imported caller function executed' >"$m3_test_startup_marker"
}
export -f m3_test_imported_function
set +e
PATH="$m3_test_fake_bin:$PATH" \
    M3_TEST_FAKE_MARKER="$m3_test_fake_marker" \
    HYPERION_12B_ARTIFACT="$m3_test_owner_artifact" \
    BASH_ENV="$m3_test_startup_file" \
    ENV="$m3_test_startup_file" \
    PERL5OPT=-MM3Hostile \
    PERL5LIB="$m3_test_hostile_perl_dir" \
    PERLLIB="$m3_test_hostile_perl_dir" \
    PERL_LOCAL_LIB_ROOT="$m3_test_hostile_perl_dir" \
    PERL_MB_OPT="--install_base $m3_test_hostile_perl_dir" \
    PERL_MM_OPT="INSTALL_BASE=$m3_test_hostile_perl_dir" \
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
unset -f m3_test_imported_function
if (( m3_test_hostile_git_status != 1 )); then
    echo "hostile-Git runner exit was $m3_test_hostile_git_status, expected 1" >&2
    /usr/bin/sed -n '1,240p' "$m3_test_scratch/hostile-git-invocation.log" >&2
    exit 1
fi
m3_test_assert_fake_unused
if [[ -e "$m3_test_startup_marker" ]]; then
    echo "direct runner execution processed inherited shell startup state" >&2
    exit 1
fi
if [[ -e "$m3_test_hostile_perl_marker" ]]; then
    echo "controlled shasum processed hostile inherited Perl startup state" >&2
    exit 1
fi
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

# Exercise the exact embedded selected-Xcode binding helper without a model,
# network, or mutation of the system developer-tool selection. Both a directly
# selected versioned app and the unversioned symlink spelling must bind to the
# same canonical developer directory. Prefix-boundary escape and cross-Xcode
# substitution controls must fail closed.
m3_test_xcode_binding_helper="$m3_test_scratch/xcode-binding.py"
/usr/bin/awk '
    /^# M3_XCODE_BINDING_PYTHON_BEGIN$/ { capture=1; next }
    /^# M3_XCODE_BINDING_PYTHON_END$/ { capture=0; found=1; exit }
    capture { print }
    END { if (!found) exit 1 }
' scripts/run-m3-stream-parity.sh >"$m3_test_xcode_binding_helper"
if [[ ! -s "$m3_test_xcode_binding_helper" ]]; then
    echo "could not extract the runner selected-Xcode binding helper" >&2
    exit 1
fi

m3_test_xcode_layout="$m3_test_scratch/xcode-layout"
m3_test_versioned_developer="$m3_test_xcode_layout/Xcode_26.2.app/Contents/Developer"
m3_test_versioned_driver="$m3_test_versioned_developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal"
/bin/mkdir -p "${m3_test_versioned_driver%/*}"
printf '%s\n' '#!/bin/bash' 'exit 0' >"$m3_test_versioned_driver"
/bin/chmod +x "$m3_test_versioned_driver"
m3_test_versioned_developer_canonical=$(cd "$m3_test_versioned_developer" && pwd -P)
m3_test_versioned_driver_canonical=$(cd "${m3_test_versioned_driver%/*}" && pwd -P)/metal
m3_test_versioned_driver_sha=$(m3_test_sha256 "$m3_test_versioned_driver")
m3_test_versioned_selection=$(/usr/bin/python3 -I -S \
    "$m3_test_xcode_binding_helper" \
    selection "$m3_test_versioned_developer")
/usr/bin/jq -e \
    --arg invocation "$m3_test_versioned_developer" \
    --arg canonical "$m3_test_versioned_developer_canonical" '
    .developer_dir.invocation_path == $invocation and
    .developer_dir.canonical_path == $canonical and
    has("environment_pin") == false and
    has("metal_driver") == false
' <<<"$m3_test_versioned_selection" >/dev/null
m3_test_versioned_binding=$(/usr/bin/python3 -I -S \
    "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_versioned_developer" \
    "$m3_test_versioned_developer_canonical" \
    "$m3_test_versioned_driver")
/usr/bin/jq -e \
    --arg invocation "$m3_test_versioned_developer" \
    --arg canonical "$m3_test_versioned_developer_canonical" \
    --arg driver_invocation "$m3_test_versioned_driver" \
    --arg driver "$m3_test_versioned_driver_canonical" \
    --arg driver_sha "$m3_test_versioned_driver_sha" '
    .developer_dir.invocation_path == $invocation and
    .developer_dir.canonical_path == $canonical and
    .environment_pin.DEVELOPER_DIR == $canonical and
    .metal_driver.invocation_path == $driver_invocation and
    .metal_driver.canonical_path == $driver and
    .metal_driver.sha256 == $driver_sha and
    (.metal_driver.resolution | contains("fixed xcrun"))
' <<<"$m3_test_versioned_binding" >/dev/null

/bin/ln -s Xcode_26.2.app "$m3_test_xcode_layout/Xcode.app"
m3_test_unversioned_developer="$m3_test_xcode_layout/Xcode.app/Contents/Developer"
m3_test_unversioned_driver="$m3_test_unversioned_developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal"
m3_test_unversioned_binding=$(/usr/bin/python3 -I -S \
    "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_unversioned_developer" \
    "$m3_test_versioned_developer_canonical" \
    "$m3_test_unversioned_driver")
/usr/bin/jq -e \
    --arg invocation "$m3_test_unversioned_developer" \
    --arg canonical "$m3_test_versioned_developer_canonical" \
    --arg driver "$m3_test_versioned_driver_canonical" '
    .developer_dir.invocation_path == $invocation and
    .developer_dir.canonical_path == $canonical and
    .environment_pin.DEVELOPER_DIR == $canonical and
    .metal_driver.invocation_path ==
      ($invocation + "/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal") and
    .metal_driver.canonical_path == $driver and
    (.metal_driver.sha256 | test("^[0-9a-f]{64}$"))
' <<<"$m3_test_unversioned_binding" >/dev/null

if /usr/bin/python3 -I -S "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_unversioned_developer" \
    "$m3_test_unversioned_developer" \
    "$m3_test_unversioned_driver" \
    >"$m3_test_scratch/xcode-noncanonical-pin.out" 2>&1
then
    echo "selected-Xcode binding accepted a noncanonical DEVELOPER_DIR pin" >&2
    exit 1
fi
if ! /usr/bin/grep -q 'pinned DEVELOPER_DIR is not canonical' \
    "$m3_test_scratch/xcode-noncanonical-pin.out"
then
    echo "selected-Xcode pin control did not report the noncanonical path" >&2
    exit 1
fi

# Hosted macOS images can install Metal as an Apple MobileAsset mounted by
# cryptexd. The exact pinned-xcrun result is valid outside Xcode.app and is
# bound by invocation path, canonical path, executable mode, and digest.
m3_test_cryptex_driver="$m3_test_xcode_layout/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.42.0.3a76QH/Metal.xctoolchain/usr/metal/current/bin/metal"
/bin/mkdir -p "${m3_test_cryptex_driver%/*}"
printf '%s\n' '#!/bin/bash' 'exit 0' >"$m3_test_cryptex_driver"
/bin/chmod +x "$m3_test_cryptex_driver"
m3_test_cryptex_driver_canonical=$(cd "${m3_test_cryptex_driver%/*}" && pwd -P)/metal
m3_test_cryptex_driver_sha=$(m3_test_sha256 "$m3_test_cryptex_driver")
m3_test_cryptex_binding=$(/usr/bin/python3 -I -S \
    "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_unversioned_developer" \
    "$m3_test_versioned_developer_canonical" \
    "$m3_test_cryptex_driver")
/usr/bin/jq -e \
    --arg invocation "$m3_test_cryptex_driver" \
    --arg canonical "$m3_test_cryptex_driver_canonical" \
    --arg sha256 "$m3_test_cryptex_driver_sha" '
    .metal_driver.invocation_path == $invocation and
    .metal_driver.canonical_path == $canonical and
    .metal_driver.sha256 == $sha256 and
    (.metal_driver.resolution | contains("canonical DEVELOPER_DIR pin"))
' <<<"$m3_test_cryptex_binding" >/dev/null

# A byte change at the same xcrun path must change the postflight binding and
# therefore trip the runner's exact preflight/postflight JSON equality check.
printf '%s\n' '#!/bin/bash' 'printf drifted' >"$m3_test_cryptex_driver"
/bin/chmod +x "$m3_test_cryptex_driver"
m3_test_cryptex_drift_binding=$(/usr/bin/python3 -I -S \
    "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_unversioned_developer" \
    "$m3_test_versioned_developer_canonical" \
    "$m3_test_cryptex_driver")
if [[ "$m3_test_cryptex_drift_binding" == "$m3_test_cryptex_binding" ]] || \
   [[ "$(/usr/bin/jq -r '.metal_driver.sha256' \
        <<<"$m3_test_cryptex_drift_binding")" == "$m3_test_cryptex_driver_sha" ]]; then
    echo "Metal binding did not expose postflight executable-byte drift" >&2
    exit 1
fi

m3_test_noncanonical_driver="${m3_test_cryptex_driver%/*}/../bin/metal"
if /usr/bin/python3 -I -S "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_unversioned_developer" \
    "$m3_test_versioned_developer_canonical" \
    "$m3_test_noncanonical_driver" \
    >"$m3_test_scratch/xcode-noncanonical-driver.out" 2>&1
then
    echo "selected-Xcode binding accepted a noncanonical xcrun Metal path" >&2
    exit 1
fi
if ! /usr/bin/grep -q 'xcrun metal driver is not normalized and absolute' \
    "$m3_test_scratch/xcode-noncanonical-driver.out"; then
    echo "noncanonical xcrun Metal path was not reported" >&2
    exit 1
fi

m3_test_missing_driver="$m3_test_xcode_layout/missing-metal"
if /usr/bin/python3 -I -S "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_unversioned_developer" \
    "$m3_test_versioned_developer_canonical" \
    "$m3_test_missing_driver" \
    >"$m3_test_scratch/xcode-missing-driver.out" 2>&1
then
    echo "selected-Xcode binding accepted a missing xcrun Metal executable" >&2
    exit 1
fi
if ! /usr/bin/grep -q 'xcrun metal driver is unavailable' \
    "$m3_test_scratch/xcode-missing-driver.out"; then
    echo "missing xcrun Metal executable was not reported" >&2
    exit 1
fi

m3_test_nonexecutable_driver="$m3_test_xcode_layout/nonexecutable-metal"
printf '%s\n' '#!/bin/bash' 'exit 0' >"$m3_test_nonexecutable_driver"
if /usr/bin/python3 -I -S "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_unversioned_developer" \
    "$m3_test_versioned_developer_canonical" \
    "$m3_test_nonexecutable_driver" \
    >"$m3_test_scratch/xcode-nonexecutable-driver.out" 2>&1
then
    echo "selected-Xcode binding accepted a non-executable xcrun Metal file" >&2
    exit 1
fi
if ! /usr/bin/grep -q 'xcrun metal driver is not executable' \
    "$m3_test_scratch/xcode-nonexecutable-driver.out"; then
    echo "non-executable xcrun Metal file was not reported" >&2
    exit 1
fi

m3_test_substitute_developer="$m3_test_xcode_layout/Xcode_substitute.app/Contents/Developer"
m3_test_substitute_driver="$m3_test_substitute_developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal"
/bin/mkdir -p "${m3_test_substitute_driver%/*}"
printf '%s\n' '#!/bin/bash' 'exit 0' >"$m3_test_substitute_driver"
/bin/chmod +x "$m3_test_substitute_driver"
m3_test_substitute_developer_canonical=$(cd "$m3_test_substitute_developer" && pwd -P)
if /usr/bin/python3 -I -S "$m3_test_xcode_binding_helper" \
    binding \
    "$m3_test_versioned_developer" \
    "$m3_test_substitute_developer_canonical" \
    "$m3_test_versioned_driver" \
    >"$m3_test_scratch/xcode-pin-substitution.out" 2>&1
then
    echo "selected-Xcode binding accepted a substituted canonical pin" >&2
    exit 1
fi
if ! /usr/bin/grep -q \
    'pinned DEVELOPER_DIR does not match the selected Xcode developer directory' \
    "$m3_test_scratch/xcode-pin-substitution.out"
then
    echo "selected-Xcode canonical-pin substitution was not reported" >&2
    exit 1
fi

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
    'm3-stream-parity-runner-regression-pass: privileged-shell/controlled-hashing/identity/environment/git-routing/index-flags/artifact/discovery/cleanup controls passed model-free'
