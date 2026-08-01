#!/bin/bash -p
# shellcheck disable=SC2016 # Literal awk/jq programs intentionally contain `$` fields.
if [[ ${BASH:-} != /bin/bash || $- != *p* ]]; then
    printf '%s\n' \
        'M3 stream parity requires direct execution by fixed /bin/bash in privileged mode' >&2
    exit 127
fi
set -euo pipefail

m3_dirname=/usr/bin/dirname
m3_date=/bin/date
m3_mkdir=/bin/mkdir
m3_mv=/bin/mv
m3_tee=/usr/bin/tee
m3_shasum=/usr/bin/shasum
m3_perl=/usr/bin/perl
m3_awk=/usr/bin/awk
m3_jq=/usr/bin/jq
m3_python=/usr/bin/python3
m3_git=/usr/bin/git
m3_sort=/usr/bin/sort
m3_head=/usr/bin/head
m3_tail=/usr/bin/tail
m3_uname=/usr/bin/uname
m3_sysctl=/usr/sbin/sysctl
m3_sw_vers=/usr/bin/sw_vers
m3_env=/usr/bin/env
m3_mktemp=/usr/bin/mktemp
m3_getconf=/usr/bin/getconf

for m3_bootstrap_tool in \
    "$m3_dirname" "$m3_date" "$m3_mkdir" "$m3_mv" "$m3_tee" \
    "$m3_shasum" "$m3_perl" "$m3_awk" "$m3_jq" "$m3_python" "$m3_git" \
    "$m3_sort" "$m3_head" "$m3_tail" "$m3_uname" "$m3_sysctl" \
    "$m3_sw_vers" "$m3_env" "$m3_mktemp" "$m3_getconf"
do
    if [[ ! -f "$m3_bootstrap_tool" || -L "$m3_bootstrap_tool" || \
          ! -x "$m3_bootstrap_tool" ]]; then
        printf 'required fixed system executable is unavailable: %s\n' \
            "$m3_bootstrap_tool" >&2
        exit 127
    fi
done

# Prevent an inherited developer-directory override from shaping fixed Apple
# shims before the canonical selected directory is discovered under env -i.
unset DEVELOPER_DIR

IFS= read -r m3_shasum_shebang <"$m3_shasum" || m3_shasum_shebang=
if [[ "$m3_shasum_shebang" != '#!/usr/bin/perl' ]]; then
    printf '%s\n' \
        'fixed /usr/bin/shasum does not declare the required /usr/bin/perl runtime' >&2
    exit 127
fi

m3_repo_root=$(cd "$("$m3_dirname" "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$m3_repo_root"

if (( $# > 1 )); then
    echo "usage: scripts/run-m3-stream-parity.sh [EVIDENCE_DIR]" >&2
    exit 64
fi

m3_started_at=$("$m3_date" -u +%Y-%m-%dT%H:%M:%SZ)
m3_run_id=$("$m3_date" -u +%Y%m%dT%H%M%SZ)
m3_evidence_root=${1:-"benchmarks/raw/m3/$m3_run_id"}
if [[ -e "$m3_evidence_root" || -L "$m3_evidence_root" ]]; then
    echo "refusing to overwrite existing M3 stream-parity evidence: $m3_evidence_root" >&2
    exit 1
fi
"$m3_mkdir" -p "$m3_evidence_root"
m3_evidence_root=$(cd "$m3_evidence_root" && pwd -P)
m3_manifest="$m3_evidence_root/manifest.json"
m3_preflight_log="$m3_evidence_root/preflight.log"
m3_identity_log="$m3_evidence_root/identity.log"
m3_build_log="$m3_evidence_root/build.log"
m3_test_log="$m3_evidence_root/test.log"
: >"$m3_preflight_log"
: >"$m3_identity_log"
: >"$m3_build_log"
: >"$m3_test_log"

m3_expected_fixture_sha256=086ca72232de415973564b2c6028c98a7063f7d024b85411512071650c86cf3d
m3_expected_historical_manifest_sha256=9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144
m3_expected_owner_payload_manifest_sha256=3cee7e9c21051eb6e6857ef485b355b4a2e847901930d64620348cbc7f56c806
m3_fixture_path=native/hyperion_mlx/tests/fixtures/12b_greedy_golden.safetensors
m3_build_argv=(
    cargo test --locked --offline
    -p hyperion-server
    --test contract
    --no-run
    --message-format=json-render-diagnostics
)
m3_direct_test_manifest_argv=(
    '<contract-test-executable>'
    real_http_sse_matches_m2_golden
    --ignored --exact --nocapture
)

m3_final_status=running
m3_failure_stage=artifact_environment
m3_source_sha=
m3_source_tree_sha=
m3_source_clean=false
m3_fixture_actual_sha256=
m3_artifact_identity=null
m3_artifact_identity_kind=
m3_artifact_manifest_sha256=
m3_machine_architecture=
m3_machine_model=
m3_physical_memory_bytes=
m3_macos_version=
m3_macos_build=
m3_rustc_version=
m3_cargo_version=
m3_cmake_version=
m3_clang_version=
m3_python_version=
m3_finished_at=
m3_build_exit_status=null
m3_direct_test_exit_status=null
m3_test_binary_sha256=
m3_test_binary_receipt_path=
m3_xcode_postflight_verified=false
m3_build_root=
m3_build_root_parent=
m3_build_root_dev=
m3_build_root_ino=

m3_login_home=$("$m3_python" -I -S -c '
import os
import pwd

home = os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir)
if not os.path.isabs(home) or not os.path.isdir(home):
    raise SystemExit("login home is unavailable or non-canonical")
print(home)
')
m3_home_binding_valid=true
if [[ -z "${HOME:-}" || "$HOME" != "$m3_login_home" ]]; then
    m3_home_binding_valid=false
fi
m3_cargo_home="$m3_login_home/.cargo"
m3_rustup_home="$m3_login_home/.rustup"
m3_cargo="$m3_cargo_home/bin/cargo"
m3_rustc="$m3_cargo_home/bin/rustc"
m3_rustup="$m3_cargo_home/bin/rustup"
m3_cmake=/opt/homebrew/bin/cmake
m3_clang=/usr/bin/clang
m3_clangxx=/usr/bin/clang++
m3_ar=/usr/bin/ar
m3_ranlib=/usr/bin/ranlib
m3_xcrun=/usr/bin/xcrun
m3_xcode_select=/usr/bin/xcode-select
m3_make=/usr/bin/make
m3_sh=/bin/sh
m3_cc=/usr/bin/cc
m3_ld=/usr/bin/ld
m3_libtool=/usr/bin/libtool
m3_install_name_tool=/usr/bin/install_name_tool
m3_controlled_path="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$m3_cargo_home/bin"
m3_source_git_path=/usr/bin:/bin:/usr/sbin:/sbin
m3_hash_path=/usr/bin:/bin:/usr/sbin:/sbin
m3_user_tmp=$("$m3_getconf" DARWIN_USER_TEMP_DIR)
m3_user_tmp=$("$m3_python" -I -S -c '
import os
import sys

path = os.path.realpath(sys.argv[1])
if not os.path.isabs(path) or not os.path.isdir(path) or os.path.islink(path):
    raise SystemExit("canonical Darwin user temporary directory is unavailable")
print(path)
' "$m3_user_tmp")

# Keep every source-binding Git query on one fixed path after a separate raw
# repository-metadata inspection has resolved and approved the native linked-
# worktree Git directory. No source query refreshes or writes the index.
m3_source_git() {
    "$m3_env" -i \
        HOME="$m3_login_home" \
        DEVELOPER_DIR="$m3_xcode_developer_dir" \
        PATH="$m3_source_git_path" \
        TMPDIR="$m3_user_tmp" \
        LANG=C LC_ALL=C \
        GIT_CONFIG_NOSYSTEM=1 \
        GIT_CONFIG_SYSTEM=/dev/null \
        GIT_CONFIG_GLOBAL=/dev/null \
        GIT_ATTR_NOSYSTEM=1 \
        GIT_TERMINAL_PROMPT=0 \
        GIT_PAGER= PAGER= \
        GIT_OPTIONAL_LOCKS=0 \
        GIT_NO_LAZY_FETCH=1 \
        GIT_NO_REPLACE_OBJECTS=1 \
        "$m3_git" \
        --no-optional-locks \
        -C "$m3_repo_root" \
        --git-dir="${m3_source_git_dir:?}" \
        --work-tree="$m3_repo_root" \
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

# Resolve repository-native paths without loading repository configuration,
# then parse each config file explicitly with includes disabled. Helper-capable
# local settings and non-comment info excludes are rejected before source Git
# commands can consult them. The returned deterministic digest record is also
# compared at postflight under the declared active-same-UID exclusion.
m3_validate_repository_metadata() {
    m3_pinned_python -I -S - \
        "$m3_repo_root" "$m3_git" "$m3_login_home" \
        "$m3_user_tmp" "$m3_source_git_path" <<'PY'
import hashlib
import json
import os
import stat
import subprocess
import sys


def fail(message):
    print(message)
    raise SystemExit(1)


def regular_file_bytes(path, description, required=False):
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        if required:
            fail(f"{description} is missing")
        return None
    except OSError as error:
        fail(f"{description} is unreadable: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{description} is not a real regular file")
    try:
        with open(path, "rb") as stream:
            return stream.read()
    except OSError as error:
        fail(f"{description} is unreadable: {error}")


def real_directory(path, description):
    canonical = os.path.realpath(path)
    try:
        metadata = os.lstat(canonical)
    except OSError as error:
        fail(f"{description} is unavailable: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        fail(f"{description} is not a real directory")
    return canonical


repo_root, git_executable, login_home, user_tmp, source_git_path = sys.argv[1:]
repo_root_bytes = os.fsencode(repo_root)
dot_git = os.path.join(repo_root_bytes, b".git")
try:
    dot_git_metadata = os.lstat(dot_git)
except OSError as error:
    fail(f"repository-native .git is unavailable: {error}")
if stat.S_ISDIR(dot_git_metadata.st_mode):
    git_dir = real_directory(dot_git, "repository-native Git directory")
elif stat.S_ISREG(dot_git_metadata.st_mode):
    git_file = regular_file_bytes(dot_git, "repository-native Git file", required=True)
    if b"\0" in git_file or not git_file.startswith(b"gitdir: "):
        fail("repository-native Git file is malformed")
    git_dir_field = git_file[len(b"gitdir: ") :]
    if git_dir_field.endswith(b"\n"):
        git_dir_field = git_dir_field[:-1]
    if git_dir_field.endswith(b"\r"):
        git_dir_field = git_dir_field[:-1]
    if not git_dir_field or b"\n" in git_dir_field or b"\r" in git_dir_field:
        fail("repository-native Git file has an invalid gitdir field")
    if not os.path.isabs(git_dir_field):
        git_dir_field = os.path.join(repo_root_bytes, git_dir_field)
    git_dir = real_directory(git_dir_field, "linked-worktree Git directory")
else:
    fail("repository-native .git is neither a directory nor a regular gitfile")

commondir_path = os.path.join(git_dir, b"commondir")
commondir_field = regular_file_bytes(commondir_path, "linked-worktree commondir")
if commondir_field is None:
    common_dir = git_dir
else:
    if commondir_field.endswith(b"\n"):
        commondir_field = commondir_field[:-1]
    if commondir_field.endswith(b"\r"):
        commondir_field = commondir_field[:-1]
    if not commondir_field or b"\0" in commondir_field or b"\n" in commondir_field:
        fail("linked-worktree commondir is malformed")
    if not os.path.isabs(commondir_field):
        commondir_field = os.path.join(git_dir, commondir_field)
    common_dir = real_directory(commondir_field, "repository common Git directory")

controlled_environment = {
    "HOME": login_home,
    "DEVELOPER_DIR": os.environ.get("DEVELOPER_DIR", ""),
    "PATH": source_git_path,
    "TMPDIR": user_tmp,
    "LANG": "C",
    "LC_ALL": "C",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "",
    "PAGER": "",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
}
dangerous_exact = {
    "core.attributesfile",
    "core.excludesfile",
    "core.fsmonitor",
    "diff.external",
}


def config_record(path, description):
    content = regular_file_bytes(path, description)
    if content is None:
        return None
    command = [
        git_executable,
        "config",
        "--file",
        os.fsdecode(path),
        "--no-includes",
        "--null",
        "--name-only",
        "--list",
    ]
    completed = subprocess.run(
        command,
        env=controlled_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"{description} could not be parsed with includes disabled")
    config_keys = completed.stdout[:-1] if completed.stdout.endswith(b"\0") else completed.stdout
    keys = config_keys.split(b"\0") if config_keys else ()
    if any(not raw_key for raw_key in keys):
        fail(f"{description} produced an empty config key")
    rejected = []
    for raw_key in keys:
        try:
            key = raw_key.decode("utf-8").lower()
        except UnicodeDecodeError:
            fail(f"{description} contains a non-UTF-8 config key")
        if (
            key in dangerous_exact
            or key.startswith("include.")
            or key.startswith("includeif.")
            or key.startswith("filter.")
            or (
                key.startswith("diff.")
                and key.rsplit(".", 1)[-1] in {"command", "external", "textconv"}
            )
        ):
            rejected.append(key)
    if rejected:
        fail(
            f"{description} contains prohibited helper/config routing: "
            + ",".join(sorted(set(rejected)))
        )
    return hashlib.sha256(content).hexdigest()


records = {}
config_paths = [
    (os.path.join(common_dir, b"config"), "common_config"),
    (os.path.join(git_dir, b"config.worktree"), "worktree_config"),
]
seen = set()
for path, role in config_paths:
    canonical_key = os.path.normpath(path)
    if canonical_key in seen:
        continue
    seen.add(canonical_key)
    records[role] = config_record(path, f"repository-native {role}")

info_roots = []
for candidate in (common_dir, git_dir):
    if candidate not in info_roots:
        info_roots.append(candidate)
for index, info_root in enumerate(info_roots):
    attributes_path = os.path.join(info_root, b"info", b"attributes")
    if os.path.lexists(attributes_path):
        fail("repository-native info/attributes is prohibited")
    exclude_path = os.path.join(info_root, b"info", b"exclude")
    exclude = regular_file_bytes(exclude_path, "repository-native info/exclude")
    role = f"info_exclude_{index}"
    if exclude is None:
        records[role] = None
        continue
    for line in exclude.splitlines():
        if line.strip() and not line.lstrip().startswith(b"#"):
            fail("repository-native info/exclude contains active patterns")
    records[role] = hashlib.sha256(exclude).hexdigest()

print(
    json.dumps(
        {
            "git_dir": os.fsdecode(git_dir),
            "common_dir": os.fsdecode(common_dir),
            "metadata_sha256": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
}

# Compare HEAD, the native index, and physical worktree bytes without Git
# conversion, attribute, exclude, or diff machinery. Physical traversal supplies
# the untracked/ignored inventory. Only the exact evidence directory and three
# generated roots that are not inputs to this fresh-target contract build are
# pruned, and an allowance is rejected if HEAD tracks anything beneath it.
m3_attest_source() {
    local m3_attest_evidence_relative=${1-}
    m3_pinned_python -I -S - \
        "$m3_repo_root" "$m3_source_git_dir" "$m3_git" \
        "$m3_login_home" "$m3_user_tmp" "$m3_source_git_path" \
        "$m3_attest_evidence_relative" <<'PY'
import hashlib
import json
import os
import stat
import subprocess
import sys


def fail(message):
    print(message)
    raise SystemExit(1)


(
    repo_root,
    git_dir,
    git_executable,
    login_home,
    user_tmp,
    source_git_path,
    evidence_relative,
) = sys.argv[1:]
environment = {
    "HOME": login_home,
    "DEVELOPER_DIR": os.environ.get("DEVELOPER_DIR", ""),
    "PATH": source_git_path,
    "TMPDIR": user_tmp,
    "LANG": "C",
    "LC_ALL": "C",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "",
    "PAGER": "",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
}
base = [
    git_executable,
    "--no-optional-locks",
    "-C",
    repo_root,
    f"--git-dir={git_dir}",
    f"--work-tree={repo_root}",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "core.ignoreStat=false",
    "-c",
    "core.trustctime=true",
    "-c",
    "core.checkStat=default",
    "-c",
    "core.fileMode=true",
    "-c",
    "core.symlinks=true",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.attributesFile=/dev/null",
    "-c",
    "core.excludesFile=/dev/null",
    "-c",
    "core.pager=",
    "-c",
    "pager.status=false",
    "-c",
    "diff.external=",
    "-c",
    "diff.trustExitCode=false",
]


def git_output(arguments, description):
    completed = subprocess.run(
        base + arguments,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"{description} failed closed")
    return completed.stdout


source_sha = git_output(
    ["rev-parse", "--verify", "HEAD^{commit}"], "source commit binding"
).strip()
source_tree_sha = git_output(
    ["rev-parse", "--verify", "HEAD^{tree}"], "source tree binding"
).strip()
if len(source_sha) not in (40, 64) or len(source_tree_sha) != len(source_sha):
    fail("source object identifiers have an unsupported format")
try:
    int(source_sha, 16)
    int(source_tree_sha, 16)
except ValueError:
    fail("source object identifiers are malformed")
hash_name = "sha1" if len(source_sha) == 40 else "sha256"

tree_stream = git_output(
    ["ls-tree", "-r", "-z", "--full-tree", "HEAD"],
    "HEAD tree enumeration",
)
if tree_stream and not tree_stream.endswith(b"\0"):
    fail("HEAD tree enumeration returned a truncated NUL stream")
head = {}
for record in tree_stream[:-1].split(b"\0") if tree_stream else ():
    metadata, separator, pathname = record.partition(b"\t")
    fields = metadata.split(b" ")
    if not separator or not pathname or len(fields) != 3:
        fail("HEAD tree enumeration returned a malformed record")
    mode, object_type, object_id = fields
    if pathname in head:
        fail(f"HEAD tree contains a duplicate path path_hex={pathname.hex()}")
    head[pathname] = (mode, object_type, object_id)

index_stream = git_output(
    ["ls-files", "--cached", "--stage", "-v", "-z"],
    "native index enumeration",
)
if index_stream and not index_stream.endswith(b"\0"):
    fail("native index enumeration returned a truncated NUL stream")
index = {}
prohibited_flags = []
for record in index_stream[:-1].split(b"\0") if index_stream else ():
    if len(record) < 4 or record[1:2] != b" ":
        fail("native index enumeration returned a malformed record")
    tag = record[:1]
    metadata, separator, pathname = record[2:].partition(b"\t")
    fields = metadata.split(b" ")
    if not separator or not pathname or len(fields) != 3:
        fail("native index enumeration returned malformed stage metadata")
    mode, object_id, stage = fields
    flags = []
    if b"a" <= tag <= b"z":
        flags.append("assume-unchanged")
    if tag.upper() == b"S":
        flags.append("skip-worktree")
    if flags:
        prohibited_flags.append((flags, pathname))
    if stage != b"0":
        fail(f"native index contains a nonzero stage path_hex={pathname.hex()}")
    if pathname in index:
        fail(f"native index contains a duplicate path path_hex={pathname.hex()}")
    index[pathname] = (mode, object_id)
if prohibited_flags:
    for flags, pathname in prohibited_flags:
        print(
            "prohibited native Git index entry: "
            f"flags={'+'.join(flags)} path_hex={pathname.hex()}"
        )
    raise SystemExit(1)

head_index = {
    pathname: (mode, object_id)
    for pathname, (mode, _object_type, object_id) in head.items()
}
if head_index != index:
    for pathname in sorted(set(head_index) | set(index)):
        if head_index.get(pathname) != index.get(pathname):
            fail(f"native index differs from exact HEAD path_hex={pathname.hex()}")

repo_root_bytes = os.fsencode(repo_root)


def git_blob_id(content):
    digest = hashlib.new(hash_name)
    digest.update(b"blob " + str(len(content)).encode("ascii") + b"\0")
    digest.update(content)
    return digest.hexdigest().encode("ascii")


tracked_paths = set(head)
for pathname, (mode, object_type, expected_id) in sorted(head.items()):
    full_path = os.path.join(repo_root_bytes, pathname)
    try:
        metadata = os.lstat(full_path)
    except OSError as error:
        fail(f"tracked path is unavailable path_hex={pathname.hex()}: {error}")
    if mode in (b"100644", b"100755") and object_type == b"blob":
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"tracked regular-file type changed path_hex={pathname.hex()}")
        executable = bool(metadata.st_mode & 0o111)
        if executable != (mode == b"100755"):
            fail(f"tracked executable mode changed path_hex={pathname.hex()}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(full_path, flags)
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    fail(f"tracked file identity changed path_hex={pathname.hex()}")
                chunks = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
            finally:
                os.close(descriptor)
        except OSError as error:
            fail(f"tracked file is unreadable path_hex={pathname.hex()}: {error}")
        actual_id = git_blob_id(b"".join(chunks))
    elif mode == b"120000" and object_type == b"blob":
        if not stat.S_ISLNK(metadata.st_mode):
            fail(f"tracked symlink type changed path_hex={pathname.hex()}")
        try:
            actual_id = git_blob_id(os.readlink(full_path))
        except OSError as error:
            fail(f"tracked symlink is unreadable path_hex={pathname.hex()}: {error}")
    elif mode == b"160000" and object_type == b"commit":
        fail(f"tracked gitlinks are unsupported path_hex={pathname.hex()}")
    else:
        fail(
            "HEAD contains an unsupported mode/type "
            f"path_hex={pathname.hex()} mode={mode.decode('ascii', 'replace')}"
        )
    if actual_id != expected_id:
        fail(f"tracked raw bytes differ from exact HEAD path_hex={pathname.hex()}")

allowed_roots = [b"build", b"target", b"oracle/.venv"]
if evidence_relative:
    evidence_bytes = os.fsencode(evidence_relative)
    if (
        os.path.isabs(evidence_bytes)
        or os.path.normpath(evidence_bytes) != evidence_bytes
        or evidence_bytes in (b"", b".")
    ):
        fail("inside-repository evidence allowance is not a normalized relative path")
    allowed_roots.append(evidence_bytes)
for allowed in allowed_roots:
    if any(path == allowed or path.startswith(allowed + b"/") for path in tracked_paths):
        fail(f"generated/evidence allowance overlaps tracked HEAD path_hex={allowed.hex()}")

physical_files = set()


def walk(relative):
    directory = repo_root_bytes if not relative else os.path.join(repo_root_bytes, relative)
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError as error:
        fail(f"physical source inventory failed path_hex={relative.hex()}: {error}")
    for entry in entries:
        name = entry.name
        pathname = name if not relative else relative + b"/" + name
        if not relative and pathname == b".git":
            continue
        if pathname in allowed_roots:
            if not entry.is_dir(follow_symlinks=False):
                fail(f"generated/evidence allowance is not a real directory path_hex={pathname.hex()}")
            continue
        if entry.is_dir(follow_symlinks=False):
            walk(pathname)
        else:
            physical_files.add(pathname)


walk(b"")
untracked = sorted(physical_files - tracked_paths)
if untracked:
    for pathname in untracked:
        print(f"untracked or ignored physical source path path_hex={pathname.hex()}")
    raise SystemExit(1)

print(
    json.dumps(
        {
            "sha": source_sha.decode("ascii"),
            "tree_sha": source_tree_sha.decode("ascii"),
            "tracked_path_count": len(tracked_paths),
            "inventory": "HEAD/index/raw-worktree-bytes plus physical untracked/ignored paths",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
}

m3_validate_xcode_binding() {
    local m3_binding_developer_dir=${m3_xcode_developer_dir-}
    local -a m3_binding_python=("$m3_python")
    if [[ -n "$m3_binding_developer_dir" ]]; then
        m3_binding_python=(
            "$m3_env" "DEVELOPER_DIR=$m3_binding_developer_dir" "$m3_python"
        )
    fi
    "${m3_binding_python[@]}" -I -S - "$@" <<'PY'
# M3_XCODE_BINDING_PYTHON_BEGIN
import json
import hashlib
import os
import stat
import sys


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


def canonical_path(path, description):
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        fail(f"{description} is not normalized and absolute")
    return os.path.realpath(path)


arguments = sys.argv[1:]
if not arguments:
    fail("selected-Xcode binding mode is required")
mode = arguments[0]
if mode == "selection" and len(arguments) == 2:
    selected_invocation = arguments[1]
    pinned_canonical = None
    metal_driver_invocation = None
elif mode == "binding" and len(arguments) == 4:
    selected_invocation, pinned_canonical, metal_driver_invocation = arguments[1:]
else:
    fail("invalid selected-Xcode binding arguments")

selected_canonical = canonical_path(
    selected_invocation, "selected Xcode developer directory"
)
try:
    selected_metadata = os.stat(selected_canonical, follow_symlinks=False)
except OSError as error:
    fail(f"selected Xcode developer directory is unavailable: {error}")
if not stat.S_ISDIR(selected_metadata.st_mode):
    fail("selected Xcode developer directory is not a real directory")

result = {
    "developer_dir": {
        "invocation_path": selected_invocation,
        "canonical_path": selected_canonical,
    }
}
if mode == "binding":
    if canonical_path(pinned_canonical, "pinned DEVELOPER_DIR") != pinned_canonical:
        fail("pinned DEVELOPER_DIR is not canonical")
    if pinned_canonical != selected_canonical:
        fail("pinned DEVELOPER_DIR does not match the selected Xcode developer directory")

    metal_driver_canonical = canonical_path(
        metal_driver_invocation, "xcrun metal driver"
    )
    try:
        metal_driver_metadata = os.stat(metal_driver_canonical, follow_symlinks=False)
    except OSError as error:
        fail(f"xcrun metal driver is unavailable: {error}")
    if not stat.S_ISREG(metal_driver_metadata.st_mode):
        fail("xcrun metal driver is not a real regular file")
    if metal_driver_metadata.st_mode & 0o111 == 0:
        fail("xcrun metal driver is not executable")
    digest = hashlib.sha256()
    try:
        with open(metal_driver_canonical, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail(f"xcrun metal driver is unreadable: {error}")

    result["environment_pin"] = {"DEVELOPER_DIR": pinned_canonical}
    result["metal_driver"] = {
        "invocation_path": metal_driver_invocation,
        "canonical_path": metal_driver_canonical,
        "sha256": digest.hexdigest(),
        "resolution": "fixed xcrun under canonical DEVELOPER_DIR pin",
    }

print(json.dumps(result, sort_keys=True, separators=(",", ":")))
# M3_XCODE_BINDING_PYTHON_END
PY
}

m3_read_active_xcode_developer_dir() {
    "$m3_env" -i \
        HOME="$m3_login_home" \
        PATH=/usr/bin:/bin:/usr/sbin:/sbin \
        TMPDIR="$m3_user_tmp" \
        LANG=C LC_ALL=C \
        "$m3_xcode_select" --print-path
}

m3_pinned_xcrun() {
    "$m3_env" -i \
        HOME="$m3_login_home" \
        DEVELOPER_DIR="$m3_xcode_developer_dir" \
        PATH=/usr/bin:/bin:/usr/sbin:/sbin \
        TMPDIR="$m3_user_tmp" \
        LANG=C LC_ALL=C \
        "$m3_xcrun" "$@"
}

m3_toolchain_query() {
    "$m3_env" -i \
        HOME="$m3_login_home" \
        CARGO_HOME="$m3_cargo_home" \
        RUSTUP_HOME="$m3_rustup_home" \
        DEVELOPER_DIR="$m3_xcode_developer_dir" \
        PATH="$m3_controlled_path" \
        TMPDIR="$m3_user_tmp" \
        LANG=C LC_ALL=C \
        "$@"
}

m3_xcode_developer_dir_invocation=$(m3_read_active_xcode_developer_dir)
m3_xcode_selection_json=$(m3_validate_xcode_binding \
    selection "$m3_xcode_developer_dir_invocation")
m3_xcode_developer_dir=$("$m3_jq" -er \
    '.developer_dir.canonical_path | select(type == "string" and startswith("/"))' \
    <<<"$m3_xcode_selection_json")

m3_pinned_python() {
    DEVELOPER_DIR="$m3_xcode_developer_dir" "$m3_python" "$@"
}

m3_metal_driver=$(m3_pinned_xcrun --find metal)
m3_metal=$(m3_pinned_xcrun -sdk macosx metal -print-prog-name=metal)

m3_identify_tools() {
    m3_pinned_python -I -S - "$@" <<'PY'
# M3_TOOL_IDENTITY_PYTHON_BEGIN
import hashlib
import json
import os
import stat
import sys

arguments = sys.argv[1:]
if len(arguments) % 2:
    raise SystemExit("tool identity arguments must be name/path pairs")
identities = {}
for name, invocation_path in zip(arguments[::2], arguments[1::2]):
    if not os.path.isabs(invocation_path):
        raise SystemExit(f"tool path is not absolute: {name}={invocation_path}")
    canonical_path = os.path.realpath(invocation_path)
    try:
        metadata = os.stat(canonical_path, follow_symlinks=False)
    except OSError as error:
        raise SystemExit(f"required tool is unavailable: {name}: {error}")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111 == 0:
        raise SystemExit(f"required tool is not a regular executable: {name}")
    digest = hashlib.sha256()
    with open(canonical_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    identities[name] = {
        "invocation_path": invocation_path,
        "canonical_path": canonical_path,
        "sha256": digest.hexdigest(),
    }
rustup_canonical = identities["rustup"]["canonical_path"]
if identities["cargo"]["canonical_path"] != rustup_canonical:
    raise SystemExit("fixed Cargo proxy does not resolve to the fixed rustup executable")
if identities["rustc"]["canonical_path"] != rustup_canonical:
    raise SystemExit("fixed rustc proxy does not resolve to the fixed rustup executable")
if not identities["cmake"]["canonical_path"].startswith(
    "/opt/homebrew/Cellar/cmake/"
):
    raise SystemExit("fixed CMake path does not resolve inside the M5 Homebrew cellar")
print(json.dumps(identities, sort_keys=True, separators=(",", ":")))
# M3_TOOL_IDENTITY_PYTHON_END
PY
}

m3_collect_tool_identities() {
    local m3_identity_metal_driver=$1
    local m3_identity_metal=$2
    m3_identify_tools \
        bash /bin/bash \
        dirname "$m3_dirname" \
        date "$m3_date" \
        mkdir "$m3_mkdir" \
        mv "$m3_mv" \
        tee "$m3_tee" \
        shasum "$m3_shasum" \
        perl "$m3_perl" \
        awk "$m3_awk" \
        jq "$m3_jq" \
        python3 "$m3_python" \
        git "$m3_git" \
        sort "$m3_sort" \
        head "$m3_head" \
        tail "$m3_tail" \
        uname "$m3_uname" \
        sysctl "$m3_sysctl" \
        sw_vers "$m3_sw_vers" \
        env "$m3_env" \
        mktemp "$m3_mktemp" \
        getconf "$m3_getconf" \
        cargo "$m3_cargo" \
        rustc "$m3_rustc" \
        rustup "$m3_rustup" \
        cmake "$m3_cmake" \
        clang "$m3_clang" \
        clangxx "$m3_clangxx" \
        ar "$m3_ar" \
        ranlib "$m3_ranlib" \
        xcrun "$m3_xcrun" \
        xcode_select "$m3_xcode_select" \
        make "$m3_make" \
        sh "$m3_sh" \
        cc "$m3_cc" \
        ld "$m3_ld" \
        libtool "$m3_libtool" \
        install_name_tool "$m3_install_name_tool" \
        metal_driver "$m3_identity_metal_driver" \
        metal "$m3_identity_metal"
}

m3_tool_identities_json=$(m3_collect_tool_identities \
    "$m3_metal_driver" "$m3_metal")
m3_preflight_tool_identities_json=$m3_tool_identities_json

m3_xcode_binding_json=$(m3_validate_xcode_binding \
    binding \
    "$m3_xcode_developer_dir_invocation" \
    "$m3_xcode_developer_dir" \
    "$m3_metal_driver")

m3_verify_xcode_postflight() {
    local m3_post_xcode_developer_dir_invocation
    local m3_post_metal_driver
    local m3_post_metal
    local m3_post_xcode_binding_json
    local m3_post_tool_identities_json

    m3_post_xcode_developer_dir_invocation=$(m3_read_active_xcode_developer_dir) || return $?
    m3_post_metal_driver=$(m3_pinned_xcrun --find metal) || return $?
    m3_post_metal=$(m3_pinned_xcrun \
        -sdk macosx metal -print-prog-name=metal) || return $?
    m3_post_xcode_binding_json=$(m3_validate_xcode_binding \
        binding \
        "$m3_post_xcode_developer_dir_invocation" \
        "$m3_xcode_developer_dir" \
        "$m3_post_metal_driver") || return $?
    if [[ "$m3_post_xcode_binding_json" != "$m3_xcode_binding_json" ]]; then
        echo "selected Xcode binding changed across the real-model test"
        return 1
    fi
    m3_post_tool_identities_json=$(m3_collect_tool_identities \
        "$m3_post_metal_driver" "$m3_post_metal") || return $?
    if [[ "$m3_post_tool_identities_json" != "$m3_preflight_tool_identities_json" ]]; then
        echo "developer-tool identities changed across the real-model test"
        return 1
    fi
    printf '%s\n' 'selected Xcode binding and developer-tool identities unchanged'
}

m3_note() {
    printf '%s\n' "$*" | "$m3_tee" -a "$m3_preflight_log"
}

m3_controlled_shasum() {
    "$m3_env" -i \
        HOME="$m3_login_home" \
        PATH="$m3_hash_path" \
        TMPDIR="$m3_user_tmp" \
        LANG=C LC_ALL=C \
        "$m3_shasum" "$@"
}

m3_log_sha256() {
    m3_controlled_shasum -a 256 "$1" | "$m3_awk" '{print $1}'
}

m3_write_manifest() {
    local m3_manifest_status=$1
    local m3_manifest_exit=$2
    local m3_preflight_sha
    local m3_identity_sha
    local m3_build_sha
    local m3_test_sha
    local m3_manifest_tmp
    m3_preflight_sha=$(m3_log_sha256 "$m3_preflight_log") || return $?
    m3_identity_sha=$(m3_log_sha256 "$m3_identity_log") || return $?
    m3_build_sha=$(m3_log_sha256 "$m3_build_log") || return $?
    m3_test_sha=$(m3_log_sha256 "$m3_test_log") || return $?
    m3_manifest_tmp="$m3_manifest.tmp"
    "$m3_jq" -n \
        --arg status "$m3_manifest_status" \
        --arg source_sha "$m3_source_sha" \
        --arg source_tree_sha "$m3_source_tree_sha" \
        --argjson source_clean "$m3_source_clean" \
        --arg fixture_path "$m3_fixture_path" \
        --arg fixture_expected "$m3_expected_fixture_sha256" \
        --arg fixture_actual "$m3_fixture_actual_sha256" \
        --arg historical_artifact_manifest "$m3_expected_historical_manifest_sha256" \
        --arg owner_artifact_manifest "$m3_expected_owner_payload_manifest_sha256" \
        --arg artifact_identity_kind "$m3_artifact_identity_kind" \
        --arg artifact_manifest "$m3_artifact_manifest_sha256" \
        --argjson artifact_identity "$m3_artifact_identity" \
        --arg started "$m3_started_at" \
        --arg finished "$m3_finished_at" \
        --arg architecture "$m3_machine_architecture" \
        --arg machine_model "$m3_machine_model" \
        --arg physical_memory "$m3_physical_memory_bytes" \
        --arg macos_version "$m3_macos_version" \
        --arg macos_build "$m3_macos_build" \
        --arg rustc "$m3_rustc_version" \
        --arg cargo "$m3_cargo_version" \
        --arg cmake "$m3_cmake_version" \
        --arg clang "$m3_clang_version" \
        --arg python3 "$m3_python_version" \
        --arg controlled_path "$m3_controlled_path" \
        --arg source_git_path "$m3_source_git_path" \
        --arg hash_path "$m3_hash_path" \
        --arg user_tmp "$m3_user_tmp" \
        --arg xcode_developer_dir "$m3_xcode_developer_dir" \
        --argjson tools "$m3_tool_identities_json" \
        --argjson xcode_binding "$m3_xcode_binding_json" \
        --argjson xcode_postflight_verified "$m3_xcode_postflight_verified" \
        --arg preflight_sha "$m3_preflight_sha" \
        --arg identity_sha "$m3_identity_sha" \
        --arg build_sha "$m3_build_sha" \
        --arg test_sha "$m3_test_sha" \
        --arg test_binary_sha "$m3_test_binary_sha256" \
        --arg test_binary_path "$m3_test_binary_receipt_path" \
        --arg failure_stage "$m3_failure_stage" \
        --argjson build_exit_status "$m3_build_exit_status" \
        --argjson direct_test_exit_status "$m3_direct_test_exit_status" \
        --argjson exit_status "$m3_manifest_exit" \
        '
        {
          schema: "hyperion.m3-stream-parity-evidence.v1",
          status: $status,
          source: {
            sha: (if $source_sha == "" then null else $source_sha end),
            tree_sha: (if $source_tree_sha == "" then null else $source_tree_sha end),
            clean: $source_clean,
            repository_root_binding: "<physical-repository-root>",
            git_dir_discovery: "raw repository-native .git directory or linked-worktree gitfile plus commondir; no repository config loaded during discovery",
            repository_native_metadata_policy: {
              inspection: "config and config.worktree parsed explicitly with includes disabled before source Git queries; metadata digests compared at postflight",
              prohibited: [
                "include/includeIf routing",
                "filter.* helpers",
                "core.attributesFile",
                "core.excludesFile",
                "core.fsmonitor",
                "diff external/textconv helpers",
                "info/attributes",
                "active info/exclude patterns"
              ],
              default_info_exclude: "comment-only content allowed"
            },
            native_index_policy: {
              index: "repository-native-linked-worktree-aware",
              required_for_clean: "all stage-0 mode/object/path entries exactly equal HEAD; assume-unchanged and skip-worktree absent at source preflight and postflight",
              prohibited_entry_flags: ["assume-unchanged", "skip-worktree"],
              inspection: "read-only git ls-files --cached --stage -v -z parsed as NUL-delimited bytes",
              mutation: "none; index is not refreshed or modified"
            },
            raw_worktree_policy: {
              tracked_bytes: "physical regular-file and symlink bytes hashed as Git blobs and compared directly with exact HEAD object ids; attributes, clean/smudge filters, and textconv are not used",
              tracked_modes: "physical regular-file, executable, and symlink modes must match exact HEAD",
              path_inventory: "physical byte-path traversal rejects every untracked or ignored non-directory path outside the narrow generated/evidence allowances; Git excludes are not used",
              generated_root_allowances: {
                "build/": "legacy CMake output only; this contract build configures native code beneath Cargo OUT_DIR in the fresh external CARGO_TARGET_DIR",
                "target/": "excluded only because CARGO_TARGET_DIR is a fresh runner-owned external directory",
                "oracle/.venv/": "excluded only because the selected hyperion-server contract build/test does not invoke the oracle environment and uses the explicit artifact root",
                inside_repository_evidence: "only the exact invocation-created benchmarks/raw/m3 evidence subtree is excluded"
              }
            }
          },
          fixture: {
            path: $fixture_path,
            expected_sha256: $fixture_expected,
            actual_sha256: (if $fixture_actual == "" then null else $fixture_actual end)
          },
          artifact: {
            expected_historical_manifest_sha256: $historical_artifact_manifest,
            expected_owner_payload_manifest_sha256: $owner_artifact_manifest,
            identity_kind: (if $artifact_identity_kind == "" then null else $artifact_identity_kind end),
            manifest_sha256: (if $artifact_manifest == "" then null else $artifact_manifest end),
            identity: $artifact_identity
          },
          started_at_utc: $started,
          finished_at_utc: (if $finished == "" then null else $finished end),
          command: {
            build_argv: [
              "cargo", "test", "--locked", "--offline",
              "-p", "hyperion-server", "--test", "contract",
              "--no-run", "--message-format=json-render-diagnostics"
            ],
            direct_test_argv: [
              "<contract-test-executable>",
              "real_http_sse_matches_m2_golden",
              "--ignored", "--exact", "--nocapture"
            ],
            environment: {
              HYPERION_12B_ARTIFACT: "<explicit-artifact-root>"
            },
            shell: {
              interpreter: "/bin/bash",
              invocation: "direct executable",
              privileged_mode_required: true,
              inherited_startup_state: "BASH_ENV, ENV, imported functions, SHELLOPTS, CDPATH, and GLOBIGNORE ignored by privileged mode before script code"
            },
            developer_tools: {
              selection: {
                environment_launcher: "/usr/bin/env -i",
                executable: "/usr/bin/xcode-select",
                argv: ["/usr/bin/xcode-select", "--print-path"],
                inherited_environment: "cleared",
                DEVELOPER_DIR: null
              },
              pin: {
                variable: "DEVELOPER_DIR",
                canonical_value: $xcode_developer_dir,
                scopes: [
                  "xcrun_tool_discovery",
                  "source_git",
                  "runner_python_helpers",
                  "toolchain_queries",
                  "cargo_build",
                  "direct_test"
                ]
              },
              xcrun: {
                environment_launcher: "/usr/bin/env -i",
                executable: "/usr/bin/xcrun",
                inherited_environment: "cleared",
                DEVELOPER_DIR: $xcode_developer_dir,
                metal_resolution: "the exact absolute regular executable returned by fixed xcrun is bound by invocation path, canonical path, and sha256; Apple MobileAsset/cryptex toolchains outside Xcode.app are allowed"
              },
              python_helpers: {
                executable: "/usr/bin/python3",
                DEVELOPER_DIR: $xcode_developer_dir
              }
            },
            hashing: {
              environment_launcher: "/usr/bin/env -i",
              executable: "/usr/bin/shasum",
              runtime: "/usr/bin/perl",
              inherited_environment: "cleared",
              working_directory: "preserved",
              environment: {
                HOME: "<canonical-login-home>",
                PATH: $hash_path,
                TMPDIR: $user_tmp,
                LANG: "C",
                LC_ALL: "C"
              }
            },
            source_git: {
              environment_launcher: "/usr/bin/env -i",
              executable: "/usr/bin/git",
              inherited_environment: "cleared",
              environment: {
                HOME: "<canonical-login-home>",
                DEVELOPER_DIR: $xcode_developer_dir,
                PATH: $source_git_path,
                TMPDIR: $user_tmp,
                LANG: "C",
                LC_ALL: "C",
                GIT_CONFIG_NOSYSTEM: "1",
                GIT_CONFIG_SYSTEM: "/dev/null",
                GIT_CONFIG_GLOBAL: "/dev/null",
                GIT_ATTR_NOSYSTEM: "1",
                GIT_TERMINAL_PROMPT: "0",
                GIT_PAGER: "",
                PAGER: "",
                GIT_OPTIONAL_LOCKS: "0",
                GIT_NO_LAZY_FETCH: "1",
                GIT_NO_REPLACE_OBJECTS: "1"
              },
              repository_binding: {
                working_directory: "<physical-repository-root>",
                work_tree: "<physical-repository-root>",
                git_dir: "repository-native-linked-worktree-aware"
              },
              global_options: ["--no-optional-locks"],
              command_line_config_overrides: [
                "core.fsmonitor=false",
                "core.untrackedCache=false",
                "core.ignoreStat=false",
                "core.trustctime=true",
                "core.checkStat=default",
                "core.fileMode=true",
                "core.symlinks=true",
                "core.hooksPath=/dev/null",
                "core.attributesFile=/dev/null",
                "core.excludesFile=/dev/null",
                "core.pager=",
                "pager.status=false",
                "diff.external=",
                "diff.trustExitCode=false"
              ],
              local_repository_config: "direct config files inspected first with includes disabled; helper-capable routing rejected, then benign repository config loaded with the listed command-line safety overrides"
            },
            toolchain_query_environment: {
              HOME: "<canonical-login-home>",
              CARGO_HOME: "<canonical-login-home>/.cargo",
              RUSTUP_HOME: "<canonical-login-home>/.rustup",
              DEVELOPER_DIR: $xcode_developer_dir,
              PATH: $controlled_path,
              TMPDIR: $user_tmp,
              LANG: "C",
              LC_ALL: "C"
            },
            build_environment: {
              CARGO_TARGET_DIR: "<fresh-runner-target-root>",
              HOME: "<fresh-runner-target-root>/build-home",
              CARGO_HOME: "<canonical-login-home>/.cargo",
              RUSTUP_HOME: "<canonical-login-home>/.rustup",
              DEVELOPER_DIR: $xcode_developer_dir,
              PATH: $controlled_path,
              TMPDIR: $user_tmp,
              LANG: "C",
              LC_ALL: "C",
              CC: "/usr/bin/clang",
              CXX: "/usr/bin/clang++",
              AR: "/usr/bin/ar",
              RANLIB: "/usr/bin/ranlib",
              CMAKE: "/opt/homebrew/bin/cmake"
            },
            direct_test_environment: {
              HOME: "<fresh-runner-target-root>/test-home",
              DEVELOPER_DIR: $xcode_developer_dir,
              PATH: $controlled_path,
              TMPDIR: $user_tmp,
              LANG: "C",
              LC_ALL: "C"
            }
          },
          execution_identity: {
            controlled_path: $controlled_path,
            xcode_binding: ($xcode_binding + {
              postflight_verified: $xcode_postflight_verified
            }),
            tools: $tools
          },
          trust_boundary: {
            active_same_uid_mutation_excluded: true,
            statement: "This receipt does not attest against active same-UID mutation of source, artifacts, toolchain files, or running processes; external isolation or privilege separation is required."
          },
          execution: {
            build_exit_status: $build_exit_status,
            direct_test_exit_status: $direct_test_exit_status,
            test_binary_path: (if $test_binary_path == "" then null else $test_binary_path end),
            test_binary_sha256: (if $test_binary_sha == "" then null else $test_binary_sha end)
          },
          machine: {
            architecture: (if $architecture == "" then null else $architecture end),
            machine_model: (if $machine_model == "" then null else $machine_model end),
            physical_memory_bytes: (if $physical_memory == "" then null else $physical_memory end),
            macos_version: (if $macos_version == "" then null else $macos_version end),
            macos_build: (if $macos_build == "" then null else $macos_build end)
          },
          toolchain: {
            rustc: (if $rustc == "" then null else $rustc end),
            cargo: (if $cargo == "" then null else $cargo end),
            cmake: (if $cmake == "" then null else $cmake end),
            clang: (if $clang == "" then null else $clang end),
            python3: (if $python3 == "" then null else $python3 end)
          },
          logs: {
            preflight: {path: "preflight.log", sha256: $preflight_sha},
            identity: {path: "identity.log", sha256: $identity_sha},
            build: {path: "build.log", sha256: $build_sha},
            test: {path: "test.log", sha256: $test_sha}
          },
          failure_stage: (if $failure_stage == "" then null else $failure_stage end),
          exit_status: (if $status == "running" then null else $exit_status end)
        }
        ' >"$m3_manifest_tmp" || return $?
    "$m3_mv" "$m3_manifest_tmp" "$m3_manifest" || return $?
}

m3_cleanup_build_root() {
    local m3_cleanup_leaf
    if [[ -z "$m3_build_root" ]]; then
        return 0
    fi
    m3_cleanup_leaf=${m3_build_root##*/}
    if [[ -z "$m3_build_root" || -z "$m3_build_root_parent" || \
          -z "$m3_build_root_dev" || -z "$m3_build_root_ino" || \
          "$m3_build_root_parent" != /private/tmp || \
          "$m3_build_root" == "$m3_build_root_parent" || \
          "${m3_build_root%/*}" != "$m3_build_root_parent" || \
          "$m3_cleanup_leaf" != hyperion-m3-contract.* ]]; then
        echo "refusing unsafe M3 build-target cleanup path" >&2
        return 1
    fi
    m3_pinned_python -I -S - "$m3_build_root_parent" "$m3_cleanup_leaf" \
        "$m3_build_root_dev" "$m3_build_root_ino" <<'PY' || return $?
# M3_BUILD_ROOT_CLEANUP_PYTHON_BEGIN
import os
import stat
import sys


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


def open_directory(parent_descriptor, name):
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        fail(f"cleanup entry is not a directory: {name}")
    return descriptor, metadata


def remove_open_tree(parent_descriptor, name, descriptor, expected_identity):
    try:
        with os.scandir(descriptor) as entries:
            entry_names = [entry.name for entry in entries]
        for entry_name in entry_names:
            try:
                metadata = os.stat(
                    entry_name, dir_fd=descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                fail(f"cleanup entry changed during traversal: {entry_name}")
            if stat.S_ISDIR(metadata.st_mode):
                child_descriptor, opened = open_directory(descriptor, entry_name)
                child_identity = (metadata.st_dev, metadata.st_ino)
                if (opened.st_dev, opened.st_ino) != child_identity:
                    os.close(child_descriptor)
                    fail(f"cleanup directory changed while opening: {entry_name}")
                remove_open_tree(
                    descriptor,
                    entry_name,
                    child_descriptor,
                    child_identity,
                )
            else:
                os.unlink(entry_name, dir_fd=descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        opened = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != expected_identity or (
            opened.st_dev,
            opened.st_ino,
        ) != expected_identity:
            fail(f"cleanup directory identity changed before removal: {name}")
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=parent_descriptor)


parent, leaf, expected_dev_text, expected_ino_text = sys.argv[1:]
expected_identity = (int(expected_dev_text), int(expected_ino_text))
if parent != "/private/tmp" or not leaf.startswith("hyperion-m3-contract."):
    fail("refusing unsafe M3 build-target cleanup arguments")
if "/" in leaf or leaf in ("", ".", ".."):
    fail("refusing non-leaf M3 build-target cleanup argument")

parent_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
parent_descriptor = os.open(parent, parent_flags)
try:
    parent_metadata = os.fstat(parent_descriptor)
    path_metadata = os.stat(parent, follow_symlinks=False)
    if (parent_metadata.st_dev, parent_metadata.st_ino) != (
        path_metadata.st_dev,
        path_metadata.st_ino,
    ):
        fail("M3 cleanup parent identity changed while opening")
    try:
        initial = os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        fail("M3 build-target root disappeared before cleanup")
    if not stat.S_ISDIR(initial.st_mode) or (initial.st_dev, initial.st_ino) != expected_identity:
        fail("M3 build-target root identity mismatch; leaving replacement for inspection")

    quarantine = f"{leaf}.cleanup-{os.getpid()}-{os.urandom(8).hex()}"
    os.rename(leaf, quarantine, src_dir_fd=parent_descriptor, dst_dir_fd=parent_descriptor)
    quarantined = os.stat(quarantine, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(quarantined.st_mode) or (
        quarantined.st_dev,
        quarantined.st_ino,
    ) != expected_identity:
        fail("M3 build-target root changed during quarantine; leaving material for inspection")
    quarantine_descriptor, opened = open_directory(parent_descriptor, quarantine)
    if (opened.st_dev, opened.st_ino) != expected_identity:
        os.close(quarantine_descriptor)
        fail("M3 quarantined build-target identity changed; leaving material for inspection")
    try:
        os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        os.close(quarantine_descriptor)
        fail("M3 build-target path was replaced during cleanup; leaving quarantine for inspection")

    remove_open_tree(
        parent_descriptor,
        quarantine,
        quarantine_descriptor,
        expected_identity,
    )
finally:
    os.close(parent_descriptor)
# M3_BUILD_ROOT_CLEANUP_PYTHON_END
PY
    m3_build_root=
    m3_build_root_dev=
    m3_build_root_ino=
    return 0
}

m3_finalize() {
    local m3_original_status=$?
    local m3_manifest_status=failed
    local m3_manifest_exit=$m3_original_status
    local m3_cleanup_status=0
    trap '' HUP INT TERM
    trap - EXIT
    set +e
    m3_cleanup_build_root
    m3_cleanup_status=$?
    if (( m3_cleanup_status != 0 )); then
        m3_failure_stage=build_target_cleanup
        if (( m3_manifest_exit == 0 )); then
            m3_manifest_exit=$m3_cleanup_status
        fi
    fi
    if [[ "$m3_final_status" == passed && $m3_original_status -eq 0 && \
          $m3_cleanup_status -eq 0 ]]; then
        m3_manifest_status=passed
        m3_failure_stage=
    elif (( m3_manifest_exit == 0 )); then
        m3_manifest_exit=1
    fi
    m3_finished_at=$("$m3_date" -u +%Y-%m-%dT%H:%M:%SZ)
    m3_write_manifest "$m3_manifest_status" "$m3_manifest_exit"
    local m3_manifest_status_code=$?
    if (( m3_manifest_status_code != 0 )); then
        echo "failed to finalize M3 evidence manifest (status $m3_manifest_status_code)" >&2
        if (( m3_manifest_exit == 0 )); then
            m3_manifest_exit=$m3_manifest_status_code
        fi
    elif [[ "$m3_manifest_status" == passed && $m3_manifest_exit -eq 0 ]]; then
        printf 'm3-stream-parity-pass: source=%s fixture=%s evidence=%s\n' \
            "$m3_source_sha" "$m3_fixture_actual_sha256" "$m3_evidence_root"
    fi
    exit "$m3_manifest_exit"
}

trap m3_finalize EXIT
trap 'm3_failure_stage=signal_hup; exit 129' HUP
trap 'm3_failure_stage=signal_interrupt; exit 130' INT
trap 'm3_failure_stage=signal_terminate; exit 143' TERM

m3_write_manifest running -1

m3_failure_stage=static_execution_identity
if [[ "$m3_home_binding_valid" != true ]]; then
    m3_note "M3 stream parity requires HOME to equal the canonical login home: $m3_login_home"
    exit 1
fi
if [[ ! -d "$m3_cargo_home" || -L "$m3_cargo_home" || \
      ! -d "$m3_rustup_home" || -L "$m3_rustup_home" ]]; then
    m3_note "canonical M5 Cargo or rustup home is missing or symlinked"
    exit 127
fi
m3_write_manifest running -1

m3_failure_stage=ambient_execution_overrides
m3_ambient_override=$(m3_pinned_python -I -S -c '
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
matches = sorted(
    name for name in os.environ if name in exact or name.startswith(prefixes)
)
if matches:
    print(matches[0])
')
if [[ -n "$m3_ambient_override" ]]; then
    m3_note "M3 stream parity rejects ambient $m3_ambient_override"
    exit 1
fi

m3_failure_stage=cargo_configuration_preflight
for m3_cargo_config_name in config config.toml; do
    m3_default_cargo_config="$m3_cargo_home/$m3_cargo_config_name"
    if [[ -e "$m3_default_cargo_config" || -L "$m3_default_cargo_config" ]]; then
        m3_note "M3 stream parity rejects default user Cargo $m3_cargo_config_name"
        exit 1
    fi
done
m3_cargo_config_parent=${m3_repo_root%/*}
[[ -n "$m3_cargo_config_parent" ]] || m3_cargo_config_parent=/
while :; do
    for m3_cargo_config_name in config config.toml; do
        m3_ancestor_cargo_config="$m3_cargo_config_parent/.cargo/$m3_cargo_config_name"
        if [[ -e "$m3_ancestor_cargo_config" || -L "$m3_ancestor_cargo_config" ]]; then
            m3_note "M3 stream parity rejects Cargo $m3_cargo_config_name outside the verified source worktree"
            exit 1
        fi
    done
    if [[ "$m3_cargo_config_parent" == / ]]; then
        break
    fi
    m3_cargo_config_parent=${m3_cargo_config_parent%/*}
    [[ -n "$m3_cargo_config_parent" ]] || m3_cargo_config_parent=/
done

m3_failure_stage=source_preflight
m3_evidence_relative=
case "$m3_evidence_root" in
    "$m3_repo_root"/*)
        m3_evidence_relative=${m3_evidence_root#"$m3_repo_root"/}
        case "$m3_evidence_relative" in
            benchmarks/raw/m3/*)
                ;;
            *)
                m3_note "evidence inside the repository is allowed only under benchmarks/raw/m3"
                exit 1
                ;;
        esac
        ;;
esac
if ! m3_source_metadata_json=$(m3_validate_repository_metadata 2>&1); then
    m3_note "source preflight rejected dangerous or unreadable repository-native Git metadata"
    if [[ -n "$m3_source_metadata_json" ]]; then
        printf '%s\n' "$m3_source_metadata_json" >>"$m3_preflight_log"
    fi
    exit 1
fi
if ! m3_source_git_dir=$("$m3_jq" -er \
    '.git_dir | select(type == "string" and startswith("/"))' \
    <<<"$m3_source_metadata_json")
then
    m3_note "source preflight could not bind the repository-native Git directory"
    exit 1
fi
if ! m3_source_attestation_json=$(m3_attest_source \
    "$m3_evidence_relative" 2>&1)
then
    m3_note "source preflight exact HEAD/index/raw-worktree attestation failed"
    if [[ -n "$m3_source_attestation_json" ]]; then
        printf '%s\n' "$m3_source_attestation_json" >>"$m3_preflight_log"
    fi
    exit 1
fi
if ! m3_source_sha=$("$m3_jq" -er \
    '.sha | select(type == "string" and test("^[0-9a-f]{40}([0-9a-f]{24})?$"))' \
    <<<"$m3_source_attestation_json") || \
   ! m3_source_tree_sha=$("$m3_jq" -er \
    '.tree_sha | select(type == "string" and test("^[0-9a-f]{40}([0-9a-f]{24})?$"))' \
    <<<"$m3_source_attestation_json")
then
    m3_note "source preflight attestation output is malformed"
    exit 1
fi
m3_source_clean=true

m3_failure_stage=artifact_environment
if [[ -z "${HYPERION_12B_ARTIFACT:-}" ]]; then
    m3_note "HYPERION_12B_ARTIFACT must be set explicitly to the pinned real artifact"
    exit 64
fi
m3_artifact_root=$HYPERION_12B_ARTIFACT
m3_sanitize() {
    m3_pinned_python -I -S -c '
import sys
text = sys.stdin.read()
for raw, replacement in (
    (sys.argv[1], "<REPO>"),
    (sys.argv[2], "<ARTIFACT>"),
    (sys.argv[3], "<BUILD_TARGET>"),
):
    if raw:
        text = text.replace(raw, replacement)
sys.stdout.write(text)
' "$m3_repo_root" "$m3_artifact_root" "$m3_build_root"
}
if [[ ! -d "$m3_artifact_root" || -L "$m3_artifact_root" ]]; then
    m3_note "HYPERION_12B_ARTIFACT must be a real directory, not a symlink"
    exit 64
fi
m3_artifact_root=$(cd "$m3_artifact_root" && pwd -P)
HYPERION_12B_ARTIFACT=$m3_artifact_root
export HYPERION_12B_ARTIFACT
m3_has_historical_manifest=false
m3_has_owner_payload_manifest=false
if [[ -e "$m3_artifact_root/SHA256SUMS" || -L "$m3_artifact_root/SHA256SUMS" ]]; then
    m3_has_historical_manifest=true
fi
if [[ -e "$m3_artifact_root/PAYLOAD_SHA256SUMS" || -L "$m3_artifact_root/PAYLOAD_SHA256SUMS" ]]; then
    m3_has_owner_payload_manifest=true
fi
if [[ "$m3_has_historical_manifest" == "$m3_has_owner_payload_manifest" ]]; then
    m3_note "artifact must contain exactly one recognized identity manifest (SHA256SUMS or PAYLOAD_SHA256SUMS)"
    exit 64
fi
if [[ "$m3_has_historical_manifest" == true ]]; then
    m3_artifact_identity_kind=historical_sha256sums
    m3_artifact_manifest_name=SHA256SUMS
    m3_expected_selected_manifest_sha256=$m3_expected_historical_manifest_sha256
else
    m3_artifact_identity_kind=owner_payload_sha256sums
    m3_artifact_manifest_name=PAYLOAD_SHA256SUMS
    m3_expected_selected_manifest_sha256=$m3_expected_owner_payload_manifest_sha256
fi
if [[ ! -f "$m3_artifact_root/$m3_artifact_manifest_name" || \
      -L "$m3_artifact_root/$m3_artifact_manifest_name" ]]; then
    m3_note "artifact identity manifest must be a real regular file"
    exit 64
fi
m3_artifact_manifest_sha256=$(m3_log_sha256 "$m3_artifact_root/$m3_artifact_manifest_name")
if [[ "$m3_artifact_manifest_sha256" != "$m3_expected_selected_manifest_sha256" ]]; then
    m3_note "artifact identity manifest digest is not an approved immutable payload"
    exit 64
fi

m3_runtime_files=(
    config.json \
    chat_template.jinja \
    generation_config.json \
    model-00001-of-00002.safetensors \
    model-00002-of-00002.safetensors \
    model.safetensors.index.json \
    tokenizer.json \
    tokenizer_config.json
)
for m3_required_file in "${m3_runtime_files[@]}"; do
    if [[ ! -f "$m3_artifact_root/$m3_required_file" || -L "$m3_artifact_root/$m3_required_file" ]]; then
        m3_note "artifact required file is missing or a symlink: $m3_required_file"
        exit 64
    fi
done

m3_verify_selected_artifact() {
    local m3_verify_manifest_path="$m3_artifact_root/$m3_artifact_manifest_name"
    local m3_verify_manifest_sha
    local m3_verify_historical_present=false
    local m3_verify_owner_present=false
    if [[ -e "$m3_artifact_root/SHA256SUMS" || -L "$m3_artifact_root/SHA256SUMS" ]]; then
        m3_verify_historical_present=true
    fi
    if [[ -e "$m3_artifact_root/PAYLOAD_SHA256SUMS" || -L "$m3_artifact_root/PAYLOAD_SHA256SUMS" ]]; then
        m3_verify_owner_present=true
    fi
    if [[ "$m3_verify_historical_present" == "$m3_verify_owner_present" || \
          ! -f "$m3_verify_manifest_path" || -L "$m3_verify_manifest_path" ]]; then
        echo "artifact identity manifests became missing, ambiguous, or symlinked" >&2
        return 1
    fi
    m3_verify_manifest_sha=$(m3_log_sha256 "$m3_verify_manifest_path")
    if [[ "$m3_verify_manifest_sha" != "$m3_expected_selected_manifest_sha256" ]]; then
        echo "artifact identity manifest changed or is not approved" >&2
        return 1
    fi
    for m3_verify_file in "${m3_runtime_files[@]}"; do
        if [[ ! -f "$m3_artifact_root/$m3_verify_file" || -L "$m3_artifact_root/$m3_verify_file" ]]; then
            echo "artifact runtime file became missing or symlinked: $m3_verify_file" >&2
            return 1
        fi
    done

    if [[ "$m3_artifact_identity_kind" == historical_sha256sums ]]; then
        m3_pinned_python -I -S oracle/model_identity.py \
            --model "$m3_artifact_root" \
            --manifest-sha256 "$m3_expected_historical_manifest_sha256"
        return
    fi

    local m3_public_expected_bound
    local m3_public_actual_bound
    m3_public_expected_bound=$(printf '%s\n' \
        chat_template.jinja \
        config.json \
        generation_config.json \
        model-00001-of-00002.safetensors \
        model-00002-of-00002.safetensors \
        model.safetensors.index.json \
        tokenizer.json \
        tokenizer_config.json | LC_ALL=C "$m3_sort")
    m3_public_actual_bound=$("$m3_awk" '
        NF == 2 && length($1) == 64 && $1 !~ /[^0-9a-f]/ {
            name = $2
            sub(/^\*/, "", name)
            sub(/^\.\//, "", name)
            print name
        }
    ' "$m3_verify_manifest_path" | LC_ALL=C "$m3_sort")
    if [[ "$m3_public_actual_bound" != "$m3_public_expected_bound" ]]; then
        echo "owner payload manifest does not bind exactly the eight runtime files" >&2
        return 1
    fi
    local m3_public_checksum_status
    (
        cd "$m3_artifact_root"
        m3_controlled_shasum -a 256 -c PAYLOAD_SHA256SUMS
    )
    m3_public_checksum_status=$?
    if (( m3_public_checksum_status != 0 )); then
        return "$m3_public_checksum_status"
    fi
    "$m3_jq" -cn \
        --arg manifest_sha256 "$m3_expected_owner_payload_manifest_sha256" \
        '{
          schema: "hyperion.owner-payload-identity.v1",
          identity_kind: "owner_payload_sha256sums",
          manifest_name: "PAYLOAD_SHA256SUMS",
          manifest_sha256: $manifest_sha256,
          payload_file_count: 8,
          exact_inventory: true,
          inventory_scope: "payload_manifest",
          release_metadata_allowed: true,
          symlinks_rejected: true,
          payload_hashes_verified: true
        }'
}
m3_failure_stage=artifact_identity_preflight
set +e
m3_identity_output=$(m3_verify_selected_artifact 2>&1)
m3_identity_status=$?
set -e
m3_identity_output=$(printf '%s\n' "$m3_identity_output" | m3_sanitize)
printf '%s\n' "$m3_identity_output" | "$m3_tee" -a "$m3_identity_log"
if (( m3_identity_status != 0 )); then
    exit "$m3_identity_status"
fi
m3_artifact_identity=$(printf '%s\n' "$m3_identity_output" | "$m3_tail" -n 1)
if ! "$m3_jq" -e \
    --arg kind "$m3_artifact_identity_kind" \
    --arg manifest "$m3_expected_selected_manifest_sha256" \
    '(.exact_inventory == true and .symlinks_rejected == true) and
     (if $kind == "historical_sha256sums" then
        .schema == "hyperion.model-tree-identity.v1" and
        .manifest_sha256 == $manifest and
        .payload_file_count > 0
      else
        .schema == "hyperion.owner-payload-identity.v1" and
        .identity_kind == "owner_payload_sha256sums" and
        .manifest_sha256 == $manifest and
        .payload_file_count == 8 and
        .payload_hashes_verified == true
      end)' \
    <<<"$m3_artifact_identity" >/dev/null
then
    echo "artifact identity output failed schema validation" | "$m3_tee" -a "$m3_identity_log"
    exit 1
fi
m3_write_manifest running -1

m3_failure_stage=fixture_preflight
if [[ ! -f "$m3_fixture_path" || -L "$m3_fixture_path" ]]; then
    m3_note "committed M2 golden fixture is missing or a symlink"
    exit 1
fi
m3_fixture_actual_sha256=$(m3_log_sha256 "$m3_fixture_path")
if [[ "$m3_fixture_actual_sha256" != "$m3_expected_fixture_sha256" ]]; then
    m3_note "M2 golden fixture identity mismatch"
    exit 1
fi

m3_machine_architecture=$("$m3_uname" -m)
m3_machine_model=$("$m3_sysctl" -n hw.model)
m3_physical_memory_bytes=$("$m3_sysctl" -n hw.memsize)
m3_macos_version=$("$m3_sw_vers" -productVersion)
m3_macos_build=$("$m3_sw_vers" -buildVersion)
if [[ "$m3_machine_architecture" != arm64 ]]; then
    m3_note "M3 stream parity requires Apple arm64; found $m3_machine_architecture"
    exit 1
fi
m3_rustc_version=$(m3_toolchain_query "$m3_rustc" --version --verbose)
m3_cargo_version=$(m3_toolchain_query "$m3_cargo" --version)
m3_cmake_version=$(m3_toolchain_query "$m3_cmake" --version | "$m3_head" -n 1)
m3_clang_version=$(m3_toolchain_query "$m3_clang" --version | "$m3_head" -n 1)
m3_python_version=$(m3_toolchain_query "$m3_python" --version 2>&1)
m3_note "source_sha=$m3_source_sha"
m3_note "source_tree_sha=$m3_source_tree_sha"
m3_note "fixture_sha256=$m3_fixture_actual_sha256"
m3_note "machine_architecture=$m3_machine_architecture"
m3_note "machine_model=$m3_machine_model"
m3_note "physical_memory_bytes=$m3_physical_memory_bytes"
m3_note "macos_version=$m3_macos_version"
m3_note "macos_build=$m3_macos_build"
m3_write_manifest running -1

m3_resolve_contract_executable() {
    m3_pinned_python -I -S - "$@" <<'PY'
# M3_CONTRACT_RESOLVER_PYTHON_BEGIN
import json
import os
import stat
import sys


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


mode, root, source = sys.argv[1:]
if not os.path.isabs(root) or os.path.realpath(root) != root:
    fail("fresh Cargo target root is not canonical and absolute")
if os.path.islink(root) or not os.path.isdir(root):
    fail("fresh Cargo target root is missing or symlinked")

if mode == "discover":
    candidates = []
    try:
        with open(source, "r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target = message.get("target")
                if (
                    message.get("reason") == "compiler-artifact"
                    and isinstance(target, dict)
                    and target.get("name") == "contract"
                    and target.get("kind") == ["test"]
                    and isinstance(message.get("executable"), str)
                    and message["executable"]
                ):
                    candidates.append(message["executable"])
    except (OSError, UnicodeError) as error:
        fail(f"could not read Cargo machine output: {error}")
    if len(candidates) != 1:
        fail(
            "Cargo machine output must identify exactly one contract test executable; "
            f"found {len(candidates)}"
        )
    candidate = candidates[0]
elif mode == "validate":
    candidate = source
else:
    fail("unknown contract executable validation mode")

if not os.path.isabs(candidate) or os.path.normpath(candidate) != candidate:
    fail("Cargo-emitted contract test executable path is not normalized and absolute")
try:
    if os.path.commonpath((root, candidate)) != root or candidate == root:
        fail("Cargo-emitted contract test executable escapes the fresh target root")
except ValueError:
    fail("Cargo-emitted contract test executable is on a different path root")

relative = os.path.relpath(candidate, root)
cursor = root
try:
    for component in relative.split(os.sep):
        cursor = os.path.join(cursor, component)
        metadata = os.lstat(cursor)
        if stat.S_ISLNK(metadata.st_mode):
            fail("Cargo-emitted contract test executable path contains a symlink")
except OSError as error:
    fail(f"Cargo-emitted contract test executable is unavailable: {error}")
if not stat.S_ISREG(metadata.st_mode):
    fail("Cargo-emitted contract test executable is not a regular file")
if metadata.st_mode & 0o111 == 0:
    fail("Cargo-emitted contract test executable is not executable")
if os.path.realpath(candidate) != candidate:
    fail("Cargo-emitted contract test executable does not resolve canonically")

open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(candidate, open_flags)
    opened_metadata = os.fstat(descriptor)
finally:
    if "descriptor" in locals():
        os.close(descriptor)
if not stat.S_ISREG(opened_metadata.st_mode):
    fail("opened contract test executable is not a regular file")
if (metadata.st_dev, metadata.st_ino) != (opened_metadata.st_dev, opened_metadata.st_ino):
    fail("contract test executable changed during validation")
print(candidate)
# M3_CONTRACT_RESOLVER_PYTHON_END
PY
}

m3_append_build_diagnostic() {
    local m3_diagnostic_path=$1
    local m3_diagnostic_pipeline_status
    if [[ ! -s "$m3_diagnostic_path" ]]; then
        return 0
    fi
    set +e
    m3_sanitize <"$m3_diagnostic_path" | "$m3_tee" -a "$m3_build_log"
    m3_diagnostic_pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if (( m3_diagnostic_pipeline_status[0] != 0 )); then
        return "${m3_diagnostic_pipeline_status[0]}"
    fi
    if (( m3_diagnostic_pipeline_status[1] != 0 )); then
        return "${m3_diagnostic_pipeline_status[1]}"
    fi
}

m3_failure_stage=build_target_setup
if [[ ! -d /private/tmp || -L /private/tmp ]] || \
   [[ "$(cd /private/tmp 2>/dev/null && pwd -P)" != /private/tmp ]]; then
    m3_note "canonical system build-target parent /private/tmp is unavailable"
    exit 1
fi
m3_build_root_parent=/private/tmp
m3_build_root=$("$m3_mktemp" -d "$m3_build_root_parent/hyperion-m3-contract.XXXXXXXX")
if [[ ! -d "$m3_build_root" || -L "$m3_build_root" || ! -O "$m3_build_root" || \
      "${m3_build_root%/*}" != "$m3_build_root_parent" || \
      "${m3_build_root##*/}" != hyperion-m3-contract.* ]]; then
    m3_note "mktemp did not create a safe runner-owned Cargo target root"
    exit 1
fi
m3_build_root_canonical=$(cd "$m3_build_root" && pwd -P)
if [[ "$m3_build_root_canonical" != "$m3_build_root" ]]; then
    m3_note "fresh Cargo target root did not remain canonical"
    exit 1
fi
case "$m3_build_root" in
    "$m3_repo_root"|"$m3_repo_root"/*|"$m3_artifact_root"|"$m3_artifact_root"/*)
        m3_note "fresh Cargo target root overlaps verified source or artifact input"
        exit 1
        ;;
esac
m3_build_root_identity=$(m3_pinned_python -I -S -c '
import os
import stat
import sys

path = sys.argv[1]
metadata = os.stat(path, follow_symlinks=False)
if not stat.S_ISDIR(metadata.st_mode):
    raise SystemExit("fresh Cargo target root is not a directory")
descriptor = os.open(
    path,
    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
)
try:
    opened = os.fstat(descriptor)
finally:
    os.close(descriptor)
if (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino):
    raise SystemExit("fresh Cargo target root changed while binding identity")
print(metadata.st_dev, metadata.st_ino)
' "$m3_build_root")
read -r m3_build_root_dev m3_build_root_ino <<<"$m3_build_root_identity"
if [[ -z "$m3_build_root_dev" || -z "$m3_build_root_ino" ]]; then
    m3_note "could not bind fresh Cargo target root identity"
    exit 1
fi
m3_build_home="$m3_build_root/build-home"
m3_test_home="$m3_build_root/test-home"
"$m3_mkdir" "$m3_build_home" "$m3_test_home"

m3_failure_stage=cargo_build
m3_cargo_machine_output="$m3_build_root/cargo-machine-output.jsonl"
set +e
"$m3_env" -i \
    HOME="$m3_build_home" \
    CARGO_HOME="$m3_cargo_home" \
    RUSTUP_HOME="$m3_rustup_home" \
    CARGO_TARGET_DIR="$m3_build_root" \
    HYPERION_12B_ARTIFACT="$m3_artifact_root" \
    DEVELOPER_DIR="$m3_xcode_developer_dir" \
    PATH="$m3_controlled_path" \
    TMPDIR="$m3_user_tmp" \
    LANG=C LC_ALL=C \
    CC="$m3_clang" CXX="$m3_clangxx" \
    AR="$m3_ar" RANLIB="$m3_ranlib" CMAKE="$m3_cmake" \
    "$m3_cargo" "${m3_build_argv[@]:1}" 2>&1 \
    | "$m3_tee" "$m3_cargo_machine_output" \
    | m3_sanitize \
    | "$m3_tee" "$m3_build_log"
m3_build_pipeline_status=("${PIPESTATUS[@]}")
set -e
m3_build_exit_status=${m3_build_pipeline_status[0]}
if (( m3_build_pipeline_status[0] != 0 )); then
    exit "${m3_build_pipeline_status[0]}"
fi
if (( m3_build_pipeline_status[1] != 0 )); then
    m3_failure_stage=build_machine_output_capture
    exit "${m3_build_pipeline_status[1]}"
fi
if (( m3_build_pipeline_status[2] != 0 )); then
    m3_failure_stage=build_log_sanitization
    exit "${m3_build_pipeline_status[2]}"
fi
if (( m3_build_pipeline_status[3] != 0 )); then
    m3_failure_stage=build_log_write
    exit "${m3_build_pipeline_status[3]}"
fi

m3_failure_stage=contract_executable_discovery
m3_discovery_error="$m3_build_root/contract-executable-discovery.log"
set +e
m3_contract_binary=$(m3_resolve_contract_executable \
    discover "$m3_build_root" "$m3_cargo_machine_output" 2>"$m3_discovery_error")
m3_discovery_status=$?
set -e
m3_append_build_diagnostic "$m3_discovery_error"
if (( m3_discovery_status != 0 )); then
    exit "$m3_discovery_status"
fi

m3_failure_stage=contract_executable_hash
set +e
m3_test_binary_sha256=$(m3_log_sha256 "$m3_contract_binary")
m3_binary_hash_status=$?
set -e
if (( m3_binary_hash_status != 0 )); then
    exit "$m3_binary_hash_status"
fi
m3_test_binary_relative_path=${m3_contract_binary#"$m3_build_root"/}
if [[ -z "$m3_test_binary_relative_path" || \
      "$m3_test_binary_relative_path" == "$m3_contract_binary" ]]; then
    echo "contract test executable did not remain relative to the fresh target root" \
        | "$m3_tee" -a "$m3_build_log"
    exit 1
fi
m3_test_binary_receipt_path="<fresh-runner-target-root>/$m3_test_binary_relative_path"
m3_tool_identities_json=$("$m3_jq" -c \
    --arg path "$m3_test_binary_receipt_path" \
    --arg sha256 "$m3_test_binary_sha256" \
    '. + {
      contract_test: {
        invocation_path: $path,
        canonical_path: $path,
        sha256: $sha256
      }
    }' <<<"$m3_tool_identities_json")

m3_failure_stage=contract_executable_pretest_validation
m3_validation_error="$m3_build_root/contract-executable-pretest.log"
set +e
m3_pretest_contract_binary=$(m3_resolve_contract_executable \
    validate "$m3_build_root" "$m3_contract_binary" 2>"$m3_validation_error")
m3_pretest_validation_status=$?
set -e
m3_append_build_diagnostic "$m3_validation_error"
if (( m3_pretest_validation_status != 0 )); then
    exit "$m3_pretest_validation_status"
fi
if [[ "$m3_pretest_contract_binary" != "$m3_contract_binary" ]]; then
    echo "contract test executable identity changed before direct execution" \
        | "$m3_tee" -a "$m3_build_log"
    exit 1
fi
m3_pretest_binary_sha256=$(m3_log_sha256 "$m3_contract_binary")
if [[ "$m3_pretest_binary_sha256" != "$m3_test_binary_sha256" ]]; then
    echo "contract test executable digest changed before direct execution" \
        | "$m3_tee" -a "$m3_build_log"
    exit 1
fi

m3_direct_test_argv=(
    "$m3_contract_binary"
    "${m3_direct_test_manifest_argv[@]:1}"
)
m3_failure_stage=real_model_test
set +e
"$m3_env" -i \
    HOME="$m3_test_home" \
    HYPERION_12B_ARTIFACT="$m3_artifact_root" \
    DEVELOPER_DIR="$m3_xcode_developer_dir" \
    PATH="$m3_controlled_path" \
    TMPDIR="$m3_user_tmp" \
    LANG=C LC_ALL=C \
    "${m3_direct_test_argv[@]}" 2>&1 \
    | m3_sanitize \
    | "$m3_tee" "$m3_test_log"
m3_test_pipeline_status=("${PIPESTATUS[@]}")
set -e
m3_direct_test_exit_status=${m3_test_pipeline_status[0]}
if (( m3_test_pipeline_status[0] != 0 )); then
    exit "${m3_test_pipeline_status[0]}"
fi
if (( m3_test_pipeline_status[1] != 0 )); then
    m3_failure_stage=test_log_sanitization
    exit "${m3_test_pipeline_status[1]}"
fi
if (( m3_test_pipeline_status[2] != 0 )); then
    m3_failure_stage=test_log_write
    exit "${m3_test_pipeline_status[2]}"
fi

m3_failure_stage=contract_executable_posttest_validation
m3_posttest_validation_error="$m3_build_root/contract-executable-posttest.log"
set +e
m3_posttest_contract_binary=$(m3_resolve_contract_executable \
    validate "$m3_build_root" "$m3_contract_binary" 2>"$m3_posttest_validation_error")
m3_posttest_validation_status=$?
set -e
m3_append_build_diagnostic "$m3_posttest_validation_error"
if (( m3_posttest_validation_status != 0 )); then
    exit "$m3_posttest_validation_status"
fi
if [[ "$m3_posttest_contract_binary" != "$m3_contract_binary" ]]; then
    echo "contract test executable identity changed after direct execution" \
        | "$m3_tee" -a "$m3_build_log"
    exit 1
fi
m3_posttest_binary_sha256=$(m3_log_sha256 "$m3_contract_binary")
if [[ "$m3_posttest_binary_sha256" != "$m3_test_binary_sha256" ]]; then
    echo "contract test executable digest changed after direct execution" \
        | "$m3_tee" -a "$m3_build_log"
    exit 1
fi

m3_failure_stage=postflight_identity
set +e
m3_post_xcode_output=$(m3_verify_xcode_postflight 2>&1)
m3_post_xcode_status=$?
set -e
m3_post_xcode_output=$(printf '%s\n' "$m3_post_xcode_output" | m3_sanitize)
printf '%s\n' "$m3_post_xcode_output" | "$m3_tee" -a "$m3_identity_log"
if (( m3_post_xcode_status != 0 )); then
    exit "$m3_post_xcode_status"
fi
m3_xcode_postflight_verified=true

set +e
m3_post_identity_output=$(m3_verify_selected_artifact 2>&1)
m3_post_identity_status=$?
set -e
m3_post_identity_output=$(printf '%s\n' "$m3_post_identity_output" | m3_sanitize)
printf '%s\n' "$m3_post_identity_output" | "$m3_tee" -a "$m3_identity_log"
if (( m3_post_identity_status != 0 )); then
    exit "$m3_post_identity_status"
fi
m3_post_artifact_identity=$(printf '%s\n' "$m3_post_identity_output" | "$m3_tail" -n 1)
if [[ "$m3_post_artifact_identity" != "$m3_artifact_identity" ]]; then
    echo "artifact identity changed across the real-model test" | "$m3_tee" -a "$m3_identity_log"
    exit 1
fi

m3_source_clean=false
m3_failure_stage=source_postflight
if ! m3_post_source_metadata_json=$(m3_validate_repository_metadata 2>&1); then
    m3_note "source postflight rejected dangerous or unreadable repository-native Git metadata"
    if [[ -n "$m3_post_source_metadata_json" ]]; then
        printf '%s\n' "$m3_post_source_metadata_json" >>"$m3_preflight_log"
    fi
    exit 1
fi
if [[ "$m3_post_source_metadata_json" != "$m3_source_metadata_json" ]]; then
    m3_note "repository-native Git metadata changed across the real-model test"
    exit 1
fi
if ! m3_post_source_attestation_json=$(m3_attest_source \
    "$m3_evidence_relative" 2>&1)
then
    m3_note "source postflight exact HEAD/index/raw-worktree attestation failed"
    if [[ -n "$m3_post_source_attestation_json" ]]; then
        printf '%s\n' "$m3_post_source_attestation_json" >>"$m3_preflight_log"
    fi
    exit 1
fi
if [[ "$m3_post_source_attestation_json" != "$m3_source_attestation_json" ]]; then
    m3_note "exact source attestation changed across the real-model test"
    exit 1
fi
m3_post_fixture_sha256=$(m3_log_sha256 "$m3_fixture_path")
if [[ "$m3_post_fixture_sha256" != "$m3_expected_fixture_sha256" ]]; then
    m3_note "M2 golden fixture changed across the real-model test"
    exit 1
fi
m3_source_clean=true

m3_final_status=passed
m3_failure_stage=
