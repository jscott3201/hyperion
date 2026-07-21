#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_check_script="$repo_root/scripts/check-append-only.sh"
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-append-only-test.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
fixture_repo="$scratch_dir/repo"
check_script="$fixture_repo/scripts/check-append-only.sh"

git init --quiet "$fixture_repo"
mkdir -p \
    "$fixture_repo/benchmarks" \
    "$fixture_repo/eval-results" \
    "$fixture_repo/docs/decisions" \
    "$fixture_repo/scripts"
cp "$source_check_script" "$check_script"
printf 'benchmark baseline\n' >"$fixture_repo/benchmarks/BENCHMARKS.md"
printf 'eval baseline\n' >"$fixture_repo/eval-results/EVAL_RESULTS.md"
printf '# Proposed decision\n\n- Status: proposed\n\nOriginal proposal.\n' \
    >"$fixture_repo/docs/decisions/0001-proposed.md"
printf '# Accepted decision\n\n- Status: accepted\n\nImmutable decision.\n' \
    >"$fixture_repo/docs/decisions/0002-accepted.md"
git -C "$fixture_repo" add .
git -C "$fixture_repo" \
    -c user.name='Hyperion CI' \
    -c user.email='hyperion-ci@example.invalid' \
    commit --quiet -m baseline
base_ref=$(git -C "$fixture_repo" rev-parse HEAD)

printf 'benchmark append\n' >>"$fixture_repo/benchmarks/BENCHMARKS.md"
printf '# Proposed decision revised\n\n- Status: proposed\n\nRevised proposal.\n' \
    >"$fixture_repo/docs/decisions/0001-proposed.md"
printf '\nAcceptance evidence appended.\n' >>"$fixture_repo/docs/decisions/0002-accepted.md"
git -C "$fixture_repo" add \
    benchmarks/BENCHMARKS.md \
    docs/decisions/0001-proposed.md \
    docs/decisions/0002-accepted.md
git -C "$fixture_repo" \
    -c user.name='Hyperion CI' \
    -c user.email='hyperion-ci@example.invalid' \
    commit --quiet -m append
HYPERION_BASE_REF="$base_ref" "$check_script" >/dev/null
append_ref=$(git -C "$fixture_repo" rev-parse HEAD)

printf '# Accepted decision changed\n\n- Status: accepted\n\nTampered decision.\n' \
    >"$fixture_repo/docs/decisions/0002-accepted.md"
git -C "$fixture_repo" add docs/decisions/0002-accepted.md
git -C "$fixture_repo" \
    -c user.name='Hyperion CI' \
    -c user.email='hyperion-ci@example.invalid' \
    commit --quiet -m tamper-accepted-decision
if output=$(HYPERION_BASE_REF="$base_ref" "$check_script" 2>&1); then
    echo "append-only regression: modified accepted decision was not rejected" >&2
    exit 1
fi
if [[ "$output" != *"accepted content was modified instead of appended: docs/decisions/0002-accepted.md"* ]]; then
    echo "append-only regression: unexpected accepted-decision rejection: $output" >&2
    exit 1
fi

git -C "$fixture_repo" checkout --quiet "$append_ref" -- docs/decisions/0002-accepted.md
printf 'tampered baseline\nbenchmark append\n' >"$fixture_repo/benchmarks/BENCHMARKS.md"
git -C "$fixture_repo" add benchmarks/BENCHMARKS.md docs/decisions/0002-accepted.md
git -C "$fixture_repo" \
    -c user.name='Hyperion CI' \
    -c user.email='hyperion-ci@example.invalid' \
    commit --quiet -m tamper-ledger
if output=$(HYPERION_BASE_REF="$base_ref" "$check_script" 2>&1); then
    echo "append-only regression: modified accepted bytes were not rejected" >&2
    exit 1
fi
if [[ "$output" != *"accepted content was modified instead of appended: benchmarks/BENCHMARKS.md"* ]]; then
    echo "append-only regression: unexpected rejection: $output" >&2
    exit 1
fi

echo "append-only regression: proposed edit and accepted append allowed; accepted prefixes immutable"
