#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-layout-test.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
run_root="$scratch_dir/run"
mkdir -p "$run_root/budget"
printf '{"refinement_candidates_bytes":[4831838208,5905580032]}\n' \
    >"$run_root/budget/coarse-selection.json"
scripts/verify-m1-archive-layout.sh "$run_root" --emit-expected \
    >"$scratch_dir/expected-files.txt"
while IFS= read -r relative; do
    path="$run_root/${relative#./}"
    mkdir -p "$(dirname "$path")"
    if [[ ! -e "$path" ]]; then
        printf '{}\n' >"$path"
    fi
done <"$scratch_dir/expected-files.txt"
(
    cd "$run_root"
    find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z \
        | xargs -0 shasum -a 256 >SHA256SUMS
)
scripts/verify-m1-archive-layout.sh "$run_root" >/dev/null

printf 'stray\n' >"$run_root/stray-secret.txt"
if scripts/verify-m1-archive-layout.sh "$run_root" >"$scratch_dir/stray.out" 2>&1; then
    echo "M1 archive layout accepted an extra file" >&2
    exit 1
fi
mv "$run_root/stray-secret.txt" "$scratch_dir/stray-secret.txt"

ln -s run-manifest.json "$run_root/manifest-link"
if scripts/verify-m1-archive-layout.sh "$run_root" >"$scratch_dir/symlink.out" 2>&1; then
    echo "M1 archive layout accepted a symlink" >&2
    exit 1
fi
mv "$run_root/manifest-link" "$scratch_dir/manifest-link"

printf '/Users/example/private\n' >"$run_root/preflight/ci.log"
(
    cd "$run_root"
    find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z \
        | xargs -0 shasum -a 256 >SHA256SUMS
)
if scripts/verify-m1-archive-layout.sh "$run_root" >"$scratch_dir/privacy.out" 2>&1; then
    echo "M1 archive layout accepted a private machine path" >&2
    exit 1
fi

echo "m1-archive-layout-negative-controls: pass"
