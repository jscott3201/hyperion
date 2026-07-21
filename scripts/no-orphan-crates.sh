#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if ! command -v jq >/dev/null 2>&1; then
    echo "no-orphan-crates: jq is required" >&2
    exit 2
fi

scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-orphans.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT

# Negative control: prove the comparison fails closed before trusting it on the workspace.
printf '%s\n' hyperion-bench hyperion-orphan hyperion-server >"$scratch_dir/control-members"
printf '%s\n' hyperion-bench hyperion-server >"$scratch_dir/control-reachable"
if [[ "$(comm -23 "$scratch_dir/control-members" "$scratch_dir/control-reachable")" != "hyperion-orphan" ]]; then
    echo "no-orphan-crates: negative control failed" >&2
    exit 2
fi

cargo metadata --locked --format-version 1 --no-deps |
    jq -r '.workspace_members[] as $id | .packages[] | select(.id == $id) | .name' |
    sort -u >"$scratch_dir/members"

for root in hyperion-server hyperion-bench; do
    cargo tree --locked -e normal -p "$root" --prefix none --format '{p}' |
        awk '{print $1}' >>"$scratch_dir/reachable"
done
sort -u "$scratch_dir/reachable" -o "$scratch_dir/reachable"

comm -23 "$scratch_dir/members" "$scratch_dir/reachable" >"$scratch_dir/orphans"
if [[ -s "$scratch_dir/orphans" ]]; then
    echo "workspace crates not reachable from hyperion-server or hyperion-bench:" >&2
    sed 's/^/  - /' "$scratch_dir/orphans" >&2
    exit 1
fi

echo "no-orphan-crates: all workspace members are on a live root path"
