#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${1:?usage: scripts/verify-m1-archive-layout.sh RUN_ROOT [--emit-expected]}
mode=${2:-verify}
if [[ "$mode" != verify && "$mode" != --emit-expected ]]; then
    echo "usage: scripts/verify-m1-archive-layout.sh RUN_ROOT [--emit-expected]" >&2
    exit 64
fi
if [[ ! -d "$run_root" ]]; then
    echo "M1 archive layout root is not a directory: $run_root" >&2
    exit 2
fi

emit_expected() {
    printf '%s\n' \
        ./run-manifest.json \
        ./core-summary.json \
        ./summary.json \
        ./verification.json \
        ./preflight/preflight.log \
        ./preflight/oracle-verification.log \
        ./preflight/model-verification.log \
        ./preflight/ci.log \
        ./preflight/corpus.log \
        ./preflight/build.log \
        ./postflight/oracle-verification.log \
        ./postflight/model-verification.log \
        ./budget/coarse-selection.json \
        ./budget/selection.json \
        ./acca/confirmation.json
    for model in 12b e4b; do
        for context in 512 1024 4096 8192 16384 32768; do
            printf './core/%s-%s.jsonl\n./core/%s-%s.stderr.log\n' \
                "$model" "$context" "$model" "$context"
        done
    done
    for cap in 4294967296 5368709120 6442450944 7516192768 \
        8589934592 9663676416 10737418240 11811160064; do
        printf './budget/coarse/%s.jsonl\n./budget/coarse/%s.stderr.log\n' "$cap" "$cap"
    done
    refinement_caps=()
    while IFS= read -r cap; do
        refinement_caps+=("$cap")
    done < <(jq -er '.refinement_candidates_bytes | select(length == 2) | .[]' \
        "$run_root/budget/coarse-selection.json")
    if [[ ${#refinement_caps[@]} -ne 2 ]]; then
        echo "M1 archive requires exactly two refinement caps" >&2
        return 1
    fi
    for cap in "${refinement_caps[@]}"; do
        if [[ ! "$cap" =~ ^[0-9]+$ ]]; then
            echo "M1 archive refinement cap is not an integer" >&2
            return 1
        fi
        printf './budget/refine/%s.jsonl\n./budget/refine/%s.stderr.log\n' "$cap" "$cap"
    done
    for stem in 01-a1 02-c1 03-c2 04-a2; do
        printf './acca/%s.jsonl\n./acca/%s.stderr.log\n' "$stem" "$stem"
    done
    for model in 12b e4b; do
        printf './server/%s.server-smoke.json\n' "$model"
        for repeat in 1 2; do
            printf './server/%s-repeat-%s.json\n' "$model" "$repeat"
            printf './server/%s-repeat-%s.server.log\n' "$model" "$repeat"
            for suffix in warmup tool-call final; do
                printf './server/%s-repeat-%s.%s.jsonl\n' "$model" "$repeat" "$suffix"
            done
        done
    done
}

if [[ "$mode" == --emit-expected ]]; then
    emit_expected | LC_ALL=C sort
    exit 0
fi
if [[ ! -f "$run_root/SHA256SUMS" ]]; then
    echo "M1 archive lacks its content manifest" >&2
    exit 1
fi
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-layout.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
emit_expected | LC_ALL=C sort >"$scratch_dir/expected-files.txt"
(
    cd "$run_root"
    if find . ! -type f ! -type d -print -quit | grep -q .; then
        echo "M1 archive refuses symlinks and special files" >&2
        exit 1
    fi
    shasum -a 256 -c SHA256SUMS >/dev/null
    find . -type f ! -name SHA256SUMS ! -name archive-receipt.json -print | LC_ALL=C sort \
        >"$scratch_dir/actual-files.txt"
    cut -c 67- SHA256SUMS | LC_ALL=C sort >"$scratch_dir/manifest-files.txt"
)
if ! cmp "$scratch_dir/actual-files.txt" "$scratch_dir/manifest-files.txt"; then
    echo "M1 content manifest does not cover exactly the files selected for archival" >&2
    exit 1
fi
if ! cmp "$scratch_dir/expected-files.txt" "$scratch_dir/actual-files.txt"; then
    echo "M1 archive file set differs from the preregistered whole-run allowlist" >&2
    exit 1
fi
if rg -n -a \
    -e "$repo_root" \
    -e "${HOME:?HOME is required for archive privacy validation}" \
    -e '/Users/' \
    -e '/home/' \
    -e 'github_pat_' \
    -e 'gh[pousr]_[A-Za-z0-9]+' \
    -e 'Authorization:[[:space:]]*Bearer' \
    -e '(GH_TOKEN|GITHUB_TOKEN)=' \
    "$run_root" >"$scratch_dir/privacy-findings.txt"; then
    echo "M1 archive privacy scan found a forbidden path or credential pattern" >&2
    sed -n '1,20p' "$scratch_dir/privacy-findings.txt" >&2
    exit 1
fi
echo "m1-archive-layout-verified: $run_root"
