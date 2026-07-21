#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?usage: scripts/archive-m1-evidence.sh RUN_ID}
if [[ ! "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "M1 archive run ID contains unsafe characters" >&2
    exit 64
fi
cd "$repo_root"
run_root="benchmarks/raw/m1/$run_id"
verification="$run_root/verification.json"
manifest="$run_root/run-manifest.json"
if [[ "$(jq -er '.valid' "$verification")" != true ]]; then
    echo "refusing to archive M1 evidence without a passing master verification" >&2
    exit 1
fi
source_commit=$(jq -er '.source_commit' "$manifest")
if [[ "$source_commit" != "$(git rev-parse HEAD)" ]]; then
    echo "M1 archive source commit differs from HEAD" >&2
    exit 1
fi
if [[ -z "${GH_TOKEN:-}" || -z "${GITHUB_REPOSITORY:-}" ]]; then
    echo "durable M1 archive requires GH_TOKEN and GITHUB_REPOSITORY" >&2
    exit 2
fi

binary="$repo_root/target/m1-release/release/hyperion-bench"
if [[ ! -x "$binary" ]]; then
    echo "durable M1 archive requires the run-bound release benchmark executable" >&2
    exit 2
fi
"$binary" m1 verify-run --input-dir "$run_root" >/dev/null

if [[ ! -f "$run_root/SHA256SUMS" ]]; then
    echo "M1 archive lacks its content manifest" >&2
    exit 1
fi
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-archive.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
(
    cd "$run_root"
    shasum -a 256 -c SHA256SUMS >/dev/null
    find . -type f ! -name SHA256SUMS ! -name archive-receipt.json -print | LC_ALL=C sort \
        >"$scratch_dir/actual-files.txt"
    cut -c 67- SHA256SUMS | LC_ALL=C sort >"$scratch_dir/manifest-files.txt"
)
if ! cmp "$scratch_dir/actual-files.txt" "$scratch_dir/manifest-files.txt"; then
    echo "M1 content manifest does not cover exactly the files selected for archival" >&2
    exit 1
fi
"$binary" m1 summarize --input-dir "$run_root/core" >"$scratch_dir/recomputed-summary.json"
if ! cmp "$run_root/core-summary.json" "$scratch_dir/recomputed-summary.json" \
    || ! cmp "$run_root/summary.json" "$scratch_dir/recomputed-summary.json"; then
    echo "stored M1 summaries do not reproduce from the archived core traces" >&2
    exit 1
fi

tag="m1-evidence-$source_commit"
asset="hyperion-m1-$source_commit.zip"
if gh release view "$tag" --repo "$GITHUB_REPOSITORY" >/dev/null 2>&1; then
    echo "refusing to replace immutable M1 evidence release $tag" >&2
    exit 1
fi

archive="$scratch_dir/$asset"
(
    cd "$run_root"
    find . -type f ! -name archive-receipt.json -print | LC_ALL=C sort | zip -X -q "$archive" -@
)
archive_sha256=$(shasum -a 256 "$archive" | awk '{print $1}')
gh release create "$tag" "$archive#Hyperion M1 raw evidence" \
    --repo "$GITHUB_REPOSITORY" \
    --target "$source_commit" \
    --title "Hyperion M1 baseline evidence $source_commit" \
    --notes "Permanent content-addressed raw M1 baseline evidence. SHA-256: $archive_sha256"

download_dir="$scratch_dir/download"
mkdir -p "$download_dir"
gh release download "$tag" --repo "$GITHUB_REPOSITORY" --pattern "$asset" --dir "$download_dir"
retrieved_sha256=$(shasum -a 256 "$download_dir/$asset" | awk '{print $1}')
if [[ "$retrieved_sha256" != "$archive_sha256" ]]; then
    echo "retrieved M1 release asset hash differs from the uploaded archive" >&2
    exit 1
fi
release_json=$(gh api "repos/$GITHUB_REPOSITORY/releases/tags/$tag")
release_url=$(jq -er '.html_url' <<<"$release_json")
asset_url=$(jq -er --arg asset "$asset" '.assets[] | select(.name == $asset) | .browser_download_url' <<<"$release_json")
jq -n \
    --arg run_id "$run_id" \
    --arg source_commit "$source_commit" \
    --arg tag "$tag" \
    --arg asset "$asset" \
    --arg archive_sha256 "$archive_sha256" \
    --arg retrieved_sha256 "$retrieved_sha256" \
    --arg release_url "$release_url" \
    --arg asset_url "$asset_url" '
    {
      schema: "hyperion.m1-archive-receipt.v1",
      run_id: $run_id,
      source_commit: $source_commit,
      archive_kind: "github_release_asset",
      release_tag: $tag,
      asset_name: $asset,
      archive_sha256: $archive_sha256,
      retrieved_sha256: $retrieved_sha256,
      retrieval_verified: ($archive_sha256 == $retrieved_sha256),
      release_url: $release_url,
      asset_url: $asset_url
    }
    ' >"$run_root/archive-receipt.json"
printf 'm1-archive-pass: tag=%s asset=%s sha256=%s receipt=%s\n' \
    "$tag" "$asset" "$archive_sha256" "$run_root/archive-receipt.json"
