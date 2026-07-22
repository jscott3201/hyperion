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
scripts/verify-m1-archive-layout.sh "$run_root" >/dev/null
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/hyperion-m1-archive.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT
"$binary" m1 summarize --input-dir "$run_root/core" >"$scratch_dir/recomputed-summary.json"
if ! cmp "$run_root/core-summary.json" "$scratch_dir/recomputed-summary.json" \
    || ! cmp "$run_root/summary.json" "$scratch_dir/recomputed-summary.json"; then
    echo "stored M1 summaries do not reproduce from the archived core traces" >&2
    exit 1
fi

tag="m1-evidence-$source_commit"
asset="hyperion-m1-$source_commit.zip"
receipt_asset="hyperion-m1-$source_commit-receipt.json"
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
    --notes "Permanent content-addressed raw M1 baseline evidence. ZIP SHA-256: $archive_sha256. Retrieval receipt asset: $receipt_asset"

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
    --arg receipt_asset "$receipt_asset" \
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
      receipt_asset_name: $receipt_asset,
      archive_sha256: $archive_sha256,
      retrieved_sha256: $retrieved_sha256,
      retrieval_verified: ($archive_sha256 == $retrieved_sha256),
      release_url: $release_url,
      asset_url: $asset_url
    }
    ' >"$run_root/archive-receipt.json"
cp "$run_root/archive-receipt.json" "$scratch_dir/$receipt_asset"
gh release upload "$tag" "$scratch_dir/$receipt_asset#M1 durable retrieval receipt" \
    --repo "$GITHUB_REPOSITORY"
receipt_download_dir="$scratch_dir/receipt-download"
mkdir -p "$receipt_download_dir"
gh release download "$tag" --repo "$GITHUB_REPOSITORY" --pattern "$receipt_asset" \
    --dir "$receipt_download_dir"
if ! cmp "$scratch_dir/$receipt_asset" "$receipt_download_dir/$receipt_asset"; then
    echo "retrieved M1 receipt asset differs from the uploaded receipt" >&2
    exit 1
fi
release_json=$(gh api "repos/$GITHUB_REPOSITORY/releases/tags/$tag")
if [[ "$(jq -r --arg archive "$asset" --arg receipt "$receipt_asset" \
    '[.assets[].name] | contains([$archive, $receipt])' <<<"$release_json")" != true ]]; then
    echo "M1 release does not contain both permanent evidence assets" >&2
    exit 1
fi
printf 'm1-archive-pass: tag=%s asset=%s sha256=%s receipt_asset=%s receipt=%s\n' \
    "$tag" "$asset" "$archive_sha256" "$receipt_asset" "$run_root/archive-receipt.json"
