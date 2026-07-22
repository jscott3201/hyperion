#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python_path=$(uv python find 3.12.13)
if [[ "$("$python_path" -c 'import platform; print(platform.python_version())')" != "3.12.13" ]]; then
    echo "uv did not resolve the required oracle Python 3.12.13" >&2
    exit 1
fi

staging="$repo_root/oracle/.venv.build.$$"
backup="$repo_root/oracle/.venv.backup.$$"
if [[ -e "$staging" || -e "$backup" ]]; then
    echo "oracle staging path unexpectedly exists" >&2
    exit 1
fi
cleanup() {
    rm -rf -- "$staging"
}
trap cleanup EXIT
UV_PROJECT_ENVIRONMENT="$staging" uv sync --project oracle --python "$python_path" --frozen --no-dev
# The measurement interpreter redirects cache lookups and disables bytecode writes.
# Rejecting site-package bytecode makes any later injected .pyc fail identity checks.
find "$staging" -type f -name '*.pyc' -delete
find "$staging" -depth -type d -name '__pycache__' -empty -delete
if [[ -e oracle/.venv ]]; then
    mv oracle/.venv "$backup"
fi
if ! mv "$staging" oracle/.venv; then
    if [[ -e "$backup" ]]; then
        mv "$backup" oracle/.venv
    fi
    echo "failed to install the clean oracle environment; previous environment restored" >&2
    exit 1
fi
if ! scripts/verify-oracle.sh; then
    failed="$repo_root/oracle/.venv.failed.$$"
    mv oracle/.venv "$failed"
    if [[ -e "$backup" ]]; then
        mv "$backup" oracle/.venv
    fi
    rm -rf -- "$failed"
    echo "clean oracle recreation failed identity verification; previous environment restored" >&2
    exit 1
fi
rm -rf -- "$backup"
trap - EXIT
