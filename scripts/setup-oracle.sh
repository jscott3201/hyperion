#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python_path=$(uv python find 3.12.13)
if [[ "$("$python_path" -c 'import platform; print(platform.python_version())')" != "3.12.13" ]]; then
    echo "uv did not resolve the required oracle Python 3.12.13" >&2
    exit 1
fi
uv sync --project oracle --python "$python_path" --frozen --no-dev
scripts/verify-oracle.sh
