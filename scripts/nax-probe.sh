#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$repo_root/oracle/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "oracle environment is missing; run scripts/setup-oracle.sh" >&2
    exit 2
fi

exec "$python_bin" "$repo_root/scripts/nax-probe.py"
