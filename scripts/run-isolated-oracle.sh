#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

exec /usr/bin/env -i \
    LANG=C \
    LC_ALL=C \
    PYTHONHASHSEED=0 \
    TOKENIZERS_PARALLELISM=false \
    TZ=UTC \
    "$repo_root/oracle/.venv/bin/python" \
    -B -S -s -P -X pycache_prefix=/dev/null \
    "$repo_root/oracle/isolated_oracle.py" \
    "$@"
