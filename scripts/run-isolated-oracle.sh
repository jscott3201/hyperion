#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Cross-machine runners set HYPERION_RELAX_RUNTIME_TREE=1 because the relocated
# uv base interpreter prefix is not byte-reproducible off the self-hosted M5.
# The strict runtime-tree pin stays enforced wherever this is unset/0.
relax=""
if [[ "${HYPERION_RELAX_RUNTIME_TREE:-0}" == "1" ]]; then
    relax="--relax-runtime-tree"
fi

exec /usr/bin/env -i \
    LANG=C \
    LC_ALL=C \
    PYTHONHASHSEED=0 \
    TOKENIZERS_PARALLELISM=false \
    TZ=UTC \
    "$repo_root/oracle/.venv/bin/python" \
    -B -S -s -P -X pycache_prefix=/dev/null \
    "$repo_root/oracle/isolated_oracle.py" \
    "$@" $relax
