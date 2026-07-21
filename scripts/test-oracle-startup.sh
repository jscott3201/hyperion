#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python3 -B oracle/test_isolated_oracle.py
first=$(scripts/run-isolated-oracle.sh identity)
second=$(scripts/run-isolated-oracle.sh identity)
first_probe=$(jq -er '.hash_seed_probe' <<<"$first")
second_probe=$(jq -er '.hash_seed_probe' <<<"$second")
if [[ "$first_probe" != 1244036990071903237 || "$first_probe" != "$second_probe" ]]; then
    echo "oracle hash seed probe is not fixed across fresh processes" >&2
    exit 1
fi
if [[ "$(find oracle/.venv/lib/python3.12/site-packages -name '*.pyc' -print -quit)" != "" ]]; then
    echo "oracle site-packages contains forbidden bytecode" >&2
    exit 1
fi

echo "oracle-startup-model-free-controls: pass"
