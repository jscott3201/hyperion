#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

expected_lock_sha256=5e6e51756f1420e078f09badc0748e010eaaa7a5f1b81adfebbe9ca24a0e8883
actual_lock_sha256=$(shasum -a 256 oracle/uv.lock | awk '{print $1}')
if [[ "$actual_lock_sha256" != "$expected_lock_sha256" ]]; then
    echo "oracle lock digest differs from the reviewed M0 lock" >&2
    exit 1
fi
if [[ ! -x oracle/.venv/bin/python ]]; then
    echo "oracle environment is missing; run scripts/setup-oracle.sh" >&2
    exit 2
fi

oracle/.venv/bin/python - <<'PY'
from importlib.metadata import distribution
import json
import sys

expected_commit = "8239c72de5a0e42c539e30489021db73c7fe258c"
expected_lock = "5e6e51756f1420e078f09badc0748e010eaaa7a5f1b81adfebbe9ca24a0e8883"
mlx_lm = distribution("mlx-lm")
if mlx_lm.version != "0.31.3":
    raise SystemExit(f"expected mlx-lm 0.31.3, found {mlx_lm.version}")
direct_url_text = mlx_lm.read_text("direct_url.json")
if direct_url_text is None:
    raise SystemExit("mlx-lm direct_url.json is missing")
direct_url = json.loads(direct_url_text)
commit = direct_url.get("vcs_info", {}).get("commit_id")
if commit != expected_commit:
    raise SystemExit(f"expected mlx-lm commit {expected_commit}, found {commit}")
mlx_version = distribution("mlx").version
if mlx_version != "0.32.0":
    raise SystemExit(f"expected mlx 0.32.0, found {mlx_version}")
if not ((3, 11) <= sys.version_info[:2] < (3, 15)):
    raise SystemExit(f"unsupported oracle Python {sys.version.split()[0]}")
print(
    "oracle-verified: "
    f"python={sys.version.split()[0]} mlx=0.32.0 "
    f"mlx-lm=0.31.3@{expected_commit} lock={expected_lock}"
)
PY
