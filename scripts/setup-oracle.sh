#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

uv sync --project oracle --frozen --no-dev
oracle/.venv/bin/python - <<'PY'
from importlib.metadata import distribution
import mlx.core as mx

commit = distribution("mlx-lm").read_text("direct_url.json") or ""
if "8239c72de5a0e42c539e30489021db73c7fe258c" not in commit:
    raise SystemExit("mlx-lm direct URL does not contain the reviewed commit")
if mx.__version__ != "0.32.0":
    raise SystemExit(f"expected mlx 0.32.0, found {mx.__version__}")
print("oracle-lock: mlx=0.32.0 mlx-lm=8239c72de5a0e42c539e30489021db73c7fe258c")
PY
