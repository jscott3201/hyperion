#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python3 - <<'PY'
from pathlib import Path
import re

checkout = re.compile(r"^(\s*)-\s+uses:\s+actions/checkout@")
checked = 0
for path in sorted(Path(".github/workflows").glob("*.yml")):
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        match = checkout.match(line)
        if match is None:
            continue
        checked += 1
        indent = len(match.group(1))
        block = []
        for following in lines[index + 1 :]:
            stripped = following.lstrip()
            following_indent = len(following) - len(stripped)
            if stripped.startswith("-") and following_indent <= indent:
                break
            block.append(stripped)
        if "persist-credentials: false" not in block:
            raise SystemExit(f"{path}:{index + 1}: checkout persists credentials")
if checked == 0:
    raise SystemExit("no checkout actions found")
print(f"workflow-checkout-credentials: pass ({checked} checkouts)")
PY
