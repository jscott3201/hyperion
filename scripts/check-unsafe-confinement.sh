#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if rg -n --glob '*.rs' --glob '!crates/hyperion-ffi/**' \
    '(^|[^[:alnum:]_])unsafe([[:space:]]|\{|extern)' crates; then
    echo "unsafe Rust is confined to crates/hyperion-ffi" >&2
    exit 1
fi

echo "unsafe-confinement: only hyperion-ffi contains unsafe Rust"
