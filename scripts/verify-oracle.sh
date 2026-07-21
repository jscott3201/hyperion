#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

expected_lock_sha256=b3603b4ebbc7f5883afe3d8cc10fc1767239837f5256985bd591b777c993dbaf
actual_lock_sha256=$(shasum -a 256 oracle/uv.lock | awk '{print $1}')
if [[ "$actual_lock_sha256" != "$expected_lock_sha256" ]]; then
    echo "oracle lock digest differs from the reviewed M0 lock" >&2
    exit 1
fi
if [[ ! -x oracle/.venv/bin/python ]]; then
    echo "oracle environment is missing; run scripts/setup-oracle.sh" >&2
    exit 2
fi

identity=$(oracle/.venv/bin/python -I -S oracle/isolated_oracle.py \
    script oracle/oracle_identity.py)
jq -e '
    .schema == "hyperion.m1-oracle-identity.v1" and
    .python == "3.12.13" and
    .python_executable_sha256 == "01564940172b2811e1f39a4dc90e84c7a26a19cf071bbc5de67e456d82627bec" and
    .python_runtime_tree_sha256 == "01a580d385a91f4b8bc195c8b2f56c4c2d156f6c1e1ad8768fc4501987c4e12f" and
    .python_runtime_file_count == 1897 and
    .site_packages_tree_sha256 == "db258e22404a3937d46d72ff44083400aafcf34636b8444a91a29c858b297006" and
    .site_packages_file_count == 5470 and
    .isolated_flags == {
      "ignore_environment": 1,
      "isolated": 1,
      "no_site": 1,
      "no_user_site": 1,
      "safe_path": true
    } and
    .mlx_version == "0.32.0" and
    .mlx_metal_version == "0.32.0" and
    .mlx_lm_version == "0.31.3" and
    .mlx_lm_commit == "8239c72de5a0e42c539e30489021db73c7fe258c" and
    .mlx_tree_sha256 == "bacebd4f46680155a129301ffefc516402142183584f2b47673bc91b561f0cd9" and
    .mlx_tree_file_count == 40 and
    .mlx_metal_tree_sha256 == "628a99548b65855148fb03f71cac83ce46eae42140f119fa8d1b51285c2abefd" and
    .mlx_metal_tree_file_count == 406 and
    .mlx_lm_tree_sha256 == "40dc49399a07cdf22e3516070cfe222e89ec2f0ff29cd6e257e1b069edc3472f" and
    .mlx_lm_tree_file_count == 176
' <<<"$identity" >/dev/null
printf 'oracle-verified: lock=%s identity=%s\n' "$expected_lock_sha256" "$identity"
