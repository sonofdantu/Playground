#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

paths=("$REPO_ROOT/build")

while IFS= read -r path; do
  paths+=("$path")
done < <(find "$REPO_ROOT/.venv/lib" -path "*/site-packages/nvidia/*/lib" -type d 2>/dev/null | sort)

(
  IFS=:
  echo "${paths[*]}"
)
