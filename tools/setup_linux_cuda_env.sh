#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${VENV_PATH:-.venv}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_ROOT="$REPO_ROOT/$VENV_PATH"
VENV_PYTHON="$VENV_ROOT/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  "$SCRIPT_DIR/setup_linux_env.sh"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is required for CUDA validation" >&2
  exit 1
fi

nvidia-smi

"$VENV_PYTHON" -m pip install \
  "nvidia-cuda-runtime-cu12" \
  "nvidia-cublas-cu12" \
  "nvidia-curand-cu12" \
  "nvidia-cufft-cu12" \
  "nvidia-cudnn-cu12"

cat <<EOF

CUDA user-space libraries installed under $VENV_ROOT.

Use this in shell commands that run CUDA ORT GenAI:
export LD_LIBRARY_PATH="\$(./tools/cuda_library_path.sh):\${LD_LIBRARY_PATH:-}"
EOF
