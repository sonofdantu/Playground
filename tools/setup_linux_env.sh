#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${VENV_PATH:-.venv}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_ROOT="$REPO_ROOT/$VENV_PATH"
VENV_PYTHON="$VENV_ROOT/bin/python"

if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  echo "error: a C++20 compiler is required (g++ or clang++)" >&2
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating Python virtual environment at $VENV_ROOT"
  "$PYTHON_BIN" -m venv "$VENV_ROOT"
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install \
  cmake \
  ninja \
  huggingface_hub \
  onnx \
  pillow

cat <<EOF

Linux environment ready.

Next commands:
./tools/fetch_runtime_deps.sh
$VENV_PYTHON tools/prepare_qwen35_onnxopt_genai.py --output-dir models/qwen3.5-2b-onnxopt-q4f16 --variant q4f16
./tools/build.sh --test --ort-genai
$VENV_PYTHON tools/make_test_image.py
./build/scene_describer --config configs/qwen3.5-2b-onnxopt.ini --image tmp/smoke.png --max-new-tokens 32 --json
EOF
