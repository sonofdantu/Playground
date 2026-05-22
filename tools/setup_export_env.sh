#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${VENV_PATH:-.venv}"
ORT_GENAI_VERSION="${ORT_GENAI_VERSION:-0.13.1}"
SKIP_PACKAGE_INSTALL="${SKIP_PACKAGE_INSTALL:-0}"
SKIP_SOURCE_FETCH="${SKIP_SOURCE_FETCH:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_ROOT="$REPO_ROOT/$VENV_PATH"
VENV_PYTHON="$VENV_ROOT/bin/python"
REQUIREMENTS="$REPO_ROOT/tools/export-requirements.txt"
SOURCE_ROOT="$REPO_ROOT/.deps/onnxruntime-genai-src-v$ORT_GENAI_VERSION"
BUILDER="$SOURCE_ROOT/src/python/py/models/builder.py"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating Python virtual environment at $VENV_ROOT"
  "$PYTHON_BIN" -m venv "$VENV_ROOT"
fi

"$VENV_PYTHON" -m pip install --upgrade pip

if [[ "$SKIP_PACKAGE_INSTALL" != "1" ]]; then
  "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS"
fi

if [[ "$SKIP_SOURCE_FETCH" != "1" && ! -f "$BUILDER" ]]; then
  mkdir -p "$REPO_ROOT/.deps"
  git clone --depth 1 --branch "v$ORT_GENAI_VERSION" \
    https://github.com/microsoft/onnxruntime-genai.git \
    "$SOURCE_ROOT"
fi

if [[ ! -f "$BUILDER" ]]; then
  echo "error: could not find ORT GenAI model builder at $BUILDER" >&2
  exit 1
fi

echo
echo "Python: $VENV_PYTHON"
echo "Builder: $BUILDER"
echo
echo "Next command:"
echo "$VENV_PYTHON tools/prepare_qwen35_onnxopt_genai.py --output-dir models/qwen3.5-2b-onnxopt-q4f16 --variant q4f16"
