#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models/qwen3.5-2b-onnxopt-q4f16"
CONFIG_PATH="configs/qwen3.5-2b-onnxopt.ini"
IMAGE_PATH="tmp/smoke.png"
MAX_NEW_TOKENS=16
EXECUTION_PROVIDER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --image)
      IMAGE_PATH="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --execution-provider)
      EXECUTION_PROVIDER="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/smoke_ort_genai.sh [options]

Options:
  --model-dir <path>       ORT GenAI model package, default models/qwen3.5-2b-onnxopt-q4f16
  --config <path>          Runtime config, default configs/qwen3.5-2b-onnxopt.ini
  --image <path>           Smoke image path, default tmp/smoke.png
  --max-new-tokens <n>     Generation length, default 16
  --execution-provider <n> Override execution provider, for example cuda
EOF
      exit 0
      ;;
    *)
      echo "error: unknown option $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
SCENE_DESCRIBER="$REPO_ROOT/build/scene_describer"
MODEL_PROBE="$REPO_ROOT/build/scene_model_probe"

cd "$REPO_ROOT"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "error: missing .venv; run ./tools/setup_linux_env.sh first" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "error: missing model package: $MODEL_DIR" >&2
  echo "run: $VENV_PYTHON tools/prepare_qwen35_onnxopt_genai.py --output-dir $MODEL_DIR --variant q4f16" >&2
  exit 1
fi

if [[ ! -x "$SCENE_DESCRIBER" || ! -x "$MODEL_PROBE" ]]; then
  echo "error: missing ORT-enabled build; run ./tools/build.sh --test --ort-genai first" >&2
  exit 1
fi

"$VENV_PYTHON" tools/validate_model_package.py "$MODEL_DIR" --require-multimodal --require-provenance

if [[ ! -f "$IMAGE_PATH" && "$IMAGE_PATH" == "tmp/smoke.png" ]]; then
  "$VENV_PYTHON" tools/make_test_image.py
elif [[ ! -f "$IMAGE_PATH" ]]; then
  echo "error: image file does not exist: $IMAGE_PATH" >&2
  exit 1
fi

probe_stage="token"
if [[ "$EXECUTION_PROVIDER" == "cuda" ]]; then
  probe_stage="model"
fi

probe_args=("$MODEL_PROBE" --model-dir "$MODEL_DIR" --image "$IMAGE_PATH" --stage "$probe_stage" --max-new-tokens 1)
cli_args=("$SCENE_DESCRIBER" --config "$CONFIG_PATH" --image "$IMAGE_PATH" --max-new-tokens "$MAX_NEW_TOKENS" --json)
if [[ -n "$EXECUTION_PROVIDER" ]]; then
  probe_args+=(--execution-provider "$EXECUTION_PROVIDER")
  cli_args+=(--execution-provider "$EXECUTION_PROVIDER")
fi

if [[ "$EXECUTION_PROVIDER" == "cuda" ]]; then
  export LD_LIBRARY_PATH="$("$SCRIPT_DIR/cuda_library_path.sh"):${LD_LIBRARY_PATH:-}"
fi

"${probe_args[@]}"
"${cli_args[@]}"
