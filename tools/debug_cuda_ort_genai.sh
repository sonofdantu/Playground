#!/usr/bin/env bash
set -uo pipefail

MODEL_DIR="models/qwen3.5-2b-onnxopt-q4f16"
IMAGE_PATH="tmp/smoke.png"
MAX_NEW_TOKENS=1
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
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
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/debug_cuda_ort_genai.sh [options]

Runs CUDA ORT GenAI diagnostics and writes logs under tmp/cuda-debug-*.

Options:
  --model-dir <path>       Model package, default models/qwen3.5-2b-onnxopt-q4f16
  --image <path>           Smoke image, default tmp/smoke.png
  --max-new-tokens <n>     Token probe length, default 1
  --out-dir <path>         Explicit output directory
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
cd "$REPO_ROOT"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="tmp/cuda-debug-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

CUDA_LIBRARY_PATH="$("$SCRIPT_DIR/cuda_library_path.sh")"
export LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"

run_log() {
  local name="$1"
  shift
  {
    echo "$ $*"
    "$@"
    echo "exit=$?"
  } >"$OUT_DIR/$name.log" 2>&1
}

run_shell_log() {
  local name="$1"
  local command="$2"
  {
    echo "$ $command"
    bash -lc "$command"
    echo "exit=$?"
  } >"$OUT_DIR/$name.log" 2>&1
}

{
  echo "repo=$REPO_ROOT"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "model_dir=$MODEL_DIR"
  echo "image_path=$IMAGE_PATH"
  echo "ld_library_path=$LD_LIBRARY_PATH"
  echo "cuda_module_loading=${CUDA_MODULE_LOADING:-}"
  echo "uname=$(uname -a)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$OUT_DIR/environment.txt"

run_shell_log nvidia-smi "nvidia-smi"
run_shell_log nvidia-smi-query "nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv"
run_shell_log build-libs "find build -maxdepth 1 -type f -name 'lib*.so*' -printf '%f\\n' | sort"
run_shell_log ldd-scene-model-probe "ldd build/scene_model_probe"
run_shell_log ldd-genai-cuda "ldd build/libonnxruntime-genai-cuda.so"
run_shell_log ldd-ort-cuda "ldd build/libonnxruntime_providers_cuda.so"
run_shell_log python-packages ".venv/bin/python -m pip freeze | grep -E 'onnxruntime|nvidia-' | sort"
run_shell_log dmesg-before "dmesg | tail -80"

if [[ ! -f "$IMAGE_PATH" && "$IMAGE_PATH" == "tmp/smoke.png" ]]; then
  run_log make-test-image .venv/bin/python tools/make_test_image.py
fi

run_log validate-model .venv/bin/python tools/validate_model_package.py "$MODEL_DIR" --require-multimodal --require-provenance

run_log model-stage ./build/scene_model_probe \
  --model-dir "$MODEL_DIR" \
  --image "$IMAGE_PATH" \
  --execution-provider cuda \
  --stage model
model_status="$(tail -n 1 "$OUT_DIR/model-stage.log" | sed 's/^exit=//')"

run_log token-stage ./build/scene_model_probe \
  --model-dir "$MODEL_DIR" \
  --image "$IMAGE_PATH" \
  --execution-provider cuda \
  --stage token \
  --max-new-tokens "$MAX_NEW_TOKENS"
token_status="$(tail -n 1 "$OUT_DIR/token-stage.log" | sed 's/^exit=//')"

if command -v gdb >/dev/null 2>&1; then
  {
    echo "$ gdb -batch ..."
    gdb -batch \
      -ex "set env LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
      -ex "run" \
      -ex "thread apply all bt" \
      --args ./build/scene_model_probe \
        --model-dir "$MODEL_DIR" \
        --image "$IMAGE_PATH" \
        --execution-provider cuda \
        --stage token \
        --max-new-tokens "$MAX_NEW_TOKENS"
    echo "exit=$?"
  } >"$OUT_DIR/gdb-token-stage.log" 2>&1
else
  echo "gdb not found" >"$OUT_DIR/gdb-token-stage.log"
fi

run_shell_log dmesg-after "dmesg | tail -120"

cat >"$OUT_DIR/summary.txt" <<EOF
CUDA ORT GenAI debug summary

model_stage_exit=$model_status
token_stage_exit=$token_status

Expected current WSL2 blocker:
- model stage exits 0 and prints device=CUDA
- token stage exits 139 when ORT GenAI CUDA generation segfaults

Logs: $OUT_DIR
EOF

cat "$OUT_DIR/summary.txt"

if [[ "$token_status" == "0" ]]; then
  exit 0
fi
exit "$token_status"
