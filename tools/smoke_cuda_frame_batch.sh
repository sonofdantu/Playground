#!/usr/bin/env bash
set -euo pipefail

FRAME_COUNT=120
MAX_NEW_TOKENS=24
FRAMES_DIR="tmp/frames"
CONFIG_PATH="configs/qwen3.5-2b-onnxopt-cuda.ini"
REQUEST_ID="cuda-frame-batch"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --frame-count)
      FRAME_COUNT="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --frames-dir)
      FRAMES_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOH'
Usage: tools/smoke_cuda_frame_batch.sh [options]

Options:
  --frame-count <n>       Number of frames to generate and summarize, default 120
  --max-new-tokens <n>    Max generated tokens, default 24
  --frames-dir <path>     Frame output directory, default tmp/frames
  --config <path>         CUDA runtime config, default configs/qwen3.5-2b-onnxopt-cuda.ini
EOH
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

if [[ "$FRAME_COUNT" -lt 1 ]]; then
  echo "error: --frame-count must be at least 1" >&2
  exit 2
fi
if [[ ! -x ./.venv/bin/python ]]; then
  echo "error: missing .venv; run ./tools/setup_linux_env.sh first" >&2
  exit 1
fi
if [[ ! -x ./build/scene_analyzer ]]; then
  echo "error: missing ORT-enabled build; run ./tools/build.sh --ort-genai first" >&2
  exit 1
fi

./.venv/bin/python tools/make_test_frames.py --output-dir "$FRAMES_DIR" --count "$FRAME_COUNT"

args=(./build/scene_analyzer --config "$CONFIG_PATH" --max-new-tokens "$MAX_NEW_TOKENS" --json --request-id "$REQUEST_ID")
for index in $(seq 0 $((FRAME_COUNT - 1))); do
  frame_path=$(printf '%s/frame_%04d.png' "$FRAMES_DIR" "$index")
  args+=(--image "$frame_path" --timestamp-ms $((index * 33)))
done

output_path="tmp/cuda_frame_batch_${FRAME_COUNT}.json"
LD_LIBRARY_PATH="$($SCRIPT_DIR/cuda_library_path.sh):${LD_LIBRARY_PATH:-}" "${args[@]}" | tee "$output_path"

./.venv/bin/python - "$output_path" "$FRAME_COUNT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_frames = sys.argv[2]
data = json.loads(path.read_text())
metadata = data.get("metadata", {})
checks = {
    "execution_provider": "raw-ort-cuda",
    "model_type": "qwen3_5",
    "frame_count": expected_frames,
    "image_count": expected_frames,
}
for key, expected in checks.items():
    actual = metadata.get(key)
    if actual != expected:
        raise SystemExit(f"{key} expected {expected!r}, got {actual!r}")
summary = data.get("summary", "").strip()
if len(summary) < 10:
    raise SystemExit("summary is unexpectedly short")
print(f"frame batch smoke passed: frames={expected_frames} summary={summary!r}")
PY
