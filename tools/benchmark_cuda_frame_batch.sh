#!/usr/bin/env bash
set -euo pipefail

FRAME_COUNT=30
MAX_NEW_TOKENS=72
FRAMES_DIR="tmp/frames-benchmark"
CONFIG_PATH="configs/qwen3.5-2b-onnxopt-cuda.ini"
FRAME_WIDTH=1920
FRAME_HEIGHT=1080
FRAME_FORMAT="jpg"
JPEG_QUALITY=85
WARMUPS=1
REPEATS=3
TARGET_MS=4000
PROMPT_TEXT=""
PROMPT_FILE=""
ANALYZER_PROMPT=0
SYNTHETIC_TRACKS=1
TRACK_STRIDE=5
DETAIL_CROPS=1
DETAIL_CROP_DIR="tmp/detail-crops-benchmark"
DETAIL_SAMPLE_FRAMES=""

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
    --width)
      FRAME_WIDTH="$2"
      shift 2
      ;;
    --height)
      FRAME_HEIGHT="$2"
      shift 2
      ;;
    --frame-format)
      FRAME_FORMAT="$2"
      shift 2
      ;;
    --jpeg-quality)
      JPEG_QUALITY="$2"
      shift 2
      ;;
    --warmups)
      WARMUPS="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --target-ms)
      TARGET_MS="$2"
      shift 2
      ;;
    --prompt)
      PROMPT_TEXT="$2"
      shift 2
      ;;
    --prompt-file)
      PROMPT_FILE="$2"
      shift 2
      ;;
    --analyzer-prompt)
      ANALYZER_PROMPT=1
      shift
      ;;
    --no-tracks)
      SYNTHETIC_TRACKS=0
      shift
      ;;
    --track-stride)
      TRACK_STRIDE="$2"
      shift 2
      ;;
    --no-detail-crops)
      DETAIL_CROPS=0
      shift
      ;;
    --detail-crop-dir)
      DETAIL_CROP_DIR="$2"
      shift 2
      ;;
    --detail-sample-frames)
      DETAIL_SAMPLE_FRAMES="$2"
      shift 2
      ;;
    --no-target)
      TARGET_MS=0
      shift
      ;;
    -h|--help)
      cat <<'EOH'
Usage: tools/benchmark_cuda_frame_batch.sh [options]

Options:
  --frame-count <n>       Number of frames to summarize, default 30
  --max-new-tokens <n>    Max generated tokens, default 72
  --frames-dir <path>     Frame output directory, default tmp/frames-benchmark
  --config <path>         CUDA runtime config, default configs/qwen3.5-2b-onnxopt-cuda.ini
  --width <n>             Generated source frame width, default 1920
  --height <n>            Generated source frame height, default 1080
  --frame-format <jpg|png> Generated source frame format, default jpg
  --jpeg-quality <n>      JPEG quality for source/detail frames, default 85
  --warmups <n>           Warmup requests before timing, default 1
  --repeats <n>           Measured requests, default 3
  --target-ms <n>         Fail if median warmed latency is above this, default 4000
  --prompt <text>         Override the model prompt used by the benchmark
  --prompt-file <path>    Read the model prompt from a file
  --analyzer-prompt       Build and benchmark the analyzer's full frame-batch prompt
  --no-tracks             Do not attach synthetic person/vehicle tracks to analyzer prompt
  --track-stride <n>      Attach synthetic tracks every n frames, default 5
  --no-detail-crops       Do not add high-resolution drone/handheld detail crops
  --detail-crop-dir <p>   Detail crop output directory, default tmp/detail-crops-benchmark
  --detail-sample-frames  Comma-separated source frames for detail crops, default middle frame
  --no-target             Report latency without enforcing a target
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
if [[ "$FRAME_WIDTH" -lt 64 || "$FRAME_HEIGHT" -lt 64 ]]; then
  echo "error: --width and --height must be at least 64" >&2
  exit 2
fi
if [[ "$FRAME_FORMAT" != "jpg" && "$FRAME_FORMAT" != "png" ]]; then
  echo "error: --frame-format must be jpg or png" >&2
  exit 2
fi
if [[ "$JPEG_QUALITY" -lt 1 || "$JPEG_QUALITY" -gt 100 ]]; then
  echo "error: --jpeg-quality must be between 1 and 100" >&2
  exit 2
fi
if [[ "$WARMUPS" -lt 0 || "$REPEATS" -lt 1 ]]; then
  echo "error: --warmups must be >= 0 and --repeats must be >= 1" >&2
  exit 2
fi
if [[ "$TRACK_STRIDE" -lt 1 ]]; then
  echo "error: --track-stride must be at least 1" >&2
  exit 2
fi
if [[ -z "$DETAIL_SAMPLE_FRAMES" ]]; then
  DETAIL_SAMPLE_FRAMES="$((FRAME_COUNT / 2))"
fi
if [[ ! -x ./.venv/bin/python ]]; then
  echo "error: missing .venv; run ./tools/setup_linux_env.sh first" >&2
  exit 1
fi
if [[ ! -x ./build/scene_describer_benchmark ]]; then
  echo "error: missing ORT-enabled build; run ./tools/build.sh --ort-genai first" >&2
  exit 1
fi
if [[ "$ANALYZER_PROMPT" -eq 1 && ! -x ./build/scene_analyzer ]]; then
  echo "error: missing scene_analyzer; run ./tools/build.sh --ort-genai first" >&2
  exit 1
fi
if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
  echo "error: --prompt-file does not exist: $PROMPT_FILE" >&2
  exit 1
fi

./.venv/bin/python tools/make_test_frames.py \
  --output-dir "$FRAMES_DIR" \
  --count "$FRAME_COUNT" \
  --width "$FRAME_WIDTH" \
  --height "$FRAME_HEIGHT" \
  --format "$FRAME_FORMAT" \
  --jpeg-quality "$JPEG_QUALITY"

benchmark_image_paths=()
for index in $(seq 0 $((FRAME_COUNT - 1))); do
  frame_path=$(printf '%s/frame_%04d.%s' "$FRAMES_DIR" "$index" "$FRAME_FORMAT")
  benchmark_image_paths+=("$frame_path")
done

if [[ "$ANALYZER_PROMPT" -eq 1 ]]; then
  prompt_args=(./build/scene_analyzer --config "$CONFIG_PATH" --print-prompt --request-id cuda-frame-benchmark)
  denominator=$((FRAME_COUNT - 1))
  if [[ "$denominator" -lt 1 ]]; then
    denominator=1
  fi
  for index in $(seq 0 $((FRAME_COUNT - 1))); do
    frame_path=$(printf '%s/frame_%04d.%s' "$FRAMES_DIR" "$index" "$FRAME_FORMAT")
    prompt_args+=(--image "$frame_path" --timestamp-ms $((index * 33)))
    if [[ "$SYNTHETIC_TRACKS" -eq 1 && ( "$index" -eq 0 || "$index" -eq $((FRAME_COUNT - 1)) || $((index % TRACK_STRIDE)) -eq 0 ) ]]; then
      person_center_x=$(((FRAME_WIDTH * 74 / 100) - ((FRAME_WIDTH * 30 / 100) * index / denominator)))
      person_x=$((person_center_x - FRAME_WIDTH * 7 / 100))
      person_y=$((FRAME_HEIGHT * 60 / 100))
      person_w=$((FRAME_WIDTH * 14 / 100))
      person_h=$((FRAME_HEIGHT * 23 / 100))
      truck_x=$(((FRAME_WIDTH * 3 / 100) + ((FRAME_WIDTH * 60 / 100) * index / denominator)))
      truck_y=$((FRAME_HEIGHT * 63 / 100))
      truck_w=$((FRAME_WIDTH * 20 / 100))
      truck_h=$((FRAME_HEIGHT * 18 / 100))
      prompt_args+=(--track "$index,person-1,person,$person_x,$person_y,$person_w,$person_h,0.95")
      prompt_args+=(--track "$index,vehicle-1,vehicle,$truck_x,$truck_y,$truck_w,$truck_h,0.91")
    fi
  done
  latest_timestamp_ms=$(((FRAME_COUNT - 1) * 33))
  if [[ "$DETAIL_CROPS" -eq 1 ]]; then
    while IFS=$'\t' read -r source_index crop_kind crop_path crop_note; do
      [[ -n "$crop_path" ]] || continue
      prompt_args+=(--detail-image "$crop_path" --timestamp-ms "$latest_timestamp_ms" --frame-note "$crop_note")
      benchmark_image_paths+=("$crop_path")
    done < <(
      ./.venv/bin/python tools/make_detail_crops.py \
        --frames-dir "$FRAMES_DIR" \
        --output-dir "$DETAIL_CROP_DIR" \
        --count "$FRAME_COUNT" \
        --frame-format "$FRAME_FORMAT" \
        --output-format "$FRAME_FORMAT" \
        --jpeg-quality "$JPEG_QUALITY" \
        --sample-frames "$DETAIL_SAMPLE_FRAMES"
    )
  fi
  PROMPT_TEXT="$("${prompt_args[@]}")"
elif [[ -n "$PROMPT_FILE" ]]; then
  PROMPT_TEXT="$(cat "$PROMPT_FILE")"
fi

args=(
  ./build/scene_describer_benchmark
  --config "$CONFIG_PATH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --warmups "$WARMUPS"
  --repeats "$REPEATS"
  --json
)
if [[ -n "$PROMPT_TEXT" ]]; then
  args+=(--prompt "$PROMPT_TEXT")
fi
for image_path in "${benchmark_image_paths[@]}"; do
  args+=(--image "$image_path")
done

output_path="tmp/cuda_frame_batch_benchmark_${FRAME_COUNT}_${FRAME_WIDTH}x${FRAME_HEIGHT}_${FRAME_FORMAT}q${JPEG_QUALITY}.json"
LD_LIBRARY_PATH="$($SCRIPT_DIR/cuda_library_path.sh):${LD_LIBRARY_PATH:-}" "${args[@]}" | tee "$output_path"

./.venv/bin/python - "$output_path" "$FRAME_COUNT" "$TARGET_MS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_frames = int(sys.argv[2])
target_ms = float(sys.argv[3])
data = json.loads(path.read_text())
metadata = data.get("metadata", {})
actual_image_count = int(metadata.get("image_count", "0"))

checks = {
    "execution_provider": "raw-ort-cuda",
    "model_type": "qwen3_5",
}
for key, expected in checks.items():
    actual = metadata.get(key)
    if actual != expected:
        raise SystemExit(f"{key} expected {expected!r}, got {actual!r}")

if data.get("image_count") != actual_image_count:
    raise SystemExit(f"top-level image_count expected {actual_image_count}, got {data.get('image_count')!r}")
if actual_image_count < expected_frames:
    raise SystemExit(f"image_count must be at least source frame count {expected_frames}, got {actual_image_count}")

summary = data.get("last_text", "").strip()
if len(summary) < 10:
    raise SystemExit("summary is unexpectedly short")

latency = data.get("latency_ms", {})
median = float(latency["median"])
print(f"CUDA frame benchmark passed: frames={expected_frames} median_ms={median:.3f} summary={summary!r}")
if target_ms > 0 and median > target_ms:
    raise SystemExit(f"median latency {median:.3f} ms exceeded target {target_ms:.3f} ms")
PY
