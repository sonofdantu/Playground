#!/usr/bin/env bash
set -euo pipefail

FRAME_COUNT=30
MAX_NEW_TOKENS=72
FRAMES_DIR="tmp/frames"
CONFIG_PATH="configs/qwen3.5-2b-onnxopt-cuda.ini"
REQUEST_ID="cuda-frame-batch"
FRAME_WIDTH=1920
FRAME_HEIGHT=1080
FRAME_FORMAT="jpg"
JPEG_QUALITY=85
SYNTHETIC_TRACKS=1
TRACK_STRIDE=5
DETAIL_CROPS=1
DETAIL_CROP_DIR="tmp/detail-crops"
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
    --config)
      CONFIG_PATH="$2"
      shift 2
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
    -h|--help)
      cat <<'EOH'
Usage: tools/smoke_cuda_frame_batch.sh [options]

Options:
  --frame-count <n>       Number of frames to generate and summarize, default 30
  --max-new-tokens <n>    Max generated tokens, default 72
  --frames-dir <path>     Frame output directory, default tmp/frames
  --width <n>             Generated source frame width, default 1920
  --height <n>            Generated source frame height, default 1080
  --frame-format <jpg|png> Generated source frame format, default jpg
  --jpeg-quality <n>      JPEG quality for source/detail frames, default 85
  --config <path>         CUDA runtime config, default configs/qwen3.5-2b-onnxopt-cuda.ini
  --no-tracks             Do not attach synthetic person/vehicle track metadata
  --track-stride <n>      Attach synthetic tracks every n frames, default 5
  --no-detail-crops       Do not add high-resolution drone/handheld detail crops
  --detail-crop-dir <p>   Detail crop output directory, default tmp/detail-crops
  --detail-sample-frames  Comma-separated source frames for detail crops, default middle frame
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
if [[ ! -x ./build/scene_analyzer ]]; then
  echo "error: missing ORT-enabled build; run ./tools/build.sh --ort-genai first" >&2
  exit 1
fi

./.venv/bin/python tools/make_test_frames.py \
  --output-dir "$FRAMES_DIR" \
  --count "$FRAME_COUNT" \
  --width "$FRAME_WIDTH" \
  --height "$FRAME_HEIGHT" \
  --format "$FRAME_FORMAT" \
  --jpeg-quality "$JPEG_QUALITY"

args=(./build/scene_analyzer --config "$CONFIG_PATH" --max-new-tokens "$MAX_NEW_TOKENS" --json --request-id "$REQUEST_ID")
denominator=$((FRAME_COUNT - 1))
if [[ "$denominator" -lt 1 ]]; then
  denominator=1
fi
for index in $(seq 0 $((FRAME_COUNT - 1))); do
  frame_path=$(printf '%s/frame_%04d.%s' "$FRAMES_DIR" "$index" "$FRAME_FORMAT")
  args+=(--image "$frame_path" --timestamp-ms $((index * 33)))
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
    args+=(--track "$index,person-1,person,$person_x,$person_y,$person_w,$person_h,0.95")
    args+=(--track "$index,vehicle-1,vehicle,$truck_x,$truck_y,$truck_w,$truck_h,0.91")
  fi
done
latest_timestamp_ms=$(((FRAME_COUNT - 1) * 33))
if [[ "$DETAIL_CROPS" -eq 1 ]]; then
  while IFS=$'\t' read -r source_index crop_kind crop_path crop_note; do
    [[ -n "$crop_path" ]] || continue
    args+=(--detail-image "$crop_path" --timestamp-ms "$latest_timestamp_ms" --frame-note "$crop_note")
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

output_path="tmp/cuda_frame_batch_${FRAME_COUNT}_${FRAME_WIDTH}x${FRAME_HEIGHT}_${FRAME_FORMAT}q${JPEG_QUALITY}.json"
LD_LIBRARY_PATH="$($SCRIPT_DIR/cuda_library_path.sh):${LD_LIBRARY_PATH:-}" "${args[@]}" | tee "$output_path"

./.venv/bin/python - "$output_path" "$FRAME_COUNT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_frames = sys.argv[2]
data = json.loads(path.read_text())
metadata = data.get("metadata", {})
detail_images = int(metadata.get("detail_image_count", "0"))
expected_image_count = str(int(expected_frames) + detail_images)
checks = {
    "execution_provider": "raw-ort-cuda",
    "model_type": "qwen3_5",
    "frame_count": expected_frames,
    "image_count": expected_image_count,
}
for key, expected in checks.items():
    actual = metadata.get(key)
    if actual != expected:
        raise SystemExit(f"{key} expected {expected!r}, got {actual!r}")
summary = data.get("summary", "").strip()
if len(summary) < 10:
    raise SystemExit("summary is unexpectedly short")
if detail_images > 0:
    summary_lower = summary.lower()
    if "drone" not in summary_lower:
        raise SystemExit("summary did not mention the detail-crop drone")
    if not any(term in summary_lower for term in ("holding", "holds", "carrying", "carries", "object")):
        raise SystemExit("summary did not mention the detail-crop held object")
print(f"frame batch smoke passed: frames={expected_frames} summary={summary!r}")
PY
