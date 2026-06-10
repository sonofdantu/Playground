# Analyzer Runtime

The analyzer layer is the C++ shape for turning image batches, track metadata, and recent summaries into a Qwen scene-description request:

```text
frames + track metadata + recent summaries
  -> prompt templates
  -> ONNX Runtime GenAI backend
  -> structured analyzer result
```

This is separate from SAM/GDINO/retraining work. The current backend can use the working Qwen3-VL GenAI package today and should later swap to Qwen3.5-2B when a complete multimodal GenAI package exists.

## Components

- `AnalyzerRequest`: request ID, frames, image paths, timestamps, tracks, and prior summaries.
- `TrackMetadata`: track ID, label, bounding box, confidence, and optional attributes.
- `HistoryStore`: bounded recent-summary store with monotonic timestamp enforcement.
- `BuildAnalyzerPrompt`: renders prompt templates into the final model prompt.
- `scene_analyzer.exe`: smoke CLI for the analyzer flow.

Prompt templates live in `prompts\analyzer`:

- `base_guardrails.txt`
- `task_rules.txt`
- `local_batch.txt`
- `history_context.txt`

## Smoke Command

```powershell
.\build\scene_analyzer.exe `
  --config configs\qwen3-vl-2b.fast.ini `
  --image tmp\smoke.png `
  --timestamp-ms 1000 `
  --request-id smoke-analyzer `
  --history "Earlier frame showed the same yard and toy train." `
  --track "0,t1,train,78,85,94,38,0.91" `
  --json
```

Observed local Qwen3-VL output:

```json
{
  "request_id": "smoke-analyzer",
  "summary": "A red house with a brown door and a white window is situated on a green lawn, with a blue toy train on a white path in front of it. A bright yellow sun is in the sky above the house.",
  "latest_timestamp_ms": 1000
}
```

The analyzer CLI supports repeated `--image` arguments and `--detail-image` crops. The current CUDA path has been verified with 30 generated 1920x1080 security frames plus two high-resolution detail crops through `tools/smoke_cuda_frame_batch.sh`.

Linux CUDA frame-batch smoke:

```bash
./tools/smoke_cuda_frame_batch.sh
```

Required evidence includes `metadata.execution_provider=raw-ort-cuda`, `metadata.frame_count=30`, `metadata.detail_image_count=2`, `metadata.image_count=32`, `metadata.track_count=14`, and `metadata.prefill_chunk_tokens=512`. The smoke also fails if the summary omits the detail-crop drone or the small held object.

Resident latency gate:

```bash
./tools/benchmark_cuda_frame_batch.sh --analyzer-prompt
```

On RTX4000-class hardware, this 30-frame 1080p JPEG quality-85 security-profile benchmark is expected to report warmed median latency below 4 seconds with the detail crops enabled. On the verified RTX 4070 Laptop WSL2 development machine, use `--no-target`; observed medians range from `5162.286` to `7468.595` ms depending on laptop/WSL state.

## Track Format

`--track` uses:

```text
frame_index,track_id,label,x,y,width,height,confidence
```

Example:

```powershell
--track "0,t1,vehicle,120,90,64,48,0.87"
```

`--detail-image` appends a crop as an additional model image while preserving `frame_count` as the number of source video frames:

```bash
--detail-image tmp/detail-crops/detail_0015_drone.png \
--timestamp-ms 957 \
--frame-note "High-resolution sky crop from source frame 15; inspect for small airborne objects such as drones."
```

## Current Limits

- This is a CLI smoke harness, not a ZMQ service yet.
- Request parsing is argument-based, not protocol-buffer or JSON-based.
- The Qwen3.5 ONNX-OPT package remains prototype-classified until provenance/legal and target-device performance review are complete.
- The direct ORT GenAI CUDA generator path still crashes; use the raw ONNX Runtime CUDA route for Qwen3.5 GPU frame batches.
