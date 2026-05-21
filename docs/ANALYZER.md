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

The analyzer CLI supports repeated `--image` arguments. The current ORT GenAI backend now passes multiple image paths into `OgaImages::Load`, so the runtime boundary is ready for batch-style prompts. Quality and latency still need target-device testing for real multi-frame batches.

## Track Format

`--track` uses:

```text
frame_index,track_id,label,x,y,width,height,confidence
```

Example:

```powershell
--track "0,t1,vehicle,120,90,64,48,0.87"
```

## Current Limits

- This is a CLI smoke harness, not a ZMQ service yet.
- Request parsing is argument-based, not protocol-buffer or JSON-based.
- The working model is still Qwen3-VL smoke-classified.
- Qwen3.5-2B remains the target once multimodal GenAI export is solved.
