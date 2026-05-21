# Model Packaging

The runtime expects ONNX Runtime GenAI-style model directories.

## Target Layout

```text
models/<name>/
  genai_config.json
  tokenizer.json
  tokenizer_config.json
  preprocessor_config.json
  model.onnx
  vision.onnx
  embedding.onnx
  LICENSE
  MODEL_CARD.md
```

## Candidate Export Order

1. Try `Qwen/Qwen3.5-2B` with ONNX Runtime GenAI v0.13.1 or newer.
2. If Qwen3.5 export/runtime quality is blocked, try `Qwen/Qwen3-VL-2B-Instruct`.
3. Use `Qwen/Qwen2.5-VL-7B-Instruct` only if 2B quality is inadequate and the edge device can handle it.

See [MODEL_EXPORT.md](MODEL_EXPORT.md) for the scripted export path.

## Acceptance Criteria

- Model package loads without Python at runtime.
- CLI returns a non-empty description for a local image.
- Output is stable with deterministic decoding.
- Memory use is measured on the target device.
- Model and dependency licenses are captured in `docs/LICENSES.md`.
