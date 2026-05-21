# Model Packages

Model artifacts are intentionally not committed.

Expected package layout for ONNX Runtime GenAI:

```text
models/<model-name>/
  genai_config.json
  tokenizer.json
  tokenizer_config.json
  preprocessor_config.json
  model.onnx
  vision.onnx
  embedding.onnx
```

The first candidate packages are:

- `qwen3.5-2b`: Apache-2.0 model weights, preferred edge-size target.
- `qwen3-vl-2b-instruct`: Apache-2.0 model weights, conservative VLM fallback.

