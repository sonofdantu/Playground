# Architecture

## Runtime Flow

```text
image path
  -> Image loader
  -> SceneDescriptionRequest
  -> ISceneDescriber backend
  -> SceneDescription
  -> CLI text or JSON output
```

`scene_describer_benchmark` uses the same request/backend boundary, but constructs the backend once and runs repeated `Describe` calls so model load and per-request latency can be measured separately.

## Analyzer Flow

```text
image frames + track metadata + recent summaries
  -> AnalyzerRequest
  -> prompt templates
  -> SceneDescriptionRequest
  -> ISceneDescriber backend
  -> AnalyzerResult
```

The analyzer layer is intentionally above the model backend. It should be able to run against the current Qwen3-VL smoke package now and a Qwen3.5-2B GenAI package later without changing prompt/history/request ownership.

## Backends

`ISceneDescriber` is the stable boundary.

- `mock`: deterministic backend for smoke tests, CI, and CLI plumbing.
- `ort-genai`: intended ONNX Runtime GenAI implementation.

The ORT GenAI path is expected to use a packaged model directory containing:

- `genai_config.json`
- tokenizer files
- processor/preprocessor files
- `model.onnx`
- `vision.onnx`
- `embedding.onnx`

## Model Strategy

Preferred target:

- `Qwen/Qwen3.5-2B`
- Reason: Apache-2.0, small edge footprint, strong current VLM direction.
- Risk: newer hybrid decoder architecture needs export/runtime validation.

Fallback target:

- `Qwen/Qwen3-VL-2B-Instruct`
- Reason: Apache-2.0, explicit VLM family, lower risk than Qwen3.5 if export details are unstable.

Heavier quality target:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- Reason: mature and Apache-2.0, but significantly heavier for edge.

## Dependency Policy

Keep hard dependencies narrow:

- C++20 standard library.
- ONNX Runtime GenAI for production inference.
- A permissively licensed image decoder to be selected later.

Do not add AGPL/GPL dependencies to runtime code.
