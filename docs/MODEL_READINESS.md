# Model Readiness

Use `tools\model_readiness.py` to compare source model capability, exported ORT GenAI package shape, and provenance status.

```powershell
python tools\model_readiness.py models\qwen3.5-2b models\qwen3.5-2b-onnxopt-q4f16 models\qwen3-vl-2b-instruct
```

Current local result:

| Package | Source Architecture | Runtime Type | Source Has Vision | Runtime Embedding | Runtime Vision | Processor Config | Classification | License Signal | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `models\qwen3.5-2b` | `Qwen3_5ForConditionalGeneration` | `qwen3_5` | yes | no | no | no | `prototype` | Apache-2.0 | Source is VLM, export is decoder-only. |
| `models\qwen3.5-2b-onnxopt-q4f16` | `Qwen3_5ForConditionalGeneration` | `qwen3_5` | yes | yes | yes | yes | `prototype` | Apache-2.0 | Runtime-ready local prototype; not production-cleared. |
| `models\qwen3-vl-2b-instruct` | `Qwen3VLForConditionalGeneration` | `qwen3_vl` | yes | yes | yes | yes | `smoke` | Apache-2.0 upstream signal | Runtime-ready for engineering smoke tests. |

## Qwen3.5 Gap

`Qwen/Qwen3.5-2B` is still the preferred target because it is small, VLM-capable, and Apache-2.0 in the captured provenance. There are now two local Qwen3.5 paths:

- `models\qwen3.5-2b`: direct ORT GenAI source-builder export. This remains decoder-only and is not a scene-description package.
- `models\qwen3.5-2b-onnxopt-q4f16`: experimental package assembled from ONNX-OPT decoder, embedding, and vision graphs. This runs through the C++ ORT GenAI backend locally.

Evidence:

- The source `config.json` includes `vision_config`, `image_token_id`, `video_token_id`, and vision boundary token IDs.
- ORT GenAI source `v0.13.1` maps `Qwen3_5ForConditionalGeneration` to `Qwen35TextModel`.
- The generated package has `model.onnx` and tokenizer files, but no `embedding.onnx`, `vision.onnx`, or image processor config.
- The generated `genai_config.json` decoder consumes `inputs_embeds`, which means a separate embedding path is expected but absent.
- `tools\prepare_qwen35_onnxopt_genai.py` patches the ONNX-OPT package into ORT GenAI shape: recurrent-state names, tokenizer metadata, image-feature injection, and `num_logits_to_keep`.
- `scene_describer.exe --config configs\qwen3.5-2b-onnxopt.ini --image tmp\smoke.png --json` returned a correct local scene description.

The practical next step is either:

- promote the ONNX-OPT preparation path only after legal/provenance review and target-device benchmarks, or
- extend/update the direct export path so Qwen3.5 emits an official complete ORT GenAI package internally.

## Promotion Rules

Do not promote a model package beyond engineering smoke tests until all are true:

- `tools\validate_model_package.py <dir> --require-multimodal --require-production` passes.
- `MODEL_PROVENANCE.json` classification is `production_candidate` or `production`.
- The model license chain is explicitly approved for commercial use.
- The exact package is benchmarked on the target edge device.
- The package can run from `tools\package_runtime.ps1` output without relying on development-only paths.
