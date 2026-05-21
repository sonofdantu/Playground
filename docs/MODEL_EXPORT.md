# Model Export

This project targets ONNX Runtime GenAI model directories. Runtime inference is C++ only; Python is used only to export or inspect model packages.

## Current Export Path

Use the ONNX Runtime GenAI source model builder from the same version family as the C++ runtime. On this machine, `pip index versions onnxruntime-genai` currently tops out at `0.11.4`, while Qwen3.5/Qwen3-VL support is in ORT GenAI `0.13.x`. For that reason, the export scripts clone `microsoft/onnxruntime-genai` at `v0.13.1` into ignored `.deps/` and run `src/python/py/models/builder.py` directly.

The relevant model architectures have been checked from the public model configs:

- `Qwen/Qwen3.5-2B`: `Qwen3_5ForConditionalGeneration`
- `Qwen/Qwen3-VL-2B-Instruct`: `Qwen3VLForConditionalGeneration`

Both names are handled by the ORT GenAI `v0.13.1` builder source.

## One-Time Setup

```powershell
.\tools\setup_export_env.ps1
```

This creates `.venv/`, installs export dependencies from `tools/export-requirements.txt`, and clones the matching ORT GenAI source tag into `.deps/`.

If the Python dependencies are already installed and you only want to validate paths:

```powershell
.\tools\setup_export_env.ps1 -SkipPackageInstall
```

## Export Qwen3.5 2B

```powershell
.\tools\export_model.ps1 `
  -ModelName Qwen/Qwen3.5-2B `
  -OutputDir models\qwen3.5-2b `
  -Precision int4 `
  -ExecutionProvider cpu
```

Default extra options are `hf_remote=true` and `hf_token=false`. Remote code is an export-time trust decision; review the model repository before using this in a locked-down build pipeline.

Current result: this path exports the Qwen3.5 text decoder only. It generates `model.onnx` and `model.onnx.data`, but the generated `genai_config.json` does not include `model.embedding`, `model.vision`, or an image processor config. Keep this output as evidence for the export workstream; do not use it as the scene-description runtime package.

## Experimental Qwen3.5 ONNX-OPT Package

The currently working Qwen3.5 C++ path uses `onnx-community/Qwen3.5-2B-ONNX-OPT` as graph input and prepares it into an ORT GenAI-style directory:

```powershell
.\.venv\Scripts\python.exe tools\prepare_qwen35_onnxopt_genai.py `
  --output-dir models\qwen3.5-2b-onnxopt-q4f16 `
  --variant q4f16
```

The preparation script makes these package-level changes:

- patches decoder recurrent-state I/O names to ORT GenAI's `past_key_values.%d.conv_state` / `recurrent_state` convention;
- patches the embedding graph so `image_features` are scattered into `<|image_pad|>` token positions;
- normalizes tokenizer metadata from `TokenizersBackend` to the Qwen tokenizer path supported by ORT GenAI;
- normalizes the tokenizer split regex to the Qwen regex shape accepted by ORT GenAI;
- inlines decoder `num_logits_to_keep=0` because ORT GenAI's multimodal pipeline path does not currently attach that preset decoder input.

Validate it:

```powershell
python tools\validate_model_package.py models\qwen3.5-2b-onnxopt-q4f16 --require-multimodal --require-provenance
```

Probe it stage by stage:

```powershell
.\build\scene_model_probe.exe `
  --model-dir models\qwen3.5-2b-onnxopt-q4f16 `
  --image tmp\smoke.png `
  --stage generate `
  --max-new-tokens 32
```

Run the normal CLI:

```powershell
.\build\scene_describer.exe `
  --config configs\qwen3.5-2b-onnxopt.ini `
  --image tmp\smoke.png `
  --max-new-tokens 32 `
  --json
```

Current local status: this package runs on CPU through the C++ ORT GenAI backend and describes the smoke image correctly. It is still classified as `prototype`; do not treat it as production-cleared until the package source, graph patches, license chain, and target-device benchmarks are reviewed.

For a compact status view across candidates:

```powershell
python tools\model_readiness.py models\qwen3.5-2b models\qwen3.5-2b-onnxopt-q4f16 models\qwen3-vl-2b-instruct
```

See [MODEL_READINESS.md](MODEL_READINESS.md).

## Fallback Export: Qwen3-VL 2B

```powershell
.\tools\export_model.ps1 `
  -ModelName Qwen/Qwen3-VL-2B-Instruct `
  -OutputDir models\qwen3-vl-2b-instruct `
  -Precision int4 `
  -ExecutionProvider cpu
```

## Smoke-Test Package

`onnx-community/Qwen3-VL-2B-Instruct-ONNX` includes an ONNX Runtime GenAI CPU package at `onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32/` with `text.onnx`, `embedding.onnx`, `vision.onnx`, and `processor_config.json`.

That repository currently has no model card/license metadata, so treat it as an engineering smoke-test input only. For production, use an internally exported package or another package with an explicit license chain back to the Apache-2.0 Qwen model.

```powershell
.\.venv\Scripts\python.exe tools\fetch_hf_oga_package.py `
  --repo-id onnx-community/Qwen3-VL-2B-Instruct-ONNX `
  --source-prefix onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32 `
  --output-dir models\qwen3-vl-2b-instruct
```

Validate the package:

```powershell
python tools\validate_model_package.py models\qwen3-vl-2b-instruct --require-multimodal --require-provenance
```

See [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) for manifest and classification rules.

## Build Runtime Dependencies

```powershell
.\tools\fetch_runtime_deps.ps1
```

Then build the ORT GenAI-enabled C++ binary with the roots printed by the script:

```powershell
.\tools\build.ps1 -Clean -Test -OrtGenAI `
  -OrtGenAIRoot .deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64 `
  -OrtRuntimeRoot .deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

## Run

```powershell
.\build\scene_describer.exe `
  --config configs\qwen3.5-2b-onnxopt.ini `
  --image path\to\image.jpg `
  --json
```

The ORT GenAI backend passes image paths into ORT GenAI's multimodal processor. The bootstrap PPM/PGM decoder is only used by the mock backend.

Smoke-test command used locally:

```powershell
.\.venv\Scripts\python.exe tools\make_test_image.py
.\build\scene_describer.exe --config configs\qwen3-vl-2b.ini --image tmp\smoke.png --max-new-tokens 48 --json
```

Observed output described the synthetic scene correctly: a red house, green lawn, blue train, path, and sun.

The same flow is wrapped by:

```powershell
.\tools\smoke_ort_genai.ps1
```
