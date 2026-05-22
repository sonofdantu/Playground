# Runtime Assets

This repository commits source code, configs, docs, and scripts. It intentionally does not commit runtime binaries, ONNX model weights, build outputs, Python environments, or download caches.

This document is the contract for recreating those missing assets after a fresh clone.

## Required Assets To Run Qwen3.5

| Asset | Local Path | Created By | Committed? | Purpose |
| --- | --- | --- | --- | --- |
| ONNX Runtime GenAI SDK | `.deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64` | `tools\fetch_runtime_deps.ps1` | no | Provides `include\ort_genai.h`, `lib\onnxruntime-genai.lib`, and `lib\onnxruntime-genai.dll`. |
| ONNX Runtime SDK | `.deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1` | `tools\fetch_runtime_deps.ps1` | no | Provides the matching `lib\onnxruntime.dll` used by ORT GenAI. |
| Qwen3.5 multimodal model package | `models\qwen3.5-2b-onnxopt-q4f16` | `tools\prepare_qwen35_onnxopt_genai.py` | no | Contains `genai_config.json`, `processor_config.json`, tokenizer files, decoder/embedding/vision ONNX graphs, and external ONNX data shards. |
| ORT-enabled C++ build | `build\scene_describer.exe`, `build\scene_analyzer.exe`, `build\scene_model_probe.exe` | `tools\build.ps1 -OrtGenAI ...` | no | Executables linked against ONNX Runtime GenAI. |
| App-local runtime DLLs | `build\onnxruntime-genai.dll`, `build\onnxruntime.dll` | `tools\build.ps1 -OrtGenAI ...` | no | Ensures Windows loads the matching runtime DLLs instead of an unrelated system copy. |
| Smoke image | `tmp\smoke.png` | `tools\make_test_image.py` | no | Deterministic sample image used to prove image-to-description works. |

## ORT GenAI Dependency Path

Run:

```powershell
.\tools\fetch_runtime_deps.ps1
```

Expected roots:

```text
.deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64
.deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

The ORT GenAI root must contain:

```text
include\ort_genai.h
include\ort_genai_c.h
lib\onnxruntime-genai.lib
lib\onnxruntime-genai.dll
```

The ONNX Runtime root must contain:

```text
lib\onnxruntime.lib
lib\onnxruntime.dll
```

The runtime versions are intentionally pinned together in `tools\fetch_runtime_deps.ps1`:

```text
ONNX Runtime GenAI: 0.13.1
ONNX Runtime:       1.25.1
```

## Valid Multimodal Model Package

The working local Qwen3.5 model package is:

```text
models\qwen3.5-2b-onnxopt-q4f16
```

Create it with:

```powershell
.\.venv\Scripts\python.exe tools\prepare_qwen35_onnxopt_genai.py `
  --output-dir models\qwen3.5-2b-onnxopt-q4f16 `
  --variant q4f16
```

Required package shape:

```text
models\qwen3.5-2b-onnxopt-q4f16\
  genai_config.json
  processor_config.json
  tokenizer.json
  tokenizer_config.json
  MODEL_PROVENANCE.json
  UPSTREAM_LICENSE
  onnx\
    decoder_model_merged_q4f16.onnx
    decoder_model_merged_q4f16.onnx_data
    embed_tokens_q4f16.onnx
    embed_tokens_q4f16.onnx_data
    vision_encoder_q4f16.onnx
    vision_encoder_q4f16.onnx_data
```

Validate it with:

```powershell
.\.venv\Scripts\python.exe tools\validate_model_package.py `
  models\qwen3.5-2b-onnxopt-q4f16 `
  --require-multimodal `
  --require-provenance
```

Important: `models\qwen3.5-2b` is not the working scene-description package. That directory is the direct ORT GenAI source-builder export and is decoder-only. The working package is `models\qwen3.5-2b-onnxopt-q4f16`.

## ORT-Enabled Build Path

Build with:

```powershell
.\tools\build.ps1 -Test -OrtGenAI `
  -OrtGenAIRoot .deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64 `
  -OrtRuntimeRoot .deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

This does three runtime-critical things:

- configures CMake with `SCENE_DESC_ENABLE_ORT_GENAI=ON`;
- points `FindOnnxRuntimeGenAI.cmake` at the ORT GenAI include/lib root;
- copies `onnxruntime-genai.dll` and the matching `onnxruntime.dll` beside the built executables.

The ORT backend is compiled only when `SCENE_DESC_ENABLE_ORT_GENAI=ON`. Without that flag, `backend=ort-genai` is unavailable and only the mock backend is useful.

## Runtime Config Path

Use:

```text
configs\qwen3.5-2b-onnxopt.ini
```

Current config:

```ini
backend=ort-genai
model_dir=models/qwen3.5-2b-onnxopt-q4f16
execution_provider=cpu
prompt=Describe the visible scene in one concise sentence. Mention only clear objects and actions.
max_new_tokens=48
temperature=0.6
top_p=0.95
deterministic=false
emit_json=true
```

The key fields are:

- `backend=ort-genai`: selects the C++ ONNX Runtime GenAI backend.
- `model_dir=models/qwen3.5-2b-onnxopt-q4f16`: points at the valid multimodal model package.
- `execution_provider=cpu`: starts with CPU for reproducibility. Use target-specific EPs only after validating their runtime packages.

## Proof Commands

Create the smoke image:

```powershell
.\.venv\Scripts\python.exe tools\make_test_image.py
```

Probe GenAI stage by stage:

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

Expected proof signals:

- JSON contains `"backend": "ort-genai"`.
- Metadata contains `"model_type": "qwen3_5"`.
- Text describes the sample image: red house, blue vehicle, yellow sun.

## Packaging For Another Machine

Use:

```powershell
.\tools\package_runtime.ps1 `
  -OutputDir dist\scene-describer-qwen35 `
  -ConfigPath configs\qwen3.5-2b-onnxopt.ini `
  -ModelDir models\qwen3.5-2b-onnxopt-q4f16 `
  -Force
```

By default, the package references the model directory instead of copying model files. Use `-IncludeModel` only for internal artifact packaging after license/provenance review.
