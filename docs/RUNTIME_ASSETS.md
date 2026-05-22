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

Linux equivalents:

| Asset | Local Path | Created By | Committed? | Purpose |
| --- | --- | --- | --- | --- |
| Linux Python/build environment | `.venv` | `tools/setup_linux_env.sh` | no | Provides local CMake, Ninja, and lightweight Qwen3.5 preparation packages. |
| ONNX Runtime GenAI SDK | `.deps/onnxruntime-genai-0.13.1-linux-x64/onnxruntime-genai-0.13.1-linux-x64` | `tools/fetch_runtime_deps.sh` | no | Provides `include/ort_genai.h`, `lib/libonnxruntime-genai.so`, and headers. |
| ONNX Runtime SDK | `.deps/onnxruntime-linux-x64-1.25.1/onnxruntime-linux-x64-1.25.1` | `tools/fetch_runtime_deps.sh` | no | Provides matching `lib/libonnxruntime.so`. |
| ORT-enabled C++ build | `build/scene_describer`, `build/scene_analyzer`, `build/scene_model_probe` | `tools/build.sh --ort-genai` | no | Linux executables linked against ONNX Runtime GenAI. |
| App-local runtime shared libraries | `build/libonnxruntime-genai.so`, `build/libonnxruntime.so` | `tools/build.sh --ort-genai` | no | Lets the executable rpath or `LD_LIBRARY_PATH=$PWD/build` resolve matching runtime libraries. |

## ORT GenAI Dependency Path

Run:

Linux:

```bash
./tools/setup_linux_env.sh
./tools/fetch_runtime_deps.sh
```

Windows:

```powershell
.\tools\fetch_runtime_deps.ps1
```

Expected roots:

```text
.deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64
.deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

Expected Linux roots:

```text
.deps/onnxruntime-genai-0.13.1-linux-x64/onnxruntime-genai-0.13.1-linux-x64
.deps/onnxruntime-linux-x64-1.25.1/onnxruntime-linux-x64-1.25.1
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

On Linux, the ORT GenAI root must contain:

```text
include/ort_genai.h
include/ort_genai_c.h
lib/libonnxruntime-genai.so
```

The Linux ONNX Runtime root must contain:

```text
lib/libonnxruntime.so
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

Linux:

```bash
./.venv/bin/python tools/prepare_qwen35_onnxopt_genai.py \
  --output-dir models/qwen3.5-2b-onnxopt-q4f16 \
  --variant q4f16
```

Windows:

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

Linux:

```bash
./.venv/bin/python tools/validate_model_package.py \
  models/qwen3.5-2b-onnxopt-q4f16 \
  --require-multimodal \
  --require-provenance
```

Windows:

```powershell
.\.venv\Scripts\python.exe tools\validate_model_package.py `
  models\qwen3.5-2b-onnxopt-q4f16 `
  --require-multimodal `
  --require-provenance
```

Important: `models\qwen3.5-2b` is not the working scene-description package. That directory is the direct ORT GenAI source-builder export and is decoder-only. The working package is `models\qwen3.5-2b-onnxopt-q4f16`.

## ORT-Enabled Build Path

Build with:

Linux:

```bash
./tools/build.sh --test --ort-genai
```

Or with explicit roots:

```bash
./tools/build.sh --test --ort-genai \
  --ort-genai-root .deps/onnxruntime-genai-0.13.1-linux-x64/onnxruntime-genai-0.13.1-linux-x64 \
  --ort-runtime-root .deps/onnxruntime-linux-x64-1.25.1/onnxruntime-linux-x64-1.25.1
```

Windows:

```powershell
.\tools\build.ps1 -Test -OrtGenAI `
  -OrtGenAIRoot .deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64 `
  -OrtRuntimeRoot .deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

This does three runtime-critical things:

- configures CMake with `SCENE_DESC_ENABLE_ORT_GENAI=ON`;
- points `FindOnnxRuntimeGenAI.cmake` at the ORT GenAI include/lib root;
- copies `onnxruntime-genai.dll` / `onnxruntime.dll` on Windows or `libonnxruntime-genai.so` / `libonnxruntime.so` on Linux beside the built executables.

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

Linux:

```bash
./.venv/bin/python tools/make_test_image.py
```

Windows:

```powershell
.\.venv\Scripts\python.exe tools\make_test_image.py
```

Probe GenAI stage by stage:

Linux:

```bash
./build/scene_model_probe \
  --model-dir models/qwen3.5-2b-onnxopt-q4f16 \
  --image tmp/smoke.png \
  --stage generate \
  --max-new-tokens 32
```

Windows:

```powershell
.\build\scene_model_probe.exe `
  --model-dir models\qwen3.5-2b-onnxopt-q4f16 `
  --image tmp\smoke.png `
  --stage generate `
  --max-new-tokens 32
```

Run the normal CLI:

Linux:

```bash
./build/scene_describer \
  --config configs/qwen3.5-2b-onnxopt.ini \
  --image tmp/smoke.png \
  --max-new-tokens 32 \
  --json
```

Linux smoke gate:

```bash
./tools/smoke_ort_genai.sh
```

Windows:

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
