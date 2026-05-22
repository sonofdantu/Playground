# Quickstart

This guide recreates the local Qwen3.5 C++ scene-description result from a fresh clone. The repository intentionally does not include model weights, build outputs, dependency DLLs, Python environments, or Hugging Face cache files.

## What To Read First

For a human or AI agent starting cold:

1. Read this file.
2. Read `docs/RUNTIME_ASSETS.md` to understand what files are generated locally and why they are not committed.
3. Read `PROJECT_STATE.md` for the current technical status and known limits.
4. Read `docs/MODEL_EXPORT.md` for the Qwen3.5 package-preparation details.
5. Read `docs/MODEL_READINESS.md` before treating any model package as production-ready.

## Prerequisites

- Windows PowerShell.
- Python 3.10 or newer available as `python`.
- Visual Studio 2022 Build Tools with MSVC, CMake, and Ninja installed.
- Network access to GitHub and Hugging Face for dependencies and model artifacts.
- Enough disk space for generated artifacts. A lean Qwen3.5 setup is roughly 2.5 GB; full local caches can exceed 10 GB.

## Linux Copy/Paste Setup

Use this path on a Linux x64 machine with Python 3, CMake, and a C++20 compiler installed:

```bash
git clone https://github.com/sonofdantu/Playground.git
cd Playground

./tools/fetch_runtime_deps.sh
./tools/setup_export_env.sh

./.venv/bin/python tools/prepare_qwen35_onnxopt_genai.py \
  --output-dir models/qwen3.5-2b-onnxopt-q4f16 \
  --variant q4f16

./tools/build.sh --test --ort-genai

./.venv/bin/python tools/make_test_image.py

LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" ./build/scene_describer \
  --config configs/qwen3.5-2b-onnxopt.ini \
  --image tmp/smoke.png \
  --max-new-tokens 32 \
  --json
```

If your checkout loses executable bits, run:

```bash
chmod +x tools/*.sh
```

## Windows Fresh Clone Setup

```powershell
git clone https://github.com/sonofdantu/Playground.git
cd Playground
```

Fetch ONNX Runtime GenAI and ONNX Runtime runtime dependencies:

```powershell
.\tools\fetch_runtime_deps.ps1
```

This creates `.deps\onnxruntime-genai-0.13.1-win-x64\...` and `.deps\onnxruntime-win-x64-1.25.1\...`. See `docs/RUNTIME_ASSETS.md` for the exact asset contract.

Set up the Python export/preparation environment:

```powershell
.\tools\setup_export_env.ps1
```

Prepare the working Qwen3.5 ONNX-OPT package:

```powershell
.\.venv\Scripts\python.exe tools\prepare_qwen35_onnxopt_genai.py `
  --output-dir models\qwen3.5-2b-onnxopt-q4f16 `
  --variant q4f16
```

Build the C++ runtime with ORT GenAI:

```powershell
.\tools\build.ps1 -Test -OrtGenAI `
  -OrtGenAIRoot .deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64 `
  -OrtRuntimeRoot .deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

This enables `SCENE_DESC_ENABLE_ORT_GENAI=ON`, links `onnxruntime-genai.lib`, and copies `onnxruntime-genai.dll` plus the matching `onnxruntime.dll` into `build`.

Generate the sample image:

```powershell
.\.venv\Scripts\python.exe tools\make_test_image.py
```

Run the C++ Qwen3.5 scene describer:

```powershell
.\build\scene_describer.exe `
  --config configs\qwen3.5-2b-onnxopt.ini `
  --image tmp\smoke.png `
  --max-new-tokens 32 `
  --json
```

Expected result shape:

```json
{
  "text": "A blue truck drives past a red house under a yellow sun.",
  "backend": "ort-genai",
  "metadata": {
    "model_type": "qwen3_5"
  }
}
```

The wording can vary because the config uses sampling. The important proof is `backend=ort-genai`, `model_type=qwen3_5`, and a scene description matching `tmp\smoke.png`.

## Linux Analyzer Smoke

```bash
LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" ./build/scene_analyzer \
  --config configs/qwen3.5-2b-onnxopt.ini \
  --image tmp/smoke.png \
  --timestamp-ms 1000 \
  --request-id smoke-analyzer \
  --track "0,t1,vehicle,78,85,94,38,0.91" \
  --json
```

## Windows Analyzer Smoke

```powershell
.\build\scene_analyzer.exe `
  --config configs\qwen3.5-2b-onnxopt.ini `
  --image tmp\smoke.png `
  --timestamp-ms 1000 `
  --request-id smoke-analyzer `
  --track "0,t1,vehicle,78,85,94,38,0.91" `
  --json
```

## Validate The Setup

```powershell
.\.venv\Scripts\python.exe tools\validate_project.py
.\.venv\Scripts\python.exe tools\validate_model_package.py models\qwen3.5-2b-onnxopt-q4f16 --require-multimodal --require-provenance
.\.venv\Scripts\python.exe tools\model_readiness.py models\qwen3.5-2b-onnxopt-q4f16
```

For staged runtime debugging:

Linux:

```bash
LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" ./build/scene_model_probe \
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

## Ignored Local Artifacts

These are intentionally not committed:

- `models\qwen3.5-2b-onnxopt-q4f16`
- `.deps`
- `.venv`
- `.cache`
- `build`
- `dist`
- `tmp`

Recreate them with the commands above. Do not commit ONNX model weights, runtime DLLs, build outputs, Python bytecode, or downloaded caches to normal Git.

## Common Problems

- If `build.ps1` cannot find Visual Studio tools, install Visual Studio 2022 Build Tools with the C++ workload.
- If the Qwen3.5 preparation script gets rate-limited by Hugging Face, rerun it later or configure Hugging Face authentication outside this repo.
- If Windows loads the wrong `onnxruntime.dll`, rebuild with `tools\build.ps1 -OrtGenAI ... -OrtRuntimeRoot ...`; the script copies the matching DLLs beside the executables.
- If `git push` fails with a 403, local Git credentials are authenticated to a GitHub account that lacks access to the remote.
