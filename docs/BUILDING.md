# Building

## Required Tools

- Python 3 for local tooling.
- C++20 compiler.
- CMake 3.24 or newer. On Linux, `tools/setup_linux_env.sh` can install local CMake/Ninja into `.venv`.
- Optional: ONNX Runtime GenAI v0.13.1 or newer for Qwen3.5/Qwen3-VL work.

## Linux Quick Build

On Linux x64, use the shell wrappers:

```bash
./tools/setup_linux_env.sh
./tools/fetch_runtime_deps.sh
./.venv/bin/python tools/prepare_qwen35_onnxopt_genai.py --output-dir models/qwen3.5-2b-onnxopt-q4f16 --variant q4f16
./tools/build.sh --test --ort-genai
./.venv/bin/python tools/make_test_image.py
./build/scene_describer --config configs/qwen3.5-2b-onnxopt.ini --image tmp/smoke.png --json
./tools/smoke_ort_genai.sh
```

If executable bits are not preserved, run `chmod +x tools/*.sh`.

## Current Machine Check

Observed in this workspace:

- `python3 --version`: Python 3.12.3
- `g++` is available.
- Default PATH does not expose `cmake` or `ninja`.
- `.venv/bin/cmake` and `.venv/bin/ninja` are installed by `tools/setup_linux_env.sh`.
- `tools/build.sh --test --ort-genai` passed using local Linux ORT GenAI 0.13.1 and ONNX Runtime 1.25.1 packages.

On Windows, ORT GenAI builds need both `onnxruntime-genai.dll` and a compatible `onnxruntime.dll` copied beside the executable.

```powershell
.\tools\build.ps1 -Clean -Test -OrtGenAI `
  -OrtGenAIRoot .deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64 `
  -OrtRuntimeRoot .deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

## Build Commands

```powershell
cmake -S . -B build -DSCENE_DESC_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

With ONNX Runtime GenAI:

```bash
cmake -S . -B build \
  -DSCENE_DESC_ENABLE_ORT_GENAI=ON \
  -DOnnxRuntimeGenAI_ROOT=/path/to/onnxruntime-genai
```
