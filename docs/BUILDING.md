# Building

## Required Tools

- CMake 3.24 or newer.
- C++20 compiler.
- Optional: ONNX Runtime GenAI v0.13.1 or newer for Qwen3.5/Qwen3-VL work.

## Linux Quick Build

On Linux x64, use the shell wrappers:

```bash
./tools/fetch_runtime_deps.sh
./tools/setup_export_env.sh
./.venv/bin/python tools/prepare_qwen35_onnxopt_genai.py --output-dir models/qwen3.5-2b-onnxopt-q4f16 --variant q4f16
./tools/build.sh --test --ort-genai
./.venv/bin/python tools/make_test_image.py
LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" ./build/scene_describer --config configs/qwen3.5-2b-onnxopt.ini --image tmp/smoke.png --json
```

If executable bits are not preserved, run `chmod +x tools/*.sh`.

## Current Machine Check

Observed in this workspace:

- `python --version`: Python 3.10.11
- Default PATH does not expose `cmake`, `cl`, `g++`, `clang++`, or `ninja`.
- Visual Studio Build Tools are installed under `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`.
- MSVC, CMake, and Ninja work after initializing `VC\Auxiliary\Build\vcvars64.bat`.

Use the wrapper:

```powershell
.\tools\build.ps1 -Test
```

ORT GenAI builds need both `onnxruntime-genai.dll` and a compatible `onnxruntime.dll` copied beside the executable. This matters on this machine because `C:\Windows\System32\onnxruntime.dll` is version 1.17.1 and is too old for ORT GenAI 0.13.1.

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

```powershell
cmake -S . -B build `
  -DSCENE_DESC_ENABLE_ORT_GENAI=ON `
  -DOnnxRuntimeGenAI_ROOT=C:\path\to\onnxruntime-genai
```

The ONNX backend currently has a placeholder implementation. Enabling the library link is the next implementation phase, not the end state.
