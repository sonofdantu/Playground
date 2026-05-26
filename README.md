# Edge Scene Describer

This repository is a C++ foundation for edge scene description with ONNX Runtime GenAI-ready backends.

The intended production target is a commercially usable Qwen VLM package, starting with:

- `Qwen/Qwen3.5-2B` where ORT GenAI support and export quality are sufficient.
- `Qwen/Qwen3-VL-2B-Instruct` as the conservative 2B VLM fallback.

Both candidates are tracked as Apache-2.0 in the project license notes. Final model use still needs legal review before shipping.

## Current State

The codebase currently contains:

- C++ core interfaces for scene description.
- CLI argument parsing and config loading.
- A deterministic mock backend for local smoke tests.
- A bootstrap PPM/PGM image loader.
- ONNX Runtime GenAI backend compiled behind `SCENE_DESC_ENABLE_ORT_GENAI`.
- Analyzer layer for frames, tracks, prompt templates, and history context.
- Scripted dependency fetch, export setup, and model export workflows.
- CLI-inclusive and in-process benchmark tooling.
- Runtime packaging script for local edge-device bundles.

For a fresh clone, start with [docs\QUICKSTART.md](docs/QUICKSTART.md).
For the generated runtime assets that are intentionally not committed, read [docs\RUNTIME_ASSETS.md](docs/RUNTIME_ASSETS.md).
Read [PROJECT_STATE.md](PROJECT_STATE.md) first after any context reset.
For agent-oriented Linux reproduction, read [AGENTS.md](AGENTS.md) and [docs/LINUX_REPRODUCTION.md](docs/LINUX_REPRODUCTION.md).
For the current CUDA status and commands, read [docs/CUDA.md](docs/CUDA.md).
For model readiness and edge-performance direction, read [docs\MODEL_READINESS.md](docs/MODEL_READINESS.md) and [docs\EDGE_ROADMAP.md](docs/EDGE_ROADMAP.md).
For the analyzer surface, read [docs\ANALYZER.md](docs/ANALYZER.md).

## Planned CLI

```powershell
scene_describer --config configs/qwen3.5-2b-onnxopt.ini --image sample.png
```

For now, the mock path is:

```powershell
python tools/make_test_ppm.py
scene_describer --config configs/mock.local.ini --image tmp/smoke.ppm
```

## Build

Linux quick path:

```bash
./tools/setup_linux_env.sh
./tools/fetch_runtime_deps.sh
./.venv/bin/python tools/prepare_qwen35_onnxopt_genai.py --output-dir models/qwen3.5-2b-onnxopt-q4f16 --variant q4f16
./tools/build.sh --test --ort-genai
./.venv/bin/python tools/make_test_image.py
./build/scene_describer --config configs/qwen3.5-2b-onnxopt.ini --image tmp/smoke.png --max-new-tokens 32 --json
./tools/smoke_ort_genai.sh
```

See [docs\QUICKSTART.md](docs/QUICKSTART.md) for full Windows and Linux startup paths.

The local Visual Studio Build Tools install includes CMake, Ninja, and MSVC. The easiest path on this machine is:

```powershell
.\tools\build.ps1 -Test
```

For the ORT GenAI backend after local dependency packages are present:

```powershell
.\tools\build.ps1 -Clean -Test -OrtGenAI `
  -OrtGenAIRoot .deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64 `
  -OrtRuntimeRoot .deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

CMake remains the canonical build system:

```powershell
cmake -S . -B build -DSCENE_DESC_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

On Linux, `tools/setup_linux_env.sh` installs local CMake/Ninja and the lightweight Python packages needed for Qwen3.5 ONNX-OPT preparation. See [docs/BUILDING.md](docs/BUILDING.md).

## Model Export

The export path is documented in [docs/MODEL_EXPORT.md](docs/MODEL_EXPORT.md). The short version is:

```powershell
.\tools\setup_export_env.ps1
.\tools\export_model.ps1 -ModelName Qwen/Qwen3.5-2B -OutputDir models\qwen3.5-2b -Precision int4 -ExecutionProvider cpu
```

The direct ORT GenAI source export for Qwen3.5 produces a decoder-only package, so it is not enough for scene description by itself. The working local Qwen3.5 route is the experimental ONNX-OPT preparation path documented in [docs/MODEL_EXPORT.md](docs/MODEL_EXPORT.md):

```powershell
.\.venv\Scripts\python.exe tools\prepare_qwen35_onnxopt_genai.py --output-dir models\qwen3.5-2b-onnxopt-q4f16 --variant q4f16
.\build\scene_describer.exe --config configs\qwen3.5-2b-onnxopt.ini --image tmp\smoke.png --max-new-tokens 32 --json
```

For the conservative Qwen3-VL engineering smoke test:

```powershell
.\.venv\Scripts\python.exe tools\make_test_image.py
.\build\scene_describer.exe --config configs\qwen3-vl-2b.ini --image tmp\smoke.png --max-new-tokens 48 --json
```

Or run the optional local smoke gate:

```powershell
.\tools\smoke_ort_genai.ps1
```

Linux:

```bash
./tools/smoke_ort_genai.sh
```

Benchmark the CLI-inclusive CPU path:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_cli.py --exe build\scene_describer.exe --config configs\qwen3-vl-2b.ini --image tmp\smoke.png --max-new-tokens 48
```

Benchmark the in-process path after an ORT-enabled build:

```powershell
.\build\scene_describer_benchmark.exe --config configs\qwen3-vl-2b.ini --image tmp\smoke.png --max-new-tokens 48 --warmups 1 --repeats 3 --json
```

For the edge-oriented smoke profile, use:

```powershell
.\build\scene_describer_benchmark.exe --config configs\qwen3-vl-2b.fast.ini --image tmp\smoke.png --warmups 1 --repeats 3 --json
```

Run the analyzer-shaped Qwen path:

```powershell
.\build\scene_analyzer.exe --config configs\qwen3.5-2b-onnxopt.ini --image tmp\smoke.png --timestamp-ms 1000 --request-id smoke-analyzer --track "0,t1,train,78,85,94,38,0.91" --json
```

Create a local runtime package without copying model weights:

```powershell
.\tools\package_runtime.ps1 -ConfigPath configs\qwen3-vl-2b.ini -ModelDir models\qwen3-vl-2b-instruct -Force
```
