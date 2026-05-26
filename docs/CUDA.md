# CUDA Runtime Path

This project can fetch and build against the Linux CUDA ONNX Runtime GenAI packages. On the WSL2 RTX 4070 Laptop test machine, CUDA is visible and ORT GenAI can instantiate Qwen models on `device=CUDA`, but token generation currently segfaults inside the ORT GenAI CUDA runtime before a scene description is produced.

## Setup

```bash
./tools/setup_linux_env.sh
./tools/setup_linux_cuda_env.sh
./tools/fetch_runtime_deps.sh --cuda
./tools/build.sh --clean --test --ort-genai \
  --ort-genai-root "$PWD/.deps/onnxruntime-genai-0.13.1-linux-x64-cuda/onnxruntime-genai-0.13.1-linux-x64-cuda" \
  --ort-runtime-root "$PWD/.deps/onnxruntime-linux-x64-gpu-1.25.1/onnxruntime-linux-x64-gpu-1.25.1"
```

`tools/setup_linux_cuda_env.sh` installs CUDA 12 user-space libraries into `.venv` using NVIDIA Python wheels. `tools/cuda_library_path.sh` prints the required `LD_LIBRARY_PATH` entries for those libraries plus the local `build` directory.

## Probe

```bash
export LD_LIBRARY_PATH="$(./tools/cuda_library_path.sh):${LD_LIBRARY_PATH:-}"

./build/scene_model_probe \
  --model-dir models/qwen3.5-2b-onnxopt-q4f16 \
  --image tmp/smoke.png \
  --execution-provider cuda \
  --stage model
```

Expected CUDA load evidence:

```text
ok=model type=qwen3_5 device=CUDA
```

## Current Blocker

The following command reaches CUDA model load and image preprocessing, then segfaults during GenAI token generation in this WSL2 environment:

```bash
./build/scene_model_probe \
  --model-dir models/qwen3.5-2b-onnxopt-q4f16 \
  --image tmp/smoke.png \
  --execution-provider cuda \
  --stage token \
  --max-new-tokens 1
```

Observed evidence:

- `nvidia-smi` reports an NVIDIA GeForce RTX 4070 Laptop GPU.
- C++ ORT GenAI 0.13.1 + ONNX Runtime GPU 1.25.1 loads Qwen3.5 with `device=CUDA`.
- ONNX Runtime GPU 1.26.0 and the Python `onnxruntime-genai-cuda==0.14.0` package show the same generation-time segfault.
- Qwen3.5 q4f16, q4, fp16, and Qwen3-VL int4 packages all fail in the CUDA generation path after successful CUDA model/session creation.

Until this is resolved upstream or with a different model package/runtime build, the production smoke path remains CPU:

```bash
./tools/smoke_ort_genai.sh
```

## Native Linux Debug Runbook

On a native Linux CUDA box, pull the repo and recreate the CUDA path:

```bash
git pull
./tools/setup_linux_env.sh
./tools/setup_linux_cuda_env.sh
./tools/fetch_runtime_deps.sh --cuda
./.venv/bin/python tools/prepare_qwen35_onnxopt_genai.py \
  --output-dir models/qwen3.5-2b-onnxopt-q4f16 \
  --variant q4f16
./tools/build.sh --clean --test --ort-genai \
  --ort-genai-root "$PWD/.deps/onnxruntime-genai-0.13.1-linux-x64-cuda/onnxruntime-genai-0.13.1-linux-x64-cuda" \
  --ort-runtime-root "$PWD/.deps/onnxruntime-linux-x64-gpu-1.25.1/onnxruntime-linux-x64-gpu-1.25.1"
./.venv/bin/python tools/make_test_image.py
```

Run the diagnostic collector:

```bash
./tools/debug_cuda_ort_genai.sh
```

The collector writes logs to `tmp/cuda-debug-*`, including:

- `nvidia-smi.log`
- `ldd-genai-cuda.log`
- `ldd-ort-cuda.log`
- `model-stage.log`
- `token-stage.log`
- `gdb-token-stage.log` when `gdb` is installed
- `dmesg-after.log`

If native Linux succeeds where WSL2 fails, run the full CUDA smoke:

```bash
LD_LIBRARY_PATH="$(./tools/cuda_library_path.sh):${LD_LIBRARY_PATH:-}" \
./build/scene_describer \
  --config configs/qwen3.5-2b-onnxopt-cuda.ini \
  --image tmp/smoke.png \
  --max-new-tokens 32 \
  --json
```

Successful CUDA evidence should include scene JSON plus either `scene_model_probe --stage model` showing `device=CUDA` or a debugger/profiler trace showing CUDA provider execution.
