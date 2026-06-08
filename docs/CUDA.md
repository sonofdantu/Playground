# CUDA Runtime Path

This repository now has a verified Linux/WSL2 CUDA path for Qwen3.5-2B scene description.

If you are starting from a fresh clone, use `docs/AI_CUDA_REPRODUCTION.md` first. It is the most explicit copy/paste guide and includes C++ proof, CUDA proof, expected output, and failure triage.

The stable CUDA route is a hybrid C++ runtime:

- ORT GenAI loads the model package on CPU only for Qwen-compatible image preprocessing, prompt tokenization, and token decoding.
- Raw ONNX Runtime C++ sessions run `vision_encoder_*`, `embed_tokens_*`, and `decoder_model_merged_*` with `CUDAExecutionProvider`.
- The manual generation loop owns Qwen3.5 recurrent state tensors, uses chunked prefill for long frame batches, and uses greedy decoding.

This avoids the current ORT GenAI CUDA generator crash while keeping model execution in ONNX Runtime CUDA.

## Setup

```bash
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

`tools/setup_linux_cuda_env.sh` installs CUDA user-space libraries into `.venv` using NVIDIA Python wheels. `tools/cuda_library_path.sh` prints the required `LD_LIBRARY_PATH` entries for those libraries plus the local `build` directory.

## Smoke Test

```bash
./tools/smoke_ort_genai.sh \
  --execution-provider cuda \
  --config configs/qwen3.5-2b-onnxopt-cuda.ini \
  --max-new-tokens 32
```

Verified output on this machine:

```text
ok=model type=qwen3_5 device=CUDA
```

```json
{
  "text": "A blue truck drives past a red house under a yellow sun.",
  "backend": "ort-genai",
  "metadata": {
    "execution_provider": "raw-ort-cuda",
    "model_type": "qwen3_5"
  }
}
```

Direct command:

```bash
LD_LIBRARY_PATH="$(./tools/cuda_library_path.sh):$PWD/build:${LD_LIBRARY_PATH:-}" \
./build/scene_describer \
  --config configs/qwen3.5-2b-onnxopt-cuda.ini \
  --image tmp/smoke.png \
  --max-new-tokens 32 \
  --json
```

## 30-120 Frame Video Smoke

Generate synthetic frames and summarize them in one C++/CUDA analyzer request:

```bash
./tools/smoke_cuda_frame_batch.sh --frame-count 30 --max-new-tokens 32
./tools/smoke_cuda_frame_batch.sh --frame-count 120 --max-new-tokens 48
```

Verified 120-frame evidence on this machine:

```text
frame batch smoke passed: frames=120
```

```json
{
  "summary": "The surveillance scene shows a static, pixelated environment featuring a red house with a dark roof, a blue pickup truck positioned on a gray road, and a yellow sun in the upper right corner.",
  "metadata": {
    "execution_provider": "raw-ort-cuda",
    "frame_count": "120",
    "image_count": "120",
    "model_type": "qwen3_5",
    "prefill_chunk_tokens": "512"
  }
}
```

The default Qwen3.5 preparation path uses `--processor-profile video`, which creates a 224px processor profile. That keeps 120-frame requests inside the tested 8GB RTX 4070 Laptop GPU envelope. Use `--processor-profile image` only when high-detail single-image inference matters more than long frame batches.

## Why Raw ORT CUDA Exists

The original CUDA attempt used ORT GenAI's `OgaGenerator` directly. It reached CUDA model load and image preprocessing, but crashed before token generation completed.

Raw ONNX Runtime debugging found a concrete CUDA kernel limitation in the decoder graph:

```text
GroupQueryAttention: attention_bias is not supported in GroupQueryAttention cuda kernel
```

For this project's single-image, no-padding prompt, the exported `GroupQueryAttention` bias is all zeros. `tools/prepare_qwen35_onnxopt_genai.py` now removes that optional bias input from the generated local model package so the CUDA kernel can run.

The direct ORT GenAI CUDA generator path still crashes in this environment after `stage=generator`; do not use it as the production CUDA path. Use `scene_describer` with `execution_provider=cuda`, which routes Qwen3.5 through the raw ONNX Runtime CUDA loop.

For long prompts, the raw CUDA loop splits prefill into 512-token chunks and feeds only the image-feature rows needed by each chunk. This is required for 30-120 frame requests because all-at-once prefill can saturate an 8GB GPU.

## Debug Runbook

Collect CUDA diagnostics:

```bash
./tools/debug_cuda_ort_genai.sh
```

The collector writes logs to `tmp/cuda-debug-*`, including:

- `nvidia-smi.log`
- `ldd-genai-cuda.log`
- `ldd-ort-cuda.log`
- `model-stage.log`
- `token-stage.log` for the known legacy GenAI CUDA generator failure
- `raw-ort-cuda-cli.log` for the stable CUDA CLI path
- `dmesg-after.log`

A successful run exits with `raw_ort_cuda_cli_exit=0` in `summary.txt`.
