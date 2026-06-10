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
  --variant q4f16 \
  --processor-profile security
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

## 1080p Security Frame Batch

Generate synthetic 1920x1080 security frames, attach sampled person/vehicle tracks, append high-resolution detail crops, and summarize them in one C++/CUDA analyzer request:

```bash
./tools/smoke_cuda_frame_batch.sh
```

Required evidence:

```text
frame batch smoke passed: frames=30
```

```json
{
  "summary": "The scene depicts a static suburban setting with a red house, a garage, and a blue pickup truck parked on the left. A person wearing a yellow shirt and dark pants walks from left to right across the asphalt road. The person carries a small, rectangular object in their right hand. A black drone hovers in the upper right portion of",
  "metadata": {
    "detail_image_count": "2",
    "execution_provider": "raw-ort-cuda",
    "frame_count": "30",
    "image_count": "32",
    "model_type": "qwen3_5",
    "prefill_chunk_tokens": "512"
  }
}
```

Measure the resident-runtime latency gate:

```bash
./tools/benchmark_cuda_frame_batch.sh --analyzer-prompt
```

Target warmed evidence on RTX4000-class hardware:

```text
CUDA frame benchmark passed: frames=30 median_ms=<4000
```

The benchmark constructs the C++ backend once, runs one warmup request, then times repeated 30-frame summaries. This excludes one-time model/session startup and matches the intended resident video-service behavior.

Processor profiles:

- `security` is the default 224px wide-frame profile paired with high-resolution detail crops for 30-frame 1920x1080 JPEG quality-85 summaries and the 4-second RTX4000 target.
- `video` is the same 224px profile when running longer low-detail frame batches without detail crops.
- `detail` is the 448px wide-frame profile for quality testing; it is not currently a 4-second profile on the tested laptop GPU.
- `image` is the legacy high-detail single-image profile.

Measured tradeoffs on the verified 8GB RTX 4070 Laptop GPU:

- 30 JPEG quality-85 frames, `security`/224px plus two JPEG quality-85 detail crops, analyzer prompt, 63 generated tokens: median `5162.286`-`7468.595` ms with `--no-target` depending on laptop/WSL state.
- 30 frames, 336px plus two detail crops, analyzer prompt, 64 generated tokens: median `7124.846` ms.
- 60 frames, 224px without detail crops, concise prompt, 24 generated tokens: median `4100.184` ms.
- 30 frames, 448px wide-only, concise prompt, 24 generated tokens: median `16636.760` ms.
- 60 frames at 336px wide-frame detail: failed on the 8GB GPU with ORT arena allocation pressure.

The critical accuracy point is that small objects are not expected to survive the compressed whole-frame view. They need either detector/track metadata or high-resolution detail crops. For 448px wide-frame detail at a 4-second cadence or 60 frames above the 224px profile, expect a materially faster desktop GPU and more than 8GB VRAM; 16GB+ VRAM is the next practical test tier. A6000 should be faster than RTX4000, and Thor-class targets should be materially faster.

## Why Raw ORT CUDA Exists

The original CUDA attempt used ORT GenAI's `OgaGenerator` directly. It reached CUDA model load and image preprocessing, but crashed before token generation completed.

Raw ONNX Runtime debugging found a concrete CUDA kernel limitation in the decoder graph:

```text
GroupQueryAttention: attention_bias is not supported in GroupQueryAttention cuda kernel
```

For this project's single-image, no-padding prompt, the exported `GroupQueryAttention` bias is all zeros. `tools/prepare_qwen35_onnxopt_genai.py` now removes that optional bias input from the generated local model package so the CUDA kernel can run.

The direct ORT GenAI CUDA generator path still crashes in this environment after `stage=generator`; do not use it as the production CUDA path. Use `scene_describer` with `execution_provider=cuda`, which routes Qwen3.5 through the raw ONNX Runtime CUDA loop.

For long prompts, the raw CUDA loop splits prefill into 512-token chunks and feeds only the image-feature rows needed by each chunk. This is required for 30-60 frame requests because all-at-once prefill can saturate an 8GB GPU.

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
