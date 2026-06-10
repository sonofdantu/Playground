# AI CUDA Reproduction Guide

This is the shortest complete path for a fresh Linux clone to reproduce the verified Qwen3.5-2B C++ CUDA scene-description runtime.

Use this file when an AI agent or a human needs to recreate the exact working environment without relying on any generated local files.

## What Success Means

A successful run proves all of these are true:

- `scene_describer` is a compiled C++ Linux executable.
- Qwen3.5 model loading reaches `device=CUDA`.
- The runtime executes the Qwen3.5 ONNX graphs through raw ONNX Runtime CUDA sessions.
- The output JSON contains `metadata.execution_provider=raw-ort-cuda`.
- The generated text describes `tmp/smoke.png`, normally mentioning a blue truck, red house, yellow sun, or green ground.
- The analyzer path can ingest 30 generated 1920x1080 security frames in one request and return a CUDA-generated video summary.
- The analyzer path can add high-resolution detail crops so small held objects and small airborne objects are visible to the model.
- The resident C++ benchmark can summarize that 30-frame JPEG quality-85 window with warmed median latency below 4 seconds on RTX4000-class hardware.

Known-good sentence on this machine:

```text
A blue truck drives past a red house under a yellow sun.
```

## Start From Fresh Clone

```bash
git clone https://github.com/sonofdantu/Playground.git
cd Playground
```

If the repo already exists:

```bash
cd ~/code/Playground
git pull --ff-only
```

## Required Machine State

Before setup, verify the machine has NVIDIA GPU visibility:

```bash
nvidia-smi || /usr/lib/wsl/lib/nvidia-smi
```

Expected: the command prints an NVIDIA GPU and driver. The target gate is RTX4000-class hardware or better. The verified WSL2 development machine has an RTX 4070 Laptop GPU and is slower than the RTX4000 target.

## One-Shot Reproduction

Run these commands in order from the repository root:

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
./tools/smoke_ort_genai.sh \
  --execution-provider cuda \
  --config configs/qwen3.5-2b-onnxopt-cuda.ini \
  --max-new-tokens 32
./tools/smoke_cuda_frame_batch.sh
./tools/benchmark_cuda_frame_batch.sh --analyzer-prompt
```

## Expected Smoke Output

The smoke command must print this probe evidence:

```text
ok=model type=qwen3_5 device=CUDA
```

It must also print JSON shaped like this:

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

The exact sentence may vary slightly. The required checks are:

```text
ok=model type=qwen3_5 device=CUDA
"execution_provider": "raw-ort-cuda"
"model_type": "qwen3_5"
```

## 1080p Security Frame Batch

The video-frame gate generates 30 synthetic 1920x1080 JPEG frames at quality 85, attaches sampled person/vehicle track metadata, adds two high-resolution JPEG quality-85 detail crops, and feeds them into one C++ analyzer request:

```bash
./tools/smoke_cuda_frame_batch.sh
```

Required evidence:

```text
frame batch smoke passed: frames=30
```

The JSON metadata must include:

```json
{
  "metadata": {
    "execution_provider": "raw-ort-cuda",
    "frame_count": "30",
    "detail_image_count": "2",
    "image_count": "32",
    "model_type": "qwen3_5",
    "prefill_chunk_tokens": "512"
  }
}
```

Observed 30-frame output on the verified RTX 4070 Laptop WSL2 machine:

```text
The scene depicts a static suburban setting with a red house, a garage, and a blue pickup truck parked on the left. A person wearing a yellow shirt and dark pants walks from left to right across the asphalt road. The person carries a small, rectangular object in their right hand. A black drone hovers in the upper right portion of
```

The warmed resident-runtime latency gate is:

```bash
./tools/benchmark_cuda_frame_batch.sh --analyzer-prompt
```

Required evidence on RTX4000-class hardware:

```text
CUDA frame benchmark passed: frames=30 median_ms=<4000
```

On the verified RTX 4070 Laptop WSL2 development machine, run the same benchmark with `--no-target`; observed JPEG-85 medians range from `5162.286` to `7468.595` ms depending on laptop/WSL state.

## Direct C++ Command

After the one-shot path succeeds, this is the direct runtime command:

```bash
LD_LIBRARY_PATH="$(./tools/cuda_library_path.sh):$PWD/build:${LD_LIBRARY_PATH:-}" \
./build/scene_describer \
  --config configs/qwen3.5-2b-onnxopt-cuda.ini \
  --image tmp/smoke.png \
  --max-new-tokens 32 \
  --json
```

## Proof That It Is C++ And CUDA

C++ executable proof:

```bash
file build/scene_describer
ldd build/scene_describer | grep -E 'onnxruntime|libstdc\+\+'
```

Expected evidence:

```text
ELF 64-bit ... executable ... x86-64
libonnxruntime-genai.so
libonnxruntime.so
libstdc++.so.6
```

CUDA model proof:

```bash
LD_LIBRARY_PATH="$(./tools/cuda_library_path.sh):$PWD/build:${LD_LIBRARY_PATH:-}" \
./build/scene_model_probe \
  --model-dir models/qwen3.5-2b-onnxopt-q4f16 \
  --image tmp/smoke.png \
  --execution-provider cuda \
  --stage model
```

Expected evidence:

```text
ok=model type=qwen3_5 device=CUDA
```

GPU process proof while generation is running:

```bash
(
  LD_LIBRARY_PATH="$(./tools/cuda_library_path.sh):$PWD/build:${LD_LIBRARY_PATH:-}" \
    ./build/scene_describer \
      --config configs/qwen3.5-2b-onnxopt-cuda.ini \
      --image tmp/smoke.png \
      --max-new-tokens 64 \
      --json > tmp/cuda_cpp_confirm.json
) &
run_pid=$!
for _ in $(seq 1 30); do
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || \
    /usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
  kill -0 "$run_pid" 2>/dev/null || break
  sleep 0.25
done
wait "$run_pid"
cat tmp/cuda_cpp_confirm.json
```

On WSL2, `nvidia-smi` may show the process name as `[Not Found]`, but a process row during generation still confirms GPU compute visibility.

## Important Implementation Detail

Do not use `scene_model_probe --stage token --execution-provider cuda` as the pass/fail CUDA test. That path exercises the legacy ORT GenAI CUDA generator and is known to crash after `stage=generator` in this environment.

The stable CUDA path is:

```text
scene_describer --config configs/qwen3.5-2b-onnxopt-cuda.ini
```

For Qwen3.5 with `execution_provider=cuda`, `scene_describer` routes into `src/backends/qwen35_raw_ort_cuda.cpp`, which runs:

- `vision_encoder_*` through ONNX Runtime CUDA
- `embed_tokens_*` through ONNX Runtime CUDA
- `decoder_model_merged_*` through ONNX Runtime CUDA

ORT GenAI is still used for Qwen-compatible image preprocessing, prompt tokenization, and token decoding.

For frame-batch requests, the raw CUDA path uses persistent CUDA sessions and chunked prefill (`prefill_chunk_tokens=512`) so the decoder does not prefill a long video prompt in one giant sequence. The default model-preparation profile is `--processor-profile security`, which uses a 224px wide-frame budget plus high-resolution detail crops for 30-frame 1920x1080 security summaries.

Measured 8GB RTX 4070 Laptop tradeoffs:

- `security`/224px plus two JPEG quality-85 detail crops, 30 JPEG quality-85 frames, analyzer prompt, 63 generated tokens: median `5162.286`-`7468.595` ms depending on laptop/WSL state.
- 336px plus two detail crops, 30 frames, analyzer prompt, 64 generated tokens: median `7124.846` ms.
- `video`/224px without detail crops, 60 frames, concise prompt, 24 generated tokens: median `4100.184` ms.
- `detail`/448px wide-only, 30 frames, concise prompt, 24 generated tokens: median `16636.760` ms.
- 336px wide-frame detail, 60 frames: failed on the 8GB GPU with ORT arena allocation pressure.

Do not assume tiny objects will be reliable from the compressed whole-frame view. The working security path preserves small-object evidence by adding high-resolution detail crops. The target gate is 30 JPEG quality-85 1920x1080 frames in under 4 seconds on RTX4000-class hardware. If 448px wide-frame detail or 60+ higher-detail frames must meet that cadence, use a materially faster desktop GPU and test with more than 8GB VRAM; 16GB+ VRAM is the practical next tier. A6000 should be faster than RTX4000, and Thor-class targets should be materially faster.

## If It Fails

Run the debug collector:

```bash
./tools/debug_cuda_ort_genai.sh
```

Open the generated directory:

```bash
ls tmp/cuda-debug-*
cat tmp/cuda-debug-*/summary.txt
cat tmp/cuda-debug-*/raw-ort-cuda-cli.log
```

The expected successful summary is:

```text
raw_ort_cuda_cli_exit=0
```

The legacy token-stage may still show:

```text
token_stage_exit=139
```

That is not a failure of the stable CUDA path as long as `raw_ort_cuda_cli_exit=0`.

## Generated Files Policy

Do not commit generated files. These are intentionally recreated locally:

- `.venv`
- `.deps`
- `.cache`
- `build`
- `tmp`
- `models/qwen3.5-2b-onnxopt-q4f16`

The reproducible source of truth is this guide plus the checked-in scripts and source files.
