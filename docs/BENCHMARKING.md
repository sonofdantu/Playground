# Benchmarking

There are two benchmark paths:

- `tools\benchmark_cli.py` measures the complete CLI path: process start, model load, image preprocessing, generation, and JSON output.
- `scene_describer_benchmark` constructs the backend once, reports model-load time separately, then measures repeated in-process `Describe` calls against the same model instance.
- `tools/benchmark_cuda_frame_batch.sh` generates 1920x1080 frame batches and wraps `scene_describer_benchmark` for the Linux raw-ORT CUDA path.

## CLI-Inclusive CPU Baseline

Command:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_cli.py `
  --exe build\scene_describer.exe `
  --config configs\qwen3-vl-2b.ini `
  --image tmp\smoke.png `
  --max-new-tokens 48 `
  --warmups 1 `
  --repeats 3 `
  --save tmp\benchmark-qwen3vl-cpu.json
```

Observed on 2026-05-21 with the smoke Qwen3-VL CPU package:

- Median latency: about 7.32 seconds.
- Mean latency: about 7.30 seconds.
- Generated tokens: 39.
- Mean generated-token throughput, inclusive: about 5.35 tokens/sec.

This is the right baseline for command-line tooling and batch wrappers, but it includes process startup and model loading on every invocation.

## In-Process CPU Baseline

Build the ORT-enabled target first:

```powershell
.\tools\build.ps1 -Test -OrtGenAI `
  -OrtGenAIRoot .deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64 `
  -OrtRuntimeRoot .deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1
```

Command:

```powershell
.\build\scene_describer_benchmark.exe `
  --config configs\qwen3-vl-2b.ini `
  --image tmp\smoke.png `
  --max-new-tokens 48 `
  --warmups 1 `
  --repeats 3 `
  --json
```

Observed on 2026-05-21 with the smoke Qwen3-VL CPU package:

- Model load: about 4.61 seconds.
- Median in-process request latency: about 3.25 seconds.
- Mean in-process request latency: about 3.24 seconds.
- Generated tokens: 39.
- Mean generated-token throughput: about 12.03 tokens/sec.

This is the better proxy for a long-running edge service because the model instance is reused.

## In-Process CUDA Frame Batch

Prepare the Qwen3.5 ONNX-OPT package with the default security profile and build with CUDA as described in `docs/AI_CUDA_REPRODUCTION.md`, then run:

```bash
./tools/benchmark_cuda_frame_batch.sh --analyzer-prompt
```

Target gate for RTX4000-class hardware:

- Source frames: 30 generated 1920x1080 JPEG quality-85 frames plus two high-resolution JPEG quality-85 detail crops.
- Processor profile: `security`, 224px wide-frame budget with detail crops for small objects.
- Prompt: full analyzer frame-batch prompt with sampled person/vehicle tracks and detail-crop notes.
- Warmed median request latency: below `4000` ms.
- Metadata proof: `execution_provider=raw-ort-cuda`, `model_type=qwen3_5`, `image_count=32`.
- Accuracy proof: the smoke summary mentions a small rectangular object carried by the person and a black drone in the sky.

Observed on the verified RTX 4070 Laptop WSL2 development machine with `--no-target`: warmed medians range from `5162.286` to `7468.595` ms depending on laptop/WSL state, with `generated_tokens=63` and `input_tokens=3995`. This laptop is below the RTX4000 target class, so the default 4-second gate is expected to fail there unless `--no-target` is supplied.

This is the current resident security-video gate. It excludes one-time process/model startup and is the right proxy for a service that emits a fresh summary every 4 seconds on RTX4000-class hardware.

For fast edge smoke tests, use `configs\qwen3-vl-2b.fast.ini`. It keeps the prompt short and caps generation at 48 tokens by default:

```powershell
.\build\scene_describer_benchmark.exe `
  --config configs\qwen3-vl-2b.fast.ini `
  --image tmp\smoke.png `
  --warmups 1 `
  --repeats 3 `
  --json
```

Observed fast-profile CPU result on 2026-05-21:

- Model load: about 4.16 seconds.
- Median in-process request latency: about 2.57 seconds.
- Generated tokens: 31.
- Mean generated-token throughput: about 12.06 tokens/sec.
