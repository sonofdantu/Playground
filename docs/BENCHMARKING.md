# Benchmarking

There are two benchmark paths:

- `tools\benchmark_cli.py` measures the complete CLI path: process start, model load, image preprocessing, generation, and JSON output.
- `scene_describer_benchmark` constructs the backend once, reports model-load time separately, then measures repeated in-process `Describe` calls against the same model instance.

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
