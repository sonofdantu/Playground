# Edge Roadmap

The working runtime is now a baseline, not the end state. The target is a commercially usable edge package with low steady-state latency.

## Track 1: Keep The Harness Working

- Preserve `configs\qwen3-vl-2b.ini` as the known-good VLM smoke path.
- Keep `tools\smoke_ort_genai.ps1` passing after runtime changes.
- Keep `scene_describer_benchmark` as the latency gate for every candidate model and execution provider.
- Keep `scene_analyzer` passing as the prompt/history/track harness.

## Track 2: Qwen3.5 Runtime Package

Goal: make `Qwen/Qwen3.5-2B` produce a complete ORT GenAI multimodal package.

Required output shape:

- decoder ONNX graph
- embedding ONNX graph
- vision ONNX graph
- processor config
- tokenizer files
- `genai_config.json` with `model.decoder`, `model.embedding`, and `model.vision`
- `MODEL_PROVENANCE.json` with local hashes and Apache-2.0 license signal

Current state: `tools/prepare_qwen35_onnxopt_genai.py` assembles the complete experimental Qwen3.5 ONNX-OPT package from public decoder, embedding, and vision graphs.

## Track 3: Speed

CPU is the baseline and raw ONNX Runtime CUDA is now the verified Linux GPU path:

- model load: about 4.61 seconds
- steady in-process request: about 3.25 seconds for 39 generated tokens
- throughput: about 12 generated tokens/sec

Next measurements should be collected on the target hardware with:

- CPU INT4 baseline
- raw ONNX Runtime CUDA on NVIDIA hardware
- DirectML on Windows GPUs, if the deployment target has supported hardware
- QNN, if the target is Qualcomm/NPU-class hardware

Keep prompts short and cap `max_new_tokens` aggressively for edge use. The current scene-description task does not need long generation by default.

## Track 4: Product Integration

The CLI is useful for validation. A production app will likely want one of:

- a long-running local service that keeps the model loaded
- a C ABI wrapper for integration into non-C++ callers
- a C++ library API consumed directly by the host application

The in-process benchmark already measures the performance profile expected from a long-running service.
