# Agent Handoff

This repository is a C++ ONNX Runtime GenAI playground for running a Qwen3.5-2B vision-language runtime on Linux and Windows.

Start here after a fresh clone or context reset:

1. Read `PROJECT_STATE.md`.
2. Read `docs/LINUX_REPRODUCTION.md` for the tested Linux Qwen3.5 path.
3. Read `docs/RUNTIME_ASSETS.md` before assuming generated files exist.
4. Read `docs/CUDA.md` before attempting GPU/CUDA work.
5. For the easiest fresh-clone GPU reproduction, read `docs/AI_CUDA_REPRODUCTION.md` and follow it exactly.

Do not commit generated runtime assets:

- `.venv`
- `.deps`
- `.cache`
- `build`
- `dist`
- `tmp`
- `models/*` except `models/README.md`

The Linux proof path is:

```bash
./tools/setup_linux_env.sh
./tools/fetch_runtime_deps.sh
./.venv/bin/python tools/prepare_qwen35_onnxopt_genai.py --output-dir models/qwen3.5-2b-onnxopt-q4f16 --variant q4f16
./tools/build.sh --test --ort-genai
./.venv/bin/python tools/make_test_image.py
./tools/smoke_ort_genai.sh
```

Expected smoke evidence includes `backend=ort-genai`, `model_type=qwen3_5`, and a description of `tmp/smoke.png` mentioning the red house, blue vehicle, green ground, or yellow sun.

CUDA status: Qwen3.5 scene description now works on WSL2 through the C++ raw ONNX Runtime CUDA loop. `tools/smoke_ort_genai.sh --execution-provider cuda --config configs/qwen3.5-2b-onnxopt-cuda.ini` must return scene JSON with `metadata.execution_provider=raw-ort-cuda`.

Frame-batch status: `tools/smoke_cuda_frame_batch.sh --frame-count 120 --max-new-tokens 48` must return analyzer JSON with `metadata.execution_provider=raw-ort-cuda`, `metadata.frame_count=120`, and `metadata.prefill_chunk_tokens=512`.

The direct ORT GenAI CUDA generator path still crashes after `stage=generator`; do not regress the stable raw-ORT CUDA path while investigating that legacy failure. For CUDA debugging on another machine, run `tools/debug_cuda_ort_genai.sh` and inspect the generated `tmp/cuda-debug-*` logs before changing model/runtime code.

When another AI needs to reproduce this on a new Linux box, send it to `docs/AI_CUDA_REPRODUCTION.md` first. That file contains the exact setup commands, expected output, C++ proof, CUDA proof, and failure triage.
