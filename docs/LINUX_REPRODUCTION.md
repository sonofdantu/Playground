# Linux Reproduction

This is the shortest tested path for rebuilding the Linux Qwen3.5 ONNX Runtime GenAI environment from a fresh clone. It is written for future agents and humans who need to recreate the working state without relying on local generated files.

## Goal

Produce a C++ runtime that can describe `tmp/smoke.png` through the real ORT GenAI backend:

- backend: `ort-genai`
- model type: `qwen3_5`
- model package: `models/qwen3.5-2b-onnxopt-q4f16`

## Fresh Linux Setup

Prerequisites:

- Linux x64.
- Python 3.
- A C++20 compiler such as `g++` or `clang++`.
- Network access to GitHub and Hugging Face.
- Several GB of free disk space for `.deps`, `.cache`, and `models`.

Commands:

```bash
./tools/setup_linux_env.sh
./tools/fetch_runtime_deps.sh
./.venv/bin/python tools/prepare_qwen35_onnxopt_genai.py \
  --output-dir models/qwen3.5-2b-onnxopt-q4f16 \
  --variant q4f16
./tools/build.sh --test --ort-genai
./.venv/bin/python tools/make_test_image.py
./tools/smoke_ort_genai.sh
```

`tools/setup_linux_env.sh` creates `.venv` and installs local `cmake`, `ninja`, `huggingface_hub`, `onnx`, and `pillow`. `tools/build.sh` automatically uses `.venv/bin/cmake`, `.venv/bin/ctest`, and `.venv/bin/ninja` when those tools are not on PATH.

## Runtime Proof

Run the normal CLI:

```bash
./build/scene_describer \
  --config configs/qwen3.5-2b-onnxopt.ini \
  --image tmp/smoke.png \
  --max-new-tokens 32 \
  --json
```

Known-good output shape:

```json
{
  "text": "A blue car drives past a red house on a green hill under a yellow sun.",
  "backend": "ort-genai",
  "image": {
    "path": "tmp/smoke.png"
  },
  "metadata": {
    "backend": "ort-genai",
    "model_dir": "models/qwen3.5-2b-onnxopt-q4f16",
    "model_type": "qwen3_5"
  }
}
```

The exact sentence can vary because the config uses sampling. The important checks are `backend=ort-genai`, `model_type=qwen3_5`, and a plausible description of the generated smoke image.

## Analyzer Proof

```bash
./build/scene_analyzer \
  --config configs/qwen3.5-2b-onnxopt.ini \
  --image tmp/smoke.png \
  --timestamp-ms 1000 \
  --request-id smoke-analyzer \
  --track "0,t1,vehicle,78,85,94,38,0.91" \
  --max-new-tokens 16 \
  --json
```

Expected evidence includes `metadata.backend=ort-genai`, `metadata.model_type=qwen3_5`, and `metadata.track_count=1`.

## Generated Files

These are intentionally ignored and must be recreated locally:

- `.venv`
- `.deps`
- `.cache`
- `build`
- `tmp`
- `models/qwen3.5-2b-onnxopt-q4f16`

Do not commit ONNX weights, external data shards, downloaded runtime libraries, build outputs, Python environments, or Hugging Face cache files.

## Troubleshooting

- If `cmake` is missing, rerun `./tools/setup_linux_env.sh`; the build script will use `.venv/bin/cmake`.
- If ORT headers or shared libraries are missing, rerun `./tools/fetch_runtime_deps.sh`.
- If the model directory is missing or incomplete, rerun `prepare_qwen35_onnxopt_genai.py`; unauthenticated Hugging Face downloads can be rate-limited.
- If `backend=ort-genai` is unavailable, rebuild with `./tools/build.sh --test --ort-genai`.
