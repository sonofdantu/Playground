# Agent Handoff

This repository is a C++ ONNX Runtime GenAI playground for running a Qwen3.5-2B vision-language runtime on Linux and Windows.

Start here after a fresh clone or context reset:

1. Read `PROJECT_STATE.md`.
2. Read `docs/LINUX_REPRODUCTION.md` for the tested Linux Qwen3.5 path.
3. Read `docs/RUNTIME_ASSETS.md` before assuming generated files exist.

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
