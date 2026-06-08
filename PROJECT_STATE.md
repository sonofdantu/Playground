# Project State

Last updated: 2026-05-26

## Goal

Build a professional C++ edge scene-description runtime that can run a commercially usable Qwen vision-language model through ONNX Runtime GenAI.

## Current Milestone

Bootstrap codebase exists with a mock backend, CLI/config/image loading boundaries, CMake files, model package conventions, documentation, an ORT GenAI backend that runs against local ORT dependencies, benchmark paths, and an analyzer surface. The current step is moving the working Qwen3.5 ONNX-OPT prototype from local CPU proof into production-cleared provenance, accelerator benchmarking, and packaging.

Verified so far:

- `python tools/validate_project.py` passed.
- `python tools/make_test_ppm.py` created `tmp/smoke.ppm`.
- `git diff --check -- .` passed.
- MSVC/Ninja/CMake configure succeeded through Visual Studio Build Tools.
- `cmake --build build --config Release` succeeded.
- `ctest --test-dir build --output-on-failure` passed.
- `build\scene_describer.exe --config configs\mock.local.ini --image tmp\smoke.ppm --json` returned mock scene JSON.
- Downloaded ignored local dependencies: ONNX Runtime GenAI 0.13.1 CPU package and ONNX Runtime 1.25.1 CPU package.
- ORT-enabled build compiles and tests when using those local dependencies.
- ORT-linked mock CLI launches successfully after copying `onnxruntime-genai.dll` and matching `onnxruntime.dll` beside `scene_describer.exe`.
- `.venv` exists with Python 3.10.11.
- ORT GenAI source tag `v0.13.1` is cloned under ignored `.deps` for model-builder use.
- PyPI currently does not provide `onnxruntime-genai==0.13.1` for this environment, so export tooling runs the source builder directly.
- Public model configs match ORT GenAI builder support: `Qwen3_5ForConditionalGeneration` and `Qwen3VLForConditionalGeneration`.
- Python export dependencies are installed in `.venv`.
- Full Qwen3.5 INT4 CPU export completed, but the generated package is decoder-only and not sufficient for scene description.
- The validator now fails packages that are missing `model.embedding`, `model.vision`, or image processor config when `--require-multimodal` is used.
- `onnx-community/Qwen3-VL-2B-Instruct-ONNX` has an ORT GenAI CPU subfolder with a complete multimodal layout, but no model-card/license metadata was observed.
- Qwen3-VL smoke package was fetched to `models\qwen3-vl-2b-instruct` and validated as multimodal.
- `tools\make_test_image.py` generated `tmp\smoke.png`.
- ORT-enabled C++ CLI successfully described `tmp\smoke.png` using `configs\qwen3-vl-2b.ini`.
- ORT backend now trims decoded output to generated tokens only.
- `tools\smoke_ort_genai.ps1` wraps multimodal package validation, ORT build/test, PNG generation, CLI execution, JSON parsing, and prompt-echo detection.
- Local model packages now support `MODEL_PROVENANCE.json` with repo/revision/license signals and file hashes.
- Qwen3-VL smoke package has provenance linking to upstream Apache-2.0 Qwen metadata, but remains classified as `smoke`.
- Qwen3.5 decoder-only export has provenance and copied upstream license, but remains classified as `prototype`.
- CLI JSON now includes backend metadata.
- CPU CLI-inclusive benchmark baseline: median ~7.32 seconds for 39 generated tokens, about 5.35 generated tokens/sec inclusive.
- Added `scene_describer_benchmark`, which constructs the backend once and measures repeated in-process requests.
- CPU in-process Qwen3-VL smoke benchmark: model load ~4.61 seconds; median request latency ~3.25 seconds for 39 generated tokens; mean throughput ~12.03 generated tokens/sec.
- Added `tools\package_runtime.ps1` and `docs\PACKAGING.md` for local edge-runtime bundles. Model weights are referenced by default and copied only with `-IncludeModel`.
- Config loading now tolerates a UTF-8 BOM on the first line; the C++ config test covers this.
- Packaged runtime smoke test passed from `dist\scene-describer\run.ps1` against `tmp\smoke.png`.
- Added `tools\model_readiness.py`, `docs\MODEL_READINESS.md`, and `docs\EDGE_ROADMAP.md`.
- Current model readiness: Qwen3.5 source is VLM-capable but exported package is decoder-only; Qwen3-VL package is runtime-multimodal but still classified `smoke`.
- Added `configs\qwen3-vl-2b.fast.ini` for edge-oriented latency smoke tests.
- Added analyzer layer: `AnalyzerRequest`, track metadata, prompt templates, bounded monotonic `HistoryStore`, and `scene_analyzer.exe`.
- ORT GenAI backend now accepts multiple image paths in `SceneDescriptionRequest::image_paths`.
- `scene_analyzer.exe` smoke test passed through Qwen3-VL with image + track + history metadata.
- Packaged analyzer wrapper `dist\scene-describer\run_analyzer.ps1` passed with the Qwen3-VL smoke model referenced externally.
- Added `tools\prepare_qwen35_onnxopt_genai.py` to assemble an experimental ORT GenAI-style package from `onnx-community/Qwen3.5-2B-ONNX-OPT`.
- The Qwen3.5 ONNX-OPT preparation path patches recurrent-state names, tokenizer metadata, tokenizer regex compatibility, image-feature embedding injection, and decoder `num_logits_to_keep`.
- `models\qwen3.5-2b-onnxopt-q4f16` validates as multimodal with provenance and runs locally through ORT GenAI CPU.
- Added `scene_model_probe.exe` for staged ORT GenAI debugging: config, model, processor, inputs, generator, token, and generate.
- `build\scene_describer.exe --config configs\qwen3.5-2b-onnxopt.ini --image tmp\smoke.png --max-new-tokens 32 --json` returned a correct scene description.
- `build\scene_analyzer.exe --config configs\qwen3.5-2b-onnxopt.ini --image tmp\smoke.png ... --json` returned analyzer JSON through Qwen3.5.
- ORT backend now strips a leading Qwen `<think>...</think>` wrapper from returned scene text.
- Linux `.venv` now carries local CMake/Ninja plus lightweight Qwen3.5 preparation dependencies.
- Linux `tools/build.sh --test --ort-genai` passed with ORT GenAI 0.13.1 and ONNX Runtime 1.25.1.
- Linux `./build/scene_describer --config configs/qwen3.5-2b-onnxopt.ini --image tmp/smoke.png --max-new-tokens 16 --json` returned a correct Qwen3.5 scene description.
- Added `tools/smoke_ort_genai.sh` as the Linux equivalent of the Windows ORT GenAI smoke gate.
- Added `AGENTS.md` and `docs/LINUX_REPRODUCTION.md` so future AI agents can recreate the Linux Qwen3.5 runtime from a clean pull.
- Added Linux CUDA dependency/setup plumbing. CUDA model load reaches `ok=model type=qwen3_5 device=CUDA` on WSL2 RTX 4070 Laptop.
- Root-caused the raw decoder CUDA blocker to ORT CUDA `GroupQueryAttention` rejecting the optional `attention_bias` input. The Qwen3.5 preparation script now removes that optional input for the no-padding smoke/runtime path.
- Added a C++ raw ONNX Runtime CUDA generation loop for Qwen3.5. ORT GenAI still provides preprocessing/tokenization/decoding, but `vision_encoder_*`, `embed_tokens_*`, and `decoder_model_merged_*` execute through raw ORT CUDA sessions.
- `./tools/smoke_ort_genai.sh --execution-provider cuda --config configs/qwen3.5-2b-onnxopt-cuda.ini --max-new-tokens 32` passed and returned `A blue truck drives past a red house under a yellow sun.` with `metadata.execution_provider=raw-ort-cuda`.
- Added `tools/debug_cuda_ort_genai.sh` to collect CUDA provider, library-resolution, model-stage, legacy token-stage, raw CUDA CLI, `dmesg`, and optional `gdb` evidence on Linux.
- Added chunked raw-ORT CUDA prefill for long multimodal prompts and a video processor profile that keeps 30-120 frame requests inside the tested 8GB RTX 4070 Laptop GPU envelope.
- Added `tools/make_test_frames.py` and `tools/smoke_cuda_frame_batch.sh`. Verified `./tools/smoke_cuda_frame_batch.sh --frame-count 120 --max-new-tokens 48` returns analyzer JSON with `metadata.execution_provider=raw-ort-cuda`, `metadata.frame_count=120`, and `metadata.prefill_chunk_tokens=512`.

## Key Decisions

- Keep model/backend code behind `ISceneDescriber` so Qwen3.5 and Qwen3-VL can be swapped without rewriting the app.
- Use ONNX Runtime GenAI as the intended inference runtime because it owns generation loop concerns: KV/recurrent state, sampling, tokenizer, multimodal processing, and execution providers.
- Track progress in checked-in docs because this work may span context compactions.
- Do not vendor model files or large generated artifacts.
- Start with a mock backend and PPM loader to make core plumbing testable before export/runtime details are solved.
- Use ORT GenAI source `v0.13.1` for export until matching Python wheels are available.
- Do not mark a model package as scene-runtime-ready unless `genai_config.json` includes decoder, embedding, vision, and processor config.

## Known Blockers

- PNG/JPEG decoding is not added yet. Current bootstrap loader supports binary PPM/PGM only.
- The direct exported Qwen3.5 package at `models\qwen3.5-2b` is decoder-only. It is useful evidence, not a usable scene-description runtime package.
- The working Qwen3.5 ONNX-OPT package at `models\qwen3.5-2b-onnxopt-q4f16` is still a prototype because it depends on local graph/package patches and needs provenance/legal and target-device review.
- The onnx-community Qwen3-VL ONNX package is missing explicit license metadata. Use it for smoke testing only until legal provenance is resolved.
- Current image support depends on ORT GenAI image loading for the real backend; the shared C++ image loader is still only a bootstrap PPM/PGM loader for mock tests.
- The direct ORT GenAI CUDA generator path is still not production-ready in this WSL2 test environment: it loads Qwen3.5 on CUDA but still crashes after `stage=generator`. The production CUDA smoke path is the checked-in raw ONNX Runtime CUDA loop.

## Next Steps

1. Benchmark the raw ONNX Runtime CUDA Qwen3.5 loop on target video frame rates and decide production frame sampling policy.
2. Decide whether to productionize the ONNX-OPT preparation path or pursue an internal complete Qwen3.5 export.
3. Add ZMQ protocol compatibility for analyzer requests/results if needed.
4. Add runtime image decoding policy for non-ORT preprocessing paths.
5. Decide the production model package source after legal/provenance review.
6. Re-test the direct ORT GenAI CUDA generator on newer ORT GenAI releases, but keep the raw ORT CUDA path as the stable Linux GPU route.
