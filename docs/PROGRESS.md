# Progress Log

## 2026-05-21

### Completed

- Confirmed `Playground` was empty before adding files.
- Confirmed parent repository has unrelated dirty changes; this project avoids touching them.
- Checked local tools: Python exists, CMake and C++ compilers are not on PATH.
- Created C++20 project scaffold.
- Added CLI/config/image/backend boundaries.
- Added deterministic mock backend.
- Added placeholder ORT GenAI backend boundary.
- Added model package conventions and license notes.
- Added Python sanity tooling that does not require C++ compilation.
- Ran `python tools/validate_project.py`: passed.
- Ran `python tools/make_test_ppm.py`: created `tmp/smoke.ppm`.
- Ran `git diff --check -- .`: passed.
- Found Visual Studio Build Tools under `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`.
- Configured CMake with MSVC and Ninja using `vcvars64.bat`.
- Built `scene_describer.exe` and `scene_describer_config_test.exe`.
- Fixed MSVC `/EHsc` warning by centralizing target compile options.
- Ran CTest: passed.
- Ran the mock CLI with `tmp/smoke.ppm`: returned JSON.
- Refactored request image handling so ORT GenAI can receive image paths directly instead of being blocked by the bootstrap PPM loader.
- Added `tools/build.ps1` to preserve the working Visual Studio build invocation.
- Ran `powershell -NoProfile -ExecutionPolicy Bypass -File tools\build.ps1 -Test`: passed.
- Downloaded ONNX Runtime GenAI 0.13.1 CPU package into ignored `.deps`.
- Updated CMake package finder to use `ort_genai.h`.
- Implemented first ORT GenAI backend path behind `SCENE_DESC_ENABLE_ORT_GENAI`.
- Compiled and tested with `-OrtGenAI` against real ORT GenAI headers/library.
- Hit a runtime DLL mismatch: Windows loaded `C:\Windows\System32\onnxruntime.dll` 1.17.1, but ORT GenAI requested API 23.
- Downloaded ONNX Runtime 1.25.1 CPU package into ignored `.deps`.
- Confirmed ORT-linked CLI launches after copying matching `onnxruntime.dll` and `onnxruntime-genai.dll` into `build`.
- Updated `tools/build.ps1` to copy app-local ORT DLLs for ORT-enabled builds.
- Fixed `tools/build.ps1` ordering so ORT DLLs are deployed before CTest runs.
- Ran normal wrapper build: `tools\build.ps1 -Test` passed.
- Ran ORT-enabled wrapper build with local `.deps`: passed.
- Ran ORT-linked mock CLI directly from `build`: returned JSON.
- Created `.venv` with Python 3.10.11.
- Found PyPI packaging mismatch: `onnxruntime-genai` is only available up to `0.11.4` in this environment, while Qwen3.5/Qwen3-VL export support is in `0.13.x`.
- Cloned `microsoft/onnxruntime-genai` source tag `v0.13.1` into ignored `.deps` for model-builder use.
- Confirmed public model config architecture names: `Qwen3_5ForConditionalGeneration` for `Qwen/Qwen3.5-2B`, and `Qwen3VLForConditionalGeneration` for `Qwen/Qwen3-VL-2B-Instruct`.
- Added reproducible scripts for runtime dependency fetch, Python export setup, ORT GenAI model export, and model package validation.
- Installed Python export dependencies in `.venv`.
- Confirmed the ORT GenAI source builder imports and reports help successfully.
- Ran config-only builder probes for Qwen3.5 and Qwen3-VL; both generated GenAI configs.
- Ran full Qwen3.5 INT4 CPU export; it generated a decoder-only package with `model.onnx` and `model.onnx.data`.
- Tightened model package validation so scene-runtime packages must include embedding, vision, and image processor config.
- Found a public ONNX Runtime GenAI subfolder in `onnx-community/Qwen3-VL-2B-Instruct-ONNX`; it has the needed multimodal package layout but no model-card/license metadata, so it is smoke-test only.
- Fetched the Qwen3-VL ONNX Runtime GenAI CPU package into `models\qwen3-vl-2b-instruct`.
- Validated that package with `python tools\validate_model_package.py models\qwen3-vl-2b-instruct --require-multimodal`: passed.
- Added `tools\make_test_image.py` to generate a PNG smoke scene.
- Ran C++ `backend=ort-genai` against `tmp\smoke.png` with Qwen3-VL: succeeded.
- Fixed ORT backend output trimming so returned text excludes the prompt prefix and only decodes generated tokens.
- Rebuilt ORT-enabled C++ binary after trimming fix: passed.
- Re-ran Qwen3-VL smoke command after trimming fix: returned a concise scene description without prompt echo.
- Added `tools\smoke_ort_genai.ps1` as an optional local smoke gate for the ORT GenAI path.
- Added `tools\model_provenance.py` and integrated provenance writing into `tools\fetch_hf_oga_package.py`.
- Added provenance validation flags: `--require-provenance` and `--require-production`.
- Regenerated Qwen3-VL smoke package provenance: package repo has no license metadata; upstream `Qwen/Qwen3-VL-2B-Instruct` resolves to Apache-2.0 metadata.
- Captured Qwen3.5 decoder-only export provenance and copied its upstream license file into the local model directory.
- Updated the smoke gate to require provenance.
- Added CLI JSON metadata output for backend metadata including input/generated token counts.
- Added `tools\benchmark_cli.py` and `docs\BENCHMARKING.md`.
- Ran Qwen3-VL CPU CLI-inclusive benchmark: median ~7.32 seconds for 39 generated tokens, mean ~5.35 generated tokens/sec inclusive.
- Added `scene_describer_benchmark` as an in-process benchmark executable that reuses a constructed backend.
- Ran mock benchmark smoke test: passed and emitted JSON.
- Ran Qwen3-VL CPU in-process benchmark: model load ~4.61 seconds; median request latency ~3.25 seconds for 39 generated tokens; mean ~12.03 generated tokens/sec.
- Added runtime packaging guidance and `tools\package_runtime.ps1` for local edge bundles without copying model weights by default.
- Fixed config loading to tolerate a UTF-8 BOM on the first line and added C++ test coverage.
- Ran packaged runtime smoke test from `dist\scene-describer\run.ps1`: passed against `tmp\smoke.png`.
- Added `tools\model_readiness.py`, `docs\MODEL_READINESS.md`, and `docs\EDGE_ROADMAP.md`.
- Confirmed current model matrix: Qwen3.5 source config has vision capability, but exported ORT GenAI package is decoder-only; Qwen3-VL package is runtime-multimodal but smoke-classified.
- Added `configs\qwen3-vl-2b.fast.ini` as the fast edge-oriented smoke profile.
- Added analyzer core with frames, tracks, prompt templates, prior summaries, and monotonic bounded history.
- Added `scene_analyzer.exe` and prompt templates under `prompts\analyzer`.
- Extended the ORT GenAI backend to accept multiple image paths for batch-style prompts.
- Ran analyzer smoke tests with mock and Qwen3-VL ORT GenAI: both passed.
- Ran packaged analyzer wrapper from `dist\scene-describer\run_analyzer.ps1`: passed with external Qwen3-VL model reference.
- Added `tools\prepare_qwen35_onnxopt_genai.py` for an experimental Qwen3.5 ONNX-OPT to ORT GenAI package path.
- Prepared `models\qwen3.5-2b-onnxopt-q4f16` from ONNX-OPT q4f16 decoder, embedding, and vision graphs.
- Patched Qwen3.5 decoder recurrent-state I/O names to match ORT GenAI recurrent-state discovery.
- Patched Qwen3.5 embedding graph to accept `image_features` and scatter them into image-token embeddings.
- Patched Qwen3.5 tokenizer metadata and tokenizer regex so ORT GenAI can construct the multimodal processor.
- Patched Qwen3.5 decoder graph to inline `num_logits_to_keep=0`, avoiding a missing multimodal-pipeline extra input.
- Added `scene_model_probe.exe` for staged ORT GenAI package diagnostics.
- Confirmed Qwen3.5 ONNX-OPT package reaches config, model, processor, inputs, generator, token, and generation stages.
- Ran `scene_describer.exe` with `configs\qwen3.5-2b-onnxopt.ini`: returned a correct smoke-image scene description.
- Ran `scene_analyzer.exe` with `configs\qwen3.5-2b-onnxopt.ini`: returned analyzer JSON through Qwen3.5.
- Added backend cleanup for leading Qwen `<think>...</think>` wrappers in decoded output.
- Re-ran ORT-enabled build/test after Qwen3.5 changes: passed.
- Re-ran the Qwen3-VL smoke gate after backend cleanup: passed.

### Mistakes / Corrections

- Initial plan assumed local CMake verification might be possible. Toolchain inspection showed it is not currently possible on PATH, so verification is split into Python sanity checks now and C++ build checks after toolchain setup.
- First CMake configure attempt used `VsDevCmd.bat` and failed to find `cl.exe`.
- Second configure attempt used `vcvars64.bat` but accidentally overwrote PATH with early `%PATH%` expansion. The working command uses `cmd /v:on` and delayed `!PATH!` expansion.
- ORT GenAI package alone was not enough at runtime because Windows loaded the stale System32 `onnxruntime.dll`; app-local DLL copying fixed that.
- Tried to install `onnxruntime-genai==0.13.1` from PyPI; no such wheel is currently available for this environment. The correction is to use the matching GitHub source tag for export tooling.
- Initially marked the Qwen3.5 full export as usable because it had `genai_config.json` and ONNX files. Inspecting the config showed it was decoder-only and missing `model.embedding`/`model.vision`; validation now catches this.
- First generated package config used PowerShell UTF-8 output with a BOM, which exposed that the C++ config parser rejected BOM-prefixed keys. The correction is both script-side ASCII config generation and parser-side BOM tolerance.
- The first Qwen3.5 ONNX-OPT GenAI package crashed through the main CLI. `scene_model_probe.exe` isolated the sequence: model load worked, processor failed on `TokenizersBackend`, then failed on tokenizer regex, then generator failed on `num_logits_to_keep`. Each issue is now patched in the preparation script.

### Next

- Keep Qwen3.5 as the target and decide whether the ONNX-OPT preparation path can be productionized after review.
- Resolve production license provenance for any prebuilt or patched ONNX package.
- Benchmark target accelerator execution providers after the CPU path stays stable.
- Add analyzer ZMQ compatibility if the work integration needs it.
- Add runtime image decoding policy for non-ORT preprocessing paths.
