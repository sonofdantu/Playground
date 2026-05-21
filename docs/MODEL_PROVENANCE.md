# Model Provenance

Every local model package should carry `MODEL_PROVENANCE.json` before it is used beyond a smoke test.

The manifest records:

- Hugging Face package repository and resolved revision.
- Optional upstream model repository and resolved revision.
- License signals from model card metadata or tags.
- Local file sizes and SHA-256 hashes.
- Package classification: `smoke`, `prototype`, `production_candidate`, or `production`.
- Notes explaining known gaps.

## Commands

Capture provenance for a fetched ONNX package:

```powershell
.\.venv\Scripts\python.exe tools\fetch_hf_oga_package.py `
  --repo-id onnx-community/Qwen3-VL-2B-Instruct-ONNX `
  --source-prefix onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32 `
  --output-dir models\qwen3-vl-2b-instruct `
  --upstream-repo-id Qwen/Qwen3-VL-2B-Instruct `
  --classification smoke
```

Capture provenance for an existing local package:

```powershell
.\.venv\Scripts\python.exe tools\model_provenance.py `
  --output-dir models\qwen3.5-2b `
  --classification prototype `
  --package-repo-id Qwen/Qwen3.5-2B `
  --upstream-repo-id Qwen/Qwen3.5-2B
```

Validate a multimodal package with provenance:

```powershell
python tools\validate_model_package.py models\qwen3-vl-2b-instruct --require-multimodal --require-provenance
```

Production validation additionally requires `classification` to be `production_candidate` or `production`:

```powershell
python tools\validate_model_package.py models\qwen3-vl-2b-instruct --require-multimodal --require-production
```

## Current Packages

- `models\qwen3-vl-2b-instruct`: complete multimodal smoke package. Provenance links to upstream Apache-2.0 Qwen metadata, but the prebuilt ONNX package repo itself has no explicit license metadata.
- `models\qwen3.5-2b`: decoder-only prototype export. It has useful export evidence and upstream license capture, but is not a scene-runtime package.
