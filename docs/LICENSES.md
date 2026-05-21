# License Notes

This is not legal advice. Treat this file as engineering intake for legal review.

## Runtime Dependencies

| Component | Intended Use | License Signal | Commercial Notes |
| --- | --- | --- | --- |
| ONNX Runtime | Inference engine | MIT | Permissive |
| ONNX Runtime GenAI | Generation runtime | MIT | Permissive |
| C++ standard library | Core runtime | Toolchain dependent | Standard commercial use |

## Candidate Models

| Model | License Signal | Status |
| --- | --- | --- |
| `Qwen/Qwen3.5-2B` | Apache-2.0 on Hugging Face | Preferred edge target |
| `Qwen/Qwen3-VL-2B-Instruct` | Apache-2.0 on Hugging Face | Conservative VLM fallback |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0 on Hugging Face | Heavier quality fallback |
| `onnx-community/Qwen3-VL-2B-Instruct-ONNX` | No model-card/license metadata observed | Engineering smoke only; do not production ship without legal review |

Avoid Qwen model variants marked with a custom `qwen` license unless legal explicitly approves them.

## Dependency Rules

- No GPL or AGPL runtime dependencies.
- Prefer MIT, BSD, Apache-2.0, or similarly permissive licenses.
- Any vendored third-party code must include its original license file or notice.
- Model package directories must include a copy or pointer to the exact model license.
- Local model packages should include `MODEL_PROVENANCE.json`; see [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).
