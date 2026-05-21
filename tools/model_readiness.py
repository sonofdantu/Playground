#!/usr/bin/env python3
"""Report model-package readiness for the C++ scene runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_invalid_json": True}
    return value if isinstance(value, dict) else {}


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def first_architecture(config: dict[str, Any]) -> str:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return "<unknown>"


def license_signal(provenance: dict[str, Any]) -> str:
    package = provenance.get("package")
    if isinstance(package, dict) and package.get("license"):
        return str(package["license"])

    upstream = provenance.get("upstream_model")
    if isinstance(upstream, dict) and upstream.get("license"):
        return str(upstream["license"])

    return "<missing>"


def inspect_model_dir(model_dir: Path) -> dict[str, Any]:
    hf_config = load_json(model_dir / "config.json")
    genai_config = load_json(model_dir / "genai_config.json")
    provenance = load_json(model_dir / "MODEL_PROVENANCE.json")

    model_section = genai_config.get("model") if isinstance(genai_config.get("model"), dict) else {}
    decoder = model_section.get("decoder") if isinstance(model_section.get("decoder"), dict) else {}
    decoder_inputs = decoder.get("inputs") if isinstance(decoder.get("inputs"), dict) else {}

    processor_files = ["processor_config.json", "preprocessor_config.json"]
    has_processor = any((model_dir / name).exists() for name in processor_files)
    has_tokenizer = any((model_dir / name).exists() for name in ("tokenizer.json", "tokenizer_config.json"))
    has_onnx = any(model_dir.rglob("*.onnx"))
    has_decoder = isinstance(decoder, dict) and bool(decoder)
    has_embedding = isinstance(model_section.get("embedding"), dict)
    has_vision = isinstance(model_section.get("vision"), dict)
    consumes_inputs_embeds = "inputs_embeds" in decoder_inputs

    source_has_vision = isinstance(hf_config.get("vision_config"), dict)
    is_runtime_multimodal = has_decoder and has_embedding and has_vision and has_processor
    has_required_runtime_files = has_onnx and has_tokenizer and has_decoder

    if is_runtime_multimodal:
        recommendation = "runtime-ready smoke candidate"
    elif source_has_vision and has_decoder and consumes_inputs_embeds and not has_embedding:
        recommendation = "source is VLM, export is decoder-only"
    elif has_required_runtime_files:
        recommendation = "text-only runtime candidate"
    else:
        recommendation = "incomplete package"

    return {
        "path": str(model_dir),
        "source_architecture": first_architecture(hf_config),
        "source_model_type": hf_config.get("model_type", "<unknown>"),
        "source_has_vision_config": source_has_vision,
        "runtime_type": model_section.get("type", "<unknown>"),
        "has_onnx": has_onnx,
        "has_tokenizer": has_tokenizer,
        "has_processor_config": has_processor,
        "has_decoder": has_decoder,
        "decoder_consumes_inputs_embeds": consumes_inputs_embeds,
        "has_embedding": has_embedding,
        "has_vision": has_vision,
        "is_runtime_multimodal": is_runtime_multimodal,
        "provenance_classification": provenance.get("classification", "<missing>"),
        "license": license_signal(provenance),
        "recommendation": recommendation,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "path",
        "source_architecture",
        "runtime_type",
        "src_vision",
        "decoder",
        "embedding",
        "vision",
        "processor",
        "classification",
        "license",
        "recommendation",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["path"],
                row["source_architecture"],
                row["runtime_type"],
                bool_text(bool(row["source_has_vision_config"])),
                bool_text(bool(row["has_decoder"])),
                bool_text(bool(row["has_embedding"])),
                bool_text(bool(row["has_vision"])),
                bool_text(bool(row["has_processor_config"])),
                row["provenance_classification"],
                row["license"],
                row["recommendation"],
            ]
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a compact table.")
    args = parser.parse_args()

    rows = []
    for model_dir in args.model_dirs:
        if not model_dir.is_dir():
            raise SystemExit(f"model directory does not exist: {model_dir}")
        rows.append(inspect_model_dir(model_dir))

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
