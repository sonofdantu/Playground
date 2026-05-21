#!/usr/bin/env python3
"""Validate the minimum shape of an ONNX Runtime GenAI model directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    allow_config_only = False
    allow_decoder_only = False
    require_provenance = False
    require_production = False
    require_multimodal = False
    if "--allow-config-only" in args:
        allow_config_only = True
        args.remove("--allow-config-only")
    if "--allow-decoder-only" in args:
        allow_decoder_only = True
        args.remove("--allow-decoder-only")
    if "--require-multimodal" in args:
        require_multimodal = True
        args.remove("--require-multimodal")
    if "--require-provenance" in args:
        require_provenance = True
        args.remove("--require-provenance")
    if "--require-production" in args:
        require_production = True
        require_provenance = True
        args.remove("--require-production")

    if len(args) != 1:
        raise SystemExit(
            "usage: validate_model_package.py <model-dir> "
            "[--allow-config-only] [--allow-decoder-only] [--require-multimodal] "
            "[--require-provenance] [--require-production]"
        )

    model_dir = Path(args[0])
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")

    config_path = model_dir / "genai_config.json"
    if not config_path.exists():
        raise SystemExit(f"missing {config_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid {config_path}: {exc}") from exc

    model_section = config.get("model")
    if not isinstance(model_section, dict):
        raise SystemExit(f"{config_path}: expected top-level 'model' object")

    onnx_files = sorted(str(path.relative_to(model_dir)) for path in model_dir.rglob("*.onnx"))
    if not onnx_files:
        external_files = sorted(str(path.relative_to(model_dir)) for path in model_dir.rglob("*.onnx.data"))
        if not external_files and not allow_config_only:
            raise SystemExit(f"{model_dir}: expected at least one .onnx file")

    processing_candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ]
    present_processing = [name for name in processing_candidates if (model_dir / name).exists()]
    if not present_processing:
        raise SystemExit(f"{model_dir}: expected tokenizer or processor files")

    decoder = model_section.get("decoder")
    decoder_inputs = decoder.get("inputs", {}) if isinstance(decoder, dict) else {}
    problems: list[str] = []
    if (
        "inputs_embeds" in decoder_inputs
        and not isinstance(model_section.get("embedding"), dict)
        and not allow_decoder_only
    ):
        problems.append("decoder consumes inputs_embeds but model.embedding is missing")

    if require_multimodal:
        if not isinstance(model_section.get("vision"), dict):
            problems.append("scene runtime requires model.vision, but it is missing")
        if not any((model_dir / name).exists() for name in ("preprocessor_config.json", "processor_config.json")):
            problems.append("scene runtime requires an image processor config")

    provenance_path = model_dir / "MODEL_PROVENANCE.json"
    provenance = None
    if require_provenance:
        if not provenance_path.exists():
            problems.append("MODEL_PROVENANCE.json is required")
        else:
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"MODEL_PROVENANCE.json is invalid JSON: {exc}")

    if isinstance(provenance, dict):
        package_files = provenance.get("local_package", {}).get("files", [])
        if not package_files:
            problems.append("MODEL_PROVENANCE.json must include local file hashes")

        upstream = provenance.get("upstream_model")
        upstream_license = upstream.get("license") if isinstance(upstream, dict) else None
        package = provenance.get("package")
        package_license = package.get("license") if isinstance(package, dict) else None
        if not upstream_license and not package_license:
            problems.append("MODEL_PROVENANCE.json must include an upstream or package license signal")

        if require_production and provenance.get("classification") not in {"production_candidate", "production"}:
            problems.append("MODEL_PROVENANCE.json classification is not production_candidate or production")

    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise SystemExit(f"model package is incomplete for this runtime:\n{details}")

    model_type = model_section.get("type", "<unknown>")
    if onnx_files:
        print(f"model package looks usable: type={model_type}, onnx_files={onnx_files}")
    else:
        print(f"config-only model package looks usable: type={model_type}")


if __name__ == "__main__":
    main()
