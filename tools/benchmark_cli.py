#!/usr/bin/env python3
"""Benchmark the scene_describer CLI with repeatable JSON output."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default="build/scene_describer.exe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--save", default=None)
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        args.exe,
        "--config",
        args.config,
        "--image",
        args.image,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--json",
    ]
    if args.model_dir:
        command.extend(["--model-dir", args.model_dir])

    start = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    payload = json.loads(completed.stdout)
    text = str(payload.get("text", "")).strip()
    if not text:
        raise RuntimeError("benchmark command returned empty text")

    metadata = payload.get("metadata", {})
    generated_tokens = None
    if isinstance(metadata, dict) and metadata.get("generated_tokens"):
        generated_tokens = int(metadata["generated_tokens"])

    run = {
        "elapsed_ms": elapsed_ms,
        "text_chars": len(text),
        "backend": payload.get("backend"),
    }
    if generated_tokens is not None:
        run["generated_tokens"] = generated_tokens
        run["generated_tokens_per_second_inclusive"] = generated_tokens / (elapsed_ms / 1000.0)

    return run


def main() -> None:
    args = parse_args()
    for _ in range(args.warmups):
        run_once(args)

    runs = [run_once(args) for _ in range(args.repeats)]
    elapsed = [run["elapsed_ms"] for run in runs]
    throughput = [
        run["generated_tokens_per_second_inclusive"]
        for run in runs
        if "generated_tokens_per_second_inclusive" in run
    ]
    summary = {
        "min_ms": min(elapsed),
        "median_ms": statistics.median(elapsed),
        "max_ms": max(elapsed),
        "mean_ms": statistics.fmean(elapsed),
    }
    if throughput:
        summary["mean_generated_tokens_per_second_inclusive"] = statistics.fmean(throughput)

    result = {
        "command": {
            "exe": args.exe,
            "config": args.config,
            "image": args.image,
            "model_dir": args.model_dir,
            "max_new_tokens": args.max_new_tokens,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "summary": summary,
        "runs": runs,
    }

    output = json.dumps(result, indent=2)
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
