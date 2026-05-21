#!/usr/bin/env python3
"""Fetch a Hugging Face ONNX Runtime GenAI package subfolder."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from model_provenance import write_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository id.")
    parser.add_argument("--source-prefix", required=True, help="Repository subfolder containing genai_config.json.")
    parser.add_argument("--output-dir", required=True, help="Local model directory to populate.")
    parser.add_argument("--cache-dir", default=".cache/huggingface", help="Hugging Face cache directory.")
    parser.add_argument("--revision", default=None, help="Optional repository revision.")
    parser.add_argument("--upstream-repo-id", default=None, help="Original model repo used to establish license lineage.")
    parser.add_argument("--upstream-revision", default=None, help="Optional original model revision.")
    parser.add_argument("--classification", default="smoke", help="Package classification such as smoke or production.")
    parser.add_argument("--note", action="append", default=[], help="Additional provenance note.")
    parser.add_argument("--skip-provenance", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    args = parse_args()
    source_prefix = args.source_prefix.strip("/")
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=args.repo_id, revision=args.revision)
    package_files = [name for name in repo_files if name.startswith(source_prefix + "/")]
    if not package_files:
        raise SystemExit(f"no files found under {args.repo_id}:{source_prefix}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for repo_file in package_files:
        relative = repo_file[len(source_prefix) + 1 :]
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        cached = hf_hub_download(
            repo_id=args.repo_id,
            filename=repo_file,
            revision=args.revision,
            cache_dir=cache_dir,
        )
        shutil.copy2(cached, target)
        print(f"{repo_file} -> {target}")

    print(f"fetched {len(package_files)} files into {output_dir}")
    if not args.skip_provenance:
        notes = list(args.note)
        if not notes:
            notes.append("Fetched prebuilt ONNX Runtime GenAI subfolder for engineering validation.")
        manifest = write_provenance(
            output_dir,
            classification=args.classification,
            package_repo_id=args.repo_id,
            package_source_prefix=source_prefix,
            package_revision=args.revision,
            upstream_repo_id=args.upstream_repo_id,
            upstream_revision=args.upstream_revision,
            cache_dir=str(cache_dir),
            notes=notes,
        )
        print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
