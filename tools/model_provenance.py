#!/usr/bin/env python3
"""Capture reproducible provenance for a local model package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


PROVENANCE_FILENAME = "MODEL_PROVENANCE.json"
LICENSE_CANDIDATES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "License", "license.txt")


def _card_data_to_dict(card_data: Any) -> dict[str, Any] | None:
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data
    to_dict = getattr(card_data, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    data = getattr(card_data, "data", None)
    if isinstance(data, dict):
        return data
    return {"raw": str(card_data)}


def _license_from_info(info: Any) -> str | None:
    card_data = _card_data_to_dict(getattr(info, "cardData", None))
    if card_data and card_data.get("license"):
        return str(card_data["license"])
    for tag in getattr(info, "tags", []) or []:
        if tag.startswith("license:"):
            return tag.removeprefix("license:")
    return None


def _model_info(repo_id: str | None, revision: str | None) -> dict[str, Any] | None:
    if not repo_id:
        return None

    info = HfApi().model_info(repo_id=repo_id, revision=revision, files_metadata=False)
    return {
        "repo_id": repo_id,
        "revision_requested": revision,
        "resolved_revision": info.sha,
        "license": _license_from_info(info),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(getattr(info, "tags", []) or []),
        "card_data": _card_data_to_dict(getattr(info, "cardData", None)),
    }


def _copy_upstream_license(
    output_dir: Path,
    repo_id: str | None,
    revision: str | None,
    cache_dir: str | None,
) -> dict[str, str] | None:
    if not repo_id:
        return None

    for candidate in LICENSE_CANDIDATES:
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=candidate,
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception:
            continue

        target = output_dir / "UPSTREAM_LICENSE"
        shutil.copy2(cached, target)
        return {
            "repo_id": repo_id,
            "source_path": candidate,
            "local_path": target.name,
        }
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_files(output_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == PROVENANCE_FILENAME:
            continue
        relative = path.relative_to(output_dir).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def write_provenance(
    output_dir: Path,
    *,
    classification: str,
    package_repo_id: str | None,
    package_source_prefix: str | None,
    package_revision: str | None,
    upstream_repo_id: str | None,
    upstream_revision: str | None,
    cache_dir: str | None,
    notes: list[str],
    copy_upstream_license: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream_license_file = None
    if copy_upstream_license:
        upstream_license_file = _copy_upstream_license(output_dir, upstream_repo_id, upstream_revision, cache_dir)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "package": {
            **(_model_info(package_repo_id, package_revision) or {}),
            "source_prefix": package_source_prefix,
        },
        "upstream_model": _model_info(upstream_repo_id, upstream_revision),
        "upstream_license_file": upstream_license_file,
        "local_package": {
            "path": str(output_dir),
            "files": _local_files(output_dir),
        },
        "notes": notes,
    }

    manifest_path = output_dir / PROVENANCE_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classification", default="prototype")
    parser.add_argument("--package-repo-id", default=None)
    parser.add_argument("--package-source-prefix", default=None)
    parser.add_argument("--package-revision", default=None)
    parser.add_argument("--upstream-repo-id", default=None)
    parser.add_argument("--upstream-revision", default=None)
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--no-copy-upstream-license", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    args = parse_args()
    manifest_path = write_provenance(
        Path(args.output_dir),
        classification=args.classification,
        package_repo_id=args.package_repo_id,
        package_source_prefix=args.package_source_prefix,
        package_revision=args.package_revision,
        upstream_repo_id=args.upstream_repo_id,
        upstream_revision=args.upstream_revision,
        cache_dir=args.cache_dir,
        notes=args.note,
        copy_upstream_license=not args.no_copy_upstream_license,
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
