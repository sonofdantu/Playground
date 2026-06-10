#!/usr/bin/env python3
"""Create high-resolution detail crops from generated security frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", default="tmp/frames")
    parser.add_argument("--output-dir", default="tmp/detail-crops")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--frame-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--output-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--sample-frames", default="15")
    parser.add_argument("--crop-types", default="handheld,drone")
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            result.append(int(stripped))
    return result


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def centered_box(
    center_x: int,
    center_y: int,
    crop_width: int,
    crop_height: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = max(0, min(width - crop_width, center_x - crop_width // 2))
    top = max(0, min(height - crop_height, center_y - crop_height // 2))
    return left, top, left + crop_width, top + crop_height


def drone_center(index: int, count: int, width: int, height: int) -> tuple[int, int]:
    progress = index / max(1, count - 1)
    return int(width * (0.78 - progress * 0.18)), int(height * (0.18 + progress * 0.05))


def handheld_center(index: int, count: int, width: int, height: int) -> tuple[int, int]:
    progress = index / max(1, count - 1)
    person_x = int(width * (0.74 - progress * 0.30))
    person_y = int(height * 0.66)
    head_r = int(height * 0.027)
    body_top = person_y + head_r
    return person_x + int(width * 0.067), body_top + int(height * 0.064)


def crop_note(kind: str, source_index: int) -> str:
    if kind == "drone":
        return f"High-resolution sky crop from source frame {source_index}; inspect for small airborne objects such as drones."
    if kind == "handheld":
        return f"High-resolution person/hand crop from source frame {source_index}; inspect for small held objects such as phones, packages, or tools."
    return f"High-resolution crop from source frame {source_index}; inspect for small details."


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("detail_*.png", "detail_*.jpg", "detail_*.jpeg"):
        for old_crop in output_dir.glob(pattern):
            old_crop.unlink()

    sample_frames = parse_csv_ints(args.sample_frames)
    crop_types = parse_csv_strings(args.crop_types)
    frame_suffix = "jpg" if args.frame_format == "jpg" else "png"
    output_suffix = "jpg" if args.output_format == "jpg" else "png"
    for sample_index in sample_frames:
        if sample_index < 0 or sample_index >= args.count:
            raise SystemExit(f"sample frame out of range: {sample_index}")
        frame_path = frames_dir / f"frame_{sample_index:04d}.{frame_suffix}"
        if not frame_path.exists():
            raise SystemExit(f"missing frame: {frame_path}")

        frame = Image.open(frame_path).convert("RGB")
        width, height = frame.size
        crop_width = max(320, width // 3)
        crop_height = max(180, height // 3)
        for kind in crop_types:
            if kind == "drone":
                center_x, center_y = drone_center(sample_index, args.count, width, height)
            elif kind == "handheld":
                center_x, center_y = handheld_center(sample_index, args.count, width, height)
            else:
                raise SystemExit(f"unknown crop type: {kind}")

            crop = frame.crop(centered_box(center_x, center_y, crop_width, crop_height, width, height))
            crop = crop.resize((width, height), Image.Resampling.BICUBIC)
            crop_path = output_dir / f"detail_{sample_index:04d}_{kind}.{output_suffix}"
            if args.output_format == "jpg":
                crop.save(crop_path, quality=args.jpeg_quality)
            else:
                crop.save(crop_path)
            print(f"{sample_index}\t{kind}\t{crop_path}\t{crop_note(kind, sample_index)}")


if __name__ == "__main__":
    main()
