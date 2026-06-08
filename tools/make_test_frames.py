#!/usr/bin/env python3
"""Create synthetic PNG frames for CUDA multi-frame smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tmp/frames")
    parser.add_argument("--count", type=int, default=120)
    return parser.parse_args()


def draw_frame(index: int, count: int) -> Image.Image:
    image = Image.new("RGB", (320, 200), (170, 205, 230))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 125, 320, 200), fill=(70, 145, 85))
    draw.ellipse((245, 18, 292, 65), fill=(250, 210, 70))
    draw.rectangle((42, 84, 132, 145), fill=(185, 70, 58))
    draw.polygon([(36, 84), (87, 42), (138, 84)], fill=(90, 88, 105))
    draw.rectangle((73, 104, 100, 145), fill=(85, 54, 42))
    draw.rectangle((108, 100, 126, 118), fill=(235, 245, 255))
    draw.line((0, 166, 320, 155), fill=(210, 210, 210), width=10)

    progress = index / max(1, count - 1)
    truck_x = int(18 + progress * 220)
    truck_y = 116
    draw.rectangle((truck_x, truck_y, truck_x + 70, truck_y + 30), fill=(45, 90, 150))
    draw.rectangle((truck_x + 45, truck_y - 14, truck_x + 67, truck_y), fill=(45, 90, 150))
    draw.rectangle((truck_x + 51, truck_y - 10, truck_x + 63, truck_y - 2), fill=(235, 245, 255))
    draw.ellipse((truck_x + 8, truck_y + 25, truck_x + 26, truck_y + 43), fill=(28, 28, 32))
    draw.ellipse((truck_x + 48, truck_y + 25, truck_x + 66, truck_y + 43), fill=(28, 28, 32))

    return image


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in output_dir.glob("frame_*.png"):
        old_frame.unlink()

    for index in range(args.count):
        frame = draw_frame(index, args.count)
        frame.save(output_dir / f"frame_{index:04d}.png")

    print(output_dir)


if __name__ == "__main__":
    main()
