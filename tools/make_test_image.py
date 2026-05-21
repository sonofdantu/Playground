#!/usr/bin/env python3
"""Create a small PNG scene for multimodal runtime smoke tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out_dir = Path("tmp")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "smoke.png"

    image = Image.new("RGB", (320, 200), (170, 205, 230))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 125, 320, 200), fill=(70, 145, 85))
    draw.ellipse((245, 18, 292, 65), fill=(250, 210, 70))
    draw.rectangle((42, 84, 132, 145), fill=(185, 70, 58))
    draw.polygon([(36, 84), (87, 42), (138, 84)], fill=(90, 88, 105))
    draw.rectangle((73, 104, 100, 145), fill=(85, 54, 42))
    draw.rectangle((108, 100, 126, 118), fill=(235, 245, 255))
    draw.rectangle((190, 115, 260, 145), fill=(45, 90, 150))
    draw.ellipse((198, 140, 216, 158), fill=(28, 28, 32))
    draw.ellipse((238, 140, 256, 158), fill=(28, 28, 32))
    draw.line((0, 166, 320, 155), fill=(210, 210, 210), width=10)
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
