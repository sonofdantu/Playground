#!/usr/bin/env python3
"""Create a tiny binary PPM image for CLI smoke tests."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    out_dir = Path("tmp")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "smoke.ppm"
    width, height = 4, 3
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend((x * 60, y * 80, 128))

    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(pixels))
    print(path)


if __name__ == "__main__":
    main()

