#!/usr/bin/env python3
"""Create synthetic security frames for CUDA multi-frame smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tmp/frames")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--no-small-objects", action="store_true")
    return parser.parse_args()


def draw_drone(draw: ImageDraw.ImageDraw, x: int, y: int, scale: int) -> None:
    arm = max(8, scale)
    rotor = max(5, scale // 2)
    body_w = max(12, int(scale * 0.9))
    body_h = max(7, int(scale * 0.45))
    draw.ellipse((x - body_w, y - body_h, x + body_w, y + body_h), fill=(28, 30, 34))
    draw.line((x - arm * 2, y, x + arm * 2, y), fill=(28, 30, 34), width=max(3, scale // 5))
    draw.line((x, y - arm, x, y + arm), fill=(28, 30, 34), width=max(3, scale // 5))
    for rotor_x, rotor_y in ((x - arm * 2, y), (x + arm * 2, y), (x, y - arm), (x, y + arm)):
        draw.ellipse((rotor_x - rotor, rotor_y - rotor, rotor_x + rotor, rotor_y + rotor), outline=(20, 22, 24), width=max(2, scale // 7))


def draw_frame(index: int, count: int, width: int, height: int, small_objects: bool) -> Image.Image:
    image = Image.new("RGB", (width, height), (176, 206, 228))
    draw = ImageDraw.Draw(image)

    ground_y = int(height * 0.64)
    road_y = int(height * 0.80)
    draw.rectangle((0, ground_y, width, height), fill=(68, 138, 82))
    draw.rectangle((0, road_y, width, height), fill=(84, 88, 92))
    draw.line((0, road_y + int(height * 0.055), width, road_y + int(height * 0.025)), fill=(210, 210, 198), width=max(6, width // 150))
    draw.ellipse((int(width * 0.82), int(height * 0.06), int(width * 0.92), int(height * 0.24)), fill=(248, 210, 74))

    house_left = int(width * 0.10)
    house_top = int(height * 0.35)
    house_right = int(width * 0.37)
    house_bottom = int(height * 0.70)
    roof_peak = (int(width * 0.235), int(height * 0.20))
    draw.rectangle((house_left, house_top, house_right, house_bottom), fill=(180, 68, 58))
    draw.polygon([(house_left - int(width * 0.025), house_top), roof_peak, (house_right + int(width * 0.025), house_top)], fill=(82, 82, 100))
    draw.rectangle((int(width * 0.20), int(height * 0.53), int(width * 0.25), house_bottom), fill=(82, 52, 42))
    draw.rectangle((int(width * 0.29), int(height * 0.49), int(width * 0.35), int(height * 0.58)), fill=(230, 242, 250))
    draw.rectangle((int(width * 0.43), int(height * 0.47), int(width * 0.66), house_bottom), fill=(190, 190, 184))
    draw.rectangle((int(width * 0.46), int(height * 0.53), int(width * 0.63), house_bottom), fill=(96, 104, 116))
    draw.polygon(
        [
            (int(width * 0.40), int(height * 0.47)),
            (int(width * 0.54), int(height * 0.34)),
            (int(width * 0.69), int(height * 0.47)),
        ],
        fill=(76, 80, 92),
    )

    progress = index / max(1, count - 1)
    if small_objects:
        drone_x = int(width * (0.78 - progress * 0.18))
        drone_y = int(height * (0.18 + progress * 0.05))
        draw_drone(draw, drone_x, drone_y, max(16, width // 95))

    truck_w = int(width * 0.18)
    truck_h = int(height * 0.105)
    truck_x = int(width * 0.03 + progress * width * 0.60)
    truck_y = int(height * 0.68)
    draw.rectangle((truck_x, truck_y, truck_x + truck_w, truck_y + truck_h), fill=(42, 86, 150))
    draw.rectangle((truck_x + int(truck_w * 0.58), truck_y - int(truck_h * 0.48), truck_x + int(truck_w * 0.92), truck_y), fill=(42, 86, 150))
    draw.rectangle((truck_x + int(truck_w * 0.67), truck_y - int(truck_h * 0.36), truck_x + int(truck_w * 0.87), truck_y - int(truck_h * 0.08)), fill=(230, 242, 250))
    wheel_r = int(height * 0.035)
    for wheel_x in (truck_x + int(truck_w * 0.18), truck_x + int(truck_w * 0.78)):
        draw.ellipse((wheel_x - wheel_r, truck_y + truck_h - wheel_r, wheel_x + wheel_r, truck_y + truck_h + wheel_r), fill=(26, 26, 30))

    person_x = int(width * (0.74 - progress * 0.30))
    person_y = int(height * 0.66)
    head_r = int(height * 0.027)
    body_top = person_y + head_r
    body_bottom = person_y + int(height * 0.16)
    draw.ellipse((person_x - head_r, person_y - head_r, person_x + head_r, person_y + head_r), fill=(72, 46, 34))
    draw.rectangle((person_x - int(width * 0.018), body_top, person_x + int(width * 0.018), body_bottom), fill=(236, 206, 76))
    draw.line((person_x - int(width * 0.018), body_top + int(height * 0.04), person_x - int(width * 0.06), body_bottom - int(height * 0.035)), fill=(32, 42, 55), width=max(5, width // 220))
    draw.line((person_x + int(width * 0.018), body_top + int(height * 0.04), person_x + int(width * 0.055), body_bottom - int(height * 0.02)), fill=(32, 42, 55), width=max(5, width // 220))
    draw.line((person_x - int(width * 0.008), body_bottom, person_x - int(width * 0.04), body_bottom + int(height * 0.07)), fill=(32, 42, 55), width=max(5, width // 220))
    draw.line((person_x + int(width * 0.008), body_bottom, person_x + int(width * 0.04), body_bottom + int(height * 0.07)), fill=(32, 42, 55), width=max(5, width // 220))
    draw.rectangle(
        (
            person_x + int(width * 0.045),
            body_top + int(height * 0.035),
            person_x + int(width * 0.085),
            body_top + int(height * 0.085),
        ),
        fill=(212, 130, 56) if small_objects else (236, 206, 76),
    )
    if small_objects:
        draw.rectangle(
            (
                person_x + int(width * 0.060),
                body_top + int(height * 0.044),
                person_x + int(width * 0.073),
                body_top + int(height * 0.080),
            ),
            fill=(30, 36, 48),
        )

    return image


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.width < 64 or args.height < 64:
        raise SystemExit("--width and --height must be at least 64")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("frame_*.png", "frame_*.jpg", "frame_*.jpeg"):
        for old_frame in output_dir.glob(pattern):
            old_frame.unlink()

    suffix = "jpg" if args.format == "jpg" else "png"
    for index in range(args.count):
        frame = draw_frame(index, args.count, args.width, args.height, not args.no_small_objects)
        output_path = output_dir / f"frame_{index:04d}.{suffix}"
        if args.format == "jpg":
            frame.save(output_path, quality=args.jpeg_quality)
        else:
            frame.save(output_path)

    print(output_dir)


if __name__ == "__main__":
    main()
