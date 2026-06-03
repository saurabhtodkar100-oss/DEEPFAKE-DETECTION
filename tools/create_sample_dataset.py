from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset for smoke-testing the portal.")
    parser.add_argument("--output-dir", default="dataset", help="Dataset root to create")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size in pixels")
    parser.add_argument("--train-count", type=int, default=12, help="Images per class in train split")
    parser.add_argument("--val-count", type=int, default=4, help="Images per class in val split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


def write_bmp(path: Path, pixels: list[tuple[int, int, int]], width: int, height: int) -> None:
    row_stride = width * 3
    padding = (4 - (row_stride % 4)) % 4
    pixel_array_size = (row_stride + padding) * height
    file_size = 14 + 40 + pixel_array_size

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"BM")
        handle.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        handle.write(struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_array_size, 2835, 2835, 0, 0))

        pad = b"\x00" * padding
        for row in range(height - 1, -1, -1):
            start = row * width
            end = start + width
            for red, green, blue in pixels[start:end]:
                handle.write(struct.pack("BBB", blue, green, red))
            handle.write(pad)


def smooth_wave(x: float, y: float, variant: int) -> float:
    return (
        math.sin((x + variant * 0.07) * math.pi * 1.3)
        + math.cos((y + variant * 0.11) * math.pi * 1.7)
        + math.sin((x + y) * math.pi * 0.8)
    )


def make_real_pixels(size: int, variant: int) -> list[tuple[int, int, int]]:
    pixels: list[tuple[int, int, int]] = []
    cx = 0.28 + (variant % 4) * 0.12
    cy = 0.32 + ((variant // 2) % 4) * 0.09
    radius = 0.16 + (variant % 3) * 0.03

    for y in range(size):
        yn = y / max(1, size - 1)
        for x in range(size):
            xn = x / max(1, size - 1)
            wave = smooth_wave(xn, yn, variant)
            dist = math.sqrt((xn - cx) ** 2 + (yn - cy) ** 2)
            glow = max(0.0, 1.0 - dist / radius)

            red = 80 + 110 * xn + 18 * wave + 55 * glow
            green = 70 + 95 * yn + 15 * math.cos(xn * math.pi * 2.0) + 35 * glow
            blue = 95 + 70 * (1.0 - yn) + 12 * math.sin(yn * math.pi * 3.0) + 25 * glow
            pixels.append((clamp(red), clamp(green), clamp(blue)))

    return pixels


def make_fake_pixels(size: int, variant: int) -> list[tuple[int, int, int]]:
    base = make_real_pixels(size, variant)
    pixels: list[tuple[int, int, int]] = []
    block = 8 + (variant % 4) * 4
    shift = 6 + (variant % 5) * 2

    for y in range(size):
        for x in range(size):
            block_x = (x // block) * block
            block_y = (y // block) * block
            src_index = min(size - 1, block_y + (variant % block)) * size + min(size - 1, block_x + (variant * 3) % block)
            red, green, blue = base[src_index]

            if ((x // shift) + (y // shift)) % 2 == 0:
                red += 26
                blue -= 18
            if y % (14 + variant % 6) < 2:
                green += 22
            if x % (18 + variant % 7) < 2:
                red -= 20

            quantize = 24 + (variant % 3) * 8
            red = (red // quantize) * quantize
            green = (green // quantize) * quantize
            blue = (blue // quantize) * quantize
            pixels.append((clamp(red), clamp(green), clamp(blue)))

    return pixels


def generate_split(root: Path, split: str, label: str, count: int, size: int, seed: int) -> None:
    generator = random.Random(seed)
    split_dir = root / split / label
    split_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        variant = generator.randint(0, 10_000)
        if label == "real":
            pixels = make_real_pixels(size, variant)
        else:
            pixels = make_fake_pixels(size, variant)

        file_path = split_dir / f"{label}_{index + 1:03d}.bmp"
        write_bmp(file_path, pixels, size, size)


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)

    generate_split(root, "train", "real", args.train_count, args.image_size, args.seed + 10)
    generate_split(root, "train", "fake", args.train_count, args.image_size, args.seed + 20)
    generate_split(root, "val", "real", args.val_count, args.image_size, args.seed + 30)
    generate_split(root, "val", "fake", args.val_count, args.image_size, args.seed + 40)

    print(f"Sample dataset created at: {root.resolve()}")
    print("This dataset is synthetic and only intended for smoke tests.")


if __name__ == "__main__":
    main()
