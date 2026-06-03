from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CONFIG_KEYS = {
    "real_image_dirs",
    "fake_image_dirs",
    "real_video_dirs",
    "fake_video_dirs",
}


@dataclass(frozen=True)
class MediaSource:
    label: str
    media_type: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a larger real/fake dataset from raw image and video folders."
    )
    parser.add_argument("--config", help="Optional JSON config listing input directories")
    parser.add_argument("--real-image-dir", action="append", default=[], help="Directory of authentic images")
    parser.add_argument("--fake-image-dir", action="append", default=[], help="Directory of manipulated or AI images")
    parser.add_argument("--real-video-dir", action="append", default=[], help="Directory of authentic videos")
    parser.add_argument("--fake-video-dir", action="append", default=[], help="Directory of manipulated or AI videos")
    parser.add_argument("--output-root", default="dataset_large", help="Output dataset root")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--frames-per-video", type=int, default=12, help="Frames to extract from each video")
    parser.add_argument("--blur-threshold", type=float, default=40.0, help="Minimum Laplacian variance for saved frames")
    parser.add_argument("--min-dimension", type=int, default=160, help="Minimum height/width required for saved media")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink", help="How to materialize image files")
    parser.add_argument("--face-crop-videos", action="store_true", help="Extract the largest detected face instead of full frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_source_config(config_path: Path | None) -> dict[str, list[str]]:
    payload = {key: [] for key in CONFIG_KEYS}
    if config_path is None:
        return payload

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    for key in CONFIG_KEYS:
        values = raw.get(key, [])
        if isinstance(values, list):
            payload[key] = [str(item) for item in values if item]
    return payload


def unique_paths(paths: Iterable[str]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for item in paths:
        candidate = str(Path(item).expanduser())
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(Path(candidate))
    return result


def collect_media_files(roots: Iterable[Path], extensions: set[str]) -> list[Path]:
    collected: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"Skipping missing source directory: {root}")
            continue

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                collected.append(path)
    return sorted(set(collected))


def validation_count(total: int, val_ratio: float) -> int:
    if total <= 1 or val_ratio <= 0:
        return 0
    proposed = int(round(total * val_ratio))
    return max(1, min(total - 1, proposed))


def split_sources(paths: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    items = list(paths)
    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = validation_count(len(items), val_ratio)
    val_items = items[:val_count]
    train_items = items[val_count:]
    return train_items, val_items


def file_token(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "sample"
    return f"{stem[:48]}_{digest}"


def ensure_link_or_copy(source: Path, target: Path, copy_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return

    if copy_mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass

    shutil.copy2(source, target)


def blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def largest_face_crop(frame: np.ndarray, detector: cv2.CascadeClassifier) -> np.ndarray:
    if detector.empty():
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return frame

    x, y, w, h = max(faces, key=lambda box: int(box[2] * box[3]))
    pad_x = int(w * 0.18)
    pad_y = int(h * 0.18)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame.shape[1], x + w + pad_x)
    y1 = min(frame.shape[0], y + h + pad_y)
    cropped = frame[y0:y1, x0:x1]
    return cropped if cropped.size else frame


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    prefix: str,
    frames_per_video: int,
    blur_threshold: float,
    min_dimension: int,
    face_crop: bool,
    face_detector: cv2.CascadeClassifier,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return 0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    candidate_count = max(frames_per_video * 4, frames_per_video)
    if total_frames > 1:
        frame_positions = np.linspace(0, total_frames - 1, num=min(total_frames, candidate_count), dtype=int).tolist()
    else:
        frame_positions = list(range(candidate_count))

    saved = 0
    visited: set[int] = set()
    for position in frame_positions:
        if position in visited:
            continue
        visited.add(position)

        if total_frames > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)

        success, frame = capture.read()
        if not success or frame is None or frame.size == 0:
            continue

        if face_crop:
            frame = largest_face_crop(frame, face_detector)

        height, width = frame.shape[:2]
        if min(height, width) < min_dimension:
            continue
        if blur_score(frame) < blur_threshold:
            continue

        target = output_dir / f"{prefix}_{saved + 1:02d}.jpg"
        ok = cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            continue

        saved += 1
        if saved >= frames_per_video:
            break

    capture.release()
    return saved


def materialize_split(
    sources: list[MediaSource],
    split_name: str,
    output_root: Path,
    copy_mode: str,
    frames_per_video: int,
    blur_threshold: float,
    min_dimension: int,
    face_crop_videos: bool,
    face_detector: cv2.CascadeClassifier,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        "real": {"images": 0, "frames": 0, "videos": 0},
        "fake": {"images": 0, "frames": 0, "videos": 0},
    }

    for source in sources:
        class_dir = output_root / split_name / source.label
        token = file_token(source.path)
        if source.media_type == "image":
            target = class_dir / f"{source.label}_{token}{source.path.suffix.lower()}"
            ensure_link_or_copy(source.path, target, copy_mode)
            counts[source.label]["images"] += 1
            continue

        counts[source.label]["videos"] += 1
        saved = extract_video_frames(
            video_path=source.path,
            output_dir=class_dir,
            prefix=f"{source.label}_{token}",
            frames_per_video=frames_per_video,
            blur_threshold=blur_threshold,
            min_dimension=min_dimension,
            face_crop=face_crop_videos,
            face_detector=face_detector,
        )
        counts[source.label]["frames"] += saved

    return counts


def summarize_sources(items: Iterable[MediaSource]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        summary[item.label][item.media_type] += 1
    return {label: dict(values) for label, values in summary.items()}


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0.0 and 1.0.")

    config = load_source_config(Path(args.config)) if args.config else load_source_config(None)
    real_image_roots = unique_paths([*config["real_image_dirs"], *args.real_image_dir])
    fake_image_roots = unique_paths([*config["fake_image_dirs"], *args.fake_image_dir])
    real_video_roots = unique_paths([*config["real_video_dirs"], *args.real_video_dir])
    fake_video_roots = unique_paths([*config["fake_video_dirs"], *args.fake_video_dir])

    real_images = collect_media_files(real_image_roots, IMAGE_EXTENSIONS)
    fake_images = collect_media_files(fake_image_roots, IMAGE_EXTENSIONS)
    real_videos = collect_media_files(real_video_roots, VIDEO_EXTENSIONS)
    fake_videos = collect_media_files(fake_video_roots, VIDEO_EXTENSIONS)

    if not any([real_images, fake_images, real_videos, fake_videos]):
        raise RuntimeError("No input media found. Provide source directories with --*-dir flags or a config file.")

    train_real_images, val_real_images = split_sources(real_images, args.val_ratio, args.seed)
    train_fake_images, val_fake_images = split_sources(fake_images, args.val_ratio, args.seed + 1)
    train_real_videos, val_real_videos = split_sources(real_videos, args.val_ratio, args.seed + 2)
    train_fake_videos, val_fake_videos = split_sources(fake_videos, args.val_ratio, args.seed + 3)

    train_sources = [
        *[MediaSource("real", "image", path) for path in train_real_images],
        *[MediaSource("fake", "image", path) for path in train_fake_images],
        *[MediaSource("real", "video", path) for path in train_real_videos],
        *[MediaSource("fake", "video", path) for path in train_fake_videos],
    ]
    val_sources = [
        *[MediaSource("real", "image", path) for path in val_real_images],
        *[MediaSource("fake", "image", path) for path in val_fake_images],
        *[MediaSource("real", "video", path) for path in val_real_videos],
        *[MediaSource("fake", "video", path) for path in val_fake_videos],
    ]

    output_root = Path(args.output_root)
    for split in ("train", "val"):
        for label in ("real", "fake"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    train_counts = materialize_split(
        sources=train_sources,
        split_name="train",
        output_root=output_root,
        copy_mode=args.copy_mode,
        frames_per_video=args.frames_per_video,
        blur_threshold=args.blur_threshold,
        min_dimension=args.min_dimension,
        face_crop_videos=args.face_crop_videos,
        face_detector=face_detector,
    )
    val_counts = materialize_split(
        sources=val_sources,
        split_name="val",
        output_root=output_root,
        copy_mode=args.copy_mode,
        frames_per_video=args.frames_per_video,
        blur_threshold=args.blur_threshold,
        min_dimension=args.min_dimension,
        face_crop_videos=args.face_crop_videos,
        face_detector=face_detector,
    )

    manifest = {
        "output_root": str(output_root.resolve()),
        "copy_mode": args.copy_mode,
        "frames_per_video": args.frames_per_video,
        "blur_threshold": args.blur_threshold,
        "min_dimension": args.min_dimension,
        "face_crop_videos": args.face_crop_videos,
        "sources": {
            "train": summarize_sources(train_sources),
            "val": summarize_sources(val_sources),
        },
        "materialized": {
            "train": train_counts,
            "val": val_counts,
        },
        "config": {
            "real_image_dirs": [str(path) for path in real_image_roots],
            "fake_image_dirs": [str(path) for path in fake_image_roots],
            "real_video_dirs": [str(path) for path in real_video_roots],
            "fake_video_dirs": [str(path) for path in fake_video_roots],
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Prepared dataset at: {output_root.resolve()}")
    print(f"Manifest written to: {manifest_path.resolve()}")
    print(json.dumps(manifest["materialized"], indent=2))


if __name__ == "__main__":
    main()
