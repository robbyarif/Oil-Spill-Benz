from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw

ALLOWED_IMAGE_EXT = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate segmentation masks (PNG) from YOLO polygon labels.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root containing YOLO txt labels (default: this folder).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "images",
        help="Root containing source images (default: DV4/images relative to this file).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination root for mask PNGs (default: same tree as labels).",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional txt listing image paths (relative to repo root) to limit processing.",
    )
    parser.add_argument(
        "--class-offset",
        type=int,
        default=0,
        help="Value added to class id before scaling (keeps background at 0 by default).",
    )
    parser.add_argument(
        "--fill-scale",
        type=int,
        default=255,
        help="Multiplier applied to (class_id + class_offset) so foreground renders white in viewers.",
    )
    return parser.parse_args()


def load_split(split_file: Path, repo_root: Path) -> set[Path]:
    lines = split_file.read_text(encoding="utf-8").splitlines()
    return {repo_root / line.strip() for line in lines if line.strip()}


def find_image(label_path: Path, labels_root: Path, images_root: Path) -> Path | None:
    rel = label_path.relative_to(labels_root).with_suffix("")
    for ext in ALLOWED_IMAGE_EXT:
        candidate = images_root / rel.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def parse_label_line(line: str) -> Tuple[int, List[float]] | None:
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    return class_id, coords


def draw_polygons(
    mask: Image.Image,
    polygons: Iterable[Tuple[int, Sequence[float]]],
    class_offset: int,
    fill_scale: int,
) -> None:
    width, height = mask.size
    draw = ImageDraw.Draw(mask)
    for class_id, coords in polygons:
        points: List[Tuple[float, float]] = []
        for x, y in zip(coords[0::2], coords[1::2]):
            points.append((x * width, y * height))
        if len(points) >= 3:
            draw.polygon(points, fill=255)


def process_label(
    label_path: Path,
    labels_root: Path,
    images_root: Path,
    output_root: Path,
    allowed_images: set[Path] | None,
    class_offset: int,
    fill_scale: int,
) -> bool:
    image_path = find_image(label_path, labels_root, images_root)
    if image_path is None:
        print(f"skip (no image): {label_path}")
        return False

    if allowed_images is not None and image_path not in allowed_images:
        return False

    raw_lines = label_path.read_text(encoding="utf-8").splitlines()
    parsed = [parse_label_line(line) for line in raw_lines]
    polygons = [p for p in parsed if p is not None]
    if not polygons:
        print(f"skip (invalid or empty): {label_path}")
        return False

    with Image.open(image_path) as img:
        width, height = img.size

    mask = Image.new("L", (width, height), color=0)
    draw_polygons(mask, polygons, class_offset, fill_scale)

    rel = label_path.relative_to(labels_root).with_suffix(".png")
    dst = output_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    mask.save(dst)
    return True


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    labels_root = args.labels_root.resolve()
    images_root = args.images_root.resolve()
    output_root = args.output_root.resolve()

    allowed_images: set[Path] | None = None
    if args.split_file:
        allowed_images = load_split(args.split_file.resolve(), repo_root)

    count = 0
    for label_path in labels_root.rglob("*.txt"):
        if process_label(
            label_path,
            labels_root,
            images_root,
            output_root,
            allowed_images,
            args.class_offset,
            args.fill_scale,
        ):
            count += 1
    print(f"masks written: {count}")


if __name__ == "__main__":
    main()
