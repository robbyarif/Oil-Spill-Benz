#!/usr/bin/env python3
"""
Resize dataset images to a fixed size and copy to a new folder.
Preserves the split structure (train/valid/test) and labels.

Example:
  python scripts/resize_images_dataset.py \
    datasets/processed/exp2.4-dv3train_organized \
    datasets/processed/exp2.4-dv3train_organized_yolo_resized \
    --size 312 312
"""

import argparse
from pathlib import Path
import shutil
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resize_image(src_path: Path, dst_path: Path, size: tuple[int, int]) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        img.save(dst_path, quality=95)


def copy_labels(src_labels_dir: Path, dst_labels_dir: Path) -> None:
    if not src_labels_dir.exists():
        return
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    for label_file in src_labels_dir.iterdir():
        if label_file.is_file() and label_file.suffix.lower() == ".txt":
            shutil.copy2(label_file, dst_labels_dir / label_file.name)


def process_split(src_split: Path, dst_split: Path, size: tuple[int, int]) -> None:
    src_images = src_split / "images"
    src_labels = src_split / "labels"

    dst_images = dst_split / "images"
    dst_labels = dst_split / "labels"

    if not src_images.exists():
        return

    for img_path in src_images.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        dst_path = dst_images / img_path.name
        resize_image(img_path, dst_path, size)

    copy_labels(src_labels, dst_labels)


def resize_dataset(src_root: Path, dst_root: Path, size: tuple[int, int]) -> None:
    splits = ["train", "valid", "test"]
    for split in splits:
        process_split(src_root / split, dst_root / split, size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize dataset images to a fixed size")
    parser.add_argument("src", type=str, help="Source dataset root folder")
    parser.add_argument("dst", type=str, help="Destination dataset root folder")
    parser.add_argument("--size", nargs=2, type=int, default=[312, 312], help="Target size: width height")

    args = parser.parse_args()
    size = (args.size[0], args.size[1])

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)
    resize_dataset(src_root, dst_root, size)

    print(f"Done. Resized images saved to: {dst_root}")


if __name__ == "__main__":
    main()
