from __future__ import annotations

from pathlib import Path

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_images(images_dir: Path) -> list[Path]:
    """Collect all image files from a directory."""
    images = []
    for path in sorted(images_dir.glob("*")):
        if path.is_file() and path.suffix in ALLOWED_IMAGE_EXT:
            images.append(path)
    return images


def write_split(images: list[Path], dst: Path, repo_root: Path) -> None:
    """Write image paths to split file (relative to repo root)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for img_path in images:
            rel = img_path.relative_to(repo_root).as_posix()
            f.write(f"{rel}\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "datasets" / "processed" / "port-oil"
    dst_root = repo_root / "datasets" / "processed" / "exp2.2_port-oil-split"
    
    splits = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
    }
    
    for split_name, output_name in splits.items():
        images_dir = src_root / split_name / "images"
        if not images_dir.exists():
            print(f"skip (missing): {images_dir}")
            continue
        
        images = collect_images(images_dir)
        write_split(images, dst_root / output_name, repo_root)
        print(f"{split_name}: {len(images)} images")


if __name__ == "__main__":
    main()
