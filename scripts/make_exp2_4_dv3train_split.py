#!/usr/bin/env python3
"""Generate train/val/test splits for exp2.4-dv3train with label validation."""

from pathlib import Path
import sys

# Configuration
TRAIN_ROOT = Path("/home/robby/workspace/Oil-Spill-Benz/datasets/processed/dv3-by-date")
VAL_ROOTS = [
    Path("/home/robby/workspace/Oil-Spill-Benz/DV4/images/20240725_喀麥隆籍「多芬DOLPHIN」貨輪擱淺漏油案")
]
TEST_ROOTS = [
    Path("/home/robby/workspace/Oil-Spill-Benz/DV4/images/20240725_多哥籍「阿諾(ALANO)」貨輪疑似流錨協處案"),
    Path("/home/robby/workspace/Oil-Spill-Benz/DV4/images/20240725_蒙古籍「凱塔(KETA)」貨輪發電機故障漏油案"),
    Path("/home/robby/workspace/Oil-Spill-Benz/DV4/images/20241208_巴拿馬籍「液態寶石」油輪擱淺案"),
]
OUTPUT_ROOT = Path("/home/robby/workspace/Oil-Spill-Benz/datasets/processed/exp2.4-dv3train")
ALLOWED_IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_label_for_image(img_path: Path) -> Path | None:
    """Find corresponding label file for an image."""
    # For dv3-by-date: images/labels are in same parent dir (YYYYMMDD)
    if "dv3-by-date" in str(img_path):
        label_path = img_path.parent.parent / "labels" / img_path.name
        label_path = label_path.with_suffix(".txt")
    # For DV4: images and labels are in separate root dirs
    else:
        # Get relative path from DV4/images/
        rel_path = img_path.relative_to(Path("/home/robby/workspace/Oil-Spill-Benz/DV4/images"))
        label_path = Path("/home/robby/workspace/Oil-Spill-Benz/DV4/labels") / rel_path
        label_path = label_path.with_suffix(".txt")
    
    if label_path.exists():
        return label_path
    return None


def collect_images_with_labels(root: Path) -> list[Path]:
    """Collect all images in root that have corresponding labels."""
    images = []
    
    if not root.exists():
        print(f"Warning: {root} does not exist")
        return images
    
    for img_path in sorted(root.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in ALLOWED_IMG_EXT:
            if find_label_for_image(img_path):
                images.append(img_path)
    
    return images


def main():
    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Collect train images
    print("Collecting train images...")
    train_images = collect_images_with_labels(TRAIN_ROOT)
    print(f"  Found {len(train_images)} train images with labels")
    
    # Collect val images
    print("Collecting val images...")
    val_images = []
    for val_root in VAL_ROOTS:
        val_images.extend(collect_images_with_labels(val_root))
    print(f"  Found {len(val_images)} val images with labels")
    
    # Collect test images
    print("Collecting test images...")
    test_images = []
    for test_root in TEST_ROOTS:
        test_images.extend(collect_images_with_labels(test_root))
    print(f"  Found {len(test_images)} test images with labels")
    
    # Write split files
    print("\nWriting split files...")
    
    train_file = OUTPUT_ROOT / "train.txt"
    with open(train_file, "w") as f:
        for img_path in train_images:
            f.write(str(img_path) + "\n")
    print(f"  train.txt: {len(train_images)} images")
    
    val_file = OUTPUT_ROOT / "val.txt"
    with open(val_file, "w") as f:
        for img_path in val_images:
            f.write(str(img_path) + "\n")
    print(f"  val.txt: {len(val_images)} images")
    
    test_file = OUTPUT_ROOT / "test.txt"
    with open(test_file, "w") as f:
        for img_path in test_images:
            f.write(str(img_path) + "\n")
    print(f"  test.txt: {len(test_images)} images")
    
    total = len(train_images) + len(val_images) + len(test_images)
    print(f"\nTotal: {total} images")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
