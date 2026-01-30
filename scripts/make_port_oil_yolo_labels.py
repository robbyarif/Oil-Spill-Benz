from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OIL_CLASS_ID = 0


def mask_to_yolo_polygons(mask_path: Path) -> list[str]:
    """Convert binary mask to YOLO polygon format lines."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    
    height, width = img.shape
    
    # Find contours (oil regions = 255)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Flatten contour and normalize coordinates
        points = contour.squeeze()
        if points.ndim == 1:  # Single point
            continue
        
        # Normalize to [0, 1]
        normalized = []
        for pt in points:
            x_norm = pt[0] / width
            y_norm = pt[1] / height
            normalized.append(f"{x_norm:.6f}")
            normalized.append(f"{y_norm:.6f}")
        
        # Build YOLO line: class_id x1 y1 x2 y2 ...
        line = f"{OIL_CLASS_ID} " + " ".join(normalized)
        lines.append(line)
    
    return lines


def process_split(src_root: Path, split: str) -> int:
    """Process one split (train/val/test)."""
    masks_dir = src_root / split / "masks"
    labels_dir = src_root / split / "labels"
    
    if not masks_dir.exists():
        print(f"skip (missing): {masks_dir}")
        return 0
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for mask_file in sorted(masks_dir.glob("*.png")):
        lines = mask_to_yolo_polygons(mask_file)
        if not lines:
            print(f"skip (no contours): {mask_file}")
            continue
        
        label_file = labels_dir / f"{mask_file.stem}.txt"
        label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        count += 1
    
    return count


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "datasets" / "processed" / "port-oil"
    
    total = 0
    for split in ["train", "val", "test"]:
        count = process_split(src_root, split)
        print(f"{split}: {count} labels")
        total += count
    
    print(f"total labels: {total}")


if __name__ == "__main__":
    main()
