from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# RGB color for oil class
OIL_RGB = (255, 0, 124)


def process_mask(src_mask: Path, dst_mask: Path) -> None:
    """Convert RGB mask to binary grayscale (oil=255, other=0)."""
    img = Image.open(src_mask).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    
    # Create binary mask: True where all channels match oil RGB
    oil_mask = (
        (arr[:, :, 0] == OIL_RGB[0])
        & (arr[:, :, 1] == OIL_RGB[1])
        & (arr[:, :, 2] == OIL_RGB[2])
    )
    
    # Convert to uint8: oil=255, background=0
    binary = (oil_mask * 255).astype(np.uint8)
    
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode="L").save(dst_mask)


def copy_image(src_img: Path, dst_img: Path) -> None:
    """Copy image to destination."""
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)


def process_split(src_root: Path, dst_root: Path, split: str) -> int:
    """Process one split (train/val/test)."""
    src_masks = src_root / split / "masks"
    src_images = src_root / split / "images"
    dst_masks = dst_root / split / "masks"
    dst_images = dst_root / split / "images"
    
    if not src_masks.exists():
        print(f"skip (missing): {src_masks}")
        return 0
    
    count = 0
    for mask_file in sorted(src_masks.glob("*")):
        if not mask_file.is_file():
            continue
        
        stem = mask_file.stem
        # Find matching image
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = src_images / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file is None:
            print(f"skip (no image): {mask_file}")
            continue
        
        # Process mask
        dst_mask_file = dst_masks / f"{stem}.png"
        process_mask(mask_file, dst_mask_file)
        
        # Copy image
        dst_img_file = dst_images / img_file.name
        copy_image(img_file, dst_img_file)
        
        count += 1
    
    return count


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "datasets" / "raw" / "port-oil"
    dst_root = repo_root / "datasets" / "processed" / "port-oil"
    
    total = 0
    for split in ["train", "val", "test"]:
        count = process_split(src_root, dst_root, split)
        print(f"{split}: {count} pairs")
        total += count
    
    print(f"total processed: {total}")


if __name__ == "__main__":
    main()
