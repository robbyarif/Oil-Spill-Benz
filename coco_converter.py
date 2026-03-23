"""COCO dataset converter helper for YOLO-annotated datasets."""

import json
import os
import shutil
from typing import Optional

import numpy as np

from utils import image2label, read_img, yolo_label2contours


def prepare_coco_dataset(
    src: str,
    output_dir: Optional[str] = None,
    categories: Optional[list] = None,
) -> str:
    """
    Convert a dataset with train.txt, val.txt, test.txt into COCO format.
    
    Args:
        src: Source directory containing txt files listing image paths
        output_dir: Output directory for COCO format (default: {src}/coco_format_temp)
        categories: List of category dicts [{"id": 0, "name": "oil", ...}]
    
    Returns:
        Path to COCO format directory
    """
    if output_dir is None:
        output_dir = os.path.join(src, "coco_format_temp")
    
    if categories is None:
        categories = [{"id": 0, "name": "oil", "supercategory": "oil"}]
    
    # If already COCO format, return directly
    if os.path.exists(os.path.join(src, "train", "_annotations.coco.json")):
        return src
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        txt_file = os.path.join(src, f"{split}.txt")
        if not os.path.exists(txt_file):
            continue
        
        # Map val -> valid for consistency with some frameworks
        folder_name = "valid" if split == "val" else split
        split_dir = os.path.join(output_dir, folder_name)
        os.makedirs(split_dir, exist_ok=True)
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        
        anno_id = 1
        with open(txt_file, "r", encoding="utf-8") as f:
            img_paths = [line.strip() for line in f if line.strip()]
        
        for img_id, img_path in enumerate(img_paths, start=1):
            abs_img_path = os.path.abspath(img_path)
            if not os.path.exists(abs_img_path):
                continue
            
            base_name = os.path.basename(abs_img_path)
            dst_img = os.path.join(split_dir, base_name)
            
            # Read image to get dimensions
            img = read_img(abs_img_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Copy image to split directory
            shutil.copy2(abs_img_path, dst_img)
            
            # Add image metadata
            coco_data["images"].append({
                "id": img_id,
                "file_name": base_name,
                "height": int(h),
                "width": int(w),
            })
            
            # Process labels
            label_path = image2label(abs_img_path, lbl_ext=".txt")
            if os.path.exists(label_path):
                contours = yolo_label2contours(label_path, (h, w))
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    
                    seg = contour.flatten().tolist()
                    x_min = float(np.min(contour[:, 0]))
                    y_min = float(np.min(contour[:, 1]))
                    x_max = float(np.max(contour[:, 0]))
                    y_max = float(np.max(contour[:, 1]))
                    bw = x_max - x_min
                    bh = y_max - y_min
                    
                    coco_data["annotations"].append({
                        "id": anno_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "segmentation": [seg],
                        "bbox": [x_min, y_min, bw, bh],
                        "iscrowd": 0,
                        "area": float(bw * bh),
                    })
                    anno_id += 1
        
        # Write COCO annotations
        out_json = os.path.join(split_dir, "_annotations.coco.json")
        with open(out_json, "w") as f:
            json.dump(coco_data, f)
    
    # Fallback: symlink valid to test if test doesn't exist
    test_dir = os.path.join(output_dir, "test")
    valid_dir = os.path.join(output_dir, "valid")
    if not os.path.exists(test_dir) and os.path.exists(valid_dir):
        os.symlink(os.path.abspath(valid_dir), os.path.abspath(test_dir))
    
    return output_dir
