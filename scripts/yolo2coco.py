import json
import os
import cv2
import shutil
import numpy as np

# --- CONFIGURATION ---
CLASSES = ["Oil Spill"]  # Update this list!
# BASE_DIR = "/home/robby/workspace/Oil-Spill-Benz/datasets/processed/exp2.4-dv3train_organized"  # Base dataset directory
BASE_DIR = "/home/robby/workspace/Oil-Spill-Benz/datasets/processed/dv4_random_split"  # Base dataset directory
SPLITS = ["train", "test", "valid"]  # Splits to process
# OUTPUT_DIR = "/home/robby/workspace/Oil-Spill-Benz/datasets/processed/exp2.4-dv3train_organized_coco"  # Output directory for JSON files
OUTPUT_DIR = "/home/robby/workspace/Oil-Spill-Benz/datasets/processed/dv4_random_split_coco_312"  # Output directory for JSON files
TARGET_SIZE = (312, 312)  # Target size for resizing (width, height), e.g., (640, 640). Set to None to keep original size
# ---------------------

def get_split_paths(base_dir, split_name):
    """
    Get labels and images directories for a split.
    Supports both 'val' and 'valid' directory names.
    """
    # Try the standard split name first, then alternative
    split_dir = os.path.join(base_dir, split_name)
    if split_name == "val":
        alternative_split_dir = os.path.join(base_dir, "valid")
    elif split_name == "valid":
        alternative_split_dir = os.path.join(base_dir, "val")
    else:
        alternative_split_dir = split_dir
    
    # Check which one exists
    if os.path.exists(split_dir):
        split_dir = split_dir
    elif os.path.exists(alternative_split_dir):
        split_dir = alternative_split_dir
    
    labels_dir = os.path.join(split_dir, "labels")
    images_dir = os.path.join(split_dir, "images")
    
    return labels_dir, images_dir

def yolo_seg_to_coco(labels_dir, images_dir, split_name, target_size=None):
    """
    Convert YOLO segmentation format to COCO format for a single split
    
    Args:
        labels_dir: Path to labels directory (e.g., labels/train)
        images_dir: Path to images directory (e.g., images/train)
        split_name: Name of the split (train/test/val)
        target_size: Optional tuple (width, height) to resize images and scale coordinates
    
    Returns:
        COCO format dictionary with image metadata for resizing
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [],
        "resize_metadata": []  # Store original sizes for resizing during copy
    }

    # 1. Create Categories
    for i, class_name in enumerate(CLASSES):
        coco_format["categories"].append({
            "id": i, 
            "name": class_name,
            "supercategory": "none"
        })

    annotation_id = 0
    image_id = 0

    if not os.path.exists(labels_dir):
        print(f"Warning: Labels directory '{labels_dir}' does not exist. Skipping {split_name}.")
        return None

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"Processing {split_name} split: {len(label_files)} label files found")

    for label_file in label_files:
        image_id += 1
        base_name = os.path.splitext(label_file)[0]
        
        # Find image
        image_filename = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if os.path.exists(os.path.join(images_dir, base_name + ext)):
                image_filename = base_name + ext
                break
        
        if not image_filename:
            print(f"  Warning: No image found for {label_file}")
            continue

        # Get Image Dimensions
        img_path = os.path.join(images_dir, image_filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not read image {img_path}")
            continue
        
        # Get original dimensions
        orig_height, orig_width = img.shape[:2]
        
        # Determine final dimensions (after potential resize)
        if target_size is not None:
            width, height = target_size
            scale_x = width / orig_width
            scale_y = height / orig_height
        else:
            width, height = orig_width, orig_height
            scale_x = 1.0
            scale_y = 1.0
        
        # Store metadata for image resizing during copy phase
        coco_format["resize_metadata"].append({
            "image_id": image_id,
            "orig_width": orig_width,
            "orig_height": orig_height,
            "target_width": width,
            "target_height": height,
            "file_name": image_filename
        })

        coco_format["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        # Process Polygons
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # If line has fewer than 6 numbers (class + 2 points), it's not a polygon
                if len(coords) < 4: 
                    continue

                # Denormalize Polygon Coordinates (0-1 to pixels)
                # Apply scaling if resizing is enabled
                poly_pixels = []
                for i in range(0, len(coords), 2):
                    x = coords[i] * orig_width * scale_x
                    y = coords[i+1] * orig_height * scale_y
                    poly_pixels.append(x)
                    poly_pixels.append(y)

                # 1. Calculate Bounding Box from Polygon
                # reshape to standard list of (x,y) for min/max
                poly_np = np.array(poly_pixels).reshape(-1, 2)
                x_min = np.min(poly_np[:, 0])
                y_min = np.min(poly_np[:, 1])
                x_max = np.max(poly_np[:, 0])
                y_max = np.max(poly_np[:, 1])
                w_box = x_max - x_min
                h_box = y_max - y_min

                # 2. Calculate Area (Required for COCO)
                # We use OpenCV's contourArea for accuracy
                poly_np_int = poly_np.astype(np.int32)
                area = cv2.contourArea(poly_np_int)

                annotation_id += 1
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": [poly_pixels], # COCO expects a list of lists
                    "bbox": [x_min, y_min, w_box, h_box],
                    "area": area,
                    "iscrowd": 0
                })

    return coco_format

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each split
    for split in SPLITS:
        labels_dir, images_dir = get_split_paths(BASE_DIR, split)
        
        print(f"\n{'='*60}")
        print(f"Converting {split.upper()} split...")
        if TARGET_SIZE is not None:
            print(f"Resizing images to: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} pixels")
        else:
            print(f"Using original image sizes (no resizing)")
        print(f"{'='*60}")
        
        coco_data = yolo_seg_to_coco(labels_dir, images_dir, split, TARGET_SIZE)
        
        if coco_data is None:
            continue
        
        # Create split-specific output directory
        split_output_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Copy/resize images to the output directory
        if TARGET_SIZE is not None:
            print(f"Resizing and copying images to {split_output_dir}...")
        else:
            print(f"Copying original images to {split_output_dir}...")
        
        for i, metadata in enumerate(coco_data["resize_metadata"]):
            src_image_path = os.path.join(images_dir, metadata["file_name"])
            dst_image_path = os.path.join(split_output_dir, metadata["file_name"])
            
            if os.path.exists(src_image_path):
                try:
                    if TARGET_SIZE is not None:
                        # Read and resize the image
                        img = cv2.imread(src_image_path)
                        if img is not None:
                            resized_img = cv2.resize(img, (metadata["target_width"], metadata["target_height"]), 
                                                    interpolation=cv2.INTER_LINEAR)
                            cv2.imwrite(dst_image_path, resized_img)
                        else:
                            print(f"  Warning: Could not read image {src_image_path}")
                    else:
                        # Copy the original image without resizing
                        shutil.copy2(src_image_path, dst_image_path)
                except Exception as e:
                    print(f"  Warning: Failed to process {src_image_path} -> {dst_image_path}: {e}")
            else:
                print(f"  Warning: Could not find image {src_image_path}")
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(coco_data['resize_metadata'])} images")
        
        # Remove resize_metadata before saving (not part of COCO standard)
        coco_data_clean = {
            "images": coco_data["images"],
            "annotations": coco_data["annotations"],
            "categories": coco_data["categories"]
        }
        
        # Save to JSON with COCO naming convention
        output_file = os.path.join(split_output_dir, f"_annotations.coco.json")
        with open(output_file, 'w') as f:
            json.dump(coco_data_clean, f, indent=2)
        
        print(f"✓ Saved {output_file}")
        print(f"  - Images: {len(coco_data['images'])} copied")
        print(f"  - Annotations: {len(coco_data['annotations'])}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete! All COCO data saved to '{OUTPUT_DIR}/'")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()