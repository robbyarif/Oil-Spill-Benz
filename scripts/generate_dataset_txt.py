#!/usr/bin/env python3
"""Generate train.txt, valid.txt, and test.txt files for YOLO datasets."""

import os
import argparse

def create_dataset_txt_files(output_dir="."):
    """Create train.txt, valid.txt, and test.txt files with absolute image paths."""
    split_names = ["train", "test", "valid"]
    
    for split_name in split_names:
        split_images_dir = os.path.join(output_dir, split_name, "images")
        txt_file_path = os.path.join(output_dir, f"{split_name}.txt")
        
        if not os.path.isdir(split_images_dir):
            print(f"Skip {split_name}.txt: {split_images_dir} does not exist")
            continue
        
        # Get all image files
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_files = sorted(
            os.path.join(split_images_dir, name)
            for name in os.listdir(split_images_dir)
            if os.path.splitext(name)[1].lower() in valid_exts
        )
        
        if not image_files:
            print(f"No images found in {split_images_dir}")
            continue
        
        # Write absolute paths to txt file
        with open(txt_file_path, 'w') as f:
            for image_path in image_files:
                abs_path = os.path.abspath(image_path)
                f.write(abs_path + "\n")
        
        print(f"Created {txt_file_path} with {len(image_files)} images")

def main():
    parser = argparse.ArgumentParser(
        description="Generate train.txt, valid.txt, and test.txt files for YOLO dataset"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory containing train/, test/, valid/ subdirectories",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise NotADirectoryError(f"Invalid output directory: {args.output_dir}")

    create_dataset_txt_files(args.output_dir)

if __name__ == "__main__":
    main()
