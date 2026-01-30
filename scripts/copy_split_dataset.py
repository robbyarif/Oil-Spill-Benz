#!/usr/bin/env python3
"""
Copy dataset files according to split information (train/test/val).

This script reads image paths from train.txt, test.txt, and val.txt files,
and copies corresponding image and label files to the appropriate directories
organized by split and subfolder (images, labels).
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List


def read_split_files(split_dir: str) -> Dict[str, List[str]]:
    """
    Read image paths from train.txt, test.txt, and val.txt files.
    
    Args:
        split_dir: Directory containing the split text files
        
    Returns:
        Dictionary with keys 'train', 'test', 'val' and lists of image paths as values
    """
    splits = {}
    
    for split_name in ['train', 'test', 'val']:
        split_file = os.path.join(split_dir, f'{split_name}.txt')
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found")
            splits[split_name] = []
            continue
            
        with open(split_file, 'r', encoding='utf-8') as f:
            image_paths = [line.strip() for line in f if line.strip()]
            splits[split_name] = image_paths
    
    return splits


def get_label_path(image_path: str) -> str:
    """
    Convert image path to corresponding label path.
    Assumes labels have same structure but with .txt extension and in labels folder.
    
    Args:
        image_path: Path to image file (e.g., 'DV4/images/..../image.jpg')
        
    Returns:
        Corresponding label path (e.g., 'DV4/labels/..../image.txt')
    """
    # Replace 'images' with 'labels' and change extension to .txt
    label_path = image_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    return label_path


def create_output_structure(output_dir: str, splits: List[str]) -> None:
    """
    Create the output directory structure.
    
    Args:
        output_dir: Root output directory
        splits: List of split names (e.g., ['train', 'test', 'val'])
    """
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)


def copy_files(
    splits_data: Dict[str, List[str]],
    source_dir: str,
    output_dir: str,
    skip_missing: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Copy image and label files to output directory organized by split.
    
    Args:
        splits_data: Dictionary from read_split_files()
        source_dir: Root directory where source images/labels are located
        output_dir: Root directory where files will be copied
        skip_missing: If True, skip missing files; if False, raise error
        
    Returns:
        Dictionary with copy statistics for each split
    """
    stats = {}
    
    for split_name, image_paths in splits_data.items():
        images_copied = 0
        labels_copied = 0
        missing_count = 0
        
        split_output_dir = os.path.join(output_dir, split_name)
        
        for image_path in image_paths:
            # Construct full source paths
            source_image = os.path.join(source_dir, image_path)
            source_label = os.path.join(source_dir, get_label_path(image_path))
            
            # Construct destination paths
            image_filename = os.path.basename(image_path)
            label_filename = os.path.basename(source_label)
            
            dest_image = os.path.join(split_output_dir, 'images', image_filename)
            dest_label = os.path.join(split_output_dir, 'labels', label_filename)
            
            # Create destination subdirectories if needed
            os.makedirs(os.path.dirname(dest_image), exist_ok=True)
            os.makedirs(os.path.dirname(dest_label), exist_ok=True)
            
            # Copy image file
            if os.path.exists(source_image):
                shutil.copy2(source_image, dest_image)
                images_copied += 1
            else:
                if skip_missing:
                    print(f"Warning: Image not found: {source_image}")
                    missing_count += 1
                else:
                    raise FileNotFoundError(f"Image not found: {source_image}")
            
            # Copy label file
            if os.path.exists(source_label):
                shutil.copy2(source_label, dest_label)
                labels_copied += 1
            else:
                if skip_missing:
                    print(f"Warning: Label not found: {source_label}")
                else:
                    # Labels might be optional, so we warn but don't fail
                    print(f"Warning: Label not found: {source_label}")
        
        stats[split_name] = {
            'images': images_copied,
            'labels': labels_copied,
            'missing': missing_count
        }
        
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Copy dataset files according to train/test/val splits.'
    )
    parser.add_argument(
        '--split-dir',
        required=True,
        help='Directory containing train.txt, test.txt, val.txt split files'
    )
    parser.add_argument(
        '--source-dir',
        default='/home/robby/workspace/Oil-Spill-Benz',
        help='Root source directory where DV4/images and DV4/labels are located'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory where split data will be organized'
    )
    parser.add_argument(
        '--skip-missing',
        action='store_true',
        help='Skip missing files instead of raising an error'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.split_dir):
        raise ValueError(f"Split directory not found: {args.split_dir}")
    if not os.path.exists(args.source_dir):
        raise ValueError(f"Source directory not found: {args.source_dir}")
    
    print(f"Reading split files from: {args.split_dir}")
    splits_data = read_split_files(args.split_dir)
    
    # Print statistics
    for split_name, paths in splits_data.items():
        print(f"  {split_name}: {len(paths)} images")
    
    print(f"\nCreating output directory structure in: {args.output_dir}")
    create_output_structure(args.output_dir, splits_data.keys())
    
    print(f"\nCopying files from: {args.source_dir}")
    stats = copy_files(
        splits_data,
        args.source_dir,
        args.output_dir,
        skip_missing=args.skip_missing
    )
    
    # Print copy statistics
    print("\nCopy Statistics:")
    print("-" * 50)
    for split_name, split_stats in stats.items():
        print(f"{split_name}:")
        print(f"  Images copied: {split_stats['images']}")
        print(f"  Labels copied: {split_stats['labels']}")
        if split_stats['missing'] > 0:
            print(f"  Missing files: {split_stats['missing']}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
