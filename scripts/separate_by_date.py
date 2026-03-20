#!/usr/bin/env python3
"""
Script to separate images and labels into date-based folders.
Example: UAV_20230726_Angel_0101.jpg -> <output_folder>/20230726/UAV_20230726_Angel_0101.jpg
"""

import os
import shutil
import argparse
from pathlib import Path
import re


def extract_date_from_filename(filename):
    """
    Extract date from filename in format UAV_YYYYMMDD_*.ext
    
    Args:
        filename: Name of the file
        
    Returns:
        Date string (YYYYMMDD) or None if pattern not found
    """
    # Pattern: UAV_YYYYMMDD_*
    pattern = r'UAV_(\d{8})_'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def organize_files_by_date(input_folder, output_folder, copy=True):
    """
    Organize files from input_folder into date-based subdirectories in output_folder.
    
    Args:
        input_folder: Path to input folder containing images/ and labels/ subdirectories
        output_folder: Path to output folder where date-based subdirectories will be created
        copy: If True, copy files; if False, move files
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Check if input folder exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process both images and labels folders
    for subfolder in ['images', 'labels']:
        subfolder_path = input_path / subfolder
        
        if not subfolder_path.exists():
            print(f"Warning: {subfolder_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {subfolder}...")
        
        # Get all files in the subfolder
        files = [f for f in subfolder_path.iterdir() if f.is_file()]
        
        processed_count = 0
        skipped_count = 0
        
        for file_path in files:
            filename = file_path.name
            
            # Extract date from filename
            date = extract_date_from_filename(filename)
            
            if date is None:
                print(f"  Skipping {filename}: Could not extract date")
                skipped_count += 1
                continue
            
            # Create date-based subdirectory
            date_folder = output_path / date / subfolder
            date_folder.mkdir(parents=True, exist_ok=True)
            
            # Destination path
            dest_path = date_folder / filename
            
            # Copy or move the file
            if copy:
                shutil.copy2(file_path, dest_path)
                action = "Copied"
            else:
                shutil.move(str(file_path), str(dest_path))
                action = "Moved"
            
            print(f"  {action}: {filename} -> {date}/{subfolder}/")
            processed_count += 1
        
        print(f"Processed {processed_count} files, skipped {skipped_count} files in {subfolder}")


def main():
    parser = argparse.ArgumentParser(
        description='Organize images and labels into date-based folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Move files (default)
  python separate_by_date.py datasets/raw/dv3-combined datasets/processed/dv3-by-date
  
  # Copy files instead of moving
  python separate_by_date.py datasets/raw/dv3-combined datasets/processed/dv3-by-date --copy
        """
    )
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='Input folder containing images/ and labels/ subdirectories'
    )
    
    parser.add_argument(
        'output_folder',
        type=str,
        help='Output folder where date-based subdirectories will be created'
    )
    
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them (default: move)'
    )
    
    args = parser.parse_args()
    
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Mode: {'Copy' if args.copy else 'Move'}")
    
    organize_files_by_date(args.input_folder, args.output_folder, args.copy)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
