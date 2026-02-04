#!/usr/bin/env python3
"""
Summarize annotation files by incident date.

Scans JSON annotation files in a directory structure, extracts the date from filenames,
and generates a summary showing:
- Number of files per day
- Annotation status per day (annotated/unannotated)
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def extract_date_from_filename(filename: str) -> str:
    """
    Extract date from filename format: UAV_YYYYMMDD_Name_Number.json
    
    Args:
        filename: The filename to parse
        
    Returns:
        Date string in format YYYYMMDD, or None if not found
    """
    parts = filename.replace('.json', '').split('_')
    if len(parts) >= 2 and parts[0] == 'UAV':
        date_str = parts[1]
        if len(date_str) == 8 and date_str.isdigit():
            return date_str
    return None


def is_annotated(json_path: str) -> bool:
    """
    Check if a JSON file contains annotations.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        True if file has annotations, False otherwise
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Check for common annotation fields
            if isinstance(data, dict):
                # Check if it has shapes/annotations
                if 'shapes' in data and len(data['shapes']) > 0:
                    return True
                if 'annotations' in data and len(data['annotations']) > 0:
                    return True
                if 'objects' in data and len(data['objects']) > 0:
                    return True
            return False
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return False


def scan_directory(root_dir: str) -> dict:
    """
    Scan directory for image and JSON annotation files, grouped by date.
    
    Args:
        root_dir: Root directory to scan
        
    Returns:
        Dictionary with date as key and list of (filename, annotated) tuples as value.
        Includes images with no annotations.
    """
    date_files = defaultdict(list)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    processed_images = set()
    
    # First pass: find JSON annotation files
    json_files_by_date = defaultdict(set)
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                date_str = extract_date_from_filename(file)
                if date_str:
                    file_path = os.path.join(root, file)
                    annotated = is_annotated(file_path)
                    date_files[date_str].append((file, annotated))
                    json_files_by_date[date_str].add(file.rsplit('.', 1)[0])  # Store base name
    
    # Second pass: find image files and check if they have annotations
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                date_str = extract_date_from_filename(file)
                if date_str:
                    file_base = file.rsplit('.', 1)[0]
                    processed_images.add(file_base)
                    
                    # Check if corresponding JSON annotation exists and is annotated
                    has_annotation = file_base in json_files_by_date.get(date_str, set())
                    
                    # Only add if we haven't already added it (to avoid duplicates)
                    if not any(f[0] == file for f in date_files[date_str]):
                        date_files[date_str].append((file, has_annotation))
    
    return date_files


def format_date(date_str: str) -> str:
    """
    Format YYYYMMDD to readable date format.
    
    Args:
        date_str: Date string in YYYYMMDD format
        
    Returns:
        Formatted date string (YYYY-MM-DD Day)
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        return date_obj.strftime('%Y-%m-%d %A')
    except:
        return date_str


def print_summary(date_files: dict) -> None:
    """
    Print formatted summary of files by date, including unannotated images.
    
    Args:
        date_files: Dictionary from scan_directory()
    """
    print("\n" + "="*80)
    print("ANNOTATION SUMMARY BY DATE (Including all images)")
    print("="*80 + "\n")
    
    # Sort by date
    sorted_dates = sorted(date_files.keys())
    
    total_files = 0
    total_annotated = 0
    
    for date_str in sorted_dates:
        files_list = date_files[date_str]
        total_count = len(files_list)
        annotated_count = sum(1 for _, annotated in files_list if annotated)
        unannotated_count = total_count - annotated_count
        
        total_files += total_count
        total_annotated += annotated_count
        
        formatted_date = format_date(date_str)
        print(f"Date: {formatted_date}")
        print(f"  Total Files:       {total_count:3d}")
        print(f"  Annotated:         {annotated_count:3d}")
        print(f"  Unannotated:       {unannotated_count:3d}")
        print(f"  Annotation Rate:   {(annotated_count/total_count)*100:5.1f}%")
        print()
    
    # Print overall statistics
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total Files:       {total_files}")
    print(f"Total Annotated:   {total_annotated}")
    print(f"Total Unannotated: {total_files - total_annotated}")
    print(f"Overall Rate:      {(total_annotated/total_files)*100:.1f}%" if total_files > 0 else "N/A")
    print(f"Unique Dates:      {len(sorted_dates)}")
    print("="*80 + "\n")


def export_csv(date_files: dict, output_path: str) -> None:
    """
    Export summary to CSV file.
    
    Args:
        date_files: Dictionary from scan_directory()
        output_path: Path to save CSV file
    """
    import csv
    
    sorted_dates = sorted(date_files.keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Formatted Date', 'Total Files', 'Annotated', 'Unannotated', 'Annotation Rate (%)'])
        
        for date_str in sorted_dates:
            files_list = date_files[date_str]
            total_count = len(files_list)
            annotated_count = sum(1 for _, annotated in files_list if annotated)
            unannotated_count = total_count - annotated_count
            rate = (annotated_count / total_count * 100) if total_count > 0 else 0
            
            writer.writerow([
                date_str,
                format_date(date_str),
                total_count,
                annotated_count,
                unannotated_count,
                f"{rate:.1f}"
            ])
    
    print(f"✓ CSV summary exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize annotation files by incident date.'
    )
    parser.add_argument(
        'root_dir',
        help='Root directory containing annotation JSON files (can have subdirectories)'
    )
    parser.add_argument(
        '--export-csv',
        help='Export summary to CSV file (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.root_dir):
        raise ValueError(f"Directory not found: {args.root_dir}")
    if not os.path.isdir(args.root_dir):
        raise ValueError(f"Not a directory: {args.root_dir}")
    
    print(f"Scanning directory: {args.root_dir}")
    date_files = scan_directory(args.root_dir)
    
    if not date_files:
        print("No annotation files found!")
        return
    
    print(f"Found {sum(len(v) for v in date_files.values())} files across {len(date_files)} dates\n")
    
    # Print summary
    print_summary(date_files)
    
    # Export CSV if requested
    if args.export_csv:
        export_csv(date_files, args.export_csv)


if __name__ == '__main__':
    main()
