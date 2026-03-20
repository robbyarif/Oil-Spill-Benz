#!/usr/bin/env python3
"""
Script to generate a summary of files in a date-organized dataset folder.
Analyzes the structure and provides statistics about images and labels.
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import json


def get_file_stem(filename):
    """Get filename without extension."""
    return Path(filename).stem


def analyze_date_folder(date_folder_path):
    """
    Analyze a single date folder and return statistics.
    
    Args:
        date_folder_path: Path to date folder
        
    Returns:
        Dictionary with statistics
    """
    images_path = date_folder_path / 'images'
    labels_path = date_folder_path / 'labels'
    
    # Get files
    images = []
    labels = []
    
    if images_path.exists():
        images = [f.name for f in images_path.iterdir() if f.is_file()]
    
    if labels_path.exists():
        labels = [f.name for f in labels_path.iterdir() if f.is_file()]
    
    # Get stems (filenames without extensions)
    image_stems = set(get_file_stem(f) for f in images)
    label_stems = set(get_file_stem(f) for f in labels)
    
    # Find matches and mismatches
    matched = image_stems & label_stems
    images_only = image_stems - label_stems
    labels_only = label_stems - image_stems
    
    return {
        'num_images': len(images),
        'num_labels': len(labels),
        'num_matched': len(matched),
        'num_images_only': len(images_only),
        'num_labels_only': len(labels_only),
        'images_only': sorted(list(images_only)),
        'labels_only': sorted(list(labels_only))
    }


def summarize_dataset(folder_path, verbose=False, output_file=None):
    """
    Generate a summary of the dataset organized by date.
    
    Args:
        folder_path: Path to the root folder containing date subdirectories
        verbose: If True, show detailed information
        output_file: If provided, write summary to this file
    """
    root_path = Path(folder_path)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all date folders (directories with 8-digit names)
    date_folders = sorted([
        d for d in root_path.iterdir() 
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    ])
    
    if not date_folders:
        print(f"No date folders found in {folder_path}")
        return
    
    # Analyze each date folder
    summary = {}
    total_images = 0
    total_labels = 0
    total_matched = 0
    total_images_only = 0
    total_labels_only = 0
    
    for date_folder in date_folders:
        date = date_folder.name
        stats = analyze_date_folder(date_folder)
        summary[date] = stats
        
        total_images += stats['num_images']
        total_labels += stats['num_labels']
        total_matched += stats['num_matched']
        total_images_only += stats['num_images_only']
        total_labels_only += stats['num_labels_only']
    
    # Print summary
    output_lines = []
    
    def add_line(line=""):
        output_lines.append(line)
        print(line)
    
    add_line("=" * 80)
    add_line(f"DATASET SUMMARY: {folder_path}")
    add_line("=" * 80)
    add_line()
    
    add_line(f"Total date folders: {len(date_folders)}")
    add_line(f"Date range: {date_folders[0].name} to {date_folders[-1].name}")
    add_line()
    
    add_line("OVERALL STATISTICS")
    add_line("-" * 80)
    add_line(f"Total images:              {total_images:6d}")
    add_line(f"Total labels:              {total_labels:6d}")
    add_line(f"Matched pairs:             {total_matched:6d}")
    add_line(f"Images without labels:     {total_images_only:6d}")
    add_line(f"Labels without images:     {total_labels_only:6d}")
    add_line()
    
    add_line("PER-DATE BREAKDOWN")
    add_line("-" * 80)
    add_line(f"{'Date':<12} {'Images':>8} {'Labels':>8} {'Matched':>8} {'Img-Only':>10} {'Lbl-Only':>10}")
    add_line("-" * 80)
    
    for date, stats in summary.items():
        add_line(
            f"{date:<12} {stats['num_images']:>8d} {stats['num_labels']:>8d} "
            f"{stats['num_matched']:>8d} {stats['num_images_only']:>10d} "
            f"{stats['num_labels_only']:>10d}"
        )
    
    add_line("-" * 80)
    add_line(
        f"{'TOTAL':<12} {total_images:>8d} {total_labels:>8d} "
        f"{total_matched:>8d} {total_images_only:>10d} "
        f"{total_labels_only:>10d}"
    )
    add_line()
    
    # Verbose output - show mismatches
    if verbose:
        add_line("=" * 80)
        add_line("DETAILED MISMATCHES")
        add_line("=" * 80)
        add_line()
        
        for date, stats in summary.items():
            if stats['num_images_only'] > 0 or stats['num_labels_only'] > 0:
                add_line(f"Date: {date}")
                add_line("-" * 80)
                
                if stats['num_images_only'] > 0:
                    add_line(f"  Images without labels ({stats['num_images_only']}):")
                    for filename in stats['images_only'][:10]:  # Show first 10
                        add_line(f"    - {filename}")
                    if len(stats['images_only']) > 10:
                        add_line(f"    ... and {len(stats['images_only']) - 10} more")
                
                if stats['num_labels_only'] > 0:
                    add_line(f"  Labels without images ({stats['num_labels_only']}):")
                    for filename in stats['labels_only'][:10]:  # Show first 10
                        add_line(f"    - {filename}")
                    if len(stats['labels_only']) > 10:
                        add_line(f"    ... and {len(stats['labels_only']) - 10} more")
                
                add_line()
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        
        print(f"\n✓ Summary saved to: {output_file}")
    
    # Also save JSON summary
    if output_file:
        json_file = output_path.with_suffix('.json')
        json_summary = {
            'folder': str(folder_path),
            'num_dates': len(date_folders),
            'date_range': {
                'start': date_folders[0].name,
                'end': date_folders[-1].name
            },
            'totals': {
                'images': total_images,
                'labels': total_labels,
                'matched': total_matched,
                'images_only': total_images_only,
                'labels_only': total_labels_only
            },
            'dates': summary
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        print(f"✓ JSON summary saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a summary of a date-organized dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic summary
  python summarize_dataset.py datasets/processed/dv3-by-date
  
  # Verbose output with mismatches
  python summarize_dataset.py datasets/processed/dv3-by-date --verbose
  
  # Save summary to file
  python summarize_dataset.py datasets/processed/dv3-by-date -o logs/dv3_summary.txt
        """
    )
    
    parser.add_argument(
        'folder',
        type=str,
        help='Path to the root folder containing date subdirectories'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information including mismatches'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Save summary to output file (also creates a .json file)'
    )
    
    args = parser.parse_args()
    
    summarize_dataset(args.folder, verbose=args.verbose, output_file=args.output)


if __name__ == '__main__':
    main()
