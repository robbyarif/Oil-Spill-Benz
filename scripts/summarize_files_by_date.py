#!/usr/bin/env python3
"""
Script to generate a summary of files in a folder, aggregated by date.
Extracts dates from filenames and provides statistics per date.
"""

import os
import argparse
import json
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime


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


def format_date_readable(date_str):
    """
    Convert YYYYMMDD format to readable format (YYYY-MM-DD)
    
    Args:
        date_str: Date string in YYYYMMDD format
        
    Returns:
        Formatted date string
    """
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        return dt.strftime('%Y-%m-%d')
    except:
        return date_str


def get_file_info(file_path):
    """
    Get file information including size and extension
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file info
    """
    stat = file_path.stat()
    return {
        'size': stat.st_size,
        'extension': file_path.suffix.lower()
    }


def summarize_files_by_date(folder_path, verbose=False, output_file=None):
    """
    Analyze files in a folder and summarize by date.
    
    Args:
        folder_path: Path to the folder containing files
        verbose: If True, show detailed information
        output_file: If provided, write summary to this file
    """
    root_path = Path(folder_path)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Aggregate files by date
    date_stats = defaultdict(lambda: {
        'count': 0,
        'total_size': 0,
        'extensions': defaultdict(int),
        'files': []
    })
    
    total_files = 0
    total_size = 0
    files_without_date = []
    extension_stats = defaultdict(int)
    
    # Process all files
    for file_path in root_path.iterdir():
        if not file_path.is_file():
            continue
        
        total_files += 1
        filename = file_path.name
        
        # Skip system files
        if filename.startswith('.') or filename.lower() == 'thumbs.db':
            continue
        
        file_info = get_file_info(file_path)
        total_size += file_info['size']
        extension_stats[file_info['extension']] += 1
        
        # Extract date
        date = extract_date_from_filename(filename)
        
        if date is None:
            files_without_date.append(filename)
            continue
        
        date_stats[date]['count'] += 1
        date_stats[date]['total_size'] += file_info['size']
        date_stats[date]['extensions'][file_info['extension']] += 1
        if verbose:
            date_stats[date]['files'].append(filename)
    
    # Sort by date
    sorted_dates = sorted(date_stats.keys())
    
    # Print summary
    output_lines = []
    
    def add_line(line=""):
        output_lines.append(line)
        print(line)
    
    add_line("=" * 100)
    add_line(f"FILE SUMMARY BY DATE: {folder_path}")
    add_line("=" * 100)
    add_line()
    
    add_line("OVERALL STATISTICS")
    add_line("-" * 100)
    add_line(f"Total files:               {total_files:8d}")
    add_line(f"Files with date info:      {sum(s['count'] for s in date_stats.values()):8d}")
    add_line(f"Files without date:        {len(files_without_date):8d}")
    add_line(f"Total size:                {total_size / (1024**3):8.2f} GB")
    add_line(f"Number of unique dates:    {len(sorted_dates):8d}")
    add_line()
    
    if sorted_dates:
        add_line(f"Date range:                {format_date_readable(sorted_dates[0])} to {format_date_readable(sorted_dates[-1])}")
        add_line()
    
    add_line("FILE TYPE DISTRIBUTION")
    add_line("-" * 100)
    for ext, count in sorted(extension_stats.items(), key=lambda x: x[1], reverse=True):
        ext_name = ext if ext else "(no extension)"
        add_line(f"  {ext_name:<10} {count:>8d} files")
    add_line()
    
    # Detailed per-date breakdown
    add_line("PER-DATE BREAKDOWN")
    add_line("-" * 100)
    add_line(f"{'Date':<12} {'Readable Date':<15} {'Files':>8} {'Size (MB)':>12} {'Avg Size (KB)':>15}")
    add_line("-" * 100)
    
    for date in sorted_dates:
        stats = date_stats[date]
        readable_date = format_date_readable(date)
        size_mb = stats['total_size'] / (1024**2)
        avg_size_kb = stats['total_size'] / stats['count'] / 1024 if stats['count'] > 0 else 0
        
        add_line(
            f"{date:<12} {readable_date:<15} {stats['count']:>8d} {size_mb:>12.2f} {avg_size_kb:>15.2f}"
        )
    
    add_line("-" * 100)
    total_size_mb = total_size / (1024**2)
    avg_size_kb = total_size / (total_files - len(files_without_date)) / 1024 if total_files > len(files_without_date) else 0
    add_line(f"{'TOTAL':<12} {'-':<15} {total_files - len(files_without_date):>8d} {total_size_mb:>12.2f} {avg_size_kb:>15.2f}")
    add_line()
    
    # Files without date info
    if files_without_date:
        add_line("FILES WITHOUT DATE INFORMATION")
        add_line("-" * 100)
        for filename in files_without_date[:20]:
            add_line(f"  {filename}")
        if len(files_without_date) > 20:
            add_line(f"  ... and {len(files_without_date) - 20} more")
        add_line()
    
    # Verbose output - file details per date
    if verbose and sorted_dates:
        add_line("=" * 100)
        add_line("DETAILED FILE LISTINGS BY DATE")
        add_line("=" * 100)
        add_line()
        
        for date in sorted_dates:
            stats = date_stats[date]
            readable_date = format_date_readable(date)
            add_line(f"Date: {date} ({readable_date}) - {stats['count']} files")
            add_line("-" * 100)
            
            # Show extension breakdown for this date
            for ext, count in sorted(stats['extensions'].items(), key=lambda x: x[1], reverse=True):
                ext_name = ext if ext else "(no extension)"
                add_line(f"  {ext_name}: {count} files")
            
            # Show file list (limit to first 15)
            if stats['files']:
                add_line("  Files:")
                for filename in stats['files'][:15]:
                    add_line(f"    - {filename}")
                if len(stats['files']) > 15:
                    add_line(f"    ... and {len(stats['files']) - 15} more")
            
            add_line()
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"\n✓ Summary saved to: {output_file}")
    
    # Also save JSON summary
    if output_file:
        json_file = output_path.with_suffix('.json')
        
        json_summary = {
            'folder': str(folder_path),
            'summary': {
                'total_files': total_files,
                'files_with_date': sum(s['count'] for s in date_stats.values()),
                'files_without_date': len(files_without_date),
                'total_size_bytes': total_size,
                'total_size_gb': total_size / (1024**3),
                'num_unique_dates': len(sorted_dates),
                'date_range': {
                    'start': sorted_dates[0] if sorted_dates else None,
                    'end': sorted_dates[-1] if sorted_dates else None
                }
            },
            'by_date': {
                date: {
                    'count': date_stats[date]['count'],
                    'total_size_bytes': date_stats[date]['total_size'],
                    'total_size_mb': date_stats[date]['total_size'] / (1024**2),
                    'extensions': dict(date_stats[date]['extensions'])
                }
                for date in sorted_dates
            },
            'extension_stats': dict(extension_stats)
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON summary saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a summary of files in a folder, aggregated by date',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic summary
  python summarize_files_by_date.py datasets/raw/AIUAV/資工_AI_無人機
  
  # Verbose output with file listings
  python summarize_files_by_date.py datasets/raw/AIUAV/資工_AI_無人機 --verbose
  
  # Save summary to file
  python summarize_files_by_date.py datasets/raw/AIUAV/資工_AI_無人機 -o logs/aiuav_summary.txt
        """
    )
    
    parser.add_argument(
        'folder',
        type=str,
        help='Path to the folder containing files'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information including file listings'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Save summary to output file (also creates a .json file)'
    )
    
    args = parser.parse_args()
    
    summarize_files_by_date(args.folder, verbose=args.verbose, output_file=args.output)


if __name__ == '__main__':
    main()
