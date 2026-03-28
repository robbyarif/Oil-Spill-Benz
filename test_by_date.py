#!/usr/bin/env python3
"""
Test dataset by date subsets and print metrics for each subset.

This script:
1. Parses test.txt file to extract dates from filenames (UAV_yyyymmdd_*.jpg)
2. Groups test images by date
3. Creates subset test files for each date
4. Runs test on each subset using the specified trainer
5. Prints metrics for each subset and overall

Example:
  python scripts/test_by_date.py \
    --test-file datasets/processed/exp2.4-dv3train/test.txt \
    --model-weights runs/exp-manager/yolov11-exp2.4-dv3train/weights/best.pt \
    --dataset-dir datasets/processed/exp2.4-dv3train \
    --output-dir runs/test_by_date_exp2.4 \
    --trainer yolo \
    --model-version 11
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb


def extract_date_and_ship(filename: str) -> tuple:
    """Extract (date, ship) from UAV_YYYYMMDD_SHIPNAME_*.jpg format."""
    match = re.search(r'UAV_(\d{8})_([^_]+)_', filename)
    if match:
        return (match.group(1), match.group(2))
    return None


def group_by_date_ship(test_file: str) -> dict:
    """
    Parse test.txt and group image paths by (date, ship).
    
    Returns:
        Dict mapping (date, ship) tuple -> list of image paths
    """
    date_ship_groups = defaultdict(list)
    
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract filename from path
            filename = os.path.basename(line)
            date_ship = extract_date_and_ship(filename)
            
            if date_ship:
                date_ship_groups[date_ship].append(line)
    
    return dict(sorted(date_ship_groups.items()))


def create_subset_test_file(output_dir: str, date: str, ship: str, image_paths: list) -> str:
    """Create a temporary test.txt file for a specific date-ship subset."""
    os.makedirs(output_dir, exist_ok=True)
    subset_file = os.path.join(output_dir, f"test_subset_{date}_{ship}.txt")
    
    with open(subset_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')
    
    return subset_file


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to readable format."""
    try:
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d (%a)')
    except:
        return date_str


def format_date_ship(date: str, ship: str) -> str:
    """Format date and ship for display."""
    formatted_date = format_date(date)
    return f"{formatted_date} | {ship}"


def test_with_yolo(trainer, subset_file: str, dataset_dir: str, 
                   output_dir: str, date: str, ship: str) -> dict:
    """Run YOLO test on subset and return metrics."""
    subset_output = os.path.join(output_dir, f"results_{date}_{ship}")
    os.makedirs(subset_output, exist_ok=True)
    
    # Create temporary test.txt in dataset_dir
    temp_test_path = os.path.join(dataset_dir, "test.txt")
    original_test_path = os.path.join(dataset_dir, "test.txt.bak")
    
    # Backup original
    if os.path.exists(temp_test_path):
        os.rename(temp_test_path, original_test_path)
    
    try:
        # Copy subset to test.txt
        with open(subset_file, 'r') as src:
            with open(temp_test_path, 'w') as dst:
                dst.write(src.read())
        
        # Run test
        trainer.test(
            dataset_dir, 
            dst=subset_output,
            file_name="test.txt",
            color_coded=True,
            log=True,
            save=True
        )
        
        # Return metrics from trainer
        return trainer._analyze(dst=subset_output, log=False, color_coded=False, save=False)
    
    finally:
        # Restore original test.txt
        if os.path.exists(original_test_path):
            os.rename(original_test_path, temp_test_path)


def test_with_trainer(trainer_class, trainer_type: str, model_weights: str,
                      model_version: int, subset_file: str, dataset_dir: str,
                      output_dir: str, date: str, ship: str) -> dict:
    """Generic test wrapper for different trainer types."""
    
    trainer = trainer_class()
    
    if trainer_type == 'yolo':
        trainer.load_model(weights=model_weights, version=model_version)
    else:
        trainer.load_model(weights=model_weights)
    
    return test_with_yolo(trainer, subset_file, dataset_dir, output_dir, date, ship)


def print_metrics_summary(all_metrics: dict) -> None:
    """Print formatted metrics for each date-ship combination."""
    print("\n" + "=" * 120)
    print("TEST RESULTS BY DATE AND SHIP")
    print("=" * 120)
    
    if not all_metrics:
        print("No results available.")
        return
    
    # Print header
    metric_keys = list(all_metrics[list(all_metrics.keys())[0]].keys())
    header = f"{'Date | Ship':<45} | {'Count':>5}"
    for key in metric_keys:
        header += f" | {key:>12}"
    print(header)
    print("-" * (55 + len(metric_keys) * 16))
    
    # Print each date-ship
    for (date, ship) in sorted(all_metrics.keys()):
        metrics = all_metrics[(date, ship)]
        count = metrics.get('_count', '')
        row = f"{format_date_ship(date, ship):<45} | {count:>5}"
        for key in metric_keys:
            if key == '_count':
                continue
            value = metrics[key]
            if value is None:
                row += f" | {'N/A':>12}"
            else:
                row += f" | {value:>12.3f}"
        print(row)
    
    # Calculate and print overall metrics
    print("-" * (55 + len(metric_keys) * 16))
    overall_metrics = {}
    for key in metric_keys:
        if key == '_count':
            continue
        values = [m[key] for m in all_metrics.values() if m.get(key) is not None]
        if values:
            overall_metrics[key] = np.mean(values)
        else:
            overall_metrics[key] = None
    
    total_count = sum(m.get('_count', 0) for m in all_metrics.values())
    row = f"{'OVERALL':<45} | {total_count:>5}"
    for key in metric_keys:
        if key == '_count':
            continue
        value = overall_metrics[key]
        if value is None:
            row += f" | {'N/A':>12}"
        else:
            row += f" | {value:>12.3f}"
    print(row)
    print("=" * 120 + "\n")


def export_csv(all_metrics: dict, output_csv: str) -> None:
    """Export metrics to CSV."""
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    
    if not all_metrics:
        print("No results to export.")
        return
    
    metric_keys = [k for k in all_metrics[list(all_metrics.keys())[0]].keys() if k != '_count']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Date', 'Ship', 'Formatted Date', 'Count'] + metric_keys)
        
        # Data rows
        for (date, ship) in sorted(all_metrics.keys()):
            metrics = all_metrics[(date, ship)]
            count = metrics.get('_count', '')
            row = [date, ship, format_date(date), count]
            for key in metric_keys:
                value = metrics[key]
                row.append(value if value is not None else '')
            writer.writerow(row)
        
        # Overall row
        overall_metrics = {}
        for key in metric_keys:
            values = [m[key] for m in all_metrics.values() if m.get(key) is not None]
            overall_metrics[key] = np.mean(values) if values else None
        
        total_count = sum(m.get('_count', 0) for m in all_metrics.values())
        row = ['OVERALL', '', '', total_count]
        for key in metric_keys:
            value = overall_metrics[key]
            row.append(value if value is not None else '')
        writer.writerow(row)
    
    print(f"✓ CSV report written to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test dataset by date subsets and print metrics."
    )
    parser.add_argument(
        '--test-file',
        required=True,
        help='Path to test.txt file'
    )
    parser.add_argument(
        '--model-weights',
        required=True,
        help='Path to model weights'
    )
    parser.add_argument(
        '--dataset-dir',
        required=True,
        help='Path to dataset directory (must contain train.txt/val.txt)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--trainer',
        default='yolo',
        choices=['yolo', 'deeplabv3', 'segformer', 'rfdetr'],
        help='Trainer type (default: yolo)'
    )
    parser.add_argument(
        '--model-version',
        type=int,
        default=11,
        help='YOLO model version (11 or 12, default: 11)'
    )
    parser.add_argument(
        '--export-csv',
        help='Optional path to export CSV report'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.test_file):
        raise ValueError(f"test.txt not found: {args.test_file}")
    if not os.path.exists(args.model_weights):
        raise ValueError(f"model weights not found: {args.model_weights}")
    if not os.path.exists(args.dataset_dir):
        raise ValueError(f"dataset dir not found: {args.dataset_dir}")
    
    # Import trainer class
    if args.trainer == 'yolo':
        from yolo import YoloTrainer
        trainer_class = YoloTrainer
    elif args.trainer == 'deeplabv3':
        from deeplabv3 import DeeplabTrainer
        trainer_class = DeeplabTrainer
    elif args.trainer == 'segformer':
        from segformer import SegformerTrainer
        trainer_class = SegformerTrainer
    elif args.trainer == 'rfdetr':
        from rfdetr_trainer import RFDETRTrainer
        trainer_class = RFDETRTrainer
    else:
        raise ValueError(f"Unknown trainer: {args.trainer}")
    
    # Initialize wandb
    model_name = f"{args.trainer}-v{args.model_version}" if args.trainer == 'yolo' else args.trainer
    wandb.init(
        project="Oil-Spill-UAV",
        name=f"test-by-date-{model_name}-{os.path.basename(args.dataset_dir)}",
        config={
            "trainer": args.trainer,
            "model_version": args.model_version,
            "model_weights": os.path.basename(args.model_weights),
            "dataset": os.path.basename(args.dataset_dir),
        }
    )
    
    print(f"Parsing test.txt: {args.test_file}")
    date_ship_groups = group_by_date_ship(args.test_file)
    
    print(f"Found {len(date_ship_groups)} unique date-ship combination(s):")
    for (date, ship) in sorted(date_ship_groups.keys()):
        print(f"  {format_date_ship(date, ship)}: {len(date_ship_groups[(date, ship)])} images")
    
    print(f"\nRunning tests by date-ship subset...\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = {}
    
    for (date, ship) in sorted(date_ship_groups.keys()):
        image_paths = date_ship_groups[(date, ship)]
        print(f"Testing {format_date_ship(date, ship)} ({len(image_paths)} images)...")
        
        # Create subset test file
        subset_file = create_subset_test_file(
            os.path.join(args.output_dir, 'subsets'),
            date,
            ship,
            image_paths
        )
        
        # Run test
        metrics = test_with_trainer(
            trainer_class,
            args.trainer,
            args.model_weights,
            args.model_version,
            subset_file,
            args.dataset_dir,
            args.output_dir,
            date,
            ship
        )
        
        metrics['_count'] = len(image_paths)
        all_metrics[(date, ship)] = metrics
        
        # Log to wandb
        subset_name = f"{date}_{ship}"
        log_data = {f"{subset_name}/{k}": v for k, v in metrics.items()}
        wandb.log(log_data)
    
    # Print summary
    print_metrics_summary(all_metrics)
    
    # Create wandb table for detailed metrics
    metric_keys = [k for k in all_metrics[list(all_metrics.keys())[0]].keys() if k != '_count']
    table_columns = ["Model", "Date", "Ship", "Formatted Date", "Count"] + metric_keys
    table_data = []
    
    for (date, ship) in sorted(all_metrics.keys()):
        metrics = all_metrics[(date, ship)]
        count = metrics.get('_count', 0)
        row = [model_name, date, ship, format_date(date), count]
        for key in metric_keys:
            value = metrics.get(key)
            row.append(value if value is not None else None)
        table_data.append(row)
    
    # Add overall row to table
    overall_metrics = {}
    total_count = 0
    for key in metric_keys:
        values = [m[key] for m in all_metrics.values() if m.get(key) is not None]
        overall_metrics[key] = np.mean(values) if values else None
    
    for m in all_metrics.values():
        total_count += m.get('_count', 0)
    
    overall_row = [model_name, "OVERALL", "", "", total_count]
    for key in metric_keys:
        overall_row.append(overall_metrics.get(key))
    table_data.append(overall_row)
    
    # Log table to wandb
    results_table = wandb.Table(columns=table_columns, data=table_data)
    wandb.log({"test_results_by_date_ship": results_table})
    
    # Log overall metrics as summary
    overall_data = {f"overall/{k}": v for k, v in overall_metrics.items() if v is not None}
    overall_data["overall/total_images"] = total_count
    overall_data["overall/num_subsets"] = len(all_metrics)
    wandb.log(overall_data)
    
    # Also log overall metrics as wandb summary (persistent)
    for key, value in overall_metrics.items():
        if value is not None:
            wandb.run.summary[f"overall_{key}"] = value
    wandb.run.summary["total_images"] = total_count
    wandb.run.summary["num_subsets"] = len(all_metrics)
    
    # Export CSV if requested
    if args.export_csv:
        export_csv(all_metrics, args.export_csv)
    
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
