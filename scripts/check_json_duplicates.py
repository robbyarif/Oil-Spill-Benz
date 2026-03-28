#!/usr/bin/env python3
"""
Check for duplicate JSON annotation files with "__1.json" suffix.

For each *.__1.json file, this script checks whether the corresponding
*.json exists in the same directory, and produces a summary plus a
comparison (size/hash/annotation counts).

Example:
  python scripts/check_json_duplicates.py datasets/raw/DV3/Categorize_UAV \
      --export-csv logs/json_duplicate_report.csv
"""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


DUP_SUFFIX = "__1.json"


@dataclass
class AnnotationStats:
    total: int
    shapes: int
    annotations: int
    objects: int
    parse_error: Optional[str] = None


def sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def annotation_stats(path: str) -> AnnotationStats:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = len(data.get("shapes", [])) if isinstance(data, dict) else 0
        annotations = len(data.get("annotations", [])) if isinstance(data, dict) else 0
        objects = len(data.get("objects", [])) if isinstance(data, dict) else 0

        if isinstance(data, list):
            total = len(data)
        else:
            total = shapes + annotations + objects

        return AnnotationStats(
            total=total,
            shapes=shapes,
            annotations=annotations,
            objects=objects,
            parse_error=None,
        )
    except Exception as exc:
        return AnnotationStats(total=0, shapes=0, annotations=0, objects=0, parse_error=str(exc))


def scan_duplicates(root_dir: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Returns a dict keyed by duplicate path with original path and status.
    """
    results = {}
    for root, _, files in os.walk(root_dir):
        for name in files:
            if not name.endswith(DUP_SUFFIX):
                continue
            dup_path = os.path.join(root, name)
            orig_name = name[: -len(DUP_SUFFIX)] + ".json"
            orig_path = os.path.join(root, orig_name)
            results[dup_path] = {
                "dup_path": dup_path,
                "orig_path": orig_path,
                "orig_exists": os.path.exists(orig_path),
            }
    return results


def compare_pair(dup_path: str, orig_path: str, orig_exists: bool) -> Dict[str, Optional[str]]:
    dup_size = os.path.getsize(dup_path)
    orig_size = os.path.getsize(orig_path) if orig_exists else None

    dup_hash = sha256_file(dup_path)
    orig_hash = sha256_file(orig_path) if orig_exists else None

    dup_stats = annotation_stats(dup_path)
    orig_stats = annotation_stats(orig_path) if orig_exists else None

    same_hash = orig_exists and (dup_hash is not None) and (dup_hash == orig_hash)

    return {
        "dup_size": dup_size,
        "orig_size": orig_size,
        "dup_hash": dup_hash,
        "orig_hash": orig_hash,
        "same_hash": same_hash,
        "dup_stats": dup_stats,
        "orig_stats": orig_stats,
    }


def print_summary(rows: list) -> None:
    total = len(rows)
    with_orig = sum(1 for r in rows if r["orig_exists"])
    missing_orig = total - with_orig
    same_hash = sum(1 for r in rows if r["same_hash"] is True)
    diff_hash = sum(1 for r in rows if r["orig_exists"] and r["same_hash"] is False)
    parse_errors = sum(
        1
        for r in rows
        if (r["dup_stats"].parse_error is not None)
        or (r["orig_stats"] and r["orig_stats"].parse_error is not None)
    )

    print("\n" + "=" * 80)
    print("JSON DUPLICATE SUMMARY (__1.json)")
    print("=" * 80)
    print(f"Total duplicates found: {total}")
    print(f"With original .json:    {with_orig}")
    print(f"Missing original .json: {missing_orig}")
    print(f"Identical (hash):       {same_hash}")
    print(f"Different (hash):       {diff_hash}")
    print(f"JSON parse errors:      {parse_errors}")
    print("=" * 80 + "\n")


def export_csv(rows: list, output_path: str) -> None:
    fieldnames = [
        "dup_path",
        "orig_path",
        "orig_exists",
        "same_hash",
        "dup_size",
        "orig_size",
        "dup_hash",
        "orig_hash",
        "dup_total",
        "orig_total",
        "dup_shapes",
        "orig_shapes",
        "dup_annotations",
        "orig_annotations",
        "dup_objects",
        "orig_objects",
        "dup_parse_error",
        "orig_parse_error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            orig_stats = r["orig_stats"]
            row = {
                "dup_path": r["dup_path"],
                "orig_path": r["orig_path"],
                "orig_exists": r["orig_exists"],
                "same_hash": r["same_hash"],
                "dup_size": r["dup_size"],
                "orig_size": r["orig_size"],
                "dup_hash": r["dup_hash"],
                "orig_hash": r["orig_hash"],
                "dup_total": r["dup_stats"].total,
                "orig_total": orig_stats.total if orig_stats else None,
                "dup_shapes": r["dup_stats"].shapes,
                "orig_shapes": orig_stats.shapes if orig_stats else None,
                "dup_annotations": r["dup_stats"].annotations,
                "orig_annotations": orig_stats.annotations if orig_stats else None,
                "dup_objects": r["dup_stats"].objects,
                "orig_objects": orig_stats.objects if orig_stats else None,
                "dup_parse_error": r["dup_stats"].parse_error,
                "orig_parse_error": orig_stats.parse_error if orig_stats else None,
            }
            writer.writerow(row)

    print(f"✓ CSV report written to: {output_path}")


def _safe_backup(path: str) -> str:
    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        os.rename(path, backup_path)
        return backup_path

    counter = 1
    while True:
        candidate = f"{backup_path}.{counter}"
        if not os.path.exists(candidate):
            os.rename(path, candidate)
            return candidate
        counter += 1


def _rename_duplicates(rows: list) -> None:
    renamed = 0
    backed_up = 0

    for r in rows:
        dup_path = r["dup_path"]
        orig_path = r["orig_path"]
        if not r["orig_exists"]:
            os.rename(dup_path, orig_path)
            renamed += 1
            continue

        backup_path = _safe_backup(orig_path)
        backed_up += 1
        os.rename(dup_path, orig_path)
        renamed += 1
        r["orig_backup"] = backup_path

    print("\n" + "=" * 80)
    print("RENAME SUMMARY")
    print("=" * 80)
    print(f"Renamed duplicates to originals: {renamed}")
    print(f"Originals backed up:             {backed_up}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check __1.json duplicates and compare with originals."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing JSON annotation files",
    )
    parser.add_argument(
        "--export-csv",
        help="Optional path to export CSV report",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help=(
            "Rename __1.json to .json. If .json exists, back it up to .json.bak"
        ),
    )

    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        raise ValueError(f"Not a directory: {args.root_dir}")

    dup_map = scan_duplicates(args.root_dir)
    if not dup_map:
        print("No __1.json duplicates found.")
        return

    rows = []
    for dup_path, info in dup_map.items():
        comparison = compare_pair(dup_path, info["orig_path"], info["orig_exists"])
        rows.append({
            "dup_path": dup_path,
            "orig_path": info["orig_path"],
            "orig_exists": info["orig_exists"],
            **comparison,
        })

    print_summary(rows)

    if args.rename:
        _rename_duplicates(rows)

    if args.export_csv:
        export_csv(rows, args.export_csv)


if __name__ == "__main__":
    main()
