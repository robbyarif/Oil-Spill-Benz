#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np


def _discover_pred_mask_dirs(root: Path) -> dict[str, Path]:
    model_dirs: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        pred_dir = child / "test" / "pred_masks"
        if pred_dir.is_dir():
            png_count = sum(1 for _ in pred_dir.glob("*.png"))
            if png_count > 0:
                model_dirs[child.name] = pred_dir
    return model_dirs


def _label_tile(mask_path: Path, model_name: str, title_height: int = 44) -> tuple[np.ndarray, int, float]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    binary = (mask > 127).astype(np.uint8)
    positive_pixels = int(binary.sum())
    positive_ratio = float(positive_pixels / binary.size)

    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    vis[binary == 1] = (0, 200, 255)

    tile = cv2.copyMakeBorder(
        vis,
        top=title_height,
        bottom=12,
        left=10,
        right=10,
        borderType=cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )

    cv2.putText(
        tile,
        model_name,
        (12, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        f"px={positive_pixels} ({positive_ratio * 100:.2f}%)",
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return tile, positive_pixels, positive_ratio


def _hstack_padded(images: list[np.ndarray], pad: int = 10, bg=(15, 15, 15)) -> np.ndarray:
    h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] == h:
            padded.append(img)
            continue
        delta = h - img.shape[0]
        padded_img = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=delta,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=bg,
        )
        padded.append(padded_img)

    separator = np.full((h, pad, 3), bg, dtype=np.uint8)
    out = []
    for i, img in enumerate(padded):
        out.append(img)
        if i < len(padded) - 1:
            out.append(separator)
    return np.hstack(out)


def _build_contact_sheet(rows: list[np.ndarray], cols: int = 3, bg=(10, 10, 10), gap: int = 12) -> np.ndarray:
    if not rows:
        raise ValueError("No rows to compose")

    cols = max(1, cols)
    chunks: list[np.ndarray] = []
    for i in range(0, len(rows), cols):
        group = rows[i : i + cols]
        max_h = max(x.shape[0] for x in group)
        normalized = []
        for x in group:
            if x.shape[0] < max_h:
                x = cv2.copyMakeBorder(
                    x,
                    top=0,
                    bottom=max_h - x.shape[0],
                    left=0,
                    right=0,
                    borderType=cv2.BORDER_CONSTANT,
                    value=bg,
                )
            normalized.append(x)

        row_gap = np.full((max_h, gap, 3), bg, dtype=np.uint8)
        strip = []
        for j, item in enumerate(normalized):
            strip.append(item)
            if j < len(normalized) - 1:
                strip.append(row_gap)
        chunks.append(np.hstack(strip))

    max_w = max(x.shape[1] for x in chunks)
    prepared = []
    for x in chunks:
        if x.shape[1] < max_w:
            x = cv2.copyMakeBorder(
                x,
                top=0,
                bottom=0,
                left=0,
                right=max_w - x.shape[1],
                borderType=cv2.BORDER_CONSTANT,
                value=bg,
            )
        prepared.append(x)

    col_gap = np.full((gap, max_w, 3), bg, dtype=np.uint8)
    out = []
    for i, strip in enumerate(prepared):
        out.append(strip)
        if i < len(prepared) - 1:
            out.append(col_gap)
    return np.vstack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize side-by-side pred_masks from experiment manager runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/exp-manager-20260324"),
        help="Path containing model subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/visualizations/pred_masks_compare",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of common samples (0 = all).")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sample order before limiting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when --shuffle is used.")
    parser.add_argument("--sheet-samples", type=int, default=24, help="How many comparisons to include in the contact sheet.")
    parser.add_argument("--sheet-cols", type=int, default=3, help="Columns in the contact sheet grid.")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    output_dir = args.output_dir.resolve() if args.output_dir else (root / "visualizations" / "pred_masks_compare")
    comparisons_dir = output_dir / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    model_pred_dirs = _discover_pred_mask_dirs(root)
    if not model_pred_dirs:
        raise RuntimeError(f"No model folders with test/pred_masks were found in: {root}")

    model_names = sorted(model_pred_dirs.keys())
    mask_sets = []
    for name in model_names:
        names = {p.name for p in model_pred_dirs[name].glob("*.png")}
        mask_sets.append(names)

    common_names = sorted(set.intersection(*mask_sets))
    if not common_names:
        raise RuntimeError("No common PNG mask filenames across model folders.")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(common_names)

    if args.max_samples > 0:
        common_names = common_names[: args.max_samples]

    summary_path = output_dir / "summary.csv"
    html_path = output_dir / "index.html"

    summary_rows: list[dict[str, str | int | float]] = []
    preview_rows: list[np.ndarray] = []

    for idx, file_name in enumerate(common_names, start=1):
        tiles: list[np.ndarray] = []
        row: dict[str, str | int | float] = {
            "sample": file_name,
            "comparison_image": f"comparisons/{file_name}",
        }

        for model_name in model_names:
            mask_path = model_pred_dirs[model_name] / file_name
            tile, pos_px, pos_ratio = _label_tile(mask_path, model_name)
            tiles.append(tile)
            row[f"{model_name}_px"] = pos_px
            row[f"{model_name}_ratio"] = round(pos_ratio, 6)

        combined = _hstack_padded(tiles)
        cv2.putText(
            combined,
            f"sample {idx}/{len(common_names)}: {file_name}",
            (10, combined.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        out_path = comparisons_dir / file_name
        cv2.imwrite(str(out_path), combined)
        summary_rows.append(row)

        if len(preview_rows) < max(1, args.sheet_samples):
            preview_rows.append(combined)

    fieldnames = ["sample", "comparison_image"]
    for model_name in model_names:
        fieldnames.extend([f"{model_name}_px", f"{model_name}_ratio"])

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    contact_sheet_path = output_dir / "contact_sheet.png"
    sheet = _build_contact_sheet(preview_rows, cols=args.sheet_cols)
    cv2.imwrite(str(contact_sheet_path), sheet)

    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html lang=\"en\"><head><meta charset=\"utf-8\">\n")
        f.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
        f.write("<title>Pred Mask Comparison</title>\n")
        f.write("<style>\n")
        f.write("body{font-family:Segoe UI,Arial,sans-serif;background:#0f1115;color:#e8ecf1;margin:24px;}\n")
        f.write("h1{margin:0 0 8px 0;} p{color:#b5bec8;} .card{background:#171b22;border:1px solid #2a303b;padding:16px;border-radius:10px;margin:14px 0;}\n")
        f.write("img{max-width:100%;height:auto;border-radius:8px;border:1px solid #2a303b;}\n")
        f.write("table{border-collapse:collapse;width:100%;font-size:13px;} th,td{border:1px solid #2a303b;padding:6px 8px;text-align:left;} th{background:#1f2530;}\n")
        f.write("a{color:#77b7ff;}\n")
        f.write("</style></head><body>\n")
        f.write("<h1>Prediction Mask Comparison</h1>\n")
        f.write(f"<p>Root: {root} <br> Models: {', '.join(model_names)} <br> Samples: {len(common_names)}</p>\n")
        f.write("<div class=\"card\"><h2>Contact Sheet</h2><img src=\"contact_sheet.png\" alt=\"contact sheet\"></div>\n")
        f.write("<div class=\"card\"><h2>Sample Comparisons</h2><table><thead><tr><th>Sample</th><th>Image</th></tr></thead><tbody>\n")
        for row in summary_rows:
            sample = str(row["sample"])
            rel = str(row["comparison_image"])
            f.write(f"<tr><td>{sample}</td><td><a href=\"{rel}\">open</a></td></tr>\n")
        f.write("</tbody></table></div>\n")
        f.write("</body></html>\n")

    print(f"Saved comparison gallery to: {output_dir}")
    print(f"- HTML: {html_path}")
    print(f"- Contact sheet: {contact_sheet_path}")
    print(f"- Per-sample comparisons: {comparisons_dir}")
    print(f"- Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
