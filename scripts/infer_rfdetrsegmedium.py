import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from rfdetr import RFDETRSegMedium
except ImportError as exc:
    raise ImportError("RF-DETR library not found. Please install the rfdetr package.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR SegMedium inference on a split file using COCO category names."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/lados_rfdetrsegmedium/checkpoint_best_regular.pth"),
        help="Path to RF-DETR checkpoint (.pth).",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("datasets/new_baseline/test.txt"),
        help="Text file containing one image path per line.",
    )
    parser.add_argument(
        "--coco-annotation",
        type=Path,
        default=Path("datasets/lados_432/test/_annotations.coco.json"),
        help="COCO annotation JSON used to load class IDs/names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/lados_rfdetrsegmedium/inference_new_baseline_test"),
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory used to resolve relative image paths from split file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Inference device (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.25,
        help="Minimum confidence score for keeping detections.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualization images with segmentation overlays.",
    )
    return parser.parse_args()


def load_categories(coco_path: Path) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
    with coco_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    categories = coco_data.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in COCO file: {coco_path}")

    categories = sorted(categories, key=lambda cat: int(cat["id"]))
    id_to_name = {int(cat["id"]): str(cat["name"]) for cat in categories}
    return id_to_name, categories


def read_split_file(split_file: Path) -> List[str]:
    with split_file.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    if not lines:
        raise ValueError(f"Split file is empty: {split_file}")
    return lines


def resolve_image_path(raw_path: str, base_dir: Path, split_parent: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    option1 = (base_dir / candidate).resolve()
    if option1.exists():
        return option1

    option2 = (split_parent / candidate).resolve()
    if option2.exists():
        return option2

    return option1


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:
            return None
    return np.asarray(value)


def _extract_predictions(prediction: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = _to_numpy(getattr(prediction, "mask", None))
    class_ids = _to_numpy(getattr(prediction, "class_id", None))
    scores = _to_numpy(getattr(prediction, "confidence", None))

    if masks is None and isinstance(prediction, dict):
        masks = _to_numpy(prediction.get("mask") or prediction.get("masks"))
        class_ids = _to_numpy(prediction.get("class_id") or prediction.get("class_ids"))
        scores = _to_numpy(prediction.get("confidence") or prediction.get("scores"))

    if masks is None:
        return np.zeros((0, 1, 1), dtype=np.uint8), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]

    num_instances = int(masks.shape[0])

    if class_ids is None:
        class_ids = np.zeros((num_instances,), dtype=np.int64)
    class_ids = np.asarray(class_ids).reshape(-1).astype(np.int64)

    if scores is None:
        scores = np.ones((num_instances,), dtype=np.float32)
    scores = np.asarray(scores).reshape(-1).astype(np.float32)

    if class_ids.shape[0] < num_instances:
        class_ids = np.pad(class_ids, (0, num_instances - class_ids.shape[0]), mode="edge")
    if scores.shape[0] < num_instances:
        scores = np.pad(scores, (0, num_instances - scores.shape[0]), mode="edge")

    return masks, class_ids[:num_instances], scores[:num_instances]


def _stable_color(class_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=class_id + 1234)
    color = rng.integers(64, 256, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def visualize_predictions(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    class_name_map: Dict[int, str],
    score_thr: float,
) -> np.ndarray:
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]

    for index in range(len(masks)):
        score = float(scores[index])
        if score < score_thr:
            continue

        class_id = int(class_ids[index])
        class_name = class_name_map.get(class_id, f"cls_{class_id}")

        mask = masks[index]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5)

        color = _stable_color(class_id)
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array(color)).astype(np.uint8)

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x_min, y_min = int(xs.min()), int(ys.min())
        label = f"{class_name}:{score:.2f}"
        cv2.putText(
            overlay,
            label,
            (x_min, max(y_min - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay


def prediction_to_json_items(
    image_path: Path,
    masks: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    class_name_map: Dict[int, str],
    score_thr: float,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for index in range(len(masks)):
        score = float(scores[index])
        if score < score_thr:
            continue

        mask = masks[index]
        mask_bin = (mask > 0.5).astype(np.uint8)
        area = int(mask_bin.sum())

        items.append(
            {
                "image": str(image_path),
                "class_id": int(class_ids[index]),
                "class_name": class_name_map.get(int(class_ids[index]), f"cls_{int(class_ids[index])}"),
                "score": score,
                "mask_area": area,
            }
        )
    return items


def main() -> None:
    args = parse_args()

    checkpoint = args.checkpoint.resolve()
    split_file = args.split_file.resolve()
    coco_annotation = args.coco_annotation.resolve()
    output_dir = args.output_dir.resolve()
    base_dir = args.base_dir.resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not coco_annotation.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_annotation}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    class_name_map, categories = load_categories(coco_annotation)
    image_paths_raw = read_split_file(split_file)

    model = RFDETRSegMedium(pretrain_weights=str(checkpoint), device=args.device)
    model.optimize_for_inference()

    all_predictions: List[Dict[str, Any]] = []
    missing_images: List[str] = []

    for raw_path in image_paths_raw:
        image_path = resolve_image_path(raw_path, base_dir=base_dir, split_parent=split_file.parent)
        if not image_path.exists():
            missing_images.append(raw_path)
            continue

        prediction = model.predict(str(image_path))
        masks, class_ids, scores = _extract_predictions(prediction)

        items = prediction_to_json_items(
            image_path=image_path,
            masks=masks,
            class_ids=class_ids,
            scores=scores,
            class_name_map=class_name_map,
            score_thr=args.score_thr,
        )
        all_predictions.extend(items)

        if args.save_vis:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            vis = visualize_predictions(
                image_bgr=image,
                masks=masks,
                class_ids=class_ids,
                scores=scores,
                class_name_map=class_name_map,
                score_thr=args.score_thr,
            )
            vis_name = image_path.stem + "_pred.jpg"
            cv2.imwrite(str(output_dir / "visualizations" / vis_name), vis)

    summary = {
        "checkpoint": str(checkpoint),
        "split_file": str(split_file),
        "coco_annotation": str(coco_annotation),
        "num_images_in_split": len(image_paths_raw),
        "num_missing_images": len(missing_images),
        "missing_images": missing_images,
        "num_predictions_after_threshold": len(all_predictions),
        "categories": categories,
        "score_threshold": args.score_thr,
        "device": args.device,
    }

    with (output_dir / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump(all_predictions, file, indent=2, ensure_ascii=False)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Done. Saved predictions to: {output_dir / 'predictions.json'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    if args.save_vis:
        print(f"Saved visualizations to: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
