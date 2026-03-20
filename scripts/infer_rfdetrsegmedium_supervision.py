import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch

try:
    from rfdetr import RFDETRSegMedium
except ImportError as exc:
    raise ImportError("RF-DETR library not found. Please install the rfdetr package.") from exc


PALETTE_HEX = [
    "#ff0000", "#ff9b00", "#ff8000", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-DETR SegMedium inference with Supervision overlays and class labels."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/lados_rfdetrsegmedium/checkpoint_best_regular.pth"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("datasets/new_baseline/test.txt"),
        help="Path to split text file (one image path per line).",
    )
    parser.add_argument(
        "--coco-annotation",
        type=Path,
        default=Path("datasets/lados_432/test/_annotations.coco.json"),
        help="COCO annotation file used to resolve class names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/lados_rfdetrsegmedium/inference_supervision"),
        help="Directory for annotated outputs.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory to resolve relative image paths from split file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for inference (cuda/cpu).",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.25,
        help="Minimum confidence score to keep a detection.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process at most N images (0 means all).",
    )
    return parser.parse_args()


def load_categories(coco_path: Path) -> Dict[int, str]:
    with coco_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)
    categories = coco_data.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in COCO annotation: {coco_path}")
    categories = sorted(categories, key=lambda cat: int(cat["id"]))
    return {int(cat["id"]): str(cat["name"]) for cat in categories}


def read_split_file(split_path: Path) -> List[str]:
    with split_path.open("r", encoding="utf-8") as file:
        rows = [line.strip() for line in file if line.strip()]
    if not rows:
        raise ValueError(f"Split file is empty: {split_path}")
    return rows


def resolve_image_path(raw_path: str, base_dir: Path, split_parent: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    first = (base_dir / candidate).resolve()
    if first.exists():
        return first

    second = (split_parent / candidate).resolve()
    if second.exists():
        return second

    return first


def to_numpy(value: Any) -> Optional[np.ndarray]:
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


def extract_prediction_arrays(prediction: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    masks = to_numpy(getattr(prediction, "mask", None))
    class_ids = to_numpy(getattr(prediction, "class_id", None))
    confidences = to_numpy(getattr(prediction, "confidence", None))
    boxes = to_numpy(getattr(prediction, "xyxy", None))

    if masks is None and isinstance(prediction, dict):
        masks = to_numpy(prediction.get("mask") or prediction.get("masks"))
        class_ids = to_numpy(prediction.get("class_id") or prediction.get("class_ids"))
        confidences = to_numpy(prediction.get("confidence") or prediction.get("scores"))
        boxes = to_numpy(prediction.get("xyxy") or prediction.get("boxes"))

    if masks is None:
        empty_masks = np.zeros((0, 1, 1), dtype=np.uint8)
        empty_ids = np.zeros((0,), dtype=np.int64)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return empty_masks, empty_ids, empty_scores, None

    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    num = int(masks.shape[0])

    if class_ids is None:
        class_ids = np.zeros((num,), dtype=np.int64)
    class_ids = np.asarray(class_ids).reshape(-1).astype(np.int64)
    if class_ids.shape[0] < num:
        class_ids = np.pad(class_ids, (0, num - class_ids.shape[0]), mode="edge")

    if confidences is None:
        confidences = np.ones((num,), dtype=np.float32)
    confidences = np.asarray(confidences).reshape(-1).astype(np.float32)
    if confidences.shape[0] < num:
        confidences = np.pad(confidences, (0, num - confidences.shape[0]), mode="edge")

    if boxes is not None:
        boxes = np.asarray(boxes)
        if boxes.ndim == 1:
            boxes = boxes.reshape(-1, 4)
        if boxes.shape[0] < num:
            boxes = None

    return masks, class_ids[:num], confidences[:num], boxes


def masks_to_xyxy(masks_bool: np.ndarray) -> np.ndarray:
    xyxy: List[List[float]] = []
    for mask in masks_bool:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            xyxy.append([0.0, 0.0, 0.0, 0.0])
            continue
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        xyxy.append([x1, y1, x2, y2])
    return np.asarray(xyxy, dtype=np.float32)


def build_detections(
    masks: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    boxes: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    score_thr: float,
) -> sv.Detections:
    h, w = image_shape

    selected_masks: List[np.ndarray] = []
    selected_ids: List[int] = []
    selected_scores: List[float] = []
    selected_boxes: List[List[float]] = []

    for idx in range(len(masks)):
        score = float(confidences[idx])
        if score < score_thr:
            continue

        mask = masks[idx]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask > 0.5

        selected_masks.append(mask)
        selected_ids.append(int(class_ids[idx]))
        selected_scores.append(score)

        if boxes is not None:
            b = boxes[idx].tolist()
            selected_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    if not selected_masks:
        return sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            mask=np.zeros((0, h, w), dtype=bool),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int64),
        )

    masks_bool = np.asarray(selected_masks, dtype=bool)
    if boxes is None:
        xyxy = masks_to_xyxy(masks_bool)
    else:
        xyxy = np.asarray(selected_boxes, dtype=np.float32)

    return sv.Detections(
        xyxy=xyxy,
        mask=masks_bool,
        confidence=np.asarray(selected_scores, dtype=np.float32),
        class_id=np.asarray(selected_ids, dtype=np.int64),
    )


def annotate(
    image_bgr: np.ndarray,
    detections: sv.Detections,
    classes: Dict[int, str],
) -> np.ndarray:
    color = sv.ColorPalette.from_hex(PALETTE_HEX)
    h, w = image_bgr.shape[:2]
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))

    mask_annotator = sv.MaskAnnotator(color=color)
    polygon_annotator = sv.PolygonAnnotator(color=sv.Color.WHITE)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_position=sv.Position.CENTER_OF_MASS,
    )

    labels = [
        f"{classes.get(int(class_id), 'unknown')} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    out = image_bgr.copy()
    out = mask_annotator.annotate(scene=out, detections=detections)
    out = polygon_annotator.annotate(scene=out, detections=detections)
    out = label_annotator.annotate(scene=out, detections=detections, labels=labels)
    return out


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
        raise FileNotFoundError(f"COCO annotation not found: {coco_annotation}")

    output_dir.mkdir(parents=True, exist_ok=True)

    class_map = load_categories(coco_annotation)
    image_list = read_split_file(split_file)
    if args.max_images > 0:
        image_list = image_list[: args.max_images]

    model = RFDETRSegMedium(pretrain_weights=str(checkpoint), device=args.device)
    model.optimize_for_inference()

    results: List[Dict[str, Any]] = []
    missing: List[str] = []

    for raw_path in image_list:
        image_path = resolve_image_path(raw_path, base_dir=base_dir, split_parent=split_file.parent)
        if not image_path.exists():
            missing.append(raw_path)
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            missing.append(raw_path)
            continue

        prediction = model.predict(str(image_path))
        masks, class_ids, confidences, boxes = extract_prediction_arrays(prediction)

        detections = build_detections(
            masks=masks,
            class_ids=class_ids,
            confidences=confidences,
            boxes=boxes,
            image_shape=image_bgr.shape[:2],
            score_thr=args.score_thr,
        )

        annotated = annotate(image_bgr=image_bgr, detections=detections, classes=class_map)

        output_image_path = output_dir / f"{image_path.stem}_pred.jpg"
        cv2.imwrite(str(output_image_path), annotated)

        image_items = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            image_items.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_map.get(int(class_id), "unknown"),
                    "score": float(confidence),
                }
            )

        results.append(
            {
                "image": str(image_path),
                "output_image": str(output_image_path),
                "num_detections": int(len(detections)),
                "detections": image_items,
            }
        )

    summary = {
        "checkpoint": str(checkpoint),
        "split_file": str(split_file),
        "coco_annotation": str(coco_annotation),
        "device": args.device,
        "score_thr": args.score_thr,
        "num_images_requested": len(image_list),
        "num_images_missing_or_failed": len(missing),
        "missing_or_failed": missing,
        "class_map": class_map,
    }

    with (output_dir / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Done. Annotated images saved in: {output_dir}")
    print(f"Predictions: {output_dir / 'predictions.json'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
