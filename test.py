from ultralytics import YOLO
from utils import *
import os
from tqdm import tqdm

def analyze_results(model, src, dst):
    metrics = {}

    val_metric = model.val(data=f"{src}/data.yaml", verbose=False, save=False)
    precision = val_metric.results_dict['metrics/precision(M)']
    recall = val_metric.results_dict['metrics/recall(M)']
    f1 = 2 * (precision * recall) / (precision + recall)
    metrics["Precision"], metrics["Recall"], metrics["F1"] = precision, recall, f1

    results = model.predict(source=f"{src}/images/test")
    acc = 0
    oil_iou, bg_iou = 0, 0
    oil_num, bg_num = 0, 0
    for result in tqdm(results, total=len(results), desc="Analyzing Results"):
        img_sz = result.orig_shape
        img_name = os.path.basename(result.path)

        # Get model predicts contours
        pred_contours = []
        if hasattr(result, 'masks') and result.masks is not None:
            pred_contours = result.masks.xy

        # Get ground truth contours
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        label_path = sb.join(result.path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"
        gt_contours = yolo_label2contours(label_path, img_sz)

        pred_mask = contours2mask(pred_contours, img_sz)
        gt_mask = contours2mask(gt_contours, img_sz)

        acc += get_acc(pred_mask, gt_mask)

        # Calculate oil and background IoU
        oil_buf, bg_buf = get_iou(pred_mask, gt_mask)
        if oil_buf is not None:
            oil_iou += oil_buf
            oil_num += 1
        if bg_buf is not None:
            bg_iou += bg_buf
            bg_num += 1

        # Draw coded mask
        colors = [
            # channel = (B, G, R)
            (0, 255, 0),  # TP
            (0, 0, 255),  # FP
            (0, 255, 255),  # FN
            (128, 128, 128)  # TN
        ]
        coded_mask = get_coded_mask(pred_mask, gt_mask, colors)
        img_dst = os.path.join(dst, f"IoU={oil_buf:.3f}_{img_name}")
        cv2.imwrite(img_dst, coded_mask)

    acc /= len(results)
    oil_iou = oil_iou / oil_num if oil_num else None
    bg_iou = bg_iou / bg_num if bg_num else None
    mean_iou = (oil_iou + bg_iou) / 2
    metrics["Acc"], metrics["Oil Iou"], metrics["BG IoU"], metrics["mIoU"] = acc, oil_iou, bg_iou, mean_iou
    return metrics

def main():
    model = YOLO("runs/segment/baseline/weights/best.pt")
    metrics = analyze_results(model, src="datasets/baseline", dst="runs/predict/baseline")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    return 0

if __name__ == "__main__":
    main()