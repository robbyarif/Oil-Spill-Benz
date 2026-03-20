import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.yolo as fouy
import cv2

import numpy as np
import os
from tqdm import tqdm

from ultralytics import YOLO

# detection_model = YOLO("yolov8n.pt")
# seg_model = YOLO("yolov8n-seg.pt")

# results = detection_model("https://ultralytics.com/images/bus.jpg")

dataset_dir = "./datasets/baseline_seed_42"

classes = ["Oil Spill"]

# 1. Load the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    label_type="polylines", # Crucial for segmentation
    data_path="images",
    labels_path="labels",
    name="oil-spill-segmentation",
    classes=classes  # Set the classes here
)

# Also set default_classes for the dataset
dataset.default_classes = classes
dataset.save()

print("Ground Truth Dataset:")
print(dataset)

# 2. Load predictions from model output
# This loads the model predictions as a separate field for comparison
predictions_dir = "runs/predict/baseline"  # Directory containing prediction labels
if os.path.exists(predictions_dir):
    fouy.add_yolo_labels(
        dataset,
        label_field="predictions",
        labels_path=f"{predictions_dir}/labels",  # Prediction labels directory
        classes=classes,
        label_type="polylines"
    )
    dataset.save()
    print("\nDataset with Predictions:")
    print(dataset)
else:
    print(f"\nWarning: Predictions directory '{predictions_dir}' not found")
    print("Skipping predictions loading. Run inference first.")

# 3. Evaluate predictions against ground truth
if "predictions" in dataset.get_field_names():
    print("\nEvaluating segmentations...")
    results = dataset.evaluate_segmentations(
        "predictions",
        gt_field="ground_truth",  # Default field from YOLO import
        eval_key="seg_eval",
        compute_mAP=True
    )
    
    print("\nEvaluation Results:")
    results.print_report()
    
    # Compute per-sample metrics for filtering/sorting
    results.compute_per_sample_metrics()
    
    print(f"\nTP: {dataset.sum('seg_eval_tp')}")
    print(f"FP: {dataset.sum('seg_eval_fp')}")
    print(f"FN: {dataset.sum('seg_eval_fn')}")
    
    try:
        mAP = results.mAP()
        print(f"\nmAP: {mAP:.4f}")
    except Exception as e:
        print(f"mAP calculation note: {e}")
    
    # Sort by lowest IoU to see worst performing samples
    print("\n" + "="*60)
    print("Samples with LOWEST IoU (worst predictions):")
    print("="*60)
    
    # Create a view sorted by IoU in ascending order
    worst_view = dataset.sort_by("seg_eval_iou").limit(10)
    for sample in worst_view:
        iou = sample.seg_eval_iou
        print(f"  {sample.filename}: IoU = {iou:.4f}")
    
    print("\n" + "="*60)
    print("Samples with HIGHEST IoU (best predictions):")
    print("="*60)
    
    # Create a view sorted by IoU in descending order
    best_view = dataset.sort_by("seg_eval_iou", reverse=True).limit(10)
    for sample in best_view:
        iou = sample.seg_eval_iou
        print(f"  {sample.filename}: IoU = {iou:.4f}")


# dataset = fo.Dataset.from_dir(
#     dataset_type=fo.types.YOLOv5Dataset,
#     dataset_dir=dataset_dir,
#     data_path="images/default",
#     labels_path="annotations/instances_default.json",
#     label_types="segmentations",
#     label_field="categories",
#     name="coffee",
#     include_id=True,
#     overwrite=True
# )


# View summary info about the dataset
# print(dataset)

# Print the first few samples in the dataset
# print(dataset.head())

# def convert_yolo_segmentations_to_fiftyone(
#     yolo_segmentations,
#     class_list
#     ):

#     detections = []
#     boxes = yolo_segmentations.boxes.xywhn
#     if not boxes.shape or yolo_segmentations.masks is None:
#         return fo.Detections(detections=detections)

#     _uncenter_boxes(boxes)
#     masks = yolo_segmentations.masks.masks
#     labels = _get_class_labels(yolo_segmentations.boxes.cls, class_list)

#     for label, box, mask in zip(labels, boxes, masks):
#         ## convert to absolute indices to index mask
#         w, h = mask.shape
#         tmp =  np.copy(box)
#         tmp[2] += tmp[0]
#         tmp[3] += tmp[1]
#         tmp[0] *= h
#         tmp[2] *= h
#         tmp[1] *= w
#         tmp[3] *= w
#         tmp = [int(b) for b in tmp]
#         y0, x0, y1, x1 = tmp
#         sub_mask = mask[x0:x1, y0:y1]

#         detections.append(
#             fo.Detection(
#                 label=label,
#                 bounding_box = list(box),
#                 mask = sub_mask.astype(bool)
#             )
#         )

#     return fo.Detections(detections=detections)


# model = YOLO("runs/oil-spill/segment/exp1/weights/best.pt")

# dataset.apply_model(model, 
#                     label_field="predictions", 
#                     confidence_thresh=0.5, 
#                     # overlap_thresh=0.5, 
#                     classes=classes)

# # Evaluate segmentation predictions against ground truth
# seg_results = dataset.evaluate_segmentations(
#     "predictions",
#     eval_key="seg_eval",
#     compute_mAP=True,
#     gt_field="ground_truth",
# )

# print("Segmentation evaluation complete.")
# try:
#     seg_map = seg_results.mAP()
#     print(f"Segmentation mAP = {seg_map}")
# except Exception:
#     print("mAP not available for this evaluation method.")

# # Optional: print per-class report (single-class dataset will show one row)
# seg_results.print_report()

# # Per-sample metrics for quick sorting/filtering in the App
# seg_results.compute_per_sample_metrics()
# print("TP: %d" % dataset.sum("seg_eval_tp"))
# print("FP: %d" % dataset.sum("seg_eval_fp"))
# print("FN: %d" % dataset.sum("seg_eval_fn"))

# # detection_results = dataset.evaluate_segmentations(
# #     "predictions",
# #     eval_key="eval",
# #     compute_mAP=True,
# #     gt_field="ground_truth",
# # )

# # # Compute Metrics per Sample (IoU)
# # detection_results.compute_per_sample_metrics()
# # # Print m

# # mAP = detection_results.mAP()
# # print(f"mAP = {mAP}")


# # # Print some statistics about the total TP/FP/FN counts
# # print("TP: %d" % dataset.sum("eval_tp"))
# # print("FP: %d" % dataset.sum("eval_fp"))
# # print("FN: %d" % dataset.sum("eval_fn"))




# # 2. Rename Class '0' to 'Oil Spill'
# # This maps the integer ID found in your text file to a readable string
# dataset.default_classes = ["Oil Spill"] 

# # 3. (Optional) Compute metadata 
# # This helps the App render images with the correct aspect ratio faster
# dataset.compute_metadata()

# # 4. Save and Launch
# dataset.save()
session = fo.launch_app(dataset, remote=True, port=5151)

# # Find unique samples based on image similarity
# import fiftyone.brain as fob

# similarity_results = fob.compute_similarity(dataset, brain_key="img_sim")
# similarity_results.find_unique(20)

# vis_results = fob.compute_visualization(dataset, brain_key="img_vis")

# # Resize all images to a consistent size before flattening to ensure homogeneous shape
# target_size = (224, 224)  # Standard size for embeddings
# embeddings = np.array([
#     cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), target_size).ravel()
#     for f in dataset.values("filepath")
# ])

# custom_vis_results = fob.compute_visualization(
#     dataset,
#     embeddings=embeddings,
#     num_dims=2,
#     method="umap",
#     brain_key="custom_vis",
#     verbose=True,
#     seed=42
# )
# # plot = similarity_results.visualize_unique(visualization=vis_results)
# # plot.show()

# # session.plots.attach(plot, name="unique")
# # session.show()

# # unique_view = dataset.select(similarity_results.unique_ids)
# # session.view = unique_view

# unique_view = dataset.select(similarity_results.unique_ids)
# session.view = unique_view
# dataset.save_view(view=unique_view, name="unique_view")

session.wait()