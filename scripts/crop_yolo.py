import cv2
import numpy as np
import os
import argparse

def crop_from_yolo_seg(image_path, label_path, output_dir="."):
    # Load the image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    h, w, _ = img.shape
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    image_stem = os.path.splitext(os.path.basename(image_path))[0]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    all_polygons = []
    x_points = []
    y_points = []

    for line in lines:
        if not line.strip():
            continue

        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        coords = parts[1:]
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        # Separate x and y, then denormalize
        xs = [coords[j] * w for j in range(0, len(coords), 2)]
        ys = [coords[j+1] * h for j in range(0, len(coords), 2)]

        all_polygons.append((class_id, xs, ys))
        x_points.extend(xs)
        y_points.extend(ys)

    if not all_polygons:
        return

    # Get outer extent across all polygons
    x_min, x_max = int(min(x_points)), int(max(x_points))
    y_min, y_max = int(min(y_points)), int(max(y_points))

    # Convert bbox to a square crop region (1:1) without adding border padding
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h)
    side = min(side, w, h)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    x_min = int(round(cx - side / 2.0))
    y_min = int(round(cy - side / 2.0))
    x_max = x_min + side
    y_max = y_min + side

    # Shift square window to stay fully inside the source image
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if x_max > w:
        shift = x_max - w
        x_min -= shift
        x_max = w
    if y_max > h:
        shift = y_max - h
        y_min -= shift
        y_max = h

    x_min = max(0, x_min)
    y_min = max(0, y_min)

    # Crop image once using the square extent
    crop = img[y_min:y_max, x_min:x_max]

    # Build full-size segmentation mask from all polygons and crop it
    full_mask = np.zeros((h, w), dtype=np.uint8)
    for _, xs, ys in all_polygons:
        polygon = np.array(
            [[int(round(xs[j])), int(round(ys[j]))] for j in range(len(xs))],
            dtype=np.int32
        )
        cv2.fillPoly(full_mask, [polygon], 255)
    crop_mask = full_mask[y_min:y_max, x_min:x_max]

    base_name = f"{image_stem}"
    img_out_path = os.path.join(images_dir, f"{base_name}.jpg")
    mask_out_path = os.path.join(labels_dir, f"{base_name}.png")
    label_out_path = os.path.join(labels_dir, f"{base_name}.txt")

    cv2.imwrite(img_out_path, crop)
    cv2.imwrite(mask_out_path, crop_mask)

    # Save all polygons as YOLO segmentation labels in crop-local coordinates
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    label_lines = []
    for class_id, xs, ys in all_polygons:
        local_coords = []
        for x, y in zip(xs, ys):
            lx = (x - x_min) / crop_w
            ly = (y - y_min) / crop_h
            local_coords.extend([min(1.0, max(0.0, lx)), min(1.0, max(0.0, ly))])
        label_lines.append(
            str(class_id) + " " + " ".join(f"{c:.6f}" for c in local_coords)
        )

    with open(label_out_path, 'w') as lf:
        lf.write("\n".join(label_lines) + "\n")

def crop_from_yolo_seg_dir(images_dir, labels_dir, output_dir="."):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_names = sorted(
        name for name in os.listdir(images_dir)
        if os.path.splitext(name)[1].lower() in valid_exts
    )

    if not image_names:
        print(f"No images found in {images_dir}")
        return

    processed = 0
    skipped = 0

    for image_name in image_names:
        image_path = os.path.join(images_dir, image_name)
        image_stem = os.path.splitext(image_name)[0]
        label_path = os.path.join(labels_dir, f"{image_stem}.txt")

        if not os.path.exists(label_path):
            skipped += 1
            continue

        crop_from_yolo_seg(image_path, label_path, output_dir)
        processed += 1

    print(f"Processed: {processed} images")
    print(f"Skipped (missing labels): {skipped} images")


def crop_from_dataset_dir(dataset_dir, output_dir="."):
    split_names = ["train", "test", "valid"]

    for split_name in split_names:
        split_dir = os.path.join(dataset_dir, split_name)
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")

        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            print(f"Skip split '{split_name}': missing images/labels folder")
            continue

        split_output_dir = os.path.join(output_dir, split_name)
        print(f"Processing split: {split_name}")
        crop_from_yolo_seg_dir(images_dir, labels_dir, split_output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Crop YOLO segmentation data for dataset dir with train/test/valid splits"
    )
    parser.add_argument(
        "--dataset-dir",
        help="Input dataset directory with train/test/valid, each containing images/ and labels/",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output root directory (creates train/test/valid, each with images/ and labels/)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        raise NotADirectoryError(f"Invalid dataset directory: {args.dataset_dir}")

    crop_from_dataset_dir(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()