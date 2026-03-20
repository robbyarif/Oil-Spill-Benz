#!/usr/bin/env python3
"""
YOLO Annotation Visualization Script
Visualizes YOLO format annotations (bounding boxes and segmentation masks) on images.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> List[dict]:
    """
    Parse YOLO format label file.
    
    Args:
        label_path: Path to YOLO label file
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of annotation dictionaries containing class_id and polygon points
    """
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            class_id = int(parts[0])
            # Remaining parts are normalized polygon coordinates (x1 y1 x2 y2 ...)
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    points.append((x, y))
            
            annotations.append({
                'class_id': class_id,
                'points': points
            })
    
    return annotations


def get_color(class_id: int, seed: Optional[int] = None) -> Tuple[int, int, int]:
    """
    Get a consistent color for a given class ID.
    
    Args:
        class_id: Class ID
        seed: Random seed for color generation
        
    Returns:
        BGR color tuple
    """
    if seed is not None:
        random.seed(class_id + seed)
    else:
        random.seed(class_id)
    
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def visualize_annotations(
    image_path: Path,
    label_path: Path,
    class_names: dict,
    output_path: Optional[Path] = None,
    alpha: float = 0.4,
    thickness: int = 2,
    show_labels: bool = True
) -> np.ndarray:
    """
    Visualize YOLO annotations on an image.
    
    Args:
        image_path: Path to image file
        label_path: Path to YOLO label file
        class_names: Dictionary mapping class IDs to class names
        output_path: Optional path to save the visualization
        alpha: Transparency for filled polygons (0-1)
        thickness: Line thickness for polygon borders
        show_labels: Whether to show class labels
        
    Returns:
        Annotated image as numpy array
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    # Parse annotations
    annotations = parse_yolo_label(label_path, img_width, img_height)
    
    # Create overlay for transparent fill
    overlay = img.copy()
    
    # Draw each annotation
    for ann in annotations:
        class_id = ann['class_id']
        points = ann['points']
        
        if len(points) < 3:
            continue
        
        # Get color for this class
        color = get_color(class_id)
        
        # Convert points to numpy array
        pts = np.array(points, dtype=np.int32)
        
        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [pts], color)
        
        # Draw polygon border on original image
        cv2.polylines(img, [pts], True, color, thickness)
        
        # Add label if requested
        if show_labels:
            class_name = class_names.get(class_id, f"Class {class_id}")
            
            # Calculate centroid for label placement
            moments = cv2.moments(pts)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = points[0]
            
            # Draw label background
            label_text = f"{class_name}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                img,
                (cx - 5, cy - text_height - 10),
                (cx + text_width + 5, cy + 5),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img,
                label_text,
                (cx, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    # Blend overlay with original image
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        print(f"Saved visualization to: {output_path}")
    
    return result


def visualize_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: dict,
    max_images: Optional[int] = None,
    alpha: float = 0.4,
    thickness: int = 2,
    show_labels: bool = True
):
    """
    Visualize annotations for an entire dataset.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        output_dir: Directory to save visualizations
        class_names: Dictionary mapping class IDs to class names
        max_images: Maximum number of images to process (None for all)
        alpha: Transparency for filled polygons (0-1)
        thickness: Line thickness for polygon borders
        show_labels: Whether to show class labels
    """
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    processed = 0
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            print(f"Warning: No label file for {image_path.name}, skipping")
            continue
        
        output_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        
        try:
            visualize_annotations(
                image_path,
                label_path,
                class_names,
                output_path,
                alpha,
                thickness,
                show_labels
            )
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{len(image_files)} images")
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
    
    print(f"\nCompleted! Processed {processed} images")
    print(f"Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO format annotations on images"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image file"
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Path to a single label file"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing images"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        help="Directory containing label files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path (file or directory)"
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs='+',
        default=['oil'],
        help="Class names in order (default: oil)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process (for batch mode)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Transparency for filled polygons (0-1, default: 0.4)"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Line thickness for polygon borders (default: 2)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't show class labels on annotations"
    )
    
    args = parser.parse_args()
    
    # Create class names dictionary
    class_names = {i: name for i, name in enumerate(args.class_names)}
    
    # Single image mode
    if args.image and args.label:
        image_path = Path(args.image)
        label_path = Path(args.label)
        output_path = Path(args.output)
        
        result = visualize_annotations(
            image_path,
            label_path,
            class_names,
            output_path,
            args.alpha,
            args.thickness,
            not args.no_labels
        )
        
        print("Visualization complete!")
    
    # Batch mode
    elif args.images_dir and args.labels_dir:
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir)
        output_dir = Path(args.output)
        
        visualize_dataset(
            images_dir,
            labels_dir,
            output_dir,
            class_names,
            args.max_images,
            args.alpha,
            args.thickness,
            not args.no_labels
        )
    
    else:
        parser.error("Must provide either (--image and --label) or (--images-dir and --labels-dir)")


if __name__ == "__main__":
    main()