import os
from typing import Optional

import numpy as np
import torch

from trainer import BaseTrainer
from utils import contours2mask, image2label

try:
    import rfdetr
except ImportError:
    raise ImportError(
        "RF-DETR library not found. Please install the rfdetr package."
    )

from coco_converter import prepare_coco_dataset


class RFDETRTrainer(BaseTrainer):
    """RF-DETR trainer following BaseTrainer interface."""
    
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = None
        self.train_args = {
            "epochs": 100,
            "batch_size": 4,
            "grad_accum_steps": 4,
            "lr": 1e-4,
        }
    
    def load_model(self, weights: Optional[str] = None, model: str = "nano"):
        """Load RF-DETR segmentation model.
        
        Args:
            weights: Path to checkpoint weights (optional)
            model: Model size - nano, small, medium, large, xlarge, 2xlarge (default: nano)
        """
        # Dynamically get model class from rfdetr module
        model_class_name = f"RFDETRSeg{model.capitalize()}"
        if not hasattr(rfdetr, model_class_name):
            available = [
                "nano", "small", "medium", "large", "xlarge", "2xlarge"
            ]
            raise ValueError(
                f"Model '{model}' not found. Available options: {available}"
            )
        
        model_class = getattr(rfdetr, model_class_name)
        
        if weights is not None:
            self.checkpoint_path = weights
            self.model = model_class(pretrain_weights=weights, device=self.device)
            self.model.optimize_for_inference()
            print(f"Model {model_class_name} loaded from checkpoint: {weights}")
        else:
            self.model = model_class(device=self.device)
            print(f"Model {model_class_name} initialized without pretrained weights")
    
    def _train(self, src, dst=None, save=True, **kwargs):
        """Train RF-DETR model."""
        args = {**self.train_args, **kwargs}
        
        # Convert dataset to COCO format if needed
        dataset_dir = prepare_coco_dataset(src)
        if not os.path.exists(os.path.join(dataset_dir, "train", "_annotations.coco.json")):
            raise Exception(f"Failed to create COCO format dataset in {dataset_dir}")
        
        output_dir = dst if (save and dst) else "runs/rfdetr_temp"
        os.makedirs(output_dir, exist_ok=True)
        
        # Train model using external library
        self.model.train(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            **args
        )
        
        # Store checkpoint path for inference
        self.checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

        # Clean up temporary COCO dataset if it was created
        if dataset_dir.endswith("coco_format_temp"):
            import shutil
            shutil.rmtree(dataset_dir)
    
    def _predict(self, src, file_name="test.txt"):
        """Run predictions on test set."""
        import cv2
        from utils import yolo_label2contours
        
        test_file = os.path.join(src, file_name)
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        with open(test_file, "r", encoding="utf-8") as f:
            image_list = [line.strip() for line in f]
        
        self.results.clear()
        
        for img_path in image_list:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Run prediction
            result = self.model.predict(img_path)
            
            # Get image dimensions
            image = cv2.imread(img_path)
            img_h, img_w = image.shape[:2]
            img_size = (img_h, img_w)
            
            # Process prediction masks
            if result.mask is not None:
                pred_mask = np.any(result.mask, axis=0)
                pred_mask = (pred_mask > 0).astype(np.uint8)
            
            # Load ground truth
            label_path = image2label(img_path, lbl_ext=".txt")
            if not os.path.exists(label_path):
                print(f"Warning: Label file '{label_path}' not found. Using empty mask.")
                gt_mask = np.zeros(img_size, dtype=np.uint8)
            else:
                gt_contours = yolo_label2contours(label_path, img_size)
                gt_mask = contours2mask(gt_contours, img_size)
                gt_mask = (gt_mask > 0).astype(np.uint8)
            
            self.results.append((img_path, pred_mask, gt_mask))


def main():
    # """Example usage for testing."""
    # trainer = RFDETRTrainer()
    # trainer.load_model(
    #     weights="runs/dv4_random_split_rfdetrsegnano/checkpoint_best_regular.pth"
    # )
    
    # trainer.test(
    #     "/home/robby/workspace/Oil-Spill-Benz/datasets/new_baseline",
    #     dst="runs/dv4_random_split_rfdetrsegnano/test",
    #     file_name="test.txt",
    #     color_coded=True,
    #     log=True,
    #     save=True,
    # )

    ...

if __name__ == "__main__":
    main()
