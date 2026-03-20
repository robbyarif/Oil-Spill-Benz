import os
import torch
import numpy as np
from trainer import BaseTrainer
from utils import contours2mask, image2label
import cv2
import supervision as sv

# Import from external rfdetr library
try:
    from rfdetr import RFDETRSegNano
except ImportError:
    raise ImportError(
        "RF-DETR library not found. Please install the rfdetr package."
    )


class RFDETRTrainer(BaseTrainer):
    """RF-DETR trainer following BaseTrainer interface."""
    
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = None
        self.train_args = {
            "epochs": 100,
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 1e-4,
            "gradient_checkpointing": False,
            "resolution": None,
            "num_select": 300,
        }
    
    def load_model(self, weights=None):
        """Load RF-DETR model."""
        if weights is not None:
            self.checkpoint_path = weights
            self.model = RFDETRSegNano(pretrain_weights=weights, device=self.device)
            self.model.optimize_for_inference()
            print(f"Model loaded from checkpoint: {weights}")
        else:
            self.model = RFDETRSegNano(device=self.device)
            print("Model initialized without pretrained weights")
    
    def _train(self, src, dst=None, save=True, **kwargs):
        """Train RF-DETR model."""
        args = {**self.train_args, **kwargs}
        
        # Ensure dataset is in COCO format
        dataset_dir = src
        if not os.path.exists(os.path.join(dataset_dir, "annotations")):
            raise Exception(f"COCO format dataset required. Expected 'annotations' folder in {dataset_dir}")
        
        output_dir = dst if (save and dst) else "runs/rfdetr_temp"
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract wandb settings if provided
        wandb_enabled = kwargs.get('wandb', False)
        project = kwargs.get('project', 'Oil-Spill-UAV')
        run_name = kwargs.get('run', None)
        
        # Train model using external library
        self.model.train(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            wandb=wandb_enabled,
            project=project,
            run=run_name,
            **args
        )
        
        # Store checkpoint path for inference
        self.checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    
    def _predict(self, src, file_name="test.txt"):
        """Run predictions on test set."""
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
            
            # Run prediction using rfdetr library's predict method
            result = self.model.predict(img_path)
            
            # Get image info
            image = cv2.imread(img_path)
            img_h, img_w = image.shape[:2]
            img_size = (img_h, img_w)
            img_name = os.path.basename(img_path).rsplit('.', maxsplit=1)[0]
            
            # Process prediction masks from rfdetr output
            if result.mask is not None:
                # print("Mask shape:", result.mask.shape)
                pred_mask = np.any(result.mask, axis=0)  # Combine all instance masks into one binary mask
                # print("Combined mask shape:", pred_mask.shape)
                pred_mask = (pred_mask > 0).astype(np.uint8)

            
            # Load ground truth
            label_path = image2label(img_path, lbl_ext=".txt")
            if not os.path.exists(label_path):
                print(f"Warning: Label file '{label_path}' not found. Using empty mask.")
                gt_mask = np.zeros(img_size, dtype=np.uint8)
            else:
                # Load YOLO format labels
                from utils import yolo_label2contours
                gt_contours = yolo_label2contours(label_path, img_size)
                gt_mask = contours2mask(gt_contours, img_size)
                gt_mask = (gt_mask > 0).astype(np.uint8)
            
            self.results.append((img_name, pred_mask, gt_mask))


def main():
    
    # """Example usage."""
    # # Training
    trainer = RFDETRTrainer()
    trainer.load_model(weights="runs/dv4_random_split_rfdetrsegnano/checkpoint_best_regular.pth")
    
    # # Train on dataset
    # trainer.train(
    #     "datasets/processed/exp2.4-dv3train_organized_coco",
    #     dst="runs/rfdetr_experiment",
    #     epochs=100
    # )
    
    # Test
    # trainer.test(
    #     "datasets/processed/exp2.4-dv3train_organized_coco",
    #     dst="runs/rfdetr_experiment/test",
    #     file_name="test.txt"
    # )

    trainer.test(
        "/home/robby/workspace/Oil-Spill-Benz/datasets/new_baseline",
        dst="runs/dv4_random_split_rfdetrsegnano/test",
        file_name="test.txt",
        color_coded=True,
        log=True,
        save=True,
    )


if __name__ == "__main__":
    main()
