import csv
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import os
from trainer import BaseTrainer
import torch.nn.functional as F
import numpy as np
from utils import image2label

class OilDataset(Dataset):
    def __init__(self, txt_path, image_processor):
        self.image_processor = image_processor
        self.samples = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                img_path = line.strip()
                lbl_path = image2label(img_path)
                self.samples.append((img_path, lbl_path))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(lbl_path)
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)
        mask = Image.fromarray(mask)

        encoded = self.image_processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt"
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long()
        }

class SegformerTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.image_processor = None
        self.train_args = {
            "epochs": 100,
            "batch": 16,
            "workers": 8,
        }

    def load_model(self, weights="nvidia/segformer-b0-finetuned-ade-512-512"):
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            weights,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(weights, use_fast=True)

    def _train(self, src, dst=None, save=True, **kwargs):
        args = {**self.train_args, **kwargs}

        train_file = os.path.join(src, "train.txt")
        valid_file = os.path.join(src, "val.txt")
        train_dataset = OilDataset(train_file, self.image_processor)
        valid_dataset = OilDataset(valid_file, self.image_processor)
        train_loader = DataLoader(train_dataset, batch_size=args["batch"], shuffle=True, num_workers=args["workers"], persistent_workers=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args["batch"], shuffle=False, num_workers=args["workers"], persistent_workers=True, pin_memory=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        optimizer = AdamW(self.model.parameters(), lr=6e-5)
        self.model.to(device)

        loss_records = []
        best_loss = float("inf")
        for epoch in range(args["epochs"]):
            print(f"Training epoch {epoch}:")
            train_loss = train(self.model, train_loader, optimizer, device)
            valid_loss = valid(self.model, valid_loader, device)
            loss_records.append((train_loss, valid_loss))
            print(f"train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
            if save:
                torch.save(self.model.state_dict(), os.path.join(dst, "weights", "last.pt"))
            if save and valid_loss < best_loss:
                torch.save(self.model.state_dict(), os.path.join(dst, "weights", "best.pt"))

        if save:
            csv_path = os.path.join(dst, "results.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train/seg_loss", "val/seg_loss"])
                for i, (tr, va) in enumerate(loss_records, start=1):
                    writer.writerow([i, tr, va])

    def _predict(self, src, file_name="test.txt"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device).eval()

        with open(os.path.join(src, file_name), "r", encoding="utf-8") as f:
            for line in f.readlines():
                img_path = line.strip()
                lbl_path = image2label(img_path)

                img_name = os.path.basename(img_path).rsplit(".", maxsplit=1)[0]
                image = Image.open(img_path).convert("RGB")
                orig_size = image.size[::-1]

                encoded = self.image_processor(
                    images=image,
                    return_tensors="pt"
                )
                pixel_values = encoded["pixel_values"].to(device)

                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
                    logits = outputs.logits

                    logits = F.interpolate(
                        logits,
                        size=orig_size,
                        mode="bilinear",
                        align_corners=False
                    )

                preds = torch.argmax(logits, dim=1)
                pred_mask = (preds[0] == 1).cpu().numpy().astype(np.uint8)

                gt_mask = np.array(Image.open(lbl_path), dtype=np.uint8)
                gt_mask = (gt_mask > 0).astype(np.uint8)

                self.results.append((img_name, pred_mask, gt_mask))


def train(model, dataloader, optimizer, device):
    model.train()
    scaler = GradScaler(enabled=device == "cuda")
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda", enabled=device=="cuda"):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def valid(model, dataloader, device):
    model.eval()
    total_loss = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda", enabled=device=="cuda"):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )
            loss = outputs.loss
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    src = r"datasets\kfold\fold_4"
    dst = r"runs\kfold\segformer\fold_4"

    trainer = SegformerTrainer()
    trainer.load_model()
    trainer.train(src, dst, save=False, epochs=100)
    trainer.test(src, dst, save=True, file_name="val.txt")

if __name__ == "__main__":
    main()