import numpy as np
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch
import torchvision.transforms.functional as TF
from torch.amp import autocast, GradScaler
from utils import read_img, image2label
import csv
from main import BaseTrainer

class OilDataset(Dataset):
    def __init__(self, data_file, image_size=512):
        self.image_size = image_size
        self.data_pairs = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                img_path = line.strip()
                lbl_path = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace(".jpg", ".png")

                # ---------- image ----------
                img = read_img(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

                img = TF.normalize(
                    img,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

                # ---------- mask ----------
                mask = read_img(lbl_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Failed to read mask: {lbl_path}")
                mask = torch.from_numpy(mask).float().unsqueeze(0)
                mask = (mask > 0).float()

                # ---------- resize ----------
                img = TF.resize(
                    img,
                    size=[self.image_size, self.image_size],
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                mask = TF.resize(
                    mask,
                    size=[self.image_size, self.image_size],
                    interpolation=TF.InterpolationMode.NEAREST
                )

                self.data_pairs.append((img, mask))


    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        return self.data_pairs[idx]

    def _padding(self, image, mask):
        _, h, w = image.shape
        pad_h = max(0, self.image_size - h)
        pad_w = max(0, self.image_size - w)

        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h])
            mask = TF.pad(mask, [0, 0, pad_w, pad_h])
        return image, mask

def train(model, loss_fn, dataloader, optimizer, device, scalar):
    model.train()
    total_loss = 0.0

    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(imgs)['out']
            loss = loss_fn(outputs, masks)

        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def valid(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0

    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)['out']
            loss = loss_fn(logits, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)

class DeeplabTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

        self.train_args = {
            "epochs": 200,
            "image_size": 512,
            "batch": 16,
            "lr": 1e-3,
            "workers": 6,
        }

    def load_model(self, weights=None):
        self.model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

        for p in self.model.backbone.parameters():
            p.requires_grad = False

        for p in self.model.backbone.layer4.parameters():
            p.requires_grad = True

        if weights is not None:
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))

    def _train(self, src, dst=None, save=True, **kwargs):
        train_file = os.path.join(src, "train.txt")
        valid_file = os.path.join(src, "val.txt")

        args = {**self.train_args, **kwargs}

        train_dataset = OilDataset(train_file, image_size=args["image_size"])
        valid_dataset = OilDataset(valid_file, image_size=args["image_size"])

        train_loader = DataLoader(train_dataset, batch_size=args["batch"], num_workers=args["workers"], shuffle=True, persistent_workers=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args["batch"], num_workers=args["workers"], shuffle=False, persistent_workers=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
        self.model.to(device)

        optimizer = torch.optim.SGD(
            [
                {"params": self.model.backbone.layer4.parameters(), "lr": 1e-4},
                {"params": self.model.classifier.parameters(), "lr": args["lr"]},
            ],
            momentum=0.9,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=args["epochs"],
            power=0.9
        )

        loss_fn = nn.BCEWithLogitsLoss()

        loss_record = []
        best_loss = float("inf")

        if save:
            weight_path = os.path.join(dst, "weights")
            os.makedirs(weight_path, exist_ok=True)

        for epoch in range(args["epochs"]):
            print(f"Training Epoch {epoch + 1}:")
            train_loss = train(self.model, loss_fn, train_loader, optimizer, device, scaler)
            val_loss = valid(self.model, loss_fn, valid_loader, device)
            loss_record.append((train_loss, val_loss))
            scheduler.step()

            if save and val_loss < best_loss:
                torch.save(self.model.state_dict(), os.path.join(dst, "weights", "best.pt"))
            best_loss = val_loss

            print(f"Train loss: {train_loss:.3f}, Valid loss: {val_loss:.3f}")

        if save:
            torch.save(self.model.state_dict(), os.path.join(dst, "weights", "last.pt"))

            csv_path = os.path.join(dst, "results.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train/seg_loss", "val/seg_loss"])
                for i, (tr, va) in enumerate(loss_record, start=1):
                    writer.writerow([i, tr, va])

    def _predict(self, src, file_name="test.txt"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device).eval()

        image_list = []
        label_list = []

        with open(os.path.join(src, file_name), "r", encoding="utf-8") as f:
            for line in f.readlines():
                image_path = line.strip()
                label_path = image2label(image_path)
                image_list.append(image_path)
                label_list.append(label_path)

        for img_path, lbl_path in zip(image_list, label_list):
            img = read_img(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_name = os.path.basename(img_path).rsplit(".", maxsplit=1)[0]

            img = TF.normalize(
                img,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ).unsqueeze(0).to(device)

            img = TF.resize(
                img,
                size=[512, 512],
                interpolation=TF.InterpolationMode.BILINEAR
            )

            # predicted mask
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = self.model(img)['out']

            logits = TF.resize(
                logits,
                size=[H, W],
                interpolation=TF.InterpolationMode.BILINEAR
            )

            pred_mask = (logits > 0).squeeze().cpu().numpy().astype(np.uint8)

            # ground truth mask
            gt_mask = read_img(lbl_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.squeeze(-1)
            gt_mask = (gt_mask > 0).astype(np.uint8)

            self.results.append((img_name, pred_mask, gt_mask))

def main():
    src = r"datasets\new_baseline"
    dst = r"foo\deeplab"

    trainer = DeeplabTrainer()
    trainer.load_model()
    trainer.train(src, dst, epochs=3)
    trainer.test(src, dst)

if __name__ == "__main__":
    main()
