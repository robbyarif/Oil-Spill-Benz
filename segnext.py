from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.apis import init_model
import cv2
from sympy import true
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
from utils import image2label, read_img
from trainer import BaseTrainer
import os
import csv
import numpy as np


class OilDataset(Dataset):
    def __init__(self, data_file, image_size=512):
        self.image_size = image_size
        self.data_pairs = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                img_path = line.strip()
                lbl_path = image2label(img_path)
                # ---------- image ----------
                img = read_img(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).permute(2, 0, 1).float()

                img = TF.normalize(
                    img,
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375]
                )

                # ---------- mask ----------
                mask = read_img(lbl_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Failed to read mask: {lbl_path}")
                mask = torch.from_numpy(mask).float().unsqueeze(0)
                mask = (mask > 0).long()

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
                ).squeeze(0)

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

def get_param_groups(model):
    decay = []
    no_decay = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name:
            head_params.append(param)
        elif 'pos_block' in name or 'norm' in name or 'bn' in name or 'gn' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': 0.01, 'lr': 6e-5},
        {'params': no_decay, 'weight_decay': 0.0, 'lr': 6e-5},
        {'params': head_params, 'weight_decay': 0.01, 'lr': 6e-5},
    ]

def poly_lr_scheduler(optimizer, init_lr, it, max_iter, power=1.0):
    lr = init_lr * (1 - it / max_iter) ** power
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * (10 if i == 2 else 1)
    return lr

class SegnextTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.train_args = {
            "epochs": 100,
            "batch": 8,
            "lr": 6e-5,
            "image_size": 512,
            "workers": 2,
        }

    def load_model(self, weights=None):
        self.model = init_model(
            "mmsegmentation/configs/segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py",
            "segnext_mscan-t_1x16_512x512_adamw_160k_ade20k_20230210_140244-05bd8466.pth"
        )
        if weights is not None:
            self.model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

    def _train(self, src, dst=None, save=True, **kwargs):
        args = {**self.train_args, **kwargs}

        train_file = os.path.join(src, "train.txt")
        valid_file = os.path.join(src, "val.txt")

        train_dataset = OilDataset(train_file, args["image_size"])
        valid_dataset = OilDataset(valid_file, args["image_size"])

        train_loader = DataLoader(train_dataset, batch_size=args["batch"], shuffle=True, num_workers=args["workers"], persistent_workers=True, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args["batch"], shuffle=False, num_workers=args["workers"], persistent_workers=True, pin_memory=True, drop_last=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        optimizer = torch.optim.AdamW(get_param_groups(self.model), lr=args["lr"], betas=(0.9, 0.999))
        self.model.to(device)

        self.global_iter = 0
        total_iters = len(train_loader) * args["epochs"]
        loss_records = []
        best_loss = float("inf")

        for epoch in range(args["epochs"]):
            print(f"Training epoch {epoch}:")
            train_loss = train(self.model, train_loader, optimizer, device, self.global_iter, total_iters)
            valid_loss = valid(self.model, valid_loader, device)
            loss_records.append((train_loss, valid_loss))
            print(f"train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}")

            if save:
                torch.save(self.model.state_dict(), os.path.join(dst, "weights", "last.pt"))
            if save and valid_loss < best_loss:
                torch.save(self.model.state_dict(), os.path.join(dst, "weights", "best.pt"))
            if valid_loss < best_loss:
                best_loss = valid_loss

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

                img = read_img(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H, W = img.shape[:2]
                img = torch.from_numpy(img).permute(2, 0, 1).float()

                img = TF.normalize(
                    img,
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375]
                )

                img = TF.resize(
                    img,
                    size=[self.train_args["image_size"], self.train_args["image_size"]],
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                img = img.unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = self.model(img)

                logits = F.interpolate(
                    logits,
                    size=[H, W],
                    mode='bilinear',
                    align_corners=False
                )

                pred = torch.argmax(logits, dim=1)
                pred_mask = pred.squeeze(0).cpu().numpy().astype(np.uint8)

                gt_mask = read_img(lbl_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask.squeeze(-1)
                gt_mask = (gt_mask > 0).astype(np.uint8)

                self.results.append((img_path, pred_mask, gt_mask))

def train(model, dataloader, optimizer, device, global_iter, total_iters):
    model.train()
    total_loss = 0

    for img, lbl in dataloader:
        img, mask = img.to(device), lbl.to(device)

        logits = model(img)
        logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logits, mask, ignore_index=255, reduction='mean')
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        poly_lr_scheduler(optimizer, init_lr=6e-5, it=global_iter, max_iter=total_iters, power=1.0)
        global_iter += 1

    return total_loss / len(dataloader)

@torch.no_grad()
def valid(model, dataloader, device):
    model.eval()
    total_loss = 0

    for img, lbl in dataloader:
        img, mask = img.to(device), lbl.to(device)
        logits = model(img)
        logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logits, mask, ignore_index=255, reduction='mean')
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    ...

if __name__ == '__main__':
    main()