import os.path
import shutil
import yaml
from ultralytics import YOLO
import numpy as np
from trainer import BaseTrainer
from utils import contours2mask, yolo_label2contours, image2label

class YoloTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

        self.train_args = {
            "data" : "data.yaml",
            "epochs" : 300,
            "imgsz" : 512,
            "batch" : 16,
            "optimizer" : "SGD",
            "lr0" : 0.01,
            "box" : 6,
            "device" : 0,
            "workers" : 8,
            "plots" : False,
        }

    def load_model(self, weights=None, version=11):
        if weights is not None:
            self.model = YOLO(weights)
        elif version == 11:
            self.model = YOLO("yolo11n-seg.pt")
        elif version == 12:
            self.model = YOLO("yolo12n-seg.yaml")
        else:
            raise Exception(f"Not support YOLO version {version}.")

    def _train(self, src, dst=None, save=True, **kwargs):
        args = {**self.train_args, **kwargs}

        train_file = src + "/" + "train.txt"
        valid_file = src + "/" + "val.txt"

        yaml_path = "data.yaml"
        with open(yaml_path, "w") as f:
            data_yaml = {
                "train": train_file,
                "val": valid_file,
                "nc": 1,
                "names": {0: "oil"},
            }
            yaml.dump(data_yaml, f, sort_keys=False)

        args["data"] = yaml_path
        if save:
            args["project"] = os.path.dirname(dst)
            args["name"] = os.path.basename(dst)

        self.model.train(**args)

        run_dir = self.model.trainer.save_dir
        if not save and run_dir.exists():
            shutil.rmtree(run_dir)

        os.remove(yaml_path)

    def _predict(self, src, file_name="test.txt"):
        with open(os.path.join(src, file_name), "r", encoding="utf-8") as f:
            image_list = [line.strip() for line in f]

        results = []
        for img_path in image_list:
            result = self.model.predict(img_path)
            results.extend(result)

        for result in results:
            img_h, img_w = result.orig_shape
            img_size = (img_h, img_w)
            img_name = os.path.basename(result.path).rsplit('.', maxsplit=1)[0]

            pred_contours = []
            if hasattr(result, "masks") and result.masks is not None:
                pred_contours = result.masks.xy

            pred_mask = contours2mask(pred_contours, img_size)
            pred_mask = (pred_mask > 0).astype(np.uint8)

            label_path = image2label(result.path, lbl_ext=".txt")
            gt_contours = yolo_label2contours(label_path, img_size)
            gt_mask = contours2mask(gt_contours, img_size)
            gt_mask = (gt_mask > 0).astype(np.uint8)

            self.results.append((img_name, pred_mask, gt_mask))

def main():
    ...

if __name__ == "__main__":
    main()