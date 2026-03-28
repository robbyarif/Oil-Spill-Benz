import csv
import time
from abc import ABC, abstractmethod
import torch
import gc
import json
from utils import *

class BaseTrainer(ABC):
    def __init__(self):
        self.model = None
        self.results = []

    @abstractmethod
    def load_model(self, weights=None): ...

    def remove_model(self):
        del self.model
        self.results.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def train(self, src, dst=None, save=True, **kwargs):
        train_file = os.path.join(src, "train.txt")
        valid_file = os.path.join(src, "val.txt")

        if not os.path.exists(train_file):
            raise Exception(f"file {train_file} doesn't exist.")
        if not os.path.exists(valid_file):
            raise Exception(f"file {valid_file} doesn't exist.")
        if save and dst is None:
            raise Exception("dst can't be None to save the model weights.")

        if save:
            weight_path = os.path.join(dst, "weights")
            os.makedirs(weight_path, exist_ok=True)

        start_time = time.time()
        self._train(src, dst, save, **kwargs)
        training_time = time.time() - start_time

    def test(self, src, dst=None, * ,file_name="test.txt", log=True, save=True, color_coded=False, alpha=(1.0, 0.4, 0.4)):
        file_path = os.path.join(src, file_name)
        if not os.path.exists(file_path):
            raise Exception(f"file {file_path} doesn't exist.")
        if color_coded or save:
            if dst is None:
                raise Exception("dst can't be None to save the result.")
            else:
                os.makedirs(dst, exist_ok=True)

        start_time = time.time()
        self._predict(src, file_name)
        self._analyze(dst, color_coded=color_coded, log=log, save=save, alpha=alpha)

    def kfold(self, src, dst=None, save=True, **kwargs):
        if not os.path.exists(src):
            raise Exception(f"dir {src} doesn't exist.")
        if save:
            if dst is None:
                raise Exception("dst can't be None to save the result.")
            else:
                os.makedirs(dst, exist_ok=True)

        folders = os.listdir(src)
        k = len(folders)
        metrics_list = []

        for i in range(k):
            folder_path = os.path.join(src, folders[i])
            self.load_model()
            self.train(folder_path, save=False, **kwargs)
            self._predict(folder_path, "val.txt")
            metrics = self._analyze(color_coded=False, log=False, save=False)
            metrics_list.append(metrics)
            self.remove_model()

        stats = {}
        for metric in metrics_list[0].keys():
            values = [fold_metrics[metric] for fold_metrics in metrics_list]
            mean = np.mean(values)
            std = np.std(values)
            stats[metric] = (mean, std)

        for i, metrics in enumerate(metrics_list):
            self._output_metrics(f"folder{i+1}", metrics)
        self._output_metrics("mean ± std over all folds", stats, is_std=True)

        if save:
            fieldnames = [k for k in stats.keys()]
            with open(os.path.join(dst, "records.csv"), "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["fold"] + fieldnames)
                for i in range(k):
                    metrics = metrics_list[i]
                    data = [f"fold{i+1}"] + [f"{metrics[field]:.3f}" for field in fieldnames]
                    writer.writerow(data)
                data = ["mean ± std"] + [f"{stats[field][0]:.3f} ± {stats[field][1]:.3f}" for field in fieldnames]
                writer.writerow(data)


        return stats

    @abstractmethod
    def _train(self, src, dst=None, save=True, **kwargs): ...

    @abstractmethod
    def _predict(self, src, file_name="test.txt"): ...

    def _analyze(self, dst=None, * , log=True, save=True, color_coded=False, alpha=(1.0, 0.4, 0.4)):
        acc = 0
        f1, f1_num = 0, 0
        oil_iou, bg_iou = 0, 0
        oil_num, bg_num = 0, 0

        for image_path, pred_mask, gt_mask in self.results:
            acc += get_acc(pred_mask, gt_mask)

            f1_buf = get_f1(pred_mask, gt_mask)
            if f1_buf is not None:
                f1 += f1_buf
                f1_num += 1

            oil_buf, bg_buf = get_iou(pred_mask, gt_mask)
            if oil_buf is not None:
                oil_iou += oil_buf
                oil_num += 1
            if bg_buf is not None:
                bg_iou += bg_buf
                bg_num += 1

            # visualize coded mask
            if color_coded:
                os.makedirs(os.path.join(dst, "color_coded_images"), exist_ok=True)

                file_name = os.path.basename(image_path)
                origin_img = read_img(image_path)

                coded_mask = get_coded_image(origin_img, pred_mask, gt_mask, alpha)
                cv2.imwrite(os.path.join(dst, "color_coded_images", f"IoU={oil_buf:.3f}_{file_name}.jpg"), coded_mask)

        acc = acc / len(self.results)
        f1 = f1 / f1_num if f1_num else None
        oil_iou =  oil_iou / oil_num if oil_num else None
        bg_iou = bg_iou / bg_num if bg_num else None
        m_iou = (oil_iou + bg_iou) / 2

        metrics = {
            "Acc": acc,
            "F1": f1,
            "Oil IoU": oil_iou,
            "BG IoU": bg_iou,
            "mIoU": m_iou
        }
        
        if inference_time is not None:
            metrics["Inference Time (s)"] = inference_time
            metrics["Avg Time per Image (s)"] = inference_time / len(self.results) if self.results else 0

        if log:
           self._output_metrics("metrics", metrics)

        if save:
            with open(os.path.join(dst, "metrics.json"), "w") as f:
                json_data = json.dumps(metrics, indent=4)
                f.write(json_data)

        return metrics

    @staticmethod
    def _output_metrics(title, metrics, is_std=False):
        print(f"\n----- {title} -----")
        max_len = max(map(len, metrics.keys()))
        for metric, value in metrics.items():
            if is_std:
                print(f"{metric:>{max_len}} : {(value[0] or 0):.3f} ± {(value[1] or 0):.3f}")
            else:
                print(f"{metric:>{max_len}} : {(value or 0):.3f}")

def main():
    ...

if __name__ == "__main__":
    main()