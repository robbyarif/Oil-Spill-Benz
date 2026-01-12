import os
from typing import Iterable
from sklearn.model_selection import train_test_split as tt_split
from sklearn.model_selection import KFold

class DatasetBuilder:
    def __init__(self,
        src: str,
        dst: str,
        img_ext: Iterable[str],
        lbl_ext: Iterable[str],
        override: bool = False,
        random_seed: int = None
    ) -> None:

        self.src = src
        self.dst = dst
        self.img_ext = img_ext
        self.lbl_ext = lbl_ext
        self.random_seed = random_seed
        self.image_list = []

        if not os.path.exists(src):
            raise Exception(f"path {src} not found.")
        if os.path.exists(dst) and not override:
            raise FileExistsError(f"path {src} already exists. Please change dst or set override as True.")
        os.makedirs(dst, exist_ok=True)

        self._collect_data()

    def create(self, ratio):
        test_sz = ratio[2] / sum(ratio) if len(ratio) == 3 else 0
        train_list = self.image_list
        if test_sz:
            train_list, test_list = tt_split(train_list, test_size=test_sz, random_state=self.random_seed)
            test_path = os.path.join(self.dst, "test.txt")
            self._write_txt(test_path, test_list)

        val_sz = ratio[1] / (ratio[0] + ratio[1])
        train_list, val_list = tt_split(train_list, test_size=val_sz, random_state=self.random_seed)
        train_path = os.path.join(self.dst, "train.txt")
        val_path = os.path.join(self.dst, "val.txt")
        self._write_txt(train_path, train_list)
        self._write_txt(val_path, val_list)

    def kfold(self, k):
        kf = KFold(n_splits=k, random_state=self.random_seed)
        for i, (train_idx, val_idx) in enumerate(kf.split(self.image_list)):
            folder_path = os.path.join(self.dst, f"fold{i+1}")
            os.makedirs(folder_path, exist_ok=True)

            train_path = os.path.join(folder_path, "train.txt")
            train_list = [self.image_list[idx] for idx in train_idx]
            self._write_txt(train_path, train_list)

            val_path = os.path.join(folder_path, "val.txt")
            val_list = [self.image_list[idx] for idx in val_idx]
            self._write_txt(val_path, val_list)

    def _collect_data(self):
        if not os.path.exists(self.src):
            print(f"Error: path {self.src} doesn't exist.")
            exit(1)

        for root, dirs, files in os.walk(self.src):
            for file in files:
                image_path = os.path.join(root, file)
                base_img_path, ext = os.path.splitext(image_path)
                if ext not in self.img_ext:
                    continue
                base_lbl_path = base_img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
                lbl_paths = [base_lbl_path+ext for ext in self.lbl_ext if os.path.exists(base_lbl_path+ext)]
                if len(lbl_paths) == 1:
                    self.image_list.append(image_path)
                else:
                    print(f"{len(lbl_paths)} label found for {image_path}")

    @staticmethod
    def _write_txt(dst, content):
        with open(dst, "w", encoding="utf-8") as f:
            for line in content:
                f.write(line + "\n")


# IMPORTANT:
# The source dataset directory structure must contain both "/images/" and "/labels/".
# During dataset creation, images and labels are matched by their paths.
# Ensure that path/.../images/.../foo_001.jpg corresponds to path/.../labels/.../foo_001.txt.

# NOTE:
#     src : Path to the original dataset directory
#     dst : Path where the processed dataset will be saved
#     image_ext : Allowed image file extensions
#     label_ext : Allowed label (mask) file extensions
#
#     builder.create(ratio=(0.8,0.1,0.1)) => split data to train.txt, val.txt and test.txt in ratio of 0.8, 0.1, 0.1.
#     builder.create(ratio=(0.8,0.2))     => split data to train.txt and val.txt in ratio of 0.8 and 0.2.
#     builder.kfold(k=5)                  => create k fold, each fold contain train.txt and val.txt in ratio of (k-1)/k and 1/k.

def main():
    src = r"DV4"
    dst = r"datasets/new_baseline"
    image_ext = {".jpg"}
    label_ext = {".png"}
    builder = DatasetBuilder(src, dst, image_ext, label_ext, random_seed=42)

    builder.create(ratio=(0.6, 0.2, 0.2))

if __name__ == "__main__":
    main()