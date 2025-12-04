import os
import shutil
import yaml
from sklearn.model_selection import train_test_split as tt_split

def generate_dataset(src, dst, ratio=(0.6, 0.2, 0.2), random_seed=42):
    if not os.path.exists(src):
        print(f"WARNING!: path {src} doesn't exist.")
        return

    if os.path.exists(dst) and len(os.listdir(dst)) != 0:
        print(f"WARNING: destination folder {dst} is NOT empty.")
        print(f"Do you want to overwrite the {dst} folder? input 0 for cancel, 1 for continue")
        while True:
            choice = input()
            if choice == "0":
                return
            elif choice == "1":
                shutil.rmtree(dst)
            else:
                print("Please input 0 or 1")
                continue

    if abs(sum(ratio) - 1.0) > 1e-6:
        print("WARNING!: the sum of ratio should be 1.")
        return

    image_paths = []
    label_paths = []
    image_ext = {".jpg", ".jpeg", ".png"}
    data_pair = {}

    # collect images and labels
    for root, dirs, files in os.walk(src):
        for file in files:
            full_path = os.path.join(root, file)
            base_name, ext = os.path.splitext(file)
            ext = ext.lower()
            if base_name not in data_pair:
                data_pair[base_name] = {"image": "", "label": ""}
            if ext == ".txt":
                data_pair[base_name]["label"] = str(full_path)
            elif ext in image_ext:
                data_pair[base_name]["image"] = str(full_path)
            else:
                print(f"WARNING!: unaccepted file {full_path}")

    for base_name, data in data_pair.items():
        if data["image"] and data["label"]:
            image_paths.append(data["image"])
            label_paths.append(data["label"])

    # split and construct dataset
    train_images, test_images, train_labels, test_labels = tt_split(image_paths, label_paths, test_size=ratio[2], shuffle=True, random_state=random_seed)
    train_images, val_images, train_labels, val_labels = tt_split(train_images, train_labels, test_size=ratio[1]/(ratio[0] + ratio[1]), shuffle=True, random_state=random_seed)

    subsets = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]

    files_lists = [
        train_images, val_images, test_images,
        train_labels, val_labels, test_labels
    ]

    os.makedirs(dst, exist_ok=True)
    for folder, files in zip(subsets, files_lists):
        folder_path = os.path.join(dst, folder)
        os.makedirs(folder_path, exist_ok=True)
        for f in files:
            shutil.copy2(f, os.path.join(folder_path, os.path.basename(f)))

    # create dataset yaml file
    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "oil"},
        "task": "segment"
    }
    yaml_path = os.path.join(dst, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

def main():
    src = r"DV4"
    dst = r"datasets/baseline_seed_42"
    generate_dataset(src, dst, (0.6, 0.2, 0.2))

if __name__ == "__main__":
    main()