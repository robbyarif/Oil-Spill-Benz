from pathlib import Path
import shutil
import os
import numpy as np
from ultralytics.data.utils import polygons2masks, IMG_FORMATS, img2label_paths
import cv2
import random
from ultralytics.utils import LOGGER, TQDM

def split(source_path, weights, annotated_only=False, random_state=None):
    """
    the same function of autosplit in ultralytics.data.split
    the only different is this contain a random_state parameter to control random seed.
    """
    path = Path(source_path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)
    n = len(files)
    if random_state is not None:
        random.seed(random_state)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()
    LOGGER.info(f"Splitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")

def create_dataset(ratio=(0.6, 0.2, 0.2),*, save=False, name=None, random_state=None):
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files
    create data.yaml for yolo model to get train, val and test data in source_data folder
    args:
        ratio (tuple): the weight of train, val and test dataset
        save (bool): save to a new folder
        name (str|None): user defined folder name
        random_state (int|None): use for setting random seed
    """
    split(Path('datasets/source_data'), ratio, annotated_only=True, random_state=random_state)
    if not save: return
    temp = 0
    if name is None:
        while os.path.exists(f'datasets/dataset_{temp}'):
            temp += 1
        name = f'dataset_{temp}'
    path = os.path.join('datasets', name)
    os.mkdir(path)
    shutil.copy('datasets/autosplit_train.txt', os.path.join(path, 'autosplit_train.txt'))
    shutil.copy('datasets/autosplit_val.txt', os.path.join(path, 'autosplit_val.txt'))
    shutil.copy('datasets/autosplit_test.txt', os.path.join(path, 'autosplit_test.txt'))

def set_dataset(name):
    """
    set the dataset to train and test
    args:
        name (str): the name of dataset
    """
    folder_path = os.path.join('datasets', name)
    if not os.path.exists(folder_path):
        print(f'{folder_path} do not exist!')
        return
    for cur_type in ['train', 'val', 'test']:
        src = os.path.join(folder_path, f'autosplit_{cur_type}.txt')
        dst = os.path.join('datasets', f'autosplit_{cur_type}.txt')
        shutil.copy(src, dst)

def dataset_info(name=None):
    """
    output the statistic of datasets includes the number of oil spill and not spill in total, train, val and test dataset
    args:
        name (str|None): the name of dataset if name is None choose the last created dataset
    """
    path = 'datasets'
    if name is not None:
        path = os.path.join(path, name)
    if not os.path.exists(path):
        print(f'{path} do not exist!')
        return
    stats = {'total': [0, 0]}
    for cur_type in ['train', 'val', 'test']:
        stats[cur_type] = [0, 0]
        cur_path = os.path.join(path, f'autosplit_{cur_type}.txt')
        with open(cur_path, 'r') as f:
            file_paths = ['datasets'+line.split('.')[1]+'.txt' for line in f.readlines()]
        for file in file_paths:
            spill = int(os.path.getsize(file) != 0)
            stats[cur_type][spill] += 1
            stats['total'][spill] += 1
    for cur_type in ['total', 'train', 'val', 'test']:
        print(f'{stats[cur_type][0] + stats[cur_type][1]:3} {cur_type:>5} images: {stats[cur_type][1]:3} oil spill, {stats[cur_type][0]:3} no oil spill')

def mask2yolo_label(src, dst):
    """
    convert a mask to the yolo detection annotation
    args:
        src (str|Path): the path of the mask
        dst (str|Path): the path of the new yolo label txt file
    """
    if os.path.exists(src):
        print(f'{src} do not exist!')
        return
    with open(dst, 'w') as f:
        mask = cv2.imread(src, cv2.IMREAD_COLOR_BGR)
        binary_mask = np.array((mask[:, :, 0] == 255), dtype=np.uint8)
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3:
                continue
            contour = contour.reshape(-1, 2).astype(np.float32)
            contour[:, 0] = contour[:, 0] / w
            contour[:, 1] = contour[:, 1] / h
            line = "0 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in contour]) + "\n"
            f.write(line)

def contour2mask(contours, sz):
    """
    convert coordinates of contour points to binary mask
    args:
        contours (list): a list of ndarray, each ndarray is the mask contour in format [[x1, y1],..., [xn,yn]]
        sz (tuple): the height and weight of the image
    return:
        final_mask (ndarray): a 2D binary mask
    """
    h, w = sz
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.astype(np.float32)
        contour[:,0] *= w
        contour[:,1] *= h
        polygons.append(contour)
    binary_masks = polygons2masks((h,w), polygons, 1)
    final_mask = np.zeros((h,w), dtype=np.uint8)
    for binary_mask in binary_masks:
        final_mask = cv2.bitwise_or(final_mask, binary_mask)
    return final_mask

def overlay_mask(img, binary_mask, color='b'):
    """
    set the color of binary mask and overlay to the image
    args:
        img (ndarray): the original image
        binary_mask (ndarray): the corresponding binary mask
        color (str): the user defined color of binary mask, 'b', 'g', 'r' for blue, green and red
    return (ndarray): the image that combine the original image and the binary mask
    """
    bgr = [0, 0, 0]
    if color == 'b':
        bgr = np.array([255, 0, 0], dtype=np.uint8)
    elif color == 'g':
        bgr = np.array([0, 255, 0], dtype=np.uint8)
    elif color == 'r':
        bgr = np.array([0, 0, 255], dtype=np.uint8)
    binary_mask = np.stack([binary_mask * bgr[c] for c in range(3)], axis=-1)
    return cv2.addWeighted(img, 1, binary_mask, 0.5, 0)

def count_iou(pred_mask, gt_mask):
    """
    count the iou of the predicted mask and the ground truth mask
    args:
        pred_mask (ndarray): the model predicted binary mask
        gt_mask (ndarray): the ground truth mask
    return (float): the iou value
    """
    mask_inter = np.logical_and(pred_mask, gt_mask).sum()
    mask_union = np.logical_or(pred_mask, gt_mask).sum()
    return mask_inter / mask_union if mask_union > 0 else 0
