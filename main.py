import cv2
from ultralytics import YOLO
import os
from random import choice, sample
import shutil
import yaml
import numpy as np

def check_dataset(dataset_name):
    dataset_path = os.path.join('datasets', dataset_name)
    dirs = []
    for img_type in ('images', 'labels'):
        for train_type in ('train', 'val'):
            new_dir_path = os.path.join(dataset_path, img_type, train_type)
            dirs.append([name.split('.')[0] for name in os.listdir(new_dir_path)])
    if dirs[0] == dirs[2] and dirs[1] == dirs[3]:
        return True
    if dirs[0] != dirs[2]:
        print('train data not match')
    if dirs[1] != dirs[3]:
        print('test data not match')
    return False

def create_dataset(new_dir_name, base_data_path, ratio):
    dataset_path = os.path.join('datasets', new_dir_name)
    dirs_path = []
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    for img_type in ('images', 'labels'):
        for train_type in ('train', 'val'):
            new_dir_path = os.path.join(dataset_path, img_type, train_type)
            os.makedirs(new_dir_path)
            dirs_path.append(new_dir_path)
    image_path = os.path.join(base_data_path, 'Drone')
    label_path = os.path.join(base_data_path, 'Mask')
    images = [os.path.join(image_path, name) for name in os.listdir(image_path)]
    labels = [os.path.join(label_path, name) for name in os.listdir(label_path)]
    test_num = int(len(images) * ratio)
    train_num = len(images) - test_num
    image_dst = sample([0]*train_num+[1]*test_num, len(images))
    label_dst = [i+2 for i in image_dst]
    src_list = images + labels
    dst_list = image_dst + label_dst
    for src, dst in zip(src_list, dst_list):
        shutil.copy(src, dirs_path[dst])
    yaml_content = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['oil spill'],
        'task': 'segment'
    }
    with open(os.path.join('datasets', new_dir_name, 'data.yaml'), 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False)

def dataset_info(dataset_name):
    for dir_type in ['train', 'val']:
        dir_path = os.path.join('datasets', dataset_name, 'labels', dir_type)
        mask, bgd = 0, 0
        for label in os.listdir(dir_path):
            file_path = os.path.join(dir_path, label)
            if os.path.getsize(file_path) == 0:
                bgd += 1
            else:
                mask += 1
        print(f'{dir_type}: {mask} masks, {bgd} backgrounds', end=' ')
    print()

def augmentation(dataset_name):
    image_dir = os.path.join('datasets', dataset_name, 'images', 'train')
    label_dir = os.path.join('datasets', dataset_name, 'labels', 'train')
    mask_list = []
    bgd_num = 0
    for image in os.listdir(image_dir):
        label_path = os.path.join(label_dir, image.split('.')[0]+'.txt')
        if os.path.getsize(label_path) == 0:
            bgd_num += 1
            continue
        mask_list.append(image)
    for _ in range(bgd_num - len(mask_list)):
        image_name = choice(mask_list)
        image_path = os.path.join(image_dir, choice(mask_list))
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        new_image = np.fliplr(image)
        cv2.imwrite(os.path.join(image_dir, f'augment_'+image_name), new_image)
        label_name = image_name.split('.')[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            var = [float(i) for i in line.strip().split()[1:]]
            for i in range(0, len(var), 2): var[i] = 1 - var[i]
            new_lines.append('0 ' + ' '.join(f'{v:.6f}' for v in var) + '\n')
        with open(os.path.join(label_dir, f'augment_'+label_name), 'w') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    dataset = 'augmentation'
    dataset_info(dataset)
    # create_dataset(dataset, 'Base_Data', 0.2)
    # augmentation(dataset)
    if check_dataset(dataset):
        model = YOLO("yolo11n-seg.pt")
        result = model.train(data=f"datasets/{dataset}/data.yaml", epochs=1, imgsz=640, batch=32, device=0, workers=8)
    # model = YOLO('runs/segment/train2/weights/last.pt')
    # model.val(data='datasets/without augmentation/data.yaml')
