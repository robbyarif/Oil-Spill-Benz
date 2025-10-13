import os
import cv2
from collections import defaultdict
import shutil
import numpy as np

from main import add_color_mask, contours2mask


def generate_txt():
    for file in os.listdir('Origin_Data/Drone'):
        file_name = file.split('.')[0]
        with open(os.path.join('Origin_Data/Label', f'{file_name}.txt'), 'w') as f:
            if not os.path.exists(os.path.join('Origin_Data/Mask', f'{file_name}.png')):
                continue
            mask = cv2.imread(os.path.join('Origin_Data/Mask', f'{file_name}.png'), cv2.IMREAD_UNCHANGED)
            binary_mask = (mask[:,:,0] == 255).astype(np.uint8)
            h, w = mask.shape[:2]
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    continue
                contour = contour.reshape(-1,2).astype(np.float32)
                contour[:,0] = contour[:,0] / w
                contour[:,1] = contour[:,1] / h
                line = "0 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in contour]) + "\n"
                f.write(line)

def create_contour(image_name):
    mask_path = os.path.join('Origin_Data/Mask', image_name.split('.')[0]+'.png')
    with open(os.path.join('Base_Data/Mask', image_name.split('.')[0] + '.txt'), 'w') as f:
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            binary_mask = (mask[:, :, 0] == 255).astype(np.uint8)
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

def is_same_type(img_lst):
    pre_img = os.path.join('Origin_Data/Mask', img_lst[0].split('.')[0]+'.png')
    pre_exist = os.path.exists(pre_img)
    for i in range(1, len(img_lst)):
        cur_img = os.path.join('Origin_Data/Mask', img_lst[i].split('.')[0]+'.png')
        cur_exist = os.path.exists(cur_img)
        if pre_exist != cur_exist:
            return False
        pre_exist = cur_exist
    return True

def extract():
    img_dict = {}
    for img_name in os.listdir('Origin_Data/Drone'):
        img_path = os.path.join('Origin_Data/Drone', img_name)
        img_dict[img_name] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    for group in make_group(img_dict):
        img_name = ''
        for img in group:
            mask_path = os.path.join('Origin_Data/Mask', img.split('.')[0] + '.png')
            if not os.path.exists(mask_path): continue
            img_name = img
            break
        if img_name == '': img_name = group[0]
        src = os.path.join('Origin_Data/Drone', img_name)
        dst = os.path.join('Base_Data/Drone', img_name)
        shutil.copy(src, dst)
        create_contour(img_name)

def make_group(image_dict):
    groups = defaultdict(list)
    for filename, arr in image_dict.items():
        key = arr.tobytes()
        groups[key].append(filename)
    return list(groups.values())



def main():
    images = list(os.listdir('Base_Data/Drone'))
    labels = list(os.listdir('Base_Data/Mask'))
    for img, lab in zip(images, labels):
        image = cv2.imread(os.path.join('Base_Data/Drone', img), cv2.IMREAD_UNCHANGED)
        if image.shape[2] != 3:
            print(f'{img} is {image.shape[2]} channels.')
            continue
        with open(os.path.join('Base_Data/Mask', lab)) as f:
            lines = f.readlines()
        contours = []
        for line in lines:
            coords = list(map(float, line.strip().split()[1:]))
            coords = np.array(coords).reshape(-1, 2)
            contours.append(coords)
        mask = contours2mask(image.shape[:2], contours)
        mask = add_color_mask(image, mask, 'r')
        check_img = np.concatenate((image, mask), axis=1)
        cv2.imwrite(f'Check/{lab.split('.')[0]}.jpg', check_img)
    return 0

if __name__ == "__main__":
    main()