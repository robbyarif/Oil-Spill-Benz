import os
import cv2
import numpy as np
from utils import contour2mask, overlay_mask

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
        mask = contour2mask(image.shape[:2], contours)
        mask = overlay_mask(image, mask, 'r')
        check_img = np.concatenate((image, mask), axis=1)
        cv2.imwrite(f'Check/{lab.split('.')[0]}.jpg', check_img)
    return 0

if __name__ == "__main__":
    main()