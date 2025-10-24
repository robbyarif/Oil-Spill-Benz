import os
import cv2
from ultralytics import YOLO
from utils import *

def main():
    conf = 0.25
    dataset_name = 'augment_down_sample'
    with open(f'datasets/{dataset_name}/autosplit_test.txt', 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    best_model = YOLO(f'runs/segment/{dataset_name}/weights/best.pt')
    total_iou = 0
    num = 0
    for img_path in image_paths:
        label_path = img_path.split('.')[0] + '.txt'
        result = best_model(img_path, conf=conf, verbose=False)[0]
        sz = result.orig_shape
        # get predict mask
        pred_mask = np.zeros(sz, dtype=np.uint8)
        if hasattr(result, 'masks') and result.masks is not None:
            contours_pred = result.masks.xyn
            pred_mask = contours2mask(contours_pred, sz)
        # get ground truth mask
        contours = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            coords = list(map(float, line.strip().split()[1:]))
            contour = np.array(coords, dtype=np.float32).reshape(-1, 2)
            contours.append(contour)
        gt_mask = contours2mask(contours, sz)
        # count iou
        mask_inter = np.logical_and(pred_mask, gt_mask).sum()
        mask_union = np.logical_or(pred_mask, gt_mask).sum()
        if mask_union == 0:
            continue
        total_iou += mask_inter / mask_union
        num += 1
    metrics = best_model.val(data='datasets/data.yaml', split='test', device='0', conf=conf, verbose=False).summary()[0]
    iou = total_iou / num
    metrics['iou'] = iou
    metrics['dc'] = 2 * iou / (1 + iou)
    metrics_list = ['Box-P', 'Box-R', 'Box-F1', 'Mask-P', 'Mask-R', 'Mask-F1', 'iou', 'dc']
    print('-'*40 + 'Result' + '-'*40)
    print(f'Dataset: {dataset_name} ')
    print(' '.join(f'{m:>10s}' for m in metrics_list))
    print(' '.join(f'{metrics[m]:>10.3f}' for m in metrics_list))
    return 0

if __name__ == "__main__":
    for i in range(2, 5):
        print(f'augment_{i}x:')
        dataset_info(f'augment_{i}x')