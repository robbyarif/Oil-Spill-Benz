import numpy as np
from ultralytics.data.utils import polygons2masks
import cv2
import os

def mask2yolo_label(mask, dst):
    """
    convert a mask to the yolo detection annotation
    args:
        mask (ndarray): a 2D binary mask
        dst (str|Path): the path of the new yolo label txt file
    """
    with open(dst, 'w') as f:
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3:
                continue
            contour = contour.reshape(-1, 2).astype(np.float64)
            contour[:, 0] = contour[:, 0] / w
            contour[:, 1] = contour[:, 1] / h
            line = "0 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in contour]) + "\n"
            f.write(line)

def yolo_label2contours(label_path, sz):
    """
    args:
        label_path (str): the path of yolo label file(txt file).
    return:
        contours (list): a list of ndarray, each ndarray is the mask contour in format [[x1, y1],..., [xn,yn]]
    """
    contours = []
    h, w = sz
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        contour = np.array([float(i) for i in line.strip().split()[1:]], dtype=np.float64)
        contour = contour.reshape(-1,2)
        contour[:,0] *= w
        contour[:,1] *= h
        contours.append(contour)
    return contours

def contours2mask(contours, sz):
    """
    convert coordinates of contour points to binary mask
    args:
        contours (list): a list of ndarray, each ndarray is the mask contour in format [[x1, y1],..., [xn,yn]]
        sz (tuple): the height and weight of the image
    return:
        final_mask (ndarray): a 2D binary mask
    """
    h, w = sz
    if len(contours) == 0:
        return np.zeros(sz, dtype=np.uint8)
    binary_masks = polygons2masks((h,w), contours, 1)
    final_mask = np.max(binary_masks, axis=0)
    return final_mask

def get_acc(pred_mask, gt_mask):
    """
    calculate the accuracy of the model predicted mask and the ground truth mask
    args:
        pred_mask (ndarray): the binary mask predicted by model
        gt_mask (ndarray): the binary ground truth mask
    return:
        acc (float): the accuracy
    """
    true_pixel = (pred_mask == gt_mask).sum()
    all_pixel = np.prod(pred_mask.shape)
    acc = true_pixel / all_pixel
    return acc

def get_iou(pred_mask, gt_mask):
    """
    calculate the oil IoU and background IoU of the model predicted mask and the ground truth mask
    args:
        pred_mask (ndarray): the binary mask predicted by model
        gt_mask (ndarray): the binary ground truth mask
    return:
        oil_iou (float): the IoU of oil
        bg_iou (float): the IoU of background
    """
    oil_union = np.logical_or(pred_mask, gt_mask).sum()
    oil_inter = np.logical_and(pred_mask, gt_mask).sum()
    oil_iou = oil_inter / oil_union if oil_union else None

    bg_union = np.logical_or(1 - pred_mask, 1 - gt_mask).sum()
    bg_inter = np.logical_and(1 - pred_mask, 1 - gt_mask).sum()
    bg_iou = bg_inter / bg_union if bg_union else None

    return oil_iou, bg_iou

def get_f1(pred_mask, gt_mask):
    """
    calculate the f1 score of the model predicted mask and the ground truth mask
    args:
        pred_mask (ndarray): the binary mask predicted by model
        gt_mask (ndarray): the binary ground truth mask
    return:
        f1 (float): the f1 score
    """
    TP = ((pred_mask == 1) & (gt_mask == 1)).sum()
    FP = ((pred_mask == 1) & (gt_mask == 0)).sum()
    FN = ((pred_mask == 0) & (gt_mask == 1)).sum()
    if (TP + FP + FN) == 0:
        return None
    else:
        return 2 * TP / (2 * TP + FP + FN)

def get_coded_image(origin_img, pred_mask, gt_mask, alpha=(1.0, 0.4, 0.4)):
    # alpha: (origin, pred, gt)
    coded_image = origin_img.astype(np.float32) * alpha[0]

    pred = pred_mask > 0
    gt = gt_mask > 0

    coded_image[pred] = coded_image[pred] * (1 - alpha[1]) + np.array([0, 0, 255], dtype=np.float32) * alpha[1]
    coded_image[gt] = coded_image[gt] * (1 - alpha[2]) + np.array([0, 255, 255], dtype=np.float32) * alpha[2]

    if alpha[1] and alpha[2]:
        tp_alpha = (alpha[1] + alpha[2]) / 2
        tp = (pred_mask == 1) & (gt_mask == 1)
        coded_image[tp] = origin_img[tp] * (1 - tp_alpha) + np.array([0, 255, 0], dtype=np.float32) * tp_alpha

    return coded_image.clip(0, 255).astype(np.uint8)


def read_img(path, flags=cv2.IMREAD_COLOR):
    """
    used for replacing cv2.imread since the function will fail when path contains chinese.
    """
    path = path.replace("\\", "/")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img

def image2label(img_path, lbl_ext=".png"):
    img_path = img_path.replace("\\", "/")
    label_path = img_path.replace("/images/", "/labels/")
    label_path = label_path.rsplit(".", 1)[0] + lbl_ext
    return label_path