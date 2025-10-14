from ultralytics import YOLO
from utils import*

def main():
    model = YOLO('yolo11n-seg.pt')
    model.train(
        data="datasets/data.yaml",
        epochs=200,
        imgsz=1024,
        batch=8,
        device=0,
        optimizer="AdamW",
        lr0=0.0005,
        weight_decay=0.001,
        box=1.5,
        cls=1.0,
        dfl=0.5,
    )
    return 0

if __name__ == "__main__":
    with open('datasets/autosplit_test.txt', 'r') as f:
        image_paths = [line.strip().replace('./', 'datasets/') for line in f.readlines()]
    best_model = YOLO('runs/segment/train14/weights/best.pt')
    iou = 0
    num = 0
    for img_path in image_paths:
        label_path = img_path.split('.')[0]+'.txt'
        result = best_model(img_path)[0]
        if not hasattr(result, 'masks') or result.masks is None:
            continue
        img = result.orig_img
        sz = result.orig_shape
        # get predict mask
        contours = result.masks.xyn
        pred_mask = contour2mask(contours, sz)
        # get ground truth mask
        contours = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            coords = list(map(float, line.strip().split()[1:]))
            contour = np.array(coords, dtype=np.float32).reshape(-1, 2)
            contours.append(contour)
        gt_mask = contour2mask(contours, sz)
        # count iou
        mask_inter = np.logical_and(pred_mask, gt_mask).sum()
        mask_union = np.logical_or(pred_mask, gt_mask).sum()
        if mask_union == 0:
            continue
        iou += mask_inter / mask_union
        num += 1
    best_model.val(data='datasets/data.yaml', split='test', device='0')
    mean_iou = iou/num
    dc = 2*mean_iou/(1+mean_iou)
    print(f'iou = {mean_iou:.3f}, dc = {dc:.3f}')
