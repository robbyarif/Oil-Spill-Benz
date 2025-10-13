from ultralytics import YOLO
from utils import*

def main():
    create_dataset(ratio=(0.8,0.1,0.1), save=True, name='testing')
    create_dataset()
    set_dataset('testing')
    dataset_info()
    model = YOLO('yolo11n-seg.pt')
    model.train(
        data="datasets/data.yaml",
        epochs=1,
        imgsz=1024,
        batch=8,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        box=1.0,
        cls=0.2,
        dfl=0.5,
        cache="ram"
    )
    return 0

if __name__ == "__main__":
    main()