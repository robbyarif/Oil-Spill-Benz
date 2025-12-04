import os.path
from ultralytics import YOLO
import argparse

class WARNING(Exception):
    pass

def get_args():
    parser = argparse.ArgumentParser(description='custom train parameter')
    parser.add_argument("--dataset", type=str, help="the dataset used to train")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--img_sz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--name", type=str, help="the name of the folder to save results")
    parser.add_argument("--conf", type=float, default=0.7, help="the confidence threshold for testing")
    args = parser.parse_args()
    try:
        if not args.dataset:
            raise WARNING("please choose a dataset.")
        elif not os.path.exists(f"datasets/{args.dataset}/data.yaml"):
            raise WARNING(f"dataset \"{args.dataset}\" doesn't exist or loss a data.yaml file.")
        else:
            args.dataset = f"datasets/{args.dataset}/data.yaml"
    except WARNING as e:
        print("WARNING:", e)
        exit()
    if not args.name:
        temp = 0
        while os.path.exists(f'runs/segment/train_{temp}'):
            temp += 1
        args.name = f'train_{temp}'
    return args

def main():
    args = get_args()
    model = YOLO('yolo11n-seg.pt')

    model.train(
        data = args.dataset,
        epochs = args.epochs,
        imgsz = args.img_sz,
        batch = args.batch,
        optimizer = args.optimizer,
        lr0 = args.lr0,
        box = 6,
        device = 0,
        workers = 8,
        name = args.name,
        plots = False
    )

if __name__ == "__main__":
    main()

