from yolo import YoloTrainer
from deeplabv3 import DeeplabTrainer
from segformer import SegformerTrainer

def main():
    src = "datasets/new_baseline"

    trainer = YoloTrainer()
    trainer.load_model()
    trainer.train(src, save=False, epochs=3)

if __name__ == "__main__":
    main()