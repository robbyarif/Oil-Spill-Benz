# from rfdetr import RFDETRSegNano
from rfdetr import RFDETRSegMedium

def main():
    model = RFDETRSegMedium(device="cuda")
    model.train(
        # dataset_dir="datasets/processed/exp2.4-dv3train_organized_yolo_resized_coco",
        # dataset_dir="datasets/processed/dv4_random_split_coco_312",
        dataset_dir="datasets/lados_432",
        epochs=100, 
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="runs/lados_rfdetrsegmedium",
        # gradient_checkpointing=True,
        # resolution=60,
        wandb=False,
        project="Oil-Spill-UAV",
        run="lados-rfdetrsegmedium",
        # resume="output/checkpoint.pth"
        # num_select=100  # Reduce from default 300 for smaller datasets
    )


if __name__ == "__main__":
    main()

# How to run:
# python rfdetr.py | tee -a logs/output-exp2.4-dv3train.txt 