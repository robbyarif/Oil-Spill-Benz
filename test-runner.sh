### Exp 2.2a Port Oil Spill

# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp2.3a_port-oil-pretrain/yolov11-exp2.3a-port-oil/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.3a/yolov11 \
#     --trainer yolo \
#     --model-version 11 \
#     --export-csv runs/test_by_date_exp2.3a/yolov11/metrics.csv

python test_by_date.py \
    --test-file datasets/processed/exp2.4-dv3train/test.txt \
    --model-weights runs/exp2.2-port-oil/yolov12-exp2.2-port-oil/weights/best.pt \
    --dataset-dir datasets/processed/exp2.4-dv3train \
    --output-dir runs/test_by_date_exp2.2a/yolov12 \
    --trainer yolo \
    --model-version 12 \
    --export-csv runs/test_by_date_exp2.2a/yolov12/metrics.csv


# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp2.3a_port-oil-pretrain/deeplabv3-exp2.3a-port-oil/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.3a/deeplabv3 \
#     --trainer deeplabv3 \
#     --export-csv runs/test_by_date_exp2.3a/deeplabv3/metrics.csv

python test_by_date.py \
    --test-file datasets/processed/exp2.4-dv3train/test.txt \
    --model-weights runs/exp2.2-port-oil/segformer-exp2.2-port-oil/weights/best.pt \
    --dataset-dir datasets/processed/exp2.4-dv3train \
    --output-dir runs/test_by_date_exp2.3a/segformer \
    --trainer segformer \
    --export-csv runs/test_by_date_exp2.3a/segformer/metrics.csv

# =====

# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp-manager-20260206/yolov11-exp2.4-dv3train/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.4/yolov11 \
#     --trainer yolo \
#     --model-version 11 \
#     --export-csv runs/test_by_date_exp2.4/yolov11/metrics.csv

# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp-manager-20260206/yolov12-exp2.4-dv3train/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.4/yolov12 \
#     --trainer yolo \
#     --model-version 12 \
#     --export-csv runs/test_by_date_exp2.4/yolov12/metrics.csv


# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp-manager-20260206/deeplabv3-exp2.4-dv3train/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.4/deeplabv3 \
#     --trainer deeplabv3 \
#     --export-csv runs/test_by_date_exp2.4/deeplabv3/metrics.csv

# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp-manager-20260206/segformer-exp2.4-dv3train/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.4/segformer \
#     --trainer segformer \
#     --export-csv runs/test_by_date_exp2.4/segformer/metrics.csv

# python test_by_date.py \
#   --test-file datasets/processed/exp2.4-dv3train/test.txt \
#   --model-weights runs/exp2.4-dv3train_rfdetrsegnano/checkpoint_best_total.pth \
#   --dataset-dir datasets/processed/exp2.4-dv3train \
#   --output-dir runs/test_by_date_exp2.4/rfdetr \
#   --trainer rfdetr \
#   --export-csv runs/test_by_date_exp2.4/rfdetr/metrics.csv


### Exp 2.3a Pretrained Port Oil Spill

# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp2.3a_port-oil-pretrain/yolov11-exp2.3a-port-oil/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.3a/yolov11 \
#     --trainer yolo \
#     --model-version 11 \
#     --export-csv runs/test_by_date_exp2.3a/yolov11/metrics.csv

# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp2.3a_port-oil-pretrain/yolov12-exp2.3a-port-oil/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.3a/yolov12 \
#     --trainer yolo \
#     --model-version 12 \
#     --export-csv runs/test_by_date_exp2.3a/yolov12/metrics.csv


# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp2.3a_port-oil-pretrain/deeplabv3-exp2.3a-port-oil/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.3a/deeplabv3 \
#     --trainer deeplabv3 \
#     --export-csv runs/test_by_date_exp2.3a/deeplabv3/metrics.csv

# python test_by_date.py \
#     --test-file datasets/processed/exp2.4-dv3train/test.txt \
#     --model-weights runs/exp2.3a_port-oil-pretrain/segformer-exp2.3a-port-oil/weights/best.pt \
#     --dataset-dir datasets/processed/exp2.4-dv3train \
#     --output-dir runs/test_by_date_exp2.3a/segformer \
#     --trainer segformer \
#     --export-csv runs/test_by_date_exp2.3a/segformer/metrics.csv

# python test_by_date.py \
#   --test-file datasets/processed/exp2.4-dv3train/test.txt \
#   --model-weights runs/exp2.3a_port-oil-pretrain/rfdetr-exp2.3a-port-oil/checkpoint_best_total.pth \
#   --dataset-dir datasets/processed/exp2.4-dv3train \
#   --output-dir runs/test_by_date_exp2.4/rfdetr \
#   --trainer rfdetr \
#   --export-csv runs/test_by_date_exp2.4/rfdetr/metrics.csv