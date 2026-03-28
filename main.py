# from rfdetr import RFDETRTrainer
from yolo import YoloTrainer
from deeplabv3 import DeeplabTrainer
from segformer import SegformerTrainer
from rfdetr_trainer import RFDETRTrainer

def main():
    # src = "datasets/new_baseline"
    # src = "datasets/processed/exp2.1_inc-split"
    # src = "datasets/processed/exp2.2_port-oil-split"
    # src = "datasets/processed/exp2.4-dv3train"
    # src = "datasets/processed/exp3.1a-dv3train_dv4test_woketa34"
    src = "datasets/processed/exp3.2a-dv3train_dv4test_woketa34_woliqgem"

    # exp_checkpoint_base_path = "/home/robby/workspace/Oil-Spill-Benz/runs/exp-manager"
    exp_checkpoint_base_path = "/home/robby/workspace/Oil-Spill-Benz/runs/exp2.3a_port-oil-pretrain/"

    trainer_yolo11 = YoloTrainer()
    yolov11_path = f"{exp_checkpoint_base_path}/yolov11-exp2.3a-port-oil/weights/best.pt"
    trainer_yolo11.load_model(weights=yolov11_path)
    # trainer_yolo.train(src, save=True, epochs=3, dst="runs/experiments/exp2.2_port-oil-split/yolo")
    trainer_yolo11.test(src, dst="runs/test/exp3.2a_port-oil/yolov11", save=True, file_name="test.txt", color_coded=True)

    trainer_yolo12 = YoloTrainer()
    yolov12_path = f"{exp_checkpoint_base_path}/yolov12-exp2.3a-port-oil/weights/best.pt"
    trainer_yolo12.load_model(weights=yolov12_path, version=12)
    # trainer_yolo.train(src, save=True, epochs=200, dst="runs/experiments/exp2.2_port-oil-split/yolo")
    trainer_yolo12.test(src, dst="runs/test/exp3.2a_port-oil/yolo12", save=True, file_name="test.txt", color_coded=True)

    trainer_deeplab = DeeplabTrainer()
    deeplab_path = f"{exp_checkpoint_base_path}/deeplabv3-exp2.3a-port-oil/weights/last.pt"
    trainer_deeplab.load_model(weights=deeplab_path)
    # trainer_deeplab.train(src, epochs=3, batch=16, save=False, dst="runs/experiments/exp2.1_inc-split/deeplabv3")
    trainer_deeplab.test(src, dst="runs/test/exp3.2a_port-oil/deeplabv3", save=True, file_name="test.txt", color_coded=True)

    trainer_segformer = SegformerTrainer()
    segformer_path = f"{exp_checkpoint_base_path}/segformer-exp2.3a-port-oil/weights/best.pt"
    trainer_segformer.load_model(weights=segformer_path)
    # trainer_segformer.train(src, save=True, epochs=3, dst="runs/experiments/exp2.1_inc-split/segformer")
    trainer_segformer.test(src, dst="runs/test/exp3.2a_port-oil/segformer", save=True, file_name="test.txt", color_coded=True)


    # # # test_img_path = "/home/robby/workspace/Oil-Spill-Benz/DV4/images/20240725_多哥籍「阿諾(ALANO)」貨輪疑似流錨協處案/20240825/UAV_20240825_sophia_0306.jpg"
    # trainer_rfdetr = RFDETRTrainer()
    # rfdetr_path = f"/home/robby/workspace/Oil-Spill-Benz/runs/exp2.4-dv3train_rfdetrsegnano/checkpoint_best_total.pth"
    # trainer_rfdetr.load_model(weights=rfdetr_path)
    # # trainer_rfdetr.train(src, save=True, epochs=3, dst="runs/experiments/exp2.4_rfdetr_test")
    # trainer_rfdetr.test(src, dst="runs/test/exp2.4_rfdetr-test", save=True, file_name="test.txt", color_coded=True)

    


if __name__ == "__main__":
    main()