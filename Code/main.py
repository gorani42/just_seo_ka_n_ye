from Framework import Framework
from Models.U_Net.U_Net import Unet, DentalDataset
from Models.Faster_R_CNN.Faster_R_CNN import FasterRcnn
import argparse

def main(result_type):
    if result_type == "eval": # 학습된 모델의 결과를 확인하거나 평가할 떄 사용
        init_framework = Framework()
        init_framework.show_result()
    else: # 모델을 학습 할 때 사용
        # U-NET 모델 학습
        unet_init = Unet(train_dataset=DentalDataset(image_dir='./Data/Train/images', mask_dir='./Data/Train/labels'),
                         eval_dataset=DentalDataset(image_dir='./Data/Test/images', mask_dir='./Data/Test/labels'),
                         test_dataset=DentalDataset(image_dir='./Data/Test/images', mask_dir='./Data/Test/labels'))
        unet_init.model_train(num_epochs=60)
        unet_init.model_testing()

        # Faster R-CNN 모델 학습
        faster_rcnn_init = FasterRcnn(train_dataset_path="./Data/Train", eval_dataset_path="./Data/Test", batch=20, epoch=50)
        faster_rcnn_init.model_train()
        faster_rcnn_init.model_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval 또는 train 을 입력해 주세요")
    parser.add_argument("--result_type", type=str, required=True, help="result_type 값을 입력해주세요.")
    args = parser.parse_args()
    main(args.type)
