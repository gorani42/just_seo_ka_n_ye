import torch
import torchvision
import segmentation_models_pytorch as smp
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class Framework:
    def __init__(self, image_path=r"./Code/Test data for framework/372.png"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 유무
        self.image_path = image_path  # 이미지 경로

    def image_upload(self):  # 이미지 업로드
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 불러오기
        image = cv2.resize(image, (2943, 1435))  # 이미지 크기가 다를 경우를 대비해 resize 진행

        return image

    def image_size_define(self):  # 이미지 사이즈 설정
        image = self.image_upload()
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 이미지 분포를 위한 인스턴스 생성
        image_clahe = clahe.apply(image)  # 가져온 이미지를 적용

        # 구강 영역 Crop할 비율 정의
        image_height, image_width = image.shape[:2]  # 이미지 높이, 넓이 정의
        left = int(image_width * 0.2)  # 좌
        right = int(image_width * 0.8)  # 우
        top = int(image_height * 0.2)  # 높이
        bottom = int(image_height * 0.9)  # 넓이

        return image_clahe, left, right, top, bottom

    def image_preprocessing(self):  # 이미지 전처리
        image_clahe, left, right, top, bottom = self.image_size_define()
        # 구강 영역 Crop
        cropped_image = image_clahe[top:bottom, left:right]  # 이미지에서 특정 영역 자르기
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)  # 자른 이미지를 GRAY에서 RGB 변환
        cropped_image_pil = Image.fromarray(cropped_image_rgb)  # OpenCV 이미지를 PIL 이미지로 변환

        return cropped_image_pil

    def faster_rcnn_load(self):  # Faster R-CNN Load
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features  # MobileNetV2의 특징 추출 부분(features) 사용
        backbone.out_channels = 1280  # Backbone의 출력 채널 수를 지정

        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0,
                                                           2.0),))  # Anchor Generator 정의: RPN(Region Proposal Network)에서 사용할 앵커(Anchor) 생성기
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)  # RoI Pooler 정의: Region of Interest에서 고정된 크기의 특징을 추출
        faster_rcnn_model = FasterRCNN(backbone,
                                       num_classes=2,
                                       rpn_anchor_generator=anchor_generator,
                                       box_roi_pool=roi_pooler)  # Faster R-CNN 모델 생성

        state_dict = torch.load('./Code/Model/Faster R-CNN.pth')
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('rpn.head.conv.0.0', 'rpn.head.conv')
            new_state_dict[new_key] = v

        faster_rcnn_model.load_state_dict(new_state_dict)
        faster_rcnn_model.to(self.device)
        faster_rcnn_model.eval()

        return faster_rcnn_model

    def image_normalization(self):
        cropped_image_pil = self.image_preprocessing()  # 이미지 불러오기
        faster_rcnn_model = self.faster_rcnn_load()  # Faster R-CNN 불러오기
        # 이미지 정규화 (ImageNet)
        transform = T.Compose([
            T.ToTensor(),  # 텐서 변환
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 평균 및 표준편차로 정규화
        ])

        image_tensor = transform(cropped_image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = faster_rcnn_model(image_tensor)
        boxes = predictions[0]['boxes'].cpu().numpy()

        # 4. 바운딩 박스 좌표로 이미지 crop하고 리사이즈
        cropped_images = []
        resize_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            # 치아 바운딩 박스 좌표를 사용해 이미지 크롭
            cropped_image = cropped_image_pil.crop((xmin, ymin, xmax, ymax))
            cropped_image = resize_transform(cropped_image)
            cropped_images.append(cropped_image)

        return cropped_images, boxes

    def unet_load(self):
        unet_model = torch.load('./Code/Model/U-Net.pth', map_location=self.device)
        unet_model = unet_model.to(self.device)
        unet_model.eval()

        return unet_model

    def predict_mask(self, image_tensor, model):
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(image_tensor)
            preds = torch.sigmoid(logits)
        return preds.squeeze().cpu().detach().numpy()

    def show_result(self):
        image = self.image_upload()  # 원본 이미지 불러오기
        image_clahe, left, right, top, bottom = self.image_size_define()  # 이미지 사이즈 불러오기
        unet_model = self.unet_load()  # U-NET 모델 불러오기
        cropped_images, boxes = self.image_normalization()  # 정규화 이미지 불러오기

        masks = [self.predict_mask(image_tensor, unet_model) for image_tensor in cropped_images]

        # 6. 파노라마 이미지와 같은 사이즈와 같은 크기의 검정색 마스크 생성
        full_mask = np.zeros_like(image)

        # 7. U-Net 예측 결과를 바운딩 박스 좌표를 검정색 마스크에 붙이기
        for box, mask in zip(boxes, masks):
            xmin, ymin, xmax, ymax = map(int, box)
            mask_resized = cv2.resize(mask, (xmax - xmin, ymax - ymin))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255  # 이진 마스크
            full_mask[top + ymin:top + ymax, left + xmin:left + xmax] = np.where(mask_binary, mask_binary,
                                                                                 full_mask[top + ymin:top + ymax,
                                                                                 left + xmin:left + xmax])
        # 8. 예측 결과를 원본 이미지에 오버레이
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        red_mask = np.zeros_like(color_image)
        red_mask[:, :, 1] = full_mask

        overlay_image = cv2.addWeighted(color_image, 0.5, red_mask, 1, 0)

        # 원본 이미지의 파일명에서 확장자를 제외한 이름을 추출
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]

        # 결과 이미지 저장 경로 생성 ('원본이미지 이름_overlay.png')
        save_path = f'./result/{base_filename}_overlay.png'

        # 결과 이미지 보여주기 및 저장
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.imshow(overlay_image)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 결과이미지 저장

        return plt.show()