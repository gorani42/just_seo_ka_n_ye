from torchvision.models.detection.rpn import AnchorGenerator  # - RPN(Region Proposal Network)에서 사용하는 앵커 생성기
from torchvision.models.detection import FasterRCNN  # - Faster R-CNN 모델 클래스
from torchvision.transforms import functional as F  # - 이미지 데이터 변환 및 정규화를 위한 유틸리티 함수
import matplotlib.patches as patches  # - 그래프나 이미지 위에 직사각형 등의 도형을 그리는 데 사용
import matplotlib.pyplot as plt  # - 시각화 도구 (학습 손실 그래프 및 이미지 표시)
from torch.optim import Adam  # - Adam 최적화 알고리즘
from PIL import Image  # - 이미지 파일 열기, 편집, 저장을 위한 라이브러리
import torchvision  # - 컴퓨터 비전 관련 데이터셋, 모델, 변환 기능 제공
import numpy as np  # - 수치 계산 및 배열 연산을 위한 라이브러리
import torch  # - 텐서 연산, 신경망 모델 정의, GPU 가속 지원
import os  # - 파일 경로 관리 및 디렉토리 작업을 위한 라이브러리


# 데이터셋 클래스 정의
class TeethDataset(torch.utils.data.Dataset):

    # 이미지와 바운딩박스 파일을 정렬하고 불러옴
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, 'annotation'))))

    # 이미지와 바운딩박스 파일을 가져오는 함수 정의
    def __getitem__(self, idx):
        # 각 파일이 저장된 폴더 이름 설정
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        ann_path = os.path.join(self.root, 'annotation', self.annotations[idx])
        img = Image.open(img_path).convert("RGB")

        # 바운딩박스 정보를 읽어서 변수에 저장
        boxes = []
        with open(ann_path) as f:
            for line in f:
                xmin, ymin, xmax, ymax = map(float, line.strip().split())
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# 정규화를 위한 클래스 정의
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

# 데이터 변환 정의
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Faster_rcnn:
    def __init__(self, batch, epoch):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 데이터셋 경로 지정
        self.train_dataset_path = './Data/Train' # 학습 데이터 경로
        self.eval_dataset_path = './Data/Test' # 평가 데이터 경로
        self.train_dataset = TeethDataset(self.train_dataset_path, transforms=self.get_transform(train=True)) # 학습 데이터
        self.eval_dataset = TeethDataset(self.eval_dataset_path, transforms=self.get_transform(train=False)) # 평가 데이터
        self.batch = batch # 배치
        self.epoch = epoch
        self.transforms = []
        self.data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=self.collate_fn)
        self.data_loader_eval = torch.utils.data.DataLoader(self.eval_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=self.collate_fn)
        # MobileNetV2 정의
        self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        self.backbone.out_channels = 1280

        # Anchor 정의
        self.anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # ROI Pooling 정의
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        # Faster R-CNN 정의
        self.model = FasterRCNN(self.backbone,
                           num_classes=2,
                           rpn_anchor_generator=self.anchor_generator,
                           box_roi_pool=self.roi_pooler)
        self.model.to(self.device)

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(self.params, lr=0.001)
        # Epoch 설정
        self.num_epochs = 100
        self.best_iou = 0.0
        self.model_save_path = "./Model_saved/Faster R-CNN.pth"  # 모델 저장 경로

    def get_transform(self, train):
        self.transforms.append(ToTensor())
        # ImageNet 평균 및 표준편차로 정규화
        self.transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return Compose(self.transforms)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def calculate_iou(self, box_a, box_b):
        # 좌표 계산
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        # 교차 영역 계산
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # 각 박스의 영역 계산
        boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        # IoU 계산
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    # 시각화 및 평가 함수 정의
    def visualize_and_evaluate(self):
        self.model.train()  # 모델을 train 모드로 전환하여 손실 계산이 가능하도록 함
        iou_values = []
        total_loss = 0.0
        loss_count = 0

        with torch.no_grad():
            for images, targets in self.data_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # 훈련 예측과 손실 계산
                loss_dict = self.model(images, targets)
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                    total_loss += losses.item()
                    loss_count += 1

                # 테스트 손실 계산을 위해 eval 모드로 전환
                self.model.eval()
                outputs = self.model(images)

                for img, target, output in zip(images, targets, outputs):
                    img = img.permute(1, 2, 0).cpu().numpy()

                    # NMS 적용
                    keep = torchvision.ops.nms(output['boxes'], output['scores'], 0.5)
                    nms_boxes = output['boxes'][keep]

                    # 테스트 IoU 계산
                    for pred_box in nms_boxes.cpu().numpy():
                        ious = [self.calculate_iou(pred_box, gt_box) for gt_box in target['boxes'].cpu().numpy()]
                        if ious:
                            max_iou = max(ious)
                            iou_values.append(max_iou)

                # 다시 모델을 train 모드로 전환하여 다음 배치 처리
                self.model.train()

        avg_iou = np.mean(iou_values) if iou_values else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

        print(f"Evaluation - Average Loss: {avg_loss:.4f}, Average IoU: {avg_iou:.4f}")

        return avg_iou, avg_loss

    # 학습 및 평가 함수 정의
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        loss_count = 0

        for images, targets in self.data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # 손실 누적
            total_loss += losses.item()
            loss_count += 1

        # 평균 손실 계산 및 출력
        avg_loss = total_loss / loss_count
        print(f"Epoch {self.epoch}, Average Loss: {avg_loss:.4f}")

    def model_train(self):
        for epoch in range(self.num_epochs):
            # 학습
            self.train_one_epoch()

            # 평가
            avg_iou, avg_loss = self.visualize_and_evaluate()

            # 모델 저장 (최고 IoU가 갱신된 경우)
            if avg_iou > self.best_iou:
                best_iou = avg_iou
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"New best model saved with IoU: {best_iou:.4f}")

        print("Training and evaluation complete.")
        print(f"Best model saved with IoU: {self.best_iou:.4f} at {self.model_save_path}")

    # Dice Score 계산 함수
    def calculate_dice(self, pred_box, gt_box):
        intersection = self.calculate_iou(pred_box, gt_box) * (
                (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        )
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        dice = (2 * intersection) / (pred_area + gt_area) if (pred_area + gt_area) != 0 else 0
        return dice

    # 이미지 역정규화 함수
    def denormalize(self, img, mean, std):
        img = img.clone()  # 원본 이미지에 영향을 미치지 않도록 복제
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return img

    # 시각화 및 평가 함수 정의
    def visualize_and_test(self, model):
        model.eval()
        all_image_ious = []
        all_image_dices = []
        image_count = 0
        max_images = 5  # 최대 5개의 이미지만 시각화
        with torch.no_grad():
            for images, targets in self.data_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.model(images)

                for img, target, output in zip(images, targets, outputs):
                    # 이미지를 최대 max_images만큼만 시각화
                    if image_count < max_images:
                        # 이미지를 정규화되지 않은 상태로 변환
                        img = self.denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        img = img.permute(1, 2, 0).cpu().numpy()
                        fig, ax = plt.subplots(1, figsize=(12, 9))
                        ax.imshow(img)

                        # NMS 적용
                        keep = torchvision.ops.nms(output['boxes'], output['scores'], 0.5)
                        nms_boxes = output['boxes'][keep]

                        # 실제 박스 그리기
                        for box in target['boxes'].cpu().numpy():
                            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2,
                                                     edgecolor='g', facecolor='none')
                            ax.add_patch(rect)

                        # 예측 박스 그리기
                        for box in nms_boxes.cpu().numpy():
                            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2,
                                                     edgecolor='r', facecolor='none')
                            ax.add_patch(rect)

                        plt.show()
                        image_count += 1

                    # 각 이미지별 IoU 및 Dice Score 계산
                    image_ious = []
                    image_dices = []
                    for pred_box in output['boxes'].cpu().numpy():
                        ious = [self.calculate_iou(pred_box, gt_box) for gt_box in target['boxes'].cpu().numpy()]
                        dices = [self.calculate_dice(pred_box, gt_box) for gt_box in target['boxes'].cpu().numpy()]
                        if ious and dices:
                            max_iou = max(ious)
                            max_dice = max(dices)
                            image_ious.append(max_iou)
                            image_dices.append(max_dice)

                    if image_ious and image_dices:
                        avg_image_iou = np.mean(image_ious)
                        avg_image_dice = np.mean(image_dices)
                        all_image_ious.append(avg_image_iou)
                        all_image_dices.append(avg_image_dice)
                        print(
                            f"Image {image_count} - Average IoU: {avg_image_iou:.4f}, Average Dice Score: {avg_image_dice:.4f}")

        # 전체 이미지에 대한 평균 IoU 및 Dice Score 계산
        avg_iou = np.mean(all_image_ious) if all_image_ious else 0
        avg_dice = np.mean(all_image_dices) if all_image_dices else 0
        return avg_iou, avg_dice

    def model_test(self):
        model = self.model.load_state_dict(torch.load('./Model_saved/Faster R-CNN.pth'))
        model.to(self.device)
        avg_iou, avg_dice = self.visualize_and_test(model)
        print(f"Overall Average IoU: {avg_iou:.4f}")
        print(f"Overall Average Dice Score: {avg_dice:.4f}")


