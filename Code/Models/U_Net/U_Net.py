from torch.utils.data import Dataset, DataLoader  # Dataset: 데이터셋을 정의하는 클래스 - DataLoader: Dataset을 반복 가능하게 로드하며 배치 처리, 셔플 등을 지원
import torchvision.transforms as transforms # - 이미지 데이터의 전처리 및 데이터 증강(augmentation) 기능 제공
import segmentation_models_pytorch as smp # - 다양한 분할 모델과 백본(backbone) 네트워크를 지원
import torch.nn.functional as F # - 손실 함수, 활성화 함수 등 신경망 연산을 위한 함수 제공
import matplotlib.pyplot as plt # - 이미지를 표시하거나 학습 과정의 손실(loss), 정확도(accuracy) 등을 그래프로 그리는 데 사용
from tqdm.notebook import tqdm # 반복문의 진행 상황을 시각적으로 표시:
import torch.nn as nn # - 신경망 레이어(Convolution, Linear 등)를 정의하고 구성
import numpy as np # - 배열 연산 및 행렬 계산을 효율적으로 수행
from PIL import Image
import torch # - 텐서 연산, 신경망 모델 정의, GPU 가속 지원
import os


class DentalDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = transforms.Compose([
                                                    transforms.Resize((256, 256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                  ])
        self.mask_transform = transforms.Compose([
                                                    transforms.Resize((256, 256)),
                                                    transforms.ToTensor()
                                                ])
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("The number of images and masks do not match")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()  # Convert mask to binary (0 and 1)

        return image, mask

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, activation=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def forward(self, inputs, targets):
        if self.activation:
            inputs = self.activation(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

class Unet:
    def __init__(self, train_dataset, eval_dataset, test_dataset):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 학습 데이터
        self.eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False) # 평가 데이터
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 테스트 데이터
        self.model = smp.Unet(encoder_name='efficientnet-b0',
                              encoder_weights='imagenet',
                              in_channels=3,
                              classes=1
                              ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4) # 손실함수 설정
        self.criterion = DiceLoss(activation=F.sigmoid)
        self.model_name = 'UNetEfficientnetB0'
        self.IoU_max = 0.
        self.losses_train, self.losses_val = [], []
        self.metrics = []

    @staticmethod
    def seed_everything(seed=42): # 시드 설정 (훈련환경 변화 방지)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def metric_calculate(prediction: np.ndarray, target: np.ndarray):  # 평가지표 함수
        target = np.uint8(target.flatten() > 0.5)
        prediction = np.uint8(prediction.flatten() > 0.5)
        TP = (prediction * target).sum()
        FN = ((1 - prediction) * target).sum()
        TN = ((1 - prediction) * (1 - target)).sum()
        FP = (prediction * (1 - target)).sum()

        acc = (TP + TN) / (TP + TN + FP + FN + 1e-4)
        iou = TP / (TP + FP + FN + 1e-4)
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-4)
        pre = TP / (TP + FP + 1e-4)
        spe = TN / (FP + TN + 1e-4)
        sen = TP / (TP + FN + 1e-4)

        return acc, iou, dice, pre, spe, sen

    def model_train(self, num_epochs: int = 50):

        for epoch in tqdm(range(num_epochs)):
            current_train_loss, current_val_loss = 0., 0.
            current_metric = np.zeros(6)

            # 훈련 과정
            self.model.train()
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                current_train_loss += loss.item() / len(self.train_loader)

            # 평가 과정
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.eval_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                    current_val_loss += loss.item() / len(self.eval_loader)
                    current_metric += np.array(self.metric_calculate(logits.cpu().detach().numpy(),
                                                                labels.cpu().detach().numpy())) / len(self.eval_loader)

            self.losses_train.append(current_train_loss)
            self.losses_val.append(current_val_loss)
            self.metrics.append(current_metric.tolist())

            if self.IoU_max < self.metrics[-1][1]:
                torch.save(self.model, f'{self.model_name}-best.pth')
                IoU_max = self.metrics[-1][1]

            print(f'Epoch: {epoch + 1}, train_loss: {self.losses_train[-1]:.4f}, val_loss: {self.losses_val[-1]:.4f}, IoU: {self.metrics[-1][1]:.4f}')

        log = {'train_loss': self.losses_train,
               'eval_loss': self.losses_val,
               'metric': self.metrics,
               'best_score': self.IoU_max}

        torch.save(self.model, f'{self.model_name}-last.pth')

        # with open(f'log.txt', 'w') as outfile:
        #     json.dump(log, outfile)

        torch.cuda.empty_cache()

        print('- - ' * 30)
        print(f'Training {self.model_name} done. Best IoU: {self.IoU_max:.4f}.')
        print('- - ' * 30)

    def train_result_plot(self): # Loss, IoU 그래프 Plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.losses_train, label='train_loss')
        plt.plot(self.losses_val, label='val_loss')
        plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot([metric[1] for metric in self.metrics], label='IoU')
        plt.axhline(self.IoU_max, linestyle='--', color='red', label=f'IoU_max: {self.IoU_max:.4f}')
        plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('IoU')

        plt.savefig('efficientnet-b0.png')
        plt.show()

    @staticmethod
    def evaluate_segmentation(pred, label):
        pred_binary = pred > 0.5
        label_binary = label > 0.5
        intersection = np.sum(pred_binary * label_binary)
        union = np.sum(pred_binary + label_binary) - intersection
        iou = intersection / union if union != 0 else 0
        dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(label_binary)) if (np.sum(pred_binary) + np.sum(
            label_binary)) != 0 else 0
        return iou, dice

    def plot_results(self, images, labels, preds, iou_scores, dice_scores):
        # denormalize 함수를 사용하여 이미지를 원래 상태로 복원
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = images * std + mean
        images = images.clamp(0, 1)  # [0, 1] 범위로 클램핑

        for i in range(len(images)):
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
            plt.title('Input Image')

            plt.subplot(1, 3, 2)
            plt.imshow(labels[i].squeeze().cpu(), cmap='gray')
            plt.title('Ground Truth Mask')

            plt.subplot(1, 3, 3)
            plt.imshow(preds[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f'Predicted Mask\nIoU: {iou_scores[i]:.4f}, Dice: {dice_scores[i]:.4f}')

            plt.show()

    def model_testing(self): # 테스트셋 이미지 예측 시각화 및 평가지표 계산
        # 정규화에 사용된 mean std 정의
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        model = torch.load('./UNetEfficientnetB0-best.pth')
        model = model.to(self.device)
        model.eval()

        total_dice_score = 0
        num_images = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = model(images)
                preds = torch.sigmoid(logits)

                iou_batch_scores = []
                dice_batch_scores = []
                for i in range(images.size(0)):
                    iou, dice = self.evaluate_segmentation(preds[i].cpu().numpy(), labels[i].cpu().numpy())
                    iou_batch_scores.append(iou)
                    dice_batch_scores.append(dice)
                    total_dice_score += dice
                    num_images += 1

                self.plot_results(images, labels, preds, iou_batch_scores, dice_batch_scores)
                break  # Only plot one batch to avoid too many plots

        average_dice_score = total_dice_score / num_images if num_images != 0 else 0
        print(f'Average Dice Score: {average_dice_score:.4f}')



