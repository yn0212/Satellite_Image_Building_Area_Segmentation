# Satellite_Image_Building_Area_Segmentation 💫🏡
SW중심대학 공동 AI 경진대회 2023
![header](https://capsule-render.vercel.app/api?type=waving&color=ADD8E6&height=300&section=header&text=Segmentation%20of%20building%20areas&desc=in%20satellite%20imagery&fontSize=50&demo=wave&fontColor=696969)
 2023.07.03 ~ 2023.07.28 09:59

# :pushpin: Competition Background and Objectives
목표 : 위성 이미지의 건물 영역 분할(Image Segmentation)을 수행하는 AI모델을 개발
----------------------------------------------
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/b697c6e6-c9b4-4908-a5ec-61ec41eaa91d)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/2640b31d-64ff-44cc-b019-13ddf37a15f4)

### 결과 맛보기😜
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/f6505440-824b-47da-af17-9e636f7d9629)

[블로그 정리](https://yn0212.tistory.com/category/AI/ai%20contest)

# :pushpin: EDA
## 1.데이터셋 구성 파악: 데이터셋에 포함된 위성 이미지와 해당 이미지에 대한 건물 영역의 마스크 정보를 확인.

 

Dataset Info.

train_img [폴더]
TRAIN_0000.png ~ TRAIN_7139.png
1024 x 1024
학습 위성 이미지의 촬영 해상도는 0.5m/픽셀이며, 추론 위성 이미지의 촬영 해상도는 공개하지 않습니다.
test_img [폴더]
TEST_00000.png ~ TEST_60639.png
224 x 224
polygon정보는 이진이미지 형태로 rle인코딩 되어있음.

 

 

## 2.이미지 시각화: 데이터셋의 일부 이미지와 해당 건물 영역 마스크를 시각화하여 어떤 형태의 이미지와 건물 분포가 있는지 확인 .이를 통해 데이터셋의 특성을 이해.
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/5cfad1b4-80ab-4872-bde7-8c6387b10985)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/cebff782-866f-46f3-a496-282debc1f255)


train 데이터

test 데이터
test데이터이미지의 위성의  고도가 더 낮은것을 확인하였음.

=>1024x1024 사이즈의 훈련데이터를 그대로 사용하기보다는 test이미지의 건물 크기와 비슷하게 확대해야 함
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/d3f71c39-ea86-4385-b75d-bed2f467be36)


## 3.건물 영역 분포 분석: 데이터셋에 포함된 건물 영역의 분포를 분석.

 

- 훈련 이미지의 배경 대비 건물 면적 그래프 생성
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/e89bcbe1-a9ee-4592-aa13-ce8db25b328d)


총 이미지 중 건물 면적 비율이 1%인 사진이 약 19% 

건물 면적이 15%이상인 위성 영상은 적음

평균적으로 5~10%대

 

=> 데이터에서 배경이 차지하는 비율이 높다.

=> 데이터 불균형 처리를 위한 적절한 데이터 전처리와 증강 기법을 적용하여야함.

 

## insight : 설계 방향성
EDA를 통해 Test 데이터의 위성 고도가 낮아 건물 크기가 작아진 것을 확인하였고,평균적으로 5~10% 대의 건물 면적 비율을 학인하였다. 정확한 예측을 위해서는 클래스 불균형 처리와 test dataset에 맞는 augmentation 기법이 중요하다는 것을 확인 하였다.

그래서 훈련 데이터를 적절히 확대 및 리사이징하여 고도에 따른 건물 크기 변화에 대응하는 augmentation 기법들을 설계하였고 건물 면적 비율과 관련하여, 데이터 불균형 처리를 위해 클래스 가중치 설정 또는 데이터 증강 기법을 활용하여 배경보다 건물 클래스에 더 중요한 가중치를 부여하여 모델이 클래스간의 경계를 명확하게 분할하게 하는 것을 목적으로 하였다.

 

 

 

# :pushpin: Data preprocessing test
## 1.data augmentation
(1) RandomCrop

- train데이터보다 test데이터이미지의 위성의  고도가 더 낮아 건물 크기가 다른 것을 확인하였음. 훈련데이터를 그대로 사용하기보다 test이미지의 건물 크기와 비슷하게 확대해주어야 하므로, 훈련데이터를 분할하여 자체 평가산식 결과 dice socre가 가장 높은 RandomCrop(224,224)를 이용해 훈련데이터를 임의로 증강 및 224사이즈로 확대하였습니다.

 
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/0ab487ec-d68e-4001-a8bb-9008dcd968fc)


*  A.RandomResizedCrop(224, 224, scale=(0.5, 0.21875), ratio=(1, 1))와 같이 설정하면, 원본 이미지를 랜덤하게 512x512에서 224x224 크기 사이로 확대하여 자르게 된다.

* 최소 확대 비율: 50% = 1024 * 0.5 = 512

* 최대 확대 비율: 약 21.875% = 1024 * 0.21875 ≈ 224

 

-RandomCrop(224,224)적용 시각화
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/08c6cf50-cc23-4668-9f5c-8dd19b727d0f)



 

## 2.데이터 불균형 처리 test
(1) Filtering (사용x)
-eda에서 데이터에서 배경이 차지하는 비율이 높아 데이터 불균형 처리를 해야함을 확인하였음. 데이터셋에서 배경 대비 건물 면적 비율에따라 특정 기준을 충족하지 않는 샘플들을 제거하는 작업으로 똑같은 조건에서 비교해본 결과, 훈련 데이터 분할로 자체 평가산식 점수가 가장 높은 '배경 대비 건물 비5% 이하를 제외하는 필터링'이 선택되었다. 
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/6837b9ee-87dd-411c-9b04-10a187ce8bc4)

 

하지만 7000장의 데이터셋에서 배경 대비 건물 비5% 이하를 제외하면 2929장의 훈련데이터를 사용해야하는데 , 이렇게 되면 데이터셋의 다양성을 상실할 수 있고, 모델의 일반화 능력이 저하될 확률이 커진다.

 

같은 조건에서 리더보드 점수는 전체 데이터셋을 사용하여 학습한 모델의 score가 더 높았다.

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/ef8277d4-64a3-4c43-b4e1-d42a84bd4c1c)

 

 

### (2) CropNonEmptyMaskIfExists(height=224, width=224)
-위의 (1)번의 대안으로 데이터 불균형 처리를 위해 CropNonEmptyMaskIfExists 데이터 증강 기법을 활용하여 일부 이미지에 마스크가 존재하지 않는 빈(mask가 없는) 영역을 제외하고 이미지를 잘라내도록 하였다.

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/dbaafd40-1226-485f-8d7b-1ecb969fe15d)


randomcrop과 데이터 불균형 처리를 동시에 할 수 있어서 모델 성능이 향상됨을 확인.
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/47f2d5ad-b3d3-4770-918f-9541a7335b82)

 

- 선택된 aumentation 기법 code

  데이터 증강을 위한 transform 파이프라인 정의
  ```python
   transform = A.Compose(
       [
           A.CropNonEmptyMaskIfExists(height=224, width=224),
           A.HorizontalFlip(p=0.5),
           A.VerticalFlip(p=0.5),
           A.Rotate(limit=30,p=0.3),
           A.ColorJitter(p=0.3),
           A.ToGray(p=0.2),
   
           #고정값
      
           A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지 픽셀값 정규화
           ToTensorV2()  # 이미지를 텐서로 변환
       ]
   )

# :pushpin: Hyperparameter tuning

## Learning Rate (학습률)
적절한 학습률을 찾기위해 학습률 스케줄링을 사용하였다.

```python
  # 옵티마이저 설정
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
  import torch.optim as optim
  from torch.optim.lr_scheduler import StepLR
  # 학습률 스케줄링 설정
  scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
   
  #훈련에폭 안에 
  scheduler.step()  # 매 epoch마다 학습률을 업데이트합니다.
```
위의 코드처럼 PyTorch의 학습률 스케줄러 중 하나인 StepLR 클래스를 import해고, StepLR(optimizer, step_size=10, gamma=0.1)인 StepLR 클래스를 사용하여 학습률 스케줄링을 설정했다.위의 StepLR은 주어진 step_size인 10에포크마다 학습률을 gamma 비율인 0.1배씩 감소시킨다.이러한 스케줄링을 통해 학습이 진행되면서 학습률이 점진적으로 줄어들어 더욱 안정적으로 모델을 학습할 수 있었다.

## Early Stopping (조기 종료)
patience=5로 설정하면 검증 손실이 5번 연속으로 개선되지 않으면 학습이 조기 종료되도록 설정함.
verbose: True로 설정하면 개선이 없을 때마다 로그를 출력하며, False로 설정하면 로그를 출력하지 않는다.

```python
 from early_stopping import EarlyStopping
 # Early Stopping 객체 생성
 # 검증 손실이 5번 연속으로 개선되지 않으면 학습이 조기 종료 되도록 설정
 #verbose: True로 설정하면 개선이 없을 때마다 로그를 출력하며, False로 설정하면 로그를 출력하지 않습니다.
 early_stopping = EarlyStopping(patience=5, verbose=True)
     # Early Stopping 체크
     early_stopping(val_loss, model)
     
     if early_stopping.early_stop:
         print("Early stopping")
         break
```

## Regularization (정규화) test

### 1.Dropout(사용x)
비교시 dropout이 성능을 향상시키지 못하고 오히려 성능이 저하됨을 확인했다.

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/fefffc31-77af-489a-b6cd-c0487466ff7e)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/096ab34a-642d-44ba-be25-693d4fa80c57)

### L2규제

-사용한 L2규제 코드
```python
# 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
```
weight_decay=1e-3은 L2 규제의 강도가 상대적으로 강하게 적용되는 값이다. 즉, 가중치의 크기를 상대적으로 작게 유지하여 모델의 복잡성을 줄이고 일반화 능력을 향상시키려는 목적으로 사용하였다.

## Loss function test
- FocalDiceLoss
위의 손실 함수에서는 Focal Loss와 Dice Loss를 결합하여 사용했다. 먼저, 입력값을 시그모이드 함수를 통해 확률값으로 변환한다. 이후 Focal Loss와 Dice Loss를 계산하여 각각 focal_loss와 dice_loss에 저장한다. 마지막으로 focal_loss와 dice_loss를 더해서 최종 손실 함수 total_loss를 구한다. 이렇게 결합된 손실 함수를 사용하면 모델은 클래스 불균형 문제를 해결하면서 분할 결과를 정확하게 예측할 수 있도록 학습될 수 있다.
```python
class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Focal Loss 계산
        inputs_prob = torch.sigmoid(inputs)
        focal_loss = -targets * (1 - inputs_prob) ** self.gamma * torch.log(inputs_prob + self.smooth) \
                     - (1 - targets) * inputs_prob ** self.gamma * torch.log(1 - inputs_prob + self.smooth)
        focal_loss = focal_loss.mean()

        # Dice Loss 계산
        dice_target = targets
        dice_output = inputs_prob
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Focal Loss와 Dice Loss를 더해서 총 손실을 계산
        total_loss = focal_loss + dice_loss

        return total_loss
```
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/9b19e7c6-a18f-4355-94a3-a75088213c37)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/79e37168-4100-4a69-936e-04f8cf5b1b8b)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/215a935a-e2dd-450a-9532-f6225dcea281)

### Loss function comparison
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/f2644720-fb2a-4bc8-a2f8-fddd3333298a)

## optimization algorithm
 Adam은 경사하강법(Gradient Descent) 기반의 최적화 알고리즘으로, 경사하강법의 한 변종으로서 일반적인 경사하강법보다 빠르게 수렴하는 특징이 있다.
```python
# 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
```


# :pushpin:모델 구축 및 실험관리🔥🔥
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/93af97e7-e653-4a58-8b17-bb5ede09ffc0)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/f4d83e69-27b6-465e-a113-9c17af143c25)

- Model

-smp 라이브러리에서 사용할 수 있는 거의 대부분의 모델을 실험하였으며 최종적으로 PSP를 선정하였다.

 

- Backbone

-backbone모델은 CNN계열로 실험을 해보았는데 그 결과 최종적으로 psp 모델에는 densenet , u-net모델에는 resnet50 ,deeplabV3모델에는 mobilenet을 backbone으로 사용한 모델이 가장 우수했다.

최종적으로 이렇게 세개의 모델을 ensemble한 모델과, 단일 모델으로는 PSP &densenet161 모델을 선정했다.

 

- Augmentation

-다양한 augmentation 기법들을 실험하였다. 최종적으로는 위의 augmentation기법인 2번 기법이 클래스 불균형처리에 효과적으로 작용함을 확인하고 선정하였다.

- Ensemble

-다양한 ensemble 기법을 사용하여 여러 모델을 앙상블하여 실험하였다.
다수의 모델이 예측한 결과를 투표하여 최종 예측을 결정하는 방인 Voting(투표) 앙상블,
여러 모델의 예측값을 평균하여 최종 예측을 얻는 방식인 Averaging(평균) 앙상블 기법을 사용해보고 예측값들의 평균을 구하여 노이즈를 줄이고 안정적인 예측을 얻을 수 있는 평균 앙상블 기법을 선택하였다.

```python
# DeepLabV3+ 모델 정의
model2 = smp.DeepLabV3Plus(
    encoder_name="timm-mobilenetv3_large_100",   # 백본으로 ResNet-50 사용
    encoder_weights="imagenet", # ImageNet 가중치로 초기화
    in_channels=3,             # 입력 이미지 채널 수 (RGB 이미지인 경우 3)
    classes=1                  # 출력 클래스 수 (이진 분류인 경우 1)
)
 
model1 = smp.PSPNet(encoder_name="densenet161",  # 필수 파라미터: 사용할 인코더 백본의 이름
    in_channels=3,    # 필수 파라미터: 입력 이미지의 채널 수 (일반적으로 3(RGB) 또는 1(Grayscale))
    classes=1,        # 필수 파라미터: 세그멘테이션 클래스의 수 (예: 물체 탐지의 경우 물체 클래스 수)
    encoder_weights="imagenet"  # 선택적 파라미터: 사용할 사전 훈련된 인코더 가중치의 경로 또는 'imagenet'으로 설정하여 ImageNet 가중치 사용
)
 
model3 = smp.Unet(encoder_name="resnet50",  # 필수 파라미터: 사용할 인코더 백본의 이름
    in_channels=3,    # 필수 파라미터: 입력 이미지의 채널 수 (일반적으로 3(RGB) 또는 1(Grayscale))
    classes=1,        # 필수 파라미터: 세그멘테이션 클래스의 수 (예: 물체 탐지의 경우 물체 클래스 수)
    encoder_weights="imagenet"  # 선택적 파라미터: 사용할 사전 훈련된 인코더 가중치의 경로 또는 'imagenet'으로 설정하여 ImageNet 가중치 사용
)
 
# 저장된 모델의 파라미터 불러오기 (strict=False 옵션 사용)
state_dict_1 = torch.load('./ensemble/three/psp_dense_base_trained_epoch55.pth', map_location=torch.device('cpu'))
state_dict_2 = torch.load('./ensemble/three/v3plus_mobilenet_epoch42.pth', map_location=torch.device('cpu'))
state_dict_3 = torch.load('./ensemble/three/resnet_50_unet_new_aug_noempty_trained_epoch21.pth', map_location=torch.device('cpu'))
```
![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/ab41e794-c4d0-4e7e-ae6a-501be163d3df)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/506027bb-ed7e-48a0-972a-6bc10bfc3f02)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/7118b6ba-9b24-4da9-aead-d23c1d82ec44)

![image](https://github.com/yn0212/Satellite_Image_Building_Area_Segmentation/assets/105347300/5e44a92b-3b03-4675-be7d-e87f13eb2fa8)



