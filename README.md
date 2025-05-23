# ResNet50 기반의 다중 클래스 이미지 분류 모델 구축
### ***'Fruits and Vegetables Classification Optimized with Fine-Tuned ResNet50 and Transfer Learning'***

    💡 Dataset : kaggle 'Fruits and Vegetables Image Recognition'
    💡 Using Model : ResNet50
    💡 Reason
            - Fruits & Vegetables는 항상 일상 생활에서 접하는 것들로, 생각의 접근이 쉽고 다양하다고 생각함
            - 클래스가 많고 다양해서 다중 클래스 분류 모델로 진행하면 좋겠다고 생각함
            - ResNet50과 EfficiecntNetB0를 사용하였고, EfficientNetB0가 성능이 좋은 것처럼 보여도 과적합이 많음
            - 해당 데이터셋에 대해서 kaggle에 Pytorch로 ResNet50을 구현한 코드가 존재하지 않아서 시도


<br>

## 📌 Quick View

- 초기 모델 성능 : 91.92% -> 최종 모델 성능 : 97.72% (약 6.31% 향상)

     <img width="700" alt="image (5)" src="https://github.com/user-attachments/assets/0a41e7bb-64bd-4068-a04f-098c3b4952b0" />

     - Early Stopping을 사용하여 과적합을 방지
     - 학습 도중 가장 높은 Validation 성능을 기록한 모델만 best_model.pth로 저장하여 최적의 모델을 사용한 평가만을 진행

<br>

## ⚒️ Usage Stack

분류 | 사용 기술
-- | --
언어 및 프레임 워크 | `Python`, `PyTorch`, `torchvision`
딥러닝 모델 | `ResNet50`
데이터 처리 및 시각화 | `Matplotlib`, `scikit-learn`, `SQLite`

<br>

## 👉🏻 Role & Responsibilities

- **Transfer Learning & Fine-Tuning**
     - 사전 학습된 ResNet50 모델을 활용하여 전이 학습 수행
     - 상위 FC Layer 구조를 변경하고, BatchNormalization, Dropout 등을 적용하여 모델 재설계
     - 추가적으로 일부 Conv 레이어에 Fine-Tuning을 적용하여 최적화

- **모델 최적화 및 학습 전략 설계**
     - 학습 중 과적합 방지를 위해 Early Stopping 적용
     - AdamW 옵티마이저 및 CosineDecay 학습률 스케줄러 적용
     - 검증 성능 기준으로 Best Model을 저장
 
- **데이터 전처리**
     - ImageNet을 사용하며 224x224로 이미지 크기 변경
     - batch_size와 num_workers의 값 조정 

- **결과 분석 및 시각화**
     - 모델 학습 결과를 Accuracy, Loss 그래프로 시각화 
     - Confusion Matrix 기반으로 클래스별 예측 정확도 비교
     - 다양한 실험을 통해 모델 성능을 수치적으로 분석 및 정리

<br>

## 📌 Report Info

> &nbsp;&nbsp;인공지능의 발달로 방대한 데이터가 하루에도 2.5엑사바이트씩 생성되는 데이터 시대에 오면서
> 데이터 활용헤 대한 사람들의 관심도 커지고 있다. 그중의 하나가 이미지 분류 및 인식 분야이며,
> 많은 기업에서도 이미지를 분석한 다양한 서비스를 제공하고 있다.   
> 
> &nbsp;&nbsp;본 프로젝트에서 사용한 데이터셋은 kaggle에서 자공하는
> 'Fruits and Vegetables Image Recognition'이며,
> ResNet50 모델 기반의 다중 클래스 이비지 분류 모델을 구축하는 것을 목표로 한다.
> 사전 학습된 CNN 모델을 기반으로 하여, Transfer Learning 기법을 적용하여 모델을 최적화하였다.  
>  
> &nbsp;&nbsp;모델의 일반화 성능을 향상하기 위해 FC Layer를 현재 데이터셋에 맞게 재설계하고,
> 일부 Conv Layer에 대해서 Fine-Tuning을 수행하였다.
> 또한, 과적합 문제를 완화하기 위해 Dropout, Early Stopping, Learning Rate Scheduler 등의
> 다양한 Optimization 기법을 단계적으로 적용하였다.  
>
> &nbsp;&nbsp;본 보고서에는 데이터의 전처리 과정부터 모델 설계, 학습 과정, 성능 평가,
> 그리고 실험 결과 분석에 이르기까지 전체적인 프로젝트 과정을 기술하며,
> 다양한 사전 학습 모델 및 학습 전략에 따른 성능 비교도 함께 다룬다. 

<br>

## 📁 Notion Report Page

> https://aboard-teeth-ea5.notion.site/Fruits-and-Vegetables-Classification-Optimized-with-Fine-Tuned-ResNet50-and-Transfer-Learning-1b9a0c19894d8079b9e7cad1632ea713?pvs=4

<br>
