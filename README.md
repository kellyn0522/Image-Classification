# ResNet50 기반의 다중 클래스 이미지 분류 모델 구축
### ***'Fruits and Vegetables Classification Optimized with Fine-Tuned ResNet50 and Transfer Learning'***

    💡 Dataset : kaggle 'Fruits and Vegetables Image Recognition'
    💡 Using Model : ResNet50
    💡 Reason
            - Fruis & Vegetables는 항상 일상 생활에서 접하는 것들로, 생각의 접근이 쉽고 다양하다고 생각함
            - 클래스가 많고 다양해서 다중 클래스 분류 모델로 진행하면 좋겠다고 생각함
            - ResNet50과 EfficiecntNetB0를 사용하였고, EfficientNetB0가 성능이 좋은 것처럼 보여도 과적합이 많음
            - 해당 데이터셋에 대해서 kaggle에 Pytorch로 ResNet50을 구현한 코드가 존재하지 않아서 시도


---

### Introduction

>  인공지능의 발달로 방대한 데이터가 하루에도 2.5엑사바이트씩 생성되는 데이터 시대에 오면서 데이터 활용헤 대한 사람들의 관심도 커지고 있다. 그중의 하나가 이미지 분류 및 인식 분야이며, 많은 기업에서도 이미지를 분석한 다양한 서비스를 제공하고 있다.   
>
&nbsp;
>
  본 프로젝트에서 사용한 데이터셋은 kaggle에서 자공하는 'Fruits and Vegetables Image Recognition'이며, ResNet50 모델 기반의 다중 클래스 이비지 분류 모델을 구축하는 것을 목표로 한다. 사전 학습된 CNN 모델을 기반으로 하여, Transfer Learning 기법을 적용하여 모델을 최적화하였다.  
>  
  모델의 일반화 성능을 향상하기 위해 FC Layer를 현재 데이터셋에 맞게 재설계하고, 일부 Conv Layer에 대해서 Fine-Tuning을 수행하였다. 또한, 과적합 문제를 완화하기 위해 Dropout, Early Stopping, Learning Rate Scheduler 등의 다양한 Optimization 기법을 단계적으로 적용하였다.  
  
  본 보고서에는 데이터의 전처리 과정부터 모델 설계, 학습 과정, 성능 평가, 그리고 실험 결과 분석에 이르기까지 전체적인 프로젝트 과정을 기술하며, 다양한 사전 학습 모델 및 학습 전략에 따른 성능 비교도 함께 다룬다. 

---

## 📌 Introduction

> 인공지능의 발달로 방대한 데이터가 하루에도 **2.5엑사바이트**씩 생성되는 데이터 시대에 오면서  
> 데이터 활용에 대한 사람들의 관심도 커지고 있습니다.  
> 그중 하나가 **이미지 분류 및 인식 분야**이며,  
> 많은 기업에서도 이미지를 분석한 다양한 서비스를 제공하고 있습니다.

&nbsp;

> 본 프로젝트에서는 Kaggle에서 제공하는  
> **Fruits and Vegetables Image Recognition** 데이터셋을 활용하여,  
> **ResNet50 모델 기반의 다중 클래스 이미지 분류 모델**을 구축하는 것을 목표로 합니다.  
> 사전 학습된 CNN 모델을 바탕으로 **Transfer Learning** 기법을 적용하여 모델을 최적화하였습니다.

&nbsp;

> 모델의 일반화 성능을 향상시키기 위해  
> FC Layer를 현재 데이터셋에 맞게 **재설계**하고,  
> 일부 Conv Layer에 대해서는 **Fine-Tuning**을 수행하였습니다.  
> 또한, 과적합 문제를 완화하기 위해  
> **Dropout, Early Stopping, Learning Rate Scheduler** 등의  
> 다양한 Optimization 기법을 단계적으로 적용하였습니다.

&nbsp;

> 본 보고서에서는 데이터 전처리 과정부터  
> 모델 설계, 학습 과정, 성능 평가, 실험 결과 분석에 이르기까지  
> 전체적인 프로젝트 과정을 상세히 기술합니다.  
> 더불어, **다양한 사전 학습 모델 및 학습 전략에 따른 성능 비교**도 함께 다루고 있습니다.



