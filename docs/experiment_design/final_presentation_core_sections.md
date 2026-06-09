# 최종 발표 핵심 정리

이 문서는 최종 발표에서 사용할 핵심 파트인 `모델 설계`, `실험 및 평가 방법`, `실험 결과 및 해석`에 들어가면 좋은 내용을 정리한 것이다.  
단순히 결과만 나열하기보다, **왜 이 모델들을 선택했는지**, **어떤 방식으로 평가했는지**, **결과를 어떻게 해석할 수 있는지**가 드러나도록 구성한다.

---

# 4. 모델 설계

## 4-1. 모델 설계 목적

모델 설계 파트에서는 MAR20 항공기 이미지 데이터셋을 이용해 20개 기종을 분류하는 CNN 기반 모델을 어떻게 구성했는지 설명하면 좋다.

포함하면 좋은 내용은 다음과 같다.

| 포함 내용 | 설명 |
|---|---|
| 프로젝트 목표 | MAR20 항공기 이미지 분류 모델을 구축하고 PGD 공격에 대한 강건성 비교 |
| 초기 계획 | ResNet18, ResNet34, ResNet50 중심 비교 |
| 확장된 실험 | EfficientNet-B0, MobileNetV2, MobileNetV3까지 비교 |
| 최종 비교 모델 | Baseline EfficientNet-B0, Proposed EfficientNet-B0 SE+MLP512, Scratch CNN |

최종 발표에서는 모델을 단순히 많이 실험했다고 설명하기보다, 서로 다른 목적을 가진 세 가지 모델군을 비교했다는 식으로 정리하면 좋다.

---

## 4-2. 데이터 입력 및 전처리 구조

MAR20 데이터셋은 원래 detection용 데이터셋이므로, 분류 모델에 사용하기 위해 전처리 과정이 필요했다는 점을 포함하면 좋다.

| 단계 | 처리 내용 | 목적 |
|---|---|---|
| 1 | XML annotation 확인 | 항공기 bounding box 정보 추출 |
| 2 | Horizontal Bounding Box 기준 crop | 항공기 객체 영역만 분리 |
| 3 | 정사각형 zero-padding | 종횡비 왜곡 방지 |
| 4 | 448 x 448 resize | 모델 입력 크기 통일 |
| 5 | train / validation / test 분할 | 학습 및 평가 데이터 구성 |
| 6 | rotation, flip augmentation | 다양한 기체 방향에 대한 일반화 성능 향상 |

강조하면 좋은 부분은 다음과 같다.

| 강조점 | 설명 |
|---|---|
| Bounding box crop | 배경보다 항공기 객체 자체에 집중하도록 입력 이미지 구성 |
| Zero-padding | 기체의 종횡비가 왜곡되지 않도록 유지 |
| 448 x 448 resize | 고해상도 위성 이미지의 세부 특징을 최대한 유지 |
| Data leakage 방지 | 동일 원본 이미지에서 나온 crop 이미지가 서로 다른 split에 섞이지 않도록 처리 |

---

## 4-3. 최종 비교 모델 선정 과정

최종 실험에서는 단순히 하나의 모델만 평가하지 않고, 서로 다른 목적을 가진 세 종류의 모델을 비교 대상으로 선정하였다.

| 구분 | 최종 모델 | 선정 목적 |
|---|---|---|
| Baseline 모델 | EfficientNet-B0 | 기존 pretrained CNN 중 가장 성능이 좋은 기준 모델 선정 |
| Proposed 모델 | EfficientNet-B0 SE+MLP512 | baseline 구조를 개선했을 때 성능과 강건성이 좋아지는지 확인 |
| Scratch CNN | scratch_small_maxpool_silu_pool4mlp | pretrained 없이 직접 설계한 CNN의 성능 한계 확인 |

이렇게 세 모델을 선정한 이유는 단순히 accuracy가 높은 모델만 보는 것이 아니라, **pretrained 모델**, **구조 개선 모델**, **직접 설계한 모델**을 함께 비교해 모델 구조와 학습 방식이 성능 및 공격 취약성에 어떤 영향을 주는지 확인하기 위해서이다.

---

### Baseline EfficientNet-B0가 선정된 이유

초기 실험에서는 여러 pretrained CNN 모델을 동일한 조건에서 학습하고 비교하였다.  
선정 근거를 보여주기 위해 후보 모델들의 validation 성능 수치를 함께 제시하면 좋다.

| 후보 모델 | 비교 목적 | best_val_acc_mean | best_val_acc_std | best_seed_val_acc |
|---|---|---:|---:|---:|
| EfficientNet-B0 | 효율적인 CNN 구조의 성능 확인 | 0.983123 | 0.002899 | 0.986776 |
| ResNet18 | 얕은 ResNet 구조의 성능 확인 | 0.982924 | 0.001254 | 0.985046 |
| ResNet34 | 중간 깊이 ResNet 구조의 성능 확인 | 0.981990 | 0.005608 | 0.986776 |
| MobileNetV2 | 경량 CNN 구조의 성능 확인 | 0.978715 | 0.003709 | 0.982997 |
| MobileNetV3 | 개선된 경량 CNN 구조의 성능 확인 | 0.978086 | 0.007004 | 0.986146 |
| ResNet50 | 더 깊은 ResNet 구조의 성능 확인 | 0.970781 | 0.008514 | 0.978589 |

이 중 EfficientNet-B0가 clean validation/test 성능에서 가장 좋은 결과를 보여 최종 baseline 모델로 선정되었다.

EfficientNet-B0는 모델 크기와 성능의 균형이 좋고, 항공기 이미지처럼 세부 형태와 전체 구조를 함께 봐야 하는 문제에서 안정적인 feature extraction 성능을 보였기 때문에 이후 공격 실험의 기준 모델로 사용하였다.

---

### Proposed EfficientNet-B0 SE+MLP512가 나온 이유

Baseline EfficientNet-B0가 가장 좋은 기준 모델로 선정된 뒤, 다음 단계에서는 EfficientNet-B0를 그대로 사용하는 것에서 끝나지 않고 구조를 개선했을 때 성능이 더 좋아지는지 확인하고자 하였다.

이를 위해 EfficientNet-B0를 기준으로 여러 구조 조합을 실험하였다.

| 변경 요소 | 후보 | 확인하고자 한 점 |
|---|---|---|
| Attention | none, SE, ECA | 중요한 feature channel을 강조하면 성능이 좋아지는지 확인 |
| Classifier head | linear, MLP512 | 단순 분류기보다 복잡한 분류기가 효과적인지 확인 |
| Activation | ReLU, SiLU | 비선형 함수에 따른 성능 차이 확인 |
| Fine-tuning 범위 | added_only, last2, full | 모델의 어느 범위까지 다시 학습하는 것이 좋은지 확인 |

이 실험에서 최종적으로 선택된 조합이 `grid_efficientnetb0_se_mlp512_full`이다.  
아래처럼 상위 후보들의 validation 성능을 함께 보여주면, Proposed 모델이 임의로 선택된 것이 아니라 grid 실험 결과로 선정되었다는 점이 명확해진다.

| 후보 모델 | Attention | Head | Activation | Fine-tuning | best_val_acc_mean | best_val_acc_std |
|---|---|---|---|---|---:|---:|
| grid_efficientnetb0_se_mlp512_full | SE | MLP512 | ReLU | full | 0.982997 | 0.008259 |
| grid_efficientnetb0_se_linear_full | SE | linear | ReLU | full | 0.981738 | 0.003830 |
| grid_efficientnetb0_noattn_mlp512_full | none | MLP512 | ReLU | full | 0.981318 | 0.002545 |
| grid_efficientnetb0_eca_linear_full | ECA | linear | ReLU | full | 0.980898 | 0.002212 |
| grid_efficientnetb0_eca_mlp512_silu_full | ECA | MLP512 | SiLU | full | 0.979639 | 0.004556 |
| grid_efficientnetb0_se_mlp512_silu_full | SE | MLP512 | SiLU | full | 0.979009 | 0.008824 |

이 모델은 다음과 같은 구조를 가진다.

| 구성 요소 | 적용 내용 |
|---|---|
| Backbone | EfficientNet-B0 |
| Attention | SE attention |
| Classifier head | MLP512 |
| Fine-tuning | full fine-tuning |

즉, Proposed EfficientNet-B0 SE+MLP512는 임의로 만든 모델이 아니라, EfficientNet-B0를 기준으로 attention, classifier head, activation, fine-tuning 범위를 조합해 비교한 결과 최종 선정된 개선 모델이다.

이 모델을 포함한 이유는 다음과 같다.

| 포함 이유 | 설명 |
|---|---|
| Attention 효과 확인 | SE attention을 추가하면 중요한 항공기 특징을 더 잘 볼 수 있는지 확인 |
| Classifier head 효과 확인 | MLP512 head가 linear head보다 복잡한 분류 경계를 더 잘 학습하는지 확인 |
| Fine-tuning 효과 확인 | ImageNet pretrained feature를 MAR20 항공기 데이터에 맞게 조정할 수 있는지 확인 |
| Robustness 확인 | 구조 개선이 clean accuracy뿐 아니라 PGD 공격 강건성으로도 이어지는지 확인 |

---

### Scratch CNN이 나온 이유

Scratch CNN은 pretrained 모델과 비교하기 위해 추가한 직접 설계 CNN이다.

ResNet, EfficientNet, MobileNet은 모두 ImageNet pretrained 가중치를 사용한다. 이 경우 모델은 이미 일반 이미지에서 학습한 feature extraction 능력을 가지고 시작한다.  
하지만 실제로 MAR20 항공기 데이터만으로도 충분히 좋은 분류 모델을 만들 수 있는지 확인하기 위해, pretrained 가중치를 전혀 사용하지 않는 CNN도 함께 설계하였다.

Scratch CNN 실험에서는 다음 요소들을 조합해 비교하였다.

| 변경 요소 | 후보 | 확인하고자 한 점 |
|---|---|---|
| Model size | small, base, deep | 모델 크기에 따른 성능 차이 확인 |
| Downsampling | maxpool, strided | feature map 크기를 줄이는 방식 비교 |
| Activation | ReLU, SiLU | 비선형 함수에 따른 성능 차이 확인 |
| Classifier head | GAP, pool4mlp | 마지막 분류 방식에 따른 차이 확인 |

이 중 최종적으로 선정된 모델이 `scratch_small_maxpool_silu_pool4mlp`이다.  
Scratch CNN 역시 여러 구조 후보를 비교한 뒤 validation 성능이 가장 좋은 조합을 선정했다는 점을 수치로 보여주면 좋다.

| 후보 모델 | Size | Downsampling | Activation | Head | best_val_acc_mean | best_val_acc_std |
|---|---|---|---|---|---:|---:|
| scratch_small_maxpool_silu_pool4mlp | small | maxpool | SiLU | pool4mlp | 0.842989 | 0.012441 |
| scratch_small_strided_relu_pool4mlp | small | strided | ReLU | pool4mlp | 0.837741 | 0.007244 |
| scratch_base_strided_relu_pool4mlp | base | strided | ReLU | pool4mlp | 0.834173 | 0.011093 |
| scratch_deep_maxpool_silu_pool4mlp | deep | maxpool | SiLU | pool4mlp | 0.833753 | 0.006664 |
| scratch_base_strided_silu_pool4mlp | base | strided | SiLU | pool4mlp | 0.830185 | 0.013540 |
| scratch_deep_strided_silu_pool4mlp | deep | strided | SiLU | pool4mlp | 0.828296 | 0.011341 |

| 구성 요소 | 적용 내용 |
|---|---|
| Model size | small |
| Downsampling | maxpool |
| Activation | SiLU |
| Classifier head | pool4mlp |
| Pretrained | 사용하지 않음 |

Scratch CNN을 포함한 이유는 다음과 같다.

| 포함 이유 | 설명 |
|---|---|
| 직접 설계 CNN 성능 확인 | pretrained 없이 MAR20 데이터만으로 어느 정도 성능을 낼 수 있는지 확인 |
| Pretrained 효과 비교 | ImageNet pretrained feature extractor가 항공기 분류에 얼마나 중요한지 비교 |
| 모델 구조 자체의 한계 확인 | 사전학습 없이 CNN 구조만으로 학습했을 때의 성능 한계 확인 |
| 비교군 역할 | Baseline, Proposed 모델과 비교해 pretrained 기반 모델의 장점을 보여주는 기준 |

따라서 Scratch CNN은 최종 공격 실험의 핵심 모델이라기보다, pretrained 모델의 효과를 확인하기 위한 비교군으로 사용되었다고 설명하면 좋다.

---

## 4-4. 최종 모델 선정 의미

최종적으로 세 모델은 각각 다른 질문에 답하기 위해 선정되었다.

| 모델 | 답하고자 한 질문 |
|---|---|
| Baseline EfficientNet-B0 | 기존 pretrained CNN 중 어떤 모델이 가장 안정적인 기준 모델인가? |
| Proposed EfficientNet-B0 SE+MLP512 | baseline 구조를 개선하면 성능과 robustness가 향상되는가? |
| Scratch CNN | pretrained 없이 직접 설계한 CNN만으로 항공기 분류가 가능한가? |

이 구조로 비교하면 단순히 “어떤 모델이 가장 정확한가”를 넘어서 다음 내용을 확인할 수 있다.

| 비교 관점 | 확인할 수 있는 내용 |
|---|---|
| Baseline vs Proposed | 구조 개선이 실제 성능 및 공격 강건성 개선으로 이어졌는지 확인 |
| Baseline vs Scratch | pretrained feature extractor의 효과 확인 |
| Proposed vs Scratch | 복잡한 pretrained 기반 개선 모델과 직접 설계 CNN의 차이 확인 |

---

## 4-5. 학습 설정

모델 학습 설정은 표로 정리하면 눈에 잘 들어온다.

| 항목 | 설정 |
|---|---|
| 입력 크기 | 448 x 448 |
| Optimizer | AdamW |
| Loss function | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| Early stopping | validation loss 기준 |
| Normalization | ImageNet normalization |
| Augmentation | rotation, horizontal flip, vertical flip |
| 반복 실험 | seed 1~5 |

포함하면 좋은 설명은 다음과 같다.

| 내용 | 설명 |
|---|---|
| Learning rate 조절 | validation loss가 정체되면 learning rate를 줄임 |
| Early stopping | 일정 기간 성능 개선이 없으면 학습 중단 |
| Seed 반복 | 모델 성능의 안정성을 확인하기 위해 반복 실험 수행 |
| 동일 조건 비교 | 모델 구조 차이에 따른 성능 차이를 보기 위해 가능한 동일한 학습 조건 적용 |

---

# 5. 실험 및 평가 방법

## 5-1. 전체 실험 흐름

실험 흐름은 clean 평가와 공격 평가로 나누어 설명하면 좋다.

```text
MAR20 원본 이미지
        ↓
Bounding Box crop + padding + resize
        ↓
20-class classification dataset
        ↓
모델 학습
        ↓
Clean test 성능 평가
        ↓
Standard PGD / LayerCAM Masked PGD 공격
        ↓
공격 후 accuracy 비교
        ↓
모델별 robustness 해석
```

포함하면 좋은 실험 단계는 다음과 같다.

| 단계 | 내용 | 목적 |
|---|---|---|
| 1 | Clean test 평가 | 공격 전 기본 분류 성능 확인 |
| 2 | Standard PGD 평가 | 전체 이미지 공격에 대한 취약성 확인 |
| 3 | LayerCAM Masked PGD 평가 | 중요 영역 중심 공격 결과 확인 |
| 4 | Confusion Matrix 분석 | 반복 오분류 쌍 확인 |
| 5 | LayerCAM 시각화 | 모델의 판단 근거 확인 |

---

## 5-2. Clean 성능 평가 방법

Clean 성능 평가는 공격을 적용하지 않은 원본 test set에서 진행했다는 내용을 포함하면 좋다.

| 지표 | 의미 | 사용 이유 |
|---|---|---|
| Accuracy | 전체 test 이미지 중 정답을 맞힌 비율 | 모델의 전체적인 분류 성능을 직관적으로 비교 |
| Macro Precision | 클래스별 precision을 동일 비중으로 평균 | 각 클래스에서 예측한 결과가 얼마나 정확한지 균등하게 확인 |
| Macro Recall | 클래스별 recall을 동일 비중으로 평균 | 각 클래스의 실제 샘플을 얼마나 잘 찾아냈는지 균등하게 확인 |
| Macro F1 | 클래스별 F1-score를 동일 비중으로 평균 | 클래스 불균형 영향을 줄이고 전체 클래스 성능 균형 확인 |
| Weighted Precision | 클래스별 precision을 샘플 수에 따라 가중 평균 | 실제 test set 분포를 반영한 precision 확인 |
| Weighted Recall | 클래스별 recall을 샘플 수에 따라 가중 평균 | 실제 test set 분포를 반영한 recall 확인 |
| Weighted F1 | 클래스별 F1-score를 샘플 수에 따라 가중 평균 | 실제 test set 분포 기준 전체 성능 확인 |

정량 지표와 함께 모델의 오분류 원인과 판단 근거를 확인하기 위해 다음 분석 도구도 함께 사용하면 좋다.

| 분석 도구 | 의미 | 사용 이유 |
|---|---|---|
| Confusion Matrix | 실제 클래스와 예측 클래스의 분포를 행렬로 표시 | 어떤 기종끼리 반복적으로 오분류되는지 확인 |
| LayerCAM | 모델이 판단할 때 주목한 이미지 영역을 시각화 | 오분류 원인과 공격 전후 주목 영역 변화를 해석 |

포함하면 좋은 설명은 다음과 같다.

| 내용 | 설명 |
|---|---|
| Accuracy의 역할 | 전체적인 분류 성능을 직관적으로 비교 |
| Macro 계열 지표의 역할 | 클래스별 성능을 동일 비중으로 반영해 클래스 불균형 영향을 줄임 |
| Weighted 계열 지표의 역할 | 실제 test set 분포를 반영한 전체 성능 확인 |
| Confusion Matrix의 역할 | 반복적으로 발생하는 오분류 클래스 쌍 확인 |
| LayerCAM의 역할 | 모델의 판단 근거와 공격 전후 주목 영역 변화 확인 |

Accuracy만으로는 클래스별 성능 차이를 확인하기 어렵기 때문에, Macro Precision/Recall/F1과 Weighted Precision/Recall/F1을 함께 사용해 성능을 더 균형 있게 평가했다는 점을 넣으면 좋다. 또한 Confusion Matrix와 LayerCAM을 함께 사용하면 단순한 성능 수치뿐 아니라 모델이 어떤 클래스를 왜 헷갈렸는지도 설명할 수 있다.

---

## 5-3. Standard PGD 공격 평가 방법

Standard PGD는 이미지 전체에 perturbation을 적용하는 공격이라는 점을 포함하면 좋다.

| 포함 내용 | 설명 |
|---|---|
| 공격 방식 | 이미지 전체 영역에 adversarial perturbation 적용 |
| 공격 강도 | epsilon 값으로 조절 |
| 평가 목적 | 공격 강도가 커질수록 모델 성능이 얼마나 하락하는지 확인 |
| 비교 모델 | Baseline EfficientNet-B0, Proposed EfficientNet-B0 SE+MLP512 |

사용한 epsilon 조건은 다음과 같이 정리하면 좋다.

| epsilon | 의미 |
|---|---|
| 0.125 | 약한 공격 |
| 0.25 | 중간 수준 공격 |
| 0.5 | 강한 공격 |
| 1 | 매우 강한 공격 |

Scratch CNN은 공격 결과가 없다면 다음 문장을 포함하면 좋다.

> Scratch CNN은 clean test 성능까지만 평가하였으며, 최종 PGD 공격 실험 결과는 포함하지 않았다. 따라서 adversarial robustness 비교는 두 EfficientNet 계열 모델을 중심으로 진행하였다.

---

## 5-4. LayerCAM Masked PGD 공격 평가 방법

LayerCAM Masked PGD는 LayerCAM으로 모델이 중요하게 본 영역을 찾고, 해당 영역 중심으로 공격을 적용하는 방식이라고 설명하면 좋다.

```text
입력 이미지
   ↓
모델 예측
   ↓
LayerCAM으로 중요 영역 추출
   ↓
mask 생성
   ↓
mask 영역 중심 PGD 공격
   ↓
공격 후 예측 결과 확인
```

포함하면 좋은 내용은 다음과 같다.

| 포함 내용 | 설명 |
|---|---|
| LayerCAM 사용 이유 | 모델이 판단에 사용한 영역 확인 |
| Masked PGD 목적 | 중요 영역만 공격했을 때 성능 변화 확인 |
| Standard PGD와 차이 | 전체 이미지 공격과 제한 영역 공격 비교 |
| 시각화 연결 | 공격 전후 주목 영역 변화를 함께 확인 |

이 파트에서는 LayerCAM Masked PGD가 단순한 공격 실험이 아니라, 시각화 분석과 연결된 공격 방식이라는 점을 강조하면 좋다.

---

## 5-5. 오분류 및 시각화 분석 방법

정량 지표만으로는 모델이 왜 틀렸는지 알기 어렵기 때문에, confusion matrix와 LayerCAM을 함께 활용했다는 내용을 포함하면 좋다.

| 분석 방법 | 포함 목적 |
|---|---|
| Confusion Matrix | 어떤 클래스 간 오분류가 반복되는지 확인 |
| 대표 오분류 사례 | 실제 이미지에서 왜 헷갈렸는지 설명 |
| Clean LayerCAM | 공격 전 모델이 주목한 영역 확인 |
| Attack LayerCAM | 공격 후 모델의 주목 영역 변화 확인 |
| Standard PGD vs Masked PGD 비교 | 전체 공격과 중요 영역 공격의 차이 확인 |

중간발표에서 확인된 오분류 쌍은 다음과 같이 활용할 수 있다.

| 오분류 쌍 | 포함하면 좋은 해석 방향 |
|---|---|
| A13 -> A15 | 동체 중앙부와 날개 접합부 형태 유사성 확인 |
| A5 -> A15 | 흐릿한 이미지, 세부 구조 구분 어려움 확인 |
| A19 -> A1 | 저해상도, 명암 차이, 일부 윤곽 의존 확인 |

---

# 6. 실험 결과 및 해석

## 6-1. Clean test 성능 비교

최종 결과에서는 세 모델의 clean 성능 비교가 먼저 들어가면 좋다.

| 모델 | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Baseline EfficientNet-B0 | 0.873609 | 0.876290 | 0.876701 |
| Proposed EfficientNet-B0 SE+MLP512 | 0.826964 | 0.840003 | 0.827110 |
| Scratch CNN | 0.417663 | 0.411253 | 0.400620 |

포함하면 좋은 해석 방향은 다음과 같다.

| 결과 | 해석 방향 |
|---|---|
| Baseline EfficientNet-B0가 가장 높은 성능 | 최종 기준 모델로 사용하기 적절함 |
| Proposed 모델이 baseline보다 낮음 | 구조 개선이 항상 성능 향상으로 이어지지 않음 |
| Scratch CNN이 가장 낮음 | pretrained feature extractor의 중요성을 보여줌 |

이 부분에서는 Scratch CNN을 빠뜨리지 않고 포함하는 것이 좋다. Scratch CNN은 공격 실험 결과가 없더라도, clean 성능 비교에서는 pretrained 모델과 직접 설계 모델의 차이를 보여주는 중요한 비교군이다.

---

## 6-2. Standard PGD Accuracy 결과

Standard PGD 결과는 epsilon별 accuracy 표로 넣으면 좋다.

| 모델 | eps 0.125 | eps 0.25 | eps 0.5 | eps 1 |
|---|---:|---:|---:|---:|
| Baseline EfficientNet-B0 | 0.494783 | 0.123972 | 0.001520 | 0.000000 |
| Proposed EfficientNet-B0 SE+MLP512 | 0.360514 | 0.050031 | 0.000138 | 0.000000 |

포함하면 좋은 해석 방향은 다음과 같다.

| 관찰 내용 | 해석 방향 |
|---|---|
| epsilon 증가에 따라 accuracy 급락 | Standard PGD가 강한 공격으로 작용함 |
| eps 0.5 이상에서 거의 0에 가까움 | 모델이 공격 상황에서 분류 기능을 거의 잃음 |
| Proposed 모델이 baseline보다 낮음 | 구조 개선이 Standard PGD robustness 향상으로 이어지지 않음 |

Scratch CNN은 공격 결과가 없으므로 이 표에는 넣지 않고, 공격 실험 미수행을 명시하면 좋다.

---

## 6-3. LayerCAM Masked PGD Accuracy 결과

LayerCAM Masked PGD 결과도 epsilon별 accuracy 표로 넣으면 좋다.

| 모델 | eps 0.125 | eps 0.25 | eps 0.5 | eps 1 |
|---|---:|---:|---:|---:|
| Baseline EfficientNet-B0 | 0.801119 | 0.691452 | 0.446203 | 0.140073 |
| Proposed EfficientNet-B0 SE+MLP512 | 0.743418 | 0.633405 | 0.384977 | 0.102135 |

포함하면 좋은 해석 방향은 다음과 같다.

| 관찰 내용 | 해석 방향 |
|---|---|
| epsilon 증가에 따라 accuracy 감소 | 중요 영역 공격도 성능 저하를 유발함 |
| Standard PGD보다 accuracy가 높음 | 공격 범위가 제한되어 성능 하락이 상대적으로 작음 |
| Baseline이 Proposed보다 높음 | Masked attack에서도 baseline이 더 안정적임 |

---

## 6-4. Standard PGD와 LayerCAM Masked PGD 직접 비교

두 공격 방식은 같은 epsilon에서 직접 비교하는 표가 들어가면 좋다.

### Baseline EfficientNet-B0 기준

| epsilon | Standard PGD Acc | LayerCAM Masked PGD Acc | 차이 |
|---|---:|---:|---:|
| 0.125 | 0.494783 | 0.801119 | +0.306336 |
| 0.25 | 0.123972 | 0.691452 | +0.567480 |
| 0.5 | 0.001520 | 0.446203 | +0.444683 |
| 1 | 0.000000 | 0.140073 | +0.140073 |

### Proposed EfficientNet-B0 SE+MLP512 기준

| epsilon | Standard PGD Acc | LayerCAM Masked PGD Acc | 차이 |
|---|---:|---:|---:|
| 0.125 | 0.360514 | 0.743418 | +0.382904 |
| 0.25 | 0.050031 | 0.633405 | +0.583374 |
| 0.5 | 0.000138 | 0.384977 | +0.384839 |
| 1 | 0.000000 | 0.102135 | +0.102135 |

포함하면 좋은 해석 방향은 다음과 같다.

| 비교 내용 | 해석 방향 |
|---|---|
| Standard PGD가 더 낮은 accuracy | 전체 이미지 공격이 더 강력하게 작용함 |
| LayerCAM Masked PGD가 상대적으로 높은 accuracy | 중요 영역만 공격해 공격 범위가 제한됨 |
| 두 모델 모두 같은 경향 | 공격 방식의 차이가 모델 종류보다 크게 나타남 |
| 차이값 제시 | 두 공격 방식의 강도 차이를 직관적으로 보여줌 |

이 부분은 최종 발표에서 매우 중요하다. Standard PGD와 LayerCAM Masked PGD를 따로 보여주는 것보다, 같은 epsilon에서 직접 비교해야 공격 방식의 차이가 눈에 들어온다.

---

## 6-5. 오분류 및 시각화 분석에 포함할 내용

이 부분은 단순히 숫자만 보여주는 것이 아니라, 모델이 왜 틀렸는지 보여주는 역할을 하면 좋다.

포함하면 좋은 내용은 다음과 같다.

| 포함 내용 | 설명 |
|---|---|
| Clean confusion matrix | 공격 전 반복 오분류 쌍 확인 |
| 대표 오분류 사례 | 실제 이미지와 예측 결과 제시 |
| LayerCAM 시각화 | 모델이 주목한 영역 확인 |
| 공격 전후 LayerCAM 비교 | PGD 공격 후 주목 영역 변화 확인 |
| Standard PGD vs Masked PGD 비교 | 전체 공격과 중요 영역 공격의 차이 설명 |

중간발표에서 확인된 오분류 쌍은 다음과 같이 넣을 수 있다.

| 오분류 쌍 | 포함하면 좋은 해석 방향 |
|---|---|
| A13 -> A15 | 동체 중앙부와 날개 접합부 형태 유사성 확인 |
| A5 -> A15 | 흐릿한 이미지, 세부 구조 구분 어려움 확인 |
| A19 -> A1 | 저해상도, 명암 차이, 일부 윤곽 의존 확인 |

LayerCAM 시각화 자료는 다음 구성이 좋다.

| 이미지 | 포함 목적 |
|---|---|
| 원본 이미지 | 입력 이미지 확인 |
| Clean LayerCAM | 공격 전 모델 주목 영역 확인 |
| Standard PGD 이미지 | 전체 공격 후 이미지 변화 확인 |
| Standard PGD LayerCAM | 전체 공격 후 주목 영역 변화 확인 |
| LayerCAM mask | masked attack에 사용한 영역 확인 |
| LayerCAM Masked PGD 이미지 | 중요 영역 공격 결과 확인 |
| LayerCAM Masked PGD LayerCAM | 제한 공격 후 주목 영역 변화 확인 |

슬라이드에는 모든 이미지를 한 번에 넣기보다, 대표 사례 1~2개를 선정해 비교하는 방식이 좋다.

---

## 6-6. Scratch CNN 처리

Scratch CNN은 clean 성능 비교에는 포함하는 것이 좋다.  
하지만 공격 실험 결과가 없다면 공격 표에는 넣지 않는 것이 정확하다.

포함하면 좋은 문장은 다음과 같다.

> Scratch CNN은 clean test 성능까지만 평가하였으며, 최종 PGD 공격 실험 결과는 포함하지 않았다. 따라서 Standard PGD와 LayerCAM Masked PGD 기반 adversarial robustness 비교는 Baseline EfficientNet-B0와 Proposed EfficientNet-B0 SE+MLP512를 중심으로 진행하였다.

Scratch CNN을 통해 강조하면 좋은 내용은 다음과 같다.

| 포함 내용 | 설명 |
|---|---|
| pretrained 미사용 | ImageNet 기반 feature를 사용하지 않음 |
| clean 성능 낮음 | 직접 설계 CNN의 feature 학습 한계 확인 |
| 비교군 역할 | pretrained feature extractor의 중요성 확인 |

---

## 6-7. 최종 결론에 연결할 내용

실험 결과 및 해석의 마지막에는 다음 내용들이 들어가면 좋다.

| 결론 방향 | 설명 |
|---|---|
| Baseline EfficientNet-B0가 가장 안정적 | clean 성능과 공격 후 성능 모두에서 기준 모델 역할 |
| Proposed 모델의 한계 | SE attention과 MLP512 head가 성능/robustness 향상으로 이어지지 않음 |
| Scratch CNN의 의미 | pretrained 모델의 중요성을 보여주는 비교군 |
| Standard PGD의 강력함 | 전체 이미지 공격으로 모델 성능을 크게 저하시킴 |
| LayerCAM Masked PGD의 의미 | 모델 주목 영역 기반 공격을 통해 해석 가능성 확보 |
| Clean accuracy의 한계 | 일반 정확도만으로 실제 환경 신뢰성을 판단하기 어려움 |
| Robustness 평가 필요성 | 위성/드론 기반 정찰 시스템에서는 공격 상황 평가가 필요함 |

---

# 부록. 추천 슬라이드 구성

최종 발표 자료에 넣는다면 아래 순서로 구성하면 흐름이 자연스럽다.

| 슬라이드 | 제목 | 핵심 내용 |
|---|---|---|
| 1 | 모델 설계 개요 | 데이터 전처리와 최종 비교 모델 3개 소개 |
| 2 | 최종 모델 선정 과정 | Baseline, Proposed, Scratch CNN이 왜 나왔는지 설명 |
| 3 | 학습 및 평가 설정 | 입력 크기, optimizer, scheduler, 평가 지표 정리 |
| 4 | Clean 성능 비교 | 세 모델의 Accuracy, Macro F1, Weighted F1 비교 |
| 5 | Standard PGD 결과 | epsilon별 공격 후 accuracy 하락 |
| 6 | LayerCAM Masked PGD 결과 | 중요 영역 공격 후 accuracy 변화 |
| 7 | 공격 방식 직접 비교 | Standard PGD Acc vs LayerCAM Masked PGD Acc |
| 8 | 오분류 및 LayerCAM 분석 | 대표 오분류 사례와 주목 영역 변화 |
| 9 | 최종 해석 | Baseline, Proposed, Scratch CNN의 의미와 robustness 결론 |
