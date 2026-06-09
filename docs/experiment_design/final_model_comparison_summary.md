# 최종 비교 모델 선정 및 공격 성능 정리

## 1. 최종 비교 모델 선정 과정

이번 실험에서는 최종 비교를 위해 세 종류의 대표 모델을 선정했다.  
각 모델은 서로 다른 목적을 가진 비교군이다.

| 구분 | 목적 | 후보군 | 최종 선정 모델 |
|---|---|---|---|
| Baseline 모델 | 기존 pretrained CNN 중 가장 좋은 기준 모델 찾기 | EfficientNet-B0, ResNet18, ResNet34, ResNet50, MobileNetV2, MobileNetV3 | Baseline EfficientNet-B0 |
| EfficientNet 개선 모델 | EfficientNet-B0를 구조적으로 개선했을 때 가장 좋은 조합 찾기 | Attention, classifier head, activation, fine-tuning 범위 조합 | `grid_efficientnetb0_se_mlp512_full` |
| Scratch CNN | 사전학습 없이 직접 만든 CNN 중 가장 좋은 구조 찾기 | 모델 크기, downsampling, activation, classifier head 조합 | `scratch_small_maxpool_silu_pool4mlp` |

## 2. Baseline 모델 선정

기존에 많이 사용되는 pretrained CNN 모델들을 같은 조건에서 학습하고 비교했다.

비교한 모델:

- EfficientNet-B0
- ResNet18
- ResNet34
- ResNet50
- MobileNetV2
- MobileNetV3

비교 기준:

- `best_val_acc_mean`

이 단계의 목적은 이후 실험에서 기준점으로 사용할 baseline 모델을 고르는 것이다.

## 3. EfficientNet 개선 모델 선정

EfficientNet-B0를 기준으로 여러 구조를 바꿔가며 실험했다.

| 비교 요소 | 후보 | 의미 |
|---|---|---|
| Attention | `none`, `SE`, `ECA` | 중요한 feature를 더 강조할지 비교 |
| Classifier head | `linear`, `MLP512` | 마지막 분류기를 단순하게 할지 복잡하게 할지 비교 |
| Activation | `ReLU`, `SiLU` | 어떤 비선형 함수를 사용할지 비교 |
| Fine-tuning 범위 | `added_only`, `last2`, `full` | 모델의 어느 부분까지 다시 학습할지 비교 |

최종 선정:

- `grid_efficientnetb0_se_mlp512_full`

선정 의미:

EfficientNet-B0에 `SE attention`, `MLP512 head`를 붙이고 전체를 fine-tuning한 모델이 가장 좋은 성능을 보였다.

## 4. Scratch CNN 선정

pretrained 모델을 사용하지 않고, 직접 설계한 CNN 구조들을 비교했다.

| 비교 요소 | 후보 | 의미 |
|---|---|---|
| Model size | `small`, `base`, `deep` | 모델을 얼마나 크게 만들지 비교 |
| Downsampling | `maxpool`, `strided` | feature map 크기를 줄이는 방식을 비교 |
| Activation | `ReLU`, `SiLU` | 어떤 비선형 함수를 사용할지 비교 |
| Classifier head | `GAP`, `pool4mlp` | 마지막 분류 방식을 단순하게 할지 복잡하게 할지 비교 |

최종 선정:

- `scratch_small_maxpool_silu_pool4mlp`

선정 의미:

작은 CNN 구조에 `maxpool`, `SiLU`, `pool4mlp`를 사용한 조합이 scratch CNN 중 가장 좋은 성능을 보였다.

## 5. 공격 전 성능

Test set 기준 clean 성능이다.

| 모델 | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Baseline EfficientNet-B0 | 0.873609 | 0.876290 | 0.876701 |
| Proposed EfficientNet-B0 SE+MLP512 | 0.826964 | 0.840003 | 0.827110 |
| Scratch CNN | 0.417663 | 0.411253 | 0.400620 |

## 6. 공격 후 성능: Standard PGD

Standard PGD는 이미지 전체를 대상으로 adversarial perturbation을 적용하는 공격이다.

| 모델 | eps 0.125 | eps 0.25 | eps 0.5 | eps 1 |
|---|---:|---:|---:|---:|
| Baseline EfficientNet-B0 | 0.494783 | 0.123972 | 0.001520 | 0.000000 |
| Proposed EfficientNet-B0 SE+MLP512 | 0.360514 | 0.050031 | 0.000138 | 0.000000 |

## 7. 공격 후 성능: LayerCAM Masked PGD

LayerCAM Masked PGD는 LayerCAM으로 모델이 중요하게 본 영역을 찾고, 그 영역을 중심으로 공격을 적용하는 방식이다.

| 모델 | eps 0.125 | eps 0.25 | eps 0.5 | eps 1 |
|---|---:|---:|---:|---:|
| Baseline EfficientNet-B0 | 0.801119 | 0.691452 | 0.446203 | 0.140073 |
| Proposed EfficientNet-B0 SE+MLP512 | 0.743418 | 0.633405 | 0.384977 | 0.102135 |

## 8. 결과 해석

공격 전 clean test 성능에서는 Baseline EfficientNet-B0가 가장 높았고, Proposed EfficientNet-B0 SE+MLP512가 그다음이었다. Scratch CNN은 pretrained 모델을 사용하지 않았기 때문에 성능이 가장 낮게 나타났다.

공격 후에는 Standard PGD에서 두 EfficientNet 모델 모두 성능이 크게 하락했다. 특히 epsilon이 커질수록 adversarial accuracy가 거의 0에 가까워졌다.

반면 LayerCAM Masked PGD에서는 성능 하락이 Standard PGD보다 훨씬 작았다. 이는 이미지 전체를 공격하는 Standard PGD보다, 제한된 중요 영역만 공격하는 LayerCAM 기반 공격이 perturbation 범위는 작지만 성능 저하도 상대적으로 작게 나타났다는 것을 보여준다.

Scratch CNN은 clean test 성능만 평가했고, 현재 공격 실험 결과는 존재하지 않는다.
