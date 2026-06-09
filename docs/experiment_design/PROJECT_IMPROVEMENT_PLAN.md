# 프로젝트 보완 방향 정리

## 1. 보완 목적

현재 BBB 프로젝트는 MAR20 항공기 이미지 분류 문제를 대상으로 ResNet, EfficientNet, MobileNet 계열 모델을 학습하고 성능을 비교한 구조이다. AA 프로젝트는 이 중 성능이 좋은 EfficientNet-B0 모델을 대상으로 PGD 공격과 LayerCAM 기반 공격 분석을 수행한 후속 실험이다.

다만 교수님 지침 관점에서는 단순히 기존 모델을 가져와 학습한 것처럼 보일 수 있으므로, 모델을 어떻게 설계하고 활용했는지 더 명확히 보여주는 보완이 필요하다. 따라서 아래 세 가지 축으로 프로젝트를 강화한다.

1. 직접 설계한 Scratch CNN baseline 추가
2. EfficientNet-B0 + Attention + Custom Classifier Head 모델 추가
3. 기존 BBB 성능 비교와 AA 공격 분석을 하나의 분석 흐름으로 연결

## 2. 추가 실험 1: 직접 설계한 Scratch CNN Baseline

### 목적

사전학습 모델에만 의존하지 않고, 팀이 직접 설계한 CNN을 처음부터 end-to-end로 학습했다는 근거를 확보한다.

### 모델 방향

간단한 4-block CNN을 직접 설계한다.

예시 구조:

```text
Input image
-> Conv Block 1
-> Conv Block 2
-> Conv Block 3
-> Conv Block 4
-> Global Average Pooling
-> Classifier
-> Aircraft class output
```

각 Conv Block은 다음과 같이 구성할 수 있다.

```text
Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
```

### 발표에서 설명할 포인트

- ImageNet 가중치를 사용하지 않았다.
- 입력 이미지부터 최종 항공기 클래스 출력까지 전체 모델을 직접 학습했다.
- 성능이 기존 EfficientNet보다 낮더라도, 이 실험은 직접 설계한 end-to-end baseline이라는 의미가 있다.
- 이후 사전학습 기반 모델이 왜 필요한지 비교 근거로 사용할 수 있다.

## 3. 추가 실험 2: EfficientNet-B0 + Attention + Custom Classifier Head

### 목적

기존 EfficientNet-B0를 그대로 fine-tuning하는 수준에서 벗어나, 항공기 분류 목적에 맞게 모델 구조를 일부 확장했다는 근거를 만든다.

### 모델 방향

EfficientNet-B0 backbone을 사용하되, feature map 또는 feature vector 단계에 attention 모듈을 추가한다. 이후 단순 Linear classifier가 아니라 MLP 형태의 custom classifier head를 붙인다.

예시 구조:

```text
Input image
-> EfficientNet-B0 backbone
-> Attention module
-> Global feature vector
-> Linear
-> ReLU
-> Dropout
-> Linear
-> Aircraft class output
```

### Attention 후보

SE 또는 CBAM 계열 attention을 사용할 수 있다.

SE attention:

```text
Feature map
-> Global Average Pooling
-> Linear
-> ReLU
-> Linear
-> Sigmoid
-> Channel-wise reweighting
```

CBAM attention:

```text
Feature map
-> Channel Attention
-> Spatial Attention
-> Reweighted feature map
```

### Custom Classifier Head

기존 EfficientNet-B0의 classifier가 단순 Linear에 가까운 구조라면, 이를 다음과 같이 바꾼다.

```text
Linear -> ReLU -> Dropout -> Linear
```

이렇게 하면 단순히 마지막 출력 차원만 바꾼 것이 아니라, 항공기 분류에 맞는 추가 분류 계층을 설계했다고 설명할 수 있다.

### 발표에서 설명할 포인트

- EfficientNet-B0 backbone은 이미지 특징 추출기로 활용했다.
- Attention 모듈을 추가하여 항공기 객체의 중요한 채널 또는 공간 영역에 더 집중하도록 설계했다.
- Classifier head를 직접 바꾸어 단순 fine-tuning보다 더 명확한 모델 설계 요소를 추가했다.
- 기존 EfficientNet-B0 full fine-tuning 모델과 성능을 비교한다.

## 4. 유지할 기존 BBB 비교군

기존 BBB 결과는 버리지 않고 비교군으로 유지한다.

비교표에는 최소한 다음 모델을 포함한다.

```text
Scratch CNN
기존 EfficientNet-B0
EfficientNet-B0 + Attention + Custom Head
```

가능하면 기존 BBB의 ResNet, MobileNet 결과도 함께 넣어 전체 성능 위치를 보여준다.

예상 성능표 형태:

```text
Model                                   Pretrained   Custom Design   Accuracy   Macro F1
Scratch CNN                             No           Yes             ...
EfficientNet-B0 Full Fine-tuning         Yes          No              ...
EfficientNet-B0 + Attention + Head       Yes          Yes             ...
ResNet / MobileNet 비교군                Yes          No              ...
```

## 5. AA 공격 분석과 연결

최종적으로 성능이 가장 좋은 모델 또는 보완 모델을 대상으로 AA의 PGD + LayerCAM 공격 분석을 수행한다.

분석 흐름은 다음과 같이 정리한다.

```text
1. MAR20 항공기 분류 문제 정의
2. Scratch CNN으로 직접 설계 baseline 학습
3. 기존 CNN backbone 기반 모델들과 비교
4. EfficientNet-B0 + Attention + Custom Head로 모델 구조 보완
5. 최고 성능 모델 선정
6. PGD 공격으로 취약성 분석
7. LayerCAM 기반 공격/해석으로 모델이 어디에 민감한지 분석
```

이렇게 정리하면 프로젝트가 단순히 모델을 가져와 돌린 것이 아니라, 직접 설계 baseline, 구조 보완 모델, 성능 비교, 강건성 분석까지 이어지는 데이터 분석 프로젝트가 된다.

## 6. 교수님 지침에 대한 방어 논리

### Scratch CNN

질문:

```text
사전학습 없이 end-to-end로 직접 학습한 모델이 있는가?
```

답변:

```text
직접 설계한 4-block CNN을 ImageNet 가중치 없이 처음부터 학습했다. 이 모델은 입력 이미지부터 최종 항공기 클래스 출력까지 전체 파라미터를 MAR20 데이터로 학습한 end-to-end baseline이다.
```

### 기존 EfficientNet-B0

질문:

```text
사전학습 모델을 어떻게 활용했는가?
```

답변:

```text
EfficientNet-B0의 ImageNet 사전학습 가중치를 초기값으로 사용했고, 항공기 20개 클래스를 분류하도록 classifier를 교체한 뒤 전체 모델을 fine-tuning했다.
```

### Attention EfficientNet-B0

질문:

```text
단순히 기존 모델만 사용한 것인가?
```

답변:

```text
기존 EfficientNet-B0 backbone 뒤에 attention 모듈과 custom classifier head를 추가했다. 항공기 이미지에서 중요한 특징 채널 또는 공간 영역에 더 집중하도록 구조를 보완했고, 기존 EfficientNet-B0 full fine-tuning 결과와 비교했다.
```

### AA 공격 분석

질문:

```text
모델 결과를 어떻게 해석하고 분석했는가?
```

답변:

```text
최고 성능 모델에 대해 PGD 공격을 적용하여 adversarial robustness를 평가했고, LayerCAM을 이용해 공격 전후 모델이 주목하는 영역 변화를 분석했다.
```

## 7. 최종 발표 서사

최종 발표에서는 다음 메시지를 중심으로 정리한다.

```text
우리는 항공기 이미지 분류 문제를 정의하고, 직접 설계한 Scratch CNN baseline부터 사전학습 기반 모델, attention을 추가한 보완 모델까지 단계적으로 비교했다. 이후 가장 성능이 좋은 모델을 대상으로 PGD 및 LayerCAM 기반 공격 분석을 수행하여, 단순 정확도뿐 아니라 모델의 취약성과 해석 가능성까지 분석했다.
```

