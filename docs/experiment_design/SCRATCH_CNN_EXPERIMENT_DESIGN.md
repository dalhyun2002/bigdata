# Scratch CNN Baseline 실험 설계

## 1. 실험 목적

Scratch CNN 실험의 목적은 사전학습 모델을 사용하지 않고, 직접 설계한 CNN을 MAR20 항공기 crop 이미지로 처음부터 end-to-end 학습하는 것이다.

이 실험은 기존 EfficientNet-B0 pretrained baseline과 구분되는 직접 설계 baseline 역할을 한다.

```text
Scratch CNN:
ImageNet pretrained weight 없음
직접 설계한 CNN 구조
MAR20 데이터로 처음부터 학습
end-to-end classification
```

최종 비교에서는 다음 구도를 만든다.

```text
Scratch CNN baseline
vs
EfficientNet-B0 pretrained baseline
vs
EfficientNet-B0 + Attention + MLP 보완 모델
```

## 2. 기본 고정 조건

Scratch CNN 실험에서 다음 항목은 고정한다.

```text
Input size: 448 x 448
Conv kernel: 3x3
Padding: 1
BatchNorm: 사용
Optimizer: AdamW
Learning rate: 1e-3
Early stopping: validation loss 기준 patience 10
Seeds: 1, 2, 3
Pretrained: 사용 안 함
Training: 전체 파라미터 end-to-end scratch 학습
```

고정 이유:

```text
기존 BBB 학습 조건과 최대한 맞추고, Scratch CNN 내부에서는 핵심 구조 요소만 비교하기 위해서이다.
```

## 3. 비교할 설계 요소

Scratch CNN에서는 모든 조합을 한 번에 탐색하지 않고, 다음 요소를 단계적으로 비교한다.

```text
1단계: 모델 크기
2단계: Downsampling 방식
3단계: Activation 함수
4단계: Classifier head
```

전체 조합을 모두 돌리면 실험 수가 지나치게 많아진다.

```text
3 sizes x 2 downsampling x 2 activation x 2 heads x 3 seeds = 72 runs
```

따라서 각 단계에서 하나의 설계 요소만 바꾸고, 선택된 best 설정을 다음 단계로 넘긴다.

## 4. 1단계: 모델 크기 비교

### 목적

직접 설계 CNN에서 모델 용량이 성능에 어떤 영향을 주는지 확인한다.

### 고정 조건

```text
Downsampling: MaxPool
Activation: ReLU
Head: GAP
BatchNorm: 사용
```

### 비교 모델

```text
ScratchCNN-Small
4 blocks
channels: 16-32-64-128

ScratchCNN-Base
4 blocks
channels: 32-64-128-256

ScratchCNN-Deep
5 blocks
channels: 32-64-128-256-512
```

Small을 3-block으로 두지 않고 4-block으로 둔 이유:

```text
이번 목적은 단순히 약한 baseline을 만드는 것이 아니라, scratch CNN 중에서도 가능한 좋은 성능을 찾는 것이다. 3-block 모델은 너무 약할 가능성이 있어, Small도 최소 4-block으로 구성한다.
```

## 5. 2단계: Downsampling 방식 비교

### 목적

feature map 크기를 줄이는 방식을 비교한다.

### 비교 조건

1단계에서 선택된 size를 사용한다.

고정:

```text
Activation: ReLU
Head: GAP
BatchNorm: 사용
```

비교:

```text
MaxPool
StridedConv
```

### MaxPool

```text
Conv2d(stride=1, padding=1)
-> BatchNorm
-> ReLU
-> MaxPool2d(2)
```

MaxPool은 2x2 영역에서 가장 큰 반응만 남기는 고정 규칙 기반 downsampling이다.

### Strided Conv

```text
Conv2d(stride=2, padding=1)
-> BatchNorm
-> ReLU
```

Strided Conv는 convolution filter가 학습되므로, downsampling 과정 자체도 데이터에 맞게 학습될 수 있다.

비교 질문:

```text
고정 규칙으로 가장 강한 반응만 남기는 MaxPool이 좋은가,
아니면 학습 가능한 StridedConv downsampling이 더 좋은가?
```

## 6. 3단계: Activation 비교

### 목적

선택된 Scratch CNN 구조에서 ReLU와 SiLU 중 어떤 activation이 더 적합한지 확인한다.

### 비교 조건

1, 2단계에서 선택된 size와 downsampling 방식을 사용한다.

고정:

```text
Head: GAP
BatchNorm: 사용
```

비교:

```text
ReLU
SiLU
```

### ReLU

```text
ReLU(x) = max(0, x)
```

음수 입력을 0으로 자르고 양수 입력은 그대로 통과시키는 기본 activation이다.

### SiLU

```text
SiLU(x) = x * sigmoid(x)
```

ReLU보다 부드럽고, 음수 신호도 일부 유지한다. EfficientNet 계열 내부에서도 사용되는 activation이다.

비교 질문:

```text
단순하고 표준적인 ReLU가 좋은가,
아니면 smooth activation인 SiLU가 scratch CNN에도 도움이 되는가?
```

## 7. 4단계: Classifier Head 비교

### 목적

마지막 feature map을 최종 classifier에 전달하는 방식을 비교한다.

### 비교 조건

1, 2, 3단계에서 선택된 size, downsampling, activation을 사용한다.

비교:

```text
GAP Head
Pool4 + Flatten + MLP Head
```

### GAP Head

```text
Feature map
-> Global Average Pooling
-> Dropout(0.3)
-> Linear(last_channel -> 20)
```

예를 들어 Base 모델의 마지막 feature map이 `256 x 28 x 28`이라면:

```text
256 x 28 x 28
-> 256
-> Linear(256 -> 20)
```

장점:

```text
파라미터 수가 적음
과적합 위험이 낮음
계산이 빠름
```

단점:

```text
공간 위치 정보가 많이 사라짐
```

### Pool4 + Flatten + MLP Head

```text
Feature map
-> AdaptiveAvgPool2d(4)
-> Flatten
-> Linear(input_dim -> 512)
-> Selected Activation
-> Dropout(0.3)
-> Linear(512 -> 20)
```

예를 들어 Base 모델의 마지막 feature map이 `256 x 28 x 28`이라면:

```text
256 x 28 x 28
-> 256 x 4 x 4
-> 4096
-> Linear(4096 -> 512)
-> Activation
-> Dropout(0.3)
-> Linear(512 -> 20)
```

장점:

```text
4x4 수준의 대략적인 공간 배치 정보를 일부 유지함
항공기 부품의 상대적 위치 정보가 도움이 되는지 확인 가능
```

단점:

```text
GAP보다 파라미터 수가 많음
과적합 위험이 증가할 수 있음
```

비교 질문:

```text
항공기 crop 분류에서 위치 정보를 거의 제거하는 GAP가 더 안정적인가,
아니면 4x4 공간 배치 정보를 일부 남기는 Pool4+MLP가 더 좋은가?
```

## 8. 실행 방식

Scratch CNN 학습 스크립트는 두 가지 모드를 지원하도록 만든다.

```text
1. staged ablation
2. full grid search
```

### 모드 1: staged ablation

단계적으로 하나의 설계 요소만 비교하고, best 설정을 다음 단계로 넘기는 방식이다.

장점:

```text
실험 수가 적음
각 단계의 질문이 명확함
결과 해석이 쉬움
```

단점:

```text
초기 단계에서 선택되지 않은 설정이 다른 조합에서는 더 좋을 가능성을 놓칠 수 있음
```

### 모드 2: full grid search

모든 조합을 전부 학습하고 가장 좋은 조합을 선택하는 방식이다.

조합:

```text
Size: small, base, deep
Downsampling: maxpool, strided
Activation: relu, silu
Head: gap, pool4mlp
Seeds: 1, 2, 3
```

총 실험 수:

```text
3 x 2 x 2 x 2 x 3 = 72 runs
```

장점:

```text
전체 후보군에서 가장 좋은 조합을 직접 찾을 수 있음
설계 요소 간 상호작용을 놓칠 가능성이 낮음
```

단점:

```text
실험 시간이 길어짐
결과표가 커짐
```

추천 사용:

```text
먼저 1 epoch 속도 테스트를 수행한 뒤, 시간이 감당 가능하면 grid search를 사용한다.
시간이 부족하면 staged ablation을 사용한다.
```

Scratch CNN은 EfficientNet보다 가벼울 가능성이 높으므로, stage별 실행뿐 아니라 전체 자동 실행도 가능하게 만든다.

추천 실행 방식:

```powershell
python train_scratch_cnn.py --mode staged --stage all --seeds "1 2 3" --batch-size 32
```

`stage all`은 다음 흐름으로 동작하도록 설계한다.

```text
1. Small / Base / Deep 학습
2. best size 선택
3. 선택된 size에서 MaxPool / StridedConv 학습
4. best downsampling 선택
5. 선택된 size + downsampling에서 ReLU / SiLU 학습
6. best activation 선택
7. 선택된 size + downsampling + activation에서 GAP / Pool4MLP 학습
8. 최종 Scratch CNN 선택
```

stage별 실행도 가능하게 만든다.

```powershell
python train_scratch_cnn.py --mode staged --stage 1 --seeds "1 2 3" --batch-size 32
python train_scratch_cnn.py --mode staged --stage 2 --selected-size base --seeds "1 2 3" --batch-size 32
python train_scratch_cnn.py --mode staged --stage 3 --selected-size base --selected-downsampling maxpool --seeds "1 2 3" --batch-size 32
python train_scratch_cnn.py --mode staged --stage 4 --selected-size base --selected-downsampling maxpool --selected-activation relu --seeds "1 2 3" --batch-size 32
```

Grid search 실행:

```powershell
python train_scratch_cnn.py --mode grid --seeds "1 2 3" --batch-size 32
```

1 epoch 속도 테스트:

```powershell
python train_scratch_cnn.py --mode grid --seeds "1" --batch-size 32 --epochs 1 --limit-runs 1
```

## 9. 저장 위치

EfficientNet 보완 실험과 섞이지 않도록 Scratch CNN 결과는 별도 폴더에 저장한다.

```text
Scratch_results/
├─ weights/
├─ logs/
└─ summaries/
```

저장 파일 예시:

```text
Scratch_results/weights/best_scratch_base_maxpool_relu_gap_scratch_seed1.pth
Scratch_results/logs/metrics_scratch_base_maxpool_relu_gap_scratch_seed1.csv
Scratch_results/summaries/stage1_size_summary.csv
Scratch_results/summaries/stage2_downsampling_summary.csv
Scratch_results/summaries/stage3_activation_summary.csv
Scratch_results/summaries/stage4_head_summary.csv
Scratch_results/summaries/final_scratch_selection.csv
```

## 10. 선택 기준

각 단계에서는 seed 1, 2, 3의 평균과 표준편차를 기준으로 best 설정을 선택한다.

우선순위:

```text
1. best_val_acc_mean
2. best_val_acc_std
3. train-val gap
4. test accuracy / macro F1
5. 모델 복잡도 대비 성능
```

최종 checkpoint는 선택된 Scratch CNN 구조 안에서 성능이 가장 좋은 seed의 checkpoint를 사용한다.

```text
구조 선택: seed 평균 기준
최종 checkpoint 선택: best seed 기준
```

## 11. 교수님 질문 대응

질문:

```text
사전학습 없이 직접 설계한 모델이 있나요?
```

답변:

```text
네. Scratch CNN은 ImageNet pretrained weight를 사용하지 않고, Conv-BN-Activation-Downsampling block을 직접 구성하여 MAR20 항공기 crop 이미지로 처음부터 end-to-end 학습한 모델입니다.
```

질문:

```text
Scratch CNN 구조는 임의로 하나만 정했나요?
```

답변:

```text
단일 구조를 임의로 선택하지 않고, 모델 크기, downsampling 방식, activation, classifier head를 단계적으로 비교했습니다. 각 단계에서는 하나의 설계 요소만 변경하여 결과 해석이 가능하도록 구성했습니다.
```

질문:

```text
왜 전체 조합을 모두 비교하지 않았나요?
```

답변:

```text
전체 조합을 모두 탐색하면 실험 수가 72개 수준으로 증가하여 비효율적이고, 결과 해석도 복잡해집니다. 따라서 단계적으로 best 후보를 선택하는 staged ablation 방식으로 실험을 설계했습니다.
```

## 12. 발표용 요약 문장

```text
Scratch CNN은 사전학습 모델 없이 직접 설계한 end-to-end baseline으로 구성했습니다. 단일 구조를 임의로 선택하지 않고, 모델 크기, downsampling 방식, activation 함수, classifier head를 단계적으로 비교하여 최종 Scratch CNN 구조를 선정했습니다. 이를 통해 pretrained EfficientNet-B0와 직접 설계 CNN 사이의 성능 차이를 비교하고, 사전학습의 효과를 정량적으로 확인할 수 있도록 했습니다.
```
