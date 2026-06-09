# MAR20 Aircraft Project 실행 순서

이 폴더는 기존 `BBB` 모델 학습/평가 프로젝트와 `AA` 적대적 공격 실험을 하나로 합친 통합 프로젝트입니다.

## 1. 의존성 설치

```powershell
python -m pip install -r requirements.txt
```

## 2. 데이터셋 생성

`MAR20` 원본 이미지와 XML annotation에서 항공기 crop 분류 데이터셋을 생성합니다.

```powershell
python prepare_dataset.py
```

## 3. 모델 학습

ResNet, EfficientNet, MobileNet 계열 모델을 학습하고 `weights`와 `logs`에 결과를 저장합니다.

```powershell
python train_classification.py
```

## 4. 모델 평가

학습된 모델들의 test 성능과 confusion matrix를 `Evaluate_models`에 저장합니다.

```powershell
python evaluate_models.py
```

## 5. LayerCAM 예시 생성

분류 모델이 주목한 영역을 LayerCAM으로 시각화합니다.

```powershell
python generate_layercam_examples.py
```

## 6. PGD / LayerCAM-masked PGD 공격 실험

기본값은 `efficientnetb0_pretrained_seed4` 모델에 대해 `standard_pgd`와 `layercam_masked_pgd`를 실행합니다.

```powershell
python adversarial_pgd_layercam.py
```

빠른 확인용 예시:

```powershell
python adversarial_pgd_layercam.py --methods clean --epsilons 0.125 --max-samples 2 --batch-size 1 --num-workers 0 --examples-per-class 0
```

공격 결과는 기본적으로 아래 경로에 저장됩니다.

```text
Adversarial_Attacks/
```

## 7. ECA / SE / MLP 보완 실험

보완 실험은 기존 학습 코드와 분리된 `train_experiment_stages.py`로 실행합니다. 결과는 `Experiment_results` 아래에 저장됩니다.

```text
Experiment_results/
├─ weights/
├─ logs/
└─ summaries/
```

먼저 실행 계획만 확인하려면 `--dry-run`을 사용합니다.

```powershell
python train_experiment_stages.py --stage 1 --dry-run
python train_experiment_stages.py --stage 2 --selected-attention eca --dry-run
python train_experiment_stages.py --stage 3 --selected-attention eca --selected-activation silu --dry-run
python train_experiment_stages.py --stage 4 --selected-attention eca --selected-head mlp512 --selected-activation silu --dry-run
```

Stage 1? SE? ECA attention? ?????.

```powershell
python train_experiment_stages.py --stage 1 --seeds "1 2 3" --batch-size 8 --grad-accum-steps 8 --resume
```

RunPod RTX 4090??? ???? ?????.

```bash
python train_experiment_stages.py --stage 1 --seeds "1 2 3" --batch-size 64 --resume
```

Stage 2? Stage 1?? ??? attention? ???? ReLU? SiLU activation? ?????.

```bash
python train_experiment_stages.py --stage 2 --selected-attention eca --seeds "1 2 3" --batch-size 64 --resume
```

Stage 3? ??? attention? activation? ???? head? attention ??? ablation???.

```bash
python train_experiment_stages.py --stage 3 --selected-attention eca --selected-activation silu --seeds "1 2 3" --batch-size 64 --resume
```

ReLU? ?????? `--selected-activation relu`? ????? ???? ???.

Stage 4? ??? ?? ???? ?? ??? ?????.

```bash
python train_experiment_stages.py --stage 4 --selected-attention eca --selected-head mlp512 --selected-activation silu --seeds "1 2 3" --batch-size 64 --resume
```

?? GTX 1050 Ti 4GB?? GPU ???? ???? batch size? ??? gradient accumulation? ?????.

```powershell
python train_experiment_stages.py --stage 3 --selected-attention eca --selected-activation silu --seeds "1 2 3" --batch-size 8 --grad-accum-steps 8 --resume
```

?? stage? ? ?? ??? ?? ????. ?? ??? ?? ??? ??? ??? ???? ??? ??? ??? ??? ?????.

```bash
python train_experiment_stages.py --stage all --selected-attention eca --selected-head mlp512 --selected-activation silu --seeds "1 2 3" --batch-size 64 --resume
```

?? ?? grid search? ??? ? ????. Linear head?? activation? ???? ??? ??? 27? config? ???, seed 3? ?? ? 81 runs? ?????.

```bash
python train_experiment_stages.py --stage grid --seeds "1 2 3" --batch-size 64 --resume
```

? run? epoch? ?? ??? `Experiment_results/logs/metrics_*.csv`? ????, ? stage? ?? ??? `Experiment_results/summaries/stage*_run_summary.csv`? `stage*_aggregate_summary.csv`? ?????. Early stopping? validation loss ?? patience 10?? ?????.

## 8. Scratch CNN Baseline 실험

Scratch CNN은 사전학습 없이 직접 설계한 CNN을 end-to-end로 학습하는 baseline입니다. 결과는 `Scratch_results` 아래에 저장됩니다.

```text
Scratch_results/
├─ weights/
├─ logs/
└─ summaries/
```

먼저 1개 run의 계획만 확인합니다.

```powershell
python train_scratch_cnn.py --mode grid --seeds "1" --batch-size 32 --epochs 1 --limit-runs 1 --dry-run
```

1 epoch 속도 테스트:

```powershell
python train_scratch_cnn.py --mode grid --seeds "1" --batch-size 32 --epochs 1 --limit-runs 1
```

전체 조합 grid search:

```powershell
python train_scratch_cnn.py --mode grid --seeds "1 2 3" --batch-size 32
```

단계적 ablation 전체 자동 실행:

```powershell
python train_scratch_cnn.py --mode staged --stage all --seeds "1 2 3" --batch-size 32
```

단계별 실행:

```powershell
python train_scratch_cnn.py --mode staged --stage 1 --seeds "1 2 3" --batch-size 32
python train_scratch_cnn.py --mode staged --stage 2 --selected-size base --seeds "1 2 3" --batch-size 32
python train_scratch_cnn.py --mode staged --stage 3 --selected-size base --selected-downsampling maxpool --seeds "1 2 3" --batch-size 32
python train_scratch_cnn.py --mode staged --stage 4 --selected-size base --selected-downsampling maxpool --selected-activation relu --seeds "1 2 3" --batch-size 32
```

OOM이 나면 batch size를 낮춥니다.

```powershell
python train_scratch_cnn.py --mode grid --seeds "1 2 3" --batch-size 16
```
