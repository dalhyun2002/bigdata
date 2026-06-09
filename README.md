# MAR20 항공기 분류 강건성 실험

MAR20 항공기 데이터셋을 이용해 항공기 분류 모델을 학습하고, PGD 기반 적대적 공격에 대한 강건성을 비교한 프로젝트입니다. 기본 모델, 제안 모델, scratch CNN 모델의 학습 결과와 공격 평가 결과를 함께 정리했습니다.

## 포함된 내용

- 데이터 전처리 코드
- 모델 학습 코드
- clean test 평가 코드
- Standard PGD 및 LayerCAM Masked PGD 공격 코드
- 실험 결과 CSV와 요약 파일
- 발표 및 실험 설계 문서
- 주요 모델 체크포인트 3개
  - EfficientNet-B0 baseline
  - EfficientNet-B0 + SE + MLP512 제안 모델
  - Scratch CNN 선택 모델
- MAR20 어노테이션 XML 전체
- MAR20 샘플 이미지 세트
  - `MAR20/sample/annotations/`: 샘플 어노테이션 XML
  - `MAR20/sample/originals/`: 원본 이미지 샘플
  - `MAR20/sample/crops/`: crop된 분류 이미지 샘플

## 제외된 내용

저장소 용량을 줄이기 위해 전체 원본 이미지, 전체 crop 이미지, 대량 생성 이미지, 대부분의 체크포인트는 업로드하지 않았습니다.

제외된 대표 경로는 다음과 같습니다.

- `MAR20/JPEGImages/`
- `MAR20/Classification_Dataset/`
- `attacks/**/examples/`
- `evaluation/**` 내부의 대량 이미지
- 대부분의 `.pth`, `.pt`, `.ckpt` 파일

## 프로젝트 구조

```text
MAR20/
  Annotations/                # 전체 XML 어노테이션
  ImageSets/                  # train/test split
  sample/                     # 원본, crop, annotation 샘플
attacks/                      # 적대적 공격 및 결과 비교 코드
docs/                         # 실험 설계, 실행 명령, 보고 자료
evaluation/                   # clean 평가 및 시각화 코드
results/                      # 주요 실험 로그, 요약 CSV, 선택 체크포인트
prepare_dataset.py            # 데이터셋 전처리
train_classification.py       # pretrained 모델 학습
train_experiment_stages.py    # EfficientNet 실험 단계 학습
train_scratch_cnn.py          # scratch CNN 학습
```

## 설치

```powershell
pip install -r requirements.txt
```

## 데이터 준비

전체 MAR20 원본 이미지를 로컬에 배치한 뒤 전처리를 실행합니다.

```powershell
python prepare_dataset.py
```

전처리 후 `MAR20/Classification_Dataset/`에 학습용 crop 이미지가 생성됩니다.

## 주요 실행 명령

기본 pretrained 모델 학습:

```powershell
python train_classification.py
```

EfficientNet 실험 단계 학습:

```powershell
python train_experiment_stages.py
```

Scratch CNN 학습:

```powershell
python train_scratch_cnn.py
```

선택 모델 clean 평가:

```powershell
python evaluation\evaluate_selected_clean_models.py
```

Baseline 모델 공격 평가:

```powershell
python attacks\adversarial_pgd_layercam.py --model-name baseline_efficientnetb0 --seed 4 --methods standard_pgd,layercam_masked_pgd --epsilons 0.125,0.25,0.5,1 --steps 10 --batch-size 16 --num-workers 4
```

제안 모델 공격 평가:

```powershell
python attacks\adversarial_pgd_layercam.py --model-name proposed_efficientnetb0_se_mlp512_full --seed 2 --checkpoint results\efficientnet_grid\weights\best_grid_efficientnetb0_se_mlp512_full_pretrained_seed2.pth --methods standard_pgd,layercam_masked_pgd --epsilons 0.125,0.25,0.5,1 --steps 10 --batch-size 16 --num-workers 4
```

공격 결과 요약 병합:

```powershell
python attacks\compare_attack_results.py
```

## 참고

전체 데이터셋과 대량 이미지 결과물은 GitHub 저장소에 포함하지 않았습니다. 재현이 필요한 경우 로컬에 MAR20 원본 데이터를 배치한 뒤 전처리 및 평가 스크립트를 실행하면 됩니다.
