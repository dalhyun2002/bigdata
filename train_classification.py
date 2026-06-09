import os
import csv
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torchvision.transforms import v2
except ImportError:
    v2 = None




PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "MAR20", "Classification_Dataset")
BASELINE_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "original_baseline")
WEIGHTS_DIR = os.path.join(BASELINE_RESULTS_DIR, "weights")
LOGS_DIR = os.path.join(BASELINE_RESULTS_DIR, "logs")
BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-3
EPOCHS = 1000
PATIENCE_EARLY_STOP = 10
PATIENCE_LR = 5
FACTOR_LR = 0.1
INPUT_SIZE = 448

AVAILABLE_MODELS = [
    'resnet18',
    'resnet34',
    'resnet50',
    'efficientnetb0',
    'mobilenetv2',
    'mobilenetv3',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = (device.type == "cuda")
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

base_transforms = transforms.Compose([
    transforms.ToTensor()
])




def get_model(model_name, num_classes, use_pretrained):
    resnet_weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
    resnet34_weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
    resnet50_weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
    efficientnetb0_weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
    mobilenetv2_weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if use_pretrained else None
    mobilenetv3_weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_pretrained else None

    if model_name == 'resnet18':
        model = models.resnet18(weights=resnet_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=resnet34_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=resnet50_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnetb0':
        model = models.efficientnet_b0(weights=efficientnetb0_weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=mobilenetv2_weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenetv3':
        model = models.mobilenet_v3_large(weights=mobilenetv3_weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")
    return model

def select_pretrained_from_console():
    print("\n사전학습 가중치(pretrained) 사용 여부를 선택하세요.")
    print("1) 사용")
    print("2) 사용 안 함")
    print("엔터만 누르면 기본값(사용)으로 진행합니다.\n")

    raw = input("선택: ").strip().lower()
    if raw in ("", "1", "y", "yes", "true", "pretrained"):
        return True
    if raw in ("2", "n", "no", "false", "scratch"):
        return False
    print("잘못된 입력입니다. 기본값(사용)으로 진행합니다.")
    return True

def select_models_from_console():
    print("\n학습할 모델을 선택하세요.")
    print("1) resnet18")
    print("2) resnet34")
    print("3) resnet50")
    print("4) efficientnetb0")
    print("5) mobilenetv2")
    print("6) mobilenetv3")
    print("입력 예시: 1 / resnet18 / 1,4,6 / resnet50 efficientnetb0 mobilenetv3")
    print("엔터만 누르면 전체 모델을 순차 학습합니다.\n")

    raw = input("선택: ").strip().lower()
    if raw == "":
        return AVAILABLE_MODELS

    mapping = {
        "1": "resnet18",
        "2": "resnet34",
        "3": "resnet50",
        "4": "efficientnetb0",
        "5": "mobilenetv2",
        "6": "mobilenetv3",
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "efficientnetb0": "efficientnetb0",
        "efficientnet_b0": "efficientnetb0",
        "mobilenetv2": "mobilenetv2",
        "mobilenet_v2": "mobilenetv2",
        "mobilenetv3": "mobilenetv3",
        "mobilenet_v3": "mobilenetv3",
        "mobilenet_v3_large": "mobilenetv3",
    }

    tokens = [tok for tok in raw.replace(",", " ").split() if tok]
    selected_models = []
    for tok in tokens:
        selected = mapping.get(tok)
        if selected is None:
            print(f"알 수 없는 입력 '{tok}'은 무시합니다.")
            continue
        if selected not in selected_models:
            selected_models.append(selected)

    if not selected_models:
        print("유효한 입력이 없어 전체 모델 학습으로 진행합니다.")
        return AVAILABLE_MODELS
    return selected_models

def build_gpu_transforms(device):
    if v2 is None:
        raise ImportError(
            "torchvision.transforms.v2를 찾을 수 없습니다. torchvision 버전을 업데이트하세요."
        )

    train_gpu_transforms = v2.Compose([
        v2.RandomRotation(360),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)

    val_gpu_transforms = v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)
    return train_gpu_transforms, val_gpu_transforms

def create_metrics_logger(model_name, use_pretrained, seed):
    os.makedirs(LOGS_DIR, exist_ok=True)
    pretrained_tag = "pretrained" if use_pretrained else "scratch"
    log_path = os.path.join(LOGS_DIR, f"metrics_{model_name}_{pretrained_tag}_seed{seed}.csv")
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "epoch_time_sec",
        "is_best",
        "early_stop_counter",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return log_path

def append_metrics_row(log_path, row):
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_dataloaders(train_dataset, val_dataset, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def validate_dataset_dirs():
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(
            f"Dataset folders not found. Run prepare_dataset.py first.\nExpected: {TRAIN_DIR} and {VAL_DIR}"
        )


if __name__ == '__main__':

    validate_dataset_dirs()
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=base_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=base_transforms)

    NUM_CLASSES = len(train_dataset.classes)
    if set(train_dataset.classes) != set(val_dataset.classes):
        raise ValueError("Train/val class mismatch detected. Re-check dataset preparation.")

    use_pretrained = select_pretrained_from_console()
    selected_models = select_models_from_console()
    train_gpu_tf, val_gpu_tf = build_gpu_transforms(device)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"증강 처리 디바이스: {device}")
    print(f"AMP 사용: {use_amp}")
    print(f"Pretrained 사용: {use_pretrained}")
    print(f"선택된 모델: {', '.join(selected_models)}")
    run_seeds = [1, 2, 3, 4, 5] if use_pretrained else [1]
    print(f"실행 시드: {run_seeds}")

    for model_name in selected_models:
        for seed in run_seeds:
            print("\n" + "=" * 70)
            print(f"Start Training: {model_name} (seed={seed})")
            print("=" * 70)
            set_seed(seed)
            train_loader, val_loader = make_dataloaders(train_dataset, val_dataset, seed)

            model = get_model(model_name, NUM_CLASSES, use_pretrained).to(device)
            log_path = create_metrics_logger(model_name, use_pretrained, seed)
            print(f"에폭 로그 저장 경로: {log_path}")


            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=LR)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=FACTOR_LR, patience=PATIENCE_LR
            )




            best_val_loss = float('inf')
            early_stop_counter = 0

            for epoch in range(1, EPOCHS + 1):
                epoch_start = time.time()

                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                for inputs, labels in tqdm(train_loader, desc=f"{model_name}/s{seed} Epoch {epoch}/{EPOCHS} [Train]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = train_gpu_tf(inputs)
                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    train_correct += torch.sum(preds == labels.data)
                    train_total += inputs.size(0)

                train_loss = train_loss / train_total
                train_acc = train_correct.double() / train_total


                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in tqdm(val_loader, desc=f"{model_name}/s{seed} Epoch {epoch}/{EPOCHS} [Val]"):
                        inputs, labels = inputs.to(device), labels.to(device)
                        inputs = val_gpu_tf(inputs)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        val_correct += torch.sum(preds == labels.data)
                        val_total += inputs.size(0)

                val_loss = val_loss / val_total
                val_acc = val_correct.double() / val_total

                print(
                    f"\n[{model_name}/seed{seed} Epoch {epoch}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )


                scheduler.step(val_loss)


                is_best = 0
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    is_best = 1
                    save_path = os.path.join(WEIGHTS_DIR, f"best_{model_name}_{'pretrained' if use_pretrained else 'scratch'}_seed{seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"=> Saved Best Model at {save_path}!")
                else:
                    early_stop_counter += 1
                    print(f"=> Early Stopping Counter: {early_stop_counter}/{PATIENCE_EARLY_STOP}")

                epoch_time_sec = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]["lr"]
                append_metrics_row(
                    log_path,
                    {
                        "epoch": epoch,
                        "train_loss": f"{train_loss:.8f}",
                        "train_acc": f"{train_acc.item():.8f}",
                        "val_loss": f"{val_loss:.8f}",
                        "val_acc": f"{val_acc.item():.8f}",
                        "lr": f"{current_lr:.10f}",
                        "epoch_time_sec": f"{epoch_time_sec:.4f}",
                        "is_best": is_best,
                        "early_stop_counter": early_stop_counter,
                    },
                )

                if early_stop_counter >= PATIENCE_EARLY_STOP:
                    print(f"Early stopping triggered at epoch {epoch} ({model_name}, seed={seed})")
                    break

            print(f"Finished: {model_name} (seed={seed})")

    print("Training finished for all selected models.")
