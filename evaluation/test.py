import os
import csv
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torchvision.transforms import v2
except ImportError:
    v2 = None

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "MAR20", "Classification_Dataset")
BASELINE_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "original_baseline")
WEIGHTS_DIR = os.path.join(BASELINE_RESULTS_DIR, "weights")
LOGS_DIR = os.path.join(BASELINE_RESULTS_DIR, "logs")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 64
NUM_WORKERS = 8
AVAILABLE_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnetb0",
    "mobilenetv2",
    "mobilenetv3",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name, num_classes, use_pretrained):
    resnet_weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
    resnet34_weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
    resnet50_weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
    efficientnetb0_weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
    mobilenetv2_weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if use_pretrained else None
    mobilenetv3_weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=resnet_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=resnet34_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=resnet50_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=efficientnetb0_weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=mobilenetv2_weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=mobilenetv3_weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")
    return model


def select_pretrained_from_console():
    print("\n평가할 가중치 타입을 선택하세요.")
    print("1) pretrained 학습 결과")
    print("2) scratch 학습 결과")
    print("엔터만 누르면 기본값(pretrained)으로 진행합니다.\n")
    raw = input("선택: ").strip().lower()
    if raw in ("", "1", "pretrained"):
        return True
    if raw in ("2", "scratch"):
        return False
    print("잘못된 입력입니다. pretrained로 진행합니다.")
    return True


def select_model_from_console():
    print("\n평가할 모델을 선택하세요.")
    print("1) resnet18")
    print("2) resnet34")
    print("3) resnet50")
    print("4) efficientnetb0")
    print("5) mobilenetv2")
    print("6) mobilenetv3")
    raw = input("선택: ").strip().lower()
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
    selected = mapping.get(raw)
    if selected is None:
        raise ValueError("지원하지 않는 모델 선택입니다.")
    return selected


def select_seed_from_console(default_seed=1):
    raw = input(f"\n평가할 seed를 입력하세요 (엔터={default_seed}): ").strip()
    if raw == "":
        return default_seed
    return int(raw)


def default_weight_path(model_name, use_pretrained, seed):
    pretrained_tag = "pretrained" if use_pretrained else "scratch"
    return os.path.join(
        WEIGHTS_DIR,
        f"best_{model_name}_{pretrained_tag}_seed{seed}.pth",
    )


def build_test_transform(device):
    if v2 is None:
        raise ImportError("torchvision.transforms.v2를 찾을 수 없습니다.")
    return v2.Compose(
        [
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ).to(device)


def append_test_result(row):
    os.makedirs(LOGS_DIR, exist_ok=True)
    out_path = os.path.join(LOGS_DIR, "test_results.csv")
    file_exists = os.path.exists(out_path)
    fieldnames = list(row.keys())
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"테스트 결과 저장: {out_path}")


def validate_dataset_dirs():
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            "Dataset folders not found. Run prepare_dataset.py first.\n"
            f"Expected: {TRAIN_DIR} and {TEST_DIR}"
        )


if __name__ == "__main__":
    validate_dataset_dirs()

    model_name = select_model_from_console()
    use_pretrained = select_pretrained_from_console()
    seed = select_seed_from_console(default_seed=1)
    ckpt_path = default_weight_path(model_name, use_pretrained, seed)

    custom_path = input(f"\n체크포인트 경로 (엔터=기본값)\n{ckpt_path}\n입력: ").strip()
    if custom_path:
        ckpt_path = custom_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    base_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=base_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=base_transform)

    if set(train_dataset.classes) != set(test_dataset.classes):
        raise ValueError("Train/test class mismatch detected. Re-check dataset preparation.")

    num_classes = len(train_dataset.classes)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_tf = build_test_transform(device)

    model = get_model(model_name, num_classes, use_pretrained).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = test_tf(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += inputs.size(0)

    test_loss /= test_total
    test_acc = test_correct / test_total

    print("\n" + "=" * 70)
    print(f"Model: {model_name}")
    print(f"Pretrained: {use_pretrained}")
    print(f"Seed: {seed}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Acc: {test_acc:.6f}")
    print("=" * 70)

    append_test_result(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": model_name,
            "use_pretrained": use_pretrained,
            "seed": seed,
            "checkpoint": ckpt_path,
            "test_loss": f"{test_loss:.8f}",
            "test_acc": f"{test_acc:.8f}",
            "num_samples": test_total,
        }
    )
