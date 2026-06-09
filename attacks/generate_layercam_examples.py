from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Shared model/checkpoint helpers used by both evaluation and LayerCAM scripts.
from attacks import layercam_test_samples as lcam

DATA_DIR = os.path.join(PROJECT_DIR, "MAR20", "Classification_Dataset")
TEST_DIR = os.path.join(DATA_DIR, "test")
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "results", "original_baseline", "weights")
DEFAULT_OUT = os.path.join(PROJECT_DIR, "evaluation", "Evaluate_models")
METRICS_CSV = os.path.join(DEFAULT_OUT, "classification_metrics.csv")
INPUT_SIZE = 448
NUM_PER_CLASS = 30
SAMPLE_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_family_name(model_name: str) -> str:
    if model_name in ("resnet18", "resnet34", "resnet50"):
        return "ResNet"
    if model_name.startswith("efficientnet"):
        return "EfficientNet"
    if model_name in ("mobilenetv2", "mobilenetv3"):
        return "MobileNet"
    return "Other"


def detailed_model_name(model_name: str) -> str:
    return model_name


def model_load_name(model_name: str) -> str:
    return model_name


def model_result_dir(base_dir: str, model_name: str) -> str:
    return os.path.join(base_dir, model_family_name(model_name), detailed_model_name(model_name))


def read_best_checkpoints(metrics_csv: str, weights_dir: str) -> List[dict]:
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(
            f"{metrics_csv} 가 없습니다. 먼저 evaluate_models.py 를 실행하세요."
        )

    grouped: Dict[str, List[dict]] = defaultdict(list)
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            detail = row.get("model_detail") or detailed_model_name(row["model_name"])
            row["model_detail"] = detail
            grouped[detail].append(row)

    best_rows: List[dict] = []
    for detail in sorted(grouped.keys()):

        best = max(
            grouped[detail],
            key=lambda r: (float(r["accuracy"]), -int(r["seed"])),
        )
        ckpt = os.path.join(weights_dir, best["checkpoint"])
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt}")
        best["checkpoint_path"] = ckpt
        best_rows.append(best)
    return best_rows


def list_samples_by_class(test_dir: str) -> Tuple[List[str], Dict[str, List[str]]]:
    classes, samples = lcam.list_test_samples(test_dir)
    by_class: Dict[str, List[str]] = {c: [] for c in classes}
    for path, idx in samples:
        by_class[classes[int(idx)]].append(path)
    for c in classes:
        by_class[c] = sorted(by_class[c], key=lambda x: x.replace("\\", "/"))
    return classes, by_class


def sample_paths_by_class(
    classes: Sequence[str],
    by_class: Dict[str, List[str]],
    num_per_class: int,
    seed: int,
) -> Dict[str, List[str]]:
    selected: Dict[str, List[str]] = {}
    for idx, class_name in enumerate(classes):
        paths = by_class[class_name]
        rng = random.Random(seed + idx * 1009)
        if len(paths) <= num_per_class:
            selected[class_name] = list(paths)
        else:
            selected[class_name] = sorted(
                rng.sample(paths, num_per_class), key=lambda x: x.replace("\\", "/")
            )
    return selected


def layercam_map(model, cam: LayerCAM, preprocess, image_pil: Image.Image, pred_class: int) -> np.ndarray:
    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.set_grad_enabled(True):
        return cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_class)],
        )[0]


def save_layercam_pair(
    original: Image.Image,
    layercam_rgb: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.3), dpi=128)
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(layercam_rgb)
    axes[1].set_title("LayerCAM overlay", fontsize=11)
    axes[1].axis("off")
    fig.suptitle(title, fontsize=9, y=0.99)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)


def generate_for_checkpoint(
    row: dict,
    classes: Sequence[str],
    selected_by_class: Dict[str, List[str]],
    out_dir: str,
) -> None:
    raw_model_name = row["model_name"]
    detail = row["model_detail"]
    load_name = model_load_name(raw_model_name)
    ckpt_path = row["checkpoint_path"]
    checkpoint_base = os.path.splitext(os.path.basename(ckpt_path))[0]

    print(f"[LayerCAM] {detail} | {os.path.basename(ckpt_path)}")
    model = lcam.get_model(load_name, len(classes), use_pretrained=False).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    target_layers = lcam.get_layercam_target_layers(load_name, model)
    cam = LayerCAM(model=model, target_layers=target_layers)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    base_out = os.path.join(model_result_dir(out_dir, raw_model_name), checkpoint_base)
    for class_name in classes:
        paths = selected_by_class[class_name]
        class_out = os.path.join(base_out, class_name)
        os.makedirs(class_out, exist_ok=True)
        for i, path in enumerate(tqdm(paths, desc=f"  {detail}/{class_name}", leave=False), start=1):
            image = Image.open(path).convert("RGB")
            if image.size != (INPUT_SIZE, INPUT_SIZE):
                image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
            rgb_float = np.array(image, dtype=np.float32) / 255.0

            with torch.no_grad():
                pred_idx = int(model(preprocess(image).unsqueeze(0).to(device)).argmax(1).item())
            cam_map = layercam_map(model, cam, preprocess, image, pred_idx)
            overlay = show_cam_on_image(rgb_float, cam_map, use_rgb=True)

            image_id = os.path.splitext(os.path.basename(path))[0]
            out_name = f"layercam_{i:02d}_{class_name}_{image_id}_pred-{classes[pred_idx]}.png"
            title = (
                f"{detail} | {checkpoint_base} | GT={class_name} | "
                f"Pred={classes[pred_idx]}"
            )
            save_layercam_pair(image, overlay, os.path.join(class_out, out_name), title)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="세부 모델별 LayerCAM 예시를 클래스별로 생성합니다.")
    parser.add_argument("--metrics-csv", default=METRICS_CSV)
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR)
    parser.add_argument("--test-dir", default=TEST_DIR)
    parser.add_argument("--out-dir", default=os.path.join(DEFAULT_OUT, "LayerCAM examples"))
    parser.add_argument("--num-per-class", type=int, default=NUM_PER_CLASS)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    classes, by_class = list_samples_by_class(args.test_dir)
    selected_by_class = sample_paths_by_class(
        classes,
        by_class,
        num_per_class=args.num_per_class,
        seed=args.sample_seed,
    )
    best_rows = read_best_checkpoints(args.metrics_csv, args.weights_dir)

    print(f"클래스 수: {len(classes)}")
    print(f"클래스별 샘플 수: {args.num_per_class} (seed={args.sample_seed})")
    print(f"대상 모델 수: {len(best_rows)}")
    for row in best_rows:
        generate_for_checkpoint(row, classes, selected_by_class, args.out_dir)

    print(f"완료: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
