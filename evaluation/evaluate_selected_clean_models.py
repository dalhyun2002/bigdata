from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from torchvision.transforms import v2
except ImportError as exc:
    raise ImportError("torchvision.transforms.v2 is required.") from exc


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from attacks import layercam_test_samples as baseline_utils
from train_experiment_stages import EfficientNetB0Experiment
from train_scratch_cnn import ScratchCNN


DATA_DIR = os.path.join(PROJECT_DIR, "MAR20", "Classification_Dataset")
TEST_DIR = os.path.join(DATA_DIR, "test")
BASELINE_WEIGHTS_DIR = os.path.join(PROJECT_DIR, "results", "original_baseline", "weights")
EFFICIENTNET_WEIGHTS_DIR = os.path.join(PROJECT_DIR, "results", "efficientnet_grid", "weights")
SCRATCH_WEIGHTS_DIR = os.path.join(PROJECT_DIR, "results", "scratch_grid", "weights")
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "evaluation", "Clean_Selected_Models")
INPUT_SIZE = 448

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class SelectedModel:
    label: str
    model_type: str
    checkpoint: str
    seed: int
    validation_source: str
    validation_mean: str
    validation_std: str
    validation_best_acc: str


def build_selected_models() -> List[SelectedModel]:
    return [
        SelectedModel(
            label="baseline_efficientnetb0",
            model_type="baseline_efficientnetb0",
            checkpoint=os.path.join(
                BASELINE_WEIGHTS_DIR, "best_efficientnetb0_pretrained_seed4.pth"
            ),
            seed=4,
            validation_source="results/original_baseline/logs/original_baseline_aggregate_summary.csv",
            validation_mean="0.98312343",
            validation_std="0.00289946",
            validation_best_acc="0.98677582",
        ),
        SelectedModel(
            label="proposed_efficientnetb0_se_mlp512_full",
            model_type="proposed_efficientnetb0_se_mlp512_full",
            checkpoint=os.path.join(
                EFFICIENTNET_WEIGHTS_DIR,
                "best_grid_efficientnetb0_se_mlp512_full_pretrained_seed2.pth",
            ),
            seed=2,
            validation_source="results/efficientnet_grid/summaries/stagegrid_aggregate_summary.csv",
            validation_mean="0.98299748",
            validation_std="",
            validation_best_acc="0.99055416",
        ),
        SelectedModel(
            label="scratch_small_maxpool_silu_pool4mlp",
            model_type="scratch_small_maxpool_silu_pool4mlp",
            checkpoint=os.path.join(
                SCRATCH_WEIGHTS_DIR,
                "best_scratch_small_maxpool_silu_pool4mlp_scratch_seed2.pth",
            ),
            seed=2,
            validation_source="results/scratch_grid/summaries/grid_aggregate_summary.csv",
            validation_mean="0.84298908",
            validation_std="",
            validation_best_acc="0.85642317",
        ),
    ]


def build_test_loader(test_dir: str, batch_size: int, num_workers: int):
    transform = transforms.Compose(
        [transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader, dataset.classes


def create_model(model_type: str, num_classes: int) -> nn.Module:
    if model_type == "baseline_efficientnetb0":
        return baseline_utils.get_model("efficientnetb0", num_classes, use_pretrained=False)
    if model_type == "proposed_efficientnetb0_se_mlp512_full":
        return EfficientNetB0Experiment(
            num_classes=num_classes,
            attention="se",
            head="mlp512",
            activation="relu",
            use_pretrained=False,
        )
    if model_type == "scratch_small_maxpool_silu_pool4mlp":
        return ScratchCNN(
            num_classes=num_classes,
            size="small",
            downsampling="maxpool",
            activation="silu",
            head="pool4mlp",
        )
    raise ValueError(f"Unsupported selected model type: {model_type}")


def load_state_dict(path: str):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    normalize = v2.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ).to(device)
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="clean evaluation", leave=False):
            x = normalize(x.to(device, non_blocking=True))
            logits = model(x)
            preds = logits.argmax(dim=1)
            y_true.append(y.numpy())
            y_pred.append(preds.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def metric_row(selected: SelectedModel, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    acc = float((y_true == y_pred).mean())
    pma, rma, f1ma, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=np.arange(n_classes)
    )
    pw, rw, f1w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=np.arange(n_classes)
    )
    return {
        "model_label": selected.label,
        "model_type": selected.model_type,
        "seed": selected.seed,
        "checkpoint": os.path.basename(selected.checkpoint),
        "n_test": int(y_true.size),
        "val_acc_mean": selected.validation_mean,
        "val_acc_std": selected.validation_std,
        "val_best_seed_acc": selected.validation_best_acc,
        "accuracy": f"{acc:.6f}",
        "precision_macro": f"{float(pma):.6f}",
        "recall_macro": f"{float(rma):.6f}",
        "f1_macro": f"{float(f1ma):.6f}",
        "precision_weighted": f"{float(pw):.6f}",
        "recall_weighted": f"{float(rw):.6f}",
        "f1_weighted": f"{float(f1w):.6f}",
        "validation_source": selected.validation_source,
    }


def write_rows(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=TEST_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader, class_names = build_test_loader(args.test_dir, args.batch_size, args.num_workers)
    rows = []
    for selected in build_selected_models():
        if not os.path.isfile(selected.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {selected.checkpoint}")
        print(f"[clean] {selected.label} seed={selected.seed}")
        print(f"        checkpoint={selected.checkpoint}")
        model = create_model(selected.model_type, len(class_names)).to(device)
        model.load_state_dict(load_state_dict(selected.checkpoint))
        y_true, y_pred = evaluate(model, loader)
        row = metric_row(selected, y_true, y_pred, len(class_names))
        rows.append(row)
        print(
            "        "
            f"acc={row['accuracy']} macro_f1={row['f1_macro']} weighted_f1={row['f1_weighted']}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = os.path.join(args.out_dir, "selected_clean_metrics.csv")
    write_rows(out_path, rows)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
