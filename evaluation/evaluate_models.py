from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from torchvision.transforms import v2
except ImportError as e:
    raise ImportError("torchvision.transforms.v2 필요") from e

try:
    import seaborn as sns
except ImportError as e:
    raise ImportError("pip install seaborn") from e

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Shared model/checkpoint helpers used by both evaluation and LayerCAM scripts.
from attacks import layercam_test_samples as lcam

DATA_DIR = os.path.join(PROJECT_DIR, "MAR20", "Classification_Dataset")
TEST_DIR = os.path.join(DATA_DIR, "test")
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "results", "original_baseline", "weights")
DEFAULT_OUT = os.path.join(PROJECT_DIR, "evaluation", "Evaluate_models")
INPUT_SIZE = 448
REPO_MODEL_NAMES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnetb0",
    "mobilenetv2",
    "mobilenetv3",
]
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


def discover_checkpoints(weights_dir: str) -> List[Tuple[str, str, str, int]]:
    if not os.path.isdir(weights_dir):
        return []
    out: List[Tuple[str, str, str, int]] = []
    for fn in os.listdir(weights_dir):
        if not str(fn).endswith(".pth"):
            continue
        p = lcam.parse_checkpoint_filename(fn)
        if p is None:
            continue
        m, tag, seed = p
        if m in REPO_MODEL_NAMES:
            out.append((os.path.join(weights_dir, fn), m, tag, seed))
    return sorted(out, key=lambda x: (x[1], x[2], x[3], x[0]))


def _parse_or_infer_checkpoint(path: str) -> Optional[Tuple[str, str, int]]:
    if not path or not os.path.isfile(path):
        return None
    name = os.path.basename(path)
    p = lcam.parse_checkpoint_filename(name)
    if p:
        return p[0], p[1], p[2]
    for m in REPO_MODEL_NAMES:
        for tag in ("pretrained", "scratch"):
            mm = re.match(
                rf"^best_{re.escape(m)}_{re.escape(tag)}_seed(\d+)\.pth$", name, re.I
            )
            if mm:
                return m, tag, int(mm.group(1))
    return None


def build_test_loader(
    test_dir: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, List[str], int]:
    base = transforms.Compose(
        [transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()]
    )
    ds = datasets.ImageFolder(test_dir, transform=base)
    ld = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return ld, ds.classes, len(ds.classes)


def run_classification(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, float]:
    nrm = v2.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ).to(device)
    y_true, y_pred = [], []
    tot, cor = 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  classification", leave=False):
            x, y = nrm(x.to(device)), y.to(device)
            p = model(x).argmax(1)
            y_true.append(y.cpu().numpy())
            y_pred.append(p.cpu().numpy())
            tot += y.numel()
            cor += (p == y).sum().item()
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    return yt, yp, float(cor) / max(tot, 1)


def plot_confusion_normalized(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], out_path: str
) -> None:
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cm.astype(float)
        rs = r.sum(1, keepdims=True)
        row = np.divide(r, np.where(rs == 0, 1, rs), out=np.zeros_like(r, float), where=rs != 0)
    w = max(11, int(np.ceil(n * 0.55)))
    fig, ax = plt.subplots(figsize=(w, w))
    ax.set_facecolor("#fafafa")
    sns.heatmap(
        row,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        cbar_kws={"shrink": 0.72, "label": "p(predicted | true)"},
        linewidths=0.2,
        linecolor="#e0e0e0",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_xticklabels(list(class_names), rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(list(class_names), rotation=0, fontsize=8)
    ax.set_title("Row-normalized confusion matrix", fontsize=13, pad=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@dataclass
class RunRecord:
    path: str
    model_name: str
    tag: str
    seed: int
    acc: float
    y_pred: np.ndarray


def pick_best_per_model_name(recs: List[RunRecord]) -> List[RunRecord]:
    by_m: Dict[str, List[RunRecord]] = defaultdict(list)
    for r in recs:
        by_m[detailed_model_name(r.model_name)].append(r)
    out: List[RunRecord] = []
    ordered_details = list(dict.fromkeys(detailed_model_name(x) for x in REPO_MODEL_NAMES))
    for mn in ordered_details:
        if mn not in by_m:
            continue
        out.append(max(by_m[mn], key=lambda x: (x.acc, -x.seed)))
    return out


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def write_seed_summary(rows: List[dict], out_dir: str) -> List[dict]:
    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        key = (row["model_family"], row["model_detail"], row["weight_tag"])
        grouped[key].append(row)

    summary_rows: List[dict] = []
    print("--- seed 평균/표준편차 ---")
    for key in sorted(grouped.keys()):
        model_family, model_detail, weight_tag = key
        items = sorted(grouped[key], key=lambda x: int(x["seed"]))
        out_row: Dict[str, object] = {
            "model_family": model_family,
            "model_detail": model_detail,
            "weight_tag": weight_tag,
            "n_seeds": len(items),
            "seeds": " ".join(str(x["seed"]) for x in items),
        }
        for col in metric_cols:
            mean_v, std_v = mean_std([float(x[col]) for x in items])
            out_row[f"{col}_mean"] = f"{mean_v:.6f}"
            out_row[f"{col}_std"] = f"{std_v:.6f}"
        summary_rows.append(out_row)

        print(f"  {model_detail} ({weight_tag}, seeds={out_row['seeds']})")
        print(
            f"    accuracy mean/std: {out_row['accuracy_mean']} / {out_row['accuracy_std']}"
        )
        print(
            f"    macro F1 mean/std: {out_row['f1_macro_mean']} / {out_row['f1_macro_std']}"
        )
        print(
            f"    weighted F1 mean/std: {out_row['f1_weighted_mean']} / {out_row['f1_weighted_std']}"
        )
        print("  " + "-" * 60)

    if summary_rows:
        summary_path = os.path.join(out_dir, "classification_metrics_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"seed 요약 저장: {summary_path}")
    return summary_rows


def parse_args() -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument("--out-dir", default=DEFAULT_OUT)
    a.add_argument("--weights-dir", default=WEIGHTS_DIR)
    a.add_argument("--test-dir", default=TEST_DIR)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--num-workers", type=int, default=4)
    a.add_argument("--checkpoints", default="")
    return a.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out_dir
    cm_d = os.path.join(args.out_dir, "Confusion matrix")
    os.makedirs(out, exist_ok=True)

    if args.checkpoints.strip():
        cps: List[Tuple[str, str, str, int]] = []
        for pth in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
            inf = _parse_or_infer_checkpoint(pth)
            if inf and os.path.isfile(pth):
                cps.append((pth, inf[0], inf[1], inf[2]))
        ckpts = sorted(set(cps), key=lambda x: (x[1], x[2], x[3], x[0]))
    else:
        ckpts = discover_checkpoints(args.weights_dir)
    if not ckpts:
        print("weights 에 best_*.pth 없음", file=sys.stderr)
        return 1

    ld, cnames, ncls = build_test_loader(args.test_dir, args.batch_size, args.num_workers)
    lclasses, _ = lcam.list_test_samples(args.test_dir)
    if lclasses != cnames:
        raise ValueError("class mismatch")

    recs: List[RunRecord] = []
    rows: List[dict] = []
    yref: Optional[np.ndarray] = None

    for cp, mn, tag, seed in ckpts:
        detail = detailed_model_name(mn)
        load_name = model_load_name(mn)
        print(f"[분류] {os.path.basename(cp)}  {detail} seed{seed}")
        m = lcam.get_model(load_name, ncls, use_pretrained=False).to(device)
        try:
            st = torch.load(cp, map_location=device, weights_only=False)
        except TypeError:
            st = torch.load(cp, map_location=device)
        m.load_state_dict(st)
        m.eval()
        yt, yp, acc = run_classification(m, ld)
        if yref is None:
            yref = yt.copy()
        elif not np.array_equal(yref, yt):
            raise RuntimeError("y_true mismatch")
        pma, rma, f1ma, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0, labels=np.arange(ncls)
        )
        pw, rw, f1w, _ = precision_recall_fscore_support(
            yt, yp, average="weighted", zero_division=0, labels=np.arange(ncls)
        )
        fam = model_family_name(mn)
        recs.append(RunRecord(cp, mn, tag, seed, acc, yp.astype(np.int16)))
        rows.append(
            {
                "checkpoint": os.path.basename(cp),
                "model_name": mn,
                "model_detail": detail,
                "model_family": fam,
                "weight_tag": tag,
                "seed": seed,
                "n_test": len(yt),
                "accuracy": f"{acc:.6f}",
                "precision_macro": f"{pma:.6f}",
                "recall_macro": f"{rma:.6f}",
                "f1_macro": f"{f1ma:.6f}",
                "precision_weighted": f"{pw:.6f}",
                "recall_weighted": f"{rw:.6f}",
                "f1_weighted": f"{f1w:.6f}",
            }
        )
        print("  [이 시드 test 결과]")
        print(f"    n_test        : {len(yt)}")
        print(f"    accuracy      : {acc:.6f}")
        print(f"    macro   P     : {pma:.6f}   R: {rma:.6f}   F1: {f1ma:.6f}")
        print(f"    weighted P    : {pw:.6f}   R: {rw:.6f}   F1: {f1w:.6f}")
        print("  " + "-" * 60)
        with open(os.path.join(out, "classification_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert yref is not None
    y_true = yref
    best = pick_best_per_model_name(recs)
    by_ck = {r["checkpoint"]: r for r in rows}
    print("--- best per model_name (세부 모델별 accuracy 최고 시드) ---")
    for b in best:
        ck = os.path.basename(b.path)
        detail = detailed_model_name(b.model_name)
        print(f"  {detail}  {ck}  tag={b.tag}  seed={b.seed}")
        r = by_ck.get(ck)
        if r:
            print(f"    accuracy      : {r['accuracy']}")
            print(f"    macro   P/R/F1: {r['precision_macro']} / {r['recall_macro']} / {r['f1_macro']}")
            print(f"    weighted P/R/F1: {r['precision_weighted']} / {r['recall_weighted']} / {r['f1_weighted']}")
        else:
            print(f"    accuracy      : {b.acc:.6f}")
        print("  " + "-" * 60)
    for b in best:
        base = os.path.splitext(os.path.basename(b.path))[0]
        cm_out_dir = model_result_dir(cm_d, b.model_name)
        plot_confusion_normalized(
            y_true,
            b.y_pred.astype(np.int64),
            cnames,
            os.path.join(cm_out_dir, f"{base}_confusion_matrix.png"),
        )
    with open(os.path.join(out, "classification_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_seed_summary(rows, out)
    print("완료", out, "best", len(best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
