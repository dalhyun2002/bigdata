from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOGS_DIR = os.path.join(PROJECT_DIR, "results", "original_baseline", "logs")
DEFAULT_EVAL_CSV = os.path.join(PROJECT_DIR, "evaluation", "Evaluate_models", "classification_metrics.csv")
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "evaluation", "Evaluate_models", "training_curves_best_accuracy")


@dataclass
class TrainingRun:
    model: str
    model_detail: str
    weight_tag: str
    seed: int
    path: str
    rows: List[Dict[str, float]]
    test_accuracy: Optional[float] = None

    @property
    def label(self) -> str:
        acc_text = "" if self.test_accuracy is None else f", test_acc={self.test_accuracy:.4f}"
        return f"{self.model_detail} ({self.weight_tag}, seed={self.seed}{acc_text})"

    @property
    def stem(self) -> str:
        return f"{self.model_detail}_{self.weight_tag}_best_seed{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and save training curves from saved metrics CSV files."
    )
    parser.add_argument(
        "--logs-dir",
        default=DEFAULT_LOGS_DIR,
        help=f"학습 metrics CSV 폴더 (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--metrics-csv",
        default=DEFAULT_EVAL_CSV,
        help=f"accuracy 평가 CSV 파일 (default: {DEFAULT_EVAL_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"그래프 저장 폴더 (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="저장 이미지 DPI (default: 160)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="저장 후 화면에도 표시합니다.",
    )
    return parser.parse_args()


def to_float(value: str) -> float:
    return float(str(value).strip())


def load_best_accuracy_rows(metrics_csv: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    best_rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"model_name", "model_detail", "weight_tag", "seed", "accuracy"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{metrics_csv} missing columns: {sorted(missing)}")

        for row in reader:
            key = (row["model_detail"], row["weight_tag"])
            current = best_rows.get(key)
            if current is None or to_float(row["accuracy"]) > to_float(current["accuracy"]):
                best_rows[key] = row

    return best_rows


def discover_metric_files(logs_dir: str, metrics_csv: str) -> List[TrainingRun]:
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    best_rows = load_best_accuracy_rows(metrics_csv)
    runs: List[TrainingRun] = []
    for row in sorted(best_rows.values(), key=lambda item: item["model_detail"]):
        model = row["model_name"]
        model_detail = row["model_detail"]
        weight_tag = row["weight_tag"]
        seed = int(row["seed"])
        filename = f"metrics_{model}_{weight_tag}_seed{seed}.csv"
        path = os.path.join(logs_dir, filename)
        if not os.path.isfile(path):
            print(f"Skip {model_detail}: training log not found: {path}")
            continue

        rows = read_metric_rows(path)
        if not rows:
            print(f"Skip empty metrics file: {path}")
            continue

        runs.append(
            TrainingRun(
                model=model,
                model_detail=model_detail,
                weight_tag=weight_tag,
                seed=seed,
                path=path,
                rows=rows,
                test_accuracy=to_float(row["accuracy"]),
            )
        )

    return runs


def read_metric_rows(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"Skip {path}: missing columns {sorted(missing)}")
            return rows

        for row in reader:
            rows.append(
                {
                    "epoch": to_float(row["epoch"]),
                    "train_loss": to_float(row["train_loss"]),
                    "train_acc": to_float(row["train_acc"]),
                    "val_loss": to_float(row["val_loss"]),
                    "val_acc": to_float(row["val_acc"]),
                    "lr": to_float(row.get("lr", "nan")),
                    "is_best": 1.0 if str(row.get("is_best", "")).lower() in ("1", "true") else 0.0,
                }
            )
    return rows


def column(rows: Sequence[Dict[str, float]], key: str) -> List[float]:
    return [row[key] for row in rows]


def best_epoch(rows: Sequence[Dict[str, float]]) -> Optional[float]:
    best_rows = [row for row in rows if row["is_best"] == 1.0]
    if best_rows:
        return best_rows[-1]["epoch"]
    if not rows:
        return None
    return min(rows, key=lambda row: row["val_loss"])["epoch"]


def plot_single_run(run: TrainingRun, out_dir: str, dpi: int, show: bool) -> str:
    epochs = column(run.rows, "epoch")
    best = best_epoch(run.rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(run.label, fontsize=14, fontweight="bold")

    axes[0].plot(epochs, column(run.rows, "train_loss"), label="train_loss", linewidth=2)
    axes[0].plot(epochs, column(run.rows, "val_loss"), label="val_loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, column(run.rows, "train_acc"), label="train_acc", linewidth=2)
    axes[1].plot(epochs, column(run.rows, "val_acc"), label="val_acc", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if best is not None:
        for ax in axes:
            ax.axvline(best, color="tab:red", linestyle="--", alpha=0.55, label="best epoch")

    fig.tight_layout()
    acc_text = "" if run.test_accuracy is None else f"_acc{run.test_accuracy:.6f}"
    out_path = os.path.join(out_dir, f"training_curve_{run.stem}{acc_text}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_overview(
    runs: Sequence[TrainingRun],
    out_dir: str,
    dpi: int,
    metric: str,
    title: str,
    ylabel: str,
    show: bool,
) -> str:
    fig, ax = plt.subplots(figsize=(12, 7))
    for run in runs:
        ax.plot(column(run.rows, "epoch"), column(run.rows, metric), label=run.label, linewidth=1.7)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if "acc" in metric:
        ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"overview_{metric}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = discover_metric_files(args.logs_dir, args.metrics_csv)
    if not runs:
        raise RuntimeError(f"No selected training logs found from: {args.metrics_csv}")

    saved_paths = []
    for run in runs:
        saved_paths.append(plot_single_run(run, args.out_dir, args.dpi, args.show))

    saved_paths.append(
        plot_overview(runs, args.out_dir, args.dpi, "val_loss", "Validation Loss Overview", "Loss", args.show)
    )
    saved_paths.append(
        plot_overview(runs, args.out_dir, args.dpi, "val_acc", "Validation Accuracy Overview", "Accuracy", args.show)
    )

    print(f"Saved {len(saved_paths)} plot files:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
