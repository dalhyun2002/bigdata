from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from torchvision.transforms import v2
except ImportError:
    v2 = None


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "MAR20", "Classification_Dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "scratch_grid")
WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summaries")

BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-3
EPOCHS = 1000
PATIENCE_EARLY_STOP = 10
PATIENCE_LR = 5
FACTOR_LR = 0.1
MLP_HIDDEN_DIM = 512
HEAD_DROPOUT = 0.3

SIZE_CHANNELS = {
    "small": [16, 32, 64, 128],
    "base": [32, 64, 128, 256],
    "deep": [32, 64, 128, 256, 512],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"


@dataclass(frozen=True)
class ScratchConfig:
    stage: str
    model_name: str
    size: str
    downsampling: str
    activation: str
    head: str


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsampling: str,
        activation: str,
    ):
        super().__init__()
        if activation == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif activation == "silu":
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if downsampling == "maxpool":
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                act_layer,
                nn.MaxPool2d(2),
            )
        elif downsampling == "strided":
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                act_layer,
            )
        else:
            raise ValueError(f"Unsupported downsampling: {downsampling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScratchCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        size: str,
        downsampling: str,
        activation: str,
        head: str,
    ):
        super().__init__()
        channels = SIZE_CHANNELS[size]
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch, downsampling, activation))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        last_ch = channels[-1]

        if activation == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif activation == "silu":
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if head == "gap":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p=HEAD_DROPOUT),
                nn.Linear(last_ch, num_classes),
            )
        elif head == "pool4mlp":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(last_ch * 4 * 4, MLP_HIDDEN_DIM),
                act_layer,
                nn.Dropout(p=HEAD_DROPOUT),
                nn.Linear(MLP_HIDDEN_DIM, num_classes),
            )
        else:
            raise ValueError(f"Unsupported head: {head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_gpu_transforms(target_device: torch.device):
    if v2 is None:
        raise ImportError("torchvision.transforms.v2 is required.")
    train_gpu_transforms = v2.Compose(
        [
            v2.RandomRotation(360),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ).to(target_device)
    val_gpu_transforms = v2.Compose(
        [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    ).to(target_device)
    return train_gpu_transforms, val_gpu_transforms


def make_dataloaders(train_dataset, val_dataset, seed: int, batch_size: int, num_workers: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def create_metrics_log(config: ScratchConfig, seed: int, resume: bool = False) -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"metrics_{config.model_name}_scratch_seed{seed}.csv")
    fields = [
        "stage",
        "model_name",
        "size",
        "downsampling",
        "activation",
        "head",
        "seed",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "epoch_time_sec",
        "is_best",
        "early_stop_counter",
        "grad_accum_steps",
        "effective_batch_size",
    ]
    if resume and os.path.exists(path):
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
    return path


def append_csv(path: str, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_last_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    best_val_acc: float,
    best_train_acc: float,
    best_epoch: int,
    early_stop_counter: int,
    completed: bool = False,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_train_acc": best_train_acc,
            "best_epoch": best_epoch,
            "early_stop_counter": early_stop_counter,
            "completed": completed,
        },
        path,
    )


def load_last_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def write_rows(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def build_aggregate_rows(rows: List[dict]) -> List[dict]:
    grouped: dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["model_name"], []).append(row)

    out: List[dict] = []
    for model_name in sorted(grouped.keys()):
        items = sorted(grouped[model_name], key=lambda r: int(r["seed"]))
        val_accs = [float(r["best_val_acc"]) for r in items]
        val_losses = [float(r["best_val_loss"]) for r in items]
        gaps = [float(r["train_val_gap_at_best"]) for r in items]
        epochs = [float(r["best_epoch"]) for r in items]
        acc_mean, acc_std = mean_std(val_accs)
        loss_mean, loss_std = mean_std(val_losses)
        gap_mean, gap_std = mean_std(gaps)
        epoch_mean, epoch_std = mean_std(epochs)
        best_item = max(items, key=lambda r: (float(r["best_val_acc"]), -float(r["best_val_loss"])))
        first = items[0]
        out.append(
            {
                "stage": first["stage"],
                "model_name": model_name,
                "size": first["size"],
                "downsampling": first["downsampling"],
                "activation": first["activation"],
                "head": first["head"],
                "n_seeds": len(items),
                "seeds": " ".join(str(r["seed"]) for r in items),
                "best_val_acc_mean": f"{acc_mean:.8f}",
                "best_val_acc_std": f"{acc_std:.8f}",
                "best_val_loss_mean": f"{loss_mean:.8f}",
                "best_val_loss_std": f"{loss_std:.8f}",
                "train_val_gap_mean": f"{gap_mean:.8f}",
                "train_val_gap_std": f"{gap_std:.8f}",
                "best_epoch_mean": f"{epoch_mean:.4f}",
                "best_epoch_std": f"{epoch_std:.4f}",
                "best_seed": best_item["seed"],
                "best_seed_val_acc": best_item["best_val_acc"],
                "best_seed_checkpoint": best_item["checkpoint"],
                "params": first["params"],
            }
        )
    return out


def config_name(size: str, downsampling: str, activation: str, head: str) -> str:
    return f"scratch_{size}_{downsampling}_{activation}_{head}"


def make_config(stage: str, size: str, downsampling: str, activation: str, head: str) -> ScratchConfig:
    return ScratchConfig(stage, config_name(size, downsampling, activation, head), size, downsampling, activation, head)


def stage_configs(args: argparse.Namespace) -> List[ScratchConfig]:
    stage = args.stage.lower()
    if stage == "1":
        return [
            make_config("stage1_size", size, "maxpool", "relu", "gap")
            for size in ("small", "base", "deep")
        ]
    if stage == "2":
        return [
            make_config("stage2_downsampling", args.selected_size, downsampling, "relu", "gap")
            for downsampling in ("maxpool", "strided")
        ]
    if stage == "3":
        return [
            make_config("stage3_activation", args.selected_size, args.selected_downsampling, activation, "gap")
            for activation in ("relu", "silu")
        ]
    if stage == "4":
        return [
            make_config(
                "stage4_head",
                args.selected_size,
                args.selected_downsampling,
                args.selected_activation,
                head,
            )
            for head in ("gap", "pool4mlp")
        ]
    raise ValueError(f"Unsupported staged stage: {args.stage}")


def grid_configs() -> List[ScratchConfig]:
    configs = []
    for size in ("small", "base", "deep"):
        for downsampling in ("maxpool", "strided"):
            for activation in ("relu", "silu"):
                for head in ("gap", "pool4mlp"):
                    configs.append(make_config("grid", size, downsampling, activation, head))
    return configs


def dedupe_configs(configs: Iterable[ScratchConfig]) -> List[ScratchConfig]:
    seen = set()
    out = []
    for cfg in configs:
        if cfg.model_name in seen:
            continue
        seen.add(cfg.model_name)
        out.append(cfg)
    return out


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(x) for x in raw.replace(",", " ").split() if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def best_config_from_aggregate(rows: List[dict]) -> ScratchConfig:
    if not rows:
        raise ValueError("No aggregate rows to select from.")
    best = max(
        rows,
        key=lambda r: (
            float(r["best_val_acc_mean"]),
            -float(r["best_val_acc_std"]),
            -abs(float(r["train_val_gap_mean"])),
        ),
    )
    return make_config("selected", best["size"], best["downsampling"], best["activation"], best["head"])


def train_one(
    config: ScratchConfig,
    seed: int,
    train_dataset,
    val_dataset,
    num_classes: int,
    args: argparse.Namespace,
) -> dict:
    set_seed(seed)
    train_loader, val_loader = make_dataloaders(
        train_dataset, val_dataset, seed, args.batch_size, args.num_workers
    )
    train_gpu_tf, val_gpu_tf = build_gpu_transforms(device)

    model = ScratchCNN(
        num_classes=num_classes,
        size=config.size,
        downsampling=config.downsampling,
        activation=config.activation,
        head=config.head,
    ).to(device)
    params = model_parameter_count(model)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    log_path = create_metrics_log(config, seed, resume=args.resume)
    checkpoint_path = os.path.join(WEIGHTS_DIR, f"best_{config.model_name}_scratch_seed{seed}.pth")
    last_checkpoint_path = os.path.join(WEIGHTS_DIR, f"last_{config.model_name}_scratch_seed{seed}.pth")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=FACTOR_LR, patience=PATIENCE_LR
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    epochs_ran = 0
    start_epoch = 1

    if args.resume and os.path.exists(last_checkpoint_path):
        checkpoint = load_last_checkpoint(last_checkpoint_path, model, optimizer, scheduler, scaler)
        best_val_loss = float(checkpoint["best_val_loss"])
        best_val_acc = float(checkpoint["best_val_acc"])
        best_train_acc = float(checkpoint["best_train_acc"])
        best_epoch = int(checkpoint["best_epoch"])
        early_stop_counter = int(checkpoint["early_stop_counter"])
        epochs_ran = int(checkpoint["epoch"])
        start_epoch = epochs_ran + 1
        print(f"=> Resumed from {last_checkpoint_path} at epoch {epochs_ran}.")
        if checkpoint.get("completed", False):
            print("=> This run was already completed. Skipping.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return {
                "stage": config.stage,
                "model_name": config.model_name,
                "size": config.size,
                "downsampling": config.downsampling,
                "activation": config.activation,
                "head": config.head,
                "seed": seed,
                "best_epoch": best_epoch,
                "epochs_ran": epochs_ran,
                "best_train_acc": f"{best_train_acc:.8f}",
                "best_val_loss": f"{best_val_loss:.8f}",
                "best_val_acc": f"{best_val_acc:.8f}",
                "train_val_gap_at_best": f"{best_train_acc - best_val_acc:.8f}",
                "params": params,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "checkpoint": os.path.basename(checkpoint_path),
                "log_file": os.path.basename(log_path),
            }

    print("\n" + "=" * 80)
    print(f"Stage={config.stage} Model={config.model_name} Seed={seed}")
    print(
        f"size={config.size} downsampling={config.downsampling} "
        f"activation={config.activation} head={config.head}"
    )
    print(f"Parameters: {params:,}")
    print(f"Batch size: {args.batch_size} | grad_accum_steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"Log: {log_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        epochs_ran = epoch
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        train_bar = tqdm(
            train_loader,
            desc=f"{config.model_name}/s{seed} Epoch {epoch}/{args.epochs} [Train]",
        )
        for step_idx, (inputs, labels) in enumerate(train_bar, start=1):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = train_gpu_tf(inputs)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                backward_loss = loss / args.grad_accum_steps
            scaler.scale(backward_loss).backward()
            if step_idx % args.grad_accum_steps == 0 or step_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader,
                desc=f"{config.model_name}/s{seed} Epoch {epoch}/{args.epochs} [Val]",
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = val_gpu_tf(inputs)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        is_best = 0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_epoch = epoch
            early_stop_counter = 0
            is_best = 1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"=> Saved best model: {checkpoint_path}")
        else:
            early_stop_counter += 1

        row = {
            "stage": config.stage,
            "model_name": config.model_name,
            "size": config.size,
            "downsampling": config.downsampling,
            "activation": config.activation,
            "head": config.head,
            "seed": seed,
            "epoch": epoch,
            "train_loss": f"{train_loss:.8f}",
            "train_acc": f"{train_acc:.8f}",
            "val_loss": f"{val_loss:.8f}",
            "val_acc": f"{val_acc:.8f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.10f}",
            "epoch_time_sec": f"{time.time() - epoch_start:.4f}",
            "is_best": is_best,
            "early_stop_counter": early_stop_counter,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
        }
        append_csv(log_path, row)
        save_last_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
            best_val_acc,
            best_train_acc,
            best_epoch,
            early_stop_counter,
        )

        print(
            f"[{config.model_name}/seed{seed}/epoch{epoch}] "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"val_loss={val_loss:.4f} early_stop={early_stop_counter}/{PATIENCE_EARLY_STOP}"
        )

        if early_stop_counter >= PATIENCE_EARLY_STOP:
            save_last_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                best_val_acc,
                best_train_acc,
                best_epoch,
                early_stop_counter,
                completed=True,
            )
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "stage": config.stage,
        "model_name": config.model_name,
        "size": config.size,
        "downsampling": config.downsampling,
        "activation": config.activation,
        "head": config.head,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "best_train_acc": f"{best_train_acc:.8f}",
        "best_val_loss": f"{best_val_loss:.8f}",
        "best_val_acc": f"{best_val_acc:.8f}",
        "train_val_gap_at_best": f"{best_train_acc - best_val_acc:.8f}",
        "params": params,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "checkpoint": os.path.basename(checkpoint_path),
        "log_file": os.path.basename(log_path),
    }


def run_configs(
    label: str,
    configs: List[ScratchConfig],
    seeds: List[int],
    train_dataset,
    val_dataset,
    num_classes: int,
    args: argparse.Namespace,
) -> List[dict]:
    rows: List[dict] = []
    for cfg in configs:
        for seed in seeds:
            rows.append(train_one(cfg, seed, train_dataset, val_dataset, num_classes, args))
            write_rows(os.path.join(SUMMARY_DIR, f"{label}_run_summary.csv"), rows)
            write_rows(os.path.join(SUMMARY_DIR, f"{label}_aggregate_summary.csv"), build_aggregate_rows(rows))
    return rows


def write_plan(label: str, configs: List[ScratchConfig], seeds: List[int]) -> None:
    rows = [
        {
            "stage": cfg.stage,
            "model_name": cfg.model_name,
            "size": cfg.size,
            "downsampling": cfg.downsampling,
            "activation": cfg.activation,
            "head": cfg.head,
            "seeds": " ".join(str(s) for s in seeds),
        }
        for cfg in configs
    ]
    write_rows(os.path.join(SUMMARY_DIR, f"{label}_plan.csv"), rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scratch CNN staged or grid experiments.")
    parser.add_argument("--mode", choices=["staged", "grid"], required=True)
    parser.add_argument("--stage", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--selected-size", choices=["small", "base", "deep"], default="base")
    parser.add_argument("--selected-downsampling", choices=["maxpool", "strided"], default="maxpool")
    parser.add_argument("--selected-activation", choices=["relu", "silu"], default="relu")
    parser.add_argument("--seeds", default="1 2 3")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_datasets():
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Expected dataset folders: {TRAIN_DIR}, {VAL_DIR}")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transforms.ToTensor())
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transforms.ToTensor())
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train/val class mismatch detected.")
    return train_dataset, val_dataset, len(train_dataset.classes)


def maybe_limit(configs: List[ScratchConfig], limit_runs: int, seeds: List[int]) -> tuple[List[ScratchConfig], List[int]]:
    if limit_runs <= 0:
        return configs, seeds
    out_configs = []
    out_seeds = []
    count = 0
    for cfg in configs:
        for seed in seeds:
            if count >= limit_runs:
                return dedupe_configs(out_configs), sorted(set(out_seeds))
            out_configs.append(cfg)
            out_seeds.append(seed)
            count += 1
    return dedupe_configs(out_configs), sorted(set(out_seeds))


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Mode: {args.mode}")
    print(f"Stage: {args.stage}")
    print(f"Seeds: {seeds}")
    print(f"Results dir: {RESULTS_DIR}")

    if args.mode == "grid":
        configs = grid_configs()
        configs, run_seeds = maybe_limit(configs, args.limit_runs, seeds)
        print(f"Planned grid configs: {len(configs)} | seeds: {run_seeds}")
        for cfg in configs:
            print(f"  - {cfg.model_name}")
        write_plan("grid", configs, run_seeds)
        if args.dry_run:
            print("Dry run only. No training started.")
            return 0
        train_dataset, val_dataset, num_classes = load_datasets()
        rows = run_configs("grid", configs, run_seeds, train_dataset, val_dataset, num_classes, args)
        aggregate = build_aggregate_rows(rows)
        write_rows(os.path.join(SUMMARY_DIR, "grid_aggregate_summary.csv"), aggregate)
        best = best_config_from_aggregate(aggregate)
        write_rows(
            os.path.join(SUMMARY_DIR, "final_scratch_selection.csv"),
            [
                {
                    "selection_source": "grid",
                    "model_name": best.model_name,
                    "size": best.size,
                    "downsampling": best.downsampling,
                    "activation": best.activation,
                    "head": best.head,
                }
            ],
        )
        return 0

    if args.stage != "all":
        configs = stage_configs(args)
        configs, run_seeds = maybe_limit(configs, args.limit_runs, seeds)
        print(f"Planned staged configs: {len(configs)} | seeds: {run_seeds}")
        for cfg in configs:
            print(f"  - {cfg.model_name}")
        label = f"stage{args.stage}"
        write_plan(label, configs, run_seeds)
        if args.dry_run:
            print("Dry run only. No training started.")
            return 0
        train_dataset, val_dataset, num_classes = load_datasets()
        run_configs(label, configs, run_seeds, train_dataset, val_dataset, num_classes, args)
        return 0

    print("Running staged all with automatic selection.")
    all_rows: List[dict] = []
    train_dataset = val_dataset = None
    num_classes = 0
    if not args.dry_run:
        train_dataset, val_dataset, num_classes = load_datasets()

    # Stage 1
    configs1 = stage_configs(argparse.Namespace(stage="1"))
    write_plan("stage1", configs1, seeds)
    if args.dry_run:
        print("Stage 1:")
        for cfg in configs1:
            print(f"  - {cfg.model_name}")
        return 0
    rows1 = run_configs("stage1", configs1, seeds, train_dataset, val_dataset, num_classes, args)
    all_rows.extend(rows1)
    best1 = best_config_from_aggregate(build_aggregate_rows(rows1))

    # Stage 2
    ns2 = argparse.Namespace(stage="2", selected_size=best1.size)
    configs2 = [cfg for cfg in stage_configs(ns2) if cfg.downsampling != best1.downsampling]
    write_plan("stage2", configs2, seeds)
    rows2 = run_configs("stage2", configs2, seeds, train_dataset, val_dataset, num_classes, args)
    all_rows.extend(rows2)
    best2 = best_config_from_aggregate(build_aggregate_rows(all_rows))

    # Stage 3
    ns3 = argparse.Namespace(stage="3", selected_size=best2.size, selected_downsampling=best2.downsampling)
    configs3 = [cfg for cfg in stage_configs(ns3) if cfg.activation != best2.activation]
    write_plan("stage3", configs3, seeds)
    rows3 = run_configs("stage3", configs3, seeds, train_dataset, val_dataset, num_classes, args)
    all_rows.extend(rows3)
    best3 = best_config_from_aggregate(build_aggregate_rows(all_rows))

    # Stage 4
    ns4 = argparse.Namespace(
        stage="4",
        selected_size=best3.size,
        selected_downsampling=best3.downsampling,
        selected_activation=best3.activation,
    )
    configs4 = [cfg for cfg in stage_configs(ns4) if cfg.head != best3.head]
    write_plan("stage4", configs4, seeds)
    rows4 = run_configs("stage4", configs4, seeds, train_dataset, val_dataset, num_classes, args)
    all_rows.extend(rows4)
    final_aggregate = build_aggregate_rows(all_rows)
    write_rows(os.path.join(SUMMARY_DIR, "staged_all_aggregate_summary.csv"), final_aggregate)
    best_final = best_config_from_aggregate(final_aggregate)
    write_rows(
        os.path.join(SUMMARY_DIR, "final_scratch_selection.csv"),
        [
            {
                "selection_source": "staged",
                "model_name": best_final.model_name,
                "size": best_final.size,
                "downsampling": best_final.downsampling,
                "activation": best_final.activation,
                "head": best_final.head,
            }
        ],
    )
    print(f"Final selected scratch config: {best_final.model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

