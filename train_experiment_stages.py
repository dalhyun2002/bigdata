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
from torchvision import datasets, models, transforms
from tqdm import tqdm

try:
    from torchvision.transforms import v2
except ImportError:
    v2 = None


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "MAR20", "Classification_Dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "efficientnet_grid")
WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summaries")

BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-3
EPOCHS = 1000
PATIENCE_EARLY_STOP = 10
PATIENCE_LR = 5
FACTOR_LR = 0.1

INPUT_SIZE = 448
MLP_HIDDEN_DIM = 512
MLP_DROPOUT = 0.3
DEFAULT_SEEDS = [1, 2, 3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"


class ECAAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EfficientNetB0Experiment(nn.Module):
    def __init__(
        self,
        num_classes: int,
        attention: str = "none",
        head: str = "linear",
        activation: str = "relu",
        use_pretrained: bool = True,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        base = models.efficientnet_b0(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        channels = base.classifier[1].in_features

        if attention == "none":
            self.attention = nn.Identity()
        elif attention == "eca":
            self.attention = ECAAttention(channels)
        elif attention == "se":
            self.attention = SEAttention(channels)
        else:
            raise ValueError(f"Unsupported attention: {attention}")

        if head == "linear":
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(channels, num_classes),
            )
        elif head == "mlp512":
            if activation == "relu":
                act_layer = nn.ReLU(inplace=True)
            elif activation == "silu":
                act_layer = nn.SiLU(inplace=True)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            self.classifier = nn.Sequential(
                nn.Linear(channels, MLP_HIDDEN_DIM),
                act_layer,
                nn.Dropout(p=MLP_DROPOUT),
                nn.Linear(MLP_HIDDEN_DIM, num_classes),
            )
        else:
            raise ValueError(f"Unsupported head: {head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@dataclass(frozen=True)
class ExperimentConfig:
    stage: str
    model_name: str
    attention: str
    head: str
    train_scope: str
    activation: str = "relu"


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


def freeze_for_scope(model: EfficientNetB0Experiment, scope: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if scope == "added_only":
        modules = [model.attention, model.classifier]
    elif scope == "last2":
        modules = [model.features[-2], model.features[-1], model.attention, model.classifier]
    elif scope == "full":
        modules = [model]
    else:
        raise ValueError(f"Unsupported train scope: {scope}")

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_metrics_log(config: ExperimentConfig, seed: int, resume: bool = False) -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, f"metrics_{config.model_name}_pretrained_seed{seed}.csv")
    fields = [
        "stage",
        "model_name",
        "attention",
        "head",
        "activation",
        "train_scope",
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
                "attention": first["attention"],
                "head": first["head"],
                "activation": first["activation"],
                "train_scope": first["train_scope"],
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
                "trainable_params": first["trainable_params"],
            }
        )
    return out


def train_one(
    config: ExperimentConfig,
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

    model = EfficientNetB0Experiment(
        num_classes=num_classes,
        attention=config.attention,
        head=config.head,
        activation=config.activation,
        use_pretrained=True,
    ).to(device)
    freeze_for_scope(model, config.train_scope)
    trainable_params = trainable_parameter_count(model)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    log_path = create_metrics_log(config, seed, resume=args.resume)
    checkpoint_path = os.path.join(WEIGHTS_DIR, f"best_{config.model_name}_pretrained_seed{seed}.pth")
    last_checkpoint_path = os.path.join(WEIGHTS_DIR, f"last_{config.model_name}_pretrained_seed{seed}.pth")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=FACTOR_LR, patience=PATIENCE_LR
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_train_acc = 0.0
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
                "attention": config.attention,
                "head": config.head,
                "activation": config.activation,
                "train_scope": config.train_scope,
                "seed": seed,
                "best_epoch": best_epoch,
                "epochs_ran": epochs_ran,
                "best_train_acc": f"{best_train_acc:.8f}",
                "best_val_loss": f"{best_val_loss:.8f}",
                "best_val_acc": f"{best_val_acc:.8f}",
                "train_val_gap_at_best": f"{best_train_acc - best_val_acc:.8f}",
                "trainable_params": trainable_params,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "checkpoint": os.path.basename(checkpoint_path),
                "log_file": os.path.basename(log_path),
            }

    print("\n" + "=" * 80)
    print(f"Stage={config.stage} Model={config.model_name} Seed={seed}")
    print(
        f"Attention={config.attention} Head={config.head} "
        f"Activation={config.activation} Scope={config.train_scope}"
    )
    print(f"Trainable parameters: {trainable_params:,}")
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
            "attention": config.attention,
            "head": config.head,
            "activation": config.activation,
            "train_scope": config.train_scope,
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
        "attention": config.attention,
        "head": config.head,
        "activation": config.activation,
        "train_scope": config.train_scope,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "best_train_acc": f"{best_train_acc:.8f}",
        "best_val_loss": f"{best_val_loss:.8f}",
        "best_val_acc": f"{best_val_acc:.8f}",
        "train_val_gap_at_best": f"{best_train_acc - best_val_acc:.8f}",
        "trainable_params": trainable_params,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "checkpoint": os.path.basename(checkpoint_path),
        "log_file": os.path.basename(log_path),
    }


def stage1_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig("stage1_attention", "efficientnetb0_se_mlp512_full", "se", "mlp512", "full"),
        ExperimentConfig("stage1_attention", "efficientnetb0_eca_mlp512_full", "eca", "mlp512", "full"),
    ]


def activation_suffix(activation: str) -> str:
    return "" if activation == "relu" else f"_{activation}"


def stage3_configs(attention: str, activation: str) -> List[ExperimentConfig]:
    act_suffix = activation_suffix(activation)
    return [
        ExperimentConfig(
            "stage3_ablation",
            f"efficientnetb0_mlp512{act_suffix}_full",
            "none",
            "mlp512",
            "full",
            activation,
        ),
        ExperimentConfig(
            "stage3_ablation",
            f"efficientnetb0_{attention}_full",
            attention,
            "linear",
            "full",
        ),
        ExperimentConfig(
            "stage3_ablation",
            f"efficientnetb0_{attention}_mlp512{act_suffix}_full",
            attention,
            "mlp512",
            "full",
            activation,
        ),
    ]


def stage2_configs(attention: str) -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            "stage2_activation",
            f"efficientnetb0_{attention}_mlp512_relu_full",
            attention,
            "mlp512",
            "full",
            "relu",
        ),
        ExperimentConfig(
            "stage2_activation",
            f"efficientnetb0_{attention}_mlp512_silu_full",
            attention,
            "mlp512",
            "full",
            "silu",
        ),
    ]


def stage4_configs(attention: str, head: str, activation: str) -> List[ExperimentConfig]:
    suffix = attention if head == "linear" else f"{attention}_mlp512{activation_suffix(activation)}"
    cfg_activation = "relu" if head == "linear" else activation
    return [
        ExperimentConfig(
            "stage4_scope",
            f"efficientnetb0_{suffix}_added_only",
            attention,
            head,
            "added_only",
            cfg_activation,
        ),
        ExperimentConfig(
            "stage4_scope",
            f"efficientnetb0_{suffix}_last2",
            attention,
            head,
            "last2",
            cfg_activation,
        ),
        ExperimentConfig(
            "stage4_scope",
            f"efficientnetb0_{suffix}_full",
            attention,
            head,
            "full",
            cfg_activation,
        ),
    ]


def grid_configs() -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    for attention in ("none", "eca", "se"):
        attn_name = "noattn" if attention == "none" else attention
        for scope in ("added_only", "last2", "full"):
            configs.append(
                ExperimentConfig(
                    "grid",
                    f"grid_efficientnetb0_{attn_name}_linear_{scope}",
                    attention,
                    "linear",
                    scope,
                    "relu",
                )
            )
            for activation in ("relu", "silu"):
                configs.append(
                    ExperimentConfig(
                        "grid",
                        f"grid_efficientnetb0_{attn_name}_mlp512{activation_suffix(activation)}_{scope}",
                        attention,
                        "mlp512",
                        scope,
                        activation,
                    )
                )
    return configs


def dedupe_configs(configs: Iterable[ExperimentConfig]) -> List[ExperimentConfig]:
    seen = set()
    out = []
    for cfg in configs:
        key = cfg.model_name
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def select_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    stage = args.stage.lower()
    if stage == "1":
        return stage1_configs()
    if stage == "2":
        return stage2_configs(args.selected_attention)
    if stage in ("2.5", "25"):
        return stage2_configs(args.selected_attention)
    if stage == "3":
        return stage3_configs(args.selected_attention, args.selected_activation)
    if stage == "4":
        return stage4_configs(args.selected_attention, args.selected_head, args.selected_activation)
    if stage == "grid":
        return grid_configs()
    if stage == "all":
        configs: List[ExperimentConfig] = []
        configs.extend(stage1_configs())
        configs.extend(stage2_configs(args.selected_attention))
        configs.extend(stage3_configs(args.selected_attention, args.selected_activation))
        configs.extend(stage4_configs(args.selected_attention, args.selected_head, args.selected_activation))
        return dedupe_configs(configs)
    raise ValueError(f"Unsupported stage: {args.stage}")


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(x) for x in raw.replace(",", " ").split() if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train staged EfficientNet-B0 ECA/SE/MLP experiments."
    )
    parser.add_argument("--stage", choices=["1", "2", "2.5", "25", "3", "4", "all", "grid"], required=True)
    parser.add_argument("--selected-attention", choices=["eca", "se"], default="eca")
    parser.add_argument("--selected-head", choices=["linear", "mlp512"], default="mlp512")
    parser.add_argument("--selected-activation", choices=["relu", "silu"], default="relu")
    parser.add_argument("--seeds", default="1 2 3")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    configs = select_configs(args)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Stage: {args.stage}")
    print(f"Seeds: {seeds}")
    print(f"Results dir: {RESULTS_DIR}")
    print("Planned experiments:")
    for cfg in configs:
        print(
            f"  - {cfg.model_name}: attention={cfg.attention}, "
            f"head={cfg.head}, activation={cfg.activation}, scope={cfg.train_scope}"
        )

    plan_rows = [
        {
            "stage": cfg.stage,
            "model_name": cfg.model_name,
            "attention": cfg.attention,
            "head": cfg.head,
            "activation": cfg.activation,
            "train_scope": cfg.train_scope,
            "seeds": " ".join(str(s) for s in seeds),
        }
        for cfg in configs
    ]
    write_rows(os.path.join(SUMMARY_DIR, f"stage{args.stage}_plan.csv"), plan_rows)

    if args.dry_run:
        print("Dry run only. No training started.")
        return 0

    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Expected dataset folders: {TRAIN_DIR}, {VAL_DIR}")

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transforms.ToTensor())
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transforms.ToTensor())
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train/val class mismatch detected.")
    num_classes = len(train_dataset.classes)

    rows: List[dict] = []
    for cfg in configs:
        for seed in seeds:
            rows.append(train_one(cfg, seed, train_dataset, val_dataset, num_classes, args))
            summary_path = os.path.join(SUMMARY_DIR, f"stage{args.stage}_run_summary.csv")
            write_rows(summary_path, rows)
            aggregate_path = os.path.join(SUMMARY_DIR, f"stage{args.stage}_aggregate_summary.csv")
            write_rows(aggregate_path, build_aggregate_rows(rows))

    write_rows(os.path.join(SUMMARY_DIR, f"stage{args.stage}_run_summary.csv"), rows)
    write_rows(
        os.path.join(SUMMARY_DIR, f"stage{args.stage}_aggregate_summary.csv"),
        build_aggregate_rows(rows),
    )
    print("Finished staged experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
