from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from train_experiment_stages import EfficientNetB0Experiment

DATA_DIR = os.path.join(PROJECT_DIR, "MAR20", "Classification_Dataset")
TEST_DIR = os.path.join(DATA_DIR, "test")
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "results", "original_baseline", "weights")
EFFICIENTNET_GRID_WEIGHTS_DIR = os.path.join(PROJECT_DIR, "results", "efficientnet_grid", "weights")
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "attacks", "Adversarial_Attacks")
INPUT_SIZE = 448
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BASELINE_MODEL_NAMES = {"baseline_efficientnetb0", "efficientnetb0"}


@dataclass(frozen=True)
class AttackConfig:
    name: str
    epsilon_px: float
    alpha_px: float
    steps: int
    random_start: bool
    layercam_threshold: float

    @property
    def epsilon(self) -> float:
        return self.epsilon_px / 255.0

    @property
    def alpha(self) -> float:
        return self.alpha_px / 255.0


@dataclass
class BatchResult:
    labels: torch.Tensor
    clean_preds: torch.Tensor
    clean_confs: torch.Tensor
    adv_preds: torch.Tensor
    adv_confs: torch.Tensor
    correct_clean: torch.Tensor
    mask_area: torch.Tensor
    linf: torch.Tensor
    l2: torch.Tensor
    l0_area: torch.Tensor
    mae: torch.Tensor
    mse: torch.Tensor
    psnr: torch.Tensor
    ssim: torch.Tensor


class NormalizeModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class NormalizedModel(nn.Module):
    def __init__(self, normalize: nn.Module, model: nn.Module) -> None:
        super().__init__()
        self.normalize = normalize
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name in BASELINE_MODEL_NAMES:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name in {
        "proposed_efficientnetb0_se_mlp512_full",
        "grid_efficientnetb0_se_mlp512_full",
    }:
        return EfficientNetB0Experiment(
            num_classes=num_classes,
            attention="se",
            head="mlp512",
            activation="relu",
            use_pretrained=False,
        )
    raise ValueError(f"Unsupported model for this script: {model_name}")


def get_layercam_target_layers(model_name: str, model: nn.Module) -> List[nn.Module]:
    model_name = model_name.lower()
    if model_name in {
        *BASELINE_MODEL_NAMES,
        "proposed_efficientnetb0_se_mlp512_full",
        "grid_efficientnetb0_se_mlp512_full",
    }:
        return [model.features[-1]]
    raise ValueError(f"Unsupported model for LayerCAM: {model_name}")


def load_checkpoint(path: str, model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    model = get_model(model_name, num_classes).to(device)
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def build_test_loader(
    test_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_samples: int = 0,
    balanced_subset: bool = False,
    subset_seed: int = 42,
) -> Tuple[DataLoader, datasets.ImageFolder]:
    tf = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(test_dir, transform=tf)
    if max_samples > 0:
        if balanced_subset:
            rng = random.Random(subset_seed)
            by_class: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
            for sample in dataset.samples:
                by_class[int(sample[1])].append(sample)
            per_class = max(1, max_samples // max(len(dataset.classes), 1))
            selected: List[Tuple[str, int]] = []
            for class_idx in sorted(by_class):
                items = by_class[class_idx]
                if len(items) <= per_class:
                    selected.extend(items)
                else:
                    selected.extend(rng.sample(items, per_class))
            if len(selected) < max_samples:
                selected_paths = {path for path, _ in selected}
                remaining = [sample for sample in dataset.samples if sample[0] not in selected_paths]
                need = min(max_samples - len(selected), len(remaining))
                selected.extend(rng.sample(remaining, need))
            dataset.samples = sorted(selected[:max_samples], key=lambda item: item[0].replace("\\", "/"))
        else:
            dataset.samples = dataset.samples[:max_samples]
        dataset.imgs = dataset.samples
        dataset.targets = [label for _, label in dataset.samples]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader, dataset


def model_logits(model: nn.Module, normalize: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(normalize(x))


def predict(
    model: nn.Module,
    normalize: nn.Module,
    x: torch.Tensor,
    use_amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with torch.autocast(device_type=x.device.type, enabled=use_amp):
            prob = F.softmax(model_logits(model, normalize, x), dim=1)
        conf, pred = prob.max(dim=1)
    return pred, conf


def random_delta_like(x: torch.Tensor, epsilon: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    if mask is not None:
        delta = delta * mask
    return delta


def project_delta(delta: torch.Tensor, epsilon: float, mask: torch.Tensor | None = None) -> torch.Tensor:
    delta = torch.clamp(delta, -epsilon, epsilon)
    if mask is not None:
        delta = delta * mask
    return delta


def pgd_attack(
    model: nn.Module,
    normalize: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: AttackConfig,
    mask: torch.Tensor | None = None,
    use_amp: bool = False,
) -> torch.Tensor:
    epsilon = cfg.epsilon
    alpha = cfg.alpha
    if cfg.steps <= 0 and not cfg.random_start:
        return x.detach()

    if cfg.random_start:
        delta = random_delta_like(x, epsilon, mask=mask)
        delta = torch.clamp(x + delta, 0.0, 1.0) - x
        delta = project_delta(delta, epsilon, mask=mask)
    else:
        delta = torch.zeros_like(x)

    for _ in range(cfg.steps):
        delta.requires_grad_(True)
        with torch.autocast(device_type=x.device.type, enabled=use_amp):
            logits = model_logits(model, normalize, torch.clamp(x + delta, 0.0, 1.0))
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        step = alpha * grad.sign()
        if mask is not None:
            step = step * mask
        delta = delta.detach() + step
        delta = torch.clamp(x + delta, 0.0, 1.0) - x
        delta = project_delta(delta, epsilon, mask=mask)
    return torch.clamp(x + delta.detach(), 0.0, 1.0)


def random_noise_attack(
    x: torch.Tensor,
    cfg: AttackConfig,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    delta = random_delta_like(x, cfg.epsilon, mask=mask)
    delta = torch.clamp(x + delta, 0.0, 1.0) - x
    delta = project_delta(delta, cfg.epsilon, mask=mask)
    return torch.clamp(x + delta, 0.0, 1.0)


def make_layercam_mask(
    cam: LayerCAM,
    x: torch.Tensor,
    clean_preds: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    targets = [ClassifierOutputTarget(int(cls)) for cls in clean_preds.detach().cpu().tolist()]
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=x, targets=targets)

    cam_tensor = torch.from_numpy(grayscale_cam).to(device=x.device, dtype=x.dtype).unsqueeze(1)
    if cam_tensor.shape[-2:] != x.shape[-2:]:
        cam_tensor = F.interpolate(cam_tensor, size=x.shape[-2:], mode="bilinear", align_corners=False)
    mask = (cam_tensor >= threshold).to(dtype=x.dtype)

    empty = mask.flatten(1).sum(dim=1) == 0
    if empty.any():
        flat_cam = cam_tensor.flatten(1)
        max_idx = flat_cam.argmax(dim=1)
        flat_mask = mask.flatten(1)
        for row_idx in torch.nonzero(empty, as_tuple=False).flatten():
            flat_mask[row_idx, max_idx[row_idx]] = 1.0
        mask = flat_mask.view_as(mask)
    return mask


def make_random_mask_like(mask: torch.Tensor) -> torch.Tensor:
    random_mask = torch.zeros_like(mask)
    flat_random = random_mask.flatten(1)
    num_pixels = flat_random.size(1)
    active_counts = mask.flatten(1).sum(dim=1).long().clamp(min=1, max=num_pixels)
    for row_idx, active_count in enumerate(active_counts.tolist()):
        selected = torch.randperm(num_pixels, device=mask.device)[:active_count]
        flat_random[row_idx, selected] = 1.0
    return random_mask


def simple_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    dims = (1, 2, 3)
    mu_x = x.mean(dim=dims)
    mu_y = y.mean(dim=dims)
    var_x = ((x - mu_x.view(-1, 1, 1, 1)) ** 2).mean(dim=dims)
    var_y = ((y - mu_y.view(-1, 1, 1, 1)) ** 2).mean(dim=dims)
    cov_xy = (
        (x - mu_x.view(-1, 1, 1, 1)) * (y - mu_y.view(-1, 1, 1, 1))
    ).mean(dim=dims)
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    return torch.clamp(numerator / torch.clamp(denominator, min=1e-12), 0.0, 1.0)


def perturbation_stats(
    x: torch.Tensor,
    adv: torch.Tensor,
    mask: torch.Tensor | None,
    epsilon: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    delta = adv - x
    flat = delta.flatten(1)
    linf = flat.abs().max(dim=1).values * 255.0
    l2 = flat.pow(2).sum(dim=1).sqrt()
    mae = flat.abs().mean(dim=1)
    mse = flat.pow(2).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
    ssim = simple_ssim(x, adv)
    active_threshold = max(epsilon * 0.01, 1e-8)
    active = delta.abs().amax(dim=1, keepdim=True) > active_threshold
    l0_area = active.flatten(1).float().mean(dim=1)
    if mask is None:
        mask_area = torch.ones_like(l0_area)
    else:
        mask_area = mask.flatten(1).float().mean(dim=1)
    return (
        linf.detach(),
        l2.detach(),
        l0_area.detach(),
        mask_area.detach(),
        mae.detach(),
        mse.detach(),
        psnr.detach(),
        ssim.detach(),
    )


def tensor_to_rgb_image(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return arr


def perturbation_to_display(delta: torch.Tensor, epsilon: float) -> np.ndarray:
    scaled = delta.detach().cpu() / max(epsilon, 1e-12)
    scaled = (scaled.clamp(-1.0, 1.0) + 1.0) / 2.0
    return scaled.permute(1, 2, 0).numpy()


def save_example_figure(
    out_path: str,
    original: torch.Tensor,
    adversarial: torch.Tensor,
    label_name: str,
    clean_pred_name: str,
    clean_conf: float,
    adv_pred_name: str,
    adv_conf: float,
    attack_name: str,
    epsilon_px: float,
    mask_area: float,
) -> None:
    delta = adversarial - original
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), dpi=150)
    panels = [
        (
            tensor_to_rgb_image(original),
            "Original",
            f"GT: {label_name}\nClean pred: {clean_pred_name} ({clean_conf:.3f})",
        ),
        (
            perturbation_to_display(delta, epsilon_px / 255.0),
            "Noise mask",
            f"{attack_name}\neps: {epsilon_px:g}/255 | area: {mask_area:.3f}",
        ),
        (
            tensor_to_rgb_image(adversarial),
            "Original + noise",
            f"Adv pred: {adv_pred_name} ({adv_conf:.3f})",
        ),
    ]
    for ax, (image, title, subtitle) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=11, weight="bold", pad=8)
        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            linespacing=1.25,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#d0d0d0")
    fig.suptitle(f"{attack_name} example", fontsize=12, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=1.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_rows(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def build_comparison_rows(summary_rows: List[dict]) -> List[dict]:
    by_epsilon: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for row in summary_rows:
        by_epsilon[str(row["epsilon_px"])][str(row["method"])] = row

    comparisons: List[dict] = []
    for epsilon_px in sorted(by_epsilon.keys(), key=lambda x: float(x)):
        rows = by_epsilon[epsilon_px]
        layercam = rows.get("layercam_masked_pgd")
        random_masked = rows.get("random_masked_pgd")
        standard = rows.get("standard_pgd")

        if layercam and random_masked:
            lc_asr = _summary_float(layercam, "attack_success_rate_clean_correct")
            rd_asr = _summary_float(random_masked, "attack_success_rate_clean_correct")
            lc_drop = _summary_float(layercam, "accuracy_drop")
            rd_drop = _summary_float(random_masked, "accuracy_drop")
            lc_area = max(_summary_float(layercam, "mean_layercam_mask_area"), 1e-12)
            rd_area = max(_summary_float(random_masked, "mean_layercam_mask_area"), 1e-12)
            comparisons.append(
                {
                    "epsilon_px": epsilon_px,
                    "comparison": "layercam_masked_vs_random_masked",
                    "layercam_attack_success": f"{lc_asr:.6f}",
                    "random_attack_success": f"{rd_asr:.6f}",
                    "attack_success_gain": f"{lc_asr - rd_asr:.6f}",
                    "attack_success_gain_ratio": f"{lc_asr / max(rd_asr, 1e-12):.6f}",
                    "layercam_accuracy_drop": f"{lc_drop:.6f}",
                    "random_accuracy_drop": f"{rd_drop:.6f}",
                    "accuracy_drop_gain": f"{lc_drop - rd_drop:.6f}",
                    "layercam_area": f"{lc_area:.6f}",
                    "random_area": f"{rd_area:.6f}",
                    "layercam_asr_per_area": f"{lc_asr / lc_area:.6f}",
                    "random_asr_per_area": f"{rd_asr / rd_area:.6f}",
                    "layercam_psnr": layercam.get("mean_psnr", ""),
                    "random_psnr": random_masked.get("mean_psnr", ""),
                }
            )

        if layercam and standard:
            lc_asr = _summary_float(layercam, "attack_success_rate_clean_correct")
            st_asr = _summary_float(standard, "attack_success_rate_clean_correct")
            lc_area = _summary_float(layercam, "mean_layercam_mask_area")
            st_area = max(_summary_float(standard, "mean_layercam_mask_area"), 1e-12)
            comparisons.append(
                {
                    "epsilon_px": epsilon_px,
                    "comparison": "layercam_masked_vs_standard",
                    "layercam_attack_success": f"{lc_asr:.6f}",
                    "standard_attack_success": f"{st_asr:.6f}",
                    "attack_success_retention": f"{lc_asr / max(st_asr, 1e-12):.6f}",
                    "layercam_area": f"{lc_area:.6f}",
                    "standard_area": f"{st_area:.6f}",
                    "area_reduction": f"{1.0 - (lc_area / st_area):.6f}",
                    "layercam_psnr": layercam.get("mean_psnr", ""),
                    "standard_psnr": standard.get("mean_psnr", ""),
                    "psnr_gain": f"{_summary_float(layercam, 'mean_psnr') - _summary_float(standard, 'mean_psnr'):.6f}",
                }
            )
    return comparisons


def build_matched_attack_rows(summary_rows: List[dict]) -> List[dict]:
    standard = [row for row in summary_rows if row.get("method") == "standard_pgd"]
    proposed = [row for row in summary_rows if row.get("method") == "layercam_masked_pgd"]
    rows: List[dict] = []
    for prop in proposed:
        prop_asr = _summary_float(prop, "attack_success_rate_clean_correct")
        if not standard:
            continue
        match = min(
            standard,
            key=lambda row: abs(_summary_float(row, "attack_success_rate_clean_correct") - prop_asr),
        )
        std_asr = _summary_float(match, "attack_success_rate_clean_correct")
        rows.append(
            {
                "proposed_epsilon_px": prop.get("epsilon_px", ""),
                "standard_epsilon_px": match.get("epsilon_px", ""),
                "proposed_attack_success": f"{prop_asr:.6f}",
                "standard_attack_success": f"{std_asr:.6f}",
                "attack_success_gap_abs": f"{abs(prop_asr - std_asr):.6f}",
                "proposed_adv_accuracy": prop.get("adv_accuracy", ""),
                "standard_adv_accuracy": match.get("adv_accuracy", ""),
                "proposed_mask_area": prop.get("mean_layercam_mask_area", ""),
                "standard_mask_area": match.get("mean_layercam_mask_area", ""),
                "area_reduction": f"{1.0 - (_summary_float(prop, 'mean_layercam_mask_area') / max(_summary_float(match, 'mean_layercam_mask_area'), 1e-12)):.6f}",
                "proposed_l2": prop.get("mean_l2", ""),
                "standard_l2": match.get("mean_l2", ""),
                "l2_reduction": f"{1.0 - (_summary_float(prop, 'mean_l2') / max(_summary_float(match, 'mean_l2'), 1e-12)):.6f}",
                "proposed_mae": prop.get("mean_mae", ""),
                "standard_mae": match.get("mean_mae", ""),
                "mae_reduction": f"{1.0 - (_summary_float(prop, 'mean_mae') / max(_summary_float(match, 'mean_mae'), 1e-12)):.6f}",
                "proposed_mse": prop.get("mean_mse", ""),
                "standard_mse": match.get("mean_mse", ""),
                "mse_reduction": f"{1.0 - (_summary_float(prop, 'mean_mse') / max(_summary_float(match, 'mean_mse'), 1e-12)):.6f}",
                "proposed_psnr": prop.get("mean_psnr", ""),
                "standard_psnr": match.get("mean_psnr", ""),
                "psnr_gain": f"{_summary_float(prop, 'mean_psnr') - _summary_float(match, 'mean_psnr'):.6f}",
                "proposed_ssim": prop.get("mean_ssim", ""),
                "standard_ssim": match.get("mean_ssim", ""),
                "ssim_gain": f"{_summary_float(prop, 'mean_ssim') - _summary_float(match, 'mean_ssim'):.6f}",
            }
        )
    return rows


def summarize_results(
    method: str,
    epsilon_px: float,
    labels: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    clean_confs: np.ndarray,
    adv_confs: np.ndarray,
    linf: np.ndarray,
    l2: np.ndarray,
    l0_area: np.ndarray,
    mask_area: np.ndarray,
    mae: np.ndarray,
    mse: np.ndarray,
    psnr: np.ndarray,
    ssim: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    clean_correct = clean_preds == labels
    adv_correct = adv_preds == labels
    clean_acc = float(clean_correct.mean())
    adv_acc = float(adv_correct.mean())
    if clean_correct.any():
        attack_success = float((adv_preds[clean_correct] != labels[clean_correct]).mean())
    else:
        attack_success = 0.0
    pma, rma, f1ma, _ = precision_recall_fscore_support(
        labels,
        adv_preds,
        average="macro",
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    return {
        "method": method,
        "epsilon_px": f"{epsilon_px:g}",
        "n_test": int(labels.size),
        "clean_accuracy": f"{clean_acc:.6f}",
        "adv_accuracy": f"{adv_acc:.6f}",
        "accuracy_drop": f"{clean_acc - adv_acc:.6f}",
        "attack_success_rate_clean_correct": f"{attack_success:.6f}",
        "changed_prediction_rate": f"{float((adv_preds != clean_preds).mean()):.6f}",
        "adv_precision_macro": f"{float(pma):.6f}",
        "adv_recall_macro": f"{float(rma):.6f}",
        "adv_f1_macro": f"{float(f1ma):.6f}",
        "mean_clean_conf": f"{float(clean_confs.mean()):.6f}",
        "mean_adv_conf": f"{float(adv_confs.mean()):.6f}",
        "mean_linf_px": f"{float(linf.mean()):.6f}",
        "mean_l2": f"{float(l2.mean()):.6f}",
        "mean_l0_area": f"{float(l0_area.mean()):.6f}",
        "mean_layercam_mask_area": f"{float(mask_area.mean()):.6f}",
        "mean_mae": f"{float(mae.mean()):.8f}",
        "mean_mse": f"{float(mse.mean()):.8f}",
        "mean_psnr": f"{float(psnr.mean()):.6f}",
        "mean_ssim": f"{float(ssim.mean()):.6f}",
    }


def summarize_per_class(
    method: str,
    epsilon_px: float,
    labels: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    class_names: Sequence[str],
) -> List[dict]:
    rows: List[dict] = []
    for idx, class_name in enumerate(class_names):
        in_class = labels == idx
        n_class = int(in_class.sum())
        if n_class == 0:
            clean_acc = adv_acc = attack_success = 0.0
        else:
            clean_correct = clean_preds[in_class] == labels[in_class]
            adv_correct = adv_preds[in_class] == labels[in_class]
            clean_acc = float(clean_correct.mean())
            adv_acc = float(adv_correct.mean())
            if clean_correct.any():
                attack_success = float((adv_preds[in_class][clean_correct] != idx).mean())
            else:
                attack_success = 0.0
        rows.append(
            {
                "method": method,
                "epsilon_px": f"{epsilon_px:g}",
                "class": class_name,
                "n_test": n_class,
                "clean_accuracy": f"{clean_acc:.6f}",
                "adv_accuracy": f"{adv_acc:.6f}",
                "accuracy_drop": f"{clean_acc - adv_acc:.6f}",
                "attack_success_rate_clean_correct": f"{attack_success:.6f}",
            }
        )
    return rows


def collect_arrays(results: List[BatchResult]) -> Dict[str, np.ndarray]:
    out: Dict[str, List[np.ndarray]] = defaultdict(list)
    for r in results:
        out["labels"].append(r.labels.cpu().numpy())
        out["clean_preds"].append(r.clean_preds.cpu().numpy())
        out["clean_confs"].append(r.clean_confs.cpu().numpy())
        out["adv_preds"].append(r.adv_preds.cpu().numpy())
        out["adv_confs"].append(r.adv_confs.cpu().numpy())
        out["linf"].append(r.linf.cpu().numpy())
        out["l2"].append(r.l2.cpu().numpy())
        out["l0_area"].append(r.l0_area.cpu().numpy())
        out["mask_area"].append(r.mask_area.cpu().numpy())
        out["mae"].append(r.mae.cpu().numpy())
        out["mse"].append(r.mse.cpu().numpy())
        out["psnr"].append(r.psnr.cpu().numpy())
        out["ssim"].append(r.ssim.cpu().numpy())
    return {key: np.concatenate(value) for key, value in out.items()}


def save_examples_from_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    adv: torch.Tensor,
    clean_preds: torch.Tensor,
    clean_confs: torch.Tensor,
    adv_preds: torch.Tensor,
    adv_confs: torch.Tensor,
    mask_area: torch.Tensor,
    class_names: Sequence[str],
    dataset_samples: Sequence[Tuple[str, int]],
    sample_indices: Sequence[int],
    method_out_dir: str,
    method_name: str,
    epsilon_px: float,
    per_class_saved: Dict[str, int],
    max_per_class: int,
) -> List[dict]:
    rows: List[dict] = []
    for i in range(x.size(0)):
        label_idx = int(y[i].item())
        clean_idx = int(clean_preds[i].item())
        adv_idx = int(adv_preds[i].item())
        if clean_idx != label_idx:
            continue
        class_name = class_names[label_idx]
        if per_class_saved[class_name] >= max_per_class:
            continue

        per_class_saved[class_name] += 1
        source_path = dataset_samples[int(sample_indices[i])][0]
        image_id = os.path.splitext(os.path.basename(source_path))[0]
        file_name = (
            f"{per_class_saved[class_name]:02d}_{image_id}_"
            f"clean-{class_names[clean_idx]}_adv-{class_names[adv_idx]}.png"
        )
        out_path = os.path.join(method_out_dir, "examples", class_name, file_name)
        save_example_figure(
            out_path=out_path,
            original=x[i],
            adversarial=adv[i],
            label_name=class_name,
            clean_pred_name=class_names[clean_idx],
            clean_conf=float(clean_confs[i].item()),
            adv_pred_name=class_names[adv_idx],
            adv_conf=float(adv_confs[i].item()),
            attack_name=method_name,
            epsilon_px=epsilon_px,
            mask_area=float(mask_area[i].item()),
        )
        rows.append(
            {
                "method": method_name,
                "epsilon_px": f"{epsilon_px:g}",
                "class": class_name,
                "source_path": source_path,
                "example_path": out_path,
                "clean_pred": class_names[clean_idx],
                "clean_conf": f"{float(clean_confs[i].item()):.6f}",
                "adv_pred": class_names[adv_idx],
                "adv_conf": f"{float(adv_confs[i].item()):.6f}",
                "mask_area": f"{float(mask_area[i].item()):.6f}",
            }
        )
    return rows


def evaluate_attack(
    model: nn.Module,
    model_name: str,
    normalize: nn.Module,
    loader: DataLoader,
    dataset: datasets.ImageFolder,
    cfg: AttackConfig,
    method: str,
    out_dir: str,
    device: torch.device,
    max_examples_per_class: int,
    use_amp: bool,
) -> dict:
    method_out = os.path.join(out_dir, epsilon_dir_name(cfg.epsilon_px), method)
    results: List[BatchResult] = []
    prediction_rows: List[dict] = []
    example_rows: List[dict] = []
    per_class_saved: Dict[str, int] = defaultdict(int)

    cam = None
    if method in (
        "layercam_cut_pgd",
        "layercam_masked_pgd",
        "layercam_random_noise",
        "random_masked_pgd",
    ):
        target_layers = get_layercam_target_layers(model_name, model)
        cam_model = NormalizedModel(normalize, model).to(device)
        cam_model.eval()
        cam = LayerCAM(model=cam_model, target_layers=target_layers)

    sample_cursor = 0
    desc = f"{method} eps={cfg.epsilon_px:g}/255"
    for x, y in tqdm(loader, desc=desc):
        batch_size = x.size(0)
        sample_indices = list(range(sample_cursor, sample_cursor + batch_size))
        sample_cursor += batch_size
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        clean_preds, clean_confs = predict(model, normalize, x, use_amp=use_amp)
        mask = None
        if method in (
            "layercam_cut_pgd",
            "layercam_masked_pgd",
            "layercam_random_noise",
            "random_masked_pgd",
        ):
            assert cam is not None
            mask = make_layercam_mask(cam, x, clean_preds, cfg.layercam_threshold)
            if method == "random_masked_pgd":
                mask = make_random_mask_like(mask)
        elif method not in ("clean", "random_noise", "standard_pgd"):
            raise ValueError(f"Unknown method: {method}")

        if method == "clean":
            adv = x.detach()
        elif method in ("random_noise", "layercam_random_noise"):
            adv = random_noise_attack(x, cfg, mask=mask)
        elif method == "layercam_cut_pgd":
            standard_adv = pgd_attack(model, normalize, x, y, cfg, mask=None, use_amp=use_amp)
            assert mask is not None
            cut_delta = project_delta((standard_adv - x) * mask, cfg.epsilon, mask=mask)
            adv = torch.clamp(x + cut_delta, 0.0, 1.0)
        else:
            adv = pgd_attack(model, normalize, x, y, cfg, mask=mask, use_amp=use_amp)
        adv_preds, adv_confs = predict(model, normalize, adv, use_amp=use_amp)
        linf, l2, l0_area, mask_area, mae, mse, psnr, ssim = perturbation_stats(
            x, adv, mask, cfg.epsilon
        )

        results.append(
            BatchResult(
                labels=y.detach().cpu(),
                clean_preds=clean_preds.detach().cpu(),
                clean_confs=clean_confs.detach().cpu(),
                adv_preds=adv_preds.detach().cpu(),
                adv_confs=adv_confs.detach().cpu(),
                correct_clean=(clean_preds == y).detach().cpu(),
                mask_area=mask_area.detach().cpu(),
                linf=linf.detach().cpu(),
                l2=l2.detach().cpu(),
                l0_area=l0_area.detach().cpu(),
                mae=mae.detach().cpu(),
                mse=mse.detach().cpu(),
                psnr=psnr.detach().cpu(),
                ssim=ssim.detach().cpu(),
            )
        )

        for i in range(batch_size):
            source_path = dataset.samples[int(sample_indices[i])][0]
            label_idx = int(y[i].item())
            clean_idx = int(clean_preds[i].item())
            adv_idx = int(adv_preds[i].item())
            prediction_rows.append(
                {
                    "method": method,
                    "epsilon_px": f"{cfg.epsilon_px:g}",
                    "source_path": source_path,
                    "label": dataset.classes[label_idx],
                    "clean_pred": dataset.classes[clean_idx],
                    "clean_conf": f"{float(clean_confs[i].item()):.6f}",
                    "adv_pred": dataset.classes[adv_idx],
                    "adv_conf": f"{float(adv_confs[i].item()):.6f}",
                    "clean_correct": int(clean_idx == label_idx),
                    "adv_correct": int(adv_idx == label_idx),
                    "linf_px": f"{float(linf[i].item()):.6f}",
                    "l2": f"{float(l2[i].item()):.6f}",
                    "l0_area": f"{float(l0_area[i].item()):.6f}",
                    "layercam_mask_area": f"{float(mask_area[i].item()):.6f}",
                    "mae": f"{float(mae[i].item()):.8f}",
                    "mse": f"{float(mse[i].item()):.8f}",
                    "psnr": f"{float(psnr[i].item()):.6f}",
                    "ssim": f"{float(ssim[i].item()):.6f}",
                }
            )

        example_rows.extend(
            save_examples_from_batch(
                x=x.detach().cpu(),
                y=y.detach().cpu(),
                adv=adv.detach().cpu(),
                clean_preds=clean_preds.detach().cpu(),
                clean_confs=clean_confs.detach().cpu(),
                adv_preds=adv_preds.detach().cpu(),
                adv_confs=adv_confs.detach().cpu(),
                mask_area=mask_area.detach().cpu(),
                class_names=dataset.classes,
                dataset_samples=dataset.samples,
                sample_indices=sample_indices,
                method_out_dir=method_out,
                method_name=method,
                epsilon_px=cfg.epsilon_px,
                per_class_saved=per_class_saved,
                max_per_class=max_examples_per_class,
            )
        )

    arrays = collect_arrays(results)
    summary = summarize_results(
        method=method,
        epsilon_px=cfg.epsilon_px,
        labels=arrays["labels"],
        clean_preds=arrays["clean_preds"],
        adv_preds=arrays["adv_preds"],
        clean_confs=arrays["clean_confs"],
        adv_confs=arrays["adv_confs"],
        linf=arrays["linf"],
        l2=arrays["l2"],
        l0_area=arrays["l0_area"],
        mask_area=arrays["mask_area"],
        mae=arrays["mae"],
        mse=arrays["mse"],
        psnr=arrays["psnr"],
        ssim=arrays["ssim"],
        class_names=dataset.classes,
    )
    summary.update(
        {
            "pgd_steps": cfg.steps,
            "alpha_px": f"{cfg.alpha_px:g}",
            "random_start": int(cfg.random_start),
            "layercam_threshold": f"{cfg.layercam_threshold:.3f}" if method.startswith("layercam_") else "",
        }
    )
    per_class_rows = summarize_per_class(
        method=method,
        epsilon_px=cfg.epsilon_px,
        labels=arrays["labels"],
        clean_preds=arrays["clean_preds"],
        adv_preds=arrays["adv_preds"],
        class_names=dataset.classes,
    )
    write_rows(os.path.join(method_out, "metrics.csv"), [summary])
    write_rows(os.path.join(method_out, "class_metrics.csv"), per_class_rows)
    write_rows(os.path.join(method_out, "predictions.csv"), prediction_rows)
    write_rows(os.path.join(method_out, "examples.csv"), example_rows)
    return summary


def parse_epsilons(raw: str) -> List[float]:
    values = []
    for token in raw.replace(",", " ").split():
        values.append(float(token))
    if not values:
        raise ValueError("At least one epsilon is required.")
    return values


def parse_methods(raw: str) -> List[str]:
    methods = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
    allowed = {
        "clean",
        "random_noise",
        "standard_pgd",
        "random_masked_pgd",
        "layercam_cut_pgd",
        "layercam_masked_pgd",
        "layercam_random_noise",
    }
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    return methods


def parse_method_eps(raw: str) -> List[Tuple[str, float]]:
    raw = raw.strip()
    if not raw:
        return []
    pairs: List[Tuple[str, float]] = []
    allowed = {
        "clean",
        "random_noise",
        "standard_pgd",
        "random_masked_pgd",
        "layercam_cut_pgd",
        "layercam_masked_pgd",
        "layercam_random_noise",
    }
    for block in [x.strip() for x in raw.split(";") if x.strip()]:
        if ":" not in block:
            raise ValueError("Use --method-eps like 'standard_pgd:0.125,0.25;layercam_masked_pgd:0.5,1'")
        method, eps_raw = block.split(":", 1)
        method = method.strip()
        if method not in allowed:
            raise ValueError(f"Unknown method in --method-eps: {method}")
        for epsilon_px in parse_epsilons(eps_raw):
            pairs.append((method, epsilon_px))
    return pairs


def epsilon_dir_name(epsilon_px: float) -> str:
    return f"epsilon_{epsilon_px:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standard PGD and LayerCAM-guided PGD variants."
    )
    parser.add_argument("--model-name", default="baseline_efficientnetb0")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--test-dir", default=TEST_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epsilons", default="0.125,0.25,0.5,1")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--alpha-divisor", type=float, default=4.0)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--layercam-threshold", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        default="standard_pgd,layercam_masked_pgd",
        help="Comma/space separated: clean, random_noise, standard_pgd, random_masked_pgd, layercam_cut_pgd, layercam_masked_pgd, layercam_random_noise",
    )
    parser.add_argument(
        "--method-eps",
        default="",
        help="Optional per-method epsilon grid, e.g. 'standard_pgd:0.125,0.25;layercam_masked_pgd:0.5,1'",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--examples-per-class", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--balanced-subset", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for PGD/prediction forward passes.")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul/cudnn acceleration on CUDA.")
    parser.add_argument("--run-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_name = args.model_name.lower()

    seed_everything(args.run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    loader, dataset = build_test_loader(
        args.test_dir,
        args.batch_size,
        args.num_workers,
        device,
        max_samples=args.max_samples,
        balanced_subset=args.balanced_subset,
        subset_seed=args.run_seed,
    )

    checkpoint = args.checkpoint.strip()
    if not checkpoint:
        if args.model_name in BASELINE_MODEL_NAMES:
            checkpoint = os.path.join(
                WEIGHTS_DIR,
                f"best_efficientnetb0_pretrained_seed{args.seed}.pth",
            )
        elif args.model_name in {
            "proposed_efficientnetb0_se_mlp512_full",
            "grid_efficientnetb0_se_mlp512_full",
        }:
            checkpoint = os.path.join(
                EFFICIENTNET_GRID_WEIGHTS_DIR,
                f"best_grid_efficientnetb0_se_mlp512_full_pretrained_seed{args.seed}.pth",
            )
        else:
            raise ValueError(f"Unsupported model for default checkpoint: {args.model_name}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = load_checkpoint(checkpoint, args.model_name, len(dataset.classes), device)
    normalize = NormalizeModule().to(device)
    normalize.eval()

    if args.model_name in BASELINE_MODEL_NAMES:
        run_name = f"baseline_efficientnetb0_seed{args.seed}"
    elif args.model_name == "grid_efficientnetb0_se_mlp512_full":
        run_name = f"proposed_efficientnetb0_se_mlp512_full_seed{args.seed}"
    else:
        run_name = f"{args.model_name}_seed{args.seed}"
    run_out = os.path.join(args.out_dir, run_name)
    method_eps_pairs = parse_method_eps(args.method_eps)
    if not method_eps_pairs:
        methods = parse_methods(args.methods)
        epsilons = parse_epsilons(args.epsilons)
        method_eps_pairs = [(method, epsilon_px) for epsilon_px in epsilons for method in methods]

    all_summaries: List[dict] = []
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test samples: {len(dataset)} | classes: {len(dataset.classes)}")
    print(f"Output: {run_out}")
    for method, epsilon_px in method_eps_pairs:
        cfg = AttackConfig(
            name="pgd",
            epsilon_px=epsilon_px,
            alpha_px=epsilon_px / args.alpha_divisor,
            steps=args.steps,
            random_start=not args.no_random_start,
            layercam_threshold=args.layercam_threshold,
        )
        summary = evaluate_attack(
            model=model,
            model_name=args.model_name,
            normalize=normalize,
            loader=loader,
            dataset=dataset,
            cfg=cfg,
            method=method,
            out_dir=run_out,
            device=device,
            max_examples_per_class=args.examples_per_class,
            use_amp=(args.amp and device.type == "cuda"),
        )
        all_summaries.append(summary)
        print(
            f"{method} eps={epsilon_px:g}/255 "
            f"adv_acc={summary['adv_accuracy']} "
            f"asr={summary['attack_success_rate_clean_correct']}"
        )

    write_rows(os.path.join(run_out, "metrics_summary.csv"), all_summaries)
    write_rows(os.path.join(run_out, "method_comparisons.csv"), build_comparison_rows(all_summaries))
    write_rows(os.path.join(run_out, "matched_attack_comparisons.csv"), build_matched_attack_rows(all_summaries))
    print(f"Done: {run_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
