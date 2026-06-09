from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import torch.nn as nn
from torchvision import models


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_checkpoint_filename(filename: str) -> Optional[Tuple[str, str, int]]:
    """Parse checkpoint names like best_resnet50_pretrained_seed3.pth."""
    name = os.path.basename(filename)
    match = re.match(r"^best_(.+)_(pretrained|scratch)_seed(\d+)\.pth$", name, re.I)
    if match is None:
        return None
    model_name, weight_tag, seed = match.groups()
    return model_name.lower(), weight_tag.lower(), int(seed)


def list_test_samples(test_dir: str) -> Tuple[List[str], List[Tuple[str, int]]]:
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    classes = [
        name
        for name in sorted(os.listdir(test_dir))
        if os.path.isdir(os.path.join(test_dir, name))
    ]
    samples: List[Tuple[str, int]] = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(test_dir, class_name)
        for filename in sorted(os.listdir(class_dir)):
            path = os.path.join(class_dir, filename)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_idx))
    return classes, samples


def get_model(model_name: str, num_classes: int, use_pretrained: bool):
    model_name = model_name.lower()

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
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


def get_layercam_target_layers(model_name: str, model):
    model_name = model_name.lower()
    if model_name in {"resnet18", "resnet34", "resnet50"}:
        return [model.layer4[-1]]
    if model_name.startswith("efficientnet"):
        return [model.features[-1]]
    if model_name in {"mobilenetv2", "mobilenetv3"}:
        return [model.features[-1]]
    raise ValueError(f"Unsupported model name for LayerCAM: {model_name}")
